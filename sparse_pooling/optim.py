"""Memory-conscious fused AdamW for the 800M-weight layer.

Reads the FP32 ``main_grad`` buffers written by the fused backward, updates the
parameters, and zeroes ``main_grad`` in the same pass (no separate memset).

Memory per trainable element (BF16 parameter + FP32 main_grad = 6 B always):
    moments="fp32", master_weights=True   -> +12 B   (exact mixed-precision AdamW)
    moments="fp32", master_weights=False  -> + 8 B   (BF16 params, stochastic rounding)
    moments="bf16", master_weights=False  -> + 4 B   (smallest; default)

Semantics are dense AdamW for every element, including weight decay and moment
decay of elements whose gradient is zero in this step (e.g. non-winners under
Max pooling).  A "lazy"/sparse Adam would be cheaper for Max/Top-K but changes
the optimizer, so it is deliberately not offered silently.
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(P, G, M, V, MASTER, n, lr, beta1, beta2, eps, wd, bc1, bc2, seed,
                  HAS_MASTER: tl.constexpr, STOCH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0)
    m = tl.load(M + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_MASTER:
        p = tl.load(MASTER + offs, mask=mask, other=0.0)
    else:
        p = tl.load(P + offs, mask=mask, other=0.0).to(tl.float32)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    upd = (m / bc1) / (tl.sqrt(v / bc2) + eps)
    p = p * (1.0 - lr * wd) - lr * upd
    # Every buffer is updated in place, and Triton (3.3) gives a load and the store of the same
    # element to different threads: G and MASTER (FP32) are loaded in the 8-elements-per-thread
    # layout of the BF16 math but stored in a 4-per-thread layout (TTGIR #blocked1 vs #blocked).
    # Without this barrier a thread could zero G[i] (or overwrite MASTER[i]) before the thread
    # loading it had read it, which dropped that step's gradient of a few elements
    # nondeterministically (results/SCALE_DISPATCH_FUSED_BACKWARD.md section 3).  All loads of
    # the program precede all of its stores; programs touch disjoint elements.
    tl.debug_barrier()
    tl.store(M + offs, m.to(M.dtype.element_ty), mask=mask)
    tl.store(V + offs, v.to(V.dtype.element_ty), mask=mask)
    tl.store(G + offs, tl.zeros_like(g), mask=mask)  # fused zero_grad
    if HAS_MASTER:
        tl.store(MASTER + offs, p, mask=mask)
        tl.store(P + offs, p.to(P.dtype.element_ty), mask=mask)
    elif STOCH:
        # stochastic rounding FP32 -> BF16: add uniform noise below the BF16 ulp, truncate
        bits = p.to(tl.uint32, bitcast=True)
        r = tl.randint(seed, offs).to(tl.uint32) & 0xFFFF
        bits = (bits + r) & 0xFFFF0000
        tl.store(P + offs, bits.to(tl.float32, bitcast=True).to(P.dtype.element_ty), mask=mask)
    else:
        tl.store(P + offs, p.to(P.dtype.element_ty), mask=mask)


class FusedAdamW(torch.optim.Optimizer):
    """AdamW over parameters that carry an FP32 ``main_grad`` (see
    SparsePoolingLayer.enable_main_grad); falls back to ``.grad`` otherwise."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, moments: str = "bf16", master_weights: bool = False,
                 stochastic_rounding: bool = True):
        if moments not in ("bf16", "fp32"):
            raise ValueError("moments must be 'bf16' or 'fp32'")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.moments = moments
        self.master_weights = master_weights
        self.stochastic_rounding = stochastic_rounding
        self._seed = 0

    def state_bytes(self) -> int:
        n = 0
        for st in self.state.values():
            for t in st.values():
                if torch.is_tensor(t):
                    n += t.numel() * t.element_size()
        return n

    @torch.no_grad()
    def step(self, closure=None):
        mdt = torch.bfloat16 if self.moments == "bf16" else torch.float32
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                g = getattr(p, "main_grad", None)
                if g is None:
                    if p.grad is None:
                        continue
                    g = p.grad.float().contiguous()
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros(p.shape, device=p.device, dtype=mdt)
                    st["v"] = torch.zeros(p.shape, device=p.device, dtype=mdt)
                    if self.master_weights and p.dtype != torch.float32:
                        st["master"] = p.detach().float().clone()
                st["step"] += 1
                t = st["step"]
                master = st.get("master")
                n = p.numel()
                BLOCK = 2048
                self._seed += 1
                _adamw_kernel[(triton.cdiv(n, BLOCK),)](
                    p, g, st["m"], st["v"], master if master is not None else p, n,
                    group["lr"], b1, b2, group["eps"], group["weight_decay"],
                    1 - b1 ** t, 1 - b2 ** t, self._seed,
                    HAS_MASTER=master is not None,
                    STOCH=(master is None and p.dtype == torch.bfloat16 and self.stochastic_rounding),
                    BLOCK=BLOCK, num_warps=8,
                )
                if getattr(p, "main_grad", None) is None:
                    p.grad = None
        return None
