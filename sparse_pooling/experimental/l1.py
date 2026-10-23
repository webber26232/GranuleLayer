"""EXPERIMENTAL L1 treatments, frozen masks and sparsity statistics of the K64 study.

Objective (weights only; padding and biases excluded):

    L = L_task + mu * sum_valid_weights |w|          (lambda_mean = mu * W0, W0 = 800M valid slots, fixed)

``L1AdamW`` is ``optim.FusedAdamW`` (same update, BF16 or FP32 moments, stochastic rounding, the in-kernel
load/store barrier of the race fix, fused main_grad zeroing) with one of three treatments per param group:

* ``none``     the dense baseline (mu = 0);
* ``coupled``  g <- g + mu sign(w) (sign(0) = 0) added to the FP32 gradient inside the kernel (no extra tensor)
               before the unchanged AdamW update: L1 enters the moments;
* ``prox``     diagonal-metric proximal AdamW: the moments see the DATA gradient only; the AdamW candidate
               u = w (1 - eta wd) - eta m_hat / (sqrt(v_hat) + eps) is soft-thresholded in FP32 before BF16
               rounding,  w_next = sign(u) max(|u| - eta mu / (sqrt(v_hat) + eps), 0)
               (v_hat = the updated second moment / (1 - beta2^t)); exact zeros survive stochastic rounding.

AdamW decoupled weight decay (``weight_decay``) is a separate coefficient (0 in this study unless recorded).

A frozen mask (``set_mask(p, mask)``: uint8, 1 = active) zeroes the gradient of inactive elements and keeps
their value, moments and update at zero (W_eff = mask * W; the exact gradient mask * dL/dW_eff); inactive
moments must be zero when the mask is installed (``apply_mask_`` does it).  Skipping work for inactive
elements is NOT done here: that is the job of the compact sparse formats (kgx_sparse).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import triton
import triton.language as tl

MODES = {"none": 0, "coupled": 1, "prox": 2}


@triton.jit
def _l1_adamw_kernel(P, G, M, V, MASK, n, lr, beta1, beta2, eps, wd, bc1, bc2, seed, mu,
                     MODE: tl.constexpr, HAS_MASK: tl.constexpr, STOCH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    g = tl.load(G + offs, mask=inb, other=0.0)
    m = tl.load(M + offs, mask=inb, other=0.0).to(tl.float32)
    v = tl.load(V + offs, mask=inb, other=0.0).to(tl.float32)
    p = tl.load(P + offs, mask=inb, other=0.0).to(tl.float32)
    if HAS_MASK:
        keep = tl.load(MASK + offs, mask=inb, other=0) != 0
        g = tl.where(keep, g, 0.0)
    if MODE == 1:  # coupled L1 subgradient, sign(0) = 0
        sgn = tl.where(p > 0, 1.0, tl.where(p < 0, -1.0, 0.0))
        g = g + mu * sgn
        if HAS_MASK:
            g = tl.where(keep, g, 0.0)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    denom = tl.sqrt(v / bc2) + eps
    u = p * (1.0 - lr * wd) - lr * ((m / bc1) / denom)
    if MODE == 2:  # proximal soft threshold in FP32, before BF16 rounding
        thr = lr * mu / denom
        au = tl.abs(u)
        u = tl.where(au > thr, tl.where(u > 0, au - thr, thr - au), 0.0)
    if HAS_MASK:
        u = tl.where(keep, u, 0.0)
    tl.debug_barrier()  # all loads of the program precede its stores (optim.FusedAdamW race fix)
    tl.store(M + offs, m.to(M.dtype.element_ty), mask=inb)
    tl.store(V + offs, v.to(V.dtype.element_ty), mask=inb)
    tl.store(G + offs, tl.zeros_like(g), mask=inb)  # fused zero_grad
    if STOCH:
        bits = u.to(tl.uint32, bitcast=True)
        r = tl.randint(seed, offs).to(tl.uint32) & 0xFFFF
        bits = (bits + r) & 0xFFFF0000  # exact zero (+0 / -0) stays zero
        tl.store(P + offs, bits.to(tl.float32, bitcast=True).to(P.dtype.element_ty), mask=inb)
    else:
        tl.store(P + offs, u.to(P.dtype.element_ty), mask=inb)


class L1AdamW(torch.optim.Optimizer):
    """FusedAdamW with the L1 treatments / frozen masks above (see the module docstring).  Param-group keys:
    lr, betas, eps, weight_decay, l1 ("none" | "coupled" | "prox"), mu."""

    def __init__(self, params: Iterable, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 moments: str = "bf16", stochastic_rounding: bool = True, l1: str = "none", mu: float = 0.0):
        if moments not in ("bf16", "fp32"):
            raise ValueError("moments must be 'bf16' or 'fp32'")
        if l1 not in MODES:
            raise ValueError(f"l1 must be one of {tuple(MODES)}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, l1=l1, mu=mu))
        self.moments = moments
        self.stochastic_rounding = stochastic_rounding
        self._seed = 0
        self.masks = {}

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for st in self.state.values() for t in st.values() if torch.is_tensor(t))

    def mask_bytes(self) -> int:
        return sum(m.numel() * m.element_size() for m in self.masks.values())

    def set_mask(self, p, mask):
        if mask is None:
            self.masks.pop(p, None)
            return
        if mask.shape != p.shape or mask.dtype != torch.uint8 or not mask.is_contiguous():
            raise ValueError("mask: contiguous uint8 of the parameter's shape")
        self.masks[p] = mask

    def _state(self, p):
        st = self.state[p]
        if not st:
            mdt = torch.bfloat16 if self.moments == "bf16" else torch.float32
            st["step"] = 0
            st["m"] = torch.zeros(p.shape, device=p.device, dtype=mdt)
            st["v"] = torch.zeros(p.shape, device=p.device, dtype=mdt)
        return st

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            mode = MODES[group["l1"]]
            for p in group["params"]:
                g = getattr(p, "main_grad", None)
                if g is None:
                    if p.grad is None:
                        continue
                    g = p.grad.float().contiguous()
                st = self._state(p)
                st["step"] += 1
                t = st["step"]
                n = p.numel()
                BLOCK = 2048
                self._seed += 1
                mask = self.masks.get(p)
                with torch.cuda.device(p.device):
                    _l1_adamw_kernel[(triton.cdiv(n, BLOCK),)](
                        p, g, st["m"], st["v"], mask if mask is not None else p, n,
                        group["lr"], b1, b2, group["eps"], group["weight_decay"], 1 - b1 ** t, 1 - b2 ** t,
                        self._seed, float(group["mu"]), MODE=mode, HAS_MASK=mask is not None,
                        STOCH=(p.dtype == torch.bfloat16 and self.stochastic_rounding), BLOCK=BLOCK, num_warps=8)
                if getattr(p, "main_grad", None) is None:
                    p.grad = None
        return None


# ------------------------------------------------------------------------- references (tests)
def reference_update(p, g, m, v, step, lr, b1, b2, eps, wd, mode: str, mu: float, mask=None):
    """FP64 reference of one L1AdamW element-wise update (no rounding of the result): (u, m', v')."""
    p, g, m, v = (t.double() for t in (p, g, m, v))
    if mask is not None:
        g = torch.where(mask.bool(), g, torch.zeros_like(g))
    if mode == "coupled":
        g = g + mu * torch.sign(p)
        if mask is not None:
            g = torch.where(mask.bool(), g, torch.zeros_like(g))
    m2 = b1 * m + (1 - b1) * g
    v2 = b2 * v + (1 - b2) * g * g
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    denom = torch.sqrt(v2 / bc2) + eps
    u = p * (1 - lr * wd) - lr * (m2 / bc1) / denom
    thr = None
    if mode == "prox":
        thr = lr * mu / denom
        u = torch.sign(u) * torch.clamp(u.abs() - thr, min=0)
    if mask is not None:
        u = torch.where(mask.bool(), u, torch.zeros_like(u))
    return u, m2, v2, thr


# ------------------------------------------------------------------------- statistics
NEAR_ZERO = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)  # absolute |w| thresholds (init std = 1/8 = 0.125)


@torch.no_grad()
def weight_stats(w: torch.Tensor, valid: torch.Tensor, chunk_groups: int = 64) -> dict:
    """Sparsity statistics of dense K = 64 weights w [G, NB, C, 64] over VALID neurons (valid [G, NB, C] bool):
    L1 norm, exact zeros, near-zero counts (``NEAR_ZERO``), active fan-in histogram (exact nonzeros per valid
    neuron, 0..64), all-zero valid neurons."""
    G = w.shape[0]
    l1 = 0.0
    zeros = 0
    near = [0] * len(NEAR_ZERO)
    hist = torch.zeros(65, dtype=torch.int64, device=w.device)
    nval = 0
    for g0 in range(0, G, chunk_groups):
        wc = w[g0:g0 + chunk_groups]
        vc = valid[g0:g0 + chunk_groups]
        wv = wc[vc].float()  # [n_valid, 64]
        nval += wv.shape[0]
        l1 += float(wv.abs().sum(dtype=torch.float64))
        zeros += int((wv == 0).sum())
        a = wv.abs()
        for i, t in enumerate(NEAR_ZERO):
            near[i] += int((a <= t).sum())
        hist += torch.bincount((wv != 0).sum(1), minlength=65)
    W0 = nval * w.shape[-1]
    fan = hist.double()
    mean_fan = float((fan * torch.arange(65, device=w.device)).sum() / max(nval, 1))
    return dict(valid_neurons=nval, valid_weights=W0, l1=l1, mean_abs=l1 / max(W0, 1), exact_zeros=zeros,
                density_nonzero=1 - zeros / max(W0, 1),
                near_zero={f"{t:g}": c for t, c in zip(NEAR_ZERO, near)},
                near_zero_fraction={f"{t:g}": c / max(W0, 1) for t, c in zip(NEAR_ZERO, near)},
                active_fan_in_hist=hist.cpu().tolist(), active_fan_in_mean=mean_fan,
                all_zero_neurons=int(hist[0]))


@torch.no_grad()
def sample_positions(valid: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """n fixed reproducible flat positions of VALID weights of a [G, NB, C, 64] layout (uniform over valid
    neuron slots x 64 inputs; with replacement)."""
    vflat = torch.nonzero(valid.reshape(-1)).squeeze(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ni = torch.randint(0, vflat.numel(), (n,), generator=g)
    k = torch.randint(0, 64, (n,), generator=g)
    return (vflat.cpu()[ni] * 64 + k).to(valid.device)


@torch.no_grad()
def calibrate_mu(grad: torch.Tensor, positions: torch.Tensor) -> dict:
    """g_ref = median |g| over the fixed sample of valid positions (FP32 task gradients).  Fallback when the
    median is 0: the mean of the nonzero |g| of the sample (documented; flagged)."""
    a = grad.reshape(-1)[positions].abs().double()
    med = float(a.median())
    rec = dict(statistic="median |g| of a fixed sample of valid FP32 task gradients", sample=int(a.numel()),
               median=med, mean=float(a.mean()), p90=float(a.quantile(0.9)), zero_fraction=float((a == 0).double().mean()))
    if med > 0:
        rec.update(g_ref=med, fallback=False)
    else:
        nz = a[a > 0]
        rec.update(g_ref=float(nz.mean()) if nz.numel() else 0.0, fallback=True,
                   fallback_rule="mean of the nonzero |g| of the sample")
    return rec


@torch.no_grad()
def apply_mask_(opt: L1AdamW, p: torch.nn.Parameter, mask: torch.Tensor):
    """Install a frozen mask: zero inactive values, their main_grad and their optimizer moments."""
    keep = mask.bool()
    p.data.masked_fill_(~keep, 0)
    if getattr(p, "main_grad", None) is not None:
        p.main_grad.masked_fill_(~keep, 0)
    st = opt.state.get(p)
    if st:
        for k in ("m", "v"):
            st[k].masked_fill_(~keep, 0)
    opt.set_mask(p, mask)


def lambda_mean(mu: float, W0: int = 800_000_000) -> float:
    return mu * W0


def _isfinite(x) -> bool:
    return x == x and not math.isinf(x)
