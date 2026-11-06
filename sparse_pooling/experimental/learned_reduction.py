"""EXPERIMENTAL learned sliding-window reduction of the wide hidden layer (docs/experiments/
learned_sliding_window_reduction.md).  Explicit API only: nothing here changes SparsePoolingLayer, CoverTopK16Layer or
their defaults.

    z[m, i] = sum_k W[i, k] x[m, idx[i, k]] + b[i]       h = ReLU(z)         sparse projection: unchanged graph / W / b
    u[m, j] = sum_r P[j, r] h[m, start(j) + r] + c[j]    a = gamma * psi(u)  P: independent per window, no sharing
    y[m, o] = sum_j a[m, j] V[j, o] + d[o]               V: FP32 dense readout (normal torch.matmul)

Model matrix (``ReducerConfig``, ``core_configs``):

    A0  group Mean(ReLU), no reducer       A1  A0 + dense           A2  fixed mean of s neurons + dense
    A3  learned sum of s neurons + dense   A4  learned L / s windows + dense
    A5  fixed mean over the A4 windows + dense
    MC  L = s = 10240 windows, L // 2048 independent outputs per window + dense (parameter-budget control of A4)

Execution:

* A0, A1: the K16 kgen kernels, as CoverTopK16Layer.
* A2, A5: the same kernels on a *virtual stride view* (``stride_view``) of the same storage: no new kernel, no copy, the
  graph / support / values are those of the base layer.  A5 averages the block means in compact [M, J] space.
* A3, A4, MC: the lrd kernels (kernels/cuda/experimental/lrd): fused forward, neuron-dependent dX / dW / db, dP (separately
  scheduled or fused with dW).  No M x H tensor is allocated.

Numerics: BF16 x / W / b / P operands, FP32 accumulation and FP32 main_grad, BF16 hi + lo split of h and Delta (plan
"split"), FusedAdamW for W / b / P; FP32 V / d / c, ordinary autograd, torch.optim.AdamW.  FP32 atomics: not bitwise
deterministic; reduction="deterministic" is rejected.  No dispatch or tuning entry is inherited from Cover-TopK/16:
the K16 dX tilings used here are explicit, unvalidated requests.  The lrd defaults (mma_split, dX MT 2 / NW 4 / WG 2, fused
dW + dP with 2 tasks per warp) come from measurements at H = 2,621,440 and 10,485,760, M = 8192 (experiment records);
the other compiled variants stay selectable for A/B comparisons.
"""
from __future__ import annotations

import ctypes
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as TF
from torch import nn

from ..supergroups import chunk_table
from . import kgen_graph as KG
from .kgen_layer import KgenLayer
from .kgen_layout import KLayout
from .kgen_runtime import KgArgs
from .reduction_layout import ReductionSpec, Routing

EPILOGUES = {"mma_bf16": 0, "mma_split": 1, "simt": 2}
DP_MODES = ("separate", "fused")
SCHEMA = "granule.learned_reduction/1"


def _mark(name):
    from ..kernels import cuda_ops

    cuda_ops._mark(name)


# --------------------------------------------------------------------------- native bridge
class LrArgs(ctypes.Structure):
    _fields_ = [("k", KgArgs), ("P", ctypes.c_void_p), ("U", ctypes.c_void_p), ("Dl", ctypes.c_void_p),
                ("dP", ctypes.c_void_p), ("F", ctypes.c_int), ("J", ctypes.c_int), ("PAD", ctypes.c_int),
                ("pad0", ctypes.c_int)]


class LrdBackend:
    def __init__(self, lib, target: str, why: str):
        self.native, self.target, self.target_reason = lib, target, why
        from ..backends.cuda import runtime as R

        self._R = R
        f = lib.fn("sp_lr_struct_layout", [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int])
        buf = (ctypes.c_longlong * 16)()
        n = f(buf, 16)
        want = [ctypes.sizeof(LrArgs)] + [getattr(LrArgs, nm).offset for nm, _ in LrArgs._fields_]
        if n != len(want) or list(buf[:n]) != want:
            raise RuntimeError(f"{lib.path}: LrArgs layout mismatch (native {list(buf[:max(n, 0)])}, ctypes {want})")
        A = ctypes.POINTER(LrArgs)
        self._fwd = lib.fn("sp_lr_fwd", [A, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
        self._dx = lib.fn("sp_lr_dx", [A] + [ctypes.c_int] * 6 + [ctypes.c_void_p])
        self._dw = lib.fn("sp_lr_dw", [A, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
        self._attrs = lib.fn("sp_lr_attrs", [ctypes.c_int] * 4 + [ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong)])

    def fwd(self, a, variant, n_chunks, n_tiles, device):
        self._R._check(self._fwd(ctypes.byref(a), variant, n_chunks, n_tiles, self._R._stream(device)), "sp_lr_fwd")

    def dx(self, a, split, mt, nw, reload, n_chunks, n_tiles, device):
        self._R._check(self._dx(ctypes.byref(a), split, mt, nw, reload, n_chunks, n_tiles, self._R._stream(device)),
                       f"sp_lr_dx(mt {mt}, nw {nw}, reload {reload})")

    def dw(self, a, split, mode, wpt, n_chunks, device):
        self._R._check(self._dw(ctypes.byref(a), split, mode, wpt, n_chunks, self._R._stream(device)), "sp_lr_dw")

    def attributes(self, kind: int, v0: int, v1: int, v2: int, smem: int) -> Dict:
        o = (ctypes.c_longlong * 6)()
        n = self._attrs(kind, v0, v1, v2, smem, o)
        if n != 6:
            return dict(error=n)
        return dict(registers_per_thread=o[0], local_bytes_per_thread=o[1], static_smem=o[2], max_threads_per_block=o[3],
                    active_ctas_per_sm=o[4], max_dynamic_smem_configured=o[5])

    def build_info(self) -> Dict:
        i = self.native.ident
        return dict(bundle="lrd", status="experimental", target=self.target, target_reason=self.target_reason,
                    kind=i["kind"], library_hash=i["library_hash"], flags=i["flags"], nvcc=i["nvcc"],
                    nvcc_version=i["nvcc_version"], host_compiler=i["host_compiler"], unit_hashes=i["unit_hashes"])


_lock = threading.Lock()
_backends: Dict[tuple, LrdBackend] = {}


def lrd_backend(device) -> LrdBackend:
    from ..backends.capabilities import query
    from ..backends.cuda import build as B_
    from ..backends.cuda import runtime as R_

    caps = query(device)
    target, why = B_.resolve_target(caps)
    lib = R_.load_bundle("lrd", target)
    with _lock:
        key = (target, lib.ident["kind"], lib.ident["library_hash"])
        if key not in _backends:
            _backends[key] = LrdBackend(lib, target, why)
        return _backends[key]


# --------------------------------------------------------------------------- work lists
class ReductionExec:
    """Items / chunks (forward, dX) and tasks / chunks (dW, dP) of a K16 layout with a reduction base output per block:
    an item is a run of consecutive blocks of one bucket with one w0 (records straddling a stride boundary simply
    start a new item).  Host work, once per layer and device."""

    def __init__(self, NB: int, C: int, K: int, counts, feat, blk_buckets, w0_blocks, wpt: int, dw_warps: int = 8,
                 chunk_weights: int = 1 << 20, max_item_blocks: int = 64):
        t0 = time.perf_counter()
        dev = feat.device
        Q = feat.shape[0]
        b = blk_buckets.to("cpu", torch.int64)
        w0_blocks = w0_blocks.to("cpu", torch.int64)
        counts = counts.to("cpu", torch.int64)
        q, g, j0, nb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        rep = torch.repeat_interleave(torch.arange(len(b)), nb)
        k = torch.arange(len(rep)) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
        fb = g[rep] * NB + j0[rep] + k
        wb = w0_blocks[fb]
        new = torch.ones(len(fb), dtype=torch.bool)
        new[1:] = (rep[1:] != rep[:-1]) | (wb[1:] != wb[:-1]) | (k[1:] % max_item_blocks == 0)
        start = torch.nonzero(new)[:, 0]
        ln = torch.diff(start, append=torch.tensor([len(fb)]))
        self.items = torch.stack([g[rep[start]], fb[start], ln, wb[start]], 1).to(torch.int32)
        iq = q[rep[start]]
        self.chunks = chunk_table(iq, ln, Q, max(1, chunk_weights // (C * K)))
        # tasks: (flat block, 16-neuron group, w0), supergroup order
        jj = fb - g[rep] * NB
        nval = torch.clamp(counts[g[rep]] - jj * C, max=C)
        ncg = (nval + 15) // 16
        rt = torch.repeat_interleave(torch.arange(len(fb)), ncg)
        kt = torch.arange(len(rt)) - torch.repeat_interleave(torch.cumsum(ncg, 0) - ncg, ncg)
        self.tasks = torch.stack([fb[rt], kt, wb[rt]], 1).to(torch.int32)
        tq = q[rep[rt]]
        per_cta = wpt * dw_warps
        q_off = torch.searchsorted(tq.contiguous(), torch.arange(Q + 1))
        n = q_off[1:] - q_off[:-1]
        nch = (n + per_cta - 1) // per_cta
        qs = torch.repeat_interleave(torch.arange(Q), nch)
        kk = torch.arange(len(qs)) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
        lo = q_off[qs] + kk * per_cta
        hi = torch.minimum(lo + per_cta, q_off[qs + 1])
        self.dw_chunks = torch.stack([qs, lo, hi], 1).to(torch.int32)
        self.items, self.chunks = self.items.to(dev).contiguous(), self.chunks.to(dev).contiguous()
        self.tasks, self.dw_chunks = self.tasks.to(dev).contiguous(), self.dw_chunks.to(dev).contiguous()
        self.stats = dict(items=len(self.items), chunks=len(self.chunks), tasks=len(self.tasks),
                          dw_chunks=len(self.dw_chunks), max_item_blocks=max_item_blocks, wpt=wpt,
                          build_seconds=time.perf_counter() - t0)

    def tensors(self):
        return [self.items, self.chunks, self.tasks, self.dw_chunks]


# --------------------------------------------------------------------------- autograd
class LearnedReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, p, red: "LearnedReducer"):
        sp = red.sparse
        rt = sp._runtime()
        B = x.shape[0]
        sp._check_call(B)
        be, ex = red._backend(), red._exec()
        xc = x.to(torch.bfloat16).contiguous()
        wc, bc, pc = w.contiguous(), b.contiguous(), p.contiguous()
        _mark("fwd.begin")
        xt = xc.t().contiguous()
        U = torch.zeros((red.J, B), device=x.device, dtype=torch.float32)
        _mark("fwd.zero")
        a = red._args(rt, xt, wc, bc, pc, B)
        a.k.items, a.k.chunks = ex.items.data_ptr(), ex.chunks.data_ptr()
        a.U = U.data_ptr()
        n_tiles = math.ceil(B / (16 * red.MT))
        be.fwd(a, EPILOGUES[red.epilogue], ex.chunks.shape[0], n_tiles, x.device)
        _mark("fwd.kernel")
        out = U.t().contiguous()
        _mark("fwd.finalize")
        ctx.red = red
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(xc, w, b, p)
        sp.last_info.update(reduction_forward=dict(epilogue=red.epilogue, F=red.F, J=red.J, token_tiles=n_tiles,
                                                   chunks=ex.chunks.shape[0], items=ex.items.shape[0],
                                                   atomics_bound=ex.items.shape[0] * n_tiles * 16 * red.MT * red.F))
        return out

    @staticmethod
    def backward(ctx, dU):
        red: LearnedReducer = ctx.red
        sp = red.sparse
        xc, w, b, p = ctx.saved_tensors
        rt = sp._runtime()
        be, ex = red._backend(), red._exec()
        B, D = xc.shape
        need_x, need_w, need_b, need_p = ctx.needs_input_grad[:4]
        _mark("bwd.begin")
        Dl = torch.zeros((red.J + 2 * red.PAD, B), device=xc.device, dtype=torch.float32)
        Dl[red.PAD:red.PAD + red.J] = dU.float().t()
        _mark("bwd.coef")
        xt = xc.t().contiguous()
        _mark("bwd.layout")
        a = red._args(rt, xt, w.contiguous(), b.contiguous(), p.contiguous(), B)
        a.Dl = Dl.data_ptr()
        split = int(sp.plan.grad_precision == "split")
        dxt = None
        if need_x:
            dxt = torch.zeros((D, B), device=xc.device, dtype=torch.float32)
        _mark("bwd.dx_init")
        if need_x:
            a.k.dXT = dxt.data_ptr()
            a.k.items, a.k.chunks, a.k.dx_group_warps = ex.items.data_ptr(), ex.chunks.data_ptr(), red.dx_group_warps
            be.dx(a, split, red.dx_mt, red.dx_nw, red.dx_reload, ex.chunks.shape[0], math.ceil(B / (16 * red.dx_mt)),
                  xc.device)
        _mark("bwd.dx")
        do_w = need_w or need_b
        mg_w, mg_b = getattr(w, "main_grad", None), getattr(b, "main_grad", None)
        mg_p = getattr(p, "main_grad", None)
        dw = db = dp = None
        if do_w:
            dw = mg_w if mg_w is not None else torch.zeros(w.shape, device=w.device, dtype=torch.float32)
            db = mg_b if mg_b is not None else torch.zeros(b.shape, device=b.device, dtype=torch.float32)
            a.k.dW, a.k.dB = dw.data_ptr(), db.data_ptr()
        if need_p:
            dp = mg_p if mg_p is not None else torch.zeros(p.shape, device=p.device, dtype=torch.float32)
            a.dP = dp.data_ptr()
        a.k.tasks, a.k.chunks = ex.tasks.data_ptr(), ex.dw_chunks.data_ptr()
        n = ex.dw_chunks.shape[0]
        if red.dp_mode == "fused" and do_w and need_p:
            be.dw(a, split, 3, red.wpt, n, xc.device)
            _mark("bwd.dw")
            _mark("bwd.dp")
        else:
            if do_w:
                be.dw(a, split, 1, red.wpt, n, xc.device)
            _mark("bwd.dw")
            if need_p:
                be.dw(a, split, 2, red.wpt, n, xc.device)
            _mark("bwd.dp")
        if need_p and red.needs_grad_mask:
            dp.masked_fill_(~red.member, 0.0)  # irregular shapes: non-member slots of in-range outputs
        sp.last_info.update(backward_backend="lrd", grad_precision=sp.plan.grad_precision,
                            reduction_backward=dict(dp_mode=red.dp_mode, wpt=red.wpt, dx=bool(need_x), dw=bool(do_w),
                                                    dp=bool(need_p), dx_group_warps=red.dx_group_warps,
                                                    dx_tiling=(red.dx_mt, red.dx_nw, red.dx_reload),
                                                    recomputations_of_z=int(bool(need_x)) + (
                                                        1 if red.dp_mode == "fused" else int(do_w) + int(bool(need_p)))))
        dx = dxt.t().to(ctx.x_dtype, memory_format=torch.contiguous_format) if need_x else None
        return (dx, None if (not need_w or mg_w is not None) else dw.to(w.dtype),
                None if (not need_b or mg_b is not None) else db.to(b.dtype),
                None if (not need_p or mg_p is not None) else dp.to(p.dtype), None)


class LearnedReducer(nn.Module):
    """First-stage learned reduction over an existing K16 / C16 kgen layer (``sparse``; its graph, values and support
    are not touched).  ``P``: packed [slots, F] BF16 Parameter (or buffer when ``trainable=False``)."""

    MT = 4  # compile-time token tile of the lrd forward / dX (lrd_k16.cu)

    def __init__(self, sparse: KgenLayer, spec: ReductionSpec, init: str = "random", seed: int = 0, gain: float = 1.0,
                 trainable: bool = True, epilogue: str = "mma_split", dp_mode: str = "fused", wpt: int = 2,
                 dx_group_warps: int = 2, allow_uncovered: bool = False, dx_tiling: tuple = (2, 4, 0)):
        super().__init__()
        L = sparse.layout
        if L.fan_in != 16:
            raise ValueError(f"the lrd kernels are compiled for K = 16 (got K = {L.fan_in})")
        if L.block % 8:
            raise ValueError(f"C = {L.block}: the lrd kernels need C % 8 == 0")
        if epilogue not in EPILOGUES:
            raise ValueError(f"epilogue {epilogue!r} (one of {sorted(EPILOGUES)})")
        if dp_mode not in DP_MODES:
            raise ValueError(f"dp_mode {dp_mode!r} (one of {DP_MODES})")
        if wpt not in (2, 4):
            raise ValueError("wpt must be 2 or 4")
        self.dx_mt, self.dx_nw, self.dx_reload = dx_tiling
        if tuple(dx_tiling) not in {(4, 8, 0), (4, 8, 1), (2, 8, 0), (2, 8, 1), (2, 4, 0), (2, 4, 1)}:
            raise ValueError(f"dx_tiling {dx_tiling}: (MT, NW, reload) is not compiled")
        if dx_group_warps not in (1, 2, 4, 8) or self.dx_nw % dx_group_warps:
            raise ValueError(f"dx_group_warps {dx_group_warps} must divide NW = {self.dx_nw}")
        if sparse.plan.grad_precision not in ("split", "bf16"):
            raise ValueError("grad_precision must be 'split' or 'bf16'")
        self.sparse = sparse
        self.spec = spec
        self.routing = Routing(spec, sparse.counts, L.blocks_per_group, L.block)
        if self.routing.uncovered_neurons and not allow_uncovered:
            raise ValueError(f"{self.routing.uncovered_neurons} hidden neurons are in no window (tail "
                             f"{spec.uncovered_tail}): pass allow_uncovered=True to accept them explicitly")
        self.F, self.J = self.routing.F, spec.J
        self.PAD = self.F
        self.epilogue, self.dp_mode, self.wpt, self.dx_group_warps = epilogue, dp_mode, wpt, dx_group_warps
        self.needs_grad_mask = self.routing.needs_grad_mask
        self.init_rule = dict(init=init, seed=seed, gain=gain)
        device = sparse.indices.device
        P = self._initial(init, seed, gain, device)
        if trainable:
            self.P = nn.Parameter(P)
        else:
            self.register_buffer("P", P)
        self.register_buffer("w0_blocks", self.routing.w0_blocks.to(device), persistent=False)
        self.register_buffer("member", self.routing.member_mask(device) if self.needs_grad_mask else
                             torch.empty(0, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("reduction_spec", torch.frombuffer(bytearray(self._identity().encode()),
                                                                dtype=torch.uint8).clone().to(device))
        self._ex = None
        self._ex_key = None
        self._be = None

    def _identity(self) -> str:
        return json.dumps(dict(schema=SCHEMA, spec=json.loads(self.spec.to_json()), F=self.F,
                               routing=self.routing.fingerprint()), sort_keys=True)

    @torch.no_grad()
    def _initial(self, init, seed, gain, device):
        r = self.routing
        shape = (r.slots, r.F)
        L = self.spec.window
        if init == "mean":
            P = torch.full(shape, 1.0 / L, dtype=torch.bfloat16, device=device)
        elif init == "random":
            P = torch.empty(shape, dtype=torch.bfloat16, device=device)
            g = torch.Generator(device=device).manual_seed(seed)
            std = gain / math.sqrt(L)
            flat = P.view(-1)
            for s0 in range(0, flat.numel(), 1 << 25):
                e = min(s0 + (1 << 25), flat.numel())
                flat[s0:e] = (torch.randn(e - s0, generator=g, device=device) * std).to(torch.bfloat16)
        elif init == "zeros":
            P = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        else:
            raise ValueError(f"init {init!r}")
        # zero every non-member / out-of-range slot (chunked: F small, slots large)
        f_first, f_last, cov = r.f_first.to(device), r.f_last.to(device), r.covered.to(device)
        f = torch.arange(r.F, dtype=torch.int16, device=device)
        for s0 in range(0, r.slots, 1 << 22):
            e = min(s0 + (1 << 22), r.slots)
            keep = (f[None, :] >= f_first[s0:e, None]) & (f[None, :] <= f_last[s0:e, None]) & cov[s0:e, None]
            P[s0:e].masked_fill_(~keep, 0)
        return P

    def _backend(self) -> LrdBackend:
        if self._be is None:
            self._be = lrd_backend(self.sparse.indices.device)
        return self._be

    def _exec(self) -> ReductionExec:
        sp = self.sparse
        key = (sp.indices.device, sp.kg_buckets.data_ptr(), self.w0_blocks.data_ptr(), self.wpt)
        if self._ex is None or self._ex_key != key:
            L = sp.layout
            self._ex = ReductionExec(L.blocks_per_group, L.block, L.fan_in, sp.counts, sp.sg_feat, sp.kg_buckets,
                                     self.w0_blocks, self.wpt)
            self._ex_key = key
        return self._ex

    def _args(self, rt, xt, w, b, p, B) -> LrArgs:
        k = self.sparse._args(rt, xt, w, b, B)
        return LrArgs(k, p.data_ptr(), None, None, None, self.F, self.J, self.PAD, 0)

    def enable_main_grad(self):
        if isinstance(self.P, nn.Parameter) and getattr(self.P, "main_grad", None) is None:
            self.P.main_grad = torch.zeros(self.P.shape, device=self.P.device, dtype=torch.float32)

    def zero_main_grad(self):
        if getattr(self.P, "main_grad", None) is not None:
            self.P.main_grad.zero_()

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError("LearnedReducer takes [M, D]; ReductionModel flattens leading dimensions")
        return LearnedReductionFunction.apply(x, self.sparse.weight, self.sparse.bias, self.P, self)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        got = state_dict.get(prefix + "reduction_spec")
        if got is not None and bytes(got.cpu().tolist()).decode() != self._identity():
            raise ValueError(f"checkpoint reduction {bytes(got.cpu().tolist()).decode()} != layer {self._identity()}")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ex = None

    def execution_record(self, include_attrs: bool = False) -> dict:
        ex = self._exec()
        be = self._backend()
        D, S = self.sparse.layout.input_dim, self.sparse.sg_feat.shape[1]
        map_b = (D * 2 + 15) & ~15
        smem = dict(fwd=map_b + S * 16 * self.MT * 2,
                    dx=map_b + S * 16 * self.dx_mt * 2 + (self.dx_nw // self.dx_group_warps) * S * 16 * self.dx_mt * 4,
                    dw=map_b + 3 * 2 * S * 16 * 2)
        rec = dict(backend="lrd", status="experimental, unvalidated",
                   kernel_tiling=dict(fwd_MT=self.MT, fwd_NW=8, dx_MT=self.dx_mt, dx_NW=self.dx_nw, dx_reload=self.dx_reload,
                                      WPT=self.wpt),
                   epilogue=self.epilogue, dp_mode=self.dp_mode, dx_group_warps=self.dx_group_warps,
                   shared_bytes=smem, work_lists=dict(ex.stats), routing=self.routing.report(),
                   reduction="fp32_atomics (not bitwise deterministic)", build=be.build_info())
        if include_attrs:
            split = int(self.sparse.plan.grad_precision == "split")
            sim = self.epilogue == "simt"
            rec["kernel_attributes"] = dict(
                fwd=be.attributes(0, EPILOGUES[self.epilogue], self.F if sim else 1, 0, smem["fwd"]),
                dx=be.attributes(1, split, self.dx_mt * 16 + self.dx_nw, self.dx_reload, smem["dx"]),
                dw=be.attributes(2, split, 1, self.wpt, smem["dw"]),
                dp=be.attributes(2, split, 2, self.wpt, smem["dw"]),
                dw_dp_fused=be.attributes(2, split, 3, self.wpt, smem["dw"]))
        return rec


# --------------------------------------------------------------------------- virtual stride view (A2, A5)
@torch.no_grad()
def stride_view(base: KgenLayer, s: int, dx_variant: str = "mt2_nw4_wg2") -> KgenLayer:
    """A KgenLayer whose 'outputs' are the stride blocks [q s, (q + 1) s) of ``base``'s neuron stream, over the SAME
    storage (weight / bias views, same records and input sets): its Mean is the fixed mean of s neurons, computed by the
    unchanged K16 kernels.  Requires contiguous storage: equal per-output counts, C divides them, C divides s, s
    divides H.  This does not regenerate a graph: the support fingerprint and values are unchanged."""
    L = base.layout
    H, C = L.hidden_dim, L.block
    n = base.counts
    if int(n.min()) != int(n.max()) or int(n[0]) % C or s % C or H % s:
        raise ValueError(f"stride_view needs equal per-output counts divisible by C = {C} and s divisible by C with "
                         f"s | H (H = {H}, s = {s})")
    G2, NB2 = H // s, s // C
    V = KLayout(L.input_dim, H, G2, L.fan_in, C)
    idx = base.indices.view(G2, NB2, L.fan_in)
    w = base.weight.detach().view(G2, NB2, C, L.fan_in)
    b = base.bias.detach().view(G2, NB2, C)
    bk = base.kg_buckets.to(torch.int64)
    nb_old = L.blocks_per_group
    rep = torch.repeat_interleave(torch.arange(len(bk), device=bk.device), bk[:, 3])
    k = torch.arange(len(rep), device=bk.device) - torch.repeat_interleave(torch.cumsum(bk[:, 3], 0) - bk[:, 3], bk[:, 3])
    fb = bk[rep, 1] * nb_old + bk[rep, 2] + k
    q, g2, j2 = bk[rep, 0], fb // NB2, fb % NB2
    order = torch.argsort(q * (G2 * NB2 + 1) + fb)
    q, g2, j2, fb = q[order], g2[order], j2[order], fb[order]
    new = torch.ones(len(fb), dtype=torch.bool, device=fb.device)
    new[1:] = (q[1:] != q[:-1]) | (g2[1:] != g2[:-1]) | (fb[1:] != fb[:-1] + 1)
    st = torch.nonzero(new)[:, 0]
    ln = torch.diff(st, append=torch.tensor([len(fb)], device=st.device))
    buckets = torch.stack([q[st], g2[st], j2[st], ln], 1).to(torch.int32).contiguous()
    sp = base.spec
    spec = KG.GraphSpec(L.fan_in, C, sp.S, sp.R, sp.seed, sp.quantum, sp.generation, source=f"stride-view-s{s}:{sp.source}")
    lay = KgenLayer(L.input_dim, H, G2, L.fan_in, C, base.connectivity, device=base.indices.device, plan=base.plan,
                    graph=(idx, base.sg_feat, buckets, spec), weights=(w, b))
    lay.dx_variant = dx_variant
    lay.graph_family = "stride-view"
    lay.envelope_KC = base.envelope_KC
    if lay.weight.data_ptr() != base.weight.data_ptr() or lay.indices.data_ptr() != base.indices.data_ptr():
        raise AssertionError("stride_view must not copy the base storage")
    if V.weight_shape != tuple(lay.weight.shape):
        raise AssertionError("stride_view layout")
    return lay


# --------------------------------------------------------------------------- second stage (three-stage follow-up)
class _Stage2Function(torch.autograd.Function):
    """Independent-weight sliding windows on the compact features, evaluated in token tiles (no M x J2 x L2 tensor,
    no autograd-saved unfolded windows)."""

    @staticmethod
    def forward(ctx, a, P2, s2: int, tile: int):
        ctx.s2, ctx.tile = s2, tile
        ctx.save_for_backward(a, P2)
        J2, L2 = P2.shape
        out = torch.zeros((a.shape[0], J2), device=a.device, dtype=a.dtype)
        with torch.no_grad():
            for m0 in range(0, a.shape[0], tile):
                t = a[m0:m0 + tile].unfold(1, L2, s2)[:, :J2]  # view [m, J2, L2]
                out[m0:m0 + tile] = (t * P2[None]).sum(-1)
        return out

    @staticmethod
    def backward(ctx, d2):
        a, P2 = ctx.saved_tensors
        J2, L2 = P2.shape
        da = torch.zeros_like(a) if ctx.needs_input_grad[0] else None
        dP2 = torch.zeros_like(P2) if ctx.needs_input_grad[1] else None
        with torch.no_grad():
            pos = (torch.arange(J2, device=a.device) * ctx.s2)[:, None] + torch.arange(L2, device=a.device)[None, :]
            for m0 in range(0, a.shape[0], ctx.tile):
                ad, dd = a[m0:m0 + ctx.tile], d2[m0:m0 + ctx.tile]
                if dP2 is not None:
                    t = ad.unfold(1, L2, ctx.s2)[:, :J2]
                    dP2 += (dd[:, :, None] * t).sum(0)
                if da is not None:
                    v = (dd[:, :, None] * P2[None])  # [m, J2, L2]
                    da[m0:m0 + ctx.tile].index_add_(1, pos.reshape(-1), v.reshape(v.shape[0], -1))
        return da, dP2, None, None


class SecondStage(nn.Module):
    def __init__(self, J1: int, L2: int, s2: int, device, seed: int = 0, psi: str = "identity", tile: int = 256):
        super().__init__()
        if J1 < L2 or (J1 - L2) % s2:
            raise ValueError(f"second stage L2 = {L2}, s2 = {s2} does not exactly cover its {J1} inputs")
        self.J2 = (J1 - L2) // s2 + 1
        self.L2, self.s2, self.tile = L2, s2, tile
        g = torch.Generator(device=device).manual_seed(seed)
        self.P2 = nn.Parameter(torch.randn(self.J2, L2, device=device, generator=g) / math.sqrt(L2))
        self.c2 = nn.Parameter(torch.zeros(self.J2, device=device))
        self.psi = psi

    def forward(self, a):
        u = _Stage2Function.apply(a, self.P2, self.s2, self.tile) + self.c2
        return apply_psi(u, self.psi)


def apply_psi(u, psi: str):
    if psi == "identity":
        return u
    if psi == "gelu":
        return TF.gelu(u, approximate="none")  # exact GELU (erf), recorded in ReducerConfig
    if psi == "relu":
        return torch.relu(u)
    raise ValueError(f"psi {psi!r}")


# --------------------------------------------------------------------------- model matrix
@dataclass(frozen=True)
class ReducerConfig:
    """One reducer of the experiment matrix.  ``kind``: "group_mean" (A0 / A1), "block_mean" (A2 / A5 on a stride view),
    "learned" (lrd kernels; ``trainable_p=False`` runs the fixed-coefficient path of the same kernels)."""

    id: str
    kind: str
    L: int = 0
    s: int = 0
    channels: int = 1
    psi: str = "identity"
    init: str = "random"  # learned P: "random" (signed, std gain / sqrt(L)) or "mean" (1 / L)
    trainable_p: bool = True
    dense: bool = True
    dense_init: str = "normal"  # "normal" (std 1 / sqrt(J), rule shared by the J != O reducers) or "identity"
    epilogue: str = "mma_split"
    dp_mode: str = "fused"
    second: Optional[tuple] = None  # (L2, s2, psi2)
    backend_note: str = ""

    def describe(self) -> dict:
        d = asdict(self)
        d["gelu"] = "exact erf GELU" if self.psi == "gelu" else None
        return d


def core_configs(H: int, L: int = 10_240, s: int = 2_048, **kw) -> Dict[str, ReducerConfig]:
    c = {
        "A0": ReducerConfig("A0", "group_mean", dense=False),
        "A1": ReducerConfig("A1", "group_mean", dense=True, dense_init="identity"),
        "A2": ReducerConfig("A2", "block_mean", L=s, s=s),
        "A3": ReducerConfig("A3", "learned", L=s, s=s, **kw),
        "A3mean": ReducerConfig("A3mean", "learned", L=s, s=s, init="mean", **kw),
        "A3gelu": ReducerConfig("A3gelu", "learned", L=s, s=s, psi="gelu", **kw),
        "A4": ReducerConfig("A4", "learned", L=L, s=s, **kw),
        "A4mean": ReducerConfig("A4mean", "learned", L=L, s=s, init="mean", **kw),
        "A4gelu": ReducerConfig("A4gelu", "learned", L=L, s=s, psi="gelu", **kw),
        "A5": ReducerConfig("A5", "block_mean", L=L, s=s),
        "A2gelu": ReducerConfig("A2gelu", "block_mean", L=s, s=s, psi="gelu"),
        "A5gelu": ReducerConfig("A5gelu", "block_mean", L=L, s=s, psi="gelu"),
    }
    if H % L == 0:
        c["MC"] = ReducerConfig("MC", "learned", L=L, s=L, channels=L // s, **kw)
    c["TS_id"] = ReducerConfig("TS_id", "learned", L=L, s=s, second=(256, 4, "identity"), **kw)
    c["TS_gelu"] = ReducerConfig("TS_gelu", "learned", L=L, s=s, psi="gelu", second=(256, 4, "gelu"), **kw)
    return c


class ReductionModel(nn.Module):
    """x [..., D] -> y [..., O] for one ``ReducerConfig`` over a base K16 layer.  The base layer's weight / bias storage is
    used directly (A2 / A5: views of it); give every paired model its own copy (``clone_base``)."""

    def __init__(self, base: KgenLayer, cfg: ReducerConfig, output_dim: int = 1024, seed: int = 0, gamma: float = 1.0,
                 p_seed: Optional[int] = None, train_sparse: bool = True, **reducer_kw):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.base_meta = dict(getattr(base, "cover_meta", {}) or {})
        L = base.layout
        self.input_dim, self.H = L.input_dim, L.hidden_dim
        self.base_shape = tuple(base.weight.shape)
        dev = base.indices.device
        if cfg.kind == "group_mean":
            self.sparse = base
            J = L.groups
            self.reducer = None
        elif cfg.kind == "block_mean":
            self.sparse = stride_view(base, cfg.s)
            self.m = cfg.L // cfg.s
            if cfg.L % cfg.s:
                raise ValueError("block_mean: L must be a multiple of s (use the fixed-coefficient lrd path otherwise)")
            J = (self.H - cfg.L) // cfg.s + 1
            self.reducer = None
        elif cfg.kind == "learned":
            self.sparse = base
            spec = ReductionSpec(self.H, cfg.L, cfg.s, cfg.channels)
            self.reducer = LearnedReducer(base, spec, init=cfg.init, seed=seed + 1000 if p_seed is None else p_seed,
                                          trainable=cfg.trainable_p, epilogue=cfg.epilogue, dp_mode=cfg.dp_mode,
                                          **reducer_kw)
            J = spec.J
        else:
            raise ValueError(f"kind {cfg.kind!r}")
        self.J1 = J
        self.has_c = cfg.kind == "learned" or cfg.psi != "identity"
        self.c = nn.Parameter(torch.zeros(J, device=dev)) if self.has_c else None
        self.register_buffer("gamma", torch.tensor(float(gamma), device=dev))
        self.second = None
        if cfg.second is not None:
            self.second = SecondStage(J, cfg.second[0], cfg.second[1], dev, seed=seed + 2000, psi=cfg.second[2])
            J = self.second.J2
        self.J = J
        if cfg.dense:
            g = torch.Generator(device=dev).manual_seed(seed + 3000)
            if cfg.dense_init == "identity":
                if J != output_dim:
                    raise ValueError("identity readout needs J == O")
                V = torch.eye(J, device=dev)
            else:
                V = torch.randn(J, output_dim, device=dev, generator=g) / math.sqrt(J)
            self.V = nn.Parameter(V)
            self.d = nn.Parameter(torch.zeros(output_dim, device=dev))
        else:
            self.V, self.d = None, None
        self.sparse.enable_main_grad()
        if self.reducer is not None:
            self.reducer.enable_main_grad()
        self.set_train_sparse(train_sparse)

    def set_train_sparse(self, flag: bool):
        self.sparse.weight.requires_grad_(flag)
        self.sparse.bias.requires_grad_(flag)

    # parameter groups: FusedAdamW (BF16 + FP32 main_grad) and torch AdamW (FP32)
    def sparse_parameters(self):
        return [p for p in (self.sparse.weight, self.sparse.bias) if p.requires_grad]

    def reducer_parameters(self):
        return [self.reducer.P] if self.reducer is not None and isinstance(self.reducer.P, nn.Parameter) else []

    def dense_parameters(self):
        out = [p for p in (self.c, self.V, self.d) if p is not None]
        if self.second is not None:
            out += [self.second.P2, self.second.c2]
        return out

    def zero_main_grad(self):
        self.sparse.zero_main_grad()
        if self.reducer is not None:
            self.reducer.zero_main_grad()

    def features(self, xf):
        """U [M, J1] before c / psi / gamma (no M x H intermediate)."""
        cfg = self.cfg
        if cfg.kind == "learned":
            return self.reducer(xf)
        U = self.sparse(xf)
        if cfg.kind == "block_mean" and self.m > 1:
            U = TF.avg_pool1d(U[:, None], self.m, 1)[:, 0]  # the mean of m equal blocks is the mean of the window
        return U

    def forward(self, x):
        lead = x.shape[:-1]
        U = self.features(x.reshape(-1, x.shape[-1]))
        if self.c is not None:
            U = U + self.c
        A = apply_psi(U, self.cfg.psi)
        if float(self.gamma) != 1.0:
            A = A * self.gamma
        if self.second is not None:
            A = self.second(A)
        _mark("readout.begin")
        y = A @ self.V + self.d if self.V is not None else A
        _mark("readout.end")
        return y.reshape(*lead, y.shape[-1])

    # -------------------------------------------------- identity and accounting
    def weight_counts(self) -> dict:
        P = self.reducer.routing.packed_valid if self.reducer is not None else 0
        s2 = self.second.J2 * self.second.L2 if self.second is not None else 0
        V = self.J * self.output_dim if self.V is not None else 0
        biases = (self.J1 if self.has_c else 0) + (self.second.J2 if self.second is not None else 0) + (
            self.output_dim if self.V is not None else 0)
        return dict(sparse_weights=16 * self.H, sparse_biases=self.H, P=P, P2=s2, V=V, added_weights=P + s2 + V,
                    added_biases=biases, intermediate=self.J1, final_intermediate=self.J)

    def fingerprints(self) -> dict:
        """Immutable identity of the sparse projection in the base layout (identical across paired reducers)."""
        from .kgen_lowered import _fp_tensor

        idx = self.sparse.indices.reshape(-1, 16)
        feat = self.sparse.sg_feat
        return dict(indices=_fp_tensor(idx)[0], feat=_fp_tensor(feat)[0], support=self.base_meta.get("support_fingerprint"))

    def memory_report(self) -> dict:
        from ..memory_accounting import layer_storage

        sp = self.sparse
        P = [self.reducer.P] if self.reducer is not None else []
        groups = dict(sparse_weights=[sp.weight], sparse_bias=[sp.bias],
                      sparse_main_grad=[t for t in (getattr(sp.weight, "main_grad", None), getattr(sp.bias, "main_grad", None))
                                        if t is not None],
                      reducer_P=P,
                      reducer_main_grad=[p.main_grad for p in P if getattr(p, "main_grad", None) is not None],
                      dense_fp32=self.dense_parameters(),
                      metadata=[sp.indices, sp.sg_feat, sp.kg_buckets, sp.kg_spec, sp.counts, sp.counts_f])
        if sp._rt is not None:
            s = sp._rt.sg
            groups["metadata"] += [s.items, s.chunks, s.tasks, s.dw_chunks]
        if self.reducer is not None and self.reducer._ex is not None:
            groups["reducer_work_lists"] = self.reducer._ex.tensors() + [self.reducer.w0_blocks]
        return layer_storage(groups)

    def explain(self) -> dict:
        d = dict(config=self.cfg.describe(), H=self.H, J1=self.J1, J=self.J, gamma=float(self.gamma),
                 weights=self.weight_counts(), sparse_family=self.sparse.graph_family,
                 sparse_dx_request=self.sparse.dx_variant, status="experimental, unvalidated")
        if self.reducer is not None:
            d["reducer"] = self.reducer.execution_record()
        return d


@torch.no_grad()
def clone_base(base: KgenLayer) -> KgenLayer:
    """An independent copy of a K16 base layer with its own weight / bias storage and the same immutable graph buffers
    (shared, not copied): paired reducers train from identical sparse values on an identical graph."""
    L = base.layout
    lay = type(base)(L.input_dim, L.hidden_dim, L.groups, L.fan_in, L.block, base.connectivity,
                     device=base.indices.device, plan=base.plan,
                     graph=(base.indices, base.sg_feat, base.kg_buckets, base.spec),
                     weights=(base.weight.detach().clone(), base.bias.detach().clone()))
    if hasattr(base, "env"):
        lay.register_buffer("env", base.env)
    for a in ("cover_meta", "envelope_KC", "dx_variant", "graph_stats"):
        if hasattr(base, a):
            setattr(lay, a, getattr(base, a))
    return lay


@torch.no_grad()
def calibrate_gamma(model: ReductionModel, x: torch.Tensor) -> float:
    """gamma = 1 / RMS(psi(u)) on a TRAINING-input calibration batch (rule shared by all J != O / non-identity readout
    reducers; frozen afterwards and recorded).  Never fit on validation / test data."""
    g0 = model.gamma.clone()
    model.gamma.fill_(1.0)
    U = model.features(x.reshape(-1, x.shape[-1]))
    A = apply_psi(U + model.c if model.c is not None else U, model.cfg.psi)
    rms = float(A.float().square().mean().sqrt())
    model.gamma.copy_(g0)
    return 1.0 / max(rms, 1e-12)


@torch.no_grad()
def feature_stats(U: torch.Tensor) -> dict:
    a = U.float()
    return dict(mean=float(a.mean()), rms=float(a.square().mean().sqrt()), var=float(a.var()),
                max_abs=float(a.abs().max()), zero_fraction=float((a == 0).float().mean()))


__all__ = ["ReducerConfig", "ReductionModel", "core_configs", "LearnedReducer", "stride_view", "clone_base",
           "calibrate_gamma", "feature_stats", "ReductionExec", "lrd_backend"]
