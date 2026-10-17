"""Custom autograd: forward saves no hidden activations; backward recomputes.

What ``forward`` saves (``ctx``):
  * references to x, W, b (needed anyway to recompute neuron activations)
  * the compact pooled value ``pre`` [B, U] (minimal_stats / save_winners)
  * optionally: winner ids [B, U] (max, save_winners), Top-K slot ids
    [B, U, k] (topk, save_winners), the log2-normaliser [B, U]
    (softmax-weighted, minimal_stats)
  * for a semantic two-level model: the level-1 features [B, G, S] if
    ``level1_checkpoint == "save"``

Nothing of size [B, H] is ever allocated.  Weight / bias gradients are
accumulated in FP32 buffers: ``W.main_grad`` / ``b.main_grad`` when present
(memory-conscious training, gradient returned as None), otherwise temporary
buffers returned through autograd in the parameter dtype.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import LevelSpec, PhysicalPlan, PoolingConfig
from .kernels import ops
from .layout import SparseLayout
from .reference.pytorch_reference import level2_forward

# measured defaults (see results/policy_choice.md); used by checkpoint_policy="auto"
AUTO_POLICY = {
    "max": "save_winners",
    "topk_mean": "save_winners",
    "mean": "minimal_stats",
    "logmeanexp": "minimal_stats",
    "power_mean": "minimal_stats",
    "softmax_weighted": "minimal_stats",
}


CHECKPOINT_BUDGET = 2 << 30  # bytes of saved winner ids before falling back to recomputation
RECOMPUTE_CHUNK_BYTES = 512 << 20  # id-tensor size per token chunk when winners are recomputed


def fit_policy(rt: "Level1Runtime", policy: str, batch: int) -> str:
    """save_winners stores B * U * k int32 ids; with many semantic shards that can
    reach GBs (B=8192, 49152 units, k=8: 12.9 GB).  Above the checkpoint budget
    the winners are recomputed in backward instead (reported in last_info)."""
    lv = rt.level
    if policy == "save_winners" and lv.type in ("max", "topk_mean"):
        k = lv.k if lv.type == "topk_mean" else 1
        need = batch * rt.units.U * k * 4
        if need > CHECKPOINT_BUDGET:
            rt.last_info["policy_fallback"] = f"save_winners needs {need / 2**30:.1f} GiB of ids -> minimal_stats"
            return "minimal_stats"
    rt.last_info.pop("policy_fallback", None)
    return policy


def resolve_policy(level: LevelSpec, requested: str) -> str:
    if requested == "auto":
        return AUTO_POLICY[level.type]
    if requested == "save_winners" and level.type not in ("max", "topk_mean"):
        return "minimal_stats"  # no winners to save; documented fallback
    return requested


class Level1Runtime:
    """Executes the fused level-1 kernels for one layer.  Holds no tensors of
    size [B, H]; all per-call state lives in the autograd ctx."""

    def __init__(self, layout: SparseLayout, idx: torch.Tensor, level: LevelSpec,
                 plan: PhysicalPlan, nps: Optional[int], workspace_budget: int,
                 backend: Optional[object] = None):
        self.layout = layout
        self.idx = idx
        self.level = level
        self.plan = plan
        self.units = ops.make_units(layout, nps, idx.device)
        self.counts_i32 = layout.counts_tensor(idx.device).contiguous()
        self.workspace_budget = workspace_budget
        self.backend = backend  # optional CUDA backend (kernels/cuda_ops.py)
        self.last_info: Dict = {}

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, w, b, track_ids: bool) -> Dict[str, torch.Tensor]:
        """Return merged statistics: 'pre' [B, U] and operator extras."""
        B = x.shape[0]
        lv = self.level
        track_ids = track_ids and lv.type in ("max", "topk_mean")
        if self.backend is not None and self.backend.supports_forward(self, track_ids):
            out = self.backend.forward(self, x, w, b, track_ids)
            self.last_info["forward_backend"] = "cuda"
            return out
        ps = ops.choose_splits(B, self.units, lv, self.plan, track_ids, self.workspace_budget)
        x_op, sxf, sxb = ops.input_operand(x, self.plan)
        st = ops.fused_forward_states(x_op, sxf, sxb, B, self.idx, w, b, self.counts_i32,
                                      self.layout, lv, self.units, self.plan, track_ids, ps)
        self.last_info.update(forward_backend="triton", phys_splits=ps,
                              workspace_bytes=sum(t.numel() * t.element_size() for t in st.values()))
        out = ops.merge_finalize(st, lv, self.units.counts)
        del st
        return out

    # --------------------------------------------------------------- backward
    def backward(self, x, w, b, g_pre: torch.Tensor, saved: Dict[str, torch.Tensor], policy: str,
                 dw_buf: torch.Tensor, db_buf: torch.Tensor, recomputed: bool = False) -> torch.Tensor:
        """Accumulate dW/db into FP32 buffers, return dX (FP32, [B, D])."""
        lv = self.level
        B, D = x.shape
        info = self.last_info
        info["policy"] = policy
        # recompute statistics that were not saved
        need = {
            "max": "winner",
            "topk_mean": "topk_slot",
            "mean": None,
            "logmeanexp": "pre",
            "power_mean": "pre",
            "softmax_weighted": "L2",
        }[lv.type]
        stats = dict(saved)
        if lv.type in ("max", "topk_mean") and need not in stats:
            k = lv.k if lv.type == "topk_mean" else 1
            if B * self.units.U * k * 4 > CHECKPOINT_BUDGET:
                info["recomputed_stats"] = True
                return self._chunked_event_backward(x, w, b, g_pre, dw_buf, db_buf, k)
        if need is not None and need not in stats:
            recomputed = True
            stats.update(self.forward(x, w, b, track_ids=True))
        info["recomputed_stats"] = recomputed

        if self.backend is not None and self.backend.supports_backward(self):
            info["backward_backend"] = "cuda"
            return self.backend.backward(self, x, w, b, g_pre, stats, dw_buf, db_buf)
        info["backward_backend"] = "triton"

        x_op, sxf, sxb = ops.input_operand(x, self.plan)
        if self.plan.input_layout == "feature_major":
            dx = torch.zeros((D, B), device=x.device, dtype=torch.float32)
            sdf, sdb = dx.stride(0), dx.stride(1)
        else:
            dx = torch.zeros((B, D), device=x.device, dtype=torch.float32)
            sdf, sdb = dx.stride(1), dx.stride(0)
        if lv.type == "max":
            ops.launch_event_backward(x_op, sxf, sxb, self.idx, w, b, self.layout, lv, self.units,
                                      stats["winner"][..., None], g_pre[..., None],
                                      dx, sdf, sdb, dw_buf, db_buf)
        elif lv.type == "topk_mean":
            kk = torch.clamp(self.units.counts, max=lv.k)[None, :, None]
            ops.launch_event_backward(x_op, sxf, sxb, self.idx, w, b, self.layout, lv, self.units,
                                      stats["topk_slot"], (g_pre[..., None] / kk).expand(-1, -1, lv.k),
                                      dx, sdf, sdb, dw_buf, db_buf)
        else:
            c1, c2, c3 = ops.dense_coefficients(lv, g_pre, stats["pre"] if "pre" in stats else None,
                                                self.units.counts, stats.get("L2"))
            ops.launch_dense_backward(x_op, sxf, sxb, B, self.idx, w, b, self.counts_i32, self.layout,
                                      lv, self.units, self.plan, c1, c2, c3, dx, sdf, sdb, dw_buf, db_buf)
        return dx.t() if self.plan.input_layout == "feature_major" else dx


def _event_backward_chunk(rt, xs, w, b, g, stats, dx_rows, dw_buf, db_buf):
    """Sparse winner-event backward for a slice of tokens; dx_rows is the FP32
    [b, D] destination view."""
    lv = rt.level
    x_op, sxf, sxb = ops.input_operand(xs, rt.plan)
    sdf, sdb = dx_rows.stride(1), dx_rows.stride(0)
    if lv.type == "max":
        ops.launch_event_backward(x_op, sxf, sxb, rt.idx, w, b, rt.layout, lv, rt.units,
                                  stats["winner"][..., None], g[..., None], dx_rows, sdf, sdb, dw_buf, db_buf)
    else:
        kk = torch.clamp(rt.units.counts, max=lv.k)[None, :, None]
        ops.launch_event_backward(x_op, sxf, sxb, rt.idx, w, b, rt.layout, lv, rt.units,
                                  stats["topk_slot"], (g[..., None] / kk).expand(-1, -1, lv.k),
                                  dx_rows, sdf, sdb, dw_buf, db_buf)


def _chunked_event_backward(self, x, w, b, g_pre, dw_buf, db_buf, k):
    """Winner recomputation in token chunks so that the [b, U, k] id tensor stays
    within RECOMPUTE_CHUNK_BYTES (microbatching; total extra work = one forward)."""
    B, D = x.shape
    bc = max(32, (min(RECOMPUTE_CHUNK_BYTES, CHECKPOINT_BUDGET) // (self.units.U * k * 4)) // 32 * 32)
    dx = torch.zeros((B, D), device=x.device, dtype=torch.float32)
    for s0 in range(0, B, bc):
        xs = x[s0:s0 + bc]
        st = self.forward(xs, w, b, track_ids=True)
        _event_backward_chunk(self, xs, w, b, g_pre[s0:s0 + bc].contiguous(), st, dx[s0:s0 + bc], dw_buf, db_buf)
        del st
    self.last_info.update(backward_backend="triton", backward_token_chunks=-(-B // bc))
    return dx


Level1Runtime._chunked_event_backward = _chunked_event_backward


def _compute_operands(x, w, b, compute_dtype):
    """Kernels consume BF16/FP16 operands; FP32 inputs are cast (tests)."""
    xc = x.to(compute_dtype).contiguous()
    wc = w if w.dtype == compute_dtype else w.to(compute_dtype)
    bc = b if b.dtype == compute_dtype else b.to(compute_dtype)
    return xc, wc.contiguous(), bc.contiguous()


def _grad_buffers(w, b):
    mg_w = getattr(w, "main_grad", None)
    mg_b = getattr(b, "main_grad", None)
    if mg_w is not None and mg_b is not None:
        return mg_w, mg_b, True
    return (torch.zeros(w.shape, device=w.device, dtype=torch.float32),
            torch.zeros(b.shape, device=b.device, dtype=torch.float32), False)


def _saved_stats(level: LevelSpec, out: Dict[str, torch.Tensor], policy: str) -> Dict[str, torch.Tensor]:
    if policy == "recompute_stats":
        return {}
    keep = {"pre": out["pre"]}
    if policy == "save_winners":
        if "winner" in out:
            keep["winner"] = out["winner"]
        if "topk_slot" in out:
            keep["topk_slot"] = out["topk_slot"]
    if level.type == "softmax_weighted" and policy == "minimal_stats":
        keep["L2"] = out["L2"]
    return keep


class SparsePoolFunction(torch.autograd.Function):
    """Single-level: y [B, G] = out_act(pool(sparse_projection(x)))."""

    @staticmethod
    def forward(ctx, x, w, b, rt: Level1Runtime, policy: str, compute_dtype):
        policy = fit_policy(rt, policy, x.shape[0])
        xc, wc, bc = _compute_operands(x, w, b, compute_dtype)
        lv = rt.level
        out = rt.forward(xc, wc, bc, track_ids=(policy == "save_winners"))
        pre = out["pre"]
        y = ops.act_torch(pre, lv.out_act_id)
        saved = _saved_stats(lv, out, policy)
        ctx.rt, ctx.policy, ctx.compute_dtype = rt, policy, compute_dtype
        ctx.saved_keys = list(saved.keys())
        ctx.save_for_backward(x, w, b, *saved.values())
        rt.last_info["checkpoint_bytes"] = sum(t.numel() * t.element_size() for t in saved.values())
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, *svals = ctx.saved_tensors
        saved = dict(zip(ctx.saved_keys, svals))
        rt: Level1Runtime = ctx.rt
        lv = rt.level
        xc, wc, bc = _compute_operands(x, w, b, ctx.compute_dtype)
        recomputed = "pre" not in saved
        if recomputed:
            saved.update(rt.forward(xc, wc, bc, track_ids=True))
        g_pre = dy.float() * ops.dact_torch(saved["pre"], lv.out_act_id)
        dw_buf, db_buf, use_main = _grad_buffers(w, b)
        dx = rt.backward(xc, wc, bc, g_pre.contiguous(), saved, ctx.policy, dw_buf, db_buf, recomputed)
        gw = None if use_main else dw_buf.to(w.dtype)
        gb = None if use_main else db_buf.to(b.dtype)
        return dx.to(x.dtype), gw, gb, None, None, None


class SemanticPoolFunction(torch.autograd.Function):
    """Two-level: F = level1(x) [B, G, S]; y = level2(transform(F)).

    level1_checkpoint == "save":      F (and level-1 stats) kept from forward
    level1_checkpoint == "recompute": only x/W/b kept; backward re-runs the
                                      level-1 streaming reduction to rebuild F
    """

    @staticmethod
    def forward(ctx, x, w, b, rt: Level1Runtime, cfg: PoolingConfig, policy: str, compute_dtype,
                tnames: List[str], *tparams):
        policy = fit_policy(rt, policy, x.shape[0])
        xc, wc, bc = _compute_operands(x, w, b, compute_dtype)
        l1 = rt.level
        out = rt.forward(xc, wc, bc, track_ids=(policy == "save_winners"))
        B = x.shape[0]
        G, S = rt.layout.groups, rt.units.S
        pre = out["pre"]
        F = ops.act_torch(pre, l1.out_act_id).reshape(B, G, S)
        tp = dict(zip(tnames, tparams))
        with torch.no_grad():
            y = level2_forward(F, cfg, tp)
        if cfg.level1_checkpoint == "save":
            saved = _saved_stats(l1, out, policy if policy != "recompute_stats" else "minimal_stats")
        else:
            saved = {}
        ctx.rt, ctx.cfg, ctx.policy, ctx.compute_dtype, ctx.tnames = rt, cfg, policy, compute_dtype, tnames
        ctx.saved_keys = list(saved.keys())
        ctx.n_t = len(tparams)
        ctx.save_for_backward(x, w, b, *tparams, *saved.values())
        rt.last_info["checkpoint_bytes"] = sum(t.numel() * t.element_size() for t in saved.values())
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, *rest = ctx.saved_tensors
        tparams = rest[: ctx.n_t]
        saved = dict(zip(ctx.saved_keys, rest[ctx.n_t:]))
        rt: Level1Runtime = ctx.rt
        l1 = rt.level
        xc, wc, bc = _compute_operands(x, w, b, ctx.compute_dtype)
        recomputed = "pre" not in saved
        if recomputed:  # rebuild level-1 features by recomputation
            saved.update(rt.forward(xc, wc, bc, track_ids=(ctx.policy == "save_winners")))
        B = x.shape[0]
        G, S = rt.layout.groups, rt.units.S
        pre = saved["pre"]
        with torch.enable_grad():
            Fv = ops.act_torch(pre, l1.out_act_id).reshape(B, G, S).detach().requires_grad_(True)
            tp = [p.detach().requires_grad_(True) for p in tparams]
            y = level2_forward(Fv, ctx.cfg, dict(zip(ctx.tnames, tp)))
            grads = torch.autograd.grad(y, [Fv] + tp, dy, allow_unused=True)
        dF = grads[0].reshape(B, G * S)
        g_pre = dF.float() * ops.dact_torch(pre, l1.out_act_id)
        policy = ctx.policy if ctx.policy != "recompute_stats" else "minimal_stats"
        dw_buf, db_buf, use_main = _grad_buffers(w, b)
        dx = rt.backward(xc, wc, bc, g_pre.contiguous(), saved, policy, dw_buf, db_buf, recomputed)
        gw = None if use_main else dw_buf.to(w.dtype)
        gb = None if use_main else db_buf.to(b.dtype)
        return (dx.to(x.dtype), gw, gb, None, None, None, None, None, *grads[1:])
