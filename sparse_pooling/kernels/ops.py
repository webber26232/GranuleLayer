"""Host-side launchers, state merging and backward-coefficient preparation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import triton

from ..config import (
    DEFAULT_WORKSPACE_BUDGET,
    LevelSpec,
    PhysicalPlan,
    human_bytes,
    state_bytes,
)
from ..layout import SparseLayout
from . import triton_fused as tk

LOG2E = 1.4426950408889634
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)


class WorkspaceBudgetError(RuntimeError):
    pass


@dataclass
class Units:
    """Level-1 pooling units: unit u = g * S + s covers blocks
    [s * BPS, (s + 1) * BPS) of group g."""

    S: int
    BPS: int
    nps: Optional[int]
    counts: torch.Tensor  # [U] float32 valid neurons per unit

    @property
    def U(self) -> int:
        return self.counts.numel()


def make_units(layout: SparseLayout, nps: Optional[int], device) -> Units:
    n_g = layout.counts_tensor(device).to(torch.float32)
    if nps is None:
        return Units(1, layout.blocks_per_group, None, n_g)
    if nps % (4 * layout.block) != 0:
        raise ValueError(f"neurons_per_shard must be a multiple of 4*C = {4 * layout.block}")
    S = math.ceil(layout.max_group / nps)
    s = torch.arange(S, device=device, dtype=torch.float32)
    cnt = torch.clamp(n_g[:, None] - s[None, :] * nps, min=0, max=nps)
    if bool((cnt == 0).any()):
        raise ValueError("a semantic shard would be empty; choose a different neurons_per_shard")
    return Units(S, nps // layout.block, nps, cnt.reshape(-1))


def input_operand(x: torch.Tensor, plan: PhysicalPlan):
    """Return (tensor, stride_feature, stride_token) for the gather."""
    if plan.input_layout == "feature_major":
        xt = x.t().contiguous()  # [D, B]  (counted in end-to-end timing)
        return xt, xt.stride(0), xt.stride(1)
    xc = x.contiguous()
    return xc, xc.stride(1), xc.stride(0)


def _p_int(level: LevelSpec) -> int:
    if level.type == "power_mean" and float(level.p).is_integer() and int(level.p) in (1, 2, 4, 8):
        return int(level.p)
    return 0


def kernel_scale(level: LevelSpec) -> float:
    if level.type == "logmeanexp":
        return LOG2E / level.temperature
    if level.type == "softmax_weighted":
        return LOG2E * level.beta
    return 1.0


def choose_splits(B: int, units: Units, level: LevelSpec, plan: PhysicalPlan, track_ids: bool,
                  budget: int = DEFAULT_WORKSPACE_BUDGET, sm_count: int = 82) -> int:
    """Physical splits: enough programs to fill the GPU, within the workspace budget."""
    if plan.phys_splits > 0:
        ps = plan.phys_splits
    else:
        n_tiles = math.ceil(B / plan.block_m)
        programs = n_tiles * units.U
        ps = 1
        while programs * ps < 8 * sm_count and ps < 64 and units.BPS // (2 * ps) >= 2:
            ps *= 2
    need = B * units.U * ps * state_bytes(level, track_ids)
    if ps > 1 and need > budget:
        raise WorkspaceBudgetError(
            f"{ps} splits need {human_bytes(need)} of pooling workspace > budget {human_bytes(budget)}"
        )
    return ps


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def fused_forward_states(x_op, sxf, sxb, B, idx, w, b, counts_i32, layout: SparseLayout,
                         level: LevelSpec, units: Units, plan: PhysicalPlan, track_ids: bool,
                         phys_splits: int, topk_sort: bool = False) -> Dict[str, torch.Tensor]:
    dev = w.device
    U, PS = units.U, phys_splits
    BM, TT = plan.block_m, plan.token_tiles_per_program
    C = layout.block
    bpsp = math.ceil(units.BPS / PS)
    bpsp += bpsp & 1
    n_tiles = triton.cdiv(B, BM)
    KP = level.k if level.type == "topk_mean" else 1
    f32 = dict(device=dev, dtype=torch.float32)
    st_m = torch.empty((PS, B, U), **f32)
    st_s = torch.empty((PS, B, U), **f32) if level.type != "max" else st_m
    st_t = torch.empty((PS, B, U), **f32) if level.type == "softmax_weighted" else st_m
    topk_final = level.type == "topk_mean" and PS == 1
    if topk_final:  # finalized in-kernel; ids only if they will be saved
        st_id = torch.empty((B, U, KP), device=dev, dtype=torch.int32) if track_ids else st_m
    else:
        st_id = torch.empty((PS, B, U), device=dev, dtype=torch.int32) if track_ids else st_m
    st_keys = (
        torch.empty((PS, B, U, KP), device=dev, dtype=torch.int64)
        if (level.type == "topk_mean" and not topk_final) else st_m
    )
    n_tt = triton.cdiv(n_tiles, TT)
    grid = (n_tt, U, PS)
    tk.fwd_pool_kernel[grid](
        x_op, sxf, sxb, idx, w, b, counts_i32,
        st_m, st_s, st_t, st_id, st_keys,
        B, U, units.S, layout.blocks_per_group, units.BPS, bpsp, n_tiles,
        kernel_scale(level), float(level.p),
        POOL=level.pool_id, ACT=level.act_id, MODE_A=not layout.shared_indices, C=C,
        KP=KP, TRACK_IDS=track_ids, BM=BM, TT=TT, EVEN_B=(B % BM == 0), P_INT=_p_int(level), TOPK_SORT=topk_sort,
        FRAG=layout.packed, UB=layout.unit_blocks, TOPK_FINAL=topk_final,
        num_warps=plan.num_warps, num_stages=plan.num_stages,
    )
    if topk_final:
        out = {"topk_pre": st_m[0]}
        if track_ids:
            out["topk_slot"] = st_id
        return out
    out = {"m": st_m}
    if level.type != "max":
        out["s"] = st_s
    if level.type == "softmax_weighted":
        out["t"] = st_t
    if track_ids:
        out["id"] = st_id
    if level.type == "topk_mean":
        out["keys"] = st_keys
    return out


def decode_keys(keys: torch.Tensor):
    s = (keys >> 32).to(torch.int32)
    bits = torch.where(s >= 0, s, s ^ 0x7FFFFFFF)
    val = bits.view(torch.float32)
    slot = (4294967295 - (keys & 0xFFFFFFFF)).to(torch.int64)
    return val, slot


def act_torch(x, aid):
    if aid == 0:
        return x
    if aid == 1:
        return torch.relu(x)
    return torch.nn.functional.gelu(x)


def dact_torch(x, aid):
    if aid == 0:
        return torch.ones_like(x)
    if aid == 1:
        return (x > 0).to(x.dtype)
    cdf = 0.5 * (1 + torch.erf(x * 0.7071067811865476))
    return cdf + x * 0.3989422804014327 * torch.exp(-0.5 * x * x)


def merge_finalize(st: Dict[str, torch.Tensor], level: LevelSpec, counts_u: torch.Tensor):
    """Merge per-split states (dim 0) and finalize.

    Returns dict with 'pre' (pooled value before out_activation, [B, U]) and
    operator statistics ('L2', 'winner', 'topk_slot', 'topk_val').
    """
    t = level.type
    n = counts_u[None, :]
    out = {}
    if t == "max":
        m = st["m"]
        M = m.amax(0)
        out["pre"] = M
        if "id" in st:
            ids = torch.where(m == M[None], st["id"], torch.full_like(st["id"], INT32_MAX))
            out["winner"] = ids.amin(0)
    elif t == "mean":
        out["pre"] = st["s"].sum(0) / n
    elif t in ("logmeanexp", "softmax_weighted"):
        m = st["m"]
        M = m.amax(0)
        r = torch.exp2(m - M[None])
        S = (st["s"] * r).sum(0)
        L2 = M + torch.log2(S)
        out["L2"] = L2
        if t == "logmeanexp":
            out["pre"] = (L2 - torch.log2(n)) / kernel_scale(level)
        else:
            out["pre"] = (st["t"] * r).sum(0) / S
    elif t == "power_mean":
        m = st["m"]
        M = m.amax(0)
        safe = torch.where(M > 0, M, torch.ones_like(M))
        ratio = m / safe[None]
        S = (st["s"] * ratio.pow(level.p)).sum(0)
        out["pre"] = torch.where(M > 0, M * (S / n).pow(1.0 / level.p), torch.zeros_like(M))
    elif t == "topk_mean" and "topk_pre" in st:  # finalized in-kernel
        out["pre"] = st["topk_pre"]
        if "topk_slot" in st:
            out["topk_slot"] = st["topk_slot"]  # int32
    elif t == "topk_mean":
        keys = st["keys"]  # [PS, B, U, KP]
        PS, B, U, KP = keys.shape
        if PS > 1:
            keys = torch.topk(keys.permute(1, 2, 0, 3).reshape(B, U, PS * KP), KP, dim=-1).values
        else:
            keys = keys[0]
        val, slot = decode_keys(keys)
        kk = torch.clamp(n, max=KP)  # documented normaliser: min(k, valid neurons)
        valid = torch.arange(KP, device=keys.device)[None, None, :] < kk[..., None]
        a = act_torch(val, level.act_id)
        out["pre"] = torch.where(valid, a, torch.zeros_like(a)).sum(-1) / kk
        out["topk_slot"] = torch.where(valid, slot, torch.full_like(slot, -1)).to(torch.int32)
        out["topk_val"] = val
    return out


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
def dense_coefficients(level: LevelSpec, g_pre: torch.Tensor, pre: torch.Tensor,
                       counts_u: torch.Tensor, L2: Optional[torch.Tensor]):
    """Per-(b, u) coefficients consumed by the dense recompute backward."""
    n = counts_u[None, :]
    z = torch.zeros_like(g_pre)
    t = level.type
    if t == "mean":
        return g_pre / n, z, z
    if t == "logmeanexp":
        if L2 is None:  # recover the log-normaliser from the pooled output
            L2 = pre * kernel_scale(level) + torch.log2(n)
        return g_pre, L2, z
    if t == "power_mean":
        pos = pre > 0
        c1 = torch.where(pos, g_pre / n, z)
        c2 = torch.where(pos, 1.0 / torch.where(pos, pre, torch.ones_like(pre)), z)
        return c1, c2, z
    if t == "softmax_weighted":
        assert L2 is not None, "softmax-weighted backward needs the log-normaliser"
        return g_pre, L2, pre
    raise ValueError(t)


def launch_dense_backward(x_op, sxf, sxb, B, idx, w, b, counts_i32, layout, level, units: Units,
                          plan: PhysicalPlan, c1, c2, c3, dx_buf, sdf, sdb, dw_buf, db_buf):
    BM = plan.block_m
    n_tiles = triton.cdiv(B, BM)
    grid = (n_tiles, units.U, 1)
    tk.bwd_dense_kernel[grid](
        x_op, sxf, sxb, idx, w, b, counts_i32, dx_buf, sdf, sdb, dw_buf, db_buf,
        c1.contiguous(), c2.contiguous(), c3.contiguous(),
        B, units.U, units.S, layout.blocks_per_group, units.BPS, units.BPS,
        kernel_scale(level), float(level.p),
        POOL=level.pool_id, ACT=level.act_id, MODE_A=not layout.shared_indices, C=layout.block,
        BM=BM, EVEN_B=(B % BM == 0), P_INT=_p_int(level), num_warps=plan.num_warps,
        FRAG=layout.packed, UB=layout.unit_blocks,
    )


def launch_event_backward(x_op, sxf, sxb, idx, w, b, layout, level, units: Units,
                          ev_slot, ev_coef, dx_buf, sdf, sdb, dw_buf, db_buf, BE: int = 128):
    ev_slot = ev_slot.to(torch.int32).contiguous()
    ev_coef = ev_coef.to(torch.float32).contiguous()
    B, U, KPE = ev_slot.shape
    n_ev = ev_slot.numel()
    grid = (triton.cdiv(n_ev, BE),)
    tk.bwd_events_kernel[grid](
        x_op, sxf, sxb, idx, w, b, dx_buf, sdf, sdb, dw_buf, db_buf,
        ev_slot, ev_coef, n_ev, KPE, U, units.S, layout.blocks_per_group,
        ACT=level.act_id, MODE_A=not layout.shared_indices, C=layout.block, BE=BE, num_warps=4,
        FRAG=layout.packed, UB=layout.unit_blocks,
    )
