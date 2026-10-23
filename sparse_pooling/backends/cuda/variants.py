"""Native CUDA kernel family: compiled instantiations, resource formulas, numerical behaviour and status.

These are facts of the kernels (kernels/cuda), independent of any device's tuning:

* every kernel needs the ISA features of ``capabilities.NATIVE_FAMILY_FEATURES`` (BF16 mma.sync,
  cp.async: sm_80+; ldmatrix / movmatrix: sm_75+) and warp size 32 (fragment layouts);
* dynamic shared memory of each launch is given by the formulas below (the same expressions as the
  launch templates; the C++ launch guard adds the kernel's static shared memory from
  cudaFuncGetAttributes and compares against the device's opt-in limit before launching);
* the tilings listed here are the compile-time instantiations that exist (anything else is
  unsupported, not slow);
* ``status``: "production" (auto-selectable, in the production bundle), "experimental" (correct,
  explicit request only, experimental bundle built on first use), "diagnostic" (tests / tools only).
  Wrong-result ablations are build flags (SP_ABLATE_*), never variants: such builds carry kind
  "ablation" (backends/cuda/build.py).
"""
from __future__ import annotations

from typing import Dict

WARP_SIZE = 32

# ---------------------------------------------------------------- grouped (supergroup) kernels
# grouped dW tilings: cfg -> (warps, units per warp, 16-token tiles per stage, stages, min CTAs/SM);
# must match SG_DW_CFGS in kernels/cuda/sg_dispatch.cuh.  min CTAs/SM is a launch bound (register
# cap) chosen per configuration; which configuration runs is a profile decision.
SG_DW_CFGS = {0: (8, 2, 1, 3, 2), 1: (8, 4, 1, 3, 1), 2: (8, 2, 4, 2, 2), 3: (8, 4, 4, 2, 1)}
SG_FWD_TILES = (16, 32, 64, 128)      # tokens (MT = tokens / 16 in 1, 2, 4, 8)
SG_FWD_WARPS = (4, 8, 16)
SG_DX_TILES = (16, 32, 64, 128)
SG_DX_WARPS = (8, 16)
SG_MAX_INPUT_DIM = 16384              # int16 shared-memory feature map
SG_C = (8, 64)

# grouped dX local-accumulation variants (PhysicalPlan.cuda_grouped_dx_scatter): kernel id
# (sparse_pool_sg_dxs.cuh SC_*), slab layout, FP32 summation-order category relative to the baseline
# ("same": every (feature, token) of a warp's copy receives the same additions in the same order;
# "mma_accumulate": contributions are summed inside the MMA accumulators) and status.
SG_DX_SCATTERS: Dict[str, Dict] = {
    "baseline": dict(sc=0, layout="float2 pairs, 2 bank groups (sg_dxs_off)", order="same", status="production"),
    "baseline_private": dict(sc=5, layout="float2 pairs, 2 bank groups; compile-time warp-private (control)",
                             order="same", status="experimental"),
    "swizzle4": dict(sc=1, layout="float2 pairs, 4 bank groups", order="same", status="experimental"),  # sc 6 private
    "swizzle4_capped": dict(sc=10, layout="float2 pairs, 4 bank groups; <= 80 registers (3 CTAs / SM at 8 warps)",
                            order="same", status="production"),
    "vectorized": dict(sc=2, layout="float4 two-tile records, 4 bank groups", order="same", status="experimental"),
    "vectorized_transpose": dict(sc=7, layout="float4 two-tile records after an 8-shuffle transpose, conflict-free",
                                 order="same", status="experimental"),
    "vectorized_hoisted": dict(sc=8, layout="float4 two-tile records, 4 bank groups; per-block record addresses",
                               order="same", status="experimental"),
    "vectorized_transpose_hoisted": dict(sc=9, layout="float4 records after a shuffle transpose; per-block "
                                         "record addresses", order="same", status="production"),
    "vectorized_parity": dict(sc=3, layout="float4 two-tile records, 2 bank groups", order="same",
                              status="experimental"),
    "feature_owner": dict(sc=4, layout="register accumulators [T, S] (MMA-routed), one store per CTA",
                          order="mma_accumulate", status="experimental"),
}
# sc ids compiled into the production entry points sp_sgx_dx_c8_* (the others: sp_sgxe_dx_c8_*)
PRODUCTION_SC = (9, 10)

# experimental fused shared-projection backward tilings (kernels/cuda/experimental/sg_bwdf_c8.cu):
# cfg -> (warps, units per warp, 16-token tiles per window / ring stage, stages, min CTAs / SM)
# (0-5: v1, tile-outer loops; 6-9: v2, block-outer / tile-inner 32-token windows, vector dX records)
SGF_CFGS = {0: (8, 2, 4, 2, 1), 1: (8, 2, 4, 2, 2), 2: (8, 4, 4, 2, 1), 3: (8, 2, 2, 2, 2), 4: (16, 2, 2, 2, 1),
            5: (8, 4, 2, 2, 1), 6: (8, 2, 2, 2, 2), 7: (16, 1, 2, 2, 1), 8: (8, 2, 2, 2, 1), 9: (8, 2, 2, 3, 2)}

# status of the other explicit grouped options
GROUPED_OPTIONS = {
    "dw_projection.padded": "production",
    "dw_projection.compact": "experimental",   # bitwise = padded, slower on the RTX 3090
    "dw_epilogue.mask": "experimental",        # attribution control (C = 8; bitwise = fast)
    "backward.separate": "production",
    "backward.fused": "experimental",          # correct (bitwise dW), not faster; FP32-atomic dX flush
    "backward.fused_recompute": "experimental",
}


def _map_bytes(D: int) -> int:
    return (D * 2 + 15) // 16 * 16


def sg_smem(kind: str, D: int, S: int, tokens: int = 16, warps: int = 8, group_warps: int = 1, dw_cfg: int = 0) -> int:
    """Dynamic shared memory of a grouped kernel launch (sg_dispatch.cuh / sgx_dispatch.cuh)."""
    m = _map_bytes(D)
    if kind == "fwd":
        return m + S * tokens * 2
    if kind == "dx":
        return m + S * tokens * 2 + (warps // group_warps) * S * tokens * 4
    _, _, ts, st, _ = SG_DW_CFGS[dw_cfg]
    return m + st * ts * S * 16 * 2


def sgf_smem(D: int, S: int, cfg: int) -> int:
    nw, _, ts, st, _ = SGF_CFGS[cfg]  # (v2, cfg >= 6: ts = 2, i.e. 32-token windows)
    return _map_bytes(D) + st * ts * S * 16 * 2 + nw * S * 16 * ts * 4


# ------------------------------------------------------------------------ generic kernels
# k_dw tiling: cfg -> (warps, work units per warp); 3: indexed staging only, launch bound 2 CTAs/SM
DW_CFGS = {0: (8, 2), 1: (8, 4), 2: (16, 1), 3: (8, 2)}
DX_WARPS = (8, 16, 32)
GENERIC_C = (8, 16, 32, 64)
FWD_TILES = {16: 1, 32: 2}            # cuda_block_m -> MT
FWD_WARPS = (8, 16)
TOPK_K = (1, 2, 4, 8, 16)             # k = 32 (2 x 32 pairs per lane) runs on Triton
CUDA_POOLS = {"max": 0, "mean": 2, "logmeanexp": 3, "power_mean": 4, "softmax_weighted": 5}
CUDA_BWD_POOLS = ("mean", "logmeanexp", "power_mean", "softmax_weighted")
INDEXED_DW_C = (64,)                  # indexed k_dw staging is compiled for C = 64


def fwd_smem(D: int, MT: int, NW: int) -> int:
    return max(D * 16 * MT * 2, 4 * NW * 16 * MT * 4)


def topk_smem(D: int, MT: int, NW: int, KP: int) -> int:
    return max(D * 16 * MT * 2 + 16 * MT * NW * 4 * 4, NW * 16 * MT * KP * 8)


def dx_smem(D: int) -> int:
    return D * 16 * (2 + 4)           # BF16 X slab + FP32 dX slab of 16 tokens


def dw_smem(D: int) -> int:
    return D * 16 * 2 * 2             # double-buffered BF16 X slab


def dw_idx_smem(NW: int, WPW: int, ST: int, UB: int) -> int:
    return NW * ST * (WPW * UB * 8) * 16 * 2


SG_REDUCE_THREADS = 256

# ------------------------------------------- EXPERIMENTAL generalized fan-in ("kgen" bundle)
# kernels/cuda/experimental/kgen (sparse_pooling.experimental.kgen_layer): input-supergroup layers with
# fan-in K in KGEN_K, logical C % 8 == 0 and C <= 256 neurons per shared record, Mean(ReLU / identity),
# BF16 parameters / inputs, split or BF16 dZ operands, FP32 atomics (fast reduction only), feature-major
# input with B % 8 == 0, S in [8, 256] with K <= S.  Tilings are compile-time per K (KgCfg in kgen.cuh,
# reported by sp_kg_config): (fwd MT, fwd warps, dX MT, dX warps, dW tasks per warp, dW warps, dW tiles
# per stage, dW stages).  Status "experimental": correct (tests/test_kgen.py), never auto-selected.
KGEN_K = (8, 16, 32, 64)
KGEN_C_MAX = 256
KGEN_S_RANGE = (8, 256)
KGEN_STATUS = "experimental"


def kgen_smem(kind: str, D: int, S: int, cfg, group_warps: int = 1) -> int:
    """Dynamic shared memory of a kgen launch (kgen_dispatch.cuh); cfg = the sp_kg_config tuple."""
    m = _map_bytes(D)
    fwd_mt, fwd_nw, dx_mt, dx_nw, _, _, dw_ts, dw_st = cfg
    if kind == "fwd":
        return m + S * 16 * fwd_mt * 2
    if kind == "dx":
        bm = 16 * dx_mt
        return m + S * bm * 2 + (dx_nw // group_warps) * S * bm * 4
    return m + dw_st * dw_ts * S * 16 * 2
