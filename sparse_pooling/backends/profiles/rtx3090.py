"""RTX 3090 tuning profile: every empirically selected default of the optimization campaign.

Measured on one NVIDIA GeForce RTX 3090 (GA102, sm_86, 82 SMs, 99 KiB opt-in shared memory per
block, 24 GiB after the 2026-10-16 repair), driver 550.67, nvcc 12.1.66, PyTorch 2.7.0+cu128,
Triton 3.3.0, with complete-training-step A/B protocols.  Evidence (pinned commits / reports):
docs/evidence.md.  Workload scope of the measurements: D = G = 1024 (and D = 256 / G = 64 medium
screens), H = 1M - 100M, M = 1024 - 8192 tokens, C = 8 / 64, input supergroups S = 8 - 512 with
R = S / 4, Mean(ReLU) (Mean(identity) shares the kernels), split and BF16 gradients, BF16
parameters / FP32 main_grad.  The bucket-statistics denominator of the S = 64 rule is the
canonical mean work units per NONEMPTY (output, supergroup) bucket (supergroups.bucket_statistics).

Every table below is a performance choice; none is required for correctness (each alternative is
legal and numerically validated, see the tests).  The matcher accepts only the measured board:
other sm_86 devices (RTX 3080 / 3070 / A10 / A40 ...) use the generic profile unless forced.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..capabilities import DeviceCapabilities
from . import TuningProfile, Validation

RTX3090_NAMES = ("NVIDIA GeForce RTX 3090", "GeForce RTX 3090")


def is_rtx3090(caps: DeviceCapabilities) -> Tuple[bool, str]:
    """Exact board name (not "RTX 3090 Ti"), compute capability 8.6, 82 SMs, 99 KiB opt-in shared
    memory per block.  All four must hold: the name alone is not a capability statement."""
    checks = [(caps.name in RTX3090_NAMES, f"name {caps.name!r}"), (tuple(caps.cc) == (8, 6), f"sm_{caps.cc[0]}{caps.cc[1]}"),
              (caps.sm_count == 82, f"{caps.sm_count} SMs"),
              (caps.smem_per_block_optin == 99 * 1024, f"{caps.smem_per_block_optin} B opt-in shared memory")]
    bad = [w for ok, w in checks if not ok]
    if bad:
        return False, "not the measured RTX 3090: " + ", ".join(bad)
    return True, "RTX 3090 (name, sm_86, 82 SMs, 99 KiB opt-in shared memory)"


# Grouped-kernel defaults per (C, S) (H = 10M, M = 8192 screens; INPUT_SUPERGROUP_OPTIMIZATION.md):
# fwd (tokens, warps) | None, dX (tokens, warps, warps per FP32 slab copy) | None, dW cfg | None.
# None = the generic kernel is faster for that phase, so backend "auto" keeps it (hybrid plans).
# Other S use the nearest smaller key.
SG_AUTO = {
    (64, 8): ((128, 4), None, None), (64, 64): ((128, 4), None, None), (64, 128): ((128, 8), None, None),
    (64, 256): ((64, 8), None, None), (64, 512): (None, None, None),
    (8, 8): ((128, 8), (128, 16, 1), 2), (8, 16): ((128, 8), (64, 8, 1), 2), (8, 32): ((128, 8), (32, 8, 1), 2),
    (8, 64): ((128, 8), (64, 8, 8), 2), (8, 128): ((128, 8), (32, 8, 8), 2), (8, 256): ((64, 8), None, 0),
    (8, 512): (None, None, None),
}
SG_CHUNK_UNITS = {"fwd": 2048, "dx": 4096}
# grouped C = 8 Mean(ReLU) fast-epilogue dW projection under cuda_grouped_dw_projection = "auto": the
# compact projection is bitwise equal but 3-4 % slower dW (DETERMINISM_METRICS_DW_PROJECTION.md 6.4)
SG_AUTO_DW_PROJECTION = "padded"
# C = 64 with large buckets (mean work units per nonempty (output, supergroup) bucket >= 3; H = 100M:
# ~6, H = 10M: ~1): the grouped dX (all S <= 64) and the grouped dW (S <= 16) are faster at
# H = 100M (-15 ms each, 4 interleaved rounds, +-2 ms) although slower at H = 10M; with BF16
# gradients the grouped dX (generic epilogue) is slower (614 vs 599 ms): split only
SG_AUTO_C64_LARGE_BUCKETS = {"dx": ((64, 8, 8), 64, ("split",)), "dw": (2, 16, ("split", "bf16")),
                             "min_bucket_units": 3.0}
# measured dX scatter per (C, S, dX tokens, warps, warps per copy, precision, activation) = the auto
# dX tiles of SG_AUTO; anything not listed keeps the baseline.  Mean(ReLU), corrected harness
# (C8_DX_SCATTER_OPTIMIZATION.md): S = 16 / 32 at H = 10M and 100M (split, 3 seeds), M = 1024, BF16
# gradients at 100M; S = 64 at H = 10M and 100M (split; the warp-private T = 32 vector layout is
# faster at 100M but slower at 10M, so it stays an explicit choice or a whole-plan rule below)
SG_AUTO_DX_SCATTER = {
    (8, 16, 64, 8, 1, "split", "relu"): "vectorized_transpose_hoisted",
    (8, 16, 64, 8, 1, "bf16", "relu"): "vectorized_transpose_hoisted",
    (8, 32, 32, 8, 1, "split", "relu"): "vectorized_transpose_hoisted",
    (8, 32, 32, 8, 1, "bf16", "relu"): "vectorized_transpose_hoisted",
    (8, 64, 64, 8, 8, "split", "relu"): "swizzle4_capped",
}
# Scale-aware whole grouped dX plan (PhysicalPlan.cuda_grouped_dx_plan = "auto" with every grouped dX
# field automatic, fast reduction; SCALE_DISPATCH_FUSED_BACKWARD.md section 5).  A rule applies to
# its (C, S, precisions, activation) and M >= min_tokens; it selects ``plan`` when the canonical mean
# work units per NONEMPTY (output, supergroup) bucket reaches min_units_per_bucket, else ``below``.
# Measured (H = 1M-100M, G = 256 / 1024, R = 8 / 16 / 32, M = 1024-8192): the private T = 32 plan
# gains with longer buckets (split: +16 % at 1.0 units / nonempty bucket ... -3.5 % at 14; crossover
# ~3; the H = 1M row was first tabulated as 0.12 = units per POSSIBLE bucket,
# DETERMINISM_METRICS_DW_PROJECTION.md); thresholds keep the ambiguous region on the shared plan.
# BF16: the shared plan beats the previous BF16 default (the original scatter) by 5.7-7.3 % at
# H = 1M-100M; crossover ~1.5.  Deterministic mode never consults these rules (dispatch: it has its
# own legal candidates and must not inherit the shared-CAS plan).
SG_AUTO_DX_PLAN = [
    dict(C=8, S=64, precisions=("split",), act="relu", min_tokens=1024, min_units_per_bucket=4.0,
         plan="private_t32", below="tiles"),
    dict(C=8, S=64, precisions=("bf16",), act="relu", min_tokens=1024, min_units_per_bucket=2.0,
         plan="private_t32", below="shared_t64"),
]
# k_dw tiling per C (DW_CFGS ids: 8 warps x 4 work units), measured at H = 100M, B = 8192
DW_CFG = {8: 1, 16: 1, 32: 1, 64: 1}
DW_INDEXED_STAGES = 2  # cp.async ring depth of the indexed k_dw (2 or 3)
# default indexed k_dw tiling per operator at C = 64 (H = 1M-100M, B = 8192): the register-capped
# 2-CTA/SM cfg 3 wins for Mean / LME; it spills more with the softmax-weighted / power epilogues,
# which keep 8 warps x 4 work units (MEMORY_TRAFFIC_OPTIMIZATION.md)
DW_CFG_INDEXED = {"mean": 3, "logmeanexp": 3, "power_mean": 1, "softmax_weighted": 1}
DX_TARGET_CTAS = 4096  # k_dx grid: ~4096 CTAs, one dX flush per CTA


class Rtx3090Profile(TuningProfile):
    name = "rtx3090"
    version = "2026.10.19"
    description = ("Measured defaults of the RTX 3090 optimization campaign (c8-deterministic-reduction-"
                   "dw-projection 2b9a84d)")
    validation = Validation(numerics=("is_rtx3090",), performance=("is_rtx3090",),
                            evidence=("docs/evidence.md#rtx-3090-campaign",),
                            toolchain="nvcc 12.1.66, driver 550.67, PyTorch 2.7.0+cu128, Triton 3.3.0")

    def matches(self, caps):
        return is_rtx3090(caps)

    def validated_on(self, caps):
        ok = is_rtx3090(caps)[0]
        return {"numerics": ok, "performance": ok}

    # grouped kernels
    def sg_phase(self, C, S, phase, mean_bucket_units, precision):
        keys = sorted(k[1] for k in SG_AUTO if k[0] == C)
        if not keys:
            return None
        k = max([s for s in keys if s <= S], default=keys[0])
        cfg = SG_AUTO[(C, k)][("fwd", "dx", "dw").index(phase)]
        big = SG_AUTO_C64_LARGE_BUCKETS
        if C == 64 and phase in ("dx", "dw") and mean_bucket_units >= big["min_bucket_units"]:
            c, s_max, precisions = big[phase]
            if S <= s_max and precision in precisions:
                cfg = c
        return cfg

    def sg_default(self, phase):
        return {"fwd": (128, 8), "dx": (64, 8, 8), "dw": 2}[phase]

    def sg_chunk_units(self, phase):
        return SG_CHUNK_UNITS[phase]

    def dx_scale_plan(self, C, S, precision, act, B, bucket_stats: Dict):
        for r in SG_AUTO_DX_PLAN:
            if (r["C"], r["S"], act) != (C, S, r["act"]) or precision not in r["precisions"]:
                continue
            if B < r["min_tokens"]:
                return "tiles", f"scale rule: M = {B} < {r['min_tokens']} (not measured)"
            ub = bucket_stats["mean_units_per_nonempty_bucket"] or 0.0
            ctx = (f"(possible-bucket mean {bucket_stats['mean_units_per_possible_bucket'] or 0:.2f}, occupancy "
                   f"{bucket_stats['bucket_occupancy'] or 0:.3f}; M = {B})")
            if ub >= r["min_units_per_bucket"]:
                return r["plan"], (f"scale rule: mean units per nonempty bucket {ub:.2f} >= "
                                   f"{r['min_units_per_bucket']} {ctx}")
            return r["below"], f"scale rule: mean units per nonempty bucket {ub:.2f} < {r['min_units_per_bucket']} {ctx}"
        return "tiles", f"no scale rule for C = {C}, S = {S}, {precision}, {act}"

    def dx_det_plan(self, C, S):
        # deterministic S = 64: the private T = 32 plan (the fast S = 64 tiles share copies through CAS);
        # DETERMINISM_METRICS_DW_PROJECTION.md section 3
        return "private_t32" if (C, S) == (8, 64) else "tiles"

    def dx_scatter(self, key):
        return SG_AUTO_DX_SCATTER.get(key)

    def dw_projection(self):
        return SG_AUTO_DW_PROJECTION

    # generic kernels
    def dw_staging(self):
        return "indexed"

    def dw_indexed_stages(self):
        return DW_INDEXED_STAGES

    def generic_dw_cfg(self, C, pool, staging):
        cfg = DW_CFG[C]
        if staging == "indexed":
            cfg = DW_CFG_INDEXED.get(pool, cfg)
        if C == 8 and pool != "mean":
            cfg = 0  # 8 x 4 spills with the exp/power epilogues at C = 8 (measured 1.6x slower)
        return cfg

    def dx_warps(self, C, blocks_per_unit, unit_blocks):
        """k_dx is held at 1 CTA/SM by its 96 KiB slab; at 64 registers 32 warps fill the register
        file and hide more latency (9-13 % faster dX at C = 64, neutral at C = 8).  Warps take whole
        work units of a pooling unit, so 32 warps only help when a unit has >= 32 work units
        (measured crossover: H = 1M slower, H >= 2M faster, at M = 1024 and 8192)."""
        return 32 if C == 64 and blocks_per_unit // unit_blocks >= 32 else 16

    def epilogue(self, kernel, pool, act, precision):
        """BACKWARD_COMPUTE_OPTIMIZATION.md: dW "fast" for every operator that has it; dX "mask" for
        Mean with split gradients, "generic" for Mean with BF16 gradients (the short BF16 generic
        path is faster there), "fast" otherwise."""
        if kernel == "dx" and pool == "mean":
            return "mask" if precision == "split" else "generic"
        return "fast"

    def dx_target_ctas(self):
        return DX_TARGET_CTAS


RTX3090 = Rtx3090Profile()
