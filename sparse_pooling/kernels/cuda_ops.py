"""Compatibility facade of the native CUDA backend (pre-refactor import path).

The implementation moved to ``sparse_pooling.backends``:
  * backends/cuda/runtime.py   ctypes bridge, library registry, launches (CudaBackend, *Args)
  * backends/cuda/build.py     bundles, targets, build identity, cache
  * backends/cuda/variants.py  kernel-family tilings, resource formulas, variant status
  * backends/dispatch.py       plan resolution (legality, precedence, reasons)
  * backends/profiles/         measured (rtx3090) and conservative (generic_cuda) tuning
This module re-exports those objects (the same objects: no second copy of any table) and the
instrumentation switches (PHASE_EVENTS, PHASE_INTERVALS, DET_WORKSPACE_FILL), whose assignments are
forwarded to the runtime.  Measured tables are re-exported read-only for inspection; to change a
policy, patch or extend the profile (docs/adding_a_gpu.md), not this module.
"""
from __future__ import annotations

import sys
import types

from ..backends.cuda import runtime as _rt
from ..backends.cuda.build import SRC_DIR, BuildError  # noqa: F401
from ..backends.cuda.runtime import (  # noqa: F401
    BWD_FALLBACK,
    BWD_FAST_DW_EPI,
    BWD_FAST_DX_EPI,
    BWD_MASK_DW_EPI,
    BWD_MASK_DX_EPI,
    BwdArgs,
    CudaBackend,
    FwdArgs,
    SgArgs,
    TopkArgs,
    _mark,
    _ptr,
    _warned,
    backend_for,
    build_info,
)
from ..backends.cuda.variants import (  # noqa: F401
    DW_CFGS,
    SG_DW_CFGS,
    SG_DX_SCATTERS,
    SGF_CFGS,
    sg_smem,
    sgf_smem,
)
from ..backends.profiles import rtx3090 as _p3090
from ..backends.profiles.rtx3090 import (  # noqa: F401  (RTX 3090 measured tables, read-only views)
    DW_CFG,
    DW_CFG_INDEXED,
    SG_AUTO,
    SG_AUTO_C64_LARGE_BUCKETS,
    SG_AUTO_DW_PROJECTION,
    SG_AUTO_DX_PLAN,
    SG_AUTO_DX_SCATTER,
    SG_CHUNK_UNITS,
)


def build_library(verbose: bool = False):
    """Path of the production library for the current CUDA device (built on first use)."""
    return _rt.backend_for("cuda").native.path


def sg_auto(C: int, S: int):
    """RTX 3090 profile: measured grouped configs (fwd, dx, dw) of the nearest smaller S."""
    keys = sorted(k[1] for k in SG_AUTO if k[0] == C)
    k = max([s for s in keys if s <= S], default=keys[0])
    return SG_AUTO[(C, k)]


def scale_rule(C: int, S: int, precision: str, act: str, cc, B: int, bucket_stats: dict):
    """(plan, reason) of the RTX 3090 S = 64 whole-plan rule (pre-refactor signature; the rule was
    validated on sm_86 only)."""
    if tuple(cc) != (8, 6):
        return "tiles", f"scale rule validated on sm_86 only (device sm_{cc[0]}{cc[1]})"
    return _p3090.RTX3090.dx_scale_plan(C, S, precision, act, B, bucket_stats)


def dx_warps_default(C: int, blocks_per_unit: int, unit_blocks: int) -> int:
    return _p3090.RTX3090.dx_warps(C, blocks_per_unit, unit_blocks)


def auto_epilogue(kernel: str, lv, grad_precision: str) -> str:
    return _p3090.RTX3090.epilogue(kernel, lv.type, lv.activation, grad_precision)


def _epilogue(requested: str, lv, grad_precision: str = "split", kernel: str = "dx", profile=None) -> str:
    """Effective dZ epilogue (default: under the RTX 3090 profile)."""
    from ..backends.dispatch import epilogue_choice

    return epilogue_choice(profile or _p3090.RTX3090, requested, lv, grad_precision, kernel)


_FORWARDED = ("PHASE_EVENTS", "PHASE_INTERVALS", "DET_WORKSPACE_FILL")


class _Facade(types.ModuleType):
    def __getattr__(self, name):
        if name in _FORWARDED:
            return getattr(_rt, name)
        if name in ("NVCC_FLAGS", "ARCH"):  # build facts of the current device's production library
            info = _rt.build_info()
            return info["flags"] if name == "NVCC_FLAGS" else info["target"]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in _FORWARDED:
            setattr(_rt, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Facade
