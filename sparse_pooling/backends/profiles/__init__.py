"""Tuning profiles: measured (or deliberately conservative) launch / variant choices per device.

A profile never makes something legal: the resolver (backends/dispatch.py) applies, in order,
  1. numerical requirements (strict determinism), 2. device / ISA / kernel / workspace legality,
  3. valid explicit PhysicalPlan overrides, 4. the profile's measured choice, 5. a conservative
  eligible fallback,
and records why.  Profiles only answer the questions in ``TuningProfile`` below.

Selection (``select``): ``PhysicalPlan.tuning_profile`` / ``SPARSE_POOL_PROFILE`` ("auto" default):
"auto" takes the first registered profile whose matcher accepts the device, else "generic_cuda".
Naming a profile forces it; a forced profile on a device it was not measured on is reported as
user-forced and unvalidated (it still passes every legality check).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..capabilities import DeviceCapabilities


@dataclass(frozen=True)
class Validation:
    """Where a profile was checked.  ``numerics`` / ``performance``: device matchers (names of
    functions in the profile module) on which the respective evidence exists."""

    numerics: Tuple[str, ...]
    performance: Tuple[str, ...]
    evidence: Tuple[str, ...]  # docs/evidence.md anchors / pinned report paths
    toolchain: str             # toolchain the measurements were taken with


class TuningProfile:
    """Base class: the questions dispatch asks.  Subclasses hold the data; methods only look it up
    (eligibility, fitting into shared memory and precedence live in dispatch)."""

    name = "base"
    version = "0"
    description = ""
    validation = Validation((), (), (), "")

    # ---------------------------------------------------------------- matching
    def matches(self, caps: DeviceCapabilities) -> Tuple[bool, str]:
        return False, "no matcher"

    def validated_on(self, caps: DeviceCapabilities) -> Dict[str, bool]:
        return {"numerics": False, "performance": False}

    # ------------------------------------------------- grouped (supergroup) kernels
    def sg_phase(self, C: int, S: int, phase: str, mean_bucket_units: float, precision: str):
        """Measured config of a grouped phase, or None = the generic kernel is preferred under
        ``auto``.  fwd: (tokens, warps); dx: (tokens, warps, warps per slab copy); dw: cfg id."""
        raise NotImplementedError

    def sg_default(self, phase: str):
        """Starting config when the phase runs grouped without a measured entry."""
        raise NotImplementedError

    def sg_chunk_units(self, phase: str) -> int:
        raise NotImplementedError

    def dx_scale_plan(self, C: int, S: int, precision: str, act: str, B: int, bucket_stats: Dict):
        """(plan name, reason) of a fully automatic grouped dX in fast mode."""
        return "tiles", f"profile {self.name}: no whole-plan rule"

    def dx_det_plan(self, C: int, S: int) -> str:
        """Grouped dX plan under reduction="deterministic" (only warp-private candidates are legal:
        "tiles" = the per-field tiles with private copies, or a DX_PLANS name with reduction 1)."""
        return "tiles"

    def dx_scatter(self, key: Tuple) -> Optional[str]:
        """Measured dX local-accumulation variant for (C, S, T, warps, wg, precision, act), or None."""
        return None

    def dw_projection(self) -> str:
        return "padded"

    # ------------------------------------------------------------ generic kernels
    def dw_staging(self) -> str:
        """Preferred k_dw staging where indexed staging is legal: "indexed" | "full"."""
        return "full"

    def dw_indexed_stages(self) -> int:
        return 2

    def generic_dw_cfg(self, C: int, pool: str, staging: str) -> int:
        raise NotImplementedError

    def dx_warps(self, C: int, blocks_per_unit: int, unit_blocks: int) -> int:
        return 16

    def epilogue(self, kernel: str, pool: str, act: str, precision: str) -> str:
        """Requested-"auto" dZ epilogue before eligibility: "generic" | "fast" | "mask"."""
        return "generic"

    def dx_target_ctas(self) -> int:
        return 4096

    def describe(self) -> Dict:
        v = self.validation
        return dict(name=self.name, version=self.version, description=self.description,
                    numerics_validated_on=list(v.numerics), performance_validated_on=list(v.performance),
                    evidence=list(v.evidence), toolchain=v.toolchain)


@dataclass(frozen=True)
class ProfileSelection:
    profile: TuningProfile
    requested: str
    forced: bool
    reason: str
    numerics_validated: bool
    performance_validated: bool

    @property
    def status(self) -> str:
        if self.performance_validated and self.numerics_validated:
            return "hardware-validated"
        if self.forced and not self.performance_validated and self.profile.validation.performance:
            return "user-forced-unvalidated"
        if self.numerics_validated:
            return "numerics-validated, performance-unmeasured"
        return "capability-eligible, hardware-unvalidated"

    def summary(self) -> Dict:
        return dict(name=self.profile.name, version=self.profile.version, requested=self.requested,
                    forced=self.forced, match_reason=self.reason, status=self.status,
                    numerics_validated=self.numerics_validated, performance_validated=self.performance_validated)


def _registry():
    from .generic_cuda import GENERIC_CUDA
    from .rtx3090 import RTX3090

    return {RTX3090.name: RTX3090, GENERIC_CUDA.name: GENERIC_CUDA}


PROFILE_ENV = "SPARSE_POOL_PROFILE"


def registered() -> Dict[str, TuningProfile]:
    return dict(_registry())


def select(caps: DeviceCapabilities, requested: str = "auto") -> ProfileSelection:
    """Profile for a device.  ``requested``: "auto" | a registered profile name; "auto" defers to
    SPARSE_POOL_PROFILE when set."""
    reg = _registry()
    if requested == "auto":
        requested = os.environ.get(PROFILE_ENV, "auto") or "auto"
    if requested != "auto":
        if requested not in reg:
            raise ValueError(f"unknown tuning profile {requested!r} (registered: {sorted(reg)})")
        prof = reg[requested]
        ok, why = prof.matches(caps)
        v = prof.validated_on(caps)
        reason = f"forced ({requested}); device {'matches' if ok else 'does not match'}: {why}"
        return ProfileSelection(prof, requested, True, reason, v["numerics"], v["performance"])
    for name, prof in reg.items():
        if name == "generic_cuda":
            continue
        ok, why = prof.matches(caps)
        if ok:
            v = prof.validated_on(caps)
            return ProfileSelection(prof, "auto", False, why, v["numerics"], v["performance"])
    prof = reg["generic_cuda"]
    v = prof.validated_on(caps)
    return ProfileSelection(prof, "auto", False, "no measured profile matches the device", v["numerics"],
                            v["performance"])
