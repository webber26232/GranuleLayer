"""Conservative capability-based policy for NVIDIA devices without a measured profile.

It reuses the production kernels (the same arithmetic and numerical contracts) with fixed, simple
choices that need no measurement: no scatter / whole-plan / large-bucket rules, the plain epilogues,
full k_dw staging, the smallest grouped dW ring, and token tiles that the resolver shrinks until
they fit the device's actual opt-in shared memory.  For input-supergroup layers the grouped kernels
are preferred because their shared memory scales with the supergroup width S instead of the input
width D (the generic k_dx needs D * 96 bytes), so they fit on more devices.

Performance of these choices is unmeasured on every device ("capability-eligible, hardware-
unvalidated").  Numerical correctness of the generic profile was tested on the RTX 3090 by forcing it
(tests: ``tuning_profile="generic_cuda"``); that is not a statement about any other GPU.
"""
from __future__ import annotations

from . import TuningProfile, Validation
from .rtx3090 import is_rtx3090


class GenericCudaProfile(TuningProfile):
    name = "generic_cuda"
    version = "1"
    description = "Conservative eligible choices under the device's actual resource limits (unmeasured)"
    validation = Validation(numerics=("is_rtx3090",), performance=(), evidence=("docs/gpu_portability.md",),
                            toolchain="numerics: nvcc 12.1.66 on RTX 3090 (forced profile)")

    def matches(self, caps):
        if caps.backend != "cuda":
            return False, f"not a CUDA device ({caps.backend})"
        return True, f"any CUDA device (sm_{caps.cc[0]}{caps.cc[1]}); kernel legality is checked separately"

    def validated_on(self, caps):
        return {"numerics": is_rtx3090(caps)[0], "performance": False}

    def sg_phase(self, C, S, phase, mean_bucket_units, precision):
        return self.sg_default(phase)  # grouped wherever legal (smaller shared-memory footprint)

    def sg_default(self, phase):
        return {"fwd": (64, 8), "dx": (64, 8, 1), "dw": 0}[phase]

    def sg_chunk_units(self, phase):
        return {"fwd": 2048, "dx": 4096}[phase]

    def generic_dw_cfg(self, C, pool, staging):
        return 0  # 8 warps x 2 work units: the smallest register / work footprint

    def dx_warps(self, C, blocks_per_unit, unit_blocks):
        return 16

    def epilogue(self, kernel, pool, act, precision):
        return "generic"


GENERIC_CUDA = GenericCudaProfile()
