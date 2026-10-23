# Adding a GPU

The steps below take a new NVIDIA device from "runs with the generic policy" to "has a measured,
validated profile". Nothing in the semantic core (`config.py`, `layer.py`, `layout.py`,
`supergroups.py`, `autograd.py`, `optim.py`) changes. The statuses used below ("compiles", "legal",
"numerically tested", "performance-measured") are defined in [gpu_portability.md](gpu_portability.md) §1.
Paths are relative to the repository root; run the snippets from there.

## 1. Discover capabilities and the build target

```python
import torch
from sparse_pooling.backends.capabilities import NATIVE_FAMILY_FEATURES, query, toolchain
from sparse_pooling.backends.cuda.build import resolve_target
from sparse_pooling.backends.profiles import select
caps = query(torch.device("cuda", 0))
print(caps.summary())                       # cc, SMs, opt-in shared memory, registers, features, ...
print(caps.missing(NATIVE_FAMILY_FEATURES))  # must be () for the native kernels (sm_80+)
print(resolve_target(caps, toolchain()))     # (target, reason), e.g. ('sm_86', 'device ... (sm_86)')
print(select(caps).summary())               # which tuning profile 'auto' picks, and its status
```

```bash
python -m sparse_pooling.backends.cuda.build            # cache directory and library path for this GPU
python -m sparse_pooling.backends.cuda.build --build    # build the production bundle now (else: first CUDA call)
```

* If `caps.missing(NATIVE_FAMILY_FEATURES)` is not empty (a pre-sm_80 device), the native kernels
  cannot run: `backend="auto"` uses Triton and records why in `last_info["backend_fallback"]`;
  `backend="cuda"` raises.
* `resolve_target` builds for the device's own plain target (`sm_<major><minor>`, e.g. `sm_86`,
  `sm_90`, `sm_120`). If the installed nvcc cannot emit it, it uses the newest target of the **same
  major** with a lower or equal minor that nvcc lists (CUDA binary compatibility); otherwise it raises
  with the supported list. It never falls back across majors and never selects PTX for JIT. Install a
  toolkit that knows the device (`NVCC=/path/to/nvcc`) rather than forcing a target.
* `SPARSE_POOL_ARCH=sm_<major><minor>` forces a target. It must be a complete plain token, binary
  compatible with the device, and listed by the toolkit. Architecture-specific / family targets
  (`sm_90a`, `sm_100f`) are refused: these kernels do not use them (§3).

## 2. Run with the generic policy

A device that no measured profile matches gets `generic_cuda` automatically. It uses the production
kernels with simple choices fitted to the device's own limits, and reports its status as
`capability-eligible, hardware-unvalidated` (`layer.explain(batch)["profile"]`). Run the existing
gates on the device:

```bash
(cd sparse_pooling/tests && python -m pytest -q)                                              # full suite
bash sparse_pooling/benchmarks/run_det_training.sh python /scratch/rec selftest out/selftest  # harness self-check
bash sparse_pooling/benchmarks/run_det_training.sh python /scratch/rec small out/small        # repeatability
bash sparse_pooling/benchmarks/run_sanitizer.sh --jobs 4 --out out/sanitizer python           # 4 tools x 4 workloads
```

`run_det_training.sh` and `run_sanitizer.sh` exit nonzero on any failure; `run_sanitizer.sh` keeps
every log and writes `summary.tsv` (one PASS / FAIL row per workload and tool).

Tests that assert the RTX 3090 profile's measured choices carry the `rtx3090_policy` marker and skip
under another profile; every numerical test runs under every profile.

## 3. Decide whether the existing kernel family suffices

The CUDA kernels are written for warp-level `mma.sync` with 32-lane fragments; they are legal on
every sm_80+ target (the required instructions are in `capabilities.FEATURE_MIN_CC`). A new
**kernel family** (for example Hopper `wgmma` with TMA, or Blackwell `tcgen05`) is needed only when
the existing one leaves most of the hardware idle. In that case:

* put the sources under `sparse_pooling/kernels/cuda/<family>/` behind the existing `extern "C"`
  ABI pattern (`abi.cu`);
* add a bundle to `BUNDLES` in `sparse_pooling/backends/cuda/build.py`. Such families need
  architecture-specific targets (`sm_90a`, `sm_100a`), whose cubins run only on that exact compute
  capability. `parse_arch` / `resolve_target` refuse those targets today: extend the grammar together
  with that compatibility rule, and make the target part of the variant's eligibility;
* declare new instructions in `sparse_pooling/backends/capabilities.py` (`FEATURE_MIN_CC`) with a ptxas
  probe in `sparse_pooling/tests/data/isa_probe.cu` (`tests/test_isa_requirements.py`);
* add eligibility and shared-memory formulas in `sparse_pooling/backends/cuda/variants.py` and their
  use in `sparse_pooling/backends/dispatch.py`.

## 4. Measure and register a profile

Create `sparse_pooling/backends/profiles/<board>.py`. Subclass `TuningProfile` and answer only the
questions it asks. The example below is **not registered** and its numbers are placeholders, not
measurements (`tests/test_dispatch.py` runs it through the resolver as written):

```python
from sparse_pooling.backends.profiles import TuningProfile, Validation

def is_example_board(caps):
    ok = caps.name == "NVIDIA Example Board" and caps.cc == (8, 9) and caps.sm_count == 128
    return ok, "example board" if ok else f"not the example board ({caps.name})"

class ExampleProfile(TuningProfile):
    name, version = "example_board", "0"
    description = "TEMPLATE ONLY - unmeasured"
    validation = Validation(numerics=(), performance=(), evidence=(), toolchain="")

    def matches(self, caps):            return is_example_board(caps)
    def validated_on(self, caps):       return {"numerics": False, "performance": False}
    def sg_phase(self, C, S, phase, mean_bucket_units, precision):
        return self.sg_default(phase)   # replace with measured per-(C, S) choices
    def sg_default(self, phase):        return {"fwd": (64, 8), "dx": (64, 8, 1), "dw": 0}[phase]
    def sg_chunk_units(self, phase):    return {"fwd": 2048, "dx": 4096}[phase]
    def generic_dw_cfg(self, C, pool, staging): return 0
```

The matcher must identify the measured board, not a compute capability: `profiles/rtx3090.py`
(`is_rtx3090`) requires the exact name, cc 8.6, 82 SMs and 99 KiB opt-in shared memory, so an RTX 3080
or A10 (also sm_86) does not inherit RTX 3090 tables.

Measure with the existing protocols on an idle GPU:

* `sparse_pooling/benchmarks/bench_supergroups.py`: complete training steps with restored training state
  (`--rounds 4 --reps 2 --iters 3`), per-phase CUDA events. Write a variant for each candidate as
  `--variant name:field=value,...` and compare forced plans.
* `sparse_pooling/benchmarks/run_impl_ab.sh` / `summarize_impl_ab.py` for paired, alternating A/B of two
  checkouts; `bench_kernels.py` / `bench_end_to_end.py` for the generic kernels.
* Record the workload scope (C, S, R, H, M, precision, activation, bucket-statistics denominator) and
  the toolchain next to every table, as `profiles/rtx3090.py` does.

Register the profile in `_registry()` of `sparse_pooling/backends/profiles/__init__.py` (before
`generic_cuda`). Then pin its decision matrix with the resolver adapter of `plan_matrix.py` and a
synthetic record of the board (the `plan_matrix.py` command line covers only the RTX 3090), store the
rows as a fixture and add a test like
`tests/test_dispatch.py::test_rtx3090_reproduces_the_reviewed_decision_matrix`:

```python
import sys; sys.path.insert(0, "sparse_pooling/benchmarks")
import plan_matrix
from sparse_pooling.backends.capabilities import synthetic
board = synthetic("NVIDIA Example Board", (8, 9), sm_count=128, smem_per_block_optin=99 * 1024)
rows = [dict(case=c, decisions=plan_matrix.decisions(c, caps=board, profile="example_board"))
        for c in plan_matrix.all_cases()]
```

## 5. Unknown and resource-constrained devices

The resolver can be exercised for a device you do not have, on any machine (no GPU, no nvcc):

```python
import sys; sys.path.insert(0, "sparse_pooling/benchmarks")
import plan_matrix
from sparse_pooling.backends.capabilities import synthetic
from sparse_pooling.backends.dispatch import Resolver
from sparse_pooling.backends.profiles import select
small = synthetic("synthetic sm_80, 48 KiB opt-in", (8, 0), smem_per_block_optin=48 * 1024)
sel = select(small)                      # generic_cuda: no measured profile matches this record
case = dict(C=8, S=64, ub=6.0, B=8192, D=1024, G=1024, H=100_000_000, precision="split", act="relu",
            det=False, pool="mean", plan={})
plan = Resolver(small, sel).resolve(plan_matrix.make_rt(case, None), case["B"], "bwd").as_dict()
print(sel.summary()["status"], plan["profile"], plan["backend"], plan["reason"])
# -> capability-eligible, hardware-unvalidated generic_cuda triton
#    k_dx slab of D = 1024 exceeds the device's shared memory
```

On this record the whole backward resolves to Triton: the native backward needs the generic `k_dx`
(`D x 96` bytes of shared memory) to be legal ([gpu_portability.md](gpu_portability.md) §8).

Such records are marked `source="synthetic"`: they test control flow, never hardware. Check that:

* `select()` picks `generic_cuda` (or your profile only if its matcher accepts the record);
* every chosen tile fits the record's `smem_per_block_optin` (the resolver shrinks token tiles, and an
  explicit request that cannot fit raises `UnsupportedError` instead of silently running something else);
* forcing `tuning_profile="rtx3090"` (or `SPARSE_POOL_PROFILE=rtx3090`) on another device is reported as
  `user-forced-unvalidated` and still passes every legality check.

`tests/test_dispatch.py` holds such cases (A100-, Ada-, Hopper-like records, another sm_86 board, a
48 KiB opt-in device, Turing), and `tests/test_build.py` covers target selection for toolkits and
devices that are not installed (e.g. sm_100 / sm_120 / sm_121). On hardware,
`sparse_pooling/kernels/cuda/launch.cuh` re-checks every launch against the driver's per-kernel
attributes before the first launch on each device.

## 6. Promotion criteria

A profile may claim `hardware-validated` (set `validated_on` to return true for its matcher) only when,
on that exact board and toolchain:

1. the full test suite passes;
2. `run_det_training.sh selftest`, `small` and `production` (or the board's memory-feasible equivalent)
   pass, with strict status `pass` and independent processes;
3. `run_sanitizer.sh` exits 0 (every row of `summary.tsv` is PASS);
4. every measured table entry is backed by a paired, alternating, restored-state comparison (evidence
   in [evidence.md](evidence.md)), with no repeatable complete-step regression above 2 % against the
   generic profile;
5. the support matrix in [gpu_portability.md](gpu_portability.md) is updated. Compiling for a target is
   never enough.
