# Migration manifest

Source: `c8-deterministic-reduction-dw-projection` @ `2b9a84d` (50 commits on top of `main` @ `300a268`).
Destination: `gpu-portability-cleanup` (this tree). Reproduce the full list with
`git diff -M40 --name-status 2b9a84d 34a6433` (the reviewed snapshot): 33 added,
35 modified, 4 renamed and 1,011 deleted paths (945 of them the phase reports and raw evidence under
`sparse_pooling/results/`).

Status: **retained** (in main, production), **optional** (in main, built or used only on explicit
request), **archived** (left on the pinned experiment commit; see [evidence.md](evidence.md)).

## Implementation

| Source component (2b9a84d) | Destination | Status | Tests | Provenance |
|---|---|---|---|---|
| `config.py`, `layer.py`, `layout.py`, `autograd.py`, `supergroups.py`, `optim.py`, `memory.py`, `reference/` | same paths; plan resolution delegated to `backends/dispatch.py`; `PhysicalPlan.tuning_profile` added; per-device guards in `optim.py` / generator stats | retained | whole suite; `test_dispatch.py`, `test_device_runtime.py` | campaign commits listed in `1889c1a`; this refactor |
| `kernels/cuda_ops.py` (bridge + policy + build + singleton) | `backends/capabilities.py` (facts), `backends/profiles/rtx3090.py` (measured tables), `backends/profiles/generic_cuda.py` (conservative), `backends/dispatch.py` (resolver), `backends/cuda/{build,runtime,variants}.py`; `cuda_ops.py` remains a facade with no policy | retained | `test_dispatch.py` (4,560 reviewed decisions reproduced), `test_build.py`, `test_cpu_import.py` | this refactor |
| `kernels/cuda/common.cuh` | split: `isa_sm80.cuh` (mma / ldmatrix / movmatrix / cp.async, `#error` below sm_80), `numerics.cuh` (FTZ compare / multiply, ex2), `common.cuh` (activations, BF16 helpers) | retained | `test_isa_requirements.py`, `test_codegen.py` | this refactor |
| `*_dispatch.cuh` launch code, per-process shared-memory opt-in caching | `launch.cuh` launch guard (per device, static + dynamic shared memory from the driver, checked before the first launch) | retained | `test_device_runtime.py`, sanitizer | this refactor |
| production `.cu` units (`sparse_pool.cu`, `fwd_*`, `bwd_*`, `sg_fwd_*`, `sg_dx_*`, `sg_dw_*`, `sg_det.cu`, `sg_dxs_c8_*` scatter 9 / 10) | same paths, production bundle (explicit manifest in `build.py`); `abi.cu` added (struct layout + ABI identity) | retained | whole suite; SASS identical to 2b9a84d for all 2,321 common kernels | campaign |
| experimental scatter layouts, owner variant, fused backward (`sg_bwdf_c8.cu`), compact dW projection, mask dW control | `kernels/cuda/experimental/` (bundle built on first explicit request) | optional | `test_dx_scatter.py`, `test_fused_backward.py`, `test_dw_projection.py` | C8_DX_SCATTER, SCALE_DISPATCH, DETERMINISM reports |
| `diag_proj_c8.cu`, slab layout dumps | `kernels/cuda/diagnostics/` (tests / tools only) | optional | `test_codegen.py`, `test_dx_scatter.py` | SCALE_DISPATCH §6 |
| `SP_ABLATE_*` wrong-result ablations | build kind `ablation` via `SPARSE_POOL_NVCC_EXTRA`; separate cache, load-time warning, recorded in `last_info["execution"]["build"]` | optional (never a validation build) | `test_build.py` | campaign |
| `kernels/cuda/screen_fusion.cu` | — | archived | — | SCALE_DISPATCH (`c9a591e`) |
| `kernels/ops.py`, `kernels/triton_fused.py` | same paths (`choose_splits` uses the device SM count) | retained | whole suite | campaign |

## Tools and tests

| Source component | Destination | Status | Notes |
|---|---|---|---|
| `benchmarks/det_training.py` (false pass on a missing reference key) | strict record store `det_records.py` (schema `granule.det_records/2`), strict / characterization verdicts, nonzero exit; `run_det_training.sh` propagates failures | retained | `test_det_records.py` + runner `selftest` (end-to-end negative cases) |
| `bench_supergroups.py`, `bench_kernels.py`, `bench_end_to_end.py`, `bench_memory.py`, `bench_matched.py`, `train_state.py`, `common.py`, `inspect_sass.py`, `kernel_resources.py`, `audit_bucket_stats.py`, `optim_race_check.py`, `profile_*` | same paths; implementation root and interpreter from the command line / `GRANULE_IMPL_ROOT`, no personal paths | retained | — |
| `run_sanitizer*.sh` (one per phase) | `run_sanitizer.sh <python> [generic sg dx_scatter det]` | retained | — |
| — | `plan_matrix.py`, `compare_sass.py`, `run_impl_ab.sh`, `summarize_impl_ab.py`, `run_det_regression.sh` (old-versus-new validation of this refactor) | retained (new) | — |
| campaign `run_*.sh`, `summarize_*.py`, screening / traffic / bank / fusion models, `numerics_*`, `sass_*`, `det_anomaly_probe.py`, `sanitize_capped/sdfb_workload.py`, `granule_memory_audit.py` (65 files) | — | archived at 2b9a84d | historical; replaced by the maintained tools |
| tests (25 files) | same paths; tests that patched module tables patch `profiles/rtx3090`; profile-aware expectations; new `test_dispatch`, `test_build`, `test_codegen`, `test_cpu_import`, `test_det_records`, `test_device_runtime`, `test_isa_requirements` | retained | no test deleted |

## Evidence and documents

| Source component | Destination | Status |
|---|---|---|
| `sparse_pooling/results/<phase>/` raw evidence (traffic, backward_compute, input_supergroups, c8_dx_scatter, scale_dispatch_fused_backward, determinism_metrics_dw_projection; 939 files) and the six phase reports | pinned links in [evidence.md](evidence.md) | archived at 2b9a84d |
| top-level `sparse_pooling/results/*.json`, `TABLES.md` (re-measured on the repaired card, `d3595ac`) | same paths | retained |
| `results/e2e_final_100M_C{8,64}_rerun.{json,log}` (in `main` @ `300a268`) | removed by the campaign's re-measurement `d3595ac`, superseded by the re-measured files | archived at `300a268` |
| `README.md`, `REPORT.md` | updated: backend / profile usage, links to pinned reports | retained |
| — | `docs/` (portability, architecture, adding a GPU, migration, this manifest, evidence, validation) | new |
