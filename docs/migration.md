# Migration and compatibility notes

This covers moving from the experiment branches (up to `c8-deterministic-reduction-dw-projection`
2b9a84d) to the integrated main.

## Unchanged

* **Public API and imports**: `SparsePoolingLayer`, `PhysicalPlan`, `PoolingConfig`, `LevelSpec`,
  `SparseLayout`, `ConnectivityPlan`, `FusedAdamW`.
* **state_dict keys and formats**: `weight` / `bias` (fragment-packed when `weight_layout` is packed),
  `indices`, `sg_feat`, `sg_buckets`, `sg_plan`, and the transform parameters. Checkpoints load
  unchanged, and the physical packing is not reinterpreted.
* **`PhysicalPlan` fields** and their meanings. Every explicit option keeps working, including the
  experimental ones (`cuda_grouped_dx_scatter` variants, `cuda_grouped_backward="fused"`,
  `cuda_grouped_dw_projection="compact"`, `cuda_dw_epilogue="mask"`).
* **The `auto` / `fast` / `deterministic` reduction semantics**, including PyTorch `warn_only`.
* **`last_info` keys** used by the benchmarks and tests.
* **RTX 3090 decisions**: `tests/test_dispatch.py` reproduces all 4,560 decisions of the reviewed
  implementation. The production kernels compile to the same SASS
  (`docs/validation/gpu_portability_cleanup.md`).

## New

* `PhysicalPlan.tuning_profile` (`"auto"`, `"rtx3090"`, `"generic_cuda"`) and the environment variable
  `SPARSE_POOL_PROFILE`. They never change the function computed.
* `last_info["execution"]` and `SparsePoolingLayer.explain(batch)`, which report the resolved plan, device,
  build, profile and validation status.
* `last_info["backend_fallback"]`: why `backend="auto"` runs Triton (no nvcc, a pre-sm_80 device, …).
* `last_info["grouped_dx"]["scatter_status"]` and `["grouped_dw"]["status"]` (production / experimental).

## Behaviour changes (deliberate)

| Before | After |
|---|---|
| Measured RTX 3090 tables applied on every GPU (only the S = 64 rule and indexed staging checked sm_86) | Only the matched RTX 3090 board uses them; other GPUs get `generic_cuda` (reported as hardware-unvalidated) |
| 99 KiB shared-memory limit hard-coded | the device's own opt-in limit, plus static shared memory from the driver |
| `choose_splits` assumed 82 SMs | the device's SM count (identical on the RTX 3090) |
| Explicit `grouped` phase silently ran Triton when the native path was illegal (e.g. D > 1056) | raises `ValueError` |
| A launch on a second GPU reused the first GPU's shared-memory opt-in state | per-device launch state; launches under the tensors' device and stream; mixed-device inputs raise early |
| `FusedAdamW` launched on the current device | on each parameter's device |

## Build cache and environment

| Variable | Meaning |
|---|---|
| `SPARSE_POOL_CACHE_DIR` | build cache root (default `$XDG_CACHE_HOME/granulelayer/cuda`, else `~/.cache/granulelayer/cuda`; previously `sparse_pooling/kernels/cuda/build/`, which can be deleted) |
| `SPARSE_POOL_ARCH` | force a build target: a complete plain `sm_<major><minor>` token (`sm_86`, `sm_90`, `sm_120`; `sm_90a`-style targets are refused); must be binary compatible with the device (same major, equal or older minor) and listed by nvcc ([gpu_portability.md](gpu_portability.md) §5) |
| `SPARSE_POOL_NVCC_EXTRA` | extra nvcc flags: kind `development`, or `ablation` with `-DSP_ABLATE_*` (wrong results; never a validation build). Must not set the target. |
| `SPARSE_POOL_BUILD_JOBS` | parallel compiles (default `min(CPUs, 8)`) |
| `SPARSE_POOL_PROFILE` | process-wide tuning-profile override for `tuning_profile="auto"` |
| `NVCC` | the nvcc to use |

Clear the cache with `python -m sparse_pooling.backends.cuda.build --clean`. Libraries are named
`<cache>/<target>/lib/<bundle>-<kind>-<hash>/libgranule_<bundle>.so`; the cache key covers the sources,
headers, flags, target, nvcc version and host compiler.

## Moved internals (`sparse_pooling.kernels.cuda_ops` remains a facade)

| Old | New |
|---|---|
| `SG_AUTO`, `SG_AUTO_DX_SCATTER`, `SG_AUTO_DX_PLAN`, `SG_AUTO_C64_LARGE_BUCKETS`, `SG_CHUNK_UNITS`, `SG_AUTO_DW_PROJECTION`, `DW_CFG`, `DW_CFG_INDEXED`, `DW_INDEXED_STAGES` | `backends/profiles/rtx3090.py` (patch there; the facade re-exports read-only views) |
| `SG_DW_CFGS`, `SGF_CFGS`, `DW_CFGS`, `SG_DX_SCATTERS`, `sg_smem`, `sgf_smem` | `backends/cuda/variants.py` |
| `CudaBackend.sg_reason / sg_use / sg_dx_plan / sg_dx_cfg / det_reason / ...` | `backends/dispatch.Resolver` (the CudaBackend methods delegate) |
| `CudaBackend.get()` singleton | `backends/cuda/runtime.backend_for(device)` registry (`get()` = current device) |
| `build_library()`, `NVCC_FLAGS`, `ARCH` | `backends/cuda/build.py`; facade `build_library()` / `build_info()` |
| `SG_MAX_SMEM` (99 KiB) | removed: `caps.smem_per_block_optin` |
| `scale_rule(..., cc, ...)` | facade kept (RTX 3090 rule, sm_86 only); profile method `dx_scale_plan` |
| single `libsparse_pool.so` with every unit | `production` / `experimental` / `diagnostics` bundles |
| `kernels/cuda/sg_dxo_*.cu`, `sg_bwdf_c8.cu`, `diag_proj_c8.cu`, part of `sg_dw_c8.cu` / `sg_dxs_c8_*.cu` | `kernels/cuda/experimental/`, `kernels/cuda/diagnostics/` |
| `screen_fusion.cu` and the campaign scripts | experiment branches only (docs/evidence.md) |

## Validation harness

`benchmarks/det_training.py` record sets use schema `granule.det_records/2` (a manifest with identity
and provenance). Record directories written by the experiment branches (schema 1, `meta.json`) are
refused by the new comparer. Regenerate them, or compare with the original harness at its pinned
commit. `run_det_training.sh` now takes `<python> <scratch> <list> <output dir>` and exits nonzero
on any strict failure.
