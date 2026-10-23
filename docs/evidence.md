# Evidence index

Main keeps the implementation, its tests and the maintained tools. The raw evidence of the
optimization campaign (timing logs, JSON, SASS dumps, sanitizer logs, screening data) stays on the
experiment branches, which are kept with their full history. Every link below is pinned to a commit
and keeps working even if a branch moves.

All campaign measurements: one NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 24 GiB after the 2026-10-16
repair; older numbers are tagged `pre-24gib-rerun`), driver 550.67, nvcc 12.1.66, PyTorch 2.7.0+cu128,
Triton 3.3.0.

## RTX 3090 campaign

| Phase | Branch @ commit | Report | Raw evidence |
|---|---|---|---|
| First implementation | `main` @ `300a268` | [REPORT.md](https://github.com/webber26232/GranuleLayer/blob/300a268/sparse_pooling/REPORT.md) | [results/](https://github.com/webber26232/GranuleLayer/tree/300a268/sparse_pooling/results) |
| Re-measurement on the repaired card | `rerun-24gib` @ `d3595ac` | [TABLES.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/TABLES.md) | top-level `sparse_pooling/results/*.json` (kept in main) |
| Memory traffic: indexed dW staging, 32-warp dX at C = 64 | `dw-indexed-staging` @ `c47f001` | [MEMORY_TRAFFIC_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/MEMORY_TRAFFIC_OPTIMIZATION.md) | [results/traffic](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/traffic) |
| Backward compute: dZ epilogues, dX warp dispatch | `backward-compute-optimization` @ `fcc363b` | [BACKWARD_COMPUTE_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/BACKWARD_COMPUTE_OPTIMIZATION.md) | [results/backward_compute](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/backward_compute) |
| Input supergroups: connectivity, grouped kernels, SG_AUTO, C = 64 large buckets | `input-supergroup-optimization` @ `aeb41a5` | [INPUT_SUPERGROUP_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/INPUT_SUPERGROUP_OPTIMIZATION.md) | [results/input_supergroups](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/input_supergroups) |
| C = 8 dX scatter layouts, `swizzle4_capped`, scatter table | `c8-dx-scatter-optimization` @ `d613615` | [C8_DX_SCATTER_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/C8_DX_SCATTER_OPTIMIZATION.md) | [results/c8_dx_scatter](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/c8_dx_scatter) |
| S = 64 scale rule, fused backward (negative), FusedAdamW race fix (`eab9f59`) | `c8-scale-dispatch-fused-backward` @ `c9a591e` | [SCALE_DISPATCH_FUSED_BACKWARD.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/SCALE_DISPATCH_FUSED_BACKWARD.md) | [results/scale_dispatch_fused_backward](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/scale_dispatch_fused_backward) |
| Deterministic reductions, canonical bucket statistics, compact dW (negative) | `c8-deterministic-reduction-dw-projection` @ `2b9a84d` | [DETERMINISM_METRICS_DW_PROJECTION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/DETERMINISM_METRICS_DW_PROJECTION.md) | [results/determinism_metrics_dw_projection](https://github.com/webber26232/GranuleLayer/tree/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/determinism_metrics_dw_projection) |
| Forward MMA epilogue (not merged) | `mma-forward-epilogue-optimization` @ `11a996b` | on that branch | on that branch |

The campaign scripts (`run_*.sh`, `summarize_*.py`, screening models, `screen_fusion.cu`) are at the
same commits under `sparse_pooling/benchmarks/`. They are historical and pinned to their experiment
commits; the maintained tools in main replace them for new measurements.

## Which table came from where

| Profile entry (`backends/profiles/rtx3090.py`) | Source |
|---|---|
| `SG_AUTO`, `SG_CHUNK_UNITS`, `SG_AUTO_C64_LARGE_BUCKETS` | input supergroups |
| `SG_AUTO_DX_SCATTER`, `swizzle4_capped` | C = 8 dX scatter |
| `SG_AUTO_DX_PLAN` (4.0 split / 2.0 BF16 units per nonempty bucket, M ≥ 1024), deterministic `private_t32` | scale dispatch; denominator audit in the determinism phase |
| `SG_AUTO_DW_PROJECTION = "padded"` | determinism / dW projection §6.4 |
| `DW_CFG`, `DW_CFG_INDEXED`, `DW_INDEXED_STAGES`, `dx_warps` (32 at C = 64) | memory traffic |
| `epilogue` (auto dZ epilogues) | backward compute |

## This integration (`gpu-portability-cleanup`)

Two kinds of evidence, kept apart:

* **Historical** — produced while the refactor was written, before its review of `34a6433`:
  [validation/gpu_portability_cleanup.md](validation/gpu_portability_cleanup.md) (decision matrix, SASS
  identity, suites, det regression against 2b9a84d, performance A/B 2b9a84d vs 2f55d71), raw records in
  [sparse_pooling/results/gpu_portability_cleanup/](../sparse_pooling/results/gpu_portability_cleanup/).
  Its sanitizer results came from the runner the review found defective and are superseded.
* **Fresh, for the review fixes** — run on the fixed branch from a clean checkout, with the repaired
  sanitizer runner: [sparse_pooling/results/GPU_PORTABILITY_REVIEW_FIXES.md](../sparse_pooling/results/GPU_PORTABILITY_REVIEW_FIXES.md),
  raw records in [sparse_pooling/results/gpu_portability_review_fixes/](../sparse_pooling/results/gpu_portability_review_fixes/).
