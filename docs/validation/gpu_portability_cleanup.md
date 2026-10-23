# Validation of the GPU-portability refactor (historical, before the review)

Baseline (A): `c8-deterministic-reduction-dw-projection` @ `2b9a84d` (the reviewed implementation).
Refactor (B): `gpu-portability-cleanup` at the commits named per section below. Raw records:
[sparse_pooling/results/gpu_portability_cleanup/](../../sparse_pooling/results/gpu_portability_cleanup/) (provenance of every file in its
[README](../../sparse_pooling/results/gpu_portability_cleanup/README.md)).

These results were produced by the sessions that wrote the refactor, **before** the review of
`34a6433`. The review found that the sanitizer runner of that time could report success after a failed
run, that the documentation was not committed, and that the target parser mishandled multi-digit
majors. The fixes, and fresh results on the fixed branch with the repaired runner, are in
[../../sparse_pooling/results/GPU_PORTABILITY_REVIEW_FIXES.md](../../sparse_pooling/results/GPU_PORTABILITY_REVIEW_FIXES.md).
This page is kept as the historical record of the refactor itself.

Environment (one machine, no competing GPU jobs during timing): NVIDIA GeForce RTX 3090 (sm_86, 82 SMs,
24 GiB, 99 KiB opt-in shared memory per block), driver 550.67, nvcc 12.1.66 (host compiler g++ 11.4.0),
PyTorch 2.7.0+cu128, Triton 3.3.0. Build of B: `production` bundle, target sm_86, flags
`-O3 --use_fast_math -lineinfo -std=c++17`, ABI 3.

## 1. Decisions and generated code

* **Dispatch decisions**: `tests/test_dispatch.py::test_rtx3090_reproduces_the_reviewed_decision_matrix`
  reproduces all 4,560 host-side decisions captured from 2b9a84d (grouped phases, S = 64 scale rule,
  scatters, dW projection, deterministic mode, generic tilings, epilogues, staging, error cases) under the
  automatically matched RTX 3090 profile.
* **SASS identity** (`benchmarks/compare_sass.py`: per kernel, the cuobjdump SASS instruction and
  encoding lines with line-info comments removed and whitespace normalised, plus registers / stack /
  static shared / local memory), the baseline's single library against B's production + experimental +
  diagnostics bundles: 2,321 common kernels, **2,321 identical SASS, 0 resource differences**; only in A:
  the two `screen_fusion.cu` timing kernels (archived). Records:
  [sass_all_bundles.json](../../sparse_pooling/results/gpu_portability_cleanup/sass/sass_all_bundles.json), [sass_production.json](../../sparse_pooling/results/gpu_portability_cleanup/sass/sass_production.json).
  The compared B libraries (`e55eb569…`, `ec72e621…`, `3f1bf046…`) are the build identities of
  `2f55d71` and `34a6433`, whose device-code inputs are identical.
* **Compile-only targets** (production bundle, nvcc 12.1): sm_80, sm_89 and sm_90 each build all 1,881
  production kernels ([log](../../sparse_pooling/results/gpu_portability_cleanup/sass/compile_only_sm80_sm89_sm90.log)). Not run on hardware; this is the
  "compiles" status only.

## 2. Test suites (RTX 3090, `c75c402`)

| Run | Result | Log |
|---|---|---|
| Full suite, `auto` profile (rtx3090) | **2,847 passed, 32 skipped, 0 failed** (5 min 28 s) | [suite_auto.log](../../sparse_pooling/results/gpu_portability_cleanup/suites/suite_auto.log) |
| Full suite, `SPARSE_POOL_PROFILE=generic_cuda` | **2,829 passed, 50 skipped, 0 failed** (5 min 18 s) | [suite_generic_cuda.log](../../sparse_pooling/results/gpu_portability_cleanup/suites/suite_generic_cuda.log) |

Reviewed baseline (2b9a84d): 2,748 passed, 31 skipped. The new passes are the added tests; no test was
deleted. The one additional skip is the real two-device routing test (one GPU available). Under the
generic profile 18 further tests skip through the `rtx3090_policy` marker: they assert only the RTX 3090
profile's measured automatic choices. Every numerical comparison runs under both profiles.

## 3. Compute Sanitizer

| Run | Result |
|---|---|
| previous sequential runner, `c75c402` ([output](../../sparse_pooling/results/gpu_portability_cleanup/sanitizer/sequential_runner_c75c402.txt)) | memcheck / initcheck / synccheck 0 errors on all four workloads; racecheck 0 hazards on sg, dx_scatter, det; generic: 37,262 racecheck errors, all in the Triton `bwd_dense_kernel` |
| `--jobs 5` runner as committed at `34a6433` ([output](../../sparse_pooling/results/gpu_portability_cleanup/sanitizer/parallel_runner_34a6433.txt)) | 16 of 16 summaries with 0 errors / 0 hazards (racecheck excludes `bwd_dense_kernel`) |

The Triton reports are the `tl.sum` reduction at `triton_fused.py:441` writing replicated values; the
baseline shows them too (37,642 at 2b9a84d vs 37,505 at c75c402 in two single runs, every sampled access
at that line: [2b9a84d](../../sparse_pooling/results/gpu_portability_cleanup/sanitizer/racecheck_generic_2b9a84d.log.gz),
[c75c402](../../sparse_pooling/results/gpu_portability_cleanup/sanitizer/racecheck_generic_c75c402.log.gz)), and the file is unchanged by the refactor.
With that kernel excluded, racecheck reports 0 hazards for the generic workload
([log](../../sparse_pooling/results/gpu_portability_cleanup/sanitizer/racecheck_generic_c75c402_bwd_dense_excluded.log)).

**Caveat:** both runner outputs come from the runner the review found defective (exit statuses not
checked, per-run logs deleted). They are not accepted as sanitizer evidence; the repaired runner's fresh
results are in the review-fix report.

## 4. Deterministic training regression (old versus new)

`benchmarks/run_det_regression.sh ... all` with validator `det_training.py` (schema
`granule.det_records/2`, strict mode): for each case the baseline checkout writes the records of both
trajectories in one process, and the refactor compares against them in an independent process — outputs,
loss, dX, gradients (`main_grad`), parameters, optimizer moments / master weights / SR state after every
step, byte for byte with the expected record counts, plus identical resolved plans and state
restoration. Cases with `G_` compare two processes of the refactor under the generic profile. Nonlinear
loss (tanh head, MSE), Mean(ReLU) layer with real R > 1 supergroup connectivity, FusedAdamW. The first run
was killed by systemd-oomd after 10 passing runs and resumed from the next case
([part 1](../../sparse_pooling/results/gpu_portability_cleanup/det_regression/run_part1.log), [part 2](../../sparse_pooling/results/gpu_portability_cleanup/det_regression/run_part2_resumed.log)).

| Case | Kind | det_training case | optimizer | dZ | plan | steps | compared steps | tensors compared | differing | status | same plan | state restored | tested | reference | peak GiB |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|
| G_M64_S16_R4_sr_split | new vs new (generic_cuda) | M64_S16_R4 | sr | split | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 34a6433 | 2.11 |
| G_M_S64_R16_det_split | new vs new (generic_cuda) | M_S64_R16 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 34a6433 | 2.14 |
| M64_S16_R4_det_split | old vs new | M64_S16_R4 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 2.11 |
| M64_S64_R16_det_bf16 | old vs new | M64_S64_R16 | det | bf16 | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 2.10 |
| M_S16_R4_sr_split | old vs new | M_S16_R4 | sr | split | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 2.14 |
| M_S32_R8_det_split_cap64M | old vs new | M_S32_R8 | det | split | deterministic_workspace_bytes=67108864,cuda_grouped_dx_tokens=64,cuda_grouped_dx_scatter=baseline | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 2.14 |
| M_S64_R16_det_bf16 | old vs new | M_S64_R16 | det | bf16 | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 2.14 |
| M_S64_R16_fp32_split | old vs new | M_S64_R16 | fp32 | split | - | 16 | 32 | 1120 | 0 | pass | True | True | 34a6433 | 2b9a84d | 3.51 |
| P_S16_R4_det_split | old vs new | P_S16_R4 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | 34a6433 | 2b9a84d | 9.81 |
| P_S32_R8_det_split | old vs new | P_S32_R8 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | 34a6433 | 2b9a84d | 9.81 |
| P_S64_R16_det_split | old vs new | P_S64_R16 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | 34a6433 | 2b9a84d | 10.00 |
| S_S16_R4_det_split | old vs new | S_S16_R4 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | c75c402 | 2b9a84d | 0.30 |
| S_S64_R16_det_split | old vs new | S_S64_R16 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | 34a6433 | 2b9a84d | 0.30 |

13 comparisons, 13 pass, 0 differing tensors

Harness self-test (`run_det_training.sh selftest`, deliberate negative controls detected):
"all strict pairs passed" ([log](../../sparse_pooling/results/gpu_portability_cleanup/det_regression/selftest.log)).

## 5. Performance (A/B, `2b9a84d` vs `2f55d71`)

Protocol: `benchmarks/run_impl_ab.sh`, `bench_supergroups.py` complete training steps (forward,
backward, FusedAdamW) with restored training state, 2 repetitions × 3 steps per round, A and B
alternating in order each round, identical graphs and initial state; paired per-round difference B − A
of the median step. Raw records: [ab/](../../sparse_pooling/results/gpu_portability_cleanup/ab/), summary [ab_summary.md](../../sparse_pooling/results/gpu_portability_cleanup/ab/ab_summary.md).

| Config | Variant | A step ms | B step ms | paired B − A (median [min, max]) | % | rounds | same plan | peak alloc A / B GiB |
|---|---|---:|---:|---|---:|---:|---|---|
| C64_S16_R4 | det | 1703.3 | 1704.2 | +2.78 [-1.59, +17.99] | +0.16 | 4 | yes | 9.50 / 9.50 |
| C64_S16_R4 | fast | 1749.2 | 1750.1 | +2.43 [-0.46, +3.39] | +0.14 | 4 | yes | 8.69 / 8.69 |
| C64_S64_R16 | fast | 1779.4 | 1782.1 | +2.70 [-6.05, +26.41] | +0.15 | 4 | yes | 8.68 / 8.68 |
| C8_S16_R4 | det | 2532.8 | 2535.7 | +3.50 [-1.35, +19.71] | +0.14 | 4 | yes | 9.66 / 9.66 |
| C8_S16_R4 | fast | 2523.8 | 2521.9 | +4.37 [-10.87, +13.46] | +0.17 | 4 | yes | 8.84 / 8.84 |
| C8_S32_R8 | det | 2760.1 | 2762.7 | +1.40 [-18.65, +4.89] | +0.05 | 4 | yes | 9.66 / 9.66 |
| C8_S32_R8 | fast | 2755.6 | 2752.1 | -2.86 [-23.03, +12.46] | -0.10 | 4 | yes | 8.84 / 8.84 |
| C8_S64_R16 | det | 3153.4 | 3156.6 | +2.73 [+2.46, +4.01] | +0.09 | 4 | yes | 9.84 / 9.84 |
| C8_S64_R16 | fast | 3143.5 | 3146.3 | +0.77 [-8.03, +9.20] | +0.02 | 4 | yes | 8.84 / 8.84 |
| C8_S64_R16_bf16 | fast | 2937.3 | 2949.4 | +11.72 [+1.99, +20.24] | +0.40 | 4 | yes | 8.84 / 8.84 |
| H10M_C8_S16_R4_M1000 | det | 38.9 | 39.3 | +0.41 [-0.17, +0.98] | +1.05 | 2 | yes | 1.88 / 1.88 |
| H10M_C8_S16_R4_M1000 | fast | 37.4 | 37.8 | +0.39 [+0.25, +0.54] | +1.05 | 2 | yes | 0.91 / 0.91 |
| H10M_C8_S64_R16 | fast | 369.0 | 370.0 | +1.03 [+0.69, +1.36] | +0.28 | 2 | yes | 1.12 / 1.12 |
| H1M_C8_S64_R16 | fast | 38.9 | 40.3 | +1.43 [+0.44, +2.41] | +3.68 | 2 | yes | 0.34 / 0.34 |
| H1M_C8_S64_R16_M1000 | fast | 5.2 | 5.2 | +0.03 [-0.01, +0.07] | +0.49 | 2 | yes | 0.12 / 0.12 |

All resolved plans and peak memory are identical between A and B. Every production-scale configuration
is within +0.4 % (paired median). H1M_C8_S64_R16 showed +3.7 % in its 2-round entry; an 8-round paired
rerun gave −1.15 ms [−2.70, +0.30] (−2.9 %) ([records](../../sparse_pooling/results/gpu_portability_cleanup/ab_h1m_8rounds/); that rerun's runner log was not kept).
At about 39 ms per step this configuration is dominated by run-to-run variance; with identical SASS and
plans only host-side work can differ.

## 6. Other hardware

None was available. Other devices are covered by synthetic capability records (control flow only) and by
compile-only builds (§1); the support matrix in [../gpu_portability.md](../gpu_portability.md) lists them as
compile-tested or capability-eligible but hardware-unvalidated.
