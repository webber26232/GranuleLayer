# Validation of Cover-TopK/16 and the RTX 3090 K16 dX tuning

## 1. Result and source identity

**The integration head passed every required check on the local RTX 3090. No gate is pending.**

| | |
|---|---|
| Target `main` | `0d748aea48950665bdd2187daff698708a6491eb` (unchanged when rechecked before pushing) |
| Research reference | `interaction-coverage-k16-dx` @ `f4c538aca3356358179078a586d55a5e473743cd` (left intact) |
| **Validated source head** | `ad3a15c8baca4746d481db42ceaf8ca89f9a979b`: clean frozen checkout, source tree SHA-256 `3a7f69e8a581…` over 180 files |
| Commit adding this report | Changes only documentation and result records relative to the validated head |
| Records | [`sparse_pooling/results/validation/cover_topk16_rtx3090_dx/`](../../sparse_pooling/results/validation/cover_topk16_rtx3090_dx/) (index in its README) |

Environment: NVIDIA GeForce RTX 3090 (sm_86, 24 GiB, 370 W limit), driver 550.67, nvcc 12.1, g++ 11.4.0, PyTorch
2.7.0+cu128, Triton 3.3.0, Python 3.11.17. The kgen library hash is `f5cf1c857ed5bb7820a1` in every timing and training
record.

Sections 3-7 are new checks of the validated head. Section 8 is previous research evidence, and none of it tested this
head.

## 2. Dependency closure

This is a focused integration. It imports the accepted path, not the research branch's 4,677 changed files (4,532 of
them experiment results). The raw campaigns stay on the research ref.

| Area | Purpose |
|---|---|
| `sparse_pooling/cover_topk16.py`, `__init__.py` | The supported entry point: `CoverTopK16Layer`, construction, packed save / load, metadata |
| `experimental/kgen_lowered.py` | Lowers K64 envelope values and masks to compact K16 / C16 records |
| `experimental/kgen_layout.py`, `kgen_graph.py`, `kgen_layer.py`, `kgen_runtime.py` | Layout and envelope graph, the layer's forward / backward, native bindings. The runtime binds K8 / K16 / K32 / K64 entry points, so all four kernels are retained as dependencies, not as recommended features |
| `experimental/kgen_dx_variants.py` | Thin wrappers over the dX variant launch, kept for reproducing older callers |
| `experimental/local_supports.py` | Cover-TopK selection. It also holds Balanced quota repair (a shared-module dependency, not called by the accepted step) |
| `experimental/row_normalize.py`, `width_masks.py`, `canonical.py`, `reference.py` | One-time energy restoration, mask utilities, canonical identity, FP64 reference for tests |
| `kernels/cuda/experimental/kgen/*`, `backends/cuda/build.py`, `backends/cuda/variants.py` | Native kernels, including `kgen_k16_dxv`; `kgen` bundle registration and the shared-memory formula |
| `backends/dispatch.py`, `backends/profiles/__init__.py`, `backends/profiles/rtx3090.py` | dX legality and precedence, the call-time `KgenDxContext`, the scoped RTX 3090 entry |
| `memory_accounting.py`, `benchmarks/paired_stats.py`, `benchmarks/verify_research_contrasts.py` | The corrected reporting of section 9 |
| `benchmarks/cover_topk16_*.py`, `optimizer_equivalence.py`, `run_cover_topk16_validation.sh`, `run_sanitizer.sh`, `sanitize_{kgen,cover}_workload.py`, `provenance.py` | This validation |
| `tests/test_cover_topk16.py`, `test_kgen.py`, `test_kgen_dx_dispatch.py`, `test_memory_accounting.py`, `test_paired_stats.py`, `test_quota_repair_accounting.py`, `test_sanitizer_runner.py` | Regression coverage |

Balanced, Mix32, U16, refresh and L1 pruning are not promoted to new defaults. The existing `SparsePoolingLayer` API and
its defaults are unchanged.

## 3. Identity, budget, coverage and round trip

Full-size model from the user-supplied K64 / C256 checkpoints, energy normalization (`identity_g{0,1,2}.json`):

| Check | Graphs 0, 1, 2 |
|---|---|
| Valid weights / biases | 200,000,000 / 12,500,000 |
| Compact records | 781,312 |
| Original records with all 64 candidates covered | 49,152 of 49,152 |
| Sorted distinct local indices, sorted global indices, zero padding | yes |
| Output denominators | 12,207 to 12,208, summing to 12,500,000 |
| Packed round trip, 0 selection calls on load | pass (`save` 0.46 s, `load` 0.38 s, file 464 MB, graph 0) |

Support fingerprints match the research records on all three graphs (`e4465520b9cc…`, `a895f331ef41…`, `d8cb86e9e816…`).
Graph 0 also matches the research canonical values (`009224916823…`), mask (`b8c5d17fcaf2…`), packed weight
(`29efaf12a5d7…`) and packed bias (`539ce294a80e…`). Graphs 1 and 2 were compared on the support only.

Persistent memory by unique storage, graph 0:

| Category | MiB |
|---|---:|
| Weights | 381.5 |
| Bias | 23.8 |
| FP32 `main_grad` | 810.7 |
| BF16 Adam moments | 810.7 |
| Metadata (indices, map, sets, buckets, work lists, denominators) | 43.88 |

One-time costs, graph 0: selection 2.4 s; normalization and lowering 3.8 s; `from_checkpoint` total 16.7 s; construction
peak allocation 6.68 GiB.

## 4. Training smoke and optimizer equivalence

**Integration / trajectory validation, not an accuracy confirmation.** It covers 32 updates at M 512, seeds 0 / 1 / 2,
legacy against auto, on seeded synthetic development data with a learned projection and head (`train_smoke.json`).

| Seed | Executed | Identical start | Max abs loss diff | Final loss diff | Validation diff | Max rel. weight-grad-norm diff |
|---:|---|---|---:|---:|---:|---:|
| 0 | legacy / `mt4_nw8_wg4` | yes | 6.0e-8 | 0 | +1.5e-8 | 1.3e-5 |
| 1 | legacy / `mt4_nw8_wg4` | yes | 1.2e-7 | 0 | -3.0e-8 | 1.2e-5 |
| 2 | legacy / `mt4_nw8_wg4` | yes | 1.2e-7 | 0 | -1.5e-8 | 1.1e-5 |

The start state covers packed weight and bias, `main_grad`, moments, the rounding counter, and the projection and head.
All values are finite. The paired validation contrast has mean -9.9e-9 and upper bound 2.8e-8 (n = 3).

`FusedAdamW` replaces the research optimizer `L1AdamW(l1="none")`. Parameters, both moments, step counts, the rounding
counter and `main_grad` zeroing were **bitwise equal in all 8 configurations** (BF16 / FP32 moments, rounding on / off,
weight decay 0 / 0.01; five steps; `optimizer_equivalence.json`).

## 5. Same-session timing

Complete layer step in the benchmark: forward, the benchmark's loss, backward, `FusedAdamW`. Fresh process per job,
randomized declared order (seed 20261106), 3 rounds of 8 samples (24 per job), canonical restore and fresh optimizer
state per round. All start fingerprints matched. Medians in ms (`timing/summary.json`):

| Graph | M | Legacy step | Auto step | Ratio | dX legacy -> auto | Fwd legacy / auto | dW+db legacy / auto | Optimizer legacy / auto | Peak alloc GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 14.91 | 12.79 | 0.858 | 5.05 -> 2.97 | 0.92 / 0.90 | 3.78 / 3.83 | 5.05 / 5.05 | 2.42 |
| 0 | 512 | 58.09 | 43.95 | 0.757 | 37.25 -> 22.25 | 4.65 / 4.85 | 11.05 / 11.72 | 5.05 / 5.05 | 2.43 |
| 0 | 8192 | 797.68 | 590.58 | 0.740 | 556.72 -> 345.68 | 68.39 / 68.79 | 166.38 / 169.79 | 5.05 / 5.06 | 2.59 |
| 1 | 8192 | 796.85 | 587.49 | 0.737 | 556.34 -> 345.01 | 68.36 / 68.36 | 166.19 / 167.87 | 5.06 / 5.06 | 2.59 |
| 2 | 8192 | 796.72 | 585.53 | 0.735 | 556.16 -> 343.74 | 68.32 / 68.35 | 166.32 / 167.36 | 5.05 / 5.06 | 2.59 |

**Graph mean at M 8192: 797.1 -> 587.9 ms, ratio 0.738.** This clears the floor of at least a 5 % improvement on every
graph. Every auto job executed `mt4_nw8_wg4` through `sp_kg_k16_dxv` with status `hardware-validated`, 64 tokens per CTA,
256 threads and 42 KiB of shared memory (legacy `sp_kg_k16`: 74 KiB).

These are benchmark measurements on one GPU. They are not the wall time of an application, and they promise nothing for
other hardware. Per-job records hold host enqueue and wall time, and clocks, power and temperature per round. Steady-state
scratch is not reported as its own counter; the peak above includes persistent tensors, the batch and scratch.

## 6. Compatibility of the existing fan-in-8 paths

Compared against `main` in alternating fresh processes (`compat_smoke.json`): D 1024, G 256, H 1M, S 64, R 16.

| Case | Plan equal | y rel. | dX rel. (BF16) | dW rel. | Step ms, PR / main |
|---|---|---:|---:|---:|---:|
| C 8, auto | yes | 4.6e-7 | 2.4e-3 | 2.5e-7 | 4.80 / 4.81 |
| C 64, auto | yes | 3.8e-7 | 2.4e-3 | 3.3e-7 | 3.69 / 3.66 |
| C 8, `generic_cuda` profile | yes | 4.6e-7 | 4.8e-3 | 2.9e-7 | 7.69 / 7.66 |
| C 64, generic backend | yes | 0 | 1.2e-3 | 0 | 3.72 / 3.70 |

The tolerances are 1e-5 for y and dW and one BF16 ulp (2^-7) for `x.grad`, because FP32 atomic sums are not bitwise and
can round to adjacent BF16 values. **The production library is identical to main's** (`e55eb569423be99b2776`). The step
times are informational.

## 7. Tests and sanitizer

* **Fresh CUDA cache build and focused tests:** 242 passed.
* **Complete suite:** **3076 passed, 32 skipped, 0 failed** (`logs/full_suite.log`).
  * All 2,963 test ids collected on `main` are still collected here (3,108 total).
  * The count differs from the research branch because this PR transfers a narrower set of dependencies.
  * This run's log does not list skip reasons, so they are not confirmed for this head. `main`'s last suite also had 32
    skips, with reasons in `sparse_pooling/results/gpu_portability_review_fixes/gpu_suites/suite_auto.log`.
* **Compute Sanitizer: 24 / 24 PASS** (`sanitizer/summary.tsv`): memcheck, racecheck, initcheck and synccheck over
  `cover`, `kgen`, `generic`, `sg`, `dx_scatter` and `det`.
  * `cover` runs legacy and WG4, ragged tails, partial groups, token tails B = 8 / 40 / 136 / 512, non-finite upstream
    gradients, `main_grad` accumulation, `FusedAdamW` and a packed round trip.
  * Its small geometry is outside the production scope, so the workload widens the RTX 3090 entry in-process to reach
    the auto path.
  * The only exclusion is the existing documented one: racecheck skips the Triton `bwd_dense_kernel`.

Dispatch coverage is in `tests/test_kgen_dx_dispatch.py`: auto, legacy, forced generic, fake non-3090 capabilities,
illegal explicit requests, deterministic rejection, out-of-scope fields, executed metadata and launch-attribute
interleaving. Numerics against the FP64 reference are in `tests/test_cover_topk16.py` and `tests/test_kgen.py`.

## 8. Previous research evidence (not this head)

All pinned to `f4c538a`: [report](https://github.com/webber26232/GranuleLayer/blob/f4c538aca3356358179078a586d55a5e473743cd/sparse_pooling/results/INTERACTION_COVERAGE_K16_DX.md),
[timing](https://github.com/webber26232/GranuleLayer/blob/f4c538aca3356358179078a586d55a5e473743cd/sparse_pooling/results/interaction_coverage_k16_dx/csv/confirm.csv),
[final contrasts](https://github.com/webber26232/GranuleLayer/blob/f4c538aca3356358179078a586d55a5e473743cd/sparse_pooling/results/interaction_coverage_k16_dx/csv/final_primary.csv),
[campaign records](https://github.com/webber26232/GranuleLayer/tree/f4c538aca3356358179078a586d55a5e473743cd/sparse_pooling/results/interaction_coverage_k16_dx).

* Complete layer step at M 8192: 801.3 -> 592.3 ms, ratio about 0.739 (graph mean of 3). dX about 559 -> 348 ms. M 512:
  58.1 -> 43.6 ms.
* Compact-backend training, six seeds: identical packed starts; locked test losses within 7.5e-7 per seed (below 8e-7);
  adjusted upper bound about 8.22e-7.
* Architecture-quality loss: Cover about 0.16973, on the **separate** dense-masked reference protocol. It is not the
  timing task.
* Full suite 3276 passed / 32 skipped; sanitizer 20 / 20 PASS.

Recomputing those contrasts from the raw records with the corrected cohort validation of section 9 gave **unchanged
values** at the report's printed precision, with complete inventories (`research_contrasts.json`; backend upper bound
8.2212e-7).

## 9. Errata of the research report

The raw measurements are not rewritten; these correct derived summaries and tools.

* **Memory table (`icdx_report.py::memory`).** It summed every numeric field outside four categories, so it counted
  `compact_values` as well as weights, both `indices` and `index_records_int16`, and the `active_tiles` count as bytes.
  * Corrected metadata: indexed K16 **43.8672 MiB** (printed 67.7), U16 wide **10.6524 MiB** (printed 392.9).
  * `sparse_pooling.memory_accounting` now counts unique storage and keeps element counts apart from bytes. The live
    layer reports 43.88 MiB because it also holds the per-output denominators and spec data. Measured peaks and timing
    are unchanged.
* **Quota-repair synchronization.** `host_syncs=18` counted selected explicit polling checks, not every host / device
  synchronization. The dynamic-length boolean indexing in `quota_repair` can also synchronize.
  * The field is now `polling_checks`, with a deprecated `host_syncs` alias and a `synchronization_note`.
  * No synchronization removal is claimed. Cover-TopK/16's accepted step does not call this helper.
* **Cohort identity (`paired_stats`).** A cohort learned its identity from its first record, so two internally consistent
  cohorts on a wrong protocol could return `complete / noninferior`. `Cohort` and `paired` now require the declared
  `Experiment` and `Method` identities, and reject wrong whole cohorts, including the same wrong protocol on both sides.
* **Functional-ANOVA wording.** The report said the bound holds for "any function of the supports". See section 10.

## 10. Scientific claims and recommendation

Architecture and backend are recommended independently.

* **Architecture.** Cover-TopK/16 is the accepted full-width sparse recipe. Balanced and Mix32 remain research
  comparisons, not preferred replacements. U16 remains a quality / width tradeoff reference.
* **Backend.** `mt4_nw8_wg4` is the automatic choice only inside the measured RTX 3090 scope: exact RTX 3090 matcher,
  D 1024 / S 64, compact K16 / C16, Mean(ReLU), split dZ, fast reduction, H 12.5M / G 1024 / R 256 Cover-TopK/16,
  M = 64, 512 or 8192. Everything else gets the legacy launch unless a legal variant is requested explicitly.
* **The bound.** The functional-ANOVA bound applies to additive sums of functions on the available supports with the
  diagnostic's per-output affine readout.
* **Training minus bound.** A positive gap is unresolved unless an attainable solution or another argument identifies its
  cause.
* **Full network.** The raw-input diagnostic's bound does not directly bound the full network with a learned projection
  and head.

## 11. Commands

From a clean checkout of the validated head. Paths are yours; the checkpoints are the K64 / C256 envelopes for graphs
0 / 1 / 2 and are not in git.

```bash
PY=python OUT=/path/results BASELINE=/path/checkout-of-main \
CKPT_G0=/path/g0.pt CKPT_G1=/path/g1.pt CKPT_G2=/path/g2.pt \
RESEARCH_L1=/path/research/sparse_pooling/experimental/l1.py \
RESEARCH_FINAL=/path/research-results/final RESEARCH_PRIMARY_CSV=/path/final_primary.csv \
EXPECT_G0="--expect-support e4465520b9cca51b91cf19cbfc6ba9771f7ec988149b548425574d5eeb643a81 --expect-canonical 009224916823c17fb145e30a4f9748a06a6b13b0ed9ba5e35379efc82fabcfc9 --expect-mask b8c5d17fcaf2406f6364966d00441acbfce72c180b098a558c87777d9767f14a --expect-weight-sha 29efaf12a5d779ed24e9021e1d41d1ad87641abf1d8e9914775a5ca9b7ff7d18 --expect-bias-sha 539ce294a80eb0df1d29b8a49c3003e38e48cdb1a0ac88a9036dc18b36237953" \
EXPECT_G1="--expect-support a895f331ef4146d9ce099e75f35ec80eca22499a4ca631bb00686c80fbd6e916" \
EXPECT_G2="--expect-support d8cb86e9e816b9007419b8bfd3909446b0e9b755a1b14b2a15ea988a07c4bc3d" \
bash sparse_pooling/benchmarks/run_cover_topk16_validation.sh
```

The script serializes GPU work and keeps the sanitizer out of the timing session. Its exit status is the number of failed
stages. The raw research records for `RESEARCH_FINAL` are not in git, so that stage needs a local copy.

## 12. Limitations

* One GPU model, synthetic data. Nothing here is a downstream-accuracy claim or a speed promise for other GPUs.
* The training smoke is 32 updates and 3 seeds. Six-seed confirmation remains research evidence, at `f4c538a`.
* FP32 atomic reductions make training not bitwise deterministic; deterministic mode is rejected on this path.
* Graphs 1 and 2 were compared with research on support identity only.
* The kgen bundle still lives in the `experimental` kernel and module namespace, and its build record says
  `status: experimental`. The supported surface is `CoverTopK16Layer`.
* The RTX 3090 entry needs fresh same-function evidence before its scope is broadened.
