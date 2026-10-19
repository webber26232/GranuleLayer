# Scale-aware S = 64 dispatch, deterministic training validation, shared-projection backward

RTX 3090 (GA102, sm_86, 24 GiB), C = 8 GranuleLayer, D = G = 1024 unless stated, K = 8,
Mean(ReLU), BF16 parameters / inputs, FP32 MMA accumulation and main gradients, split gradients
unless stated, FusedAdamW with BF16 moments. Raw evidence: `results/scale_dispatch_fused_backward/`.

Labels used below: **measured** (CUDA-event timings, exact comparisons), **computed** (counts from
metadata or source), **static** (SASS / resource dumps), **ablation** (wrong-result builds, timing
only), **interpretation** (an explanation consistent with the measurements, not separately
verified). No hardware performance counters exist on this machine (`RmProfilingAdminOnly=1`).

## 1. Summary

1. **Deterministic control: yes, bitwise identical trajectories.** In the constructed control, all
   162 comparisons of the order-preserving scatter variants against the reference are bitwise
   identical. The control uses a trainable projection → BF16 → layer → tanh head → MSE, a generic
   forward, R = 1, one dX chunk per supergroup and private copies. Every tensor matches at every
   step, across 5 shapes and 21 optimizer / precision / seed configurations: outputs, loss, dX,
   dW / db, all parameters and all optimizer state. A single writer per dX element was proven from
   the actual work lists.
   * Before that, the control found a defect elsewhere: **a race in the existing FusedAdamW
     kernel.** Triton loads `main_grad` in one thread layout and zeroes it in another, so a few
     elements per step lost their gradient nondeterministically.
   * Fixed with one barrier (§3.3). Optimizer time is unchanged within noise.
2. **Crossover descriptor: the mean work units per nonempty (output, supergroup) bucket.** H,
   units per chunk and M do not predict it.
   * **Measured relation:** Plan B (`private_t32`) changes the step by between +16 % and −3.5 %
     against Plan A (`shared_t64`) as bucket length goes from 0.12 to 14 units (§4.2).
   * **Discriminating case:** G = 256 at H = 10M has the per-chunk work of a losing H = 10M,
     G = 1024 case, but 4× longer buckets, and B wins.
   * **Rule:** B at ≥ 4 units / bucket (split) or ≥ 2 (BF16) when M ≥ 1024. The 13 held-out shapes
     (none used to choose the thresholds) cover ragged H, G = 512, M = 1024 / 2048, R = 8 / 32,
     seed 1 and BF16.
   * **Held-out performance:** the rule picks the better plan in 12 of them (regret ≤ 0.34 %). It
     misses by 1.6 % on one BF16 shape just below the conservative BF16 threshold (§4.5).
3. **Complete-step gain at H = 100M, S = 64, split: −3.1 % (−100.5 to −102.5 ms per step).**
   * **Seeds:** 3 graph / input / init seeds, same session. Per seed: 3230.2 → 3128.7,
     3234.4 → 3131.9, 3234.8 → 3134.3 ms.
   * **dX:** 1738 → 1638 ms.
   * **Smaller shapes:** no regression. Below the thresholds the rule keeps Plan A, so the new
     automatic plan is the previous one, within noise (≤ +0.34 %).
   * **BF16 (S = 64):** the rule also replaces the previous BF16 default (the original scatter).
     The step drops 6–13 % on every measured BF16 shape (§4.5, §4.6).
4. **The dX and dW projections and production ReLU decisions are interchangeable, including near
   zero:**
   * the two paths' pre-activations are **bitwise identical** in every regime tested: random,
     signed zeros, subnormal products and biases, cancellation, near-zero z, inf / NaN;
   * the production mask decisions agree element for element (§5.2). All production compares use
     flush-to-zero.
   * **Hazard for any fused kernel:** when a `set.gt.ftz` mask is only tested for non-zero, ptxas
     12.1 emitted a compare without FTZ.
5. **Projection evaluations removed: one per (neuron, token), i.e. 2 → 1.** Useful work removed is
   13.1 TFLOP at H = 100M, M = 8192.
   * In issued instructions, sharing removes dW's two HMMA.1688 per block-tile: 12.8 G instructions,
     26.2 TFLOP issued, because C = 8 pads half of dW's m16 tile.
   * The shared kernel's SASS has exactly the dX-layout projection left (§5.5).
6. **Added costs of the dW-owned schedule (F1) at H = 100M, S = 16** (computed / measured, §5.3–5.4):
   * **Global dX flushes:** 25.6 G atomics (vs 0.10 G); 121 ms in isolation, verified exact. The
     ablation shows only about 3 ms of that inside the fused kernel at H = 10M.
   * **Weight-gradient partials:** none. Ownership is exclusive; dW / db are written once,
     bitwise equal.
   * **Window reductions:** 32–64 extra shared operations per block-tile for the warp-private window
     copies.
   * **Registers:** 119–165, against 86 (dX) and 127 (dW).
   * **Scratch:** no new global scratch. The metadata reuses dW's unit chunks.
   * **Other schedules:** F2 (coarse chunks) needs 58–230 GB of dW partial read-modify-write and
     F3 (mask exchange) 205 GB of mask traffic; both were rejected by the model.
7. **No: fusion does not beat the optimized separate backward after all overhead** (§5.6, §8).
   * **Sharing saving:** 12 % of the fused kernel's time, measured with a recompute control on the
     same schedule (−339 ms at 100M).
   * **Schedule cost:** about the same, mainly dX window reductions, lost tile amortization and
     double coefficient loads.
   * **Best fused vs separate:** −2.4 % at 10M and −0.4 % to +2.5 % at 100M (one session each), not
     above noise.
8. **Defaults:**
   * **New:** the S = 64 scale-aware plan rule (split and BF16, sm_86, M ≥ 1024); the FusedAdamW race
     fix.
   * **Experimental (explicit only):** `cuda_grouped_backward = fused | fused_recompute` (C = 8,
     Mean(ReLU), split).
   * **Unchanged:** every other configuration.
   * **Failed / rejected:** the fused schedules (F1 measured, F2 / F3 modeled), and the null-control
     envelope as a pass / fail criterion (kept as characterization, §3.5).

## 2. Starting state, environment, provenance

* Started from `c8-dx-scatter-optimization` at **`d6136150d8f87e8335d0e4ed07c5a341bc20b9c0`** (the
  reviewed commit; the branch had not advanced). New branch `c8-scale-dispatch-fused-backward` in
  the worktree `../GranuleLayer-sdfb` (fused development in a second worktree on `sdfb-dev`,
  fast-forwarded into the branch). Nothing pushed or merged; other branches untouched.
* **Uncommitted outputs found in the previous worktree.** `run_c8dx_post.sh` was still running when
  `d613615` was committed. It finished later and rewrote 8 committed evidence files:
  * a second execution of the fixed-dY numerics and training validation;
  * the memory re-check and the test suite.

  They are preserved without overwriting anything, in `c8_dx_scatter/post_run_completed/` (commit
  `0ce9989`), and the old worktree's changes are stashed. Its deterministic rows are again bitwise
  and the suite again 2551 passed.
* Environment: driver 550.67, nvcc 12.1 (V12.1.66), PyTorch 2.7.0+cu128, Triton 3.3.0; flags
  `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17`. GPU state is recorded before / after each
  run (SM 1965–1995 MHz at the start, 1695–1830 MHz at the end, 34–54 °C, 370 W limit).
* Commits (in order): `0ce9989` preserved outputs · `f6cac91` whole-plan control + descriptors ·
  `eab9f59` FusedAdamW race fix · `27d5748` deterministic control, projection diagnostic, fusion
  model and screens · `0df8918` fused prototype (explicit) · `8387a1c` S = 64 rule · `00d02fb`
  training-validation plan comparisons · `1e7de43` crossover / fused screening evidence ·
  `83db1cc` held-out script · `518b16a` held-out evidence, SASS counts, final scripts · (later)
  final evidence and this report.
* **Which commit produced which data:**

  | Data | Commit / library | Notes |
  |---|---|---|
  | Held-out validation (`holdout/`) | `83db1cc`, clean, library `5b9c82fc0e35fb38` | |
  | Final matrix (`final_100M/`) | `518b16a`, clean, the same library | |
  | Stage-1 screen (`crossover_screen/`) | `f6cac91`, library `d6c776f9f7e0014d` | Before the optimizer fix; both plan arms equal; the fix does not change optimizer time (§3.3) |
  | Stage-2 screen (`crossover_stage2/`) | `27d5748`, library `cb1e745efb884e2c` | |

* **Two stage-2 incidents:**
  * **`H40000000_R16.json` records the wrong library (`a8b11455…`).** That library was built at save
    time, after unrelated edits; the run itself loaded `cb1e745…`
    (`crossover_stage2/NOTE_H40M_library_field.txt`). Since then the benchmark records the library
    it loads before timing.
  * **H = 20 / 40 / 70M overlapped a concurrent sanitizer process.** They were rerun on an idle GPU
    (`*_rerun`, `NOTE_concurrent_sanitizer.txt`). The reruns agree within 0.3 points of B − A.

## 3. Track V: deterministic nonlinear-training control

### 3.1 Construction and ownership (`benchmarks/deterministic_training.py`)

**Model.** x0 [M, D0] → `Linear(D0, D)` (FP32, torch AdamW) → BF16 → GranuleLayer → `Linear(G, G2)`
→ tanh → MSE with a nonzero target. The layer has BF16 parameters and FP32 `main_grad`, trained by
FusedAdamW (BF16 moments). Batches are pre-generated and change every step.

**Layer plan:**
* generic forward for every variant;
* grouped dX with R = 1 and `cuda_grouped_dx_chunk_units = 2^30`, warp-private copies;
* grouped dW, identical in all runs.

**Projection / head:** `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
(no determinism warnings were raised). Determinism is verified by a `baseline_repeat` run, not
assumed.

**Ownership, proven per run from the actual work lists** (`ownership()`):
* one chunk per supergroup;
* the feature sets partition D (every feature covered exactly once);
* every (feature, token) of dX has exactly one writer CTA, which issues one FP32 atomic onto zero;
* the copies are reduced in a fixed order;
* `single_writer = True` in all 21 configurations, including token tails of 8.

**Cases:** D = 256, G = 16, H = 50k, M = 136 for S = 16 / 32 / 64; D = 1024, G = 8, H = 100k, M = 64,
S = 16; D = 1024, G = 8, H = 20,007 (ragged), M = 136, S = 64.

**Modes:**
* 16 steps with stochastic rounding off;
* 8 steps with stochastic rounding for two paired seeds of the optimizer counter;
* 8 steps with FP32 moments and master weights;
* 16 steps with BF16 gradients (one case).

**Restore check (SR mode):** state captured after one step, restored with `TrainState`, and the
trajectory replayed. Both fingerprints were restored and the replay is bitwise.

### 3.2 Results (`deterministic_training_summary.txt`; repeat run `deterministic_training_repeat*`)

| Comparison (same tile, private copies) | Category | Result in all 21 configurations |
|---|---|---|
| `baseline_repeat` (control) | same | **bitwise** every tensor / step |
| `baseline` at the other token tile (T 32 ↔ 64; not at S = 64 where 8 private T = 64 copies exceed 99 KiB) | same | **bitwise** |
| `baseline_private`, `swizzle4`, `swizzle4_capped` (S = 16), `vectorized`, `vectorized_hoisted`, `vectorized_transpose`, `vectorized_transpose_hoisted`, `vectorized_parity` (S = 16) | same | **bitwise** (162 / 162 comparisons; the full repeat run identical) |
| `feature_owner` (S ≤ 32) | order (MMA accumulation) | differs from step 0 in dX; final relative L2: dX 5e-5 – 4.5e-3, layer W 6e-6 – 5.5e-5, loss ≤ 2.3e-7 |
| shared CAS copy, T = 64, WG = 8 (scheduled order inside the CTA) | order | differs from step 0 in dX; dX ≤ 2.6e-3, W ≤ 4.2e-5 |
| Plan A (`swizzle4_capped`, shared copy) vs the private reference (S = 64), and a second Plan A run | order | dX ≤ 2.1e-3 / 1.8e-3, W ≤ 3.5e-5 / 3.5e-5: Plan A against the reference is the size of Plan A against itself |

Tracked tensors:
* the layer output y, loss, layer-input gradient dX, layer `main_grad` dW / db, projection and head
  gradients;
* all parameters after the updates;
* FusedAdamW m / v (and master), torch AdamW moments, the step and the SR counter.

Plan B (`vectorized_transpose_hoisted`, private T = 32) at S = 64 is one of the bitwise rows: it is
order-preserving against the original layout with the same private tile.

### 3.3 The first run was not bitwise: a race in FusedAdamW (`optimizer_race/`, commit `eab9f59`)

**What the first run showed** (kept in `before_optimizer_fix/`). In one shape (D = 1024, G = 8,
H = 100k, M = 64) two variants differed from the reference from step 1 / 7: `W`, `opt.m0` and
`opt.v0` differed while every gradient (y, loss, dX, dW, db, projection) was bitwise equal. Reruns
moved the difference to other variants and steps, eventually to `baseline_repeat` itself, with or
without a device synchronization between runs.

**Isolation (measured).** A FusedAdamW update of identical inputs was repeated 40 times
(n = 812,345 and 819,200, BF16 moments, deterministic rounding). It produced **15 and 8 distinct
final states**, and 522 / 1137 element-steps whose new first moment equals exactly β1 · m_prev,
i.e. the gradient was read as 0. A barrier variant gave 1 state.

**Cause (TTGIR of `_adamw_kernel`, Triton 3.3).**
* **G (`main_grad`):** loaded in the 8-elements-per-thread layout of the BF16 math (`#blocked1`), but
  the fused `main_grad = 0` store uses a 4-per-thread layout (`#blocked`). Another thread could zero
  `G[i]` before the thread loading `G[i]` had read it.
* **FP32 MASTER weights:** the same mismatch in the master-weights mode. It was not observed in
  40 runs, but the hazard is present in the IR.
* **M, V, P:** load and store in one layout, so they are safe.

**Fix.** `tl.debug_barrier()` (one CTA barrier) between the loads and the stores; each program
touches only its own elements.

**After the fix:**
* 1 state in every case over 40 runs, for both modes and three sizes;
* 0 gradients read as zero;
* weight-step time 19.4 / 20.5 ms before vs 21.2 / 20.3 ms after (800M elements, alternating
  measurements): unchanged within noise.

**Regression test** `test_optim.py::test_in_place_update_is_race_free`. It checks the AdamW
recurrence for every element and that 12 runs repeat bitwise. Against the old kernel it fails
(probabilistically: 1 of 4 configurations in one run, 25–31 elements); with the fix it passes.

**Impact on the previous phase.** The production nonlinear tests' null control (21 / 36 failures)
and their "rare, large jumps" were measured with this optimizer. The race is one plausible
contributor besides the FP32 atomics; §3.5 re-measures with the fix.

### 3.4 What the control does and does not show

It shows that the order-preserving layouts change no arithmetic in a complete nonlinear training
loop when every reduction is deterministic. It does not show production R > 1 trajectories
identical: those include FP32 forward and dX atomics.

### 3.5 Production R > 1 characterization (`training_validation_production*`)

This is `numerics_training.py`, run with the race-fixed optimizer.
* Graphs: C8_S16_R4, C8_S64_R16, `C8_S64_R16_planB_vs_planA` and `C8_S16_R4_fused_vs_separate`.
* Models: mse and compose. Modes: det and SR.
* Checks are predeclared: x2 for the same organization and x10 for whole-plan or fused comparisons.
  Each check runs against an envelope built from repeated baseline runs. A `baseline_null` control
  (another baseline run) goes through the same check.
* Every run was finite.

| Graph | Configs | `baseline_null` FAIL | Candidate FAIL |
|---|---:|---:|---|
| C8_S16_R4 (x2) | 4 | 3 | `vectorized_transpose_hoisted` 3 |
| C8_S64_R16 (x2) | 4 | 1 | `vectorized_transpose_hoisted` 0 |
| C8_S64_R16 Plan B vs Plan A (x10) | 4 | 1 | `planB_private_t32` 1 |
| C8_S16_R4 fused vs separate (x10) | 4 | 2 | `fused` 2, `fused_recompute` 2 |

**How to read the failures.** Almost every failure, null or candidate, falls in steps 0–3. At those
steps the envelope for some quantity is exactly 0, because the baseline runs happened to agree
bitwise. The limit then shrinks to `ABS_FLOOR` = 1e-12, so any FP32-atomic difference gives an
enormous ratio:
* 9.79e-6 on `b` at step 2 gives 9.8e6 of limit. The null control, `fused` and `fused_recompute`
  all hit this same value in mse/det.
* 1.86e-11 on `b` at step 0 gives 18.6.

The null control fails at the same rate as the candidates: 7 of 16 configs overall. So the check
cannot tell production R > 1 candidates apart from run-to-run atomic noise; this stays
characterization (§3.4), not a pass/fail gate. With the optimizer race fixed, null-control failures still remain. The race was
therefore not their only cause: zero-envelope steps under FP32 atomics are enough to produce them. The final-step divergences overlap the null control:
* Plan B vs Plan A, mse/det: W 1.90e-6 / dX 2.57e-5, against null 1.78e-6 / 2.90e-5.
* Fused, mse/det: W 3.18e-6 / dX 4.12e-5, against null 3.14e-6 / 3.98e-5.
* Compose: every candidate is within 2% of the null on W and dX.

The one asymmetry is fused mse/SR: 16 failing (step, quantity) entries, against 2 for the null.
* They come from early steps where the envelope is zero or tiny: `b` at step 2, then dW / db at
  step 3.
* The final step is close to the null: W 6.66e-6 against 6.05e-6, dX 4.38e-5 against 3.75e-5.
* This fits fused dX using a different FP32 summation order (one partial per 16-unit CTA, §5).
  That order was already known not to be bitwise; this is not evidence of a defect. fused dW is
  bitwise at fixed upstream (`test_fused_backward.py`).

Since fused is not a default, this does not affect any decision. The deterministic control (§3.2)
remains the evidence for order-preserving equality.

## 4. Track D: the S = 64 dX crossover

### 4.1 Plans and resources (static: `resources_plan*_S64.json`)

| Plan | Tile / warps / warps per copy | Scatter | Registers | Shared memory | Computed CTAs / SM (limiter) | CTAs at H = 100M, M = 8192 |
|---|---|---|---:|---:|---|---:|
| A `shared_t64` (previous auto) | 64 / 8 / 8 | `swizzle4_capped` (64-bit CAS into one slab) | 80 | 26 KiB | 3 (registers) | 98,304 |
| B `private_t32` | 32 / 8 / 1 | `vectorized_transpose_hoisted` (conflict-free float4 records) | 78 | 70 KiB | 1 (shared memory) | 196,608 |
| original scatter (context) | 64 / 8 / 8 | `baseline` | 79 | 26 KiB | 3 | 98,304 |

**API** (`PhysicalPlan.cuda_grouped_dx_plan = auto | tiles | shared_t64 | private_t32`; config
`DX_PLANS`):
* A named plan sets tile, copies and scatter together. One that conflicts with an explicit field
  raises.
* `auto` applies the scale rule only when tokens, warps, reduction, chunk units and scatter are all
  automatic; any explicit dX field makes it behave as `tiles`.
* Every benchmark run asserts the plan, T, NW, WG, scatter and kernel template that actually ran.
* `last_info["grouped_dx"]` records `plan`, `plan_reason`, `units_per_chunk_mean`, `token_tiles`
  and `tail_utilization`.

### 4.2 Screening (one seed; `crossover_screen/`, `crossover_stage2/`, `crossover_bf16/`)

Complete step, 4 alternating rounds × 2 restored repetitions × 3 steps; B − A step difference, with
units per bucket = mean work units per nonempty (output, supergroup) bucket (computed from the
metadata, cached at construction).

| H | G | R (Q) | M | Units / bucket | Units / dX chunk | Plan A step [round range] | Plan B step | B vs A |
|---:|---:|---|---:|---:|---:|---|---|---:|
| 1M | 1024 | 16 (256) | 8192 | 0.12 | 124 | 39.1 | 45.4 | +16.0 % |
| 10M | 1024 | 16 | 8192 | 1.19 | 1224 | 365.6 [363–369] | 396.0 [393–397] | +8.3 % |
| 30M | 1024 | 32 (512) | 8192 | 1.79 | 1832 | 1053.7 | 1078.3 | +2.3 % |
| 20M (rerun) | 1024 | 16 | 8192 | 2.39 | 2444 | | | +0.65 % |
| 50M | 1024 | 32 (512) | 8192 | 2.98 | 3052 | 1667.9 | 1678.3 | +0.6 % |
| 30M | 1024 | 16 | 8192 | 3.58 | 3664 | 1000.5 [994–1007] | 989.4 [988–993] | −1.1 % |
| 30M | 1024 | 16 | 1024 / 4096 | 3.58 | 3664 | 132.2 / 508.5 | 130.8 / 503.1 | −1.0 / −1.1 % |
| 10M | **256** | 16 | 8192 | **4.77** | **1221** | 335.8 [334–338] | 331.3 [330–333] | **−1.3 %** |
| 40M (rerun) | 1024 | 16 | 8192 | 4.77 | 4884 | | | −1.7 % |
| 20M (held-out) | 1024 | 8 (128) | 8192 | 4.77 | | 666.8 | 654.4 | −1.9 % |
| 50M | 1024 | 16 | 8192 | 5.96 | 6104 | 1645.5 | 1609.6 | −2.2 % |
| 30M | 1024 | 8 (128) | 8192 | 7.15 | 3664 | 982.7 | 958.2 | −2.5 % |
| 70M (rerun) | 1024 | 16 | 8192 | 8.34 | 4274 | | | −2.9 % |
| 100M | 1024 | 16 | 8192 | 11.92 | 4069 | 3292.9 [3254–3299] | 3186.3 [3178–3189] | −3.2 % |
| 100M | 1024 | 16 | 1024 / 4096 | 11.92 | 4069 | 431.5 / 1643.6 | 419.4 / 1586.4 | −2.8 / −3.5 % |
| 50M | 1024 | 8 (128) | 8192 | 11.92 | 4069 | 1617.9 | 1564.7 | −3.3 % |
| 30M | **256** | 16 | 8192 | 14.3 | 3663 | 967.5 | 933.3 | −3.5 % |

**Which descriptor predicts the crossover:**
* **Units per bucket:** B vs A is monotone in it, across H, G, R and M.
* **Units per chunk:** a sawtooth in H (one, then two, then three chunks per supergroup). It does
  not order the results: compare G = 256, 10M (1221 units / chunk, −1.3 %) with G = 1024, 10M
  (1224, +8.3 %).
* **M:** does not move the sign at a given bucket length.
* **The original scatter** is slower than both plans everywhere it was measured (stage 1, 1M–100M):
  e.g. 3544.1 ms at 100M against A 3292.9 and B 3186.3.

**BF16** (`crossover_bf16/`; the previous BF16 default is the per-field tiles with the original
scatter, since the scatter rule had no BF16 S = 64 entry):

| H | Units / bucket | Previous BF16 default | Plan A | Plan B | B vs A |
|---:|---:|---:|---:|---:|---:|
| 1M | 0.12 | 39.33 | 37.10 (−5.7 %) | 42.48 (+8.0 %) | +14.5 % |
| 10M | 1.19 | 375.5 | 352.5 (−6.1 %) | 356.8 (−5.0 %) | +1.2 % |
| 20M | 2.39 | 710.7 | 659.1 (−7.3 %) | 638.7 (−10.1 %) | −3.1 % |
| 30M | 3.58 | 1047.5 | 973.5 (−7.1 %) | 925.0 (−11.7 %) | −5.0 % |
| 100M | 11.92 | 3407.8 | 3159.5 (−7.3 %) | 2952.0 (−13.4 %) | −6.6 % |

### 4.3 Interpretation of the crossover

**What the measurements show:**
* **Per-unit side:** Plan B's per-block-tile advantage (conflict-free private records instead of
  CAS into a shared slab) scales with total work.
* **Per-item side:** its disadvantages scale with the number of (item, token-tile) instances:
  * B has twice the token tiles (196,608 CTAs);
  * per item, each warp issues dependent loads (items[], counts[], the coefficient row);
  * B runs at a computed 1 CTA (8 warps) / SM against A's 3 (24 warps), so it has less latency
    hiding for those per-item events.
* **G = 256 separates per-item from per-CTA costs:** at H = 10M it has the same chunks and CTAs as
  G = 1024, i.e. the same per-CTA fixed cost (zeroing and reducing eight private copies, staging,
  flush), but a quarter of the items. B loses at G = 1024 and wins at G = 256, so per-CTA fixed
  costs do not explain the crossover and per-item costs are consistent with it.

That latency hiding is the limiting factor is an **interpretation** (no stall counters); the
computed residency is not achieved occupancy.

### 4.4 The rule (commit `8387a1c`; `cuda_ops.SG_AUTO_DX_PLAN`)

**Scope:** C = 8, S = 64, Mean(ReLU), device capability (8, 6), M ≥ 1024, every grouped dX field
automatic.

| Precision | Units / bucket ≥ threshold | Below |
|---|---|---|
| split | `private_t32` (threshold 4.0) | the per-field defaults (`shared_t64`, unchanged) |
| BF16 | `private_t32` (threshold 2.0) | `shared_t64` (replaces the original scatter) |

* **Thresholds:** split crosses at about 3; 3.6 is ambiguous (−1.1 % with overlapping round
  ranges) and stays on Plan A. BF16 crosses at about 1.5; 2.0 keeps 1.2 on A.
* **Outside the scope:** M < 1024, other devices and other configurations keep `tiles` (the previous
  behaviour), with the reason in `last_info`.
* **Descriptor cost:** `SgExec.mean_bucket_units()` is a host value computed once at construction.
  The rule adds no host synchronization, no device-to-host reads and no online benchmarking per
  step.

Tests: `test_dx_plan.py` (19 tests) covers:
* named plans and conflicts;
* precedence of every explicit field;
* the device / precision / activation / S scoping and the M floor;
* the shipped thresholds;
* the cached descriptor;
* numerical agreement of the plans (dW / db bitwise, dX within 1e-6).

### 4.5 Held-out evaluation (`holdout/`, `holdout_summary.txt`; commit `83db1cc`, clean)

None of these shapes was used to choose the thresholds. One seed each, 4 rounds.

| Shape | Units / bucket | Auto ran | Plan A | Plan B | Auto | Auto vs better plan | Auto vs A (vs previous BF16 default) |
|---|---:|---|---:|---:|---:|---:|---:|
| split H = 12M | 1.43 | tiles (= A) | 429.1 | 442.2 | 429.8 | +0.17 % | +0.17 % |
| split H = 33M | 3.94 | tiles (= A) | 1101.8 | 1102.2 | 1105.6 | +0.34 % | +0.34 % |
| split H = 20M, R = 8 | 4.77 | B | 666.8 | 654.4 | 651.3 | −0.48 % | −2.33 % |
| split H = 45M | 5.37 | B | 1477.9 | 1441.8 | 1440.5 | −0.09 % | −2.53 % |
| split H = 25M, G = 512 | 5.96 | B | 820.4 | 805.1 | 804.7 | −0.04 % | −1.91 % |
| split H = 100M, R = 32 | 5.96 | B | 3269.3 | 3193.5 | 3193.1 | −0.01 % | −2.33 % |
| split H = 60M, M = 2048 | 7.16 | B | 509.3 | 498.2 | 497.5 | −0.15 % | −2.32 % |
| split H = 77,777,777 (ragged) | 9.27 | B | 2553.1 | 2473.5 | 2471.5 | −0.08 % | −3.19 % |
| split H = 100M, seed 1 | 11.92 | B | 3236.0 | 3129.0 | 3135.5 | +0.21 % | −3.10 % |
| BF16 H = 5M, M = 1024 | 1.00 | A | 22.0 | 24.3 | 21.9 | −0.31 % | −0.31 % (−7.17 %) |
| BF16 H = 15M | 1.79 | A | 506.5 | **499.3** | 507.3 | **+1.60 %** | +0.15 % (−6.06 %) |
| BF16 H = 30M, M = 1024 | 3.58 | B | 130.2 | 125.0 | 124.9 | −0.07 % | −4.07 % (−11.76 %) |
| BF16 H = 50M | 5.96 | B | 1581.0 | 1491.0 | 1491.3 | +0.02 % | −5.68 % (−12.63 %) |

* **Misses:** one. BF16 at 1.79 units / bucket, just below the conservative BF16 threshold of 2.0,
  where B is better by 1.4 % (disjoint round ranges). The threshold is not retuned on held-out data.
* **Regressions:** none against the previous automatic plan anywhere; split shapes below the
  threshold run the identical plan.

### 4.6 Final same-session attribution at H = 100M (`final_100M/`; commit `518b16a`, clean)

**Session.** Clean commit `518b16a`, library `5b9c82fc0e35fb38`; 4 alternating rounds × 2 restored
repetitions × 3 steps (24 steps per variant); 8 verified restores per variant. Medians are given,
with the range of the 4 per-round medians in brackets.

| C / S / R, precision, seed | Variant | dX plan that ran | dX kernel | Backward kernels | Step [round range] | vs Plan A |
|---|---|---|---:|---:|---|---:|
| 8 / 64 / 16, split, seed 0 | Plan A (previous auto) | shared_t64 | 1737.7 | 2881.6 | 3230.2 [3219–3249] | |
| | Plan B (explicit) | private_t32 | 1636.2 | 2783.1 | 3129.6 [3111–3135] | −100.7 (−3.12 %) |
| | **new auto** | **private_t32** (scale rule, 11.92 units / bucket) | 1638.8 | 2781.1 | **3128.7** [3105–3132] | **−101.6 (−3.14 %)** |
| | original scatter (context) | tiles, `baseline` | 1976.7 | 3123.3 | 3469.2 [3450–3475] | +239.0 |
| seed 1 | A / B / **auto** | | 1741.5 / 1637.9 / 1637.9 | | 3234.4 / 3132.1 / **3131.9** | **−102.5 (−3.17 %)** |
| seed 2 | A / B / **auto** | | 1738.8 / 1635.3 / 1639.0 | | 3234.8 / 3134.1 / **3134.3** | **−100.5 (−3.11 %)** |
| 8 / 64 / 16, BF16, seed 0 | previous BF16 default (tiles, original scatter) | tiles | 1961.6 | 3023.6 | 3371.8 [3304–3378] | |
| | Plan A | shared_t64 | 1714.1 | 2773.6 | 3120.6 [3060–3127] | −251.2 vs previous |
| | Plan B | private_t32 | 1506.9 | 2569.3 | 2917.1 [2867–2924] | |
| | **new auto** | **private_t32** | 1498.6 | 2550.1 | **2893.9** [2865–2923] | **−477.9 (−14.2 %) vs previous; −7.3 % vs A** |

**Unchanged phases.** Forward (322–326 ms) and the optimizer (21.4–22.6 ms) are unchanged across
variants; dW is unchanged (backward kernels − dX ≈ 1144 ms for every plan).

**Memory** (production-style warm-up peaks, measured before any snapshot exists):
* 8.84 – 8.94 GiB allocated and 10.38 GiB reserved for every variant;
* 20.2 MiB of work lists;
* the 0.10 GiB allocated difference follows variant order (the first variant measured, Plan A,
  reads 8.84; every later one reads 8.94, including the original scatter, whose resources equal
  Plan A's). It is a harness-order effect, not a plan property;
* benchmark-only pinned-host snapshot: 8.38 GiB.

**Regression check at smaller H / M / R / G:** §4.5. Below the thresholds auto runs the previous
plan (≤ +0.34 %, within noise); no shape is slower than the previous automatic plan beyond noise.

## 5. Track F: sharing the projection between dX and dW

### 5.1 Counting the saving (computed, `fusion_model.json`)

The separate backward evaluates z = XW + b twice per (neuron, token): once in dX and once in dW.
Sharing reduces two evaluations to one. **That is one projection overall, not one per kernel.** At
H = 100M, M = 8192, K = 8 it is 2·H·M·K = 13.1 TFLOP of useful work, about 190 ms at the report's
69 TFLOPS. That figure is an arithmetic illustration, not a predicted saving.

The issued instructions differ, because C = 8 pads dW's m16 tile. Per 8-neuron × 16-token
block-tile (6.4 G block-tiles at H = 100M, S = 16):

| Ledger | Separate (dX + dW) | Shared (fused) | Removed |
|---|---|---|---|
| Projection MMAs | dX 1 + dW 2 HMMA.1688 (rows 8–15 padding) | 1 HMMA.1688 (dX layout) | 2 HMMA.1688: 12.8 G instructions, 26.2 TFLOP issued |
| Gradient MMAs | dX 2 HMMA.1688 + dW 1 HMMA.16816 | the same | 0 |
| dZ / packing | dX: 4 FSET + 2 PRMT + 4 LOP; dW: 4 compare / select + split | dX masks once + 2 MOVM (mask transposes) + mask expansion (4 SHF) | dW compares |
| Total issued MMA work | 91.8 TFLOP | 65.5 TFLOP | 26.2 TFLOP (at 69 TFLOPS ≈ 0.38 s of tensor issue, illustrative) |

### 5.2 Are the two projection paths interchangeable? (`proj_mask_diag.json`, `diag_proj_c8.cu`)

**How the diagnostic works.** For 20,000 random (unit, 16-token tile) items of an S = 16 graph
(D = 256, G = 16, H = 50,013 with ragged outputs, M = 136), one warp recomputes every block with the
exact instruction sequence of each kernel:
* **dX:** z = mma(A = X tile from `ldmatrix.trans`, B = packed weight fragment, C = bias pair of the
  lane's two neuron columns);
* **dW:** z^T = mma(A = {weight fragment, 0}, B = the same `ldmatrix.trans` registers, C = the
  lane's row bias from `biasT`).

It records z bits and the decisions of three epilogues, plain and MASKED (ragged): the dX mask
epilogue (`tile_mask`), the dW fast epilogue (`FastDz`: z > 0 ? one : 0) and the dW mask epilogue
(`gt0_mask`).

**Operand layouts.** The A-operand registers of the dX projection and the B-operand registers of
the dW projection are the same `ldmatrix.trans` result (`sg_gather<1>` = `dw_block`'s `xa`). Both
products pair the same bf16 values per k, in the same k order. The bias enters as the FP32 C operand
in both.

| Regime (9,673,472 elements each) | z bits equal | Subnormal / inf / NaN z | Decision differences dX mask vs dW fast / dW mask (plain, masked) | dX decision vs FP64 sign |
|---|---|---|---|---|
| random | all | 0 / 0 / 0 | 0 / 0 | 0 |
| signed zeros (x, w, b = ±0) | all | — | 0 / 0 | 0 |
| zero input, bias ±0 / ±2^-130 (BF16 subnormal) | all | 3.86M / 0 / 0 | 0 / 0 | 1.95M (subnormal z flushed: FTZ) |
| subnormal FP32 products (x, w ≈ 2^-70), bias 0 / tiny | all | 9.62M / 0 / 0 | 0 / 0 | 4.82M |
| huge (products overflow) | all | 0 / 7.55M / 0 | 0 / 0 | 3 |
| NaN / ±inf inputs | all (NaN bits too) | 0 / 0.92M / 1.17M | 0 / 0 | 0 |
| integer operands (exact sums), bias 0 | all | — | 0 / 0 | 0 |
| near-zero cancellation (bias = −round(x·w)) | all | — | 0 / 0 | 0 |

* **Result:** z is bitwise identical in all regimes, and all production decisions agree. All
  production compares are FTZ:
  * `FSET.BF.GT.FTZ` in the dX mask;
  * `FSETP.GT.FTZ` in the dW fast and generic bodies.
* **Tensor-core observations (not decision differences):** the tensor core never produced −0, and
  it flushes subnormal products / sums.
* **Hazard (first diagnostic build):** a `set.gt.ftz` mask that is only tested for non-zero compiled
  to `FSETP.GT` **without** `.FTZ`. That build reported 1.9M / 4.8M "differences", all positive
  subnormal z. The production kernels AND the mask into the operands and keep FTZ; the corrected
  diagnostic stores raw words (`proj_mask_diag_first_build_note.txt`).

A fused kernel must check its SASS for this; the prototype's compares are all `FSETP.GT.FTZ`
(`sass/fused_instruction_counts.txt`).

### 5.3 Resource / traffic model (computed; `fusion_model.json`, S = 16, R = 4, H = 100M, M = 8192)

Metadata: 3,125,248 valid units, 12,500,224 valid blocks, 768 dX chunks of about 4096 units, and
195,328 dW CTAs of 16 units. "Logical bytes" are requests, not DRAM traffic.

| Schedule | CTAs | Global dX atomics (upper) | dW / db RMW | Parameter requests | Input staging requests | Local dX | Registers | Shared memory | Scratch |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| separate (selected dX + dW cfg 2) | 98,304 + 195,328 | 0.10 G | 7.2 GB (once) | 258 GB (128 token tiles) | 51.4 GB | private float4 | 86 / 127 (static) | 36 / 6 KiB | 0 |
| **F1** dW-owned, 16 units / CTA (8 × 2), windows 16–64 | 195,328 | **25.6 G** | 7.2 GB (exclusive, once) | 2.0 GB | 51.2 GB | private window copies | 119–165 (static, §5.5) | 14–38 KiB | 0 |
| F1, 32 units / CTA (8 × 4) | 97,792 | 12.8 G | 7.2 GB | 2.0 GB | 25.6 GB | | 229 + dX state (≥ 255, spills) | | 0 |
| F1, 64 units / CTA (16 × 4) | 48,896 | 6.4 G | 7.2 GB | 2.0 GB | 12.8 GB | | ≥ 229 | | 0 |
| **F2** coarse 4096-unit chunks, serial windows 1024 / 512 / 256 | 768 (9.4 waves at 1 / SM) | 0.10 G | **57.6 / 115 / 230 GB** | 16 / 32 / 64 GB | 0.2 GB | one shared slab (CAS) | ≥ 24 dW + dX state | 98 / 50 / 26 KiB | 0 (in place, order changes) |
| **F3** one-bit mask exchange, 64 MiB chunks | 1,526 launch pairs | — | 7.2 GB | — | — | | | | 64 MiB |

* **F3:** 65,536 neurons × 8192 tokens per chunk; 102.4 GB of masks written and read again
  (**204.8 GB**) to remove dX's 1 HMMA.1688 per block-tile (6.4 G, about 13 TFLOP, illustratively
  0.19 s). Rejected: the traffic alone takes 0.27 s at the measured 754 GB/s.
* **F2:** at least 58 GB of dW partial RMW (about 76 ms at the measured pass rate), a single shared
  CAS slab for dX, 9.4 waves of 768 CTAs, and a changed dW order. Not implemented: F1 already
  measures what sharing can save (§5.6) and F2 adds these costs to the same saving.
* **F1:** chosen for the prototype; its flush cost needed measuring.

### 5.4 Screening measurements (measured; `screen_fusion_S16.json`, `screen_fusion.cu`)

The isolated flush kernel uses the actual feature maps and the candidate grid orders, and its
results were verified exact (contribution 1.0; every element equals its CTA count).

| Pattern | CTAs | Atomics | Time |
|---|---:|---:|---:|
| current dX (768 chunks, T = 64) | 98,304 | 0.10 G | 0.5 ms |
| F1, 16 units / CTA, windows of 64 | 195,328 | 25.6 G | 121.0 ms |
| F1, windows of 16 | 195,328 | 25.6 G | 150.2 ms |
| F1, hot working set (all CTAs on one supergroup) | 195,328 | 25.6 G | 121.2 ms |
| F1, 32 / 64 units per CTA | 97,792 / 48,896 | 12.8 / 6.4 G | 60.9 / 30.0 ms |
| F2 serial windows 512 / 1024 | 768 | 0.10 G | 0.6 / 1.1 ms |
| dW / db FP32 RMW pass (900M floats, 7.2 GB) | | | 9.55 ms (754 GB/s) |

The flush cost was far below what the 102 GB payload would suggest, and below the projection saving.
Together with the 258 → 2 GB parameter-request reduction, this justified the F1 prototype.

### 5.5 The F1 prototype (`sparse_pool_sg_bwdf.cuh`, `sg_bwdf_c8.cu`; commit `0df8918`)

**Design.**
* **Ownership and pipeline:** exactly `sg_dw`'s (CTA = 16-unit chunk of one supergroup, all tokens,
  cp.async ring).
* **Projection:** once per block-tile in the dX layout (`tile_mask`, same MASKED rule).
* **dX:** the mask epilogue of `sg_dx_tile` (2 HMMA.1688) into warp-private window slabs, which are
  reduced in copy order and flushed after each ring stage (32 or 64 tokens).
* **dW:** the 16-bit mask pairs transposed into the dW A layout with one `movmatrix` per 8-token
  half, then `dw_block`'s mask epilogue (bitwise equal to the fast epilogue): same A operand, same
  db sums, same HMMA.16816, same token order and ownership. **dW / db are bitwise equal** to the
  separate grouped dW.
* **Versions:** v1 is tile-outer; v2 is block-outer / tile-inner. v2 moves block decode, the W^T
  transpose and record addresses to once per window and uses the selected dX scatter (transposed
  float4 records). Every accumulator still sees its tiles in token order.
* **Control:** `SHARE = false` is the recompute-preserving control: the same schedule with `dw_block`'s
  own projection.
* **Non-finite coefficients:** the fused kernel exits on the device flag and two guarded launches
  produce the gradients: `sg_dx_kernel` generic and `sg_dw` generic, both with `BWD_FALLBACK`, which
  run only when the flag is set. Exactly one path contributes; no host synchronization.
* **Scope:** C = 8, Mean(ReLU), split, mask / fast epilogues, feature-major. Anything else raises
  for an explicit request. `auto` = `separate`.

**SASS** (static; `sass/fused_instruction_counts.txt`; unrolled bodies of 8 block-tiles, both
masked and unmasked paths compiled):

| Kernel | HMMA.1688 | HMMA.16816 | MOVM | Non-FTZ compares | Spill LDL / STL |
|---|---:|---:|---:|---:|---:|
| v2 `<16, 1, 2, 1, true>` (fused, cfg 7) | 32 = 8 × (2 projection paths + 2 gradient) | 8 | 20 = 16 mask + 4 W^T | 0 | 0 |
| v2 `<16, 1, 2, 1, false>` (control) | 48 = 8 × (2 + 2 + **2 dW projection**) | 8 | 4 | 0 | 0 |
| v2 `<8, 2, 2, 2, true>` (cfg 6) | 64 | 16 | 40 | 0 | 65 |
| v1 `<8, 2, 2, 2, 2, true>` (cfg 3) | 32 | 8 | 24 | 0 | 28 |

The control has exactly the two extra HMMA.1688 per block-tile: the removed duplicate projection.

**Resources** (static, `sass/fused_resources.txt`):

| Config | Registers | Stack |
|---|---:|---:|
| v1 cfg 0 (8 × 2, window 64) | 165 | 0 |
| v1 cfg 1 / 3 (capped at 128) | 128 | 56–64 |
| v1 with 4 units / warp | 255 | 72–368 |
| v2 cfg 6 (8 × 2, capped) | 128 | 64 |
| v2 cfg 8 (8 × 2) | 159 | 0 |
| **v2 cfg 7 (16 × 1)** | **119** | **0** |
| v2 cfg 7 control | 128 | 0 |

**Correctness** (`test_fused_backward.py`, 44 passed, 1 skipped: no Mean(identity) layer type):
* six shapes: ragged outputs, tiny / empty buckets, M = 16 below one window, a token tail 8 past a
  window, heavy overlap R = 16, R = 1;
* five configurations, fused and control;
* **results:**
  * dW / db bitwise equal to the separate path;
  * dX within 1e-6 of the separate path and within the strict reference tolerances (2e-5);
  * the extremes (huge, tiny, subnormal and signed-zero upstream gradients, zero bias / input,
    inf weights) agree, and with inf / NaN coefficients the guarded fallbacks give bitwise dW / db
    and bitwise dX at R = 1;
  * main-gradient accumulation over 3 backward calls is bitwise equal to the separate path;
  * explicit unsupported requests raise.

### 5.6 Measurements (complete steps, same session per row)

| Session | Separate: bwd kernels / step | Best fused: bwd kernels / step | Fused cfg 7 vs its recompute control |
|---|---|---|---|
| H = 10M, S = 16 (`fused_screen/H10M_S16*`) | 248.8 / 289.5 | v2 cfg 6: 242.8 (−2.4 %); v1 cfg 3: 246.8 | 249.4 vs 283.5 (sharing −34 ms, −12 %) |
| H = 100M, S = 16, 2 rounds (`fused_screen/H100M_S16*`) | 2155.1 / 2492.2; 2163.0 / 2502.7 | v1 cfg 3: 2147.0 / 2486.1 (−0.4 %); v2 cfg 6: 2217.5 (+2.5 %) | 2417.1 vs 2755.9 (sharing −339 ms, −12 %) |

**Attribution** at H = 10M (ablation, v1 cfg 3; `fused_screen/H10M_S16_ablate*`):

| Build | Fused kernel |
|---|---:|
| full | 243 ms |
| no global flush atomics | 240 ms |
| no dX side | 160 ms |
| no dW side | 177 ms |

**What this means.** The global atomics, the feared cost, are about 3 ms. The dX side (≈ 83 ms) and
the dW side (≈ 66 ms) remain as large as in the separate kernels. In the dW-owned schedule they lose:
* dX's 4-tile amortization of block decode and coefficient rows;
* the per-window reduction of the warp copies: 32–64 shared operations per block-tile for 16 units
  per CTA;
* single coefficient loads: the dX (row) and dW (column) coefficients are both loaded.

Larger CTAs would cut the reduction and flush cost but need 229–255 registers (spills).

**Final same-session fusion attribution** (`final_100M/C8_S16_R4_fusion`, `C8_S32_R8_fusion`; H = 100M,
M = 8192, split, seed 0; clean `518b16a`):

| C / S / R | Variant | Backward kernels (all launches incl. fallbacks) | Step [round range] | vs separate |
|---|---|---:|---|---:|
| 8 / 16 / 4 | separate (selected dX + dW) | 2107.0 | 2441.1 [2440–2442] | |
| | fused v1 cfg 3 | 2095.9 | 2427.8 [2426–2430] | −13.3 (−0.55 %) |
| | fused v2 cfg 6 | 2156.2 | 2487.5 [2481–2491] | +46.3 (+1.9 %) |
| | fused v2 cfg 7 | 2352.3 | 2685.2 [2684–2687] | +244.1 (+10.0 %) |
| | recompute control v2 cfg 7 | 2673.4 | 3007.3 [3006–3008] | sharing within the schedule: −321.1 |
| 8 / 32 / 8 | separate | 2345.1 | 2682.1 [2668–2686] | |
| | fused v1 cfg 3 | 2485.6 | 2819.8 [2805–2824] | +137.7 (+5.1 %) |
| | fused v2 cfg 6 | 2402.7 | 2737.7 [2720–2741] | +55.6 (+2.1 %) |
| | fused v2 cfg 7 | 2738.8 | 3073.5 [3062–3089] | +391.4 (+14.6 %) |
| | recompute control v2 cfg 7 | 3054.6 | 3391.6 [3379–3406] | sharing within the schedule: −315.8 |

**What the matrix shows.**
* **The only improvement is below the ±1 % run-to-run variation of this harness:** S = 16 v1 cfg 3
  (−0.55 %, one seed; the earlier separate 100M session gave −0.4 %).
* **S = 32:** every fused configuration is slower.
* **Memory:** the fused path allocates 8.93 GiB vs 8.84 GiB at the same reserved 10.38 GiB. Its
  work lists are 17.2 MiB recorded; the fallback launch also caches the dX item chunks of the
  separate path.

### 5.7 Decision

**The fused backward is rejected as a default and kept as an explicit experimental option**
(`cuda_grouped_backward = fused | fused_recompute`, `cuda_grouped_fused_cfg`).
* It shares the projection correctly (bitwise dW, verified SASS), but after all overhead it does not
  beat the optimized separate backward by more than the noise at the primary scale.
* F2 and F3 were rejected by the model (§5.3).
* Neither the prior report's "at most about 0.19 s each" nor the arithmetic illustration translated
  into a measured complete-step gain.

## 6. Numerical and sanitizer status

**Compute Sanitizer** (`run_sanitizer_sdfb.sh`, `sanitize_sdfb_workload.py`, log `sanitizer_sdfb.log`)
ran 22 workload cases under each tool. They cover:
* fused v1 cfg 0 and 3, and v2 cfg 6 and 7, both shared and recompute, at S = 16 and 32, including
  non-finite coefficients and the fallback path;
* S = 64 Plan A and Plan B, split and BF16, including non-finite inputs and auto at H = 300K;
* the diagnostic projection-mask kernel and the flush / RMW screens.

| Tool | Result |
|---|---|
| memcheck | 0 errors |
| racecheck (`--racecheck-report all`) | 0 hazards (0 errors, 0 warnings) |
| initcheck | 0 errors |
| synccheck | 0 errors |

All 22 cases completed ("ok") under every tool, and none were skipped.

* Full test suite after the dispatch / fused / optimizer changes: **2618 passed, 31 skipped**
  (previous phase 2551 / 30; the new skip is the Mean(identity) fused case).
* Fixed-upstream equivalence of the plans: dW / db bitwise, dX within 1e-6 (`test_dx_plan.py`). In
  the deterministic control, Plan B is bitwise equal to the original layout with the same private
  tile (§3.2).
* **Plan A vs Plan B is an order change** (shared CAS vs private copies). In the deterministic
  control its differences are the size of Plan A's own run-to-run differences. The production
  characterization is in §3.5.

## 7. Defaults, API, fallbacks

* **`cuda_grouped_dx_plan`:**
  * `auto` (default): the rule of §4.4 within its scope, else `tiles`;
  * `tiles`: the previous per-field behaviour, including `SG_AUTO_DX_SCATTER`;
  * `shared_t64` / `private_t32`: explicit whole plans.
* **Explicit tile, warps, reduction, chunk units or scatter** override `auto` (reported in
  `plan_reason`). A named plan conflicting with an explicit field raises.
* **`cuda_grouped_backward`:** `auto` = `separate` (default); `fused` and `fused_recompute` are
  experimental and explicit, and raise when unsupported. `last_info["backward_organization"]` and
  `["grouped_fused"]` record cfg, sharing, warps, units per warp, window, CTAs, the flush upper
  bound, the kernel and the fallback.
* **Unchanged:**
  * C = 64, S ≠ 64, Mean(identity), M < 1024, devices other than sm_86, the generic backend;
  * connectivity, checkpoints and graph identity;
  * main-gradient accumulation and the restore harness.
* **FusedAdamW:** the race fix changes no intended semantics. It removes nondeterministic
  gradient drops.

## 8. Final tables

**Dispatch** (H = 100M, M = 8192, C = 8, S = 64, R = 16, G = D = 1024; 3 seeds split, 1 seed BF16; same
session; static resources):

| Precision | Plan | Resources (regs / smem / computed CTAs per SM) | dX kernel | Step (mean of seeds) | Variability (round ranges) | Memory alloc / res | Numerical category | Sanitizer |
|---|---|---|---:|---:|---|---|---|---|
| split | A `shared_t64` (previous auto) | 80 / 26 KiB / 3 | 1739.3 | 3233.1 | ±16 ms | 8.84–8.94 / 10.38 GiB | reference | clean (previous phase; this phase §6) |
| split | B `private_t32` | 78 / 70 KiB / 1 | 1636.5 | 3131.9 | ±12 ms | 8.94 / 10.38 | order change vs A (shared CAS → private); bitwise vs same-tile old layout | §6 |
| split | **auto (rule → B)** | = B | 1638.6 | **3131.6 (−3.14 %)** | ±13 ms | 8.94 / 10.38 | = B | §6 |
| BF16 | previous default (original scatter) | 79 / 26 KiB / 3 | 1961.6 | 3371.8 | ±37 ms | 8.84 / 10.38 | reference | |
| BF16 | **auto (rule → B)** | = B | 1498.6 | **2893.9 (−14.2 %)** | ±29 ms | 8.94 / 10.38 | order change | §6 |

**Fusion** (H = 100M, M = 8192, split, seed 0, same session per S):

| C / S / R | Plan | Backward kernels | Step | vs separate | Registers / stack | Numerical category | Sanitizer |
|---|---|---:|---:|---:|---|---|---|
| 8 / 16 / 4 | separate (selected) | 2107.0 | 2441.1 | | 86 + 127 / 0 | reference | |
| 8 / 16 / 4 | fused v1 cfg 3 (best) | 2095.9 | 2427.8 | −0.55 % (< noise) | 128 / 56 | dW / db bitwise; dX order change (≤ 1e-6 vs separate) | §6 |
| 8 / 32 / 8 | separate (selected) | 2345.1 | 2682.1 | | 78 + 127 / 0 | reference | |
| 8 / 32 / 8 | fused v2 cfg 6 (best) | 2402.7 | 2737.7 | +2.1 % | 128 / 64 | same | §6 |

**Other controls** (same final session, unchanged code paths):

| Control | Plan | Step |
|---|---|---:|
| C = 64, S = 16, R = 4 | auto (unchanged per-phase plan) | 1688.7 |
| C = 64, S = 64, R = 16 | auto | 1697.1 |
| C = 8, S = 16, generic backend (existing fallback) | generic | 4813.1, vs grouped 2420.1 |

The C = 64 controls are from a different session than the previous phase's 1751 / 1760 ms; no
speedup is claimed from that difference.

**What failed or was rejected:**
* fused v1 / v2 as defaults (no gain beyond noise; S = 32 slower);
* F2 and F3 (model);
* the BF16 threshold misses one held-out shape by 1.6 %;
* the predeclared null-control envelope cannot discriminate (§3.5).

## 9. Reproduction

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd sparse_pooling/benchmarks
$PY deterministic_training.py --out ../results/scale_dispatch_fused_backward/deterministic_training
$PY optim_race_check.py --out ../results/scale_dispatch_fused_backward/optimizer_race/optim_race_check
$PY proj_mask_diag.py --out ../results/scale_dispatch_fused_backward/proj_mask_diag
$PY fusion_model.py --out ../results/scale_dispatch_fused_backward/fusion_model
$PY screen_fusion.py --out ../results/scale_dispatch_fused_backward/screen_fusion_S16
./run_sd_crossover_screen.sh; ./run_sd_crossover_stage2.sh; ./run_sd_crossover_stage2_rerun.sh; ./run_sd_bf16_screen.sh
./run_sd_holdout.sh; ./run_sd_final.sh; ./run_sd_training.sh
bash run_sanitizer_sdfb.sh $PY    # needs compute-sanitizer on PATH; CUDA_MODULE_LOADING=LAZY is set by the script
$PY summarize_dispatch.py ../results/scale_dispatch_fused_backward/final_100M/*.json
$PY summarize_holdout.py ../results/scale_dispatch_fused_backward/holdout/*.json
$PY summarize_deterministic.py ../results/scale_dispatch_fused_backward/deterministic_training.json
cd ../tests && $PY -m pytest -q .
```
