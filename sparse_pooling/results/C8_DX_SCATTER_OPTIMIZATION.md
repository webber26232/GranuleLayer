# C = 8 grouped dX: local-accumulation ("scatter") optimization and training validation

RTX 3090 (GA102, sm_86, 82 SMs), D = G = 1024, K = 8, BF16 parameters / input, FP32 MMA
accumulation, Mean(ReLU), fused AdamW with BF16 moments, input-supergroup graphs
(balanced permutation).  Raw evidence: `results/c8_dx_scatter/`.  Every number below is a
measurement on this GPU unless it is explicitly labelled *static* (SASS / cuobjdump),
*computed* (analytical model) or *ablation* (timing-only build with wrong results).

## 1. Summary

**The corrected harness.** It now restores the complete training state (parameters, main_grad,
AdamW moments, steps, hyperparameters, the stochastic-rounding counter `_seed`, RNG) before
every measured repetition. Timings do not change measurably: in the same session the old and
new protocols give 2796.6 vs 2800.6 ms (C = 8 grouped), 4997.4 vs 5015.2 ms (generic) and
1751.9 vs 1752.2 ms (C = 64).

**Session offset.** Every configuration in this session is 3 to 4 % slower than the previous
report, because of the session, not the protocol. Both protocols agree (C = 8, S = 16:
2717 ms then, about 2800 ms now; generic: 4847 then, about 5000 now). All comparisons below
are same-session A/B.

**Root cause in the baseline** (computed bank model, confirmed by measured layout controls).

* **Layout.** The float2 layout spreads the 4 features of each 64-bit half-warp phase over
  only 2 bank groups (feature parity).
* **Cost (computed).** That is 21.2 shared wavefronts per m16 tile for the read-modify-write,
  against 8 conflict-free.
* **Fix (exact).** Changing only the layout to 4 bank groups, and then making each 128-bit
  phase touch a single feature through an 8-shuffle transpose, removes most of it while
  keeping every (feature, token) addition sequence. The order-preserving variants are
  **bitwise identical** to the baseline whenever the global dX atomics are deterministic.

**Full-scale result.** The selected variants, run explicitly from commit `a3fa200`; the `auto`
selection reproduces them (§9.3). H = 100M, M = 8192, split gradients, corrected harness,
3 graph/input/init seeds, 4 alternating rounds:

| C / S / R | dX kernel (ms) | Complete step (ms) | Step saved | Speedup |
|---|---|---|---:|---:|
| 8 / 16 / 4 | 1380.9 / 1379.5 / 1375.6 → **1085.4 / 1084.6 / 1083.4** | 2800.6 / 2799.3 / 2797.0 → **2512.5 / 2507.0 / 2507.6** | −288…−292 | **1.115–1.117×** |
| 8 / 32 / 8 | 1632.1 / 1635.6 / 1632.8 → **1292.7 / 1290.0 / 1292.0** | 3099.5 / 3096.7 / 3099.1 → **2758.8 / 2758.7 / 2760.1** | −338…−341 | **1.123×** |
| 8 / 64 / 16 (seed 0) | 1999.4 → **1753.0** (`swizzle4_capped`, shared copies) | 3504.5 → **3257.8** | −247 | **1.076×** |

**How much is removable.** The dX kernel drops by 21 % (S = 16 / 32) and 12 % (S = 64).
* **Ablation reference.** A no-local-update ablation (wrong results, smaller and differently
  scheduled code) bounds what deleting the update entirely would give.
* **Share removed.** The layout work removes **62 % (S = 16) and 64 % (S = 32)** of that
  ablation gap at H = 100M: dX 1378 → 905 ablated vs 1081 selected, and 1630 → 1101 vs 1293 ms.
  At H = 10M the shares are 60 % and 67 %.
* **Remaining dX time.** It is now dominated by the projection recompute, the dZ mask epilogue
  and the 3 MMAs per block-tile (§11).

**Other variants.**
* **Register owner (B).** The MMA-routed feature-owner variant is correct and runs, but at
  S = 16 it is slower than the layout variant: dX 1169.5 vs 1085.4 ms at 100M. At S = 32 it is
  1.5× slower than the baseline (1.9× slower than the vector layouts), as its MMA count predicts
  (§7).
* **Selection.** The S = 16 / 32 / 64 results are encoded in the measured `auto` rule (§12).
  The baseline stays available and is kept for every unmeasured case.

**Unchanged.** No graph, precision, epilogue or arithmetic of the projection, dZ or dW
changed. Forward and dW are unchanged controls. No new scratch buffer was added. Production
memory is unchanged: 8.84 GiB allocated and 10.38 GiB reserved at C = 8, H = 100M.

## 2. Baseline, environment, reproducibility

* Started from `input-supergroup-optimization` at **`aeb41a522a1884f5d858ea581653c19be40e17eb`**.
  The branch had not advanced (`origin/input-supergroup-optimization` = local).
* New branch `c8-dx-scatter-optimization` in a separate worktree (`../GranuleLayer-c8dx`). The
  existing branches and worktrees are untouched. Nothing was pushed or merged, and
  `mma-forward-epilogue-optimization` was not merged.
* Commits (in order):

  | Commit | Content |
  |---|---|
  | `cdee27a` | Harness: complete restore, per-variant memory accounting |
  | `6def1ef` | Scatter variants, selection, tests |
  | `d4fc34c` | Validation: reference exception, nonlinear and multi-step training |
  | `ad12dcd` | Hoisted addresses |
  | `a3fa200` | `swizzle4_capped`, bank model, summaries |
  | `9dfc1b3` | Measured `auto` rule |
  | `fd7a651` | Selected-plan confirmation script |
  | `d97b69e` | Fingerprint memory fix |
  | `0c61f27` | Post-acceptance evidence script |
  | `4a41593` | Explicit baseline in the run scripts |
  | (last) | Evidence and report |
* **Acceptance data** (`final_100M/`) come from the clean committed state **`a3fa200`**, library
  `kernels/cuda/build/ad01f80eb1dcab3a/` (explicit variants).
* **Later runs**, all with the same kernel library `ad01f80eb1dcab3a`: their `dirty` flags
  reflect only uncommitted `README.md` or `numerics_training.py` edits.

  | Runs | Commit |
  |---|---|
  | Selected-plan confirmation (`selected_100M/`): S = 16 | `d97b69e` |
  | Selected-plan confirmation: S = 32 / 64 | `0c61f27` |
  | `extra/` C = 64 runs | `0c61f27` |
  | `extra/…_explicit`, `memory_recheck/` | `4a41593` |
  | `ablation_100M/` (library `56c976615e841ddd`, `-DSP_ABLATE_SG=2`) | `4a41593` |
* **Superseded run.** `extra/H10M_C8_S16_R16_INVALID_same_kernel.*` used `auto` as the baseline
  after the `auto` rule existed, so it compared the selected kernel with itself. It is kept
  but superseded by `…_explicit`.
* **Screening data** (`screen/`) came from exploration states. Each JSON records the commit,
  `dirty` flag and library hash; `dirty` there means uncommitted edits to benchmark scripts
  or to kernels that were committed later.
* Environment: driver 550.67; nvcc 12.1 (V12.1.66); PyTorch 2.7.0+cu128; Triton 3.3.0.
* Flags: `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17`.
* GPU state is recorded before and after every run (`gpu_before` / `gpu_after`): SM 1965–1980 MHz
  at the start, 1665–1830 MHz at the end, 37–54 °C, power limit 370 W. Each file's values are
  in `final_100M.csv`.
* Hardware performance counters are admin-only on this machine (`RmProfilingAdminOnly=1`, no
  sudo), so there are **no Nsight Compute counters**. Bank conflicts are a *computed* model
  (§5.3), instruction mixes are *static*, and attribution uses *ablation* builds.

## 3. P0: benchmark state restoration

### 3.1 Defect
`bench_supergroups.py` restored only `weight` and `bias` before each variant group, then ran an
untimed step, and reused one `FusedAdamW` throughout. Moments, per-parameter `step` and the
stochastic-rounding counter (`FusedAdamW._seed`, which is not part of `state_dict()`) kept
advancing, so the timed variants started from different training states.

`test_train_state.py::test_weight_only_reset_is_incomplete` reproduces this. After a
weights-only reset the next update differs from the reference trajectory: `step + 1`, a
different seed, and different W.

### 3.2 Fix (`benchmarks/train_state.py`, `bench_supergroups.py`)
* **Contents.** `TrainState` captures and restores, in place into the existing tensors (no
  allocation):
  * every parameter, `main_grad` and `.grad` (or its absence);
  * every optimizer state tensor (BF16 / FP32 moments, FP32 master weights) and each
    parameter's `step`;
  * param-group hyperparameters, `_seed`, and the CPU and CUDA RNG states.
* **Snapshot location.** Snapshot tensors live in **pinned host memory** (8.38 GiB at C = 8,
  H = 100M), never on the device.
* **Timing protocol.**
  1. Warm and compile every variant, including lazy AdamW state and work lists.
  2. Capture one reference state.
  3. Per round (alternating order) and variant: one untimed step rebuilds the runtime.
  4. Before each of the 2 measured repetitions, restore and verify an **exact fingerprint**
     (integer sums of raw bits, plus scalars), all outside the CUDA-event intervals.
  5. Each repetition runs 3 steps that advance normally on the same input.
* **Verification.** 8 verified restores per variant per file (`restore_checks`).
* **Historical protocol.** `--protocol historical` keeps the old reset as a labelled
  diagnostic.
* **Kernel check.** The harness aborts if any requested phase kernel or dX scatter variant did
  not run (`last_info`).

### 3.3 Effect on timings (same session, H = 100M)

| Configuration | Historical protocol | Corrected protocol |
|---|---:|---:|
| C = 8, S = 16 / R = 4, grouped baseline | 2796.6 [2790.2–2804.8] | 2800.6 [2791.2–2808.2] |
| C = 8, S = 16 / R = 4, generic backend | 4997.4 [4998.2–5000.5] | 5015.2 [4999.1–5037.6] |
| C = 64, S = 16 / R = 4, grouped auto | 1751.9 [1748.8–1755.2] | 1752.2 [1746.9–1755.0] |

The timings are the same within the round spread: restoring moments, steps and seed does not
change the work. The difference from the previous report's 2716.9 / 4846.6 / 1703.1 ms is a
**session effect** that both protocols show equally. Historical rows are context only, and no
improvement is computed against them.

### 3.4 Memory accounting
* **Per-variant metadata.** Work-list bytes are now recorded per variant in `last_info`
  (forward / dX / dW work lists actually used: 20.2 MiB at C = 8 S = 16, 13.1 MiB at C = 64 S = 16).
  Previously they were inferred from the last runtime, and a generic last runtime reported 0.
* **Validation copy.** The temporary unpacked validation-index copy is released: by code
  inspection, `del idx_c` follows validation, and `--validate` is not used in any timed run of
  this phase.
* **Production-style peak** (pre-snapshot warm-up, C = 8, H = 100M, M = 8192): **8.84 GiB
  allocated / 10.38 GiB reserved** for the baseline and every candidate; the timed-step
  allocated peak is the same. A candidate's warm-up peak reads 8.94 GiB only because the
  previous variant's runtime is still alive while the next one is built.
* **Benchmark-only overhead.**
  * The snapshot is 8.38 GiB of pinned host memory and 0 device bytes.
  * The historical protocol kept a 1.68 GiB device copy (weights and bias), which is why its
    peak reads 10.52 GiB.
  * In the corrected-harness acceptance runs the *reserved* peak reads 16.3 GiB. My first
    fingerprint used a whole-tensor `sum(dtype=int64)`, which materialized an int64 copy
    (6.4 GB) that the caching allocator kept. Allocated peaks and timings were unaffected.
  * Fixed in `d97b69e` (chunked reduction, ≤ 32 MiB of temporaries, tested).
  * Re-check (`memory_recheck/`, H = 100M, S = 16): the timed-step peaks are **8.84 GiB
    allocated / 10.38 GiB reserved** for the baseline and the selected plan, equal to the
    production-style values. Steps are 2792.2 → 2503.2 ms, consistent with §9.
* **Persistent memory** (C = 8, H = 100M): weights 1.60 GB, bias 0.20 GB, indices 0.20 GB,
  main_grad 3.60 GB, BF16 moments 3.60 GB, graph metadata 4.2 MB, work lists 20.2 MiB.
* **Per-step scratch.** 240 MiB at M = 8192 (30 MiB at M = 1024). It is identical for all dX
  variants, because they change shared memory and registers only.

## 4. P1: numerical and training validation

### 4.1 The reference exception, narrowed
Before, `test_grouped_matches_generic` accepted any reference-tolerance exceedance that the
generic kernels shared. Running the test strictly identifies **exactly one** failing case:
split, C = 8, D = 1024, H = 300,000, G = 64, B = 256, S = 64, R = 16.

`benchmarks/relu_boundary_case.py` (`relu_boundary_case.json`) shows the cause:
* It scans 93 elements with |z| < 1e-6 in FP64.
* Exactly **one** decision differs: token 206, output 30, slot 923, with z = **+2⁻²⁸** (FP64
  and FP32 reference agree: active).
* The FP32 tensor-core accumulation of the generic, grouped and new vectorized kernels makes it
  inactive.
* That single decision moves dX / dW / db by 3.0e-5 / 8.2e-5 / 8.7e-5 relative. With it
  adopted, every kernel is within **2.3e-6 / 2.4e-6 / 1.0e-7** of the FP64 reference.

The test now keeps the strict tolerances (2e-5 split, 5e-3 BF16) everywhere. For that one
case it compares against the FP64 reference with that single documented decision adopted, and
asserts the element and its exact z. Both the generic and the grouped results must meet the
strict tolerance. No other exceedance can pass.

### 4.2 Three separate questions
1. **Fixed upstream gradient** (§9.1): same weights, inputs, indices and dY; this isolates the
   kernel arithmetic.
2. **Nonlinear-loss gradients**: dY comes from the actual atomic grouped forward output.
3. **Multi-step training**: parameter, moment and rounding feedback.

The previous four-step AdamW bitwise test uses a fixed `(y · dy)` upstream gradient. It stays a
valid fixed-upstream test, not evidence about nonlinear trajectories.

### 4.3 Nonlinear multi-step comparisons (`training_validation.py`, `numerics_training.py`)

**Models.** All use production dtypes (BF16 layer parameters and input, FP32 main_grad,
FusedAdamW):

| Model | Definition |
|---|---|
| `mse` | mean((y − target)²), nonzero target |
| `square` | mean(y²), the timing loss |
| `compose` | x0 → trainable Linear(256 → 1024, FP32) → BF16 → layer → Linear(1024 → 64) → tanh → MSE |

In `compose`, the layer's dX drives the projection's update.

**Setup.**
* Optimizer modes: deterministic rounding (BF16 moments, SR off), stochastic rounding (paired
  counters), FP32 moments with master weights.
* Graphs (H = 1M, M = 1024): C = 8 S = 16 R = 4 and S = 32 R = 8 (16 steps); S = 16 R = 16,
  S = 64 R = 16 private and shared, and C = 64 S = 16 control (8 steps).
* Every run starts from the same seeds and uses the same pre-generated, changing batches.
* Runs are kept in lockstep, so pairwise divergences of loss, y, dW, db, dX, W, b, m, v and the
  projection weights are computed at every step.
* Each variant has 4 runs, including a **null control**: an independent second set of baseline
  runs checked exactly like a candidate.

**Predeclared checks** (written into `training_validation.py` before any candidate result):
* **C1:** every trajectory stays finite.
* **C2:** for "same"-order variants, candidate-vs-baseline divergence ≤ 2× the largest
  baseline-vs-baseline divergence at every step and for every quantity.
* **C3:** the same with 10× for `feature_owner`.

**Amendments** after the first smoke run, before any recorded result, applied to every variant
including the null control:
* equal numbers of pairs per side;
* a one-FP32-ulp floor for the scalar loss;
* the null control itself.

**Results** (`training_validation_split.json`, `summarize_training.py`):
* **C1 passes everywhere:** every trajectory of every variant in all 54 configurations is
  finite.
* **C2 / C3 cannot discriminate.** The *null control* fails C2 in **33 / 54** configurations.
  The candidates fail at similar rates: `vectorized` 17/36, `swizzle4` 22/36,
  `vectorized_transpose` 25/36, `feature_owner` 11/27 (its 10× factor).
* **Why.** Run-to-run divergence of a nonlinear trajectory is sporadic and heavy-tailed. The
  forward FP32 atomics produce last-bit differences only sometimes, so 4 baseline runs often
  coincide exactly at a step (envelope 0), and a single BF16 rounding flip later changes a ReLU
  decision.
* **Example.** An investigated outlier, with 12 baseline runs of the timing loss and SR, shows a
  contiguous block of about 1000 moment elements differing by about 40 % while that step's
  dW is identical. The flip happened in the previous step, and it occurs in plain baseline
  runs too.
* **Fair comparison:** final-step divergence of each variant relative to the null control in
  the same configuration (median / max over configurations):

  | Variant | W | dX | m |
  |---|---|---|---|
  | vectorized | 1.00 / 1.42 | 1.00 / 1.29 | 1.00 / 1.32 |
  | vectorized_transpose | 1.00 / 3.05 | 1.00 / 1.44 | 1.00 / 3.54 |
  | swizzle4 | 0.99 / 1.28 | 1.00 / 1.59 | 1.00 / 31.1 |
  | feature_owner | 1.00 / 1.19 | 1.03 / 2.42 | 1.00 / 3.04 |
  | generic backend | 1.00 / 1.50 | 1.01 / 1.78 | 1.00 / 2.34 |

  The two large maxima are single sporadic flips. Their per-step series are in
  `training_validation_split.json`: `vectorized_transpose`'s first-step moment divergence was
  2.9e-3 vs the null's 2.2e-5 in one `square`/SR configuration; a rerun of that configuration
  showed such jumps in baseline runs as well.
* Typical final-step parameter divergence, from both baseline-vs-baseline and
  candidate-vs-baseline pairs:
  * **Deterministic rounding:** 3–5e-4 relative after 16 steps.
  * **SR or FP32 optimizer:** 2e-6 to 4e-4 depending on the model.
* **Verdict.** No variant shows drift beyond the variation the baseline shows against itself.
* **Coverage.** These runs used the non-hoisted `vectorized` / `vectorized_transpose` builds.
  The selected `*_hoisted` / `swizzle4_capped` builds differ only in address arithmetic and
  registers, and are bitwise equal per CTA (§9.1). A repeat of the validation with exactly the
  selected variants is in `training_validation_selected.json` (§9.5).

## 5. P2: the current scatter

### 5.1 Ownership and layout
* **Grid.** `sg_dx_kernel<ACT, 8, MT, NW, SPLIT, EPI>` runs one CTA per (chunk of items of
  supergroup q, BM = 16·MT tokens).
* **Per block** (8 neurons sharing one record of 8 distinct local features):
  * per m16 tile: 1 projection MMA m16n8k8 (C = bias) → mask epilogue → 2 gradient MMAs
    (hi / lo);
  * accumulator layout: lane (t = lane/4, c = lane%4) holds dA for tokens t, t+8 of fan-in
    positions 2c (f0) and 2c+1 (f1).
* **Private slab** (`dx_group_warps = 1`): each warp owns an FP32 copy [S][BM]. `sg_dxs_off`
  stores (t, t+8) as one float2 and xors the 16-token segment with the feature parity.
* **Update:**

```
p0 = slab + f0*BM + 16*(seg ^ (f0&1)) + 2*t ;  p1 likewise for f1
v0 = LDS.64 p0 ; v1 = LDS.64 p1 ; v0 += {da0, da2} ; v1 += {da1, da3} ; STS.64 p0, v0 ; STS.64 p1, v1
```

* **Race freedom.** A record's 8 features are distinct, so no two lanes of the owning warp
  touch the same word.
* **Shared copies** (`dx_group_warps > 1`) use a 64-bit CAS loop.
* **Flush.** Copies are summed in a fixed order, then flushed with one FP32 atomic per
  (feature, token) per CTA.

### 5.2 Static SASS (the selected mask-epilogue body of `sg_dx_kernel<1, 8, 4, 8, 1, 2>`; `sass_S16_T64.json`)

**Totals.** The unit-loop body (4 blocks, full and masked paths) has **1952 instructions**:
* HMMA.1688: 96
* LDS.64: 64, STS.64: 32, ATOMS.CAS.64: 48 (the CAS path is compiled into the same loop
  because `priv` is a run-time value)
* BRA / BSSY / BSYNC: 228
* MOV: 228
* IMAD / IADD / LEA: 195

**Excerpt** of one tile's address arithmetic and private update (addresses from the dump):

```
IMAD.SHL.U32 R14, R40, 0x10, RZ ; IMAD.SHL.U32 R13, R15, 0x10, RZ   # (f & 1) << 4
ISETP.NE.AND P3, PT, R16(dx_group_warps), 0x1, PT                  # run-time private / CAS
IMAD R40, R40, 0x40, R58 ; IMAD R15, R15, 0x40, R58                 # f * BM + 2t
LOP3.LUT ... 0x10 ; IADD3 R38, R0, R40, R14 ; IADD3 R67, ... ; LEA ×2 # segment xor, copy base
@!P3 BRA <private>
...                                                                 # CAS loop (shared copies)
<private>: LDS.64 R36, [R38.X4+UR10] ; LDS.64 R16, [R67.X4+UR10]
FADD.FTZ ×4 ; STS.64 [R38.X4+UR10], R36 ; STS.64 [R67.X4+UR10], R16
```

**What the static code shows.**
* **Access width.** The intended 64-bit shared accesses are emitted.
* **Address work.** About 10 integer instructions per tile and two addresses are recomputed
  per tile.
* **Base re-materialization.** NVCC re-materializes the copy base and segment xor instead of
  keeping them live, even though registers do not limit residency (§8).

**Resources.**
* 79 registers, no stack, 36 KiB dynamic shared memory at S = 16, T = 64.
* That gives 2 CTAs, i.e. 16 warps, per SM, **limited by shared memory**. The registers alone
  would allow 3 CTAs.

### 5.3 Bank conflicts (*computed*, `bank_model.py`, from the device offsets and random records)

A 64-bit phase is a half-warp: 4 features × 4 token pairs. Its cost is the largest number of
those 4 features in one bank group:

| Layout (S = 16) | Width | Load wavefronts / m16 tile | RMW wavefronts / tile | Shared instr. / tile |
|---|---|---:|---:|---:|
| baseline (2 groups, parity) | 64 | 10.59 | 21.2 | 4 |
| swizzle4 (4 groups, f & 3) | 64 | 7.84 | 15.7 | 4 |
| vectorized_parity (2 groups) | 128 | 10.59 | 21.2 | 2 |
| vectorized (4 groups) | 128 | 7.89 | 15.8 | 2 |
| vectorized_transpose | 128 | **4.00** | **8.0** (+4 SHFL) | 2 |
| conflict-free minimum | | 4 | 8 | |

At S = 32 / 64 the values are 21.6 / 21.8 (baseline) and 16.5 / 16.7 (4 groups); the
transposed layout stays at 8.0.

For comparison, the operand gathers add 2 wavefronts per tile (ldmatrix x4).

**Implication (computed).** At H = 100M the baseline RMW alone is about 1.36e11 wavefronts,
against 1.39e11 wavefronts/s of shared bandwidth (82 SM × 1 / clk × 1.7 GHz). This is a
model, not a counter.

### 5.4 Attribution by ablation (*ablation*: wrong results)

**The previous no-scatter ablation** (`SP_ABLATE_SG=2`; library `13907ea43ad93697`) was not a
pure scatter removal:
* it removes the private RMW **and** the CAS path, and 144 of 228 branch instructions;
* the loop shrinks from 2371 to 1580 instructions and registers drop from 79 to 74.

**This session's ablation**, the same build of the current sources at H = 10M, M = 8192
(`screen/micro_ablate*`), measures the dX kernel alone:

| S | Baseline | No local update | Gap |
|---:|---:|---:|---:|
| 16 | 143.9 | 101.5 | 42.4 (29 %) |
| 32 | 177.5 | 129.9 | 47.6 (27 %) |

**The runtime branch costs nothing:**
* `baseline_private`, the same layout compiled for warp-private copies only (CAS path and
  branches gone), is 148.5 vs 149.9 ms (complete steps, corrected harness,
  `screen/steps10M_S16_factorial`);
* so the instruction-count difference of the ablation is not itself the cost;
* the cost is the update's shared traffic.

**Full-scale ablation** (`ablation_100M/`, same build flag, H = 100M, M = 8192, 2 rounds), ms:

| S | Baseline dX | No local update (ablation) | Selected dX | Share of the gap removed |
|---:|---:|---:|---:|---:|
| 16 | 1377.7 | 905.0 | 1080.8 | 62 % |
| 32 | 1630.2 | 1101.0 | 1293.0 | 64 % |

The baseline and selected values are from `memory_recheck` / `selected_100M` in the same
session. The ablation is not an attainable bound: it keeps the products live but stores
nothing, and it compiles to different code.

## 6. Experiment A: layouts and vectorized private updates (`sparse_pool_sg_dxs.cuh`)

All variants reuse the baseline's staging, gathers, coefficients and `sg_dx_tile` arithmetic.
The baseline `sg_dx_kernel` object files are untouched: they are the same translation units
with the same hashes.

### 6.1 Layouts (bijections of one [S][BM] copy; MT ≥ 2)

Notation: seg = t >> 4, tp = t & 7, h = (t >> 3) & 1.

* **swizzle4** (float2): `f·BM + 32·(seg>>1) + 8·(((seg&1)<<1 | tp>>2) ^ (f&3)) + 2·(tp&3) + h`.
  Within a 32-token block, the 4 (segment, half-warp) combinations are xor-permuted by f & 3,
  so a half-warp phase's features use 4 bank groups.
* **vectorized** (float4 records): `f·BM + 32·(seg>>1) + 8·((tp>>1) ^ (f&3)) + 4·(tp&1) + 2·(seg&1) + h`.
  One 16-byte record per (feature, token t of a tile pair) = {tile 2i: t, t+8; tile 2i+1: t, t+8}.
  One LDS.128 / FADD×4 / STS.128 replaces two float2 RMW sequences. A quarter-warp phase
  (4 features × 2 records) uses 4 bank groups.
* **vectorized_parity:** the same records with 2 groups (xor with 2·(f & 1)), as a layout control.
* **vectorized_transpose:** records in token order (`4·tp`). Before the update, 8 shuffles move
  the records so that lane (u = lane & 7, k = lane >> 3) holds the records of fan-in positions
  2k, 2k+1 at token u (source lane 4u + k). Each 128-bit phase then touches the 8 records of
  **one** feature, so there are no conflicts for any feature set.
* **\*_hoisted:** the same layouts with the two record addresses computed once per block. The
  offset is linear in the tile pair (base + 32·i floats), and the static code shows the
  per-record address arithmetic gone (§6.4).

### 6.2 Proofs and checks
* **Bijection, alignment, ownership** (`test_layout_bijection_alignment_ownership`, 84 cases:
  every layout × T ∈ {32, 64, 128} × S ∈ {8, 16, 32, 64}, 64 random records each):
  * the device offsets (`sp_sgx_layout`) are a permutation of the copy;
  * every record is 16-byte aligned and contiguous in the stated order;
  * no two lanes of an instruction touch the same word;
  * the computed conflict degree equals the stated rule: 1 for the transposed layout, the
    maximum multiplicity of f mod 4 or f mod 2 for the others.
* **Tails and zero.** Partial token tiles are masked exactly as in the baseline (dz = 0 for
  invalid tokens) and the copies are zero-initialized, so tails need no extra predicates.
* **Same reduction.** The final copy reduction and the global flush use the same layout
  function (`sgx_off`).
* **Private copies only.** The vector variants refuse shared copies at launch (`-4`) and in
  `sg_dx_scatter` (explicit requests raise). Only the float2 `swizzle4` / `swizzle4_capped`
  support shared CAS copies.
* **Addition order.** Each (feature, token) of a warp's copy receives the same additions, in
  the same order (block order), as in the baseline. The vector records only interleave
  different tokens.
* **Bitwise local equality.** With R = 1 and one chunk per supergroup, each (feature, token
  tile) is produced by one CTA and flushed by one atomic onto zero. In this deterministic
  setting all order-preserving variants are **bitwise equal** to the baseline (§9.1).

### 6.3 Results, H = 10M, M = 8192, S = 16, T = 64, 8 warps (ms)

Two kinds of rows:
* the 2 × 2 controls are complete steps (corrected harness, `screen/steps10M_S16_factorial`,
  commit `4a41593`, 4 rounds);
* the remaining rows are constant-upstream microbenchmarks (`bench_dx_micro.py`,
  `screen/micro_hoist_S16`, 6 rounds).

| Variant | Kind | dX kernel | Note |
|---|---|---:|---|
| baseline (run-time private / CAS) | step | 149.9 | |
| baseline_private (float2, 2 groups) | step | 148.5 | branch / CAS path cost ≈ 0 |
| swizzle4 (float2, 4 groups) | step | 130.4 | layout alone: −13 % |
| vectorized_parity (float4, 2 groups) | step | 148.2 | width alone: −1 % |
| vectorized (float4, 4 groups) | step | 126.7 | both: −15.5 % |
| baseline | micro | 143.96 | |
| vectorized | micro | 121.70 | |
| vectorized_hoisted | micro | 120.47 | |
| vectorized_transpose | micro | 123.34 | |
| **vectorized_transpose_hoisted** | micro | **118.27** | −17.8 % |

**Interpretation.**
* Layout is the main effect. Vector width pays only once conflicts are reduced: wider
  instructions alone do not.
* The transposed layout halves wavefronts again but adds 8 SHFL per block-pair.
* At H = 10M the hoisted vector variants are within 2 % of each other. At H = 100M the
  transposed one is clearly faster: dX 1085 vs 1148 ms (§9.2).
* At S = 32 (T = 32), the vector variants are all about 145 ms vs 176–180 for the baseline
  (−17 to −18 %).

### 6.4 Static code of the candidates (selected mask body, S = 16, T = 64, 8 warps)

| Kernel | Instructions | Shared accesses | SHFL | IMAD / IADD / LEA | BRA | Registers | Stack |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 1952 | LDS.64 64, STS.64 32, ATOMS.64 48 | 8 | 195 | 228 | 79 | 0 |
| baseline_private | 1277 | LDS.64 32, STS.64 32 | 8 | 138 | 84 | 77 | 0 |
| swizzle4 (private) | 1302 | LDS.64 32, STS.64 32 | 8 | 134 | 84 | 75 | 0 |
| vectorized | 1138 | LDS.128 16, STS.128 16 | 8 | 105 | 52 | 78 | 0 |
| vectorized_hoisted | 1113 | LDS.128 16, STS.128 16 | 8 | 90 | 52 | 80 | 0 |
| vectorized_transpose | 1210 | LDS.128 16, STS.128 16 | 72 | 118 | 68 | 82 | 0 |
| **vectorized_transpose_hoisted** | 1171 | LDS.128 16, STS.128 16 | 72 | 98 | 68 | 84 | 0 |

**Residency (computed).**
* All rows have 36 KiB of dynamic shared memory, giving 2 CTAs / 16 warps per SM.
* S = 32, T = 32: baseline 67 registers, `vectorized_transpose_hoisted` 75, both 2 CTAs / 16
  warps (shared memory 36 KiB).

The static counts are one loop-body iteration with both full and masked paths, not executed
counts. `resources_S*.json` and `sass_S16_T64.json` have every variant.

**Hoisted excerpt** (`vectorized_hoisted`: the update after the gradient MMAs, with no address
arithmetic between them):

```
LDS.128 R40, [R35.X4+UR7] ; LDS.128 R36, [R44.X4+UR7]
FADD.FTZ ×8
STS.128 [R35.X4+UR7], R40 ; STS.128 [R44.X4+UR7], R36
```

The non-hoisted version recomputes the copy base from `SR_TID.X` and `a.S` for every record
pair (`S2R`, `SHF`, `IMAD ×6`, `LOP3 ×2`).

**Selected kernel** (`vectorized_transpose_hoisted`, `sass_excerpts.txt`): one tile pair, after
the gradient MMAs.

```
HMMA.1688.F32.BF16 R36, R38, R69, R44 ; HMMA.1688.F32.BF16 R44, R2, R69, R76   # lo pass, tiles 2i / 2i+1
SHFL.IDX R36..R39, R44..R47 <- lane 4u + k   (8x, with BRA.DIV convergence checks)  # transpose
IMAD R72, R72, 0x40, R19 ; IMAD R73, R73, 0x40, R19                            # 2 address ops / pair
LDS.128 R76, [R72.X4+UR7] ; LDS.128 R80, [R73.X4+UR7]                          # conflict-free phases
FADD.FTZ x8
STS.128 [R72.X4+UR7], R76 ; STS.128 [R73.X4+UR7], R80
```

## 7. Experiment B: register aggregation / feature ownership

### 7.1 Ownership analysis
* **Tokens are already owned.** MMA row t / t+8 always belongs to lane quad t, so only the
  feature dimension moves between blocks.
* **Shuffle routing is ruled out (computed).** Moving dA values to fixed per-lane feature
  owners with shuffles costs, per tile, about 12 SHFL plus about 32 SEL to select dynamically
  addressed registers. That is far more than the 4 shared instructions it would replace.
  `acc[dynamic feature]` would also spill.
* **Implemented instead: `feature_owner` (SC_OWNER), routing by the tensor core.** Per block,
  B operand column n of n8 tile `ns` = local feature 8·ns + n.
  * **Lookup.** All lanes hold the block's 8 local rows packed:
    * 3 SHFL-xor build 4-bit nibbles for S ≤ 16 (8-bit bytes in two words for S ≤ 32);
    * a zero-nibble search finds the fan-in position p with feat[p] = n;
    * one SHFL fetches the W^T fragment column p from lane 4p + (lane % 4), or 0 if the feature
      is absent.
  * **Accumulation.** The gradient MMAs then add the block's contributions into register
    accumulators `acc[MT][S/8][4]`: lane (t, c) owns tokens t, t+8 of features 8·ns + 2c,
    8·ns + 2c + 1. These are compile-time-indexed registers with no stack (verified).
  * **Store.** The accumulators are stored once per CTA into the warp's private copy, with the
    same reduction and flush as before.
  * **Cost.** S / 8 MMAs per gradient pass instead of 1: 20 HMMA per block instead of 12 at
    S = 16, T = 64 (from the source), plus 3 + S/8 SHFL per block for S ≤ 16 (6 + S/8 for
    S ≤ 32).
* **Semantics.** Absent features get exact-zero B columns. A non-finite weight gives
  inf · 0 = NaN only inside present features' sums, exactly as in the baseline (tested with an
  inf weight). The FP32 summation order differs: contributions are accumulated inside the MMA,
  so this is a separate numerical variant (`scatter_order = "mma_accumulate"`).

### 7.2 Resources and measurements

**Registers** (8-warp builds):

| S | Tile | Registers | CTAs / SM |
|---:|---|---:|---:|
| 8 | T = 128 | 151 | 1 |
| 16 | T = 32 | 84 | 2 |
| 16 | T = 64 | 123 | 2 |
| 16 | T = 128 | 190 | 1 |
| 32 | T = 32 | 111 | 2 |
| 32 | T = 64 | 161 | 1 |

The 16-warp builds spill (stack 64–1080 B) and are not used.

**Timings** (dX kernel, ms):
* H = 10M, S = 16, T = 64: 124.7 (micro) / 132.8 (step) vs 120.5 / 120.9 for the vector
  variants.
* S = 32: 274 ms, 1.5× slower than the baseline, because of the 4× gradient MMA work.
* H = 100M, S = 16: **1169.5** vs baseline 1380.9 and `vectorized_transpose_hoisted` 1085.4
  (step 2592.5 vs 2512.5).

**Verdict.** Correct but dominated by the vector layouts, and rejected for S ≥ 32. It remains
an explicit, experimental, opt-in plan (§12); no hybrid was built, because the full owner
already loses at S = 16.

## 8. Experiment C: focused tile and reduction retuning (H = 10M, M = 8192; `screen/micro_S*`)

**S = 16** (warp-private copies), dX microbenchmark (ms):

| T / NW | baseline | swizzle4 | vectorized | vectorized_transpose | feature_owner |
|---|---:|---:|---:|---:|---:|
| 32 / 8 | 166.5 | 139.3 | 132.4 | 132.9 | 159.7 |
| 64 / 8 | 144.6 | 127.6 | **120.0** | 122.3 | 124.7 |
| 64 / 16 | 145.6 | 128.3 | 122.9 | 125.1 | 128.0 |
| 128 / 8 | 168.2 | 147.2 | 129.4 | 139.1 | 137.8 |

T = 128 is not feasible at 16 warps (copies 128 KiB).

**S = 32**: T = 32 / 8 warps is best for all variants (baseline 180.2, vectorized 146.9). The
previous optimum tiles stay optimal after the change.

**S = 64** (shared CAS copies, WG ∈ {2, 4, 8, 16}, plus private T = 32):
* At T = 64 the plain `swizzle4` needed 82 registers, which cost the third CTA per SM that the
  baseline's 79 registers allow.
* `swizzle4_capped` (`__launch_bounds__(256, 3)`, 80 registers, no stack) restores it.
* dX microbenchmark (ms):

| Variant | dX kernel |
|---|---:|
| baseline (T = 64, WG = 8) | 204.6 |
| swizzle4 | 206.6 |
| **swizzle4_capped** | **183.1** (−10.5 %) |
| swizzle4, T = 128 / 16 warps | 193.1 |
| private vector, T = 32 | 212–216 |

**Not changed, as instructed:** the CTA work chunks (4096 units) and the global merge (one
atomic per locally reduced feature / token). Full-scale shapes are in §9.

## 9. Results

### 9.1 Fixed-upstream correctness (`test_dx_scatter.py`: 481 passed, 29 skipped; `numerics_fixed_dy.json`)

**Tests:**
* every variant against the baseline grouped dX with the same grouped forward and dW;
* S ∈ {8, 16, 32, 64}, R ∈ {1, 2, 4, 8, 16}, T ∈ {32, 64, 128}, 8 / 16 warps;
* split and BF16, ReLU and identity, token tails (B = 16 … 264);
* ragged, partial, tiny and empty buckets (H from 100 to 1M);
* NaN, ±inf, subnormal and huge coefficients, signed zeros, z = 0 boundaries, inf weights;
* repeated backward accumulation; shared-copy WG ∈ {1, 2, 4, 8}.

**Findings:**
* **dW / db are bitwise equal to the baseline in every case.** The dW kernel is unchanged.
* dX is within 1e-6 relative of the baseline.
* All variants meet the strict reference tolerances.
* **Deterministic setting:** all order-preserving variants are **bitwise identical** to the
  baseline, including NaN / inf / signed-zero / subnormal patterns. `feature_owner` matches
  NaN / inf positions and zero signs exactly, and finite values within 1e-6 L2-relative
  (asserted; the per-element distribution is in the table below).

**Fixed-dY distributions** (`numerics_fixed_dy.json`, H = 1M, M = 1024, D = G = 1024):
* The dX difference of the selected variants from the baseline is the same size as the
  baseline's own run-to-run difference (global atomic order).
* dW / db are bitwise equal in every row.

| Graph / dX tile | Baseline run-to-run: max abs diff / scale, equal-bit fraction | Selected vs baseline | feature_owner vs baseline |
|---|---|---|---|
| S = 16, R = 4, T = 64 (split) | 7.0e-8, 0.985 | 7.0e-8, 0.984 | 6.3e-7, 0.040 (p99 256 ulp) |
| S = 32, R = 8, T = 32 (split) | 8.9e-8, 0.862 | 1.8e-7, 0.888 | 3.6e-7, 0.075 (p99 160 ulp) |
| S = 64, R = 16, T = 64, WG = 8 (split, swizzle4_capped) | 1.7e-7, 0.237 | 2.1e-7, 0.243 | n/a |
| S = 16 / 32, R = 1, one chunk (deterministic; split and BF16) | 0, 1.000 | **0, 1.000 (bitwise)** | 0.8–2.8e-6, 0.009–0.043 |

The BF16-gradient rows behave the same (file).

### 9.2 Full scale (H = 100M, M = 8192; `final_100M/`, `final_100M.csv`; commit `a3fa200`)

**Common to all rows:**
* 4 alternating rounds × 2 restored repetitions × 3 steps = 24 steps per variant;
* the median of all steps, with the range of the 4 per-round medians in brackets;
* dX total = dX init (0.04 ms) + dX kernel + post-processing (0.12 ms: transpose, BF16
  conversion);
* coefficient preparation (c1 and c1T: 0.32 ms) and input layout (0.13 ms) are identical
  across variants.

| C / S / R | Seed | Variant | Forward | dX kernel | dW | Optimizer | Step [round spread] | vs baseline |
|---|---|---|---:|---:|---:|---:|---|---:|
| 8 / 16 / 4 | 0 | baseline (auto) | 316.3 | 1380.9 | 1079.2 | 22.4 | 2800.6 [2791.2–2808.2] | |
| | 0 | vectorized_transpose_hoisted | 318.8 | **1085.4** | 1082.9 | 22.3 | **2512.5** [2507.7–2527.6] | −288.1 (1.115×) |
| | 0 | vectorized_hoisted | 319.7 | 1147.9 | 1083.7 | 22.5 | 2571.1 [2565.5–2590.2] | −229.5 (1.089×) |
| | 0 | feature_owner (B) | 317.6 | 1169.5 | 1078.9 | 22.2 | 2592.5 [2565.0–2615.7] | −208.1 (1.080×) |
| | 0 | generic backend (context) | 399.0 | 2452.1 | 2143.0 | 22.3 | 5015.2 [4999.1–5037.6] | |
| | 1 | baseline → vectorized_transpose_hoisted | 318.2 → 317.3 | 1379.5 → **1084.6** | 1077.8 → 1083.3 | | 2799.3 → **2507.0** | −292.3 (1.117×) |
| | 2 | baseline → vectorized_transpose_hoisted | 316.7 → 317.7 | 1375.6 → **1083.4** | 1080.8 → 1081.9 | | 2797.0 → **2507.6** | −289.4 (1.115×) |
| 8 / 32 / 8 | 0 | baseline → vectorized_transpose_hoisted | 321.8 → 320.3 | 1632.1 → **1292.7** | 1119.5 → 1122.2 | | 3099.5 → **2758.8** | −340.7 (1.123×) |
| | 1 | | 321.9 → 322.4 | 1635.6 → **1290.0** | 1117.8 → 1119.7 | | 3096.7 → **2758.7** | −338.0 (1.123×) |
| | 2 | | 321.8 → 322.5 | 1632.8 → **1292.0** | 1117.6 → 1123.8 | | 3099.1 → **2760.1** | −339.0 (1.123×) |
| | 0–2 | vectorized_hoisted | | 1316.5 / 1314.8 / 1313.6 | | | 2781.2 / 2778.7 / 2779.1 | −318…−320 |
| 8 / 64 / 16 | 0 | baseline (auto: T = 64, WG = 8) | 326.9 | 1999.4 | 1155.0 | 22.1 | 3504.5 [3496.7–3511.1] | |
| | 0 | **swizzle4_capped** (same tile) | 327.2 | **1753.0** | 1153.9 | 21.4 | **3257.8** [3226.4–3263.8] | −246.7 (1.076×) |
| | 0 | swizzle4, T = 128, 16 warps | 326.3 | 1854.9 | 1151.1 | 21.8 | 3355.1 | −149.4 (1.045×) |
| | 0 | vectorized_transpose_hoisted, T = 32, private | 325.7 | 1650.2 | 1153.0 | 22.1 | 3153.6 [3152.5–3156.0] | −350.9 (1.111×) |
| 64 / 16 / 4 | 0 | grouped auto (unchanged control) | 270.8 | 724.1 | 731.7 | 21.9 | 1752.2 [1746.9–1755.0] | |
| 64 / 64 / 16 | 0 | grouped auto (unchanged control) | 271.3 | 720.1 | 750.4 | 21.4 | 1763.3 [1759.5–1765.8] | |

**S = 64 needs a separate decision.** The warp-private T = 32 vector layout (1 CTA / 8 warps per
SM, 70 KiB) is the fastest at H = 100M (−10.0 %). It is **slower** at H = 10M: 398.5 vs
388.1 ms step, +2.7 % (`screen/steps10M_S64_b`, commit `a3fa200`). `swizzle4_capped` wins at
both scales: −5.5 % at 10M, −7.0 % at 100M. Only `swizzle4_capped` is selected
automatically.

**Forward and dW are unchanged controls.** Their medians move by ≤ 1 % between variants (±3 ms
on about 320 / 1080 ms), within run-to-run spread.

**Precision, batch size, overlap** (H = 100M, seed 0, `vectorized_transpose_hoisted` vs baseline):

| Case | dX kernel | Step | vs baseline |
|---|---|---|---:|
| S = 16, BF16 gradients | 1354.0 → **1027.8** | 2690.0 → **2363.7** | −326.3 (1.138×) |
| S = 32, BF16 gradients | 1533.0 → **1218.2** | 2912.0 → **2601.1** | −310.9 (1.120×) |
| S = 16, M = 1024 | 177.8 → **137.3** | 379.6 → **341.4** | −38.1 (1.112×) |
| S = 32, M = 1024 | 209.0 → **162.8** | 415.7 → **375.4** | −40.3 (1.107×) |
| S = 16, R = 16 (more overlap), H = 10M (`extra/…_explicit`) | 157.9 → **128.5** | 325.7 → **300.3** | −25.4 (1.085×) |

### 9.3 Selected plan (`cuda_grouped_dx_scatter="auto"`) vs explicit baseline (`selected_100M/`)

H = 100M, M = 8192, split, auto backend and tiles; the same session as §9.2.

| C / S / R | Selected kernel | dX kernel | Step [round spread] | vs baseline |
|---|---|---|---|---:|
| 8 / 16 / 4 | `sgx_dx_kernel<1, 8, 4, 8, 1, 2, 9, 1>` (vectorized_transpose_hoisted) | 1374.3 → **1082.7** | 2794.5 → **2504.9** [2502.9–2580.4] | −289.6 (1.116×) |
| 8 / 32 / 8 | `sgx_dx_kernel<1, 8, 2, 8, 1, 2, 9, 1>` | 1630.2 → **1293.0** | 3090.0 → **2759.6** [2748.5–2785.2] | −330.4 (1.120×) |
| 8 / 64 / 16 | `sgx_dx_kernel<1, 8, 4, 8, 1, 2, 10, 1>` (swizzle4_capped) | 2011.6 → **1766.3** | 3525.1 → **3280.0** [3260.1–3299.0] | −245.0 (1.075×) |

* The selection is verified per run: `last_info` scatter and kernel name.
* Commits: S = 32 at `0c61f27` (clean); S = 16 at `d97b69e` and S = 64 at `0c61f27` with
  uncommitted edits to `numerics_training.py` / `README.md` only. The library is the same, so
  the kernels are identical.

### 9.4 Secondary C = 64 dimension check (`extra/`, H = 100M, M = 8192, one session)

| S / R | Plan (unchanged per-phase auto) | Forward | dX | dW | Step | Generic backend step |
|---|---|---:|---:|---:|---:|---:|
| 16 / 4 | grouped fwd / dX / dW | 272.5 | 724.0 | 730.3 | **1751.0** | 1812.5 |
| 64 / 16 | grouped fwd / dX, generic dW | 271.5 | 714.3 | 749.5 | **1760.0** | 1820.3 |
| 128 / 32 | grouped fwd, generic dX / dW | 273.2 | 739.9 | 752.7 | **1790.1** | 1825.2 |

* At C = 64, S = 16 / 64 / 128 are within 2.2 % of each other. The grouped forward is worth
  about 35 ms at every S.
* S = 128 had no previous full-scale measurement; it adds 30–39 ms over S ≤ 64, because the
  grouped dX and dW are not selected there.
* This is a speed comparison only; graph-quality questions are separate and not studied.

### 9.5 Training validation of exactly the selected variants (`training_validation_selected.json`)

Setup: the same models, optimizer modes and checks as §4.3, at H = 1M, M = 1024. Graphs: C = 8
S = 16 R = 4 and S = 32 R = 8 (16 steps), S = 16 R = 16 and S = 64 R = 16 with shared copies (8 steps).
Variants: `vectorized_transpose_hoisted`, `vectorized_hoisted`, `swizzle4_capped` (S = 64),
`feature_owner` (S ≤ 32), the generic backend, and the null control.

* **All trajectories are finite** (C1 passes everywhere).
* **C2 fails as often for the null as for the candidates:**

  | Variant | Configurations failing C2 / C3 |
  |---|---:|
  | null control | 21 / 36 |
  | vectorized_transpose_hoisted | 18 / 27 |
  | vectorized_hoisted | 16 / 27 |
  | swizzle4_capped | 7 / 9 |
  | feature_owner (10× factor) | 16 / 27 |
* **Final-step divergence relative to the null control, median (max):**

  | Variant | W | dX | m |
  |---|---|---|---|
  | vectorized_transpose_hoisted | 1.00 (1.25) | 1.00 (1.40) | 1.00 (1.21) |
  | vectorized_hoisted | 1.00 (2.61) | 1.00 (1.21) | 1.00 (1.31) |
  | swizzle4_capped | 1.01 (1.36) | 1.00 (1.10) | 1.00 (23.3) |
  | feature_owner | 1.00 (1.29) | 1.02 (2.52) | 1.00 (1.06) |
  | generic backend | 1.00 (2.83) | 1.01 (2.04) | 1.00 (10.8) |

  The large moment maxima are single sporadic flips, which the generic backend shows as well.
* **Same conclusion as §4.3:** no variant-specific drift.

## 10. Sanitizer and regression tests

**Compute Sanitizer** (`run_sanitizer_dx_scatter.sh`, `CUDA_MODULE_LOADING=LAZY`,
`sanitizer_dx_scatter.log`):
* memcheck: 0 errors
* racecheck: 0 hazards
* initcheck: 0 errors
* synccheck: 0 errors

**Workload coverage:** 57 launches over every variant, S = 8 / 16 / 32 / 64, T = 32 / 64 / 128,
8 / 16 warps, private and shared (WG = 2, 8) `swizzle4`, split and BF16, ReLU and identity,
token tails, tiny buckets, and the non-finite fallback body.

`swizzle4_capped` was added later and checked separately (`run_sanitizer_capped.sh`,
`sanitizer_capped.log`):
* coverage: S = 64 with WG = 8 / 2, S = 16 with WG = 8 / 1, T = 64 / 128, split and BF16, ReLU and
  identity, non-finite fallback;
* all four tools are clean (0 errors, 0 hazards).

The previous grouped kernels' sanitizer reports (`input_supergroups/sanitizer.log`) are
unchanged history.

**Full test suite** (`test_suite.log`, all files, from commit `0c61f27` sources): **2551
passed, 30 skipped**.
* One skip is pre-existing.
* 29 are `test_dx_scatter.py` configurations that do not fit in shared memory or need
  `feature_owner` with S ≤ 32.
* The previous suite's 2061 tests are included; `test_input_supergroups.py` now applies the
  narrowed reference exception.

## 11. Remaining bottlenecks and one next experiment

**Computed estimate.** At H = 100M, S = 16, the selected dX kernel (1085 ms) runs about 1.9×
over the measured MMA floor:
* 6.4e9 block-tiles × 3 m16n8k8 at 69 TFLOPS ≈ 0.57 s.
* The remaining local update is conflict-free 128-bit RMW plus 8 SHFL per block-pair.

**What remains, by component** (static and ablation evidence):
* the projection recompute for the ReLU mask (1 of 3 MMAs per block-tile, plus 4 FSETP, 4 SEL,
  2 PRMT and 4 LOP3 per tile);
* the bias C-operand moves (8 MOV per tile pair);
* the per-block unit decode, gather and movmatrix;
* the local update.

**Next experiment (one): share the ReLU-mask recompute between dX and dW.**
* Both grouped backward kernels recompute z = XW + b for every (token, neuron) of the same
  blocks, and dX needs only the sign.
* A grouped backward that computes z once per (block, token tile) would remove one of dX's
  three MMAs and its mask epilogue, and dW's matching recompute. It would process a unit chunk
  with dW-style exclusive ownership and keep the dX local slab for a bounded token window
  flushed per stage.
* At most about 0.19 s each (computed: 13.1 TFLOP at 69 TFLOPS).
* Two prerequisites to measure:
  * the extra global dX atomics of stage-wise flushes must stay negligible, which requires
    CTAs covering whole buckets;
  * a register budget compatible with dW's 2 CTAs / SM.

## 12. Selected defaults and fallbacks

**New control.** `PhysicalPlan.cuda_grouped_dx_scatter`: `auto` | `baseline` |
`baseline_private` | `swizzle4` | `swizzle4_capped` | `vectorized` | `vectorized_transpose` |
`vectorized_hoisted` | `vectorized_transpose_hoisted` | `vectorized_parity` | `feature_owner`.

**What `last_info["grouped_dx"]` records:** variant, layout, order category (`same` /
`mma_accumulate`), fallback reason, the kernel template identity
(`sgx_dx_kernel<ACT, 8, MT, NW, SPLIT, EPI, SC, NS>`), T, NW, WG, chunk units, shared
memory and work-list bytes. The benchmark aborts if a requested variant did not run.

**`auto` (`SG_AUTO_DX_SCATTER`).** The measured choice is used only at the measured keys:
(C, S, auto T, NW, WG, precision, activation). Everything else keeps the baseline:

| Key | Choice | Measured at |
|---|---|---|
| C = 8, S = 16, T = 64, 8 warps, private, split / BF16, ReLU | `vectorized_transpose_hoisted` | H = 10M; H = 100M (3 seeds); BF16; M = 1024 |
| C = 8, S = 32, T = 32, 8 warps, private, split / BF16, ReLU | `vectorized_transpose_hoisted` | the same |
| C = 8, S = 64, T = 64, 8 warps, WG = 8, split, ReLU | `swizzle4_capped` | H = 10M and 100M |

**Kept at the baseline:** S = 8 / 128 / 256, Mean(identity), BF16 at S = 64, every non-auto
tile, and C = 64. Forward and dW plans are unchanged, and the generic kernels and unrestricted
connectivity are unaffected.

**Explicit unsupported requests raise `ValueError`:**
* C ≠ 8;
* T < 32;
* vector, owner or `baseline_private` with shared copies;
* `feature_owner` with S > 32;
* `swizzle4_capped` outside 8 warps / T ∈ {64, 128}.

Under `auto` the fallback is recorded in `scatter_fallback`.

**Experimental, explicit only:**
* `feature_owner` (different FP32 order);
* the warp-private S = 64 vector plan (scale-dependent);
* the controls (`baseline_private`, `vectorized_parity`, non-hoisted variants).

**Kernel builds:** C = 8, Mean(ReLU / identity), MT ∈ {2, 4, 8}, 8 / 16 warps, split (generic /
mask epilogue) and BF16 (generic). The non-finite fallback body is compiled into every
kernel.

## 13. Final table

**Conditions.** H = 100M unless noted. Corrected harness, same-session A/B against the
unchanged grouped baseline (auto plan, baseline scatter). Sanitizer: memcheck / racecheck /
initcheck / synccheck all clean on the variant's source (§10).

**Abbreviations.**
* Variability = range of the per-round step medians, baseline / candidate.
* Regs / spill = registers (baseline → candidate) and stack bytes; smem = dynamic shared memory.
* CTA/SM = computed residency (CTAs and warps per SM), limiter in brackets: smem = shared
  memory, regs = registers.
* Memory = production peak allocated / reserved; the benchmark-only snapshot is 8.38 GiB of
  pinned host memory (0 on the device).
* Numerics: "same order" = bitwise equal per CTA, and fully bitwise in the deterministic
  setting.
* Training = nonlinear multi-step validation (§4.3, §9.5).

**Speed and resources:**

| C / S / R / precision / M | Scatter / T / NW / WG | dX total (ms) | Complete step (ms) | Saved / speedup | Variability | Regs / spill / smem | CTA/SM |
|---|---|---|---|---|---|---|---|
| 8 / 16 / 4 / split / 8192 (seeds 0, 1, 2) | **vectorized_transpose_hoisted** / 64 / 8 / 1 | 1381.1 / 1379.7 / 1375.8 → 1085.5 / 1084.7 / 1083.6 | 2800.6 / 2799.3 / 2797.0 → **2512.5 / 2507.0 / 2507.6** | −288…−292 / **1.115–1.117×** | 2791–2808 / 2502–2528 | 79 → 84 / 0 / 36 KiB | 2 (16 warps) [smem] |
| 8 / 32 / 8 / split / 8192 (seeds 0, 1, 2) | **vectorized_transpose_hoisted** / 32 / 8 / 1 | 1632.3 / 1635.8 / 1633.0 → 1292.9 / 1290.2 / 1292.2 | 3099.5 / 3096.7 / 3099.1 → **2758.8 / 2758.7 / 2760.1** | −338…−341 / **1.123×** | 3093–3104 / 2753–2763 | 67 → 75 / 0 / 36 KiB | 2 (16) [smem] |
| 8 / 64 / 16 / split / 8192 | **swizzle4_capped** / 64 / 8 / 8 | 1999.5 → 1753.2 | 3504.5 → **3257.8** (selected run: 3525.1 → 3280.0) | −246.7 / **1.076×** | 3497–3511 / 3226–3264 | 79 → 80 / 0 / 26 KiB | 3 (24) [regs] |
| 8 / 16 / 4 / bf16 / 8192 | vectorized_transpose_hoisted / 64 / 8 / 1 | 1354.1 → 1028.0 | 2690.0 → **2363.7** | −326.3 / **1.138×** | 2688–2693 / 2361–2368 | 74 → 86 / 0 / 36 KiB | 2 (16) [smem] |
| 8 / 32 / 8 / bf16 / 8192 | vectorized_transpose_hoisted / 32 / 8 / 1 | 1533.2 → 1218.4 | 2912.0 → **2601.1** | −310.9 / **1.120×** | 2909–2918 / 2596–2605 | 64 (8 B stack) → 78 / 0 / 36 KiB | 2 (16) [smem] |
| 8 / 16 / 4 / split / 1024 | vectorized_transpose_hoisted / 64 / 8 / 1 | 177.8 → 137.3 | 379.6 → **341.4** | −38.1 / **1.112×** | 378–380 / 341–345 | as above | 2 (16) |
| 8 / 32 / 8 / split / 1024 | vectorized_transpose_hoisted / 32 / 8 / 1 | 209.0 → 162.8 | 415.7 → **375.4** | −40.3 / **1.107×** | 415–419 / 373–377 | as above | 2 (16) |
| 8 / 16 / 16 / split / 8192, H = 10M | vectorized_transpose_hoisted / 64 / 8 / 1 | 157.9 → 128.5 | 325.7 → **300.3** | −25.4 / 1.085× | 325–326 / 299–302 | as above | 2 (16) |
| 8 / 16 / 4 / split / 8192 (alternative A) | vectorized_hoisted / 64 / 8 / 1 | 1381.1 → 1148.1 | 2800.6 → 2571.1 | −229.5 / 1.089× | / 2566–2590 | 79 → 80 / 0 / 36 KiB | 2 (16) [smem] |
| 8 / 16 / 4 / split / 8192 (candidate B) | feature_owner / 64 / 8 / 1 | 1381.1 → 1169.7 | 2800.6 → 2592.5 | −208.1 / 1.080× | / 2565–2616 | 79 → 123 / 0 / 36 KiB | 2 (16) [regs] |
| 8 / 64 / 16 / split / 8192 (explicit only) | vectorized_transpose_hoisted / 32 / 8 / 1 | 1999.5 → 1650.4 | 3504.5 → 3153.6 (H = 10M: 388.1 → 398.5, slower) | −350.9 / 1.111× | / 3153–3156 | 75 (baseline T = 64: 79) / 0 / 70 KiB | 1 (8) [smem] |
| 64 / 16 / 4 / split / 8192 (control) | unchanged | 724.3 | 1752.2 | — | 1747–1755 | — | — |
| 64 / 64 / 16 / split / 8192 (control) | unchanged | 720.3 | 1763.3 | — | 1760–1766 | — | — |

**Memory and numerics:**

| Row | Memory (GiB) | Numerical category | Training |
|---|---|---|---|
| C = 8, S = 16 / 32, split, M = 8192 (selected) | 8.84 / 10.38 | same order | no drift beyond the null control |
| C = 8, S = 64, split (swizzle4_capped) | 8.84 / 10.38 | same order | no drift beyond the null control |
| C = 8, S = 16 / 32, BF16 gradients | 8.84 / 10.38 | same order | covered by the split-gradient runs |
| C = 8, S = 16 / 32, M = 1024 | 8.62 / — | same order | |
| C = 8, S = 16, R = 16, H = 10M | 1.13 / — | same order | covered (§9.5) |
| vectorized_hoisted | 8.84 / 10.38 | same order | no drift beyond the null control |
| feature_owner (candidate B) | 8.84 / 10.38 | MMA-accumulate order (≤ 2.8e-6 of scale) | no drift beyond the null control |
| warp-private S = 64 vector plan | 8.84 / — | same order | |
| C = 64 controls | 8.69 / 8.68 allocated | unchanged | |

**Answer to the question of the task.** Without changing connectivity, precision or the
low-memory streaming architecture, the C = 8 local dX accumulation loses about 21 % of the dX
kernel time at S = 16 / 32 and 12 % at S = 64, i.e. 7–12 % of the complete training step. That
is 62–64 % of what removing the accumulation entirely would give (ablation, H = 100M). The
remainder of the dX kernel is projection, epilogue and MMA work (§11).

## 14. Reproduction

```
cd sparse_pooling/benchmarks
./run_c8dx_screen.sh            # stage 2: dX microbenchmarks, all variants x tiles (H = 10M)
./run_c8dx_steps10M.sh          # stage 2: complete steps (H = 10M)
./run_c8dx_final_100M.sh        # stage 3: full scale, seeds, BF16, M = 1024, historical protocol
./run_c8dx_selected.sh          # selected plan (auto) vs explicit baseline
./run_c8dx_extra.sh             # S = 16 / R = 16 (H = 10M); C = 64 S = 16 / 64 / 128 (H = 100M)
./run_c8dx_ablation_100M.sh     # timing-only no-local-update ablation (wrong results)
python numerics_training.py --out ../results/c8_dx_scatter/training_validation_split
python numerics_scatter.py --out ../results/c8_dx_scatter/numerics_fixed_dy
python relu_boundary_case.py --out ../results/c8_dx_scatter/relu_boundary_case
python bank_model.py --out ../results/c8_dx_scatter/bank_model.json
python sass_scatter.py --kernel 'sgx_dx_kernel<1, 8, 4, 8, 1, 2, 9, 1>' --excerpt 40 --loop 1
./run_sanitizer_dx_scatter.sh $(which python)
python summarize_c8dx.py ../results/c8_dx_scatter/final_100M/*.json --csv ../results/c8_dx_scatter/final_100M.csv
```
