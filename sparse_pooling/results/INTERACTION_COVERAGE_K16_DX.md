# Interaction coverage at a fixed 200M budget, and K16 dX optimization

Branch `interaction-coverage-k16-dx` (from `topk16-cover-topk-confirmation` @ `0e4c009`); experimental, not merged.
Raw evidence: `results/interaction_coverage_k16_dx/` (all derived tables: `tables.md`, `csv/`).
Every quality number is a synthetic fine-tuning or synthetic diagnostic loss, not a downstream-accuracy claim; every
timing is one RTX 3090 and is not a claim about other GPUs.

## Answers

**Recommendation (architecture and backend chosen independently)**

* **Architecture: keep Cover-TopK/16.**
  * Neither new candidate earns a place on this task.
  * Balanced-TopK/16 and Cover-Mix32:p1 are both *noninferior* to Cover-TopK/16 within the preregistered 0.003 margin
    (Bonferroni over 4).
  * Both are nevertheless measurably worse at the point estimate, and every seed points the same way:
    * Balanced: +0.00094 (per seed +0.0006 to +0.0011).
    * Mix32: +0.00040 (per seed +0.00005 to +0.0007).
  * The ordering Cover < Mix32 < Balanced repeats on graphs 1 and 2.
  * Mix32 also costs a 1.33x complete step against Cover-TopK/16 under the optimized backend (787 vs 592 ms).
  * Cover-Mix32:p1 is a significant improvement over Mix-TopK (-0.00166, adjusted upper bound -0.00114), with
    about a quarter of Mix-TopK's per-output triple coverage. Neither beats plain Cover-TopK/16.
* **Backend: adopt the K16 dX tiling `mt4_nw8_wg4`.**
  * It qualified under every preregistered rule:
    * Cover-TopK/16 complete step at M 8192: 0.739x legacy (801.3 -> 592.3 ms, graph mean of 3).
    * BP-row/16: 0.739x.
    * Selected Mix32 complete step: 0.820x. Only its K16 component uses the variant.
    * M 512: 0.750x.
    * M 64 guard: 0.86x.
  * It computes the same function. The locked test losses of the legacy and optimized compact-backend training
    replays agree within 7.5e-7 per seed (primary contrast 4: noninferior, mean +1.4e-7).
    * Start states are identical in all 9 legacy / optimized pairs.
    * Only the dX kernel differs; the forward and dW kernels are the same code.
    * dX accumulates with FP32 global atomics, so the two paths can differ only by accumulation order.
  * The backend is independent of the architecture decision. It speeds up every K16 kgen path, including the K16
    component of the grouped mixtures.

**The six distinctions the spec asks for**

1. **Complete input coverage.**
   * All five mask candidates / controls keep every one of the 64 inputs in all 49,152 records (mask and numerical
     checks agree): Cover-TopK, Balanced, Cover-Mix32 p1 / p2, BP-row, and Mix-TopK.
   * U16@0.25 does not. It leaves only 47,040 records complete and has 3.96M structurally bias-only rows.
2. **Balanced input exposure.**
   * Balanced-TopK/16 achieves exactly uniform row-group multiplicity in full records (sd 0 vs 1.71 for Cover-TopK;
     neurons-per-input sd 0 vs 27.4).
   * Its cost is 2.1 % of the retained checkpoint energy: 0.3598 vs 0.3676 (Cover-TopK) and 0.2500 (BP-row).
   * The greedy repair sits 0.23 % (median, full records) / 0.28 % (tails) below the certified quota optimum.
     The maximum gap is 0.49 %.
   * Balanced exposure did **not** improve learning. Balanced is the worst of the three K16 candidates on the full
     task.
3. **Unique higher-order tuple coverage and multiplicity.**
   * Every support covers all layer pairs.
   * Unique per-output triple coverage:
     | support | triples % |
     |---|---:|
     | Cover-TopK/16 | 0.217 |
     | Balanced | 0.227 |
     | Mix32 p1 | 0.294 |
     | Mix32 p2 | 0.364 |
     | U16@0.25 | 0.549 |
     | Mix-TopK | 1.115 (identical to dense on these sampled queries) |
   * Unique quad coverage doubles from Cover-TopK to Mix32 p1 (0.0030 -> 0.0061 %).
   * The paired differences against Cover-TopK are all significant on common queries, with a joint CI on every graph.
     The one exception is Balanced quads on graph 2.
   * Tuple incidence sums (pair / triple / quad incidences per full record) are reported separately from unique
     coverage. They are 1920 / 8960 / 29120 for K16 and 2112 / 12352 / 59760 for Mix32 p1.
4. **Structural lower bound vs demonstrated attainable solutions (reduced raw-input task).**
   * The functional-ANOVA bound is validated: reconstruction error <= 3.6e-15, dense bound 0, order-1 bound 0.
   * At order 2 the trained nMSE after 4000 updates sits within 0.0018-0.0022 of the bound for all four continued
     supports. That residual is structural: the 1,211-1,532 of 4,096 order-2 terms with no eligible neuron
     (30-37 %) are the excluded interactions.
   * At orders 3 / 4 the gap is still 0.007-0.016 and shrank between 1000 and 4000 updates.
     * That part is not shown to be structural.
     * It is not assigned to optimization either: the bound is a bound, not an attainability proof.
   * Mix-TopK has bound 0, and its anchor oracle attains it (nMSE <= 8e-6 at orders 2-4; ~1e-14 against BF16-input
     targets). Its trained residual (0.044 / 0.061 / 0.066 at 1000 updates) is therefore attainable-but-unreached.
5. **Training outcomes.**
   * The full-task ranking (icdx39000, 6 seeds) is U16@0.25 0.16660 < Cover-TopK 0.16973 < Mix32 p1 0.17012 <
     Balanced 0.17067 < Mix-TopK 0.17178; dense is 0.11311.
   * The reduced task ranks differently: Mix-TopK < Balanced < Mix32 p2 < BP-row ~ Mix32 p1 < Cover-TopK << U16.
   * Coverage gains that help the reduced raw-input task do not carry over to the full task, which has a learned
     projection.
6. **Kernel performance at an identical function.**
   * See the backend recommendation above.
   * The gain is an occupancy effect, inferred from launch attributes, not from counters:
     * Legacy 74 KiB of dynamic shared memory fits 1 CTA / SM; mt4_nw8_wg4 (42 KiB) fits 2.
     * Registers (128 / thread) are unchanged.
     * Profiled dX dropped from 554 to 346 ms.
   * Hardware counters were unavailable (`ERR_NVGPUCTRPERM`), so no DRAM / L2 / stall claims are made.

**Point estimates vs uncertainty-qualified conclusions**

* *Uncertainty-qualified* (preregistered family, 6 seeds, Bonferroni over 4):
  * Balanced and Cover-Mix32:p1 are noninferior to Cover-TopK/16.
  * Cover-Mix32:p1 improves on Mix-TopK.
  * mt4_nw8_wg4 is noninferior to legacy (it is numerically the same function).
* *Point-estimate / descriptive only:*
  * Cover-TopK/16 is best among the K16 candidates; this is pointwise-significant but not in the family.
  * U16@0.25 is 0.003-0.004 better than every K16 candidate; this is exploratory, pointwise.
  * Every reduced-task ordering is descriptive.

**Full comparison vector** (graph 0; step = graph mean at M 8192, same session; test = icdx39000 mean of 6)

| method | valid slots | structural width (non-bias-only neurons) | records with all 64 inputs | groups / input sd (full) | per-output triples % | per-output quads % | step legacy ms | step mt4_nw8_wg4 ms | peak alloc GiB | test loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cover-TopK/16 | 200M | 12.50M | 49,152 | 1.709 | 0.217 | 0.0030 | 801.3 | **592.3** | 2.19 | 0.16973 |
| Balanced-TopK/16#0 | 200M | 12.50M | 49,152 | **0.000** | 0.227 | 0.0031 | 798.5 | 591.8 | 2.19 | 0.17067 |
| Cover-Mix32:p1#0 | 200M | 12.50M | 49,152 | 1.684 | 0.294 | 0.0061 | 959.4 | 786.9 | 2.21 | 0.17012 |
| U16@0.25 | 200M | 8.54M (3.96M bias-only) | 47,040 | 1.735 | 0.549 | 0.0229 | 752.2 (wide; no K16 kgen dX) | -- | 2.27 | **0.16660** |
| Mix-TopK#0 | 200M | 12.50M | 49,152 | 1.532 | 1.115 | 0.0669 | not re-timed (previous round: 1.24x Cover-TopK/16 legacy) | -- | -- | 0.17178 |
| dense | 800M (dense K64 envelope) | 12.5M | dense | -- | 1.115 | 0.0669 | 1081.8 | -- | 7.80 | 0.11311 |

Point-estimate frontier (test loss vs optimized step):
* Cover-TopK/16 + mt4_nw8_wg4 (592 ms, 0.1697) and U16@0.25 (752 ms, 0.1666) are the two non-dominated sparse
  points.
* Balanced, Mix32 and Mix-TopK are dominated on this task.
* U16's quality advantage comes with incomplete input coverage and 32 % bias-only width. It stays a tradeoff, not a
  recommendation.

**Decisions taken under the negative-outcome rules**

* Balanced is not kept: the exposure trade did not help.
* Mix32 is not kept: grouped overhead and slightly worse loss, despite 1.35x the triple coverage.
* Legacy dX is replaced, because a finalist qualified.
* The raw-input bound explains the order-2 residual, and the excluded interactions are identified. The higher-order
  gap is left as unresolved uncertainty.

## 1. Provenance, workload and protocol

* **Start:** the remote ref `topk16-cover-topk-confirmation` was verified at `0e4c00950a88dba71b894433a78cc73f7be05e56`
  (`git ls-remote`), and a clean worktree was created at that commit.
  * There is no departure from the starting point.
  * No AGENTS.md / CLAUDE.md exists in the repository; build / test / sanitizer guidance is the repository's own.
* **Workload:**
  * Shape: D = G = 1024, H = 12,500,000, K64 / C256 envelope, S64 / R256, total Q 4096.
  * Records: 49,152 original records (48,128 full, 1,024 tails of 175 / 176 neurons); 781,312 valid row groups.
  * Budget: every candidate has exactly 200,000,000 valid weight slots and 12,500,000 biases. This is checked in every
    construction / support record.
  * Pooling: Mean(ReLU) with the original per-output denominator.
  * Batch: performance M 8192, learning M 512.
  * Precision and optimizer: BF16 inputs / parameters / moments, FP32 accumulation / main_grad, split dZ, stochastic
    rounding, no L1, wd 0.
  * Checkpoints: the actual alpha-0 checkpoints of graphs 0 / 1 / 2 with their projection / head. Every job verifies
    their sha256.
* **Frozen checkouts / results:**
  * Development ran in `../GranuleLayer-icdx`.
  * Every measurement job ran from the frozen checkout `../GranuleLayer-icdx-run` @ `195a29b` (code commits `42c8d0d`, `195a29b`; `stage{1,2,3}_commit.txt` record a clean tree).
  * The raw results root is `../icdx_runs`; large states are kept outside git.
  * The stage drivers (`stage1.sh` ... `stage4.sh`) ran as detached `systemd-run --user` services, strictly serialized
    on the GPU.
  * The profiler, sanitizer and launch counts never overlapped a timing job.
* **Preregistration:**
  * `selection_rules.json` was written by the stage-1 driver before any learning, coverage or timing job. It embeds
    the manifest and the seed collision check: 39 pairs of seed sets, 0 collisions, and the old protocols unchanged.
  * All selections are mechanical (`icdx_select.py`). Each file records its rule, its input records (sha256) and its
    time.
  * `final_selection.json` was frozen at 13:08:22, before any icdx39000 / icdx89000 test was opened.

## 2. Errata and reporting contracts

### 2.1 Records are paired by their seed keys

* `benchmarks/paired_stats.py` (new) indexes every observation by an explicit `ObsKey`. The key fields are:
  * graph, data seed, SR seed, support identity, treatment;
  * test protocol, update count, backend variant (dX).
* A `Cohort`:
  * requires the declared seed set exactly once;
  * rejects duplicate keys (`DuplicateKey`) and wrong protocol / identity (`WrongIdentity`);
  * joins observations by key, never with `zip`;
  * derives n and df from the validated cohort, and keeps the preregistered family size and alpha.
* An incomplete comparison yields an inventory plus descriptive rows marked `not estimable`, with no verdict.
* Tests (`tests/test_paired_stats.py`, 11) cover:
  * different missing seeds on opposite sides, and the same missing seed;
  * duplicates, wrong protocol, and reordered files.
* `tkc_report.py` and the new `icdx_report.py` both use the keyed pairing.
* **Reproduction:** the previous round's 21 tables were regenerated from `../tkc_runs` with the hardened generator.
  * They are byte-identical to the committed `0e4c009` tables, including the 66-record final table and its three
    primary intervals.
  * Evidence: `interaction_coverage_k16_dx/errata_tkc_reproduction.txt`.
  * The previous complete cohort was correctly paired. The fix concerns future missing-file handling only.

### 2.2 The coverage repair synchronizes on the GPU

* The previous claim of "no per-swap host synchronization" for `cover_repair` was wrong.
  * On a CUDA device, every repair round reads a tensor-derived stop condition and selects the active records
    dynamically.
  * Each of those is a host / device synchronization.
  * The docstring has been corrected (`experimental/local_supports.py`).
* GPU construction wall times in this report include all synchronizations (`torch.cuda.synchronize` before the clock
  stops).
* The new quota repair (`quota_repair`) uses masked rounds over all records and reads the stop condition only every
  `check_every` = 8 rounds.
  * Its host syncs are counted: 18 per full-model Balanced build, at 64 rounds.
* The Mix32 cover repair keeps the per-round synchronization ("per round" in the construction table).
* `working_buffer_bytes` is an accounting estimate. The actual host RSS growth and device peak are measured and
  reported instead.

### 2.3 Preserved contracts

* Coverage intervals:
  * Shared-query (joint) intervals with the finite-population correction (`joint_coverage.py`).
  * Selected-output means are labelled CONDITIONAL on the 38 fixed outputs.
  * Paired coverage differences use common queries, and the query fingerprint is retained in every coverage record.
* Launch counts:
  * Backend calls, CUDA kernels and memset / memcpy are counted separately.
  * A trace with a fractional count or a missing expected layer kernel is labelled INCOMPLETE (lower bounds).
  * This happened once: the Balanced-TopK/16 legacy trace recorded 11 of the 12 expected `FillFunctor` kernels in 4
    steps. Its layer kernels are complete and identical to Cover-TopK/16.
* The SIGINT root cause, disposition tests and bounded cleanup are unchanged. The final suite again runs with
  `env --default-signal=INT` (section 10).
* Deterministic support construction does not make training bitwise deterministic. The grouped forward and the dX
  path use atomics, and the tests compare them with tolerances (dW is bitwise).
* The original support / RNG namespaces and fingerprints of every existing policy are preserved:
  * The restored Cover-TopK/16 masks equal the previous round's on all three graphs.
  * Tests pin the old fingerprints.
  * Mix32 uses its own namespace (`"Cover-Mix32", "p{p}", graph, layout, "class"`).

## 3. Candidates and construction

* **Balanced-TopK/16** (`local_supports.balanced_quotas` / `quota_repair`):
  * Rows: every row sums to exactly 16.
  * Columns: column sums equal the near-uniform quotas, with b = floor(16m/64) and h = 16m - 64b. The inputs at the
    first h positions of a keyed uniform permutation of the record's 64 coordinates get b + 1, the rest b.
    * The permutation is weight-independent.
    * The quota layout seed is the `#ls` suffix; layouts 0 and 1 were built.
  * Start point: the Cover-TopK/16 mask.
  * Repair: a greedy minimum-energy-loss swap, applied lexicographically by (j, u, d) until every column meets its
    quota.
  * Result: every record is complete (each input appears in >= 1 row group), since q_j >= 1.
* **Cover-Mix32:p1 / p2** (`experimental/mix32_supports.py`):
  * It reallocates the K16 budget to a K32 / K16 / K8 class mix, giving the same 200M slots and 12.5M neurons.
    * p1: 786,432 / 10,140,704 / 1,572,864 neurons.
    * p2: 1,572,864 / 7,781,408 / 3,145,728.
  * Classes: in every record, p full row groups become K32 and 2p become K8; the rest stay K16.
    * The groups are chosen by a keyed permutation (weight-independent).
    * Records with < 3p full groups are rejected; the budget is never changed.
  * Supports are per-row TopK by checkpoint energy, followed by variable-degree cover repair.
  * It executes as a grouped kgen layer with a declared class list.
* Construction uses checkpoint weights and seeded layout choices only. Targets, target tuples and test losses never
  enter.
* Cost (construction table):
  * Every builder meets the <= 30 s target.
    * Balanced: 17-19 s CPU + 0.6 s GPU repair.
    * Mix32: 2.2 s CPU + 0.06 s GPU.
    * Cover-TopK/16: 2.3 s.
  * GPU repairs equal the CPU result and are independent of chunking.
  * Host RSS growth peaks at 2.5 GiB.
  * The graph-2 Balanced CPU time (18.6 s) overlapped a nice-19 nvcc compile and is the slowest sample for that
    reason.
* **Quota optimality:** on a stratified sample of 64 full + 64 tail records, a MILP certified the exact quota optimum
  for all 128 records.
  * The min-cost-flow solution agrees exactly.
  * The repair is feasible everywhere but never equals the optimum mask.
  * Its energy gap is 0.23 / 0.28 % (median), 0.42 / 0.49 % (max).
  * Ratio of Balanced to Cover-TopK energy: 0.979 (full) / 0.968 (tail).
  * Ratio of Balanced to BP-row energy: 1.44.
* These are magnitude-dependent policies, so uniform-BP analytical predictions do not apply to them. Measured
  coverage and the combinatorial ceilings below are reported instead.

## 4. Coverage, exposure and tuples (tables: *Coverage*, *Inventory*, *Combinatorial ceilings*, *Paired coverage*)

* **Complete coverage vs exposure.**
  * Complete coverage is identical for all mask candidates (49,152 / 49,152 records).
  * Exposure is not:
    * Balanced: group multiplicity per input is exactly 4 in every full record (2-3 in tails).
    * Cover-TopK / Mix32: multiplicity spreads 1-13, sd ~1.7, neurons-per-input CV 0.42-0.43.
    * BP-row: near-uniform (sd 0.21) but random.
* **Unique tuples.**
  * Record-scope pairs / triples (full records): the fraction of the record's C(64, t) local tuples held together by at
    least one row group.
    | support | pairs % | triples % |
    |---|---:|---:|
    | Cover-TopK | 62.8 | 19.5 |
    | Balanced | 70.6 | 20.4 |
    | Mix32 p1 | 67.3 | 26.4 |
    | Mix32 p2 | 71.2 | 32.6 |
  * Balanced raises pair coverage most, by spreading inputs. Mix32 raises triples / quads most, through its K32 rows.
* **Incidence vs unique coverage.**
  * Per-output union bounds (a ceiling, not a prediction) are 17.5 / 0.24 / 0.003 % at t = 2 / 3 / 4 for uniform K16.
  * The measured per-output coverage reaches 11-12 % / 0.22 % / 0.003 % against those bounds.
  * Higher-order shortfall is a fan-in limit, not a repair defect.
* **Sampled-quad multiplicity.**
  * K16 supports never hold a sampled quad in >= 16 row groups.
  * Mix-TopK holds every covered quad in >= 2 groups (mean 12.7).
  * U16 has 34,803 quads in >= 16 groups.

## 5. Development and selections (tables: *Continuity*, *Development*)

* Continuity (32 updates, seed 0): every control reproduced its earlier record.
  * Same mask and same checkpoint.
  * Maximum held-out difference 1e-7.
  * The earlier 256-update records of Cover-TopK/16 and Mix-TopK were reused; nothing was rerun.
* Treatments: energy normalization was selected for Balanced and both Mix32 (`none` - `energy` > 1e-4 in each case).
* The mixed candidate is Cover-Mix32:p1 (0.17194 vs p2 0.17249).
* All selection files and their rules are in `interaction_coverage_k16_dx/raw/` (`*_selection.json`, `*_select.out`).

## 6. K16 dX optimization (tables: *Profile*, *dX tiling screen*, *Backend confirmation*, *Shape guards*, *Memory*)

* **Variants.** `kernels/cuda/experimental/kgen/kgen_k16_dxv.cu` implements the legacy-algorithm dX kernel, templated
  on:
  * rows per thread MT in {2, 4};
  * warps NW in {4, 8};
  * a runtime work-group factor WG.
* Dynamic shared memory is align16(2D) + 2·S·BM + 4·(NW/WG)·S·BM.
* Eight tilings were screened (legacy + 7).
* Selection uses `experimental/kgen_dx_variants.resolve` (legacy / auto / explicit, with cache keys). The `TUNED`
  table is empty, so nothing is auto-selected without a recorded decision.
* (MT4, NW8) is the same kernel symbol as legacy.
  * The variant path only ever *raises* the kernel's dynamic-shared-memory attribute, and checks it on every launch.
  * A regression test covers interleaved legacy / variant launches. That bug caused a CUDA error during development.
* **Resources** (ptxas `-v`, `ptxas_dxv.txt`; occupancy API):
  * legacy: 74 KiB, 128 regs, 1 CTA / SM;
  * mt4_nw8_wg4: 42 KiB, 128 regs, 2 CTAs / SM;
  * mt2 variants: 96-98 regs, no spills.
* **Profiler:**
  * Nsight Compute / Systems 2023.1 are installed, but GPU performance counters are denied (`ERR_NVGPUCTRPERM`,
    `RmProfilingAdminOnly 1`, no admin rights).
  * Kernel durations therefore come from the CUPTI activity trace. DRAM / L2, tensor utilization, shared-memory / CAS
    behaviour, warp stalls and instruction mix are **not measured**.
  * The staged-X and atomic volumes in the profile table are analytical estimates, and labelled as such.
  * No physical-C16 / targeted-change implementation was added. The only cost identified from the available evidence
    (occupancy limited by shared memory) is exactly what the tiling screen addresses. Without counters, a further
    hypothesis could not be measured.
* **Screen** (graph 0, 8 samples per process):
  * All 7 tilings were eligible.
  * The finalists were the two with the lowest Cover-TopK/16 M 8192 complete step: mt4_nw8_wg4 0.740 and mt4_nw8_wg8
    0.747.
  * The dX correctness gate (`tests/test_kgen_dx_variants.py`, 7 tests) passed before the screen.
* **Confirmation** (graphs 0 / 1 / 2, 24 samples per process, same-session legacy ratio):
  * Both finalists qualified, and mt4_nw8_wg4 was chosen (largest Cover gain).
  * The guards and three session replicates are stable: 590.3-591.2 ms vs legacy 797.4-799.3 ms.
* Memory is unchanged by the variant: peak allocation is 2.19 GiB, and only the per-launch shared-memory footprint
  changes.
* The timing factories apply no row-energy normalization (`row_energy_normalization = "none"` in every timing record),
  identically for legacy and variant.

## 7. Final locked cohort icdx39000 (tables: *New locked test*, *Primary contrasts*, *Exploratory*, *Compact-backend replay*, *Graph checks*)

* Six seeds (12..17), graph 0, 256 updates, test batches 39,000-39,007. Every method's cohort is complete.
* Rows 1-6 are dense-masked training. Rows 7 / 8 are the compact Cover-TopK/16 backend (indexed kgen) with legacy /
  mt4_nw8_wg4 dX, replayed from identical start states.
* Primary family (Bonferroni over 4, one-sided adjusted upper bound, margin 0.003):

| contrast | mean | adjusted upper | decision |
|---|---:|---:|---|
| Balanced-TopK/16#0 - Cover-TopK/16 | +0.000945 | +0.001153 | noninferior |
| Cover-Mix32:p1#0 - Cover-TopK/16 | +0.000395 | +0.000682 | noninferior |
| Cover-Mix32:p1#0 - Mix-TopK#0 | -0.001662 | -0.001141 | **improvement** |
| mt4_nw8_wg4 - legacy (compact Cover-TopK/16) | +0.00000014 | +0.000001 | noninferior |

* Pointwise (descriptive) intervals:
  * Balanced and Mix32 are *worse* than Cover-TopK/16: [+0.00078, +0.00111] and [+0.00016, +0.00063].
  * Noninferiority within 0.003 is the confirmatory statement. "Better" is not supported.
* Row 7 vs row 1 (compact legacy vs dense-masked Cover-TopK/16) is a secondary lowering / trajectory check, not an
  identical rounding trajectory: mean +1.8e-6, max |difference| 3.9e-6 over the 6 seeds.
* The grouped Mix32 K16-component check (32 updates, development seeds 0 / 1 / 2) gives identical held-out losses for
  legacy and variant. It is a short validation, not full-cohort quality confirmation.
* Graph checks (seed 12): Cover < Mix32 < Balanced on all three graphs.
* Complete final states are kept outside git, at `../icdx_runs/final/state`. Their sha256 are in `states.sha256`, and
  each record carries its own state sha.

## 8. Reduced raw-input task and the functional-ANOVA bound (tables: *Reduced raw-input task*, *Functional-ANOVA*, *Training minus bound*)

* Task: H = 262,144, one C256 record per output, raw Gaussian inputs, a per-output affine readout, and T = 16 terms per
  output.
  * Its interpretation is separate from the full task, which has a learned projection.
* Runs:
  * Three init / SR pairs (30, 0), (31, 1), (32, 2) were trained for 1000 updates.
  * The continued supports (Cover-TopK/16, Balanced, Cover-Mix32:p1, BP-row) were resumed to 4000.
  * Resume identity is checked: uninterrupted vs resumed 20 + 20 updates are bitwise equal, and a mismatched state is
    refused.
* Test: icdx89000 was evaluated once per final state, after the freeze. These results are descriptive and not in the
  confirmatory family.
* BP-row's three runs differ only in init / SR. Its support is a fixed random layout, so its ANOVA bound has zero spread
  and its losses vary in the 4th decimal.
* **Bound.** L_g is the energy of the functional-ANOVA components of t_g that no available support contains. It lower
  bounds the nMSE of any function of the supports.
  * The 32-output sample per order was declared by seed before any loss was examined. Means are over that sample.
  * Validation:
    * reconstruction error <= 3.6e-15;
    * the order-1 Monte Carlo bound agrees with the analytic floor;
    * dense and Mix-TopK give 0.
  * Order-2 bounds:
    | support | bound |
    |---|---:|
    | Cover-TopK/16 | 0.073 |
    | Mix32 p1 | 0.065 |
    | BP-row | 0.061 |
    | Mix32 p2 | 0.060 |
    | Balanced | 0.057 |
* **Interpretation.**
  * At order 2, the 4000-update models are within ~0.002 of the bound. The remaining error is the structural loss of
    the uncovered pairs: 1,211 (Balanced) to 1,532 (Cover-TopK) of 4,096 order-2 terms have no eligible neuron.
  * At orders 3 / 4, the gap above the bound (0.007-0.016) is still shrinking with training. Finite ReLU
    representability, the readout and BF16 inputs all contribute. It is reported as unresolved, not as pure
    optimization error.
  * The Mix-TopK anchor oracle shows that its zero bound is attainable.
* Term-neuron maximum matching (`eligibility`):
  * Only dense and Mix-TopK match every term.
  * U16 matches every term on 162 / 61 / 17 outputs.
  * No K16 support matches every term on more than 2 outputs.

## 9. Launches and memory (tables: *Launches per complete training step*, *Memory*)

* Kernels and backend calls per complete training step:
  | support | CUDA kernels | backend calls (fwd / bwd) |
  |---|---:|---:|
  | Cover-TopK/16, BP-row | 11 | 1 / 2 |
  | grouped Mix32, Mix-TopK | 17 | 3 / 6 |
  | U16 | 15 | -- |
  | dense | 11 | -- |
* Every support also has 2 memset / memcpy per step.
* The variant does not change launch counts.
* Because (MT4, NW8) shares the legacy symbol, the profiler names cannot distinguish the variants. The executed variant
  is taken from the layer's `dx_variant` record.
* Index, map and metadata storage is 65-68 MiB for the K16 / Mix32 formats, against 393 MiB for U16 wide.

## 10. Correctness gates

* **Compute Sanitizer** (`stage3.sh`; compute-sanitizer 2023.1.0, CUDA 12.1; logs in `raw/sanitizer/`).
  * Tools: memcheck, racecheck, initcheck and synccheck.
  * Workloads:
    * `icdx`: every compiled K16 dX tiling (legacy + 7) on the indexed Balanced / Cover-TopK path, including
      partial-group tails, D 1024 / 256, and token tails B 8 / 40 / 136;
    * also in `icdx`: the grouped Cover-Mix32 p1 / p2 (K32 / K16 / K8) layer with legacy and variant dX, non-finite
      upstream gradients, main_grad accumulation and the L1AdamW update;
    * `kgen`, `tkc`, `kwa` and `rsn`: the existing workloads.
  * Result: **20 / 20 runs PASS**. Every summary reads `0 errors`, and racecheck reports `0 hazards`.
  * The only exclusion is the existing, documented one: racecheck does not check the Triton `bwd_dense_kernel`.
* **Full suite:** `env --default-signal=INT python -m pytest -q tests` from the frozen checkout gives **3276 passed,
  32 skipped** (8 min 3 s; `raw/full_suite.txt`).
  * The previous round had 3209 passed / 32 skipped.
  * The new tests are paired statistics (11), interaction coverage (33), dX variants (7) and the ANOVA bound (16).
* **dX correctness gate** before the screen: 7 / 7 (`raw/dx_correctness.txt`). The tests compare against the dense
  masked function and FP64 gradients, and cover:
  * launch legality and shape guards;
  * the attribute-interleave regression;
  * restore / state correctness.
* `stage{1,2,3}_commit.txt`: all at `195a29b` with a clean tree.

## 11. Cross-session timing check

The previous round reported Cover-TopK/16 legacy at 875.2 ms and dense at 1566.9 ms (M 8192, graph 0). This round
measured 801.3 ms and 1081.8 ms. `stage4.sh` ran both rounds' frozen checkouts in one session, alternating twice, with
each checkout's own job definitions:

| checkout | rep | Cover-TopK/16 step ms | dX ms | dense step ms (3 anchors) |
|---|---:|---:|---:|---:|
| previous (`../GranuleLayer-tkc-run`) | 1 | 795.1 | 554.0 | 1070.8 / 1073.0 / 1073.8 |
| current (`195a29b`) | 1 | 796.1 | 556.5 | 1075.1 / 1075.5 / 1075.5 |
| previous | 2 | 798.0 | 557.3 | 1076.0 / 1077.9 / 1076.8 |
| current | 2 | 796.8 | 557.1 | 1077.0 / 1077.8 / 1075.3 |

* In one session the two checkouts agree within 0.4 %. **The absolute shift is not a code change.**
* The previous round's session was slower (dense +45 %, Cover-TopK +9 %). This was not caused by the logged state:
  * SM clocks were higher in the previous round (1700-1860 MHz vs 1600-1785 MHz now).
  * Power (~342 W of 370 W) and temperatures (38-50 C) were the same.
* The cause is not identified.
* This confirms the spec's rule: only same-session ratios are compared, and previous absolute milliseconds are planning
  references.
* Raw records are in `raw/cross_session_{prev,cur}{1,2}/`, with `nvidia-smi` in `raw/cross_session_nvidia_smi.txt`.

## 12. Limitations

* One GPU (RTX 3090) and synthetic tasks only. There is no downstream-accuracy or other-GPU claim.
* Hardware counters were unavailable. The dX explanation (occupancy) rests on launch attributes and timings only.
* Coverage intervals over selected outputs are conditional on those outputs.
* The reduced-task intervals describe three init / SR replicates, not redrawn data distributions.
* The ANOVA bound covers a 32-output sample per order, not the all-output locked metric. It excludes BF16
  quantization, and it is not an attainability proof for the sparse supports.
* Mix-TopK and U16 were not re-timed with the variant. U16's wide format has no K16 kgen dX.
* Mix-TopK's speed comes from the previous round's same-session ratio only.
* The previous round's absolute milliseconds are not comparable to this round's (section 11). Only same-session ratios are used.
* Dev-scratch observations made before the freeze (`../icdx_dev`) are not evidence of record.

## 13. Reproduction

From `sparse_pooling/benchmarks` of the frozen checkout, with `GRANULE_RESULTS_ROOT` set:

```
bash ../results/interaction_coverage_k16_dx/stage1.sh   # rules, dX gate, construction, profile, screen, coverage,
                                                        # development, reduced task, ANOVA, confirmation, selections, freeze
bash ../results/interaction_coverage_k16_dx/stage2.sh   # final cohort, graph checks, icdx89000 eval, 4000-update continuation, launches
bash ../results/interaction_coverage_k16_dx/stage3.sh   # compute-sanitizer (icdx kgen tkc kwa rsn) + full suite
bash ../results/interaction_coverage_k16_dx/stage4.sh   # cross-session check against ../GranuleLayer-tkc-run
CUDA_VISIBLE_DEVICES="" python icdx_report.py --root $GRANULE_RESULTS_ROOT --out ../results/interaction_coverage_k16_dx
```

* Each stage is `icdx_jobs.py <stage> --out <dir> [--selection <file>] --resume`.
* Each stage writes a `plan.json`, a `manifest.json` and a per-job `status.jsonl` (measured / failed).
* Single jobs can be rerun by tag.
* Large states are kept in `../icdx_runs/{final,interaction,interaction_cont}/state` and
  `../icdx_runs/interaction/resume_check`, along with the `.analysis.npz` evaluation dumps. Their sha256 are in
  `interaction_coverage_k16_dx/states.sha256`.

