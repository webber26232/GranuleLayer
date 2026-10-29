# TopK/16 confirmation and coverage-constrained TopK

Branch `topk16-cover-topk-confirmation` (from `k8-weighted-supports-anchors` @ `b18b7a6`); experimental, not merged.
Raw evidence: `results/topk16_coverage_constrained_topk/` (all derived tables: `tables.md`, `csv/`).
Every quality number below is a synthetic fine-tuning or synthetic diagnostic loss, not a downstream-accuracy claim.

## 0. Answers

**1. Where does TopK/16 land?** Same-session timing over graph seeds 0 / 1 / 2 (24 samples per process) and a NEW
locked test (`tkc29000`, batches 29,000..29,007, six fresh data / SR seeds 6..11, never used before):

| 200M-slot point | complete step ms (graph mean ± 95 % CI) | ratio to U16 | test loss (mean ± pointwise 95 %) |
|---|---:|---:|---:|
| TopK/16 indexed K16 | 877.0 ± 3.9 | 1.065 ± 0.002 | 0.1698 ± 0.0005 |
| U16 0.25 wide | 823.7 ± 2.3 | 1 | 0.1676 ± 0.0005 |
| A8 0.25 wide | 987.5 ± 2.5 | 1.199 ± 0.003 | 0.1762 ± 0.0003 |
| BP-row/16 indexed (layout 0 / 1) | 877.0 ± 2.3 | 1.065 ± 0.001 | 0.1775 ± 0.0010 (layout 0) |

* TopK/16 is 6.5 % slower than U16, the same speed as BP-row/16, and 11 % faster than A8.
* **Primary decision TopK/16 − U16: inconclusive.** Mean +0.00216; the Bonferroni-adjusted one-sided upper bound
  is +0.00307, just above the 0.003 margin, so noninferiority is NOT established. Descriptively (pointwise 95 %
  [+0.0014, +0.0030], all six seeds positive) U16 trains better by about 0.002.
* Secondary: TopK/16 beats A8 by 0.0064 [0.0058, 0.0071] and BP-row/16 by 0.0077 [0.0069, 0.0085].
* TopK/16 keeps full width (no structural bias-only rows; U16 has 31.7 %) but leaves 0.65 inputs per original record
  uncovered (half of the records incomplete).
* The previous round's development mean (0.1713 ± 0.0042) is reproduced by the new test (0.1698; the test batches
  are a different sample, validation of the same runs 0.1710).

**2. Can fixed-fan-in repair eliminate missing inputs while keeping quality and energy?** Yes for inputs, at almost no
cost. Cover-TopK keeps exactly k inputs per valid row group (exactly 100M / 200M slots) and makes every original
record use all 64 inputs (49,152 / 49,152 records, mask AND numerical; all 1,024 outputs equal the dense feature
union).

* Energy: it loses 0.39 % (k8) / 0.010 % (k16) of TopK's retained pre-pruning energy. The exact constrained optimum
  (HiGHS MILP, certified integral, flow cross-check) is within 0.019 % median / 0.18 % p95 (k8) and 0 / 0 (k16) of
  the repair on a stratified 478-record sample, so the fast repair is near-optimal.
* Quality (primary, Bonferroni): Cover-TopK/8 − TopK/8 = +0.00039 (upper +0.00047): noninferior, and descriptively
  slightly worse ([+0.0003, +0.0005], every seed and every graph). Cover-TopK/16 − TopK/16 = −0.00001 (upper
  +0.00001): noninferior (identical to 1e-5).
* Cost: one-time construction 3.9 s (k8) / 2.3 s (k16) on the CPU for the whole model (repair alone 2.7 / 0.8 s;
  0.19 / 0.09 s if the same repair runs on the GPU, bitwise identical), plus about 0.25 s importance. It is
  invisible in the step: 633.4 vs 633.5 ms (k8), 875.2 vs 877.0 ms (k16), same kernels, same index bytes.

**3. Does the raw-input linear diagnostic agree with the missing-input floor?** Yes. On the reduced task (one record
per output) TopK/8 leaves 7.5 of 64 inputs per record uncovered and 1.25 of each order-1 output's ~10 distinct
target inputs. The derived floor sum_{missing} beta^2 / E[t^2] is 0.116 (three inits; independent Monte Carlo agrees
within 0.5 %). TopK/8's order-1 test nMSE is 0.126 at 1,000 updates and 0.118 at 4,000 (floor gap 0.010 → 0.002):
the input holes explain the plateau. Cover-TopK/8 has floor 0 and reaches 0.0119 → 0.0006, the same as BP-row/8
(0.0111 → 0.0006). TopK/16: floor 0.0096, nMSE 0.0153, Cover-TopK/16 0.0057 (BP-row/16 0.0051).
Unresolved: orders 2-4. With complete inputs the explained interaction stays far below dense (k8 0.16 / 0.13 / 0.12
at 1,000 updates; k16 0.61 / 0.51 / 0.48; dense 0.99): ~79 % (k8) and ~36 % (k16, order 2) of the higher-order terms
have no neuron whose support contains all of their inputs, and Cover-TopK remains ~0.02 below BP-row at both k. That
residual is a support / multiplicity / optimization mixture; no tight model-class optimum is known.

**4. Higher-order coverage despite complete input coverage.** Almost unchanged. Paired on the same 2M / 10M queries,
Cover-TopK raises per-output triple coverage by +0.0002 pp (k8) / +0.0004 pp (k16) and exact layer triples by
+0.055 / +0.014 pp. Pairs are 100 % for every K8 / K16 policy. Fixed fan-in still caps per-output triples at the
union bound (k8 0.02395 %: TopK 0.0238, Cover 0.0238-0.0240, BP 0.0240); BP-row/16 keeps slightly more tuples than
Cover-TopK/16 (−0.0035 pp per output, −0.14 pp layer triples). Complete input coverage is not interaction coverage.

**5. Frontier** (§9; point estimates on the new test / this session, uncertainty separate):

* 100M: TopK/8 (best K8-indexed quality), Cover-TopK/8 (same speed, complete input coverage, +0.0004 loss), BP-row/8
  (most tuple coverage among the fixed-k8 policies, worst loss), U8 (best loss 0.2108 and 37.6 % layer triples, but
  10 % slower and 34.4 % structurally bias-only). None dominates another over the full vector; TopK/8 vs Cover-TopK/8
  is a coverage-for-0.0004 trade.
* 200M: U16 (fastest, best loss, but 31.7 % bias-only and incomplete inputs), Cover-TopK/16 (full width, complete
  inputs, 0.1698), BP-row/16 (more tuples than Cover-TopK/16, worse loss), A8 and Mix-TopK (dense-envelope coverage;
  A8 faster, Mix-TopK better loss). TopK/16 is point-estimate-dominated by Cover-TopK/16 (equal loss to 1e-5, equal
  time within noise, more input coverage); "equal" here means noninferior within the margin, not equal.
* Historical (previous round, batches 19,000..19,007): kept in §9 for context only, never in a paired claim here.

Separate findings: input coverage is solved by the repair; interaction coverage is bounded by fixed fan-in and is
not changed by it; representability of higher-order terms is not established; the TopK/8 order-1 plateau is a
support (missing-input) effect; backend efficiency is unchanged (no kernel work).

## 1. Workload, provenance and protocol

* **Start:** `k8-weighted-supports-anchors` @ `b18b7a6` (worktree at that exact commit, its code and evidence).
  No later changes existed on that branch. No AGENTS.md / CLAUDE.md exists in the repository; build / test guidance
  is the repository's own (`docs/`, `run_sanitizer.sh`, pytest).
* **Model:** D = G = 1024, K64 / C256 envelope, S64 / R256, total Q 4096, H = 12,500,000 valid neurons,
  W0 = 800,000,000. M 8192, Mean(ReLU), BF16 inputs / parameters / moments, FP32 accumulation / main_grad, split dZ,
  optimizer race fix, stochastic rounding on, no L1, weight decay 0. The actual alpha-0 checkpoints
  `C256_R256_prox_a0_g{0,1,2}_step128.pt` (weight / bias sha256 verified by every job) with their projection / head.
  Nothing is redrawn at low K: every mask keeps the checkpoint's values at its enabled slots.
* **Budgets:** k8 = exactly 100,000,000 valid slots, k16 = exactly 200,000,000 (checked in every support record and
  timing record). Bias count 12.5M and the original pooling denominator n_g are unchanged.
* **Code:** support construction / accounting only. No CUDA kernel changed; Cover-TopK runs on the unchanged indexed
  kgen K8 / K16, C16 path (`kgx_onetile.lower_local`). Production defaults and auto-dispatch are untouched.
* **Frozen checkouts:** `../GranuleLayer-tkc-meas1` @ `637b97c` (only the first, superseded stage-1 construct
  jobs), `../GranuleLayer-tkc-run` @ `fcf6925` (every timing / coverage / learning / interaction job) and
  `../GranuleLayer-tkc-final` @ `9687196` (launch counts v2, sanitizer, runner tests, full suite). Source-clean
  (`provenance.source_dirty = False` in every record). One library build per bundle across all timing records:
  production `e55eb569423be99b2776`, kgen `023754cd60595afae7e9`, kgx (wide / tile kernels)
  `c400495d8202e34be70f`. Raw results root `../tkc_runs` (outside git).
* **Timing protocol:** `bench_supergroups.py --canonical-start --identity` in a fresh process per job. Warm-up /
  compile, restore of the pre-warm-up values, reset of the complete optimizer / main_grad / rounding state and a
  canonical-identity check before every measured round; inside a round the 2 reps x 3 steps run sequentially
  WITHOUT weight resets (the same boundaries for every method). Screen 8 samples, confirmation 4 x 2 x 3 = 24
  samples per process. One session per graph seed, dense first / middle / last, shuffled body
  (order seed 20261029 + 10 + g). The CPU-load runner stress tests finished before the first timing job
  (`stage1_driver.out`: "stress matrix finished"); nothing else ran on the GPU during timing. Clocks / power /
  temperature per round are in every record (`gpu_rounds`): SM 1680-1965 MHz (median ~1920), 328-345 W of 370 W,
  38-51 C.
* **Preregistration:** `selection_rules.json` (10:15, before any development run and before any timing / coverage of
  Cover-TopK) and `selection_rules_addendum.json` (10:22, before any learning run; corrects a time estimate, decision
  unchanged, §3.4). Already seen when writing: the graph-0 construction statistics and record sample of the pilot,
  the reduced-task floor at init 20 (no training), and the previous round's values. All selections are mechanical
  (`tkc_select.py`) and record their inputs.
* **Test sets:** the NEW locked test `tkc29000` (batches 29,000..29,007, M 512, same teacher) and new interaction-test
  batches `tkc69000` (69,000..69,007), both reserved before any screen through versioned driver arguments
  (`rsn_learn.py --test-protocol`, `rsn_interaction.py --test-protocol`; the old protocols `kwa19000` / `kwa59000`
  stay the defaults so old commands reproduce old records). Collision checks against every train / calibration /
  validation / probe seed (data seeds 0..11, 256 updates) and the other protocols' seeds are recorded in each final
  record. Batches 19,000..19,007 are historical evidence here, not a locked test.
* **Restart (disclosed):** stage 1 was stopped once while waiting (before any timing / learning job) to add the
  learned-projection statistics of §6 to `rsn_learn.py`; the completed construct jobs kept their `637b97c` records
  (`stage1_driver_first.out`) and were re-measured in quiet conditions at `fcf6925` (`construct_quiet/`, the numbers
  used here). The first construct jobs had overlapped the CPU-light part of the runner stress matrix.

## 2. Corrections to the previous round

### 2.1 Coverage intervals must respect shared queries

**Error.** `kwa_report.cov_row()` (at `b18b7a6`) summed the 38 selected outputs' triple hits over 38 x 2M
"trials" and applied Clopper-Pearson. All outputs use the SAME 2M sampled triples; output-by-query observations are
not independent.

**Fix** (`benchmarks/joint_coverage.py`, `wps_coverage.output_rows(joint=...)`, `rsn_coverage.py` schema /2). For
every sampled tuple i the pass now accumulates c_i = number of selected outputs covering it (int16 on the GPU; the
per-query arrays are kept in `cov_*.joint.npz`, uint8, ~50 kB compressed). The selected-output mean coverage is
mean_i(c_i / O) with the query-level SE.

* **Sampling unit:** `kcsr_coverage.sample_tuples` draws DISTINCT uniform ranks (unique of draws, then a random
  permutation, first n): a simple random sample without replacement from N = C(1024, t). The SE carries the
  finite-population factor 1 − n / N (0.9888 for 2M triples, 0.99978 for 10M quads).
* **Bootstrap check:** the query-cluster bootstrap (resample queries with all their outputs) is exactly a
  multinomial resampling of the c_i histogram. It agrees with the analytic SE (test and every record).
* **Paired masks:** differences use d_i = (c_i^A − c_i^B) / O on the same queries (§5).
* **Scope of the pooled value:** it is conditional on the chosen outputs (32 seeded random, the 4 dense-lowest-feature,
  2 count-class outputs). It is not an unweighted random sample of the 1024 outputs and carries no output-sampling
  uncertainty. Individual-output Clopper-Pearson intervals remain valid; exact pair / triple layer coverage and
  exact feature counts need no interval.
* **Regression** (`tests/test_cover_topk.py::test_duplicated_outputs_add_no_monte_carlo_evidence`): O identical
  output memberships give exactly the single-output SE, while the old pooled interval shrinks like 1/sqrt(O).

**Corrected values** (query pass rerun; same mask sha256 and identical per-output hit counts as the old records):

| support | old pooled CI (invalid) | corrected joint-query CI | width ratio | outputs per covered query |
|---|---:|---:|---:|---:|
| BP-row/8 #0 g0 | [0.0237, 0.0244] % | [0.0237, 0.0244] % | 1.00 | 1.01 |
| TopK/8 g0 | [0.0235, 0.0242] % | [0.0235, 0.0242] % | 1.00 | 1.01 |
| U8 0.125 g0 | [0.1042, 0.1057] % | [0.1041, 0.1057] % | 1.09 | 1.11 |
| TopK/16 g0 | [0.2158, 0.2179] % | [0.2158, 0.2180] % | 1.06 | 1.11 |
| BP-row/16 #0 g0 | [0.2197, 0.2219] % | [0.2197, 0.2219] % | 1.06 | 1.11 |
| U16 0.25 g0 | [0.5471, 0.5504] % | [0.5467, 0.5507] % | 1.20 | 1.29 |
| A8 0.25 / Mix-TopK / Mix-BP g0 | [1.1124, 1.1171] % | [1.1116, 1.1178] % | 1.30 | 1.50 |

The point values were right; the K8 intervals happen to be numerically unchanged because almost every covered query
is covered by one selected output. The denser supports' intervals were too narrow by up to 30 %. All rows (every
graph, FP-mag/8, WRS/8:4, WRS/8:16) are in `tables.md` ("Erratum: per-output triple intervals with shared queries",
"Coverage reruns of the previous round's other supports").

### 2.2 Frontier and interpretation

Wrong or too strong in `WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md`, now corrected:

* **"U8: dominated by TopK/8."** Not over the declared vector: U8 has far more tuple coverage (37.6 % vs 19.5 % layer
  triples, 0.105 % vs 0.024 % per output). U8 is slower and 34.4 % structurally bias-only; it stays on the frontier.
* **"Dominated at 200M: BP-row/16 and A8."** Not over the full vector. A8 has full width and exactly the dense
  envelope's coverage (61.636 % triples, identical to dense and Mix-TopK) and is FASTER than Mix-TopK (978 / 988 vs
  1,075 / 1,087 ms); BP-row/16 has full width, complete input coverage and is faster than Mix-TopK. Both remain on a
  point-estimate frontier; they lose only in loss.
* **"Mix-TopK: the only 200M point with full width and full dense-envelope coverage."** A8 has both too.
* **Noninferiority is not equality.** "Tied in loss" / "matches" statements meant: the paired interval is inside the
  margin. Point-estimate frontiers and uncertainty-qualified claims are now reported separately (§9).
* **Mix-TopK interaction plateau.** Its T16 anchor oracle realizes the teacher to ~1e-14 nMSE against BF16-input
  targets with the analytic readout, i.e. an attainable solution was explicitly demonstrated inside the support.
  The ordinary-training gap (0.71-0.78 explained at 1,000 updates) is therefore a gap ABOVE an attainable solution.
  Support / multiplicity can still affect optimization, but this is no evidence of an unavoidable representation
  error at that operating point. (For masks without such a construction, a failed matching still does not establish
  their optimum.)
* **Grouped launches.** "6 launches per step plus 1 optimizer call" is wrong. The recorded adapter makes 3 forward
  and 6 backward BACKEND calls (`last_info.launches_fwd / _bwd` = 3 / 6, i.e. 9 kgen kernel calls per step). The
  actual GPU kernel-launch total per complete step also includes staging, zeroing, finalization, loss and optimizer
  kernels and is measured separately in §4.3: 17 CUDA kernels per step (indexed K16: 11).

### 2.3 The SIGINT full-suite failure: root cause and fix

**Record.** The previous full suite (`3188 passed, 32 skipped, 1 failed`) failed
`test_interrupt_terminates_children_and_exits_nonzero[2-130]` (SIGINT only): `p.communicate(timeout=30)` timed out
AFTER the four fake workloads had started and SIGINT was sent, with an EMPTY stderr. The trap's first action is an
echo to stderr, so the trap never ran. The previous report's "startup timeout" was wrong.

**Cause (demonstrated).** The suite was launched as `(nohup systemd-run --user --scope -q bash stage7.sh ... &)`.
A command started with `&` by a non-interactive shell inherits SIGINT and SIGQUIT IGNORED (SigIgn 0x6). pytest,
the test's `Popen` and the runner bash inherit that disposition, and a non-interactive bash cannot trap a signal that
was ignored on entry: `trap 'on_signal INT 130' INT` silently installs nothing. SIGINT is therefore discarded and
the runner keeps waiting on its 300 s fake workloads. SIGTERM is not ignored, which is why only the SIGINT variant
failed and why foreground reruns passed.

* **Deterministic reproduction:** running the unchanged test under `bash -c "pytest ... & wait"` gives the identical
  traceback (stderr empty, 30 s timeout) every time (`runner_sigint/repro_ign.txt`).
* **Traced trials** (`runner_sigint/traced.json`, `xtrace_*.txt`, PS4 with `$EPOCHREALTIME` / `$BASHPID`): with the
  default disposition the runner's SigCgt includes SIGINT (0x14003) and `on_signal INT 130` starts 1.03 s after
  the signal (deferred until the foreground `sleep 1` ends), then exits 130 with no survivors. With SIGINT ignored
  on entry SigIgn = 0x6, SigCgt = 0x14001 (no SIGINT): no trap entry, the runner keeps running, and a later SIGTERM
  exits 143 with no survivors. Each job is its own process group (pgid = its `timeout` leader), same session.
* **Stress matrix** (`runner_sigint/matrix_summary.json`, 8 cells x 30 trials; SIGINT / SIGTERM chosen per trial;
  sent 0-0.6 s after launch (startup) or 0-2 s after all four workloads started; 0 or 6 CPU hogs; default vs ignored
  disposition): every default-disposition trial exited 130 / 143 within 1.51 s with no surviving child. Every
  ignored-disposition SIGTERM trial did the same; every ignored-disposition SIGINT trial hung > 30 s (the harness
  then SIGKILLed the runner, which can never clean up, and killed the 8 orphans itself).
* **A second, smaller window** (old runner, `start1.jsonl`): 6 of 40 startup signals arrived before the traps were
  installed and killed the runner with its default disposition. No child existed yet (children 0, no survivors), so
  this was harmless, but the traps are now installed before anything is started.

**Fix** (no deadline increase, no unrelated process groups touched):

* `run_sanitizer.sh`: the traps (and their state) are installed at the top, before any child; when SIGINT was
  ignored on entry the runner says so at startup ("SIGINT was ignored when this runner started ...; interrupt with
  SIGTERM or SIGHUP") instead of silently ignoring it. Re-enabling SIGINT was rejected: background jobs without job
  control share the terminal's process group, and ignoring SIGINT there is the POSIX convention that keeps Ctrl-C
  from killing them.
* `tests/test_sanitizer_runner.py`: the interrupt test now sets SIGINT's disposition explicitly (`preexec_fn`
  SIG_DFL), so it no longer depends on how pytest was launched; a bounded teardown (SIGTERM to the runner, 15 s
  grace, then SIGKILL of the runner and of THIS test's recorded fake PIDs only) replaces the bare `p.kill()` that
  leaked the eight 300 s sleepers on failure. New regression
  `test_sigint_ignored_on_entry_is_reported_and_sigterm_still_cleans_up`: SIG_IGN on entry → notice printed, SIGINT
  ignored (runner alive after 2.5 s), SIGTERM → 143 with all 8 children gone. It fails on the old runner (no notice).
* Verification: §10 (runner tests with SIGINT default, ignored by inheritance and under full CPU load; full suite).

## 3. Coverage-constrained TopK

### 3.1 Constraint

For an original record r with m valid row groups and S = 64 local inputs, X[u, j] in {0, 1}: sum_j X[u, j] = k for every
valid group, sum_u X[u, j] >= 1 for every j. Within each record, not layer-wide; no neuron needs all 64 inputs; no
labels / targets are used. Valid-slot cost k n_r (unchanged density, no neuron removed). Feasible iff m k >= 64 with
distinct coordinates: the model has 48,128 full records (16 groups) and 1,024 n175 / n176 tails (11 groups), both
feasible at k8 and k16. Smaller geometries are rejected with ValueError (no epsilon edges, larger k, padding or moved
neurons; tested on a C64 layout, 4 x 8 = 32 < 64). This guarantees equality with the dense envelope's input-feature
union per record and per output; it does not guarantee pair / triple / quadruple coverage. The fixed-fan-in per-output
triple bounds (<= 763 support records per output) stay: k8 <= 0.0239462 %, k16 <= 0.2394624 %.

### 3.2 Fast deterministic repair (`experimental/local_supports.py`: `Cover-TopK`, `cover_repair`)

* Importance a[u, j] = sum over valid neurons of W_pre[u, c, j]^2 (FP64, sequential, the parent's), start from
  ordinary TopK/k (stable descending sort), normalization only after selection.
* Per record, while an input j is uncovered: apply the swap (u, d → j) of minimum a[u, d] − a[u, j] over every
  uncovered j, group u and retained donor d with record count c[d] >= 2; ties lexicographic in (j, u, d).
* Batched: a group's cheapest removable donor d*(u) does not depend on j, so the best swap is min over (j, u) of
  a[u, d*(u)] − a[u, j] flattened in (j, u) order. Records are processed in chunks of 32,768 ([chunk, 16, 64]
  buffers, 576 MiB), all records of a chunk advance one swap per round (23-24 rounds at k8, 8-9 at k16): no
  per-swap host synchronization, no target x group x donor tensor.
* Termination is checked, not assumed: an uncovered j with m k >= 64 implies a covered coordinate with count >= 2,
  hence a legal donor, and j is absent from every group; each swap covers exactly one more input and uncovers none.
  The swap count equals the initial number of missing inputs in every record (asserted in code; recorded "yes" for
  all 6 graph x k builds).
* Coordinates are sorted afterwards; the result is an ordinary `LocalSupport` (canonical maps, `lower_local`,
  indexed kgen K8 / K16 C16). Cover-TopK is magnitude-dependent: it is in `LS.WEIGHTED`, so no uniform BP prediction
  is applied. Old BP / TopK construction, keys and fingerprints are unchanged (TopK/8 g0 / g1 / g2 and TopK/16 g0
  fingerprints equal the previous round's records).
* Reproducibility: a different chunking (4,093 records) and the same repair on the GPU give bitwise-identical supports
  (all builds).

### 3.3 Construction on the actual checkpoints (`construct_quiet/`, graph 0 / 1 / 2)

| k | TopK missing inputs (of 3.15M) | records repaired | missing / record (full / tail) | TopK energy | Cover energy | loss vs TopK | record loss median / p99 | total build s (TopK / Cover) | repair s CPU / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 378,880-379,124 | 49,148-49,151 of 49,152 | 7.56 / 14.7 | 0.20487 | 0.20407 | 0.387 % | 0.36 % / 1.65 % | 1.09 / 3.90 | 2.6-2.7 / 0.19-0.21 |
| 16 | 33,914-33,965 | 24,495-24,531 | 0.65 / 2.7 | 0.36766 | 0.36762 | 0.010 % | 0 / 0.082 % | 1.37 / 2.32 | 0.74-0.81 / 0.09 |

* For reference BP-row retains 0.125 / 0.250 of the energy. The tails lose more (k8 record loss up to 3.2 %).
* Donor and target positions are spread evenly over the 64 coordinates (histograms in the records).
* Host peak: RSS growth ~0.95 GiB (k8) / ~0.2 GiB (k16) during construction (FP64 importances of 781,312 groups,
  400 MB, plus the chunk buffers); device peak 3.8 GiB (the checkpoint weights). One-time preprocessing,
  amortized over every later step; it is outside every timed region (the timing records' `build_seconds` includes
  it: 11.0 / 12.5 s vs 10.0 / 9.7 s for TopK, including checkpoint load).

### 3.4 Exact constrained optimum on actual records (`cover_reference.py`, `construct_quiet/construct_g0.json`)

* Formulation: maximum-weight bipartite b-matching / transportation with a column lower bound. Solvers (mature, not
  written here): scipy 1.15.3 `milp` (HiGHS) on the binary program, mip_rel_gap 0, FP64 objective as is (no
  quantization), default feasibility tolerance 1e-7; the returned X is rounded, checked integral
  (|x − round(x)| <= 1e-9) and feasible, and its energy recomputed with `math.fsum`. Cross-checks: the LP relaxation
  (HiGHS `linprog`; totally unimodular, bound − optimum <= 1.4e-14) and networkx 3.3 network simplex on the
  min-cost-flow form with the lower bound removed by the standard demand transformation, integer costs
  round(a 2^40 / max a) (its masks have the same FP64 energy as the MILP's in every sampled record: |flow − MILP| = 0).
* Validation: 40 tiny random / zero / all-tied cases against vectorized brute force over all C(S, k)^m masks
  (test).
* Sample: stratified, equal allocation over (full / tail) x (unrepaired, repaired-loss quartiles), seed 20261029 + k;
  228 records (k8; only 4 unrepaired full records exist) and 250 (k16). It is NOT a uniform sample: its summaries
  describe the strata.

| k | records | MILP certified | E(TopK) >= E(opt) >= E(repair) | repair = optimal mask | gap median | gap p95 | gap max | worst stratum (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 228 | 228 | 228 | 72 | 0.019 % | 0.18 % | 0.25 % | tail, top loss quartile 0.10 % |
| 16 | 250 | 250 | 250 | 243 | 0 | 0 | 0.016 % | tail, top quartile 0 (p95 0.015 %) |

No zero-optimum record occurred (relative gaps defined throughout). **Optional solver variant: not run.** The
preregistered trigger needed a gap of >= 1 % median or >= 5 % p95 AND predicted full-model construction <= 10 min. The
gaps are 50x / 28x below the thresholds. (Addendum: the preregistration's quoted time estimate, ~15 min, came from a
pilot run under CPU stress; the quiet measurement is 0.010 s per record, 8.5 / 8.0 min for 49,152 records. The gap
criterion alone decides; the decision is unchanged.)

## 4. Performance (`perf_screen/`, `perf_confirm/`)

### 4.1 Screen (graph 0, 8 samples)

Cover-TopK vs TopK on the same indexed path: 628.5 vs 630.0 ms (k8), 871.7 vs 864.3 ms (k16); identical phase
structure (dX 377 / 608 ms). No material regression: the profiling gate (> 5 % at either k) is not triggered.

### 4.2 Confirmation (graph 0 / 1 / 2, one fresh-process session per graph, 24 samples per job)

| support | format | median step per graph (ms) | graph mean ± 95 % CI | ratio to same-session U16 / U8 | fwd | dX | dW / db | opt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TopK/16 | indexed K16 | 878.4 / 875.3 / 877.2 | 877.0 ± 3.9 | 1.065 ± 0.002 | 69.6 | 611.8 | 189.2 | 5.1 |
| Cover-TopK/16 | indexed K16 | 875.4 / 877.7 / 872.6 | 875.2 ± 6.4 | 1.063 ± 0.009 | 69.6 | 610.4 | 188.9 | 5.1 |
| BP-row/16 (layouts 0 / 1) | indexed K16 | 877.7 / 875.9 / 877.3 | 877.0 ± 2.3 | 1.065 ± 0.001 | 69.9 | 611.9 | 189.0 | 5.1 |
| U16 0.25 | wide 16 x 16 | 824.7 / 822.8 / 823.5 | 823.7 ± 2.3 | 1 | 118.4 | 430.7 | 267.6 | 5.1 |
| A8 0.25 | wide 16 x 8 | 988.5 / 987.5 / 986.5 | 987.5 ± 2.5 | 1.199 ± 0.003 | 124.7 | 453.6 | 402.6 | 5.1 |
| Mix-TopK (reference) | grouped K64 / K16 / K8 | 1088.3 / 1088.3 / 1085.6 | 1087.4 ± 3.8 | 1.320 ± 0.005 | 92.9 | 687.9 | 300.1 | 5.1 |
| TopK/8 | indexed K8 | 633.3 / 634.0 / 633.3 | 633.5 ± 0.9 | 0.907 ± 0.002 | 49.1 | 380.3 | 200.2 | 2.7 |
| Cover-TopK/8 | indexed K8 | 633.2 / 633.4 / 633.4 | 633.4 ± 0.2 | 0.907 ± 0.003 | 49.3 | 380.1 | 200.0 | 2.7 |
| BP-row/8 (layouts 0 / 1) | indexed K8 | 632.6 / 631.7 / 632.6 | 632.3 ± 1.3 | 0.905 ± 0.005 | 49.2 | 379.6 | 199.4 | 2.7 |
| U8 0.125 | wide 16 x 8 | 698.7 / 699.5 / 697.5 | 698.6 ± 2.5 | 1 | 99.6 | 362.9 | 232.0 | 2.7 |
| dense | dense | 1566.1 / 1566.6 / 1568.0 | 1566.9 ± 2.5 | — | 209.9 | 765.1 | 571.2 | 19.4 |

* The interval is over the three graph means (df 2), not over timing steps; BP layout spread <= 3.0 ms.
* TopK/16's old graph-0 median (864.8 ms) is context; this session is ~1.2-1.3 % slower for every format alike (U16
  813.7 → 823.7, BP-row/16 865.7 → 877.0); the ratios are stable.
* The deterministic policies (TopK, Cover-TopK) received no artificial layout replicates.
* Host enqueue 1.3 (indexed) / 1.5-1.6 ms (wide, grouped) per step; host step wall = GPU step + < 0.1 ms (GPU-bound).

### 4.3 Memory, index bytes and launches

| support | values | bias | main_grad | moments (BF16 x2) | index / map / work-list bytes | step scratch | peak alloc / reserved | first step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cover-TopK/16 = TopK/16 = BP-row/16 | 381.5 MiB | 23.8 | 810.7 | 810.7 | 43.9 (index records 23.8, canonical map 11.9, work lists 6.9, graph 1.25) | 160 | 2.19 / 2.25 GiB | 0.88 s |
| U16 0.25 | 381.5 | 24.0 | 810.9 | 810.9 | 10.7 (tile metadata 3.0, work lists 6.4) | 206 | 2.27 / 2.35 | 1.17 |
| A8 0.25 | 381.5 | 24.0 | 811.0 | 811.0 | 12.9 | 160 | 2.18 / 2.35 | 1.19 |
| Mix-TopK | 381.5 | 23.8 | 810.7 | 810.7 | 65.3 (per type K64 / K16 / K8 records, maps, buckets, work lists + group maps 3.75) | 160 | 2.21 / 2.60 | 1.10 |
| Cover-TopK/8 = TopK/8 = BP-row/8 | 190.8 | 23.8 | 429.2 | 429.2 | 25.8 (records 11.9, map 6.0, work lists 6.7, graph 1.25) | 160 | 1.24 / 1.35 | 0.66 |
| U8 0.125 | 190.7 | 24.0 | 429.5 | 429.5 | 10.5 | 210 | 1.36 / 1.40 | 1.08 |
| dense | 1536 | 24.0 | 3120 | 3120 | 9.9 | 160 | 7.80 / 8.35 | 1.59 |

(main_grad / moments are charged on the padded flat storage of each format; padding slots hold no work.) Every
real input index and work list is charged. Cover-TopK has exactly the bytes of TopK: the repair changes which
coordinates are stored, not how many.

GPU kernels per complete training step (`launch2/`: torch.profiler, 2 warm-up steps then 4 profiled steps, per-step
averages, user annotation ranges excluded; outside every timing session):

| format | CUDA kernels / step | of which layer kernels | memset + memcpy |
|---|---:|---:|---:|
| Mix-TopK grouped | 17 | 9 (kgen fwd / dX / dW-db x K64, K16, K8) | 2 |
| TopK/16 = Cover-TopK/16 = BP-row/16 indexed | 11 | 3 (kgen K16) | 2 |
| U16 0.25 wide | 15 | 3 (kw 16 x 16) | 2 |
| A8 0.25 wide | 14 | 3 (kw 16 x 8) | 2 |
| TopK/8 / Cover-TopK/8 indexed | 10 / 9.75 | 3 (kgen K8; fwd seen in 3 of 4 steps) | 1.75 / 1.5 |
| U8 0.125 wide | 14 | 3 (fwd seen in 3 of 4) | 1.75 |
| dense | 11 | 3 | 2 |

Every format adds ~8 auxiliary kernels (loss, casts / fills, 2 L1AdamW). The grouped step therefore issues 17 kernels,
not "6 launches + 1 optimizer call": the 9 adapter backend calls plus the auxiliary work. Fractional counts are
profiler event drops (a kernel missing from one of the 4 profiled steps), so these totals are lower bounds within one
kernel; the first single-step profile (`launch/`, schema /1) lost more events (one indexed K16 run showed no forward
kernel) and is kept only as a disclosed failed measurement.

## 5. Coverage and multiplicity (`coverage/`, all timed supports, graph 0 / 1 / 2)

| support | records with all 64 inputs (mask = numerical) | outputs equal to the dense feature union | layer pairs | layer triples (exact) | per-output triples, mean of 38 [joint 95 %] | structural bias-only |
|---|---:|---:|---:|---:|---:|---:|
| TopK/16 | 24,621-24,657 (50 %) | 175-214 | 100 % | 58.75 % | 0.2167 [0.2156, 0.2178] % (g1) | 0 |
| Cover-TopK/16 | 49,152 (100 %) | 1,024 | 100 % | 58.76 % | 0.2172 [0.2161, 0.2184] % | 0 |
| BP-row/16 (#0 / #1) | 49,152 | 1,024 | 100 % | 58.90-58.92 % | 0.2197-0.2214 % | 0 |
| U16 0.25 | 47,040-47,073 | 303-317 | 100 % | 60.90 % | 0.547-0.550 % | 31.7 % |
| A8 0.25 = Mix-TopK = dense | 49,152 | 1,024 | 100 % | 61.64 % | 1.115 [1.112, 1.118] % | 0 |
| TopK/8 | 1-4 | 0 | 100 % | 19.47 % | 0.0235-0.0238 % | 0 |
| Cover-TopK/8 | 49,152 | 1,024 | 100 % | 19.52 % | 0.0237-0.0240 % | 0 |
| BP-row/8 (#0 / #1) | 49,152 | 1,024 | 100 % | 19.54 % | 0.0236-0.0240 % | 0 |
| U8 0.125 | 17,689-17,733 | 0 | 100 % | 37.6 % | 0.105-0.107 % | 34.4 % |

* Mask vs numerical: the checkpoint's weights are nonzero at every enabled slot of every support here, so the two
  coincide (`nonzero.identical_to_enabled`); the code counts them separately (an enabled zero weight adds no
  input dependence; tested).
* Distinct global supports (layer): 781,312 for every TopK / Cover / BP K8 / K16 support (one per row group).
* **Paired differences on the same queries** (`tables.md` "Paired coverage differences"):

| A − B | per-output triples (pp, joint 95 %) | layer triples (pp, exact) |
|---|---:|---:|
| Cover-TopK/8 − TopK/8 | +0.00004 .. +0.00021 (g0 CI includes 0) | +0.055 |
| Cover-TopK/16 − TopK/16 | +0.0003 .. +0.0005 (all CIs above 0) | +0.014 |
| Cover-TopK/16 − BP-row/16 | −0.0029 .. −0.0036 (all CIs below 0) | −0.14 |
| Cover-TopK/8 − BP-row/8 | −0.0002 .. −0.0001 (CIs include 0) | −0.02 |
| TopK/16 − U16 | −0.331 .. −0.333 | −2.15 |
| TopK/8 − U8 | −0.081 .. −0.083 | −18.1 |

Complete input coverage changes higher-order coverage by hundredths of a percentage point; the fixed-fan-in bound
(k8 0.02395 %, k16 0.2395 % per output) keeps every K8 / K16 row-group support near or under it, and U / A tiles
(more inputs per neuron) remain far ahead.

## 6. Development (`dev32/`, `dev256/`; previous round's TopK records reused)

* **Continuity:** TopK/8 and TopK/16 rerun at 32 updates (seed 0, none / energy) reproduce the previous records to
  <= 7.5e-8 with identical mask sha256 and checkpoint → the previous 256-update TopK records are reused (rule).
* **256 updates, validation 9,000..9,003, seeds 0 / 1 / 2:**

| support | none | energy | selected |
|---|---:|---:|---|
| Cover-TopK/8 | 0.2234 ± 0.0013 | 0.2133 ± 0.0027 | energy |
| TopK/8 (reused) | 0.2231 ± 0.0014 | 0.2130 ± 0.0027 | energy |
| Cover-TopK/16 | 0.1737 ± 0.0025 | 0.1713 ± 0.0042 | energy |
| TopK/16 (reused) | 0.1737 ± 0.0025 | 0.1713 ± 0.0042 | energy |

* Energy restoration (cap s_max 8, the existing epsilon rule): mean scale 2.39 (k8) / 1.70 (k16); clipped 0.011 %
  (k8) / 0 (k16) of connected rows; relative probe variance at the start 1.00 (none: 0.19 / 0.36); retained energy
  0.2041 / 0.3676 (Cover) vs 0.2049 / 0.3677 (TopK); participation ratio ~160 throughout.
* **Learned projection:** the trainable projection moves far in 256 updates for every method (relative change of
  proj.weight 1.67-2.78; layer-input token RMS 0.47 → ~1.6; per method in `final/*.json`
  `learned_small_parameters`). In this task a missing layer-input coordinate can therefore be partly compensated
  upstream; raw-feature observability is only tested on the fixed-input diagnostic (§8).
* Frozen before the final stage (`final_selection.json`): all treatments energy except A8 (none, previous selection)
  and dense.

## 7. New locked test (`final/`, `tkc29000`, graph 0, 256 updates, seeds 6..11, states kept)

| method | treatment | test loss per seed (6 / 7 / 8 / 9 / 10 / 11) | mean ± pointwise 95 % |
|---|---|---:|---:|
| TopK/8 | energy | 0.2125 / 0.2102 / 0.2127 / 0.2102 / 0.2109 / 0.2113 | 0.2113 ± 0.0011 |
| Cover-TopK/8 | energy | 0.2129 / 0.2105 / 0.2130 / 0.2107 / 0.2113 / 0.2117 | 0.2117 ± 0.0011 |
| BP-row/8 #0 | energy | 0.2237 / 0.2223 / 0.2235 / 0.2229 / 0.2232 / 0.2240 | 0.2233 ± 0.0006 |
| U8 0.125 | energy | 0.2111 / 0.2100 / 0.2114 / 0.2105 / 0.2113 / 0.2106 | 0.2108 ± 0.0006 |
| TopK/16 | energy | 0.1698 / 0.1698 / 0.1701 / 0.1694 / 0.1691 / 0.1704 | 0.1698 ± 0.0005 |
| Cover-TopK/16 | energy | 0.1698 / 0.1698 / 0.1701 / 0.1694 / 0.1690 / 0.1704 | 0.1698 ± 0.0005 |
| BP-row/16 #0 | energy | 0.1774 / 0.1776 / 0.1791 / 0.1761 / 0.1770 / 0.1775 | 0.1775 ± 0.0010 |
| U16 0.25 | energy | 0.1679 / 0.1672 / 0.1680 / 0.1671 / 0.1681 / 0.1672 | 0.1676 ± 0.0005 |
| A8 0.25 | none | 0.1764 / 0.1757 / 0.1761 / 0.1760 / 0.1765 / 0.1763 | 0.1762 ± 0.0003 |
| Mix-TopK #0 | energy | 0.1718 / 0.1726 / 0.1731 / 0.1715 / 0.1705 / 0.1723 | 0.1720 ± 0.0010 |
| dense | none | 0.1134 / 0.1127 / 0.1130 / 0.1132 / 0.1113 / 0.1129 | 0.1128 ± 0.0008 |

**Primary preregistered decisions** (paired over the six seeds; upper = mean + t(1 − 0.05/3, 5) sd / sqrt(6),
t = 2.9117; margin 0.003):

| comparison | mean | adjusted one-sided upper | pointwise 95 % (descriptive) | decision |
|---|---:|---:|---:|---|
| TopK/16 − U16 | +0.00216 | +0.00307 | [+0.00136, +0.00296] | inconclusive |
| Cover-TopK/8 − TopK/8 | +0.00039 | +0.00047 | [+0.00033, +0.00046] | noninferior |
| Cover-TopK/16 − TopK/16 | −0.00001 | +0.00001 | [−0.00002, +0.00000] | noninferior |

**Secondary / exploratory** (pointwise 95 %, no family-wise claim): TopK/16 − A8 −0.0064 [−0.0071, −0.0058];
TopK/16 − BP-row/16 −0.0077 [−0.0085, −0.0069]; Cover-TopK/8 − U8 +0.0009 [+0.0001, +0.0016]; Cover-TopK/16 − U16
+0.0022 [+0.0014, +0.0030]; Cover-TopK/8 − BP-row/8 −0.0116 [−0.0124, −0.0108]; Mix-TopK − TopK/16 +0.0022
[+0.0016, +0.0028].

* TopK/16 − U16 is not "equal" and not "inferior" under the rule: its adjusted bound misses the margin by 0.00007.
* Every final record names its test set (`final_test_set = tkc29000`) and stores its state (sha256 in the record,
  `final_states.sha256`; 86 GB under `../tkc_runs/final/state`, outside git).

**Graph checks** (`graph_checks/`, seed 6, same-graph checkpoint / projection / head; reported separately, not
pooled):

| method | graph 0 | graph 1 | graph 2 |
|---|---:|---:|---:|
| TopK/16 | 0.1698 | 0.1696 | 0.1700 |
| Cover-TopK/16 | 0.1698 | 0.1697 | 0.1700 |
| TopK/8 | 0.2125 | 0.2123 | 0.2127 |
| Cover-TopK/8 | 0.2129 | 0.2127 | 0.2132 |
| BP-row/16 layout 0 / 1 | 0.1774 / — | 0.1778 / 0.1779 | 0.1783 / 0.1783 |
| BP-row/8 #0 | 0.2237 | 0.2234 | 0.2236 |

Graph variation <= 0.001 for every method; the Cover − TopK differences (+0.0004 at k8, ~0 at k16) hold at every
graph.

## 8. Reduced raw-input diagnostic (`construct/floor_check_*`, `interaction/`, `interaction_cont/`)

D = G 1024, H 262,144, C256 / S64 / R256, one original record per output, no trainable projection, per-output affine
readout, T16, hash 777; fresh K64 init (seeds 20 / 21 / 22 with SR seeds 0 / 1 / 2), zero bias, no row scaling, the
parent's readout calibration; layer lr 1e-2, readout lr 1e-2; development batches 49,000..49,003; reserved test
`tkc69000` evaluated once at each declared endpoint (1,000 for all; 4,000 for the continued K8 policies, resumed from
the complete saved 1,000-update state).

### 8.1 Missing-input floor (`rsn_interaction.missing_input_floor`, every run)

For an order-1 output the target is linear, t_g = gamma_g + sum_i beta[g, i] x_i with beta[g, i] = (number of the
output's valid terms reading input i) / (sigma_1 sqrt(n_g T)) (duplicates add) and gamma_g = −sqrt(n_g T) mu_1 /
sigma_1, mu_1 / sigma_1 being the code's own Monte Carlo constants. For independent N(0, 1) inputs any function of
the visible inputs U_g has MSE >= sum_{i not in U_g} beta^2; nMSE divides by sum t^2, whose population value is
sum beta^2 + gamma^2 (gamma^2 / E[t^2] ~ 3e-7), so floor_g = sum_missing beta^2 / (sum beta^2 + gamma^2), averaged
over order-1 outputs like the evaluation. Validation (`tkc_floor_check.py`): the residual of the best visible-input
linear predictor, computed against the ACTUAL target code on 64 x 512 fresh tokens (seeds 77,000+), agrees with the
analytic floor within −0.01 .. +0.52 % (k8 / k16 TopK, three inits); with all inputs visible the residual is FP32
round-off (1.7e-11 SSE). BF16 input quantization is separate: targets of BF16-rounded inputs differ from the true
targets by nMSE 2.8e-6 (order 1) .. 8.2e-6 (order 2).

### 8.2 Results (three init / SR pairs; test batches; mean ± 95 %)

| support | updates | local inputs / record | missing order-1 target inputs / output | order-1 floor | order-1 nMSE | nMSE − floor | explained interaction 2 / 3 / 4 | order-2 / 3 / 4 terms without an eligible neuron (of 4,096) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TopK/8 | 1000 | 56.47 | 1.25 | 0.116 ± 0.018 | 0.126 ± 0.017 | 0.010 ± 0.001 | −0.33 / −0.29 / −0.27 | 3,254 / 4,003 / 4,088 |
| Cover-TopK/8 | 1000 | 64 | 0 | 0 | 0.0119 ± 0.0007 | 0.0119 | 0.161 / 0.131 / 0.122 | 3,235 / 4,006 / 4,087 |
| BP-row/8 | 1000 | 64 | 0 | 0 | 0.0111 ± 0.0000 | 0.0111 | 0.181 / 0.146 / 0.132 | 3,219 / 4,001 / 4,088 |
| TopK/16 | 1000 | 63.38 | 0.10 | 0.0096 ± 0.0040 | 0.0153 ± 0.0041 | 0.0057 ± 0.0003 | 0.566 / 0.473 / 0.449 | 1,491 / 3,308 / 3,907 |
| Cover-TopK/16 | 1000 | 64 | 0 | 0 | 0.0057 ± 0.0002 | 0.0057 | 0.607 / 0.513 / 0.484 | 1,479 / 3,307 / 3,905 |
| BP-row/16 | 1000 | 64 | 0 | 0 | 0.0051 ± 0.0000 | 0.0051 | 0.635 / 0.556 / 0.515 | 1,402 / 3,279 / 3,924 |
| dense | 1000 | 64 | 0 | 0 | 0.0012 | 0.0012 | 0.987 / 0.990 / 0.991 | 0 / 0 / 0 |
| TopK/8 | 4000 | 56.47 | 1.25 | 0.116 | 0.118 ± 0.017 | 0.0019 ± 0.0012 | −0.29 / −0.26 / −0.24 | as above |
| Cover-TopK/8 | 4000 | 64 | 0 | 0 | 0.0006 | 0.0006 | 0.201 / 0.165 / 0.155 | as above |
| BP-row/8 | 4000 | 64 | 0 | 0 | 0.0006 | 0.0006 | 0.208 / 0.171 / 0.159 | as above |

(± 0.0000 = below 5e-5: BP-row's mask does not depend on the init, and the init / SR seeds move its losses by
< 1e-5.)

* **Agreement:** TopK/8's order-1 loss converges onto its derived floor (gap 0.010 → 0.002); its order-1 terms
  without a neuron seeing their input (476-503 of 4,096) are, by construction, the terms whose input is uncovered. Repair removes the floor and
  the plateau: Cover-TopK/8 then trains like BP-row/8. The previous round's 0.134 plateau (init 17, 4,000 updates) is the same
  mechanism: its analytic floor at init 17 is 0.1311 (Monte Carlo 0.1310; `construct_quiet/floor_check_i17_previous_round.json`). This does not prove the mean-ReLU model represents the unbounded signed linear target exactly; the
  remaining 0.0006 / 0.005-0.012 above a zero floor is not labelled "pure optimization".
* **TopK/8's negative explained interaction** (−0.33) comes from the same holes: the additive (main-effect) parts of
  higher-order terms over missing inputs are unobservable; Cover-TopK/8 turns it positive (+0.16).
* **Unresolved higher orders:** with complete inputs, the explained interaction at k8 stays 0.12-0.21 and at k16
  0.48-0.61; the reference is the actual-target additive floor, and eligibility is nearly unchanged by the repair (it
  adds no co-located input tuples). Cover-TopK remains 0.006-0.03 below BP-row in explained interaction at both k
  (not explained here; BP's balanced partitions are one candidate reason, untested). No conditional
  Gaussian projection for unseen inputs was run (optional).

## 9. Frontier

Declared vector: valid slots, complete step, structural / numerical width, input-feature coverage (record and output
scope), tuple coverage (layer triples and per-output triples), task quality (new test). Point estimates (this
session / the new test); a point is listed as dominated only if another is at least as good on EVERY component.

| point | slots | step ms | bias-only (structural) | records / outputs complete | layer / per-output triples | test loss |
|---|---:|---:|---:|---:|---:|---:|
| TopK/8 | 100M | 633.5 | 0 | 0.01 % / 0 | 19.47 / 0.0238 % | 0.2113 |
| Cover-TopK/8 | 100M | 633.4 | 0 | 100 % / 1024 | 19.52 / 0.0240 % | 0.2117 |
| BP-row/8 | 100M | 632.3 | 0 | 100 % / 1024 | 19.54 / 0.0240 % | 0.2233 |
| U8 0.125 | 100M | 698.6 | 34.4 % | 36 % / 0 | 37.6 / 0.105 % | 0.2108 |
| TopK/16 | 200M | 877.0 | 0 | 50 % / 214 | 58.75 / 0.217 % | 0.1698 |
| Cover-TopK/16 | 200M | 875.2 | 0 | 100 % / 1024 | 58.76 / 0.217 % | 0.1698 |
| BP-row/16 | 200M | 877.0 | 0 | 100 % / 1024 | 58.90 / 0.221 % | 0.1775 |
| U16 0.25 | 200M | 823.7 | 31.7 % | 96 % / 303 | 60.90 / 0.549 % | 0.1676 |
| A8 0.25 | 200M | 987.5 | 0 | 100 % / 1024 | 61.64 / 1.115 % | 0.1762 |
| Mix-TopK | 200M | 1087.4 | 0 | 100 % / 1024 | 61.64 / 1.115 % | 0.1720 |
| dense (reference) | 800M | 1566.9 | 0 | 100 % / 1024 | 61.64 / 1.115 % | 0.1128 |

* **Point-estimate frontier, 100M:** all four points. TopK/8 (quality), Cover-TopK/8 (coverage at +0.0004),
  BP-row/8 (fastest by 1 ms, most K8 tuples, worst loss), U8 (best loss, most tuples; slower, bias-only rows).
* **Point-estimate frontier, 200M:** Cover-TopK/16, BP-row/16, U16, A8, Mix-TopK. TopK/16 is dominated by
  Cover-TopK/16 (equal or better on every component). BP-row/16 survives through tuple coverage, A8 through speed
  vs Mix-TopK, Mix-TopK through loss vs A8, U16 through speed and loss.
* **Uncertainty-qualified:** the 1-2 ms step differences among the K8 indexed policies and among the K16 indexed
  policies are inside the graph intervals (no speed claim); Cover-TopK/16 vs TopK/16 is noninferior (not equal);
  Cover-TopK/8 is noninferior to TopK/8 and descriptively 0.0004 worse; TopK/16 vs U16 is inconclusive under the
  primary rule; U8 vs Cover-TopK/8 and U16 vs Cover-TopK/16 favour U (secondary, pointwise).
* **Historical (previous round, batches 19,000..19,007, seeds 3 / 4 / 5; context only):** TopK/8 0.2114, BP-row/8
  0.2235, U8 0.2118, BP-row/16 0.1778, U16 0.1679, A8 0.1772, Mix-TopK 0.1724 — consistent with the new test within
  0.001; none is used in a paired claim here.

## 10. Correctness gates, sanitizers, suite

* **New tests** `tests/test_cover_topk.py` (19): batched repair = scalar reference (full records, tails, ties,
  all-tied groups; exact loss and swap counts), chunk / record-order independence, infeasibility (C64 layout,
  m k < 64), exact k per valid group / 64 inputs per record / exactly k H slots / q-neuron identity / sorted
  coordinates / nothing on padded rows at n175 / n176 tails, swap count = initial missing, energy loss = recorded
  loss, all-zero / tied importances, unchanged already-covered TopK masks, numerical vs mask coverage with enabled
  zeros, E(TopK) >= E(opt) >= E(repair) against brute force + MILP + flow (40 cases), unchanged BP / TopK
  construction, the joint-query intervals (duplicated outputs, independent outputs, bootstrap = analytic, paired
  masks, FPC = 0 for a full census), indexed K8 / K16 lowering = the dense masked function and the FP64 reference
  (forward, dX, dW, db, tails, B 40 / 64, padding zero), and a 3-step optimizer state comparison (SR off) plus
  TrainState restore / canonical identity.
* Targeted existing regressions: `test_local_supports.py`, `test_weighted_mixed_supports.py`,
  `test_row_normalize_oracle.py` (66 with the new file, all pass).
* Compute Sanitizer: SANITIZER_RESULT.
* Runner tests: RUNNER_RESULT.
* Full suite (final code): SUITE_RESULT.

## 11. Evidence layout and reproduction

`results/topk16_coverage_constrained_topk/`:

* `tables.md`, `csv/` — every table generated by `benchmarks/tkc_report.py` from the raw records.
* `selection_rules.json`, `selection_rules_addendum.json`, `*_selection.json` (+ `*_select.out`), `stage*.sh`,
  `stage*_driver.out`, `stage*_commit.txt`.
* Per stage, every job's JSON + log + `plan.json` / `manifest.json` / `status.jsonl`: `construct/`,
  `construct_quiet/`, `perf_screen/`, `perf_confirm/`, `coverage/` (+ `*.joint.npz`), `errata_cov/`, `dev32/`,
  `dev256/`, `final/`, `graph_checks/`, `interaction/`, `interaction_cont/`, `launch/` (failed v1), `launch2/`,
  `sanitizer/`, runner / suite outputs.
* `runner_sigint/` — stress / trace harnesses, per-trial JSONL, summary, xtrace logs, the deterministic reproduction.
* `final_states.sha256`, `interaction_states.sha256`; the states (86 GB / 1.5 GB) stay under `../tkc_runs/*/state`.

Reproduce (from `sparse_pooling/benchmarks` of the frozen checkout, `GRANULE_RESULTS_ROOT` set, the env_isaaclab
python): `bash stage1.sh`, `bash stage2.sh`, `bash stage3.sh` (they call `tkc_jobs.py <stage> --out <stage>
[--selection ...] --resume` and `tkc_select.py <rule>`); a single timing job is
`python tkc_jobs.py perf_confirm --out X --plan-only` + the command in its `plan.json`; tables:
`python tkc_report.py`.
