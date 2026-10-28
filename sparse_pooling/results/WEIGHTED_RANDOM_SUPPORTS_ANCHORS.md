# GranuleLayer: K8 quality confirmation, weighted random supports and budget-matched anchors

Branch `k8-weighted-supports-anchors`, started from `k16-random-support-normalization-controls` @ `03e189f`.
This is an experiment branch: production defaults and auto-dispatch are unchanged, and nothing is merged into
main. Raw evidence is in `results/weighted_random_supports_anchors/`; generated tables are in its `tables.md` and
`csv/` (`benchmarks/kwa_report.py`).

Every number is a measurement on one RTX 3090 (sm_86) unless it is labelled as a *prediction*, *model* or *bound*.
The learning workloads are the parent's synthetic diagnostics, not application-accuracy claims. "Development"
means the held-out batches 9,000–9,003, which earlier rounds already used for selection; "final test" means the
locked batches 19,000–19,007, read only once per final run, after every selection was frozen.

## 0. Answers

**1. Is BP-row K8's speed reproducible across graphs / layouts, and what task quality is lost at 100M slots?**
Its speed is reproducible. Its quality is the worst of the 100M supports tested.
* **Time** (graph seeds 0/1/2 × layout seeds 0/1, 24 restored samples per process, same-session controls):

  | 100M-slot support (rho 0.125) | complete step, graph mean ± 95 % CI | ×U8 (same session) | layout spread |
  |---|---:|---:|---:|
  | BP-row/8, indexed K8 C16 | **632.1 ± 9.4 ms** | 0.907 ± 0.003 | ≤ 1 ms |
  | FP-mag/8, indexed K8 | 646.5 ± 10.7 ms | 0.928 ± 0.004 | – |
  | FP-mag/8, index-free 16 × 8 (same function) | 823.5 ± 3.3 ms | 1.182 ± 0.021 | – |
  | U8 0.125, wide | 696.7 ± 10.5 ms | 1.000 | – |

  A second session (stage 5) gives 626.0 ± 4.7 ms (×0.906). The higher-budget frontier points in the same
  sessions were BP-row/16 indexed 874.5 ms, U16 0.25 wide 821.3 ms, A8 0.25 wide 985.3 ms and dense 1,568 ms.
* **Coverage** on the exact timed supports, every graph and layout:
  * layer triples are 19.54 % (exact; refined BP prediction 19.540 %);
  * per-output triples are 0.0240 % [0.0237, 0.0244] pooled over the 38 fixed outputs. The K8 union bound is
    763 C(8, 3) / C(1024, 3) = 0.0239462 %, so the measurement is at the bound within sampling error. It is
    not an exact count;
  * there are no bias-only rows. U8 has 34.4 % bias-only rows but 37.6 % layer triples and 0.105 %
    per-output triples.
* **Quality** (256 updates, development seeds 0/1/2, each support at its selected treatment: energy restoration
  for all three):

  | support | development loss | locked final test (seeds 3/4/5) |
  |---|---:|---:|
  | BP-row/8 | 0.2243 ± 0.0040 | 0.2235 ± 0.0022 |
  | FP-mag/8 | 0.2201 ± 0.0033 | – |
  | U8 0.125 | 0.2124 ± 0.0031 | 0.2118 ± 0.0010 |
  | *references (200M):* BP-row/16 0.1778, U16 0.1679, dense 0.1130 (final test) |

  At 100M slots BP-row/8 loses about 0.046 against BP-row/16 at 200M. It also trails U8 by about 0.012. The
  U8 gap holds at graph seeds 1/2 and both layouts (BP-row/8 0.2235–0.2242 vs U8 0.2112–0.2118, data seed 0).
  Structural width and layer coverage do not predict this ordering.

**2. Does weight-aware selection improve quality after fair scaling, and how much coverage / multiplicity does
it cost?** Yes at K8, at almost no coverage cost on this checkpoint.
* **Screen** (graph 0, same indexed K8 backend, 8 samples): WRS-8(4), WRS-8(16) and TopK-8 time within 0.2 % of
  BP-row/8. They keep 99.6 % of its layer triples and 97.6–99.3 % of its per-output triples. All three met the
  predeclared preference, and TopK-8 had the best 32-update score, so it alone advanced.
* **Retained pre-pruning energy:** BP-row/8 0.125, WRS(4) 0.140, WRS(16) 0.152, TopK 0.205. TopK/16 keeps 0.368
  against BP-row/16's 0.25. The checkpoint's energy is spread almost evenly over a row group's 64 inputs, so
  magnitude choice barely concentrates the support (§4.3).
* **Quality after fair scaling** (both supports with the same one-time energy restoration):
  * Development, 256 updates: TopK/8 0.2130 ± 0.0027 against BP-row/8 0.2243.
  * Locked final test, paired over seeds 3/4/5:

    | comparison | mean difference | paired 95 % CI | verdict |
    |---|---:|---:|---|
    | TopK/8 − BP-row/8 | −0.0121 | [−0.0143, −0.0099] | **improvement** |
    | TopK/8 − U8 0.125 | −0.0004 | [−0.0035, +0.0027] | **noninferior** (margin 0.003) |
  * Graph seeds 1/2 (data seed 0) give 0.2124 and 0.2134.
* **Time and coverage across graphs:** TopK/8 runs 629.1 ± 6.1 ms (×0.910 of U8) over graphs 0/1/2. It keeps
  19.47 % layer triples and 0.0235–0.0239 % per-output triples, with no bias-only rows.
* **Where it fails** (labelled extra, beyond the predeclared rule): on the reduced interaction task TopK/8 is
  clearly worse than BP-row/8 (§7). There each output has a single 64-input record. Independent
  largest-weight picks leave 7.4 of its 64 inputs uncovered (prediction 7.56), while BP-row's balanced
  permutations cover all 64. A missing input makes even an additive target term unrepresentable. This is a
  support effect, and the full-size model shows it only weakly (per-output features 954 vs 978).

**3. At 200M slots, can mixed anchors preserve dense-envelope coverage and approach A8 / U16 quality at an
attractive measured full-step time?** Coverage: yes, exactly. Quality: it beats A8 and BP-row/16 but not U16.
Time: it is the slowest 200M point.
* **Construction.** Each full C256 record has one K64 anchor, nine K16 groups and six K8 groups (§5). The totals
  are exact: H64 786,432, H16 6,994,976, H8 4,718,592, 200,000,000 slots and 12,500,000 biases. The n175/176
  tails are also exact.
* **Coverage.**
  * Layer and per-output tuple coverage equal the dense envelope's potential exactly. Layer triples are
    61.636 % with exact counts identical to dense; per-output triples are 1.1147 %, from the same samples.
    This holds at every graph and layout.
  * Every one of the 49,152 records keeps a full-mask, all-nonzero anchor, before and after training.
  * Distinct global supports at layer scope are 736,256, because anchors of one set coincide; at output scope
    they are 781,312.
* **Quality.** Development (256 updates): Mix-TopK 0.1739 (TopK/16 0.1713). Locked final test, paired over seeds
  3/4/5:

  | comparison | mean difference | paired 95 % CI | verdict |
  |---|---:|---:|---|
  | Mix-TopK − A8 0.25 | −0.0048 | [−0.0067, −0.0029] | **improvement** |
  | Mix-TopK − BP-row/16 | −0.0054 | [−0.0063, −0.0045] | **improvement** |
  | Mix-TopK − U16 0.25 | +0.0045 | [+0.0032, +0.0058] | **worse** (whole interval above 0) |
* **Time.** The grouped adapter on existing kgen kernels runs Mix-TopK at **1,075.3 ± 3.5 ms**, against A8 977.6,
  BP-row/16 865.7 and U16 813.7 ms (graph seeds 0/1/2, 24 samples). That is ×1.242 of BP-row/16 and ×1.10 of A8.
  * The extra time is in the per-type launches, not in adapter overhead: zeroing, layout and finalize total
    under 1 ms.
  * The K8 type is the expensive part: its dW costs about 3.6 ms per million slots, against 1.0 ms for K16 and
    0.9 ms for K64 C16.
* **What this experiment does and does not show.** It changes the fan-in allocation as well as adding anchors,
  and its selection rule differs from A8's. It is a budget-allocation result, not an isolated causal effect of
  anchors.

**4. Which quality differences persist on the locked final test and across graph / layout checks?**
* **Final-test results** (paired, three fresh training seeds, graph 0 / layout 0):
  * TopK/8 beats BP-row/8, and is noninferior to U8.
  * Mix-TopK beats A8 and BP-row/16, and is worse than U16.
  * Final-test means are within 0.001 of the same runs' development values.
* **Graph and layout checks** (data seed 0; not training-seed replicates) reproduce every ordering. The
  graph/layout spreads are about 0.001 for every method, against gaps of 0.004–0.012.

**5. What remains a support limitation, an optimization question, or a backend / resource cost rather than an
established cause?**
* **Support limitations (established for the construction, not as optimality statements):**
  * BP-row/8 per-output coverage sits at the K8 union bound.
  * On the reduced task, more than 78 % of BP-row/8 and TopK/8 interaction terms have no eligible neuron.
  * TopK/8 leaves inputs of single-record outputs uncovered (§7).
* **Optimization or unresolved:**
  * Why U8 (34 % bias-only, magnitude tiles) trains better than BP-row/8 is not established.
  * Mix-TopK's interaction plateau (0.71–0.78 at 1,000 updates) sits above an unproven optimum; its oracle is
    exact, so the remaining gap mixes optimization and support.
  * The block-additive floor is a reference, not the best approximation within these support families.
  * T64 remains unresolved.
* **Backend cost:** the mixed allocation's slowdown comes from kgen K8 efficiency and three launch sets, not
  from the support. A dedicated kernel could change the time ranking; it was not built (§8).

**Frontier** (locked final test; time = complete step, graph seeds 0/1/2):

| point | slots | step ms | bias-only | layer / per-output triples | final test |
|---|---:|---:|---:|---:|---:|
| TopK/8 (indexed K8) | 100M | 629 | 0 | 19.47 % / 0.024 % | 0.2114 |
| U8 0.125 (wide) | 100M | 691–697 | 34.4 % | 37.6 % / 0.105 % | 0.2118 |
| U16 0.25 (wide) | 200M | 814–821 | 31.7 % | 60.90 % / 0.549 % | 0.1679 |
| Mix-TopK (grouped) | 200M | 1,075 | 0 | 61.64 % / 1.115 % (= dense envelope) | 0.1724 |
| dense | 800M | 1,553–1,568 | 0 | 61.64 % / 1.115 % | 0.1130 |

* **TopK/8:** fastest point and best 100M quality, with no bias-only rows. It is noninferior to U8 in loss.
* **U8:** dominated by TopK/8 in time and tied in loss. It keeps more coverage, at the cost of bias-only rows.
* **U16:** best 200M quality and time, with bias-only rows.
* **Mix-TopK:** the only 200M point with full width and full dense-envelope coverage. It is slower and worse in
  loss than U16.
* **Dominated at 200M:** BP-row/16 (865 ms, 0.1778) and A8 (978 ms, 0.1772) are beaten in loss by both U16 and
  Mix-TopK. A8 is also beaten by U16 in time.
* **Uncertainty:** loss differences under about 0.003 are not resolved with three seeds.

## 1. Workload, provenance and protocol

* **Model** (fixed): D = G = 1024, nominal K64 envelope, logical C256, S64, R256, Q 4,096; H = 12,500,000, W0 =
  800,000,000.
  * Every original (g, n), record (g, j) and set q is preserved.
  * M 8192, Mean(ReLU), BF16 inputs / parameters / moments, FP32 accumulation and main_grad, split dZ.
  * L1AdamW lr 1e-4, no L1, weight decay 0, stochastic rounding on.
  * The actual alpha-0 checkpoints of graph seeds 0/1/2 (with their proj / head) are hash-verified at every
    load. Retained values come from the same checkpoint; nothing is reinitialized at a K8 or K16 scale.
* **Seeds kept separate:** graph seed (envelope + checkpoint), layout / candidate seed (`#ls`), data seed and SR
  seed (equal per run), interaction init seed.
* **Provenance.**
  * All timing and learning records come from source-clean frozen checkouts:

    | checkout | commit | stages |
    |---|---|---|
    | `meas1` | `a0a7b4b` | 1–2 |
    | `meas2` | `3269277` | 3–6 |
    | `meas3` | `27d749d` | continuations, sanitizer, suite |
  * The 103 timing records are all `dirty = false` and use one library per bundle, the same as the parent:
    kgx `c400495d…`, kgen `023754cd…`. No kernel changed.
  * Every record carries checkpoint hashes, `--canonical-start --identity` canonical hashes after reset,
    restore checks and immutable-metadata checks.
* **Timing protocol.** `bench_supergroups.py`:
  * screens 2 × 2 × 2 = 8 restored samples, confirmations 4 × 2 × 3 = 24;
  * fresh processes; one session per graph seed with dense first / middle / last;
  * layout replicates averaged within a graph before the graph-level interval; the within-run range is
    reported separately.
* **Plans and statuses.** Every stage wrote `plan.json` before running and appends to `status.jsonl`. Selection
  rules were written before any result (`selection_rules.json`). Every selection was applied mechanically by
  `kwa_select.py`, and its inputs are recorded in `*_selection.json`.
* **Failures and deviations (kept):**
  * `w8_confirm` / `m25_confirm` first failed at plan construction. The builders read the selection's `rule`
    key, which held the rule's description; the spec is under `rule_spec`. They were rerun in stage 5 from
    derived selection files (`rule ← rule_spec`) with the same frozen code.
  * The two 4,000-update interaction continuations first failed at `--resume`. The parent's resume path called
    `.clone()` on the optimizer's integer step. The fix is in `27d749d`; a straight 40-step run and a
    20 + resumed-20 run give identical curves (`resume_check/`).
  * A restart of the stage-7 driver killed one in-flight continuation, which was rerun.
  * The interaction seed check and the continuations are labelled extras beyond the predeclared rule.

## 2. Errata to the parent study (`RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md`)

The parent's raw files are unchanged; corrected statistics come from rerunning the corrected code on the same
supports (`errata_cov/`, `qcheck/`).

### 2.1 Distinct supports must be global

`rsn_coverage.sharing()` keyed supports by their 64-bit *local* mask. That key is exact within one record or one
set. It is wrong at output and layer scope, where equal local masks of different sets select different global
inputs. Supports are now identified by an exact D-bit bitmap of `feat[q][enabled local coordinates]`.
`labelled_local`, i.e. (q, local mask), is kept as a separate, documented alternative that does not merge
identical global supports across sets.

| support (graph 0) | scope | parent (local key) | labelled (q, local) | **global** | groups / support | neurons / support |
|---|---|---:|---:|---:|---:|---:|
| FP-mag | record | 194,512 | 194,512 | 194,512 | 4.02 | 64.3 |
| FP-mag | set | 16,384 | 16,384 | 16,384 | 47.7 | 762.9 |
| FP-mag | output | **4,096** | 194,512 | **194,512** | 4.02 | 64.3 |
| FP-mag | layer | **4** | 16,384 | **16,384** | 47.7 | 762.9 |
| BP-row | all scopes | 781,312 | 781,312 | 781,312 | 1.00 | 16.0 |
| U16 0.25 | output | **16,325** | 376,518 | **328,520** | 2.38 | 38.1 |
| U16 0.25 | layer | **16** | 61,902 | **57,807** | 13.5 | 216.2 |
| A8 0.25 | output | **98,726** | 609,658 | **609,658** | 1.28 | 20.5 |
| A8 0.25 | layer | **247** | 222,148 | **222,148** | 3.52 | 56.3 |

* **Most affected:** FP and U16 / A8 at output and layer scope.
* **BP-row unaffected:** its local masks were already all distinct.
* **Labelled vs global** differ for U16, where empty and full-set supports coincide across sets.
* **Multiplicity.** A row group's 16 neurons are one support of multiplicity 16, never 16 supports. The sampled
  tuple multiplicity is now reported both per row group and per distinct global support.
* **Coverage and hashes unchanged:** mask hashes and exact layer pair / triple counts are identical to the
  parent's for all four supports.
* **Regressions** (`tests/test_weighted_mixed_supports.py`): equal local masks in different sets stay distinct;
  different masks / sets selecting the same global inputs merge; repeated supports across records, outputs,
  partial groups, K8 / K16 / K64 and padded rows; brute-force support and tuple counts against the pipeline.

### 2.2 Total Q and occupied sets

`local_supports.geometry()` inferred Q from the largest assigned q. It now takes the authoritative Q
(`sg_feat.shape[0]`), keeps m_q and b_q of length Q, and reports total / occupied / empty counts. BP-row stream
offsets are cumulative over sets in q order, and an empty set draws no permutation. So the fix shifts no q
identity and no RNG stream: every parent support fingerprint is reproduced (`qcheck/`).

| model (graph 0) | parent Q | total Q | occupied | empty |
|---|---:|---:|---:|---:|
| reduced H 262,144 (interaction) | 4,084 | **4,096** | **1,024** | 3,072 |
| full H 12.5M (graph seeds 0/1/2) | 4,096 | 4,096 | 4,096 | 0 |

The parent's §7 sentence "All 4,096 sets are occupied" is wrong for the reduced workload: 1,024 of 4,096 are.
Regressions: leading / interior / trailing empty sets, sparse occupation, round reshaping and the reduced
geometry's 1,024 / 4,096.

### 2.3 Tightened statements

* **Term-to-neuron matching** is a constructive eligibility diagnostic. "No eligible neuron" is evidence of a
  structural restriction. A failed matching does not establish the optimal loss, and does not prove that a
  training plateau contains no optimization error. The parent's "support-limited gap" for U16 / BP-row is
  downgraded to "consistent with a support limitation".
* **The block-additive floor** is a reference for explained interaction. It is not the best approximation floor
  of BP-row's or U16's support family; no bound for those families was attempted, so their optimum stays
  unresolved.
* **Index cost.** "No index overhead / indexed is ×0.991" compared an indexed backend with a different index-free
  backend. That does not isolate index cost or establish an overhead below 1 %. It is a matched-function,
  different-backend control and is labelled as such here (§3, FP-mag/8 indexed vs wide).
* **FP vs BP-row per output:** FP's per-output triple coverage is 0.060 % and BP-row's 0.220 %; they are not
  equal. Task-loss differences within the seed intervals are uncertainty, not strict Pareto domination.
* **H2 threshold counts.** The parent said every sampled H2 row crossed the old `Var > 0.01` threshold after
  rescaling. The actual counts are 34,595 → **65,180** of 65,536 at the start, and 65,175–65,179 after 256
  updates. Restored amplitude alone is not improved representation or accuracy.

## 3. K8 baseline (stage 2)

`k8_cov/`, `k8_timing/`, `k8_q32/`, `k8_q256/`, `k8_graph/`.

* **Budget:** BP-row/8 keeps 8 distinct inputs per valid 16-neuron row group, exactly 100,000,000 valid slots
  (rho 0.125), with every neuron and bias. Nominal envelope C256 vs executed sharing C16 (one kgen K8 record per
  row group).
* **Timing:** §0.1. The phase split for BP-row/8 is fwd 49, dX 379, dW / db 200 and optimizer 5 ms.
  * Same-function control: FP-mag/8 runs at 646.5 ms on indexed K8 against 823.5 ms on the index-free 16 × 8
    kernel. That is ×0.785, a backend difference, not an index cost.
* **Memory:** peak allocated 1.24 GiB (U8 1.36, dense 7.80). Values 190.8 MiB, plus 25.8 MiB of index records,
  maps and work lists.
* **Coverage:**

  | support | layer triples | refined prediction | quads [95 % CI] | per-output triples pooled [95 % CI] |
  |---|---:|---:|---:|---:|
  | BP-row/8 (6 graph × layout) | 19.537–19.541 % | 19.540 % | 0.117–0.119 % | 0.0236–0.0240 % |
  | U8 0.125 | 37.58–37.62 % | – | 1.362 % | 0.105–0.106 % |
  | FP-mag/8 | 0.881 % | 1.023 % *(invalid model)* | 0.0052 % | 0.0107 % |

  The per-output values are estimates from 2M sampled triples per output, summarized over the 38 fixed outputs;
  they are not exact counts. The bound applies to uniform K8 row-group supports.
* **Quality:** §0.1. The 32-update screen ranks the three within 0.0004 (0.2604–0.2608 with energy); the
  ordering separates only by 256 updates.
  * Energy restoration helps every K8 support by 0.0086–0.0101 at 256 updates.
  * Clipping at s_max 8 is ≤ 0.1 % of connected rows, below the 1 % trigger, so the cap grid was not reopened.
  * Participation ratio is about 158–163 throughout. Relative input variance at the start is 0.13 (none) and
    1.00 (energy).

## 4. Weight-aware local subsets (stage 3)

### 4.1 Policies (`experimental/local_supports.py`)

* **Feature importance.** a[u, j] = Σ over valid neurons c of u of W_pre[u, c, j]². It is computed in FP64 by
  explicit sequential adds over c, with padded neurons masked, so it is independent of chunking (tested).
  score(A) adds a over A in ascending coordinate order.
* **WRS-k(L).**
  * Candidate 0 is the exact BP-row/k subset; L = 1 reproduces BP-row bit-for-bit (fingerprint tested).
  * Candidates 1..L−1 are uniform k-subsets keyed by their candidate id alone, streamed in 65,536-group chunks.
    So the first 4 candidates of L16 are those of L4, and retained energy is nondecreasing in L (tested).
  * The highest score wins; ties go to the smallest candidate id. Duplicate candidates are kept: 0 of 781,312
    groups at L4, 1 at L16.
* **TopK-k.** The k largest a, ties to the lower coordinate (stable sort).
* Every row keeps exactly k sorted distinct in-envelope coordinates; q, bias and pooling are unchanged.
* The BP coverage model is not used as a prediction for these policies; their supports are measured.

### 4.2 Screen and selection (`w8_screen/`, `w8_dev/`, `w8_advance_selection.json`, `w8_final_selection.json`)

| graph 0 | step (same session) | ×BP-row/8 | layer triples | per-output triples | retained energy | 32-update loss none / energy | selection time |
|---|---:|---:|---:|---:|---:|---:|---:|
| BP-row/8 | 625.3 | 1.000 | 19.540 % | 0.0240 % | 0.125 | 0.2629 / 0.2607 | 0.3 s |
| FP-mag/8 | 640.1 | 1.024 | 0.881 % | 0.0107 % | – | 0.2626 / 0.2604 | 0.2 s |
| WRS-8(4) | 626.3 | 1.002 | 19.470 % | 0.0237 % | 0.140 | 0.2627 / 0.2604 | 5.7 s |
| WRS-8(16) | 626.9 | 1.002 | 19.469 % | 0.0235 % | 0.152 | 0.2625 / 0.2603 | 15–17.5 s |
| TopK-8 | 624.9 | 0.999 | 19.465 % | 0.0239 % | 0.205 | 0.2619 / **0.2598** | 1.1–3.0 s |

* **Advance:** all three candidates met the preference (within 10 % in time and ≥ 80 % of both coverages), and
  TopK-8 had the best score. No tradeoff candidate qualified (none was better and outside the preferences).
* **Final weighted rule:** W* = TopK (energy), with development loss 0.2130 ± 0.0027 against its unrescaled
  version's 0.2231 ± 0.0014.
* **Setup cost:** selection and packing are a one-time 1–17 s outside every step. Once absorbed into the
  parameters they have no recurring cost; the layer builds in 9–24 s.

### 4.3 Why coverage barely moves

Retained energy rises only from 0.125 to 0.205 even for TopK. The checkpoint's per-coordinate energies are
nearly flat within a row group, so magnitude picks behave almost like random picks at the layer level. Layer
triples drop only 0.07 pp. Per-output, TopK's picks are not balanced like BP's permutations:
* in the full model, per-output features fall 978 → 954;
* in single-record outputs (reduced task), the uncovered inputs matter (§7).

## 5. Mixed allocation at 200M slots (stage 4)

### 5.1 Construction (`experimental/mixed_supports.py`)

* **Classes.** In every original record: the rank-0 row group is the K64 anchor; six full non-anchor groups,
  chosen by a keyed order independent of the weights, are K8; every other valid group, including the partial
  tail, is K16.
* **Budget.** 16·64 + 96·8 + (n − 112)·16 = 16 n.
  * Records with n < 112, or with fewer than seven row groups (C < 112), are rejected (`Unsupported`), never
    silently changed (tested).
  * Primary totals and per-output counts are exact (tested): 768 K64, 4,608 K8 and 6,831 / 6,832 K16 per
    output, which fit the balanced KLayout counts.
* **Subsets.**
  * Mix-BP: a separate BP stream per (q, k) stratum.
  * Mix-TopK: TopK within the same class map. Both variants share one class map (equal `class_fingerprint`).
* **Disclosure:** this changes fan-in allocation and anchor existence together, and its selection differs from
  A8's (A8 = the best-|w| anchor group plus magnitude 8-wide tiles).

### 5.2 Grouped execution (`experimental/kgx_grouped.py`)

* **Packing.** Three packs (K64 / K16 / K8, each C16) of the original row groups in canonical order.
  * Records are feat[q][env] and buckets are the runs of one output in one set; a duplicate (q, g) bucket or a
    partial group that is not last is rejected.
  * One flat BF16 weight and one flat bias hold the three packs back to back; 15,872 padded slots stay exactly
    zero.
* **Function.**
  * One X^T transpose. Three forward launches add into one FP32 ysum [G, B], divided once by the *original*
    counts.
  * One coefficient dy / n_original for every type. One FP32 dX^T is zeroed once, accumulated by all three dX
    launches, and cast once.
  * The kgen kernels only accumulate into ysum / dX^T / dW, and use type-local counts only for validity.
  * The dense-masked function validated every mixed policy first; all quality runs use it.
* **Correctness** (§6): the grouped layer equals the masked function and an FP64 reference.
* **Charged costs.**
  * Index records 23.8 MiB, buckets 2.3 MiB, canonical env maps 11.9 MiB plus 3.8 MiB of group maps,
    block-to-group maps 6.0 MiB and work lists 8.2 MiB, against values 381.5 MiB. With the 1.25 MiB feat / bucket
    graph metadata that is 57.2 MiB in total, versus 43.9 MiB for BP-row/16 (index records 23.8, canonical map
    11.9, work lists 6.9).
  * Peak allocated 2.21 GiB (BP-row/16 2.19, A8 2.18, U16 2.27).
  * 6 launches per step plus 1 optimizer call.

### 5.3 Screen, development and confirmation (`m25_screen/`, `m25_dev/`, `m25_confirm/`)

| graph 0 | step, 8-sample screen | layer / per-output triples | 32-update none / energy | 256-update development (energy) |
|---|---:|---:|---:|---:|
| BP-row/16 (uniform, unweighted) | 864.1 | 58.90 / 0.221 % | 0.2611 / 0.2590 | 0.1791 *(parent)* |
| TopK/16 (uniform, weighted) | 862.2 | 58.74 / 0.217 % | 0.2599 / 0.2580 | **0.1713** |
| Mix-BP (mixed, unweighted) | 1,076.2 | 61.636 / 1.115 % | 0.2611 / 0.2592 | – |
| Mix-TopK (mixed, weighted) | 1,074.6 | 61.636 / 1.115 % | 0.2601 / 0.2583 | 0.1739 |
| A8 0.25 | 974.8 | 61.636 / 1.115 % | 0.2608 / 0.2589 | 0.1778 *(parent, none)* |
| U16 0.25 | 812.9 | 60.90 / 0.549 % | 0.2608 / 0.2592 | 0.1687 *(parent)* |

* **Advance:** Mix-TopK (best mixed) and TopK/16 (better than Mix-BP).
* **Final mixed candidate:** Mix-TopK, since it was the only advanced mixed policy.
* **Confirmation** (24 samples, graphs 0/1/2 × layouts 0/1):
  * Mix-TopK 1,075.3 ± 3.5 ms, layout spread ≤ 0.9 ms;
  * layer / per-output coverage equals dense at every graph and layout;
  * quality (data seed 0) 0.1745–0.1757.
* **Weighting effect within a 2×2 cell:** weighting helps the uniform K16 allocation (TopK/16 0.1713 vs BP-row/16
  0.1791) more than it helps the mixed one in the screens (0.2583 vs 0.2592 at 32 updates).
* **Not run:** a fourth 256-update cell (Mix-BP) by the advancement rule, and a TopK/16 locked test, since at
  most two new methods go to the final test.

**Per-type attribution** (24 samples, graph 0, median ms):

| run | fwd | dX | dW / db | optimizer | step | dX K64 / K16 / K8 | dW K64 / K16 / K8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-TopK grouped | 90.7 | 672.7 | 294.0 | 5.0 | 1,063.9 | 162.9 / 351.3 / 158.4 | 44.8 / 115.0 / 134.0 |
| BP-row/16 indexed | 68.8 | 603.0 | 185.8 | 5.1 | 864.5 | – | – |
| A8 wide | 121.9 | 445.5 | 395.0 | 5.1 | 969.6 | – | – |
| U16 wide | 116.5 | 427.1 | 265.2 | 5.0 | 817.0 | – | – |

* **Per-slot dX** is similar for K64 and K16, about 3.2 ms per million slots.
* **K8 is the expensive type:** its dW is 3.6 ms per million slots against 1.0 for K16, and its dX is 4.2.
* **Adapter phases** (zero, transpose, finalize, coefficient, cast) total under 1 ms.

## 6. Correctness gates

* **Focused tests:** `tests/test_weighted_mixed_supports.py` (18). It covers:
  * global supports and empty-Q handling;
  * WRS / TopK: L1 = BP-row, nesting, nondecreasing energy, ties, padding masking, chunking and device
    independence;
  * mixed counts and tails, rejection, primary-geometry totals;
  * grouped vs masked vs FP64: y, dX, dW, db with token tails B 40 / 64 and n175 / 176 tails; unequal type
    populations; the single dX cast; main_grad accumulation, clearing and scratch reuse; NaN propagation;
    B % 8 rejection;
  * optimizer identity over 3 steps with SR off; canonical parameter / mask / main_grad / moment identity;
    snapshot restore and full state round trip; rejection of type populations that do not fit.
* **Existing suites:** `tests/test_local_supports.py` (19) and `test_row_normalize_oracle.py` (10) pass with
  the Q change.
* **Tolerances** (FP64 reference): y and db 1e-6 relative, dX and dW 2e-5, moments after 3 updates 2e-2.
* **Cancellation:** with near-zero biases the output stays within 1e-5 of FP64.
* **Stochastic rounding:** representation-dependent RNG indexing and atomic summation order mean trajectories
  are not bitwise identical across formats with SR on; none is promised.
* **Compute Sanitizer** (memcheck, racecheck, initcheck, synccheck): new `kwa` workload (grouped Mix-BP / WRS /
  TopK; indexed WRS / TopK K8 / K16), plus `rsn` and `kgen`. Results in §9.
* **Full suite:** §9.

## 7. Interaction diagnostic (reduced H 262,144 / C256, T16)

`interaction/`, `interaction_extra/`, `interaction_cont/`; workload as in the parent (corrected Q above).
Explained interaction = 1 − nMSE / additive floor (a reference, not an optimum). The values are on development
held-out batches 49,000–49,003; the extras also report the reserved batches 59,000–59,007, which agree within
0.01.

| support | lr 3e-3, 1,000 | lr 1e-2, 1,000 (init 17 / 18 / 19) | lr 1e-2, 4,000 | terms without eligible neuron (orders 2 / 3 / 4) |
|---|---:|---:|---:|---:|
| dense | 0.78–0.88 | 0.99 | – | 0 |
| Mix-TopK | 0.53–0.54 | 0.71–0.78 (all three seeds) | – | 0 |
| A8 0.25 | 0.31–0.45 | 0.66–0.69 | – | 0 |
| BP-row/8 | 0.11–0.12 | 0.13–0.18 | 0.16–0.20 | 3,219 / 4,001 / 4,088 of 4,096 |
| TopK/8 | −0.42 to −0.34 | −0.38 to −0.27 | −0.32 to −0.27 | ~3,260 / 4,005 / 4,088 |

* **Oracle.** The Mix-TopK anchor oracle is exact: nMSE 2.7e-14 against BF16-input targets, with the analytic
  readout. The perturbed oracle (eps 0.1) recovers to 1.000. So Mix-TopK can represent the T16 targets with
  its anchors; ordinary training reaches 0.71–0.78 at 1,000 updates.
* **Ranking.** Mix-TopK > A8 at both learning rates and all three seeds, so the comparison was not ambiguous
  under the predeclared rule.
* **TopK/8.** It stays below the additive reference after 4,000 updates. Its linear (order-1) nMSE plateaus at
  0.134; BP-row/8's falls to 0.0006.
  * On this geometry TopK/8 covers 56.6 of each output's 64 inputs (min 49; independent-pick prediction 56.4).
    BP-row/8 covers all 64 (`feature_cover_reduced.txt`).
  * The gap is therefore a support effect specific to sparse outputs. It is not the slower Adam progress that
    larger initial norms would cause.
* **Not run:** the T64 case.

## 8. Optimization pass

Not performed (optional). The attribution in §5.3 points to kgen K8 efficiency per slot and to separate
per-type launches, i.e. per-type slab staging. Neither is fixable inside the adapter without new kernel work,
which the assignment did not require. No reported time depends on an unimplemented speedup.

## 9. Sanitizers and full suite

* **Compute Sanitizer** (`sanitizer/summary.tsv`, frozen `27d749d`): **12 / 12 PASS**. Workloads: `kwa`, the
  new one (grouped Mix-BP / Mix-WRS(4) / Mix-TopK at C256 with n175/176 tails; indexed WRS(4/16) / TopK at K8 and
  K16; split / BF16 dZ; B 8 / 40 / 136; non-finite upstream gradients; main_grad accumulation; L1AdamW); `rsn`,
  whose local-support loop now includes WRS (L1) and TopK; and `kgen`. Each ran under memcheck, racecheck,
  initcheck and synccheck.
* **Full suite** (`pytest sparse_pooling/tests` at `27d749d`, capped scope): **3,188 passed, 32 skipped, 1
  failed** (`tests/full_suite.txt`).
  * The failure is `test_sanitizer_runner.py::test_interrupt_terminates_children_and_exits_nonzero[2-130]`. Its
    fake sanitizer jobs did not start within the test's 30 s deadline.
  * This study changed `run_sanitizer.sh` only to accept the new workload name. The whole file
    (24 tests) passed in four isolated reruns at the same commit, one of them inside the same kind of systemd
    scope (`sanitizer_runner_rerun.txt`).
  * It is recorded as a load-dependent timeout in the full run, not as fixed.

## 10. Evidence layout

`results/weighted_random_supports_anchors/`:
* `selection_rules.json` (predeclared), `*_selection.json` with their inputs, and `stage*.sh` driver scripts;
* `*_commit.txt`;
* one directory per stage with `plan.json`, `manifest.json`, `status.jsonl` and the raw JSON / logs;
* `tables.md` and `csv/`;
* `final_states.sha256` and `interaction_states.sha256`. The final-run states (34 GB) and interaction states
  (1.6 GB) are kept outside git in `../kwa_runs/final/state`, `../kwa_runs/interaction/state`;
* `resume_check/`, `feature_cover_reduced.txt`, `sanitizer/`, `tests/full_suite.txt`.
