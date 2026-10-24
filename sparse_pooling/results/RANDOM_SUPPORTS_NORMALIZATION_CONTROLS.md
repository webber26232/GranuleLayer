# GranuleLayer: randomized supports, signal normalization and optimization controls

Branch `k16-random-support-normalization-controls`, started from `k64-width-preserving-sparsity` @ `ab649a9`. This is
an experiment branch: production defaults and auto-dispatch are unchanged, and nothing is merged into main. Raw
evidence is in `results/random_supports_normalization_controls/`; generated tables are in its `tables.md` and
`csv/`.

Every number below is a measurement on one RTX 3090 (sm_86) unless it is labelled as a *prediction*, *model* or
*bound*. The learning workloads are synthetic diagnostics, not accuracy claims.

## 0. Answers

**1. Which sampling rule gives which coverage, and does the actual mapping match its model?**

| support (fan-in 16, 200M slots) | layer-wide triples: predicted (refined) / measured | quadruples: predicted / measured [95 % CI] | per-output triples, median of 38 outputs |
|---|---|---|---|
| FP-mag (parent P16) | 5.01 *(model, invalid here)* / **4.05 %** | 0.065 / 0.063 % | 0.060 % |
| FP-balanced | 5.01 *(invalid)* / **4.05 %** | 0.065 / 0.063 % | 0.061 % |
| RP-record (3 layout seeds) | 37.13 / **37.12–37.13 %** | 0.735 / 0.735–0.739 % | 0.060 % |
| RP-row (3 seeds) | 58.75 / **58.74–58.75 %** | 2.379 / 2.381–2.386 % | 0.216–0.217 % |
| BP-row (3 seeds × 3 graph seeds) | 58.91 / **58.90–58.92 %** | 2.387 / 2.383–2.391 % | 0.220–0.222 % |
| U16 0.25 (free tiles, 31.7 % bias-only) | – / 60.90 % | – / 5.14 % | 0.544 % |
| A8 0.25 / dense potential | – / 61.64 % | – / 5.56 % | 1.116 % |

* **Randomized supports match their models.** The refined independent-round model, using the actual m_q and
  b_q, matches the exact measurement to within ±0.01 pp on triples. On quadruples the predictions fall inside
  the 95 % intervals. The supplied ideal-quota table is within 0.003 pp of the refined one.
* **Fixed partitions do not match the model.** The model predicts 97.7 % of pairs; FP covers 45.0 %. The four
  tiles are ranges of each set's *sorted* feature list, so the same nearby-id tuples fall into one tile in every
  round. The one-tile union bound (5.14 % triples) is a bound only for this one-partition construction.
* **Per output, randomization barely helps.** Every K16 row-group support has at most 763 records per output.
  The union bound is 0.239 % of triples.
  * RP-record keeps one partition per record, so its per-output coverage equals FP's (0.060 %).
  * RP-row and BP-row reach 90–93 % of the union bound (0.22 %). That is 0.40× U16 and 0.20× A8.

**2. Does randomized K16 give a useful time / width / coverage / task tradeoff against U16 and A8?** It does
for time, width and layer coverage. It does not for per-output coverage or task loss.
* **Time** (confirmation, graph seeds 0/1/2 × layout seeds 0/1): BP-row on indexed K16 takes 865.2 ± 2.1 ms.
  * That is ×1.060 ± 0.007 of U16 0.25 (815.9 ms) and ×0.885 of A8 0.25 (977.2 ms).
  * It is 3 % faster than FP on the *same* indexed implementation (×0.969).
* **Width:** no row is bias-only (U16: 31.7 %).
* **Coverage:** layer-wide triples are 58.9 % (U16: 60.9 %); per output 0.22 % (U16 0.54 %, A8 1.12 %).
* **Task (fine-tuning, 256 updates, 3 seeds, no normalization):** BP-row ends at 0.1823 ± 0.0015, worse than
  U16 0.1746 ± 0.0007 and A8 0.1778 ± 0.0018.
* **Interaction (T = 16, 4,000 updates):** BP-row explains 0.53–0.65 of the interaction, against 0.78–0.82
  for A8.
* **Index cost is not visible at this size.** FP-mag gives the same function on indexed K16 and the index-free
  16 × 16 kernel, and indexed K16 is faster: ×0.991 (893 vs 901 ms). The index records are 23.8 MiB.

**3. How much of the backbone's weak width is signal scale, and does loss improve?** Most of the amplitude gap
is scale, and loss improves only partly.
* **Amplitude:** after the one-time energy restoration (s_max 8), H2 0.05's median input-driven variance rises
  from 0.051 to 0.973 of each neuron's own dense row. The parent's `Var > 0.01` count goes from 34.6k to
  65.2k of 65,536.
* **Diversity is unchanged.** The amplitude-normalized activation participation ratio stays at 158 whatever
  the treatment.
* **Loss after 256 updates (3 seeds):**

  | Support | No rescaling | Energy restoration | Change |
  |---|---:|---:|---:|
  | H2 0.05, cap 8 | 0.2558 | 0.2455 | −0.0103 ± 0.0018 |
  | H2 0.05, cap 16 | 0.2558 | 0.2437 | −0.0121 ± 0.0012 |
  | H2A 0.10 | 0.2335 | ≈ 0.2335 | not significant at caps 8 and 16 |

  The dense model reaches 0.1136. So fair scaling removes the *amplitude* deficit but most of the loss deficit
  of the two-input backbone remains.
* **Other supports** (energy − none after 256 updates):

  | Support | Change |
  |---|---:|
  | U16 0.25 | −0.0059 ± 0.0019 |
  | BP-row | −0.0032 ± 0.0021 |
  | FP-mag | −0.0032 ± 0.0026 |
  | A8 0.25 | +0.0008 ± 0.0022, none |

**4. Can the anchor model represent the teacher, and can ordinary training find it?** It can represent it.
Ordinary training finds most of it, but needs about 4× the parent's budget or a larger learning rate.
* **Representation:** the teacher-mapped oracle on A8 0.25's 16 anchors reproduces the T = 16 target to a
  normalized MSE of 8e-6 against FP32 inputs. Against the BF16-rounded inputs the layer reads, it is 2.7e-14.
  * The perturbed oracle (ε = 0.1 of the init std on every enabled weight) returns to an explained interaction
    of 1.000 within 1,000 updates.
* **Ordinary training, T = 16:**

  | Run | Explained interaction (orders 2/3/4) |
  |---|---|
  | 1,000 updates (parent protocol), init seeds 17/18/19 | 0.31–0.46 |
  | same runs continued to 4,000 updates | 0.78–0.84 |
  | 4,000 updates at lr 1e-2 | 0.94–0.97 |
  | dense, 4,000 updates | 0.998 |

  The parent's "anchors carry coverage, not capacity" reading of the T16 gap was mostly an *optimization*
  budget effect.
* **U16 and BP-row have a different, support-limited gap.** About 70 % of their terms have no eligible
  neuron, so the matching oracle is not exact. Their ordinary-training plateaus (0.42–0.65) are a support
  limitation, not an optimization one.
* **T = 64:** dense is representable (match oracle 0.999) but reaches only 0.36–0.70 in 1,000 updates, so
  this is an unresolved optimization diagnostic.

**5. Does correcting the refresh budget change the conclusion?** No, only its accounting.
* **Budget:** with B0 immutable and remaining = B0 − enabled, the H2A 0.10 T16 run stays within 208 slots of
  B0 at every refresh, under one legal tile (256 slots), and ends 128 below. The parent ended 2,352 below; a
  legacy-rule rerun ended 1,328 below.
* **Loss and support behaviour are unchanged:** held-out 0.23391 vs 0.23391, 4,916 tiles changed vs 4,907, and
  the support still oscillates.

**Pareto set** (complete step, structural width, per-output / layer coverage, held-out loss). The full table
with all criteria is in §9.
* **U16 0.25 (wide), 816 ms:** the fastest support and the best sparse fine-tuning loss (0.169 after energy
  restoration), with 31.7 % bias-only rows.
* **BP-row K16 (indexed), 865 ms:** the cheapest support with every row connected and near-full layer-wide
  coverage (58.9 %); per output only 0.22 %.
* **A8 0.25 (wide), 977 ms:** full per-output coverage (1.12 %), representable interactions, and the best
  interaction learning with a longer budget.
* **BP-row K8 (indexed, ρ 0.125), 627 ms** (conditional extension, graph seed 0): faster than U8 0.125
  (688 ms) with no bias-only rows (U8: 34.4 %), but 19.5 % layer-wide triples (U8: 37.6 %). Task loss was not
  measured.

Dominated or neutral: FP-mag / FP-balanced, which are slower than BP-row on the same implementation with
1/14.5 of its layer-wide triple coverage. H2 / H2A remain far behind on loss even after fair scaling.

## 1. Provenance, workload and protocol

* **Source checkouts:** results come from frozen, source-clean detached checkouts, written outside them
  (`GRANULE_RESULTS_ROOT=../rsn_runs`), and every record carries its provenance.
  * `2f95faa`: sanitizers, coverage, screen, first normalization comparison.
  * `e904796`: refresh, confirmation, graph-seed coverage, K8 extension, normalization set, interaction.
  * `52820b7`: cap-16 sensitivity.
  * Later commits change only the report and job tooling.
  * The checkpoint manifest line in `stage1.sh` failed because of an unescaped `$PY`. `wps_checkpoints.py` was
    then run by hand from the same frozen checkout; it is CPU-only.
  * The first refresh attempt at `2f95faa` ran out of memory in the new remap verification, which summed
    800M-element bool tensors. Those records are kept in `refresh_oom_attempt_2f95faa/`; the fix
    (chunked counts, commit `52ea34a`) was tested before the rerun.
* **Kernel libraries:** every timing record used one build per bundle. That is kgx
  `production-c400495d8202e34be70f`, the same as the parent study, and kgen `production-023754cd60595afae7e9`.
* **Checkpoints:** the actual α = 0 parent checkpoints for C256 graph seeds 0/1/2. File sha256 values are
  recomputed and match their sidecar records (`checkpoints/manifest.json`); every driver re-verifies the
  weight sha256 before use.
* **Model:** D = G = 1024, S = 64, R = 256, Q = 4096, H = 12.5M, W0 = 800M, C256 envelope, M = 8192, Mean(ReLU).
  * BF16 inputs and parameters, FP32 accumulation and main_grad, split-dZ MMA.
  * L1AdamW lr 1e-4 with BF16 moments and stochastic rounding, no L1, with the optimizer race fix.
  * Init seed 17, input seed 23.
* **Seeds:** graph seed (envelope graph and checkpoint), layout seed (local subsets only), init / training
  stochastic-rounding seed and data seed are separate arguments everywhere.
* **Timing protocol:** `bench_supergroups.py --canonical-start --identity`, unchanged. Each format runs in a
  fresh process: warm-up, restore of the pre-warm-up values, reset of optimizer state, rounding counter and
  main_grad, then capture of the canonical identity.
  * Screen: 2 rounds × 2 reps × 2 steps = 8 samples. Confirmation: 4 × 2 × 3 = 24.
  * The dense control runs first, in the middle and last of every session. Session anchors agree within
    ±0.3 % (screen 1,542–1,543 ms, confirmation 1,549–1,554 ms).
* **Machine-readable manifest** (architecture, policies, graph / layout / training seeds, ideal predictions,
  bounds, cases): `*/manifest.json` (`rsn_cases.manifest()`).
  * Every stage's plan was written before its jobs ran (`plan.json`, `status.jsonl` per job).
  * The selection rules (`selection_rules.json`) were declared before any result was read. The selections
    derived from them are in `confirm_selection.json`, `norm_selection.json`, `interaction_selection.json`
    and `cap16_selection.json`.

## 2. Refresh-budget drift (`experimental/support_refresh.py`, `benchmarks/wps_refresh.py --rule exact`)

**The rule.**
* B0 = round(ρ W0) is immutable.
* After the prune, the *entire* enabled valid-slot count is recounted (`MaskSet.enabled_slots`). Mandatory
  tiles, anchors and backbone coordinates are each counted once, and a tile's backbone overlap is charged once.
* Regrowth uses first fit against remaining = B0 − enabled_after_prune, so earlier remainders are recovered.
* Invariants checked after every refresh: enabled ≤ B0, and shortfall < 16b (the largest legal tile cost).
* If the eligible candidates cannot meet that bound (for example when excluding just-pruned tiles removes
  every fitting candidate), the prune is *reduced*: pruned tiles are re-enabled, strongest first. If the bound
  still fails, the refresh is recorded as `infeasible`. The target is never lowered.
* Every refresh records:
  * B0, the slot counts before / after prune / after regrowth;
  * the carry in and carry recovered;
  * any rollback or infeasibility;
  * mask and nonzero densities;
  * a state-remap check: surviving values and both moments unchanged, grown slots zero, nothing outside the
    mask.

**Tests** (`tests/test_support_refresh.py`, 14 tests):
* first fit against a scalar loop;
* 1,000-refresh sequences with random scores for U16 / P16 / P8 / A8 / A16 / H2 / H2A on a geometry with
  partial row groups, anchors and backbone overlaps. Every refresh satisfies both invariants, mandatory
  support is unchanged, and the element-mask count equals the accounting;
* the legacy rule on the same kind of sequence drops more than one tile below B0 within 200 refreshes, the
  regression it guards against. In a 1,000-refresh development run, U16 0.3 lost 55,098 of its 57,658 slots;
* carry recovery;
* incompatible replacement costs (reduced prune, explicit infeasible record);
* the GPU remap of the learning driver.

**Re-tested diagnostic** (256 updates, `refresh/`; the parent's records are unchanged in
`width_preserving_sparsity/refresh/`):

| source | policy | T | rule | final enabled / B0 | final shortfall | max shortfall | max legal tile | held-out @256 | tiles differing |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| parent | H2A 0.10 | 16 | legacy | 79,997,648 / 80,000,000 | 2,352 | 2,352 | 256 | 0.23391 | 4,907 |
| this study | H2A 0.10 | 16 | legacy (rerun) | 79,998,672 | 1,328 | 1,328 | 256 | 0.23391 | 4,911 |
| this study | H2A 0.10 | 16 | **exact** | 79,999,872 | 128 | 208 | 256 | 0.23391 | 4,916 |
| parent | H2A 0.10 | 128 | legacy | 79,999,712 | 288 | 288 | 256 | 0.23333 | 4,912 |
| this study | H2A 0.10 | 128 | **exact** | 79,999,984 | 16 | 96 | 256 | 0.23333 | 4,913 |
| this study | H2A 0.10 | frozen | – | 79,999,904 | 96 | 96 | 256 | 0.23364 | 0 |
| parent | A8 0.1875 | 16 | legacy | 149,999,096 / 150,000,000 | 904 | 904 | 128 | 0.19791 | 9,313 |
| this study | A8 0.1875 | 16 | **exact** | 149,999,992 | 8 | 56 | 128 | 0.19791 | 9,308 |

* No refresh was rolled back or infeasible, and every remap check passed.
* The legacy rerun does not reproduce the parent's byte trajectory, because FP32 atomics and stochastic
  rounding differ. Its deficit (1,328) differs in size from the parent's (2,352) but has the same cause.
* The parent's frozen-mask timing conclusions do not depend on this.

## 3. Randomized local supports inside the C256 K64 envelope (`experimental/local_supports.py`)

**What is held fixed.** Every valid neuron keeps its original (g, n), original record (g, j) and input set q,
and every row group uses the bias and pre-pruning weights of its own retained coordinates. Only the choice of
16 of the record's 64 local inputs per 16-neuron row group changes. No default K16 generator is involved.
* **Budget:** exactly 16 H = 200,000,000 valid slots.
* **Geometry:** 781,312 valid groups, including 992 partial ones; padded groups take nothing.
* **Records:** 49,152 records with 11–16 groups each.
* **Sets:** m_q = 187–192 groups per set (mean 190.75) and b_q = 12 records in every set.
* **Every C256 record has ≥ 4 groups,** so the RP-record premise holds. This is verified per support
  (`all_subsets_in_every_record_with_nkt_groups`).

**Policies** (module docstring):
* **FP-mag:** the parent's P16 at its floor, the highest-|w| fixed tile.
* **FP-balanced:** the fixed four-way partition, with tile (φ_r + i) mod 4 for the i-th group of record r and a
  seeded phase φ_r.
* **RP-record:** one permutation per record, cut into four disjoint subsets, allocated by the same balanced
  rule.
* **RP-row:** an independent uniform 16-subset per group.
* **BP-row:** a stream of permutations per set q, consumed by the groups of q in canonical (g, j, c/16) order.

**RNG.** Per-purpose `torch.Generator` (CPU) seeds come from sha256(schema, policy, k, graph seed, layout
seed, purpose, chunk). A permutation is the stable argsort of int64 keys drawn uniformly from [0, 2^62); a tie
goes to the lower position. Selected positions are sorted before packing, so global ids and weights stay
paired.

**Tests** (`tests/test_local_supports.py`, 19 tests):
* membership, distinct sorted in-envelope coordinates and the original q for every valid group, at C = 64 /
  128 / 256 with tails;
* the exact budget;
* balance rules: per-record spread ≤ 1, all subsets present, disjoint passes covering 64;
* reproducibility and key independence (layout seed, graph seed, policy);
* uniformity and independence statistics on a 200k-neuron geometry: co-membership 16·15/(64·63), consecutive
  records overlapping by 4 on average;
* the coverage models against a tiny exhaustive oracle: the Monte-Carlo mean over 160 layout seeds of the exact
  realized coverage equals the refined per-set model within 5 standard errors (RP-record, RP-row, BP-row);
* FP-balanced being deterministic and within the one-tile bound;
* the coverage pipeline against brute force.

### 3.1 Coverage (`coverage/`, `coverage_confirm/`; exact supports of the timed graph and checkpoint)

Pairs and triples are exact layer-wide. Quadruples use 10M fixed seeded samples, the same for every support,
with Clopper-Pearson 95 % intervals. Per-output values are medians over the parent's fixed 38 outputs. "Row
groups per covered quad" is the mean number of *distinct row-group supports* containing a covered sampled
quadruple.

| support | pairs: predicted ideal / refined / measured % | triples: predicted ideal / refined / measured % | quads: predicted / measured % [CI] | per-output pairs / triples / quads % | bias-only % | distinct supports per record | row groups per covered quad |
|---|---|---|---|---|---:|---:|---:|
| dense potential | – / – / 100 | – / – / 61.64 | – / 5.558 [5.544, 5.572] | 16.879 / 1.1163 / 0.06694 | 0 | – | – |
| FP-mag | – / 97.72 *(invalid)* / 44.98 | – / 5.01 / 4.05 | 0.065 / 0.063 | 4.076 / 0.0603 / 0.00075 | 0 | 4 | 48.35 |
| FP-balanced | 97.72 / 97.72 / 44.98 | 5.01 / 5.01 / 4.05 | 0.065 / 0.063 | 4.123 / 0.0609 / 0.00075 | 0 | 4 | 48.33 |
| RP-record #0 | 99.99998 / 99.99998 / 100.0000 | 37.13 / 37.13 / 37.13 | 0.735 / 0.735 [0.730, 0.741] | 4.304 / 0.0603 / 0.00076 | 0 | 4 | 4.25 |
| RP-row #0 | 99.99999 / 99.99999 / 100.0000 | 58.75 / 58.75 / 58.75 | 2.379 / 2.381 [2.372, 2.391] | 10.866 / 0.2162 / 0.00297 | 0 | 16 | 1.31 |
| BP-row #0 | 99.99999 / 99.99999 / 100.0000 | 58.91 / 58.91 / 58.90 | 2.387 / 2.391 [2.382, 2.401] | 11.487 / 0.2203 / 0.00299 | 0 | 16 | 1.31 |
| U16 0.25 | – / – / 100 | – / – / 60.90 | – / 5.139 | 12.343 / 0.5440 / 0.02269 | 31.7 | 8 | 6.69 |
| A8 0.25 | – / – / 100 | – / – / 61.64 | – / 5.558 | 16.879 / 1.1163 / 0.06694 | 0 | 12 | 13.16 |

Bounds:
* **Per-output union bound** for any K16 row-group support (≤ 763 records per output): pairs 17.48 %, triples
  0.2395 %, quads 0.00305 %.
* **One-tile union bound** (one common partition per set only): triples 5.142 %, quads 0.0655 %.

Graph seeds 1 and 2 give the same values for BP-row (58.91–58.92 % triples, per-output 0.220–0.222 %), FP and
the controls. Layout seeds 0/1/2 differ by at most 0.006 pp on triples. All seeds are in `tables.md`.

**Reading.**
* **Coverage tracks the number of independent support records.** At a fixed budget, the distinct supports per
  record go 4 (FP, RP-record) → 16 (RP-row, BP-row). The supporting row groups per covered tuple go
  48 → 4.3 → 1.3.
* **Multiplicity falls as coverage rises.** FP repeats one of four supports per record across ~48 row groups,
  so it has high multiplicity and low coverage. BP-row spreads 16 distinct supports, giving near-potential
  layer coverage with a covered tuple held by one or two groups (16–32 neurons).
* **Per output there is no free lunch.** RP-row and BP-row reach 0.22 % of 0.239 % possible. The 0.15 pp
  difference between BP-row and RP-row (layer-wide) is BP-row's disjointness within a permutation pass, as
  predicted (58.91 vs 58.75 %).
* **Numerical support:** every retained weight of the α = 0 checkpoint is nonzero, so the numerical and enabled
  supports coincide at the start. After 32 / 256 fine-tuning updates and after normalization, no support lost
  an edge, and the numerical bias-only count stays 0 for every all-connected support (`support_after` in the
  learning records). Normalization never changes the nonzero pattern (asserted).

## 4. Execution: indexed K16 adapter and attribution of index costs

**Adapter** (`kgx_onetile.lower_local` / `lower_env`). It is the generalization of `LoweredOneTile` to
arbitrary sorted local subsets, running on the *existing* kgen K16 (and K8) C16 kernels. No new CUDA kernel was
written.
* Each K64 row group becomes one 16-neuron kgen record whose 16 global indices are feat[q][env[u]].
* Weights are gathered from those coordinates of the original neurons; biases, valid-neuron masks and buckets
  (in 16-neuron blocks, clipped to valid tails) are carried over.
* A uint8 canonical envelope map, never read by a kernel, maps values, main_grad and moments back to the 64
  envelope coordinates.

**Validated** (`tests/test_local_supports.py`, `tests/test_row_normalize_oracle.py`):
* For FP-mag, FP-balanced, RP-record, RP-row and BP-row at k = 16 and 8 and C = 256 / 64, compared with FP64
  and the dense K64 model masked with the same inputs: forward, dX, dW, db, full-z ReLU decisions and zero dW
  outside the support. Relative errors are below 1e-6 for y and db and below 2e-5 for dX and dW.
* Canonical parameter, mask, main_grad and moment hashes are identical.
* Three optimizer steps with stochastic rounding off agree to accumulation order.
* A complete snapshot restore is exact.
* Rejections:
  * `B % 8` token counts, a documented capability (B = 40 tails run, B = 33 is rejected);
  * deterministic-reduction requests;
  * unsorted or duplicate records.
* Non-finite inputs propagate exactly as in the dense masked model.

**Identity at the timed start.** Every timing record holds the canonical identity of its restored start.
* FP-mag gives identical parameter and mask hashes on indexed K16, the new index-free 16 × 16 kernel (`wide`)
  and the parent's tile kernel (`tile`); FP-balanced likewise.
* Repeated BP-row processes reproduce their hashes.
* main_grad and moment hashes are the common zero state.
* The graph fingerprint of the *executed* K16 record table differs from the K64 envelope's, as it should. The
  envelope, retained-support, index-layout and parameter fingerprints are kept separate.

**Records and work.**
* Nominal envelope: K64 / C256, 49,152 logical records.
* Executed: K16 / C16, 781,312 physical records (one per row group), 16 neurons sharing each.
* Index records: 23.8 MiB of int16 global ids. Canonical map: 11.9 MiB, not used by kernels. Work lists:
  6.9 MiB (index-free wide: 8.6 MiB of work lists + 3.0 MiB of tile metadata, no indices).
* Executed MMA products per step: 1.64 / 4.92 / 4.92 × 10¹² (fwd / dX / dW), identical for indexed and
  index-free.
* Peak allocated memory: 2.19 GiB indexed vs 2.17 GiB wide.
* Setup, outside timing: selection takes 0.29 s (RP-record), 1.23 s (RP-row) or 0.49 s (BP-row); packing takes
  0.03 s.

### 4.1 Complete training step (ms)

Screen: graph seed 0, layout seeds 0/1/2, 8 restored samples per process. Dense anchors 1,542–1,543 ms.

| support | format | step (per layout seed) | fwd kernel | dX kernel | dW kernel | optimizer |
|---|---|---|---:|---:|---:|---:|
| U16 0.25 | wide (index-free) | 811.4 | 114.5 | 424.7 | 264.0 | 5.1 |
| U16 0.25 | tile (parent kernel) | 856.4 | 129.3 | 450.4 | 267.8 | 5.1 |
| BP-row | indexed K16 | 862.3 / 861.5 / 861.1 | 67.7 | 602.6 | 184.9 | 5.1 |
| RP-row | indexed K16 | 868.9 / 863.0 / 861.5 | 67.9 | 604.0 | 186.0 | 5.1 |
| RP-record | indexed K16 | 870.1 / 862.8 / 861.3 | 67.8 | 604.3 | 185.9 | 5.0 |
| FP-balanced | indexed K16 | 887.9 | 65.9 | 632.3 | 183.0 | 5.0 |
| FP-mag | indexed K16 | 889.3 | 66.1 | 631.7 | 184.5 | 5.1 |
| FP-mag / FP-balanced | wide | 896.1 / 896.1 | 126 | 444 | 320 | 5.0 |
| A8 0.1875 | wide | 915.6 | 116.1 | 422.5 | 370.9 | 3.9 |
| A8 0.25 | wide | 971.4 | 121.1 | 447.5 | 396.8 | 5.1 |
| FP-mag / FP-balanced | tile | 1,008.0 / 1,014.2 | 175 | 498 | 330 | 5.0 |

* **Recomputation:** z is recomputed inside the dX / dW kernels; it is not timed separately.
* **Other phases** (layout, dX init, finalize, post) take ≤ 1 ms in every format.
* **Host enqueue** is 1.2–1.5 ms; the first step after warm-up is 0.88 s wall time.

Per-phase medians of every process are in `tables.md`.

### 4.2 Confirmation

Graph seeds 0/1/2 × layout seeds 0/1, 24 samples per process. Layout replicates are averaged within a graph
seed, and the intervals are t95 over the 3 graph means.

| support | format | per graph seed (layout values) | mean ± t95 | layout spread | × U16 0.25 wide (paired) | × dense (paired) |
|---|---|---|---:|---:|---:|---:|
| U16 0.25 | wide | 816.3 / 814.4 / 817.0 | 815.9 ± 3.3 | – | 1 | 0.526 ± 0.002 |
| **BP-row** | indexed K16 | 865.0 (864.2/865.8) / 866.2 (868.3/864.1) / 864.5 (864.6/864.4) | **865.2 ± 2.1** | 2.0 ms | **1.060 ± 0.007** | 0.558 ± 0.002 |
| FP-mag | indexed K16 | 892.5 / 895.8 / 890.3 | 892.9 ± 6.9 | – | 1.094 ± 0.013 | 0.575 ± 0.005 |
| FP-balanced | indexed K16 | 895.5 / 895.2 / 901.0 | 897.2 ± 8.2 | – | 1.100 ± 0.007 | 0.578 ± 0.004 |
| FP-mag | wide | 901.5 / 902.6 / 900.1 | 901.4 ± 3.1 | – | 1.105 ± 0.008 | 0.581 ± 0.003 |
| A8 0.25 | wide | 978.6 / 976.0 / 977.2 | 977.2 ± 3.3 | – | 1.198 ± 0.004 | 0.630 ± 0.002 |

Dense anchors: 1,550.9 / 1,551.1 / 1,552.6 ms. The historical parent values (824 / 911 / 986 ms for U16 / P16 /
A8) are context only; this session's denominators are the ones above.

### 4.3 Index-cost attribution (the requested controls)

1. **Same function, two backends.** FP-mag (and FP-balanced) runs on indexed K16 vs the index-free 16 × 16
   `wide` kernel at ×0.991 (FP-mag, confirmation: 892.9 / 901.4) and ×0.991 (FP-balanced, screen: 887.9 / 896.1) — a
   different backend *and* indexing.
   * The indexed path has a much cheaper forward (66 vs 126 ms) and dW (184 vs 320 ms) but a slower dX
     (632 vs 444 ms: it stages FP32 dX slabs per set, as noted in the parent study).
   * The two balance out at C256. This does **not** show that indexing itself is free; it shows that the
     whole kgen K16 implementation matches the index-free tile kernels here.
2. **Same indexed implementation, different supports.** FP-balanced → BP-row on indexed K16 has matched
   dimensions, work counts and checkpoint: ×0.964 (865.2 / 897.2), and in the screen RP-* / BP-row 861–870 vs
   FP 888–889.
   * The whole difference is in dX (603 vs 632 ms). With a fixed partition, all row groups of a set scatter
     into the same 16 dX rows of the slab; random subsets spread the scatter over all 64.
   * This is a change of access pattern, *not* a matched-function accuracy comparison.
3. **Optional bounded indexed variant:** not built. Control 1 already shows no index overhead above the
   ±1 % level at this size, so a dedicated kernel would not change the decision.
   * No optimization pass was used. Profiling shows the indexed step is dX-bound, in the existing kgen dX
     slab scheme.
   * A faster dX scatter would be a kernel campaign, which the assignment does not make a prerequisite.

## 5. Conditional K8 extension (`k8/`, graph seed 0 screen)

The predeclared conditions held, so the extension ran:
* BP-row K16 costs ×1.062 of U16 (limit ×1.10);
* BP-row's triple coverage is 14.5× FP-mag's (required: ≥ 5×).

Fan-in 8, ρ 0.125, 100M slots:

| support | format | step ms | layer pairs / triples / quads % | per-output triples % | bias-only % |
|---|---|---:|---|---:|---:|
| BP-row/8 (layout 0/1/2) | indexed K8 | 625.4 / 630.4 / 624.8 | 100 / 19.54 / 0.119 | 0.024 | 0 |
| FP-mag/8 (= P8 0.125) | indexed K8 | 639.2 | 27.08 / 0.88 / 0.005 | 0.011 | 0 |
| FP-mag/8 | wide 16 × 8 | 809.9 | (same function) | | |
| FP-balanced/8 | indexed / wide | 639.2 / 823.2 | 27.08 / 0.88 / 0.005 | 0.012 | 0 |
| U8 0.125 | wide 16 × 8 | 687.8 | 100 / 37.62 / 1.362 | 0.107 | 34.4 |

Refined predictions for BP-row/8: 19.54 % triples, 0.119 % quads (measured: the same). Dense anchors 1,542–1,555
ms.
* **K8 indexing beats the index-free 16 × 8 kernels** for the identical FP function: ×0.79, matching the
  parent's ×0.788 K8 lowered control.
* **BP-row/8 is the fastest all-connected support measured:** 0.91× U8 0.125.
* **It covers about half U8's layer-wide triples.** Fan-in 8 subsets hold C(8,3) triples instead of C(16,3).
* This extension has a single graph seed and no task-quality runs; it is labelled conditional and is not
  confirmed.

## 6. Signal-scale controls (`norm_first/`, `norm/`, `cap16/`; `rsn_learn.py`)

**Treatments** (`experimental/row_normalize.py`).
* Each treatment is applied once, to the retained BF16 values of the frozen mask, before fine-tuning.
* Support, bias count and optimizer budget are unchanged, and optimizer state is fresh.
* `fanin`: s = √(64/k). `energy`: s = min(s_max, √(E_dense/max(E_kept, 10⁻⁶ E_dense))), with E taken from the
  same pre-pruning canonical row.
* A hybrid row is one dense masked row, so backbone and tile coordinates are scaled exactly once. Disconnected
  rows stay zero; no epsilon edges are added.
* Rounding: the stored values are BF16(s·w), with a maximum relative rounding error of 0.39 % (half a BF16 ulp).
  No runtime scale remains, so there is no per-step cost.
* Data: none is used (the factors depend only on weights). Held-out targets are never touched.

**Tests** (`tests/test_row_normalize_oracle.py`):
* hand-computed scales on tiny rows, including the cap, the relative ε and all-zero retained rows;
* support unchanged and each weight scaled once;
* W-and-bias scaling keeps ReLU decisions;
* an H2 hybrid row with backbone / tile overlap restored to its dense energy.

**Protocol.** The parent's (`wps_learn.py`): the actual checkpoint with its proj / head, teacher, held-out
batches, M_train = 512, dense masked execution and no L1.
* 32 updates (data seed 0) for continuity with the parent.
* 256 updates with data / training seeds 0/1/2: train batches + 1,000,000·seed, stochastic-rounding counter
  offset.

**Diagnostics.** Measured on a fixed sample of 65,536 valid neurons. *Probe* means fixed layer inputs: the
checkpoint's proj on the 2,048 held-out tokens, computed before any update. *e2e* means each model's current
proj (in the records). No row is ever excluded.

### 6.1 All three treatments (32 updates, seed 0)

| support | treatment | held-out: ckpt / pruned / normalized / final | scale median | clipped % | Var(z−b) relative to own dense row, median: pruned → start → after | p10 after | Var>0.01 (of 65,536): pruned → after | participation ratio | layer output var ×10⁴: pruned → after |
|---|---|---|---:|---:|---|---:|---|---|---|
| BP-row | none | 0.25557 / 0.30016 / – / 0.26111 | – | – | 0.244 → 0.244 → 0.243 | 0.149 | 65,522 → 65,521 | 162.7 | 0.58 → 0.60 |
| BP-row | fanin | … / 0.25997 / 0.25885 | 2.000 | 0 | 0.244 → 0.974 → 0.974 | 0.594 | 65,522 → 65,536 | 162.6 | 0.58 → 2.34 |
| BP-row | energy | … / 0.25997 / 0.25897 | 2.021 | 0 | 0.244 → 0.999 → 0.999 | 0.809 | 65,522 → 65,536 | 162.6 | 0.58 → 2.39 |
| H2 0.05 | none | 0.25557 / 0.34694 / – / 0.26375 | – | – | 0.051 → 0.051 → 0.051 | 0.011 | 34,595 → 34,579 | 158.0 | 0.15 → 0.15 |
| H2 0.05 | fanin | … / 0.26871 / 0.26058 | 5.657 | 0 | 0.051 → 1.428 → 1.427 | 0.344 | 34,595 → 64,784 | 157.5 | 0.15 → 3.77 |
| H2 0.05 | energy | … / 0.26251 / 0.26154 | 4.422 | 14.86 | 0.051 → 0.973 → 0.973 | 0.653 | 34,595 → 65,180 | 157.6 | 0.15 → 2.23 |
| H2 0.05 | energy, W and bias | … / 0.26244 / 0.26147 | 4.422 | 14.86 | (same) | 0.653 | (same) | 158.1 | 0.15 → 2.23 |
| dense | none | 0.25557 / 0.25557 / – / 0.25023 | – | – | 1 → 1 → 1 | 0.995 | 65,536 | 161.3 | 2.59 → 2.80 |

### 6.2 Bounded set, 256 updates, seeds 0/1/2

Held-out loss, mean ± t95; the difference is paired per seed.

| support | none | energy (s_max 8) | energy − none | energy, s_max 16 | (16) − none |
|---|---:|---:|---:|---:|---:|
| dense | **0.11364 ± 0.00243** | – | – | – | – |
| U16 0.25 | 0.17456 ± 0.00069 | **0.16868 ± 0.00254** | −0.00588 ± 0.00187 | | |
| FP-mag | 0.18065 ± 0.00195 | 0.17747 ± 0.00221 | −0.00318 ± 0.00259 | | |
| A8 0.25 | 0.17776 ± 0.00177 | 0.17853 ± 0.00203 | +0.00078 ± 0.00219 | | |
| BP-row | 0.18231 ± 0.00148 | 0.17914 ± 0.00209 | −0.00318 ± 0.00205 | | |
| H2A 0.10 | 0.23346 ± 0.00055 | 0.23367 ± 0.00203 | +0.00021 ± 0.00151 | 0.23227 ± 0.00300 | −0.00119 ± 0.00259 |
| H2 0.05 | 0.25580 ± 0.00015 | 0.24547 ± 0.00170 | −0.01033 ± 0.00176 | 0.24373 ± 0.00112 | −0.01207 ± 0.00115 |

* **Clipping.** At s_max 8 the energy rule clips 14.86 % (H2) and 14.54 % (H2A) of connected rows, against
  0–0.024 % for the K16 / A8 tile supports. The p10 of the restored energy ratio is 0.70 (the clipped rows).
* **The cap-16 sensitivity is predeclared:** clipping above 1 % triggers it. At cap 16 clipping falls to
  3.1 / 3.0 % and the restored energy is ≥ 0.996 at p10.

**Reading.**
* **Amplitude:** the "weak width" of the thin backbone was almost entirely amplitude. One rescale brings H2's
  median relative variance from 0.05 to 0.97–1.0 and every sampled row above the old absolute threshold.
  * The old absolute threshold is kept only for comparison, and crossing it is not counted as an improvement.
* **Diversity:** the amplitude-normalized participation ratio (2,048 neurons × 2,048 probe tokens, rank ≤ 2,048)
  is 157–163 for every support and treatment. Rescaling cannot change it (bias ≈ 0 at this checkpoint), and
  supports barely differ on it.
* **Loss:** a real gain for H2 (−0.012 at cap 16) and nothing measurable for H2A or A8, whose anchors already
  carry full-energy rows. The gain is mid-sized for the K16 supports (−0.003 to −0.006).
* **Absolute ranking:**
  * The backbone models stay at 0.23–0.24, far from the K16-budget supports (0.17–0.18) and dense (0.114).
    Fair scaling does not turn two inputs into a useful layer on this task.
  * U16 (magnitude-selected free tiles) has the best sparse loss with or without normalization, despite its
    31.7 % bias-only rows. On this task magnitude selection matters more than structural width.
* **What rowwise normalization adds beyond the readout calibration.** The interaction driver's per-output affine
  readout rescales the *output*, which compensates a uniform scale. Rowwise normalization changes the *relative*
  scale of rows within an output and the size of each row relative to its fixed-learning-rate Adam steps.
  * In the interaction task (fresh init, zero bias, calibrated readout) energy normalization *slowed* learning
    slightly. For A8 T16 the explained interaction at 1,000 updates was 0.20–0.36 vs 0.31–0.45, and at 4,000
    0.75–0.79 vs 0.78–0.82. This fits larger weights receiving relatively smaller Adam updates.

## 7. Optimization vs representation in the interaction task (`interaction/`; `rsn_interaction.py`, `rsn_oracle.py`)

**Workload.** The parent's reduced diagnostic, unchanged:
* D = G = 1024, C256, S64 / R256, H = 262,144 (one original record per output, 768 order-≥2 records). All
  4,096 sets are occupied.
* Raw x ~ N(0, 1), with no projection and a per-output affine readout.
* Targets: single-index ReLU terms of order 1 + (g mod 4), hash seed 777.
* Data: train seeds 50,000 + i, held-out 49,000–49,003 (2,048 tokens; values are *estimates*).
* Init: fresh by-neuron init with seed 17 (18 / 19 for the seed controls) and zero bias; supports are selected
  from the init weights.

Explained interaction = 1 − nMSE / floor_k, with the exact additive floor of each actual configuration (T1:
0.303 / 0.340 / 0.349; T16: 0.193 / 0.220 / 0.227; T64: computed and recorded the same way). Order 1 is the
separate linear control; the oracle does not construct it (n/a).

### 7.1 Teacher-mapped oracle

**Construction.**
* Each term maps to one of the 16 full-support anchor neurons (A8 0.25; the first row group for dense):
  weights 1 on its k inputs, bias −1, everything else zero.
* The readout is analytic: scale n_g / (√(n_rec T) σ_k), offset −√(n_rec T) μ_k / σ_k.
* It is verified in FP64 for T = 1 and 16, including duplicated terms (`test_oracle_algebra_fp64`, error below
  1e-12), and on the GPU (`test_gpu_oracle_reproduces_the_targets`).
* Duplicates: 12,288 terms contain 12,159 distinct hyperplanes at T16, and 49,152 contain 47,156 at T64.
* The oracle's zeros violate the broad-support operating point; it is a representability diagnostic only.

**Results** (orders 2 / 3 / 4):

| support, T | oracle, FP32-input targets nMSE | vs BF16-rounded inputs | explained at step 0 | after 1,000 updates of training from the oracle |
|---|---|---|---|---|
| A8 0.25, T1 | 8.1e-6 / 6.4e-6 / 5.7e-6 | ~1e-14 | 1.000 | 1.000 |
| A8 0.25, T16 | 8.2e-6 / 6.3e-6 / 5.5e-6 | 2.7e-14 / 1.6e-14 / 2.5e-14 | 1.000 | 0.999 / 0.999 / 1.000 |
| dense, T16 | 8.2e-6 / 6.3e-6 / 5.5e-6 | ~1e-14 | 1.000 | 0.999 / 0.999 / 1.000 |
| dense, T64 (match) | 8.4e-6 / 6.2e-6 / 5.4e-6 | | 1.000 | 0.997 / 0.999 / 0.999 |

* The residual 6–8e-6 is the BF16 input quantization floor.
* **Perturbed oracle.** Declared ε = 0.1 × init std (1/8) added to every enabled weight, analytic readout kept.
  * This is not small in output space: the start nMSE is 1–440× the floor, because every zero neuron becomes
    active and the analytic scale multiplies it by n_g.
  * Training returns it to 1.000 / 1.000 / 1.000 (A8 T16, T1) and 0.996–1.000 (dense) within 1,000 updates.
    The oracle solution is a stable, reachable basin.
* **Matching oracle, U16 0.25 / BP-row.** This is an eligibility diagnostic, not an impossibility proof:
  * Only 241 of 768 terms (U16) and 221 of 768 (BP-row) at T1 have any neuron whose support contains all k
    inputs; at T16, 3,856 and 3,683 of 12,288.
  * BP-row has 14.9 / 3.2 / 0.7 eligible neurons per term on average for orders 2 / 3 / 4; A8 has 21 / 17 / 16.
  * The construction is therefore not exact for them (explained < 0). Their gap is **absent support**: most
    terms have no eligible neuron.
  * A8 at T64 maps 17,950 of 49,152 terms. The 16 anchors plus multi-tile groups cannot host 64 terms per record.

### 7.2 Ordinary training: duration, seeds and learning rate (T16 unless stated)

The 1,000-update value of each 4,000-update run is the parent protocol, which uses constant learning rates. The
1,000-step entries of the 1,000- and 4,000-update runs are identical (e.g. A8 0.306 / 0.421 / 0.452). The
complete state at step 1,000 is stored for continuation. The reported 4,000-update runs are uninterrupted, so `--resume` from those states was
not used for any reported number. The six state files (976 MB) stay outside git in
`../rsn_runs/interaction/state/`, with sha256 values in the records and in `interaction/state_files.sha256`.

| support | start | init seed | lr | explained @1,000 | @4,000 |
|---|---|---:|---:|---|---|
| dense | parent init | 17 / 18 / 19 | 3e-3 | 0.782 / 0.852 / 0.876 (all three seeds within 0.001) | **0.998** (all seeds) |
| dense | parent init | 17 | 1e-3 / 1e-2 | 0.541–0.757 / 0.987–0.991 | 0.964–0.970 / 1.000 |
| A8 0.25 | parent init | 17 | 3e-3 | 0.306 / 0.421 / 0.452 | **0.822 / 0.789 / 0.778** |
| A8 0.25 | parent init | 18 | 3e-3 | 0.328 / 0.426 / 0.448 | 0.838 / 0.788 / 0.777 |
| A8 0.25 | parent init | 19 | 3e-3 | 0.319 / 0.417 / 0.456 | 0.824 / 0.783 / 0.781 |
| A8 0.25 | parent init | 17 | 1e-3 | −0.191 / 0.085 / 0.172 | 0.642 / 0.660 / 0.660 |
| A8 0.25 | parent init | 17 | 1e-2 | 0.656 / 0.678 / 0.682 | **0.968 / 0.945 / 0.941** |
| A8 0.25 | energy-normalized | 17 | 3e-3 | 0.201 / 0.330 / 0.364 | 0.794 / 0.764 / 0.754 |
| BP-row | parent init | 17 | 3e-3 | 0.559 / 0.509 / 0.480 | 0.649 / 0.574 / 0.532 |
| U16 0.25 | parent init | 17 | 3e-3 | 0.014 / 0.188 / 0.245 | 0.430 / 0.413 / 0.422 |

At T1 (1,000 updates):
* dense: 0.999;
* A8: 0.840 / 0.908 / 0.936 (energy: 0.755 / 0.843 / 0.880);
* BP-row: 0.519 / 0.505 / 0.464;
* U16: 0.537 / 0.468 / 0.461.

At T64 (1,000 updates):
* dense: 0.363 / 0.625 / 0.699;
* BP-row: 0.257 / 0.312 / 0.306;
* A8: −0.400 / 0.033 / 0.156.

All 44 attempts completed; none produced non-finite values. Curves are in the records (`history`).

**Interpretation, keeping the four causes apart.**
* **Representable, but ordinary training is slow to find it (A8, T ≤ 16).**
  * The permitted mask represents the teacher exactly, and the oracle basin is stable.
  * Ordinary training closes most of the gap with 4× the updates (0.78–0.84, consistent over 3 init seeds) and
    nearly all of it with a larger learning rate (0.94–0.97).
  * The parent's capacity-stress gap (0.08–0.45 at 1,000 updates) was therefore mainly *optimization budget*,
    not inability to represent.
  * A8 still trails dense at equal budget (dense reaches 0.998 at lr 3e-3). Its 16 anchors hold each term once,
    while dense has 256 candidates per record, so this is *reduced support multiplicity*: fewer redundant
    neurons to find each term.
* **Absent support (U16, BP-row).** Most terms have no eligible neuron, so the plateaus (0.42–0.65 at 4,000
  updates) are bounded by the support, not by optimization. BP-row's randomization gives more eligible neurons
  per order-2 term than U16's but fewer full-order matches. Its higher T16 values do not reflect per-output
  coverage, which is lower (0.22 vs 0.54 %).
* **Unresolved (T64).** Dense is representable (match oracle 0.999) and only partly learned in 1,000 updates.
  This is an unresolved optimization diagnostic, so it is not used to rank capacity. Above 16 terms, nominal T
  is not a lower bound on the neurons required: background neurons may see terms.
* **Generalization / task quality** is the separate fine-tuning task of §6, where A8's interaction advantage
  does not translate into the best loss (U16 is better there).

## 8. Correctness gates, sanitizers and regressions

* **New tests:**
  * `tests/test_support_refresh.py` (14);
  * `tests/test_local_supports.py` (19: structure, seeds, statistics, coverage oracles, indexed lowering vs
    FP64, optimizer / identity / restore);
  * `tests/test_row_normalize_oracle.py` (10: normalization, oracle algebra and GPU reproduction, indexed
    rejections / tails / non-finite).
* **Targeted regressions before the large runs:** the new tests plus `test_kgx_wide`, `test_wps_coverage`,
  `test_kgen` and `test_kgx` — 223 passed.
* **Compute Sanitizer 2023.1.0** (`sanitizer/summary.tsv`): 12/12 PASS (memcheck, racecheck, initcheck,
  synccheck).
  * A new workload `rsn` (`sanitize_rsn_workload.py`) covers all five local policies at k = 16 / 8, C = 64 /
    128 / 256 with tails, split / BF16 dZ, B = 8 / 40 / 136, non-finite upstream gradients, main_grad and the
    L1AdamW update, and FP supports on the index-free kernels.
  * The regressions `kgen` and `kgw` were also run.
* **Full suite on the final code:** see §8.1 (`tests/full_suite.txt`).
* **Unchanged:** production defaults, auto-dispatch and deterministic-mode behaviour. The indexed path rejects
  deterministic requests as before.
* **Stochastic rounding:** off for the update-equivalence tests. With it on, rounding indices are
  representation dependent and FP32 atomics reorder sums, so trajectories are not promised bitwise. Checkpoint
  regeneration is not promised bitwise either.

### 8.1 Full suite

`pytest sparse_pooling/tests` at the final code (`52820b7` plus report-only changes), run in a capped scope:
**3,171 passed, 32 skipped, 0 failed** in 440 s. That is the parent's 3,128 plus the 43 new tests; the skips are
the parent's profile- and device-conditional skips. The output is in
`random_supports_normalization_controls/tests/full_suite.txt`.

## 9. Pareto set

Criteria: complete step (confirmation where available), structural width, per-output and layer triple coverage,
and held-out loss after 256 updates with the better of none / energy.

| support | step ms | bias-only | layer / per-output triples | held-out (256, best treatment) | T16 interaction @4k | on frontier because |
|---|---:|---:|---|---:|---|---|
| U16 0.25 wide | 815.9 | 31.7 % | 60.9 / 0.54 % | 0.1687 (energy) | 0.42 | fastest K16 budget, best sparse loss |
| BP-row K16 indexed | 865.2 | 0 | 58.9 / 0.22 % | 0.1791 (energy) | 0.53–0.65 | cheapest all-connected K16 with near-potential layer coverage |
| A8 0.25 wide | 977.2 | 0 | 61.6 / 1.12 % | 0.1778 (none) | 0.78–0.84 (lr 3e-3), 0.94–0.97 (lr 1e-2) | full per-output coverage, representable interactions |
| BP-row K8 indexed (conditional, g0) | 627 | 0 | 19.5 / 0.024 % | not measured | not measured | fastest all-connected support measured |
| dense | 1,551 | 0 | 61.6 / 1.12 % | 0.1136 | 0.998 | quality reference |

Dominated:
* FP-mag and FP-balanced (any format) are slower than BP-row on the same implementation, with the same width and
  1/14.5 of its layer coverage, and equal per-output coverage. Their loss is 0.1775–0.1807, within noise of
  BP-row.
* RP-row and RP-record cost the same as BP-row, with marginally less (RP-row) or much less (RP-record)
  coverage.
* H2 0.05 and H2A 0.10 have loss ≥ 0.232 even after fair scaling. Parent timings: ×2.5–3.9 of U16.

## 10. Limitations and unresolved items

* Both learning tasks are synthetic; differences of ±0.002 in held-out loss are within the three-seed intervals.
* Graph-seed variation of the *learning* results was not measured: all learning runs use graph seed 0 and
  layout seed 0.
* Timing is from one GPU (RTX 3090, P2, about 1,950 MHz SM, 330–342 W), at C256 only. Indexed kgen dX staging
  grows with C, as the parent's C2048 control showed, so the indexed / index-free parity is not claimed for
  other C.
* The K8 extension is a screen at one graph seed, without learning runs.
* The perturbation of the oracle is small in weight space but not in output space (§7.1).
* T64 dense learning is unresolved within 1,000 updates.
* No weight-ranked randomized variant was run (optional).

## 11. Reproduction

Set `PY=env_isaaclab python` and `GRANULE_RESULTS_ROOT` to a directory outside the checkout. The commands run
from `sparse_pooling/benchmarks` of a clean checkout of this branch.

```bash
# one randomized indexed case (timing, canonical start + identity)
$PY bench_supergroups.py --hidden 12500000 --C 256 --S 64 --R 256 --graph-seed 0 --init-seed 17 --input-seed 23 \
  --rounds 4 --reps 2 --iters 3 --kcsr --canonical-start --identity --layer-factory k64_factory:wide \
  --factory-args '{"checkpoint": "<k64l1_ckpt>/C256_R256_prox_a0_step128.pt", "format": "indexed", "tile": 128, "policy": "BP-row", "k": 16, "layout_seed": 0}' \
  --opt-factory k64_opt:l1 --opt-args '{"l1": "none"}' --out bp_row_indexed
$PY rsn_coverage.py --support 'BP-row#0' --out cov_bp_row.json          # its exact coverage + predictions
# one normalization case
$PY rsn_learn.py --support H2@0.05 --norm energy --steps 256 --data-seed 0 --sr-seed 0 --out h2_energy.json
# the teacher oracle (+ perturbed) and the ordinary-training continuation
$PY rsn_interaction.py --support A8@0.25 --terms 16 --oracle anchor --out a8_oracle.json
$PY rsn_interaction.py --support A8@0.25 --terms 16 --oracle anchor --perturb 0.1 --out a8_oracle_p.json
$PY rsn_interaction.py --support A8@0.25 --terms 16 --steps 4000 --save-at 1000 --state-dir state --out a8_T16.json
# the corrected refresh (and the legacy rule for comparison)
$PY wps_refresh.py --policy H2A --rho 0.1 --period 16 --steps 256 --rule exact --out refresh_exact.json
$PY wps_refresh.py --policy H2A --rho 0.1 --period 16 --steps 256 --rule legacy --out refresh_legacy.json
# whole study: plans, then report
$PY rsn_jobs.py {refresh|coverage|screen|norm_first} --out <stage>
$PY rsn_jobs.py confirm --out confirm --selection confirm_selection.json   # likewise norm, interaction, cap16, k8, coverage_sel
$PY rsn_report.py --root $GRANULE_RESULTS_ROOT --parent ../results/width_preserving_sparsity --out tables.md --csv csv
bash run_sanitizer.sh --out sanitizer $PY rsn kgen kgw
```
