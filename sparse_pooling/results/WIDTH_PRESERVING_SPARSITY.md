# K64: reducing weight density while preserving input-dependent width

Branch `k64-width-preserving-sparsity`, which starts from `k64-large-c-indexfree-l1` @ `982760d`. It is an
experiment only: nothing is merged into `main`.
* Production defaults, kernels and auto-dispatch are unchanged.
* The new kernels are experimental variants of the `kgx` bundle (`tilew16`, `tile8`, `hyb16`), and the
  dispatcher never selects them.
* Raw evidence is in `results/width_preserving_sparsity/`. Its `tables.md` holds the generated tables of
  every measured row, and `csv/` holds `timing.csv` and `policies.csv`.

**Fixed model.** D = G = 1024, nominal K = 64, S = 64, R = 256, logical C = 256 (2048 and 64 for anchor and
winner checks). H = 12.5M valid neurons, so W0 = 800M valid weight slots; M = 8192 tokens.

**What is compared**, at exact valid-slot budgets, all from the same actual α = 0 checkpoint and pre-pruning
weights:
* free 16×16 and 16×8 tile pruning (U16, U8);
* one protected tile per row group (P16, P8);
* one full-support coverage anchor per logical record (A16, A8);
* a protected two-input backbone plus 16×16 tiles, without and with anchors (H2, H2A).

**What is measured**, all together:
* the complete forward / backward / optimizer step;
* the structural and numerical bias-only width;
* the input sensitivity of a fixed neuron sample;
* layer-wide and per-output tuple coverage;
* tuple multiplicity;
* two synthetic learning diagnostics.

**Correctness first.** Every policy and format was validated against an FP64 reference before timing. All
eight Compute Sanitizer runs pass. The full suite passes: 3,128 passed, 32 skipped, 0 failed.

## Answers in brief

*Times are complete C256 training steps over graph seeds 0 / 1 / 2: mean ratio ± t95, 24 samples per seed.
The unpruned dense step is 1,568.7 ms.*

### How much bias-only width can the b16 format avoid at the same budget, and what does protection cost?

**The granularity bound sets the floor.** Complete 16-input tiles force at least max(0, 1 − 4ρ) bias-only
rows. So below ρ = 0.25 no b16 mask can connect every row. P16 at ρ0.10 and P16 at ρ0.05 are rejected as
infeasible, and U16 has 65.7 % bias-only rows at ρ0.10 and 81.5 % at ρ0.05.

**At ρ0.25, P16 (one protected tile per row group) removes the 31.7 % bias-only rows of U16.**
* It costs **×1.106 ± 0.002** step time: 911 ms against 824 ms.
* It costs almost all of the multi-input coverage: layer triples fall from 60.90 % to **4.05 %**, and
  quadruples from 5.14 % to **0.063 %**. That is within the one-tile union bounds (5.142 % / 0.0655 %).
  Pairs fall from 100 % to 45.0 %.
* In the interaction diagnostic it captures **none** of the cross-block interaction: explained share ≤ 0,
  against 0.46–0.54 for U16 at the same budget.

**Above the floor, protection is nearly free.**
* At ρ0.375, P16 is ×1.044 the time of U16 (screen) and keeps 61.61 % / 5.48 % (U16: 61.64 % / 5.55 %),
  removing 15.3 % bias-only rows.
* At ρ0.5 it costs ×1.012 and keeps full potential coverage.

### Does b8 improve the speed / width / coverage tradeoff after its overheads?

**Partly. b8 halves the bias-only floor and buys width for time.**

| ρ | U8 bias-only | U16 bias-only | U8 / U16 time | U8 triples | U16 triples |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 10.1 % | 31.7 % | ×1.128 ± 0.010 | 60.36 % | 60.90 % |
| 0.125 | 34.4 % | 58.7 % | ×1.33 (screen) | 37.6 % | 47.8 % |

U8 keeps fewer quadruples than U16 at the same ρ.

The 16×8 kernels pay their overhead mostly in dW: 356 ms for U8 against 266 ms for U16 at ρ0.25, and 382 ms
for P8 against 325 ms for P16.

**At the b8 one-tile floor (P8 ρ0.125) the new format is not needed.** That support is exactly a K8, C16
kgen layer: the same retained indices, weights, bias and neuron identity, with an identical canonical hash. The
existing K8 kernels run it in **×0.788 ± 0.006** of the 16×8 time (648 vs 823 ms). The difference is in the
forward (48 vs 111 ms) and dW (198 vs 309 ms).

P8 at its floor also keeps just 0.88 % of triples (bound 1.03 %) and has a negative explained interaction.

### Can A8 at 18.75 % / 25 % keep every row connected and the potential coverage? What happens to multiplicity and quality?

**Yes, structurally and numerically.**
* All 12.5M rows have input support: 0 bias-only by both definitions.
* All 49,152 records hold a full-support anchor group (786,432 neurons), with no zero anchor weight.
* Layer-wide coverage equals the dense potential: 61.636 % triples and 5.558 % quadruples.
* Per output it equals the potential on all 38 fixed outputs: identical exact pairs and identical sampled
  triple / quadruple hits.
* The guarantee is checked structurally for all 1,024 outputs.

**Multiplicity collapses.** A covered tuple is held by one anchor group per record: 16 neurons per output,
against 256 in the dense potential. The median covered quadruple has 192 supporting neurons layer-wide,
against 3,072 for dense.

**Quality on the synthetic fine-tune** is the same as unconstrained pruning at the same budget:
* A8 0.1875: 0.26158; U16 at 0.1875: 0.26187.
* A8 0.25: 0.26075; U16 at 0.25: 0.26076.
* The dense model continued for the same updates reaches 0.25023.

**Interaction diagnostic.**
* With one term per record, the 16 anchors explain 0.82–0.96 of the cross-block interaction (dense 0.999).
* Under capacity stress (16 terms per record) they explain 0.08–0.45, still more than U16 / U8 at the same
  budget, against 0.78–0.88 for dense. **The anchors carry coverage, not broad capacity.**

**Cost.** A8 runs ×1.197 ± 0.005 the U16 time at ρ0.25, and ×1.30 at ρ0.1875 (screen).

### Can H2 / H2A make 5 % / 10 % density useful without collapsing most neurons? Does the scalar backbone dominate runtime?

**Every row stays connected.** H2 at ρ0.05 and 0.10, H2A at 0.10, and H2A at ρ0.05 at C2048 (the feasible
low-density anchor case) all have 0 bias-only rows. H2A also keeps the full potential coverage: 61.64 % /
5.56 %, per output as well.

**But this width is weak.**
* A two-input row has a median input-driven variance of 0.012, against 0.12 for the connected rows of U16.
* Only 57–63 % of the sampled neurons exceed a 0.01 sensitivity threshold.
* H2 at ρ0.05 keeps fewer triples (14.2 %) than U16 at the same budget (22.1 %).

**It is expensive.** Against U16 at the same budget, H2A at ρ0.10 runs ×2.541 ± 0.026 (1,130 ms against
445 ms), and H2 at ρ0.05 runs ×3.9 (screen). These supports are still ×1.39–1.52 faster than dense.

**The backbone does NOT dominate.** Compiling it out (ablation build) removes 23–27 % of the step: 261–272 ms.
The remaining ≈ 750–865 ms is the cost of running every neuron, meaning z, ReLU, dz, recomputation and
pooling for all 12.5M neurons and all tokens. U16 at ρ0.05 avoids that cost by running only 18.5 % of its
rows.

**Quality.** Fine-tuned held-out loss is H2 0.26375 against U16 0.26421 at ρ0.05, and H2A 0.26299 against
U16 0.26336 at ρ0.10. Under the interaction capacity stress, H2A at ρ0.10 explains less than A8 does.

### Which policies combine a low bias-only fraction with useful input sensitivity, and which only improve a structural counter?

**Both:** A8 from ρ0.1875, and A16 / P16 from ρ0.375.
* They have 0 bias-only rows.
* 96.9–99.99 % of sampled neurons are above the sensitivity threshold.
* They keep the potential coverage.

P8 at ρ0.1875 / 0.25 has the same full width, but reduced coverage: 46.3 % / 59.5 % triples.

**Structural counter only:**
* **P16 at ρ0.25 and P8 at ρ0.125** connect every row but destroy multi-block coverage and every cross-block
  interaction.
* **H2 / H2A at ρ0.05–0.10** connect every row with mostly weak two-input sensitivity. H2A adds full coverage
  through its anchors.

### Which apparent gains are an existing lower-fan-in execution of the same graph, and which need the new formats?

**Exactly-one-tile supports are lower-fan-in models.**
* **P16 at ρ0.25 = K16, C16; P8 at ρ0.125 = K8, C16.** The existing kgen kernels run the identical function:
  * at C256, ×0.994 ± 0.006 (P16) and ×0.788 ± 0.006 (P8) of the new kernels' time;
  * at C64, faster still (815 vs 944 ms; 618 vs 857 ms);
  * at C2048, slower (1,384 vs 887 ms; 940 vs 800 ms), because kgen's per-record dX slab grows with the
    C2048 record length.
* None of their apparent width gain needs a new format.

**Gains that need the new formats:**
* every support with more than one tile in some row group (P / U above the floor);
* anchored supports (A16, A8, H2A);
* backbone hybrids (H2, H2A);
* b8 multi-tile supports.

The new 16×16 kernels are also ×1.053–1.130 faster than the measured `tile` kernels of the K64 study on
identical masks (U16 0.25: 870 → 824 ms).

**Pareto sets** (§13) at the 1 % / 1-point / 0.5-point / 1e-4 tolerances:
* **Step vs held-out alone:** unconstrained U16 at every density, plus U8 0.125, wins.
* **Requiring useful width ≥ 95 % and potential triples:** A8 0.1875, A8 0.25, A16 / P16 0.375, P16 0.5.
* **At ρ ≤ 0.10 with every row connected:** H2 0.05 / 0.10. H2A 0.10 is the only one of these that keeps
  potential coverage.

**No policy wins on all axes.** Width preservation is a measured tradeoff: 1–30 % more step time at
ρ ≥ 0.1875, and ×2.5–3.9 at ρ ≤ 0.10, for connected rows and coverage. On this synthetic task it bought at most
0.17 % better held-out loss, at ρ ≤ 0.1875, and none at ρ ≥ 0.25.

## 1. Source, provenance and the audit fixes

**Frozen checkouts and results.**
* All measurements ran from frozen detached checkouts, each at a single commit:
  * stage 1 (screen, C2048, ablation, rejection, inference, coverage, fine-tune, interaction, regeneration) at
    `7a5396c`;
  * stage 2 (confirmation, winners, S128 reference, refresh) at `30def0c`;
  * stage 3 (interaction capacity, fine-tune repeat) at `d5ce881`;
  * the exact interaction floors at `a6a4c08`.
* Results went to `GRANULE_RESULTS_ROOT`, outside the checkouts.

**Provenance of the 239 timing records.**
* All 239 are `source_dirty=false` and `generated_dirty=false`.
* Every production timing run used **one kgx library, `c400495d8202e34be70f`**, across both commits; the
  kernel sources did not change after `9e2647b`.
* kgen is `023754cd60595afae7e9`, the same as the K64 study.
* Only the 3 backbone-ablation runs used a separate, labeled `ablation` build (`f7839fb3…`).

Every record carries:
* its commit;
* the status / changed paths and the source diff if dirty (none were dirty);
* the source-tree sha256;
* the build identity;
* `nvidia-smi` state before and after each round;
* all seeds: graph 0 / 1 / 2, init 17, input 23, random-score seed 1, coverage sample seeds, neuron-sample seed
  4242.

**Tests and sanitizers.**
* The sanitizers ran on kernel sources identical to the frozen ones.
* Full suite: 3,125 passed at `b88a5f3` before the campaign. At the final code `6245120`: **3,128 passed,
  32 skipped, 0 failed** (`tests/full_suite_final.txt`).

### 1.1 Representation-independent parameter identity (audit item 1)

`experimental/canonical.py` defines one canonical form for every K64 envelope representation:
* For every valid neuron in (output g, neuron n < n_g) order: the 64 weights of its record, in local input
  order (ascending global feature id), then its bias.
* Padding never enters.
* Absent or inactive entries are reconstructed as +0, and raw −0 bits are normalized to +0.

The hash uses two independent sha256 streams, `sha(header ‖ sha(rows) ‖ sha(biases))`, so it does not depend
on chunk size.

Every representation implements `canon_rows(src, g, n)`:
* dense `KgxLayer` (including its `lowered` / `kgen` routes);
* dense masked;
* `KgxTileLayer`;
* `KgxCsrLayer`;
* `KgxWideLayer` (16×8, 16×16 and backbone + tiles; backbone weights included);
* the K16 / K8 lowered control.

Since `canon_rows` takes its source tensors as input, the same function hashes three things:
* the parameters;
* the FP32 main_grad;
* **both Adam moments**, mapped from compact to canonical.

The enabled support (`mask_sha`), the graph (record-table fingerprint) and the raw storage (`storage_sha`) are
separate hashes.

`kcsr_graph.params_sha` now returns this hash for every K64 envelope layer, labeled
`logical_params_sha_schema = canonical`, and `logical_params()` no longer falls back to the compact tensor of
the tile / CSR formats.

Tests:
* dense masked (which holds −0 at inactive slots), tile16, CSR, wide16, wide8, hybrid, the lowered C512 route
  and the K16 / K8 controls of an identical model all give equal canonical and mask hashes;
* storage hashes differ, except tile16 on the old and new kernels, which share one layout;
* a one-ulp change of one enabled weight changes the hash; padding does not;
* moments after identical optimizer steps agree to rounding.

**In the campaign**, every timing job records `kcsr.canonical_start`: params / mask / main_grad / m / v at its
restored start. **For all 26 screen masks, every format of a mask had identical params and mask hashes**, and
so did every confirmation seed.

### 1.2 Honest numerical metadata (audit item 2)

**CSR kernels.** The CSR kernels compute dz in FP32 on CUDA cores; the inherited description had said "split
BF16". `KgxCsrLayer` now:
* records `grad_precision = "fp32_scalar"`, keeping the request as `grad_precision_requested`;
* describes FP32 dz in `numerics()`;
* **rejects** a `grad_precision="bf16"` request rather than silently exceeding it (test
  `test_csr_reports_its_executed_fp32_dz_and_rejects_bf16_requests`).

**Backbone layer.** The new backbone layer labels its actual scheme: `mixed: tiles split BF16 MMA dZ, backbone
fp32_scalar dz`. Its backbone products are FP32 FMAs of BF16 operands into the same FP32 accumulator as the
tile MMAs, and its backbone gradients use FP32 dz.

### 1.3 Checkpoint reproducibility (audit item 3)

**Reuse, not regeneration.** The study reuses the **actual α = 0 checkpoints** of the K64 study: C256 and C2048
at graph seeds 0 / 1 / 2, and C64 at seed 0. `checkpoints/manifest.json` records, for each of the 7:
* the path, size and file sha256, recomputed and **equal to the sidecar record**;
* the weight / bias / mask sha256;
* the producing learn record, whose weight sha equals the checkpoint's;
* the parent code: commit `609480a` for seed 0 and `317d773` for seeds 1 / 2, both `source_dirty=false`;
* the parent hardware;
* the retrieval and reconstruction description.

The files stay outside git (`k64l1_ckpt/`, kept and never overwritten). The timing factory and every learning
driver re-verify the weight and bias sha256 before use.

**Exact regeneration is not promised, and it was tested (`regen/`).** The exact producing command was rerun
for C256 graph seed 0, `k64_learn.py --C 256 --treatment prox --alpha 0`, from this study's frozen code, where
the dense kgx kernels and `k64_learn.py` are unchanged.
* The result has **different weight, bias and file sha256**: weight `367878b4…` against the original
  `b016eecf…`.
* Its held-out curve matches the original to about 1e-7: final 0.2472316 against 0.2472317.

So fast FP32 atomics and stochastic rounding make byte-identical regeneration impossible, while the function is
statistically the same. The regenerated checkpoint and the 10 fine-tuned supports are listed with sha256 in
`checkpoints/outside_git_artifacts.sha256`.

### 1.4 Correction to the K64 report (audit item 4)

The K64 report says proximal L1 "makes exact zeros only at α ≥ 10". That is inaccurate. Its own learn records
(`results/k64_large_c_indexfree_l1/learn/`) show exact zeros below α10:

| α | C64 | C256 | C2048 |
|---|---|---|---|
| 0.1 | 9,083 (1.14e-5 of W0) | 9,161 (1.15e-5) | 9,041 (1.13e-5) |
| 1 | 0.231 % | 0.228 % | 0.219 % |

These zeros are as unstructured as those at α ≥ 10 and do not form usable tiles. The historical report and
records are left unchanged; the correction lives here.

## 2. Model, terminology and budgets

**Model.** z_i = b_i + Σ_{k ∈ support(i)} w_ik·x_k, and y_g = Σ_{valid i of g} ReLU(z_i) / n_g.
* Every path accumulates the whole z (tiles + backbone + bias) in one FP32 accumulator before **one**
  activation.
* H and n_g never change; bias-only rows stay, and nothing is renormalized.
* BF16 inputs and parameters, FP32 accumulation and main_grad, Mean(ReLU), zero padding, all valid biases.
* L1AdamW: lr 1e-4, BF16 moments, stochastic rounding, weight decay 0, the race fix, no L1 anywhere.

**Terms.**
* *Pooling output* g.
* *Input supergroup* q, a 64-feature set.
* *Logical record*: a valid block (g, j) of C neurons sharing one set.
* *Row group* u: 16 consecutive neuron slots of one record. At C256 there are 781,312 valid row groups, 992 of
  them partial (15 neurons).
* *Tile* (u, t): the 16 neurons of u × b local inputs. b = 16 gives 4 tiles per group; b = 8 gives 8.
* *Anchor group*: one row group of a record with all 64 inputs enabled.
* *Backbone* A_u: 2 local inputs per row group, enabled for every valid neuron of u with independent weights.

**Densities, recorded separately.**
* `rho_mask` = enabled valid slots / W0, counted exactly.
  * Partial row groups cost their valid neurons only.
  * A tile that overlaps backbone coordinates costs only its non-backbone slots. Those positions are stored as
    structural zeros (`tvmask`), so no edge is stored or counted twice.
* `rho_nnz` = numerically nonzero enabled slots / W0, from the evaluated values. It equals `rho_mask` here: the
  α0 checkpoint and the fine-tuned weights have no exact zeros.
* *Executed work* = the MMA products and CUDA-core FMAs of one step, padding and structural zeros included
  (`executed_work` in every timing record, §7.3).

**Budgets.** Budget = round(ρ·W0) valid slots. Mandatory support is charged first. Extra tiles are taken in
descending score order, with ties broken by ascending flat tile index, while they fit. The shortfall is at most
one tile: ≤ 255 slots, below 3.2e-7·W0, and recorded for every mask. W0 / H are never changed.

**Scores.**
* Tile score: the FP32 sum of |w| over the tile's valid slots, excluding backbone columns. These are raw sums,
  so a partial group's score covers only its valid neurons.
* Anchor: the row group of largest total |w| among the record's full groups. If the record has no full group,
  its single partial group is used; that never happens in these layouts.
* Backbone: the top-2 column |w| sums of the group.
* The random-score controls replace every score with a seeded uniform value under identical constraints and
  budget.

## 3. Implementation and support matrix

| format (variant) | storage | kernels | numerics |
|---|---|---|---|
| dense / dense masked (`idf`) | [G, NB, C, 64] BF16 (+ uint8 mask in L1AdamW) | K64-study kgx idf (unchanged) | split BF16 dZ MMA |
| `tile` (`KgxTileLayer`) | 16×16 compact tiles + 4-bit masks | **the K64 study's measured kernels, unchanged** | split BF16 dZ MMA |
| `tilew16` (`KgxWideLayer`, b16) | the same compact layout | new `kgx_wide.cuh` | split BF16 dZ MMA |
| `tile8` (`KgxWideLayer`, b8) | [n_tiles, 16, 8] + 8-bit masks (`tmeta = first << 8 \| mask`) | new; m16n8k8 per 8-input tile in fwd / z, m16n8k16 per tile in dX / dW | split BF16 dZ MMA |
| `hyb16` (`KgxWideLayer`, backbone) | b16 tiles + `bbk` [n16] (a0 \| a1 << 8) + `bb` [slots, 2] BF16; structural zeros of overlapping tiles in `tvmask` | new; backbone FMAs fused into the same z; backbone dA added by its owning lane into the warp-private dX copy; backbone dv in the exclusive dW task | **mixed**: tiles split BF16 MMA, backbone FP32 scalar dz |
| `lowered` (`LoweredOneTile`) | kgen K = b, C = 16 with identical indices / weights / bias / neuron identity | existing kgen K16 / K8 kernels | split BF16 dZ MMA |

**Dispatching the tile masks.**
* A warp-uniform tile mask is dispatched through a `switch` over each 4-bit nibble of the 8-bit slice mask, so
  every case is straight-line code containing only its active MMAs.
* The first version, with per-slice or per-tile predicates, was if-converted by ptxas into predicated MMAs that
  issue for inactive tiles too. On one-tile supports that made dW 1.4× slower and dX 1.1× slower than the
  measured kernels; the switch fixed it.
* A forward variant that processes a full row group per iteration needed 140–155 registers, dropping from 2
  CTAs per SM to 1, and was slower, so it was not kept.

**Registers and spills.**
* Forward: 128, no spills.
* dX: 172–190, no spills.
* dW: 119–128; the backbone dW variant has a 28-byte spill.

**Status.**
* The three new variants are registered in `variants.KGX_VARIANTS` with shared-memory accounting and
  `kgx_reason` legality checks. They are never auto-selected.
* The new kernel sources are separate files, so every K64-study kernel instantiation is unchanged.
* The backbone ablation is a compile-time flag, `-DSP_ABLATE_KW_BB=1`. It gives deliberately wrong results and
  is labeled `ablation`.

## 4. Correctness gates (before any full-size timing)

**`tests/test_kgx_wide.py`** (58 tests) checks against a **float64 reference of the same masked dense
function**, and against the dense masked kgx kernels:
* every policy, with magnitude and random scores;
* b16 / b8 and split / BF16 dZ;
* B = 40 token tails, ragged H, partial C128 tails and partial row groups;
* all-one, all-zero and one-tile supports;
* bias-only rows;
* backbone-only groups (no extra tile at all);
* tiles overlapping backbone coordinates;
* main_grad accumulation over two backward passes;
* an integer-valued cancellation case with exact z = 0 decisions, and inf / NaN propagating like the dense
  masked kernels;
* the K16 / K8 lowered control as an identical function at C64 / C128 / C512;
* canonical identity (§1.1);
* L1AdamW updates of tile / backbone / bias parameters against the dense masked model with stochastic rounding
  off: they agree to rounding (FP32 atomic order differs, so bitwise trajectories are not claimed), and the
  structural zeros stay zero;
* complete TrainState snapshot round trips and checkpoint round trips of a hybrid layer;
* exact budgets, deterministic ties, one anchor per valid record (from full groups), and no protection or
  anchors in padded groups;
* the full-size skeletons and the explicit rejections (§5);
* the CSR numerics label.

**`tests/test_wps_coverage.py`** checks against brute force:
* layer-wide and per-output tuple multiplicity;
* byte-lookup popcount;
* width statistics;
* an **exhaustive oracle on backbone / anchor supports**: tuples of order above the surviving fan-in are
  uncovered, and anchors contain their whole record.

**Compute Sanitizer** (`sanitizer/summary.tsv`): **8 / 8 PASS**, for memcheck, racecheck (0 hazards),
initcheck and synccheck. The workloads are:
* the new `kgw` workload (`sanitize_kgw_workload.py`): 262 forward / backward / optimizer calls covering every
  policy, b16 / b8 / backbone, C64 / 128 / 512, tails, non-finite gradients, the lowered control and L1AdamW
  with `tvmask`;
* the K64 study's `kgx` workload, rebuilt in the new bundle.

## 5. Feasibility, granularity and coverage bounds (`*/manifest.json`, computed before any job)

**Layout counts.**
* C256: 49,152 valid records, each with a full group, so A = 786,432 anchor neurons.
* C2048: 6,144 records, A = 98,304.
* C64: 195,584 records, A = 3,129,344.

**Mandatory support (skeleton), valid slots and density floor:**

| policy | C64 | C256 | C2048 |
|---|---:|---:|---:|
| P16 (16·H) | 200,000,000 (25.0 %) | 200,000,000 (25.0 %) | 200,000,000 (25.0 %) |
| P8 (8·H) | 100,000,000 (12.5 %) | 100,000,000 (12.5 %) | 100,000,000 (12.5 %) |
| A16 (16·H + 48·A) | 350,208,512 (43.78 %) | **237,748,736 (29.718592 %)** | 204,718,592 (25.59 %) |
| A8 (8·H + 56·A) | 275,243,264 (34.41 %) | **144,040,192 (18.005024 %)** | 105,505,024 (13.19 %) |
| H2 (2·H) | 25,000,000 (3.125 %) | 25,000,000 (3.125 %) | 25,000,000 (3.125 %) |
| H2A (2·H + 62·A) | 219,019,328 (27.38 %) | **73,758,784 (9.219848 %)** | **31,094,848 (3.886856 %)** |

**Rejected as infeasible** (`reject/`, `winners/`). `width_masks.build` rejects each with its reason;
density is never raised and H never lowered:

| request | needs | budget |
|---|---:|---:|
| P16 at ρ0.10 | 200,000,000 | 80,000,000 |
| P8 at ρ0.05 | 100,000,000 | 40,000,000 |
| C256 H2A at ρ0.05 | 73,758,784 | 40,000,000 |
| C64 A8 at ρ0.25 | 275,243,264 | 200,000,000 |
| C64 H2A at ρ0.10 | 219,019,328 | 80,000,000 |

C2048 H2A at ρ0.05 is feasible and was measured (§8.2).

**Granularity bound** on structural bias-only rows, max(0, 1 − 64ρ/b):
* b16: 80 % at ρ0.05, 60 % at ρ0.10, 50 % at ρ0.125, 25 % at ρ0.1875.
* b8: 60 % at ρ0.05, 20 % at ρ0.10, 0 from ρ0.125.

The measured U16 values lie above these bounds: 81.5 / 65.7 / 58.7 / 43.6 %. They are close at ρ ≤ 0.10 and
farther above them at ρ ≥ 0.125, where magnitude ranking keeps several tiles in some row groups.

**One-tile union bound** (Q = 4096, fixed local partition):
* b16 at ρ0.25: 5.142008 % triples, 0.065471 % quadruples. Measured P16: **4.047 %** and **0.0632 %**.
* b8 at ρ0.125: 1.028402 % and 0.005036 %. Measured P8: **0.881 %** and **0.0052 %**.

The quadruple estimates are samples, and their intervals touch the bound.

## 6. Timing protocol

**Per job.** One fresh process (`bench_supergroups.py --canonical-start --identity`):
* the factory loads the actual α = 0 checkpoint and verifies its weight and bias sha256;
* it builds the support from the pre-pruning weights and then the requested format, freeing the dense copy
  before the compact memory is measured.

**Canonical start.**
* Two warm-up steps run.
* Then the **pre-warm-up values are restored**, the moments are zeroed, `step = 0`, the rounding counter is 0
  and main_grad is zero.
* That canonical start is hashed and captured, then restored and fingerprint-verified before every repetition.

**Order and samples.**
* Screen: 2 rounds × 2 repetitions × 2 steps = 8 samples.
* Confirmation: 4 × 2 × 3 = 24 per seed, with seed-specific checkpoints.
* Order: a seeded shuffle, with the unpruned dense control first, in the middle and last.
* Optimizer: L1AdamW without L1. Inputs: seed 23.
* One GPU workload at a time throughout.

**Labels.** *new* = the new `kgx_wide` kernels; *tile* = the K64 study's measured 16×16 kernels; *lowered* =
the K16 / K8 kgen control; *dense masked* = the identical-mask dense reference. All of these run the same
function, from the same values and state.

## 7. Frozen-support training cost

### 7.1 C256 screen (graph seed 0; complete step in ms, median of 8)

The unpruned dense step is 1,547.8 / 1,559.7 / 1,558.4 ms (first / middle / last). The identical-mask dense
masked references take 1,519–1,541 ms for every mask.

| ρ | U16 (bias-only) | P16 | U8 (bias-only) | P8 | A16 | A8 | H2 | H2A |
|---:|---|---|---|---|---|---|---|---|
| 0.05 | 265.8 (81.5 %); tile 282.3 | infeasible | – | infeasible | – | – | 1,027.3 | infeasible |
| 0.10 | 442.9 (65.7 %); tile 474.5 | infeasible | – | – | – | – | 1,089.3 | 1,128.2 |
| 0.125 | 522.2 (58.7 %); tile 562.2 | – | 694.8 (34.4 %) | 818.8; **lowered 644.4** | – | – | – | – |
| 0.1875 | 709.2 (43.6 %); tile 740.7 | – | 823.3 (19.0 %) | 889.4 | – | 921.5 (random 921.1) | – | – |
| 0.25 | 818.6 (31.7 %); tile 863.0 | 909.5 (random 904.5); tile 1,021.5; **lowered 898.1** | 923.3 (10.1 %) | 960.0 | – | 981.2 (random 985.8) | – | – |
| 0.375 | 1,008.1 (15.3 %); tile 1,074.2 | 1,052.6; tile 1,144.6 | – | – | 1,061.2; tile 1,149.8 | – | – | – |
| 0.50 | 1,169.9 (6.3 %); tile 1,231.5 | 1,183.5; tile 1,262.2 | – | – | 1,195.9; tile 1,265.8 | – | – | – |

**Random-score controls.** The random-score controls (identical constraints and budget) run at the same speed
as their magnitude masks, so speed depends on the constraint and budget, not on the ranking.

**Phases** (wide format, `tables.md`):
* dX takes 45–58 % of the step and dW 25–41 %.
* The forward takes 13–16 %; the optimizer scales with ρ (1.3 ms at 0.05, 20 ms dense).
* Host enqueue is 1.3–1.6 ms. Every phase is one launch, plus the bias-only `index_add` where there are
  bias-only rows.
* The first step, JIT included, takes 0.8–1.4 s.

**Memory.**
* Peak memory falls from 7.80 GiB (dense) to 0.70 GiB (H2 0.05) – 4.05 GiB (U16 0.5).
* For the wide formats it comprises compact values, tile metadata (3.1 MB), backbone metadata and the
  structural-zero mask (21–63 MB), main_grad and both moments (`tables.md` → memory).

### 7.2 Backbone attribution (ablation build: every backbone load / product / gradient compiled out, every row group still runs)

| mask | step | ablated | backbone | share | fwd share | dX share | dW share |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2 0.05 | 1,021.0 | 748.7 | 272.2 ms | 26.7 % | 37.9 % | 28.3 % | 16.8 % |
| H2 0.10 | 1,086.1 | 824.9 | 261.2 ms | 24.1 % | 37.0 % | 25.1 % | 15.0 % |
| H2A 0.10 | 1,125.5 | 864.6 | 260.9 ms | 23.2 % | 37.7 % | 25.4 % | 12.1 % |

**The cost of width is mostly per-neuron work, not per-weight work.** About 750 ms of an all-connected step is
spent on per-neuron, per-token work even with almost no tiles: z, ReLU, dz, the recomputation in dX and dW,
and the pooling. The fused scalar backbone adds a quarter on top. U16 at ρ0.05 runs only 18.5 % of its rows
and avoids most of this.

### 7.3 Executed work

**Per format.**
* MMA products per token, divided by W0, equal ρ_mask for the pure tile formats; padding and partial n-tiles
  add < 0.1 %.
* H2 at 0.05 runs 0.0188 W0 of MMA products in the forward, plus 1.02 T CUDA-core FMAs per step for the
  backbone (2 / 4 / 4 per neuron-token in fwd / dX / dW).
* The dense masked reference executes the full dense work for every mask.

### 7.4 Inference (forward only, compressed checkpoint)

| support | new kernels | measured tile | lowered |
|---|---:|---:|---:|
| dense | 205.1 | – | – |
| U16 0.05 | 41.3 | 43.8 | – |
| U16 0.25 | 116.1 | 129.7 | – |
| H2 0.05 | 161.7 | – | – |
| H2A 0.10 | 165.9 | – | – |
| P16 0.25 | 126.6 | 175.2 | **67.8** |
| P8 0.125 | 110.7 (16×8) | – | **50.5** |
| A8 0.25 | 123.4 | – | – |

The lowered one-tile controls are almost 2× faster in the forward.

### 7.5 One-time setup, break-even and refresh cost

**One-time setup (frozen support).**
* Building a support from the dense checkpoint costs 0.16–0.33 s for selection + compression + work lists.
* Checkpoint load and verification take 6.6 s, and the first step 0.8–1.4 s.
* Every frozen compact format repays its setup within **0.21–0.46 steps** against the unpruned dense step,
  because every one saves 360–1,290 ms per step.

**Refresh cost U** (`refresh_cost/`, full size, median of 3; U = selection + compression + work lists + state
remap):
* U16: 0.14–0.28 s. Its work lists dominate, because they cut runs of kept groups.
* P / A / H: 0.12–0.14 s.

**Amortized cost of periodic regrowth** (labeled ARITHMETIC, measured components combined). With a dense
scoring step per refresh, ((T − 1)·s + s_dense + U)/T gives:

| mask | T = 16 | T = 128 | frozen |
|---|---:|---:|---:|
| A8 0.1875 | 969 | 927 | 921.5 |
| H2A 0.10 | 1,163 | 1,133 | 1,128 |
| U16 0.05 | 364 | 278 | 266 |

## 8. Confirmation, other C, and the S128 reference

### 8.1 Confirmation over graph seeds 0 / 1 / 2 (24 samples per seed, seed-specific α0 checkpoints; `confirm/`)

The selection and its criteria were written while the screen was still running, before its results were
read (`confirm_selection.json`). Every compact-versus-dense-masked pair started from identical canonical
hashes in every seed.

| mask | format | per seed (ms) | vs its dense masked | vs unpruned dense |
|---|---|---|---:|---:|
| U16 0.10 | new | 446.2 / 444.0 / 443.6 | ×3.438 ± 0.042 | ×3.528 ± 0.033 |
| H2 0.10 | new | 1,097.3 / 1,093.3 / 1,096.7 | – | ×1.432 ± 0.008 |
| H2A 0.10 | new | 1,128.6 / 1,131.9 / 1,129.3 | ×1.359 ± 0.007 | ×1.388 ± 0.004 |
| U8 0.125 | new | 698.8 / 698.5 / 697.2 | – | ×2.247 ± 0.010 |
| P8 0.125 | new / lowered | 822.9 / 648.5 (means) | ×1.869 / ×2.372 | ×1.906 / ×2.419 |
| U16 0.25 | new / tile | 823.8 / 870.3 | ×1.867 / ×1.767 | ×1.904 / ×1.803 |
| P16 0.25 | new / tile / lowered | 911.1 / 1,029.3 / 905.5 | ×1.693 / ×1.499 / ×1.704 | ×1.722 / ×1.524 / ×1.732 |
| U8 0.25 | new | 929.4 | – | ×1.688 ± 0.007 |
| A8 0.25 | new | 987.3 / 985.7 / 985.1 | ×1.562 ± 0.010 | ×1.591 ± 0.007 |
| U16 0.375 | new | 1,016.6 | – | ×1.543 ± 0.001 |
| A16 0.375 | new / tile | 1,074.5 / 1,158.7 | ×1.440 / ×1.336 | ×1.460 / ×1.354 |
| dense | dense | 1,567.7 / 1,569.2 / 1,569.4 | – | ×1.000 |

**Paired policy and format ratios** (a / b per seed, mean ± t95; above 1 means a is slower):

| a | b | ratio |
|---|---|---:|
| P16 0.25 | U16 0.25 | **1.106 ± 0.002** |
| P16 0.25 lowered K16 | P16 0.25 new | 0.994 ± 0.006 |
| P16 0.25 measured tile | P16 0.25 new | 1.130 ± 0.004 |
| U16 0.25 measured tile | U16 0.25 new | 1.056 ± 0.008 |
| A8 0.25 | U8 0.25 | 1.061 ± 0.006 |
| A8 0.25 | U16 0.25 | **1.197 ± 0.005** |
| U8 0.25 | U16 0.25 | 1.128 ± 0.010 |
| H2A 0.10 | U16 0.10 | **2.541 ± 0.026** |
| H2A 0.10 | H2 0.10 | 1.031 ± 0.009 |
| P8 0.125 lowered K8 | P8 0.125 new | **0.788 ± 0.006** |
| P8 0.125 | U8 0.125 | 1.179 ± 0.008 |
| A16 0.375 | U16 0.375 | 1.057 ± 0.004 |
| A16 0.375 measured tile | A16 0.375 new | 1.078 ± 0.004 |

The seed-0 confirmation means reproduce the screen within 1 %.

### 8.2 Low-density anchor test at C2048 (graph seed 0; `c2048/`)

| mask | step | vs its dense masked | bias-only | triples | quadruples | per output = potential |
|---|---:|---:|---:|---:|---:|---|
| **H2A 0.05** | 1,020.2 | ×1.49 | 0 | **61.636 %** | **5.559 %** | yes |
| H2 0.05 | 1,016.7 | ×1.50 | 0 | 14.110 % | 0.349 % | – |
| U16 0.05 | 259.7 (tile 276.4) | ×5.84 | 81.5 % | 21.727 % | 0.798 % | – |

The dense C2048 step is 1,550–1,554 ms. **H2A makes 5 % density keep every row connected and the potential
coverage at C2048, at ×3.9 the U16 time.**

### 8.3 Winner checks at C64 / C2048 (graph seed 0; predeclared in `winners_selection.json`; `winners/`)

| mask | C64 | C2048 |
|---|---|---|
| U16 0.25 | 857.4 | 809.0 |
| P16 0.25 (new / tile / lowered) | 943.6 / 1,055.5 / **815.1** | 886.5 / 1,008.1 / 1,383.5 |
| U8 0.25 | 963.8 | 910.4 |
| A8 0.25 | rejected (needs 34.41 %) | 948.3 |
| P8 0.125 (new / lowered) | 857.4 / **617.8** | 800.0 / 939.9 |
| U16 0.10 | 453.1 | 432.7 |
| H2A 0.10 | rejected (needs 27.38 %) | 1,091.4 |
| dense | 1,580–1,584 | 1,552–1,554 |

**Observations.**
* The policy ordering holds at C64 and C2048.
* Anchors are much more expensive at C64, which has 4× the records.
* **The lowered control's speed depends on C, through its dX.** For P16, its dX takes 548 ms at C64, 639 ms
  at C256 and 1,107 ms at C2048. kgen's per-record dX slab read-modify-write grows with the record length
  (12,288 items of 64 C16 blocks at C2048), while the new kernels' register-accumulated dX stays at 444–472 ms.
* Its forward (48–87 ms) and dW (184–213 ms) are always faster than the new kernels' (forward 111–135 ms, dW
  295–330 ms).

### 8.4 Dense C256 / S128 / R256 high-coverage reference (graph seeds 0 / 1 / 2; `confirm_s128/`; separately labeled)

| variant | per seed (ms) | mean |
|---|---|---:|
| kgen reference | 1,674.5 / 1,674.1 / 1,673.4 | 1,674.0 |
| direct256 | 1,694.0 / 1,696.0 / 1,693.4 | 1,694.5 |

* It is ×1.067 the S64 index-free dense step, and the E24 same-session anchor took 1,684–1,695 ms.
* The point has 97.79 % layer triples and 31.28 % quadruples (K64 study coverage, seed 0), against 61.64 % /
  5.56 % for every S64 support here.
* Sparse S128 execution is outside this study.

## 9. Coverage, multiplicity and effective width (`coverage/`, `coverage_ft/`)

**Method** (`wps_coverage.py`).
* It runs on the exact timed graph, checkpoint and support: **55 / 55 analyzed supports carry the
  element-mask sha256 of the timed mask**.
* Per-neuron 64-bit local supports come from the element mask; supports of different neurons are never
  united.
* Layer-wide pairs and triples are exact.
* Quadruples use a predeclared sample: 10M, or 50M when the enabled density is ≤ 0.10, with seeded distinct
  tuples shared by all supports and Clopper–Pearson 95 % intervals.
* Per output: features for all 1,024 outputs. Exact pairs, 2M sampled triples and 10M sampled quadruples on a
  **fixed selection of 38 outputs** shared by every support:
  * 32 seeded outputs;
  * the 4 lowest-feature outputs of the dense potential;
  * both count classes, including the 12,207-neuron partial-tail class.
  * The 4 lowest-feature outputs of each support are added and flagged.
* Multiplicity = the popcount of the AND of the tuple features' neuron bitsets: the number of valid neurons
  whose own support contains the tuple.
* The exact-nonzero supports equal the enabled ones, before and after fine-tuning; thresholded coverage at
  |w| > 0.001 and 0.01 is also equal.

**Coverage and width at C256** (seed 0; full table in `tables.md`):

| support | bias-only | layer pairs | triples | quadruples [95 % CI] | per-output triples median | covered-quad multiplicity (layer, median) | per-output covered-triple multiplicity (median) |
|---|---:|---:|---:|---|---:|---:|---:|
| dense potential | 0 | 100 % | 61.636 % | 5.558 [5.544, 5.572] % | 1.116 % | 3,072 | 256 |
| U16 0.05 | 81.5 % | 99.874 % | 22.075 % | 0.8164 [0.8139, 0.8189] % | 0.0563 % | 16 | 16 |
| H2 0.05 | 0 | 99.832 % | 14.213 % | 0.3529 [0.3512, 0.3545] % | 0.0267 % | 16 | 16 |
| U16 0.10 | 65.7 % | 100 % | 41.988 % | 2.155 % | 0.147 % | 32 | 16 |
| H2 0.10 | 0 | 100 % | 39.133 % | 1.665 % | 0.110 % | 16 | 16 |
| **H2A 0.10** | 0 | 100 % | **61.636 %** | **5.559 [5.553, 5.566] %** | **1.116 %** | 192 | 16 |
| U16 0.125 | 58.7 % | 100 % | 47.813 % | 2.769 % | 0.207 % | 32 | 16 |
| U8 0.125 | 34.4 % | 100 % | 37.623 % | 1.362 % | 0.107 % | 16 | 16 |
| P8 0.125 (floor) | 0 | **27.077 %** | **0.881 %** | **0.0052 %** | 0.011 % | 384 | 32 |
| U16 0.1875 | 43.6 % | 100 % | 57.125 % | 4.227 % | 0.363 % | 48 | 16 |
| U8 0.1875 | 19.0 % | 100 % | 53.597 % | 2.868 % | 0.229 % | 32 | 16 |
| P8 0.1875 | 0 | 100 % | 46.308 % | 2.051 % | 0.163 % | 16 | 16 |
| **A8 0.1875** | 0 | 100 % | **61.636 %** | **5.558 %** | **1.116 %** | 192 | 16 |
| U16 0.25 | 31.7 % | 100 % | 60.903 % | 5.140 % | 0.544 % | 64 | 16 |
| U8 0.25 | 10.1 % | 100 % | 60.363 % | 4.267 % | 0.396 % | 32 | 16 |
| P8 0.25 | 0 | 100 % | 59.514 % | 3.975 % | 0.351 % | 32 | 16 |
| P16 0.25 (floor) | 0 | **44.982 %** | **4.047 %** | **0.0632 %** | 0.060 % | 768 | 64 |
| **A8 0.25** | 0 | 100 % | 61.636 % | 5.558 % | 1.116 % | 192 | 16 |
| U16 0.375 | 15.3 % | 100 % | 61.635 % | 5.545 % | 0.874 % | 192 | 32 |
| P16 0.375 | 0 | 100 % | 61.608 % | 5.482 % | 0.743 % | 128 | 32 |
| **A16 0.375** | 0 | 100 % | 61.636 % | 5.558 % | 1.116 % | 240 | 16 |
| U16 0.5 / P16 0.5 / A16 0.5 | 6.3 / 0 / 0 % | 100 % | 61.636 % | 5.558 % | 1.058 / 1.039 / 1.116 % | 416 / 384 / 448 | 48 |

**Anchors** (A16 0.375 / 0.5, A8 0.1875 / 0.25, H2A 0.10 and C2048 H2A 0.05, including the random-score A8):
* Every valid record holds a full-support anchor group with 64 nonzero weights per neuron; the structural
  check covers all outputs.
* No anchor weight is exactly zero; 6.0 % are ≤ 0.01, matching the init distribution.
* Layer-wide coverage equals the dense potential.
* Per output, on the fixed 38 outputs, they equal the potential exactly.
* This is support existence. **The multiplicity of a covered tuple drops to one row group per record** (16
  neurons per output, against 256 for the dense potential). The layer-wide median covered quadruple is held by
  192 neurons, against 3,072.

**Multiplicity granularity.** Multiplicities are multiples of 16 (or 15 in partial groups), because the
neurons of a row group share one support. One-tile supports (P16 / P8 at their floors) cover few tuples, but
each covered tuple is held by many neurons (768 / 384).

**Width histograms.** `coverage/*.json` → `width` holds:
* exact all-neuron mask-degree and nonzero fan-in histograms, 0–64;
* bias-only counts under both definitions;
* full-support neurons;
* input-dependent neurons per output (min / p10 / median / p90 / max).

Every connected support has 12,207 input-dependent neurons per output (the median); U16 has 2,256 at ρ0.05,
4,176 at ρ0.10 and 8,336 at ρ0.25.

## 10. Learning diagnostics (synthetic; not accuracy claims)

### 10.1 Fixed-support fine-tuning (`finetune/`; the K64 study's synthetic task)

**Setup** (identical for every treatment):
* the actual α0 checkpoint at C256, graph seed 0, with its proj / head state;
* the same teacher, batches and loss;
* the support applied as the dense masked layer, which is the compact formats' function;
* 32 updates with fresh optimizer state for every parameter;
* L1AdamW lr 1e-4 for the layer, AdamW lr 1e-3 for proj / head, no L1;
* mandatory support never pruned.

**Run-to-run repeats** of 7 treatments agree to **≤ 5e-8** in held-out loss (`finetune_repeat/`). Seed-to-seed
variation of the task was not measured.

Held-out loss (checkpoint 0.25557), pruned → fine-tuned:

| ρ | U16 | U8 | P16 / P8 | A16 / A8 | H2 / H2A |
|---:|---|---|---|---|---|
| 0.05 | 0.38080 → 0.26421 | – | – | – | H2 0.34694 → **0.26375** |
| 0.10 | 0.35766 → 0.26336 | – | – | – | H2 0.33241 → 0.26293; H2A 0.33885 → 0.26299 |
| 0.125 | 0.34768 → 0.26290 | 0.33436 → 0.26273 | P8 0.32107 → 0.26260 | – | – |
| 0.1875 | 0.32672 → 0.26187 | 0.31437 → 0.26155 | P8 0.30760 → 0.26157 | A8 0.31153 → 0.26158 (random 0.26189) | – |
| 0.25 | 0.31043 → 0.26076 | 0.30019 → 0.26074 | P16 0.29684 → 0.26085 (random 0.26112); P8 0.29699 → 0.26067 | A8 0.29938 → 0.26075 (random 0.26109) | – |
| 0.375 | 0.28784 → 0.25899 | – | P16 0.28256 → 0.25911 | A16 0.28315 → 0.25913 | – |
| 0.50 | 0.27393 → 0.25721 | – | P16 0.27227 → 0.25721 | A16 0.27250 → 0.25722 | – |

**Interpretation.**
* The dense model continued for the same 32 updates reaches **0.25023**.
* Connected supports lose much less **at pruning time** (for example 0.347 against 0.381 at ρ0.05).
* After 32 updates their held-out loss differs from U16's at the same ρ by −0.17 % … +0.05 % of the loss:
  better at ρ ≤ 0.1875, equal at ρ ≥ 0.25.
* The magnitude-ranked anchors / protection beat their random-score controls by about 3e-4.
* No protected tile and no backbone ever became numerically all-zero (0 violations; no repair needed). The
  masks were preserved exactly.

**Input sensitivity** (fixed sample of 65,536 neurons, 2,048 held-out tokens, after fine-tuning).
* Every connected neuron switches (0 < activation fraction < 1). None is always or never active, and none is
  connected but all-zero.
* What separates the supports is the **input-driven variance**, Var_t(z − b). The dense median is 0.205.
* Fraction of sampled neurons with Var > 0.01 (≈ 5 % of the dense median):

| ρ | U16 | U8 | P | A | H2 / H2A |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 18.3 % (its connected rows) | – | – | – | H2 57.1 % |
| 0.10 | 34.2 % | – | – | – | H2 63.4 %, H2A 56.9 % |
| 0.125 | 41.1 % | 64.2 % | P8 96.9 % | – | – |
| 0.1875 | 56.1 % | 79.4 % | P8 97.5 % | A8 96.9 % | – |
| 0.25 | 68.0 % | 88.8 % | P16 99.99 %, P8 98.3 % | A8 97.8 % | – |
| 0.375 | 84.3 % | – | P16 99.99 % | A16 99.99 % | – |

About 40 % of the backbone-only rows are **connected but weakly input-dependent**: their median input-driven
variance is 0.012, against 0.12 for U16's connected rows. They improve the structural counter, but not useful
width.

### 10.2 Interaction diagnostic (`interaction/`, `interaction_capacity/`; reduced, labeled configuration)

**Fixed input mixing.** The layer reads the raw inputs. The only trainable read-out is a per-output affine, so
no trainable projection can route interactions around the connectivity.

**Targets.** Every record carries a single-index term f_k = ReLU(x_1 + … + x_k − 1) (standardized) over k
inputs, taken one from each of k distinct 16-input blocks of the record. The order is k = 1 + (g mod 4);
order 1 is a linear learnability control.
* For k ≥ 2, part of the variance is not additive across the blocks. The **exact additive floor** is computed
  from the Gaussian closed-form additive projection on the held-out batches (`interaction_floors.json`).
* No support whose neurons each see one block can beat it.
* "Explained interaction" = 1 − nMSE / floor.

**Model and training.**
* Reduced configuration: H = 262,144, so n_g = 256 and one C256 record per output, with D = G = 1024, S64 / R256
  and a fresh init. Training: lr 3e-3, 1,000 updates of M = 512.
* At full H each term is diluted over 48 records, and the dense control did not learn orders ≥ 2.
* Pure interactions were tried first (Gaussian products; sign-parity sums of ReLUs). The dense control did not
  learn them for k ≥ 3, so they are not used.

**Correction to the capacity floors.** The first capacity analysis divided by the single-term floor. With
T = 16 terms per record, terms share inputs, so their additive parts add coherently and **the true floor is
lower**: 0.193 / 0.220 / 0.227 against 0.303 / 0.340 / 0.349. The table uses the exact floors, and the measured
MSEs are unchanged.

| support | 1 term per record: explained, order 2 / 3 / 4 | 16 terms per record (capacity): explained, order 2 / 3 / 4 |
|---|---|---|
| dense | 0.999 / 0.999 / 0.999 | **0.782 / 0.852 / 0.876** |
| U16 0.10 | −0.32 / −0.29 / −0.25 | −1.08 / −0.79 / −0.68 |
| H2 0.10 | −0.32 / −0.24 / −0.24 | −0.95 / −0.69 / −0.62 |
| H2A 0.10 | 0.871 / 0.945 / 0.971 | −0.20 / 0.08 / 0.19 |
| P8 0.125 (floor) | −0.38 / −0.27 / −0.26 | −0.84 / −0.59 / −0.49 |
| A8 0.1875 | 0.821 / 0.919 / 0.956 | 0.08 / 0.24 / 0.31 |
| P16 0.25 (floor) | −0.06 / −0.06 / −0.04 (random −0.07 / −0.05 / −0.04) | −0.31 / −0.18 / −0.13 |
| U16 0.25 | 0.537 / 0.468 / 0.461 | 0.01 / 0.19 / 0.25 |
| U8 0.25 | 0.505 / 0.457 / 0.444 | 0.24 / 0.30 / 0.34 |
| A8 0.25 | 0.840 / 0.908 / 0.936 | 0.31 / 0.42 / 0.45 |
| A16 0.375 | 0.882 / 0.907 / 0.925 | 0.34 / 0.47 / 0.52 |
| P16 0.5 | 0.960 / 0.892 / 0.864 | 0.52 / 0.62 / 0.66 |
| U16 0.5 | 0.971 / 0.903 / 0.880 | 0.54 / 0.64 / 0.68 |

**Findings.**
* The one-tile floors (P16 0.25, P8 0.125) never beat the additive floor, as the bound requires.
* The anchors (16 full-input neurons per record) explain most of a **single** term per record, but only about
  half of what dense explains when the record has 16 terms.
* In the capacity variant, A8 0.25 is still ahead of U8 / U16 at the same budget. H2A 0.10 falls to about
  zero.
* A few anchors can stand in for coverage, **not** for broad capacity.

### 10.3 Periodic support refresh (`refresh/`; selection predeclared in `refresh_selection.json`)

**Rule.**
* Policies: A8 0.1875 and H2A 0.10, with 256 updates on the fine-tuning task. The frozen support (T = 0) is the
  control.
* Every T updates, prune 10 % of the extra tiles by smallest |w| L1 (mandatory support never touched, no neuron
  ablation).
* Regrow absent tiles by the largest **dense** |dL/dW| L1 at their zero values, until the freed valid-slot
  budget is used, within one tile. Grown tiles start at zero with zero moments, so this is true regrowth, not
  reselection.

| policy | T | refreshes | held-out after 256 updates | tiles differing from the start | enabled / budget | mandatory intact |
|---|---:|---:|---:|---:|---|---|
| A8 0.1875 | frozen | 0 | 0.19774 | 0 | 150,000,000 / 150,000,000 | yes |
| A8 0.1875 | 128 | 1 | 0.19752 | 9,312 | 149,999,960 | yes |
| A8 0.1875 | 16 | 15 | 0.19791 | 9,313 | 149,999,096 | yes |
| H2A 0.10 | frozen | 0 | 0.23364 | 0 | 79,999,904 / 80,000,000 | yes |
| H2A 0.10 | 128 | 1 | 0.23333 | 4,912 | 79,999,712 | yes |
| H2A 0.10 | 16 | 15 | 0.23391 | 4,907 | 79,997,648 | yes |

**Result.** Refreshing changes held-out loss by at most 3e-4 in either direction.
* With T = 16, about as many tiles differ from the start as after a single refresh. The tiles regrown at zero
  are the weakest at the next refresh, so the support oscillates instead of moving.
* A refresh costs 0.22 s in the dense masked learning run, and U = 0.12 s in the compact formats (§7.5).
* **A periodic refresh is not justified by these results.** A gradient-scored rewiring with a warm start or a
  longer horizon could behave differently; it was not tried.

## 11. Which apparent gains need the new formats?

**Equivalent to an existing lower-fan-in execution.** These are the *exactly-one-tile* supports, P16 at
ρ0.25 and P8 at ρ0.125. They are the same function as a K16 / K8, C16 kgen layer with the identical indices,
weights, bias and neuron identity, and the identical canonical and mask hashes. That layer runs the function at
×0.994 / ×0.788 of the new kernels' time (C256). Their "every row connected" result also costs almost all
multi-block coverage.

**Need the new formats:**
* multi-tile protected supports above the floor;
* anchored supports (A16 / A8 / H2A), whose anchor groups keep 4 or 8 tiles;
* the backbone hybrids;
* b8 multi-tile supports.

The new 16×16 kernels also beat the K64 study's measured tile kernels by 5.6–13 % on identical masks, in the
confirmation.

## 12. Limitations

**Task scope.** Synthetic diagnostics only:
* The fine-tuning task is the K64 study's composed regression at M_train = 512 with 32 updates.
* The interaction task is a reduced configuration with H = 262,144 and one record per output.
* Neither is an application-accuracy result.
* Graph-seed variation of the learning results was not measured; only run-to-run repeats were.

**Kernels.**
* The tile and backbone kernels are one compiled tiling on one GPU (RTX 3090).
* The per-neuron cost of all-connected supports (≈ 750 ms) may be reducible: it was not optimized beyond the
  per-n-tile loop.
* The K16 / K8 lowered control's faster forward and dW suggest a better forward for one-tile supports.
* No 16×4 / 16×2 tile kernels were built, per the scope; their bias-only bounds are in §5.

**Formats and execution.**
* Width-preserving formats were trained only as dense masked in the learning diagnostics. The compact formats
  are the same function by test, not by measured training trajectory: FP32 atomics differ.
* Sparse S128 execution was not implemented. The S128 point is a dense reference only.

**Refresh.** Only one refresh rule was tested, with a zero-initialized regrowth. The amortization table is
arithmetic.

## 13. Pareto sets

Sets over complete step (new kernels), useful width (the sample fraction with input-driven variance > 0.01),
layer triples and held-out loss after fine-tuning, primary masks only. Tolerances: 1 % step, 1 point width,
0.5 point triples, and 1e-4 held-out as a **materiality threshold** (repeat noise is ≤ 5e-8).

**Step vs held-out only:** U16 at 0.05, 0.10, 0.125, 0.1875, 0.25, 0.375 and 0.5; U8 0.125; dense.
Unconstrained pruning is the best time / quality tradeoff on this task.

**Constrained to useful width ≥ 95 % and layer triples within 0.1 point of the potential:**

| mask | step ms | held-out |
|---|---:|---:|
| A8 0.1875 | 921 | 0.26158 |
| A8 0.25 | 981 | 0.26075 |
| P16 0.375 / A16 0.375 | 1,053 / 1,061 | 0.25911 / 0.25913 |
| P16 0.5 | 1,184 | 0.25721 |
| dense | 1,559 | 0.25023 |

**ρ ≤ 0.10 with every row connected:** H2 0.05 and 0.10 on time and quality. H2A 0.10 is the only member that
also keeps the potential coverage and the single-term interactions, at +3.1 % time.

**Recommended small set:**
* **U16** at the target density, when bias-only rows are acceptable;
* **A8 at 0.1875–0.25**, the cheapest full-width, full-coverage support (+20–30 % step time against U16);
* **P16 / A16 at 0.375**, nearly the U16 cost (+4–6 %) with full width;
* **H2A at 0.10** (C256) or 0.05 (C2048), when every row must stay connected at very low density (×2.5–3.9 the
  U16 time).

## 14. Raw evidence and reproduction

**Contents of `results/width_preserving_sparsity/`:**
* Every stage directory has `plan.json` and `manifest.json` (written before any job), `status.jsonl`, per-job
  records and logs. The stages are screen, c2048, ablation, reject, inference, coverage, finetune, coverage_ft,
  interaction, regen, confirm, winners, confirm_s128, refresh_cost, refresh, interaction_capacity and
  finetune_repeat.
* `tables.md` and `csv/`, generated by `wps_report.py`.
* `interaction_floors.json`, `sanitizer/`, `tests/` and `checkpoints/`.
* The predeclared selections: `confirm_selection.json`, `winners_selection.json`, `refresh_selection.json` and
  `confirm_s128_selection.json`.
* The stage scripts and driver logs.

**Outside git:**
* the α0 checkpoints (`k64l1_ckpt/`);
* the fine-tuned supports and the regenerated checkpoint (`wps_ckpt/`, with sha256 in
  `checkpoints/outside_git_artifacts.sha256`).

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
export GRANULE_RESULTS_ROOT=/path/outside/the/checkout
cd sparse_pooling/benchmarks
$PY wps_checkpoints.py --out $GRANULE_RESULTS_ROOT/checkpoints/manifest.json
for st in screen c2048 ablation reject inference coverage finetune coverage_ft interaction regen \
          refresh_cost interaction_capacity finetune_repeat; do
  $PY wps_jobs.py $st --out $st                 # one fresh process per job; plan.json before any job
done
$PY wps_jobs.py confirm --out confirm --selection confirm_selection.json
$PY wps_jobs.py winners --out winners --selection winners_selection.json
$PY wps_jobs.py refresh --out refresh --selection refresh_selection.json
$PY k64_campaign.py confirm --out confirm_s128 --selection confirm_s128_selection.json
$PY wps_interaction_floor.py --root $GRANULE_RESULTS_ROOT --out $GRANULE_RESULTS_ROOT/interaction_floors.json
$PY wps_report.py --root $GRANULE_RESULTS_ROOT --out $GRANULE_RESULTS_ROOT/tables.md --csv $GRANULE_RESULTS_ROOT/csv

# one job of each new family by hand (C256 seed 0, the alpha 0 checkpoint of the K64 study)
CK=/home/evga/Projects/pamachine/k64l1_ckpt/C256_R256_prox_a0_step128.pt
COMMON="--hidden 12500000 --C 256 --S 64 --R 256 --graph-seed 0 --init-seed 17 --input-seed 23 --rounds 2 --reps 2
        --iters 2 --kcsr --canonical-start --identity --layer-factory k64_factory:wide --opt-factory k64_opt:l1"
$PY bench_supergroups.py $COMMON --opt-args '{"l1": "none"}' --out repro/H2A_0.1 \
  --factory-args '{"checkpoint": "'$CK'", "format": "wide", "policy": "H2A", "density": 0.1}'
$PY bench_supergroups.py $COMMON --opt-args '{"l1": "none"}' --out repro/A8_0.25 \
  --factory-args '{"checkpoint": "'$CK'", "format": "wide", "policy": "A8", "density": 0.25}'
$PY bench_supergroups.py $COMMON --opt-args '{"l1": "none"}' --out repro/P8_0.125_lowered \
  --factory-args '{"checkpoint": "'$CK'", "format": "lowered", "policy": "P8", "density": 0.125}'
#   other formats of a mask: "dense_masked" (identical-mask reference), "tile" (the K64 study's 16x16 kernels);
#   random-score control: add "score": "random", "seed": 1; backbone ablation: SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_KW_BB=1
$PY wps_coverage.py --C 256 --checkpoint $CK --policy A8 --rho 0.1875 --thresholds 0.001,0.01 --out cov_A8.json
$PY wps_learn.py --policy H2A --rho 0.1 --out ft_H2A.json
$PY wps_interaction.py --policy A8 --rho 0.25 --out int_A8.json          # --terms 16: capacity variant
$PY wps_refresh.py --policy A8 --rho 0.1875 --period 16 --steps 256 --out refresh_A8_T16.json
$PY wps_refresh_cost.py --policy H2A --rho 0.1 --out cost_H2A.json
# tests and sanitizers
(cd ../tests && $PY -m pytest -q test_kgx_wide.py test_wps_coverage.py test_kgx.py test_k64_coverage.py)
bash run_sanitizer.sh $PY kgw kgx
```
