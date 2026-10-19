# Production deterministic reduction, canonical bucket statistics, standalone C = 8 dW projection

RTX 3090 (GA102, sm_86, 24 GiB, driver 550.67), nvcc 12.1.66 (`-O3 -arch=sm_86 --use_fast_math
-lineinfo -std=c++17`), PyTorch 2.7.0+cu128, Triton 3.3.0. C = 8 GranuleLayer, K = 8, Mean(ReLU),
BF16 parameters / inputs, FP32 MMA accumulation and main gradients, split gradients unless stated,
FusedAdamW (with the barrier fix of the previous phase, unchanged). Production shape D = G = 1024,
H = 100,000,000, M = 8192, S / R = (16, 4), (32, 8), (64, 16), Q = R·D/S = 256, balanced
permutation graph. Branch `c8-deterministic-reduction-dw-projection`, started from
`c8-scale-dispatch-fused-backward` @ `c9a591e` (no newer descendant existed). Raw evidence:
`results/determinism_metrics_dw_projection/`.

Labels: **measured** (CUDA-event timings, exact comparisons), **computed** (from work lists,
metadata or source), **static** (SASS / resource dumps), **ablation** (wrong-result builds, timing
only), **interpretation** (consistent with the measurements, not separately verified). No hardware
performance counters are available on this machine.

## 1. Answers

1. **Which production operations were nondeterministic; which were races?** (§2.2)
   * *Valid unordered FP32 sums* (race-free, but rounding depends on the arrival order):
     * the grouped forward's FP32 `atomicAdd` into the output. At H = 100M this is 256 atomics
       per (output, token): one per (output, supergroup) bucket item;
     * the grouped dX flush, a global FP32 atomic from every CTA whose chunk contains the feature.
       At H = 100M that is 12 / 24 / 48 CTAs per (feature, token) for S = 16 / 32 / 64;
     * the shared-memory 64-bit CAS of the S = 64 `shared_t64` plan (8 warps share one copy);
     * the generic `k_dx` (CAS + atomics), the fused backward's atomic dX flush and the Triton kernels.
   * *Already deterministic*: dW / db (one CTA per unit, register accumulation in token order),
     FusedAdamW with the barrier fix (disjoint element ranges per program), and the generic Mean forward.
   * *Actual races*: none new. The only real race is the previous phase's FusedAdamW lost-gradient
     race (main_grad loaded and cleared in different thread layouts). Its fix is preserved and covered
     by the training runs (every optimizer state tensor compared bitwise at every step).
   * Compute Sanitizer finds no invalid access, uninitialized read or hazard in the new kernels (§3.5).
2. **Does the new mode reproduce full R > 1 training bitwise at H = 100M / M = 8192?** Yes, for
   every case run; there is no failure to diagnose (§3.2–3.3).
   * *Production:* 4 changing-batch optimizer steps, SR off, each repeated from the identical
     complete state in a second process. Covered: S/R = (16, 4), (32, 8), (64, 16) seed 0, S64 seed 1,
     S64 BF16 gradients, and S64 with the compact dW. All 33 recorded tensors match at every step (165 records per case):
     y, loss, dX, dW, db, every parameter, moment and master / counter.
   * *Workload realism:* every run has R > 1 and 768 dX chunks (3 per supergroup). Normal chunk
     sizes, no reduced H / M.
   * *Small / medium:* 40 two-process pairs (33 small / medium, 2 token-window, 5 after the dW change), 16 steps each. They cover 3 seeds × 3 S/R pairs,
     SR on, BF16, FP32 moments / master and token windows, and every pair is bitwise.
3. **What order and ownership enforce it?** (§2.3)
   * *Forward and dX* store each CTA's partial into workspace rows written by that CTA only:
     * forward: row = bucket item;
     * dX: row = (chunk, local feature); warp-private copies are summed in copy order 0..7.
   * *Reduction kernel* `sp_sg_reduce` owns each final (output | feature, token). It sums that
     destination's contributor rows in ascending static order, from +0.0f, and writes once. The
     order lists are built once from the work lists. Every row was verified written exactly once,
     and the row count per destination equals the fast mode's writer count.
   * *dW / db and optimizer*: unchanged exclusive owners.
   * *Fallbacks*:
     * the non-finite guard selects the in-kernel generic epilogue of the same kernels (one path,
       still ordered);
     * the generic forward has no atomics for Mean;
     * fused backward, generic dX, Triton and shared-copy dX plans are rejected with `ValueError`.
4. **What does deterministic mode cost per S/R at H = 100M?** (§4) Paired complete-step medians
   (4 alternating rounds × 2 restored repetitions × 3 steps):

   | S / R | fast | det | Δ step (median [min, max]) | extra launches | workspace | alloc / reserved peak (fresh process) |
   |---|---:|---:|---|---|---|---|
   | 16 / 4 | 2429.6 ms | 2438.2 ms | **+0.34 %** [−0.08, +2.37] | fwd +15, dX +1 | 1 GiB + 384 MiB | 8.84 → 9.66 / 10.38 → 10.38 GiB |
   | 32 / 8 | 2673.0 | 2678.5 | **+0.21 %** [+0.16, +0.32] | fwd +15, dX +1 | 1 GiB + 768 MiB | 8.84 → 9.66 / 10.38 → 10.38 |
   | 64 / 16 | 3052.4 | 3058.0 | **+0.20 %** [+0.10, +0.25] | fwd +15, dX +3 | 1 GiB + 1020 MiB | 8.84 → 9.84 / 10.38 → 10.38 |
   | 64 / 16 BF16 | 2850.4 | 2856.5 | **+0.25 %** [+0.18, +0.31] | fwd +15, dX +3 | as above | 8.84 → 9.84 / 10.38 → 10.38 |

   * *Where the time goes:* almost all of it is the forward reduction, about 10 ms per step:
     8 GiB of partials read once, 8 windows × 2 launches. The dX reduce adds 0.5 / 1.0 / 1.9 ms.
   * *Composed model* (projection → layer → head, S64): +0.26 %.
   * *Medium shapes are costlier at S64.* Deterministic mode replaces the fast S64 `shared_t64` plan
     with `private_t32` at low bucket occupancy, costing +11 % to +24 % there (H ≤ 10M; §4.3).
5. **What are the H = 1M bucket statistics, and did any dispatch decision change?** (§5)
   * *H = 1M (C8, S64 R16, G 1024, seed 0):*
     * U = 31,744 units over 262,144 possible buckets, with Nnz = 31,744;
     * occupancy 0.1211;
     * **mean per nonempty bucket 1.000**, **mean per possible bucket 0.1211**.
   * *The mislabel:* the old report's H = 1M rows (report lines 296 split, 328 BF16) printed 0.12
     labelled as the nonempty mean. It was the possible-bucket mean; the logged runtime field was
     1.0. Every other historical row printed the nonempty mean (truncated to 2 decimals in
     5 rows). Audit: 50 historical files reconstructed, 0 mismatches.
   * *Dispatch:* **no decision changes.** The code always consumed the nonempty mean, and in every
     audited row both means select the same plan.
6. **Are statistics canonical and off the hot path?** Yes.
   * *One function:* `supergroups.bucket_statistics`, schema `granule.bucket_stats/2`. Runtime
     metadata, dispatch reasons, benchmark JSON / CSV, audit and ownership tools all read it.
   * *Off the hot path:* it is computed once when `SgExec` is built, from the host bucket table.
     Dispatch reads cached Python numbers: no per-step `.item()`, sync or online measurement.
7. **Did standalone dW remove the intended MMA, keep the numerics, and get faster?** (§6)
   * *MMA removed:* yes, half the HMMA.1688 (static: 32 → 16 in the production kernel; computed:
     6.25 G instructions and 12.8 TFLOP issued removed at H = 100M).
   * *Numerics:* bitwise equal dW / db in every test, and complete training trajectories identical
     to the padded path (5 cases, every recorded tensor).
   * *Speed: slower.* dW kernel +3.0 to +4.1 % (split, 3 seeds × 3 S/R pairs) and +8.5 % (BF16);
     complete step +1.2 to +1.5 % (split) and +3.0 % (BF16), the same in fast and deterministic mode.
   * *Costs:* the mask conversion adds 2 MOVM per block-tile and keeps one more register (127 → 128),
     with no spills in the production configuration. Attribution (§6.4): the mask-operand epilogue costs +1.5 % of dW in split mode and +6.0 % in BF16. The two `movmatrix` transposes cost more than the saved MMA: without them, an ablation that gives the wrong result, compact would be 1.3–1.8 % faster than padded in split dW.
8. **What are the defaults, experimental options and unsupported cases?** (§7)
   * *Defaults unchanged*:
     * fast mode stays the default (`reduction="auto"` → fast unless
       `torch.use_deterministic_algorithms(True)` is set);
     * the fast S64 dispatch and thresholds are unchanged;
     * padded dW projection stays the default (`cuda_grouped_dw_projection="auto"` → padded).
   * *New explicit options:* `reduction="deterministic"` (production), `deterministic_workspace_bytes`,
     and `cuda_grouped_dw_projection="compact"` (experimental, correct, slower).
   * *Unsupported in deterministic mode*, all raising `ValueError`: fused backward, generic or
     Triton dX, `shared_t64` / shared-copy dX plans, and pooling other than Mean.
   * *C64* is supported in deterministic mode (control).
   * No held-out run was needed, since no default changed.

Scope of the repeatability claim: same graph, inputs, parameters, optimizer and RNG state, resolved
plan, this GPU model, driver and toolchain. **Not claimed:**
* across GPU models, compiler or driver versions, or precision modes;
* across different reduction trees. Deterministic and fast mode differ in the last bits, as any two
  summation orders do; changing S / R / chunking / plan changes the tree.

## 2. Track A: deterministic execution

### 2.1 API and resolution

`PhysicalPlan.reduction ∈ {"auto", "fast", "deterministic"}` (`config.resolve_reduction`):

| Request | `torch.use_deterministic_algorithms` off | on | on, `warn_only=True` |
|---|---|---|---|
| `"auto"` (default) | fast | deterministic (strict) | deterministic (unsupported layer → warning, fast) |
| `"fast"` | fast | `RuntimeError` | warning, fast |
| `"deterministic"` | deterministic (strict) | deterministic (strict) | deterministic (strict) |

*Strict* means an unsupported layer or plan raises `ValueError` with the reason (`backend.det_reason`)
before any kernel runs. `last_info["reduction"]` records the resolved mode and why, and
`last_info["grouped_fwd" | "grouped_dx"]["reduction"]` records `ordered_partials` with the workspace,
window, launch, row and contributor counts.

Executable example (verified to run, README "Deterministic execution"):

```python
import torch
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.supergroups import ConnectivityPlan

lay = SparsePoolingLayer(1024, 1_000_000, 1024, shared_neurons=8, pooling="mean", device="cuda",
                         connectivity=ConnectivityPlan(64, 16),
                         plan=PhysicalPlan(reduction="deterministic"))   # or "fast" / "auto"
x = torch.randn(1024, 1024, device="cuda").bfloat16()
y = lay(x); y.float().square().sum().backward()
print(lay.last_info["reduction"], lay.last_info["grouped_dx"]["reduction"])
```

Support matrix (`reduction="deterministic"`):

| Layer / plan | Behaviour |
|---|---|
| input-supergroup layers, packed Mode B C = 8 or C = 64, Mean(ReLU) / Mean(identity), split or BF16 gradients, feature-major input, B % 8 == 0 | supported: forward grouped (ordered partials) or generic (no atomics); dX grouped (forced); dW grouped or generic (exclusive owners) |
| grouped dX `auto` | warp-private copies; C = 8, S = 64 → `private_t32` (no fast-mode threshold applied); other S: the automatic tiles with 1 warp per copy |
| explicit `private_t32`, explicit private tiles / scatters | honoured |
| explicit `shared_t64`, `cuda_grouped_dx_reduction > 1` | `ValueError` (shared-copy CAS) |
| `cuda_grouped_dx="generic"`, layers without connectivity (`k_dx`), Triton backend | `ValueError` (FP32 atomics) |
| `cuda_grouped_backward="fused"` / `"fused_recompute"` | `ValueError` (atomic dX flush); `auto` resolves to separate kernels |
| other pooling, semantic hierarchies, token-major input, B % 8 ≠ 0 | `ValueError` |
| compact dW projection | supported (bitwise equal to padded) |
| non-finite upstream gradients | the same kernels' in-kernel generic epilogue (one path; finite elements bitwise repeatable; inf / NaN positions reproduced; NaN payload bits not promised) |
| workspace cap below one token tile | `RuntimeError` (raise the cap); never an atomic fallback |

### 2.2 Audit of reduction stages (fast mode)

Counted from the actual work lists (`benchmarks/ownership_map.py`, `ownership_map.json/.log`; H = 100M, G = D = 1024):

| Stage | Destination | Fast writers per destination | Arithmetic order (fast) | Kind |
|---|---|---|---|---|
| grouped forward | y[token, output] | 256 bucket items (one per supergroup), FP32 `atomicAdd` | arrival order | valid unordered sum |
| grouped dX, in CTA | slab copy [feature, token] | 1 warp (private plans) or 8 warps (`shared_t64`, 64-bit CAS) | CAS: arrival order | valid unordered sum (shared plan only) |
| grouped dX, across CTAs | dX[token, feature] | 12 / 24 / 48 chunks (S = 16 / 32 / 64), FP32 `atomicAdd` | arrival order | valid unordered sum |
| generic `k_dx`, fused backward, Triton | dX | many, CAS + atomics | arrival order | valid unordered sum |
| grouped / generic dW, db | main_grad of a unit | 1 CTA per unit (`every_unit_one_cta = True`) | 16-token tiles in token order, one read-modify-write | deterministic |
| FusedAdamW | parameter, moments, main_grad clear | 1 program per 2048 elements | loads → barrier → stores | deterministic (race fixed previous phase) |
| SR / RNG | stochastic rounding | counter-based: `tl.randint(seed, element offset)`, seed advanced once per step on the host | — | deterministic (SR-on training bitwise) |
| layout, tails, guards | padded tail elements, guard flags | written by their single owner kernel; guard is a host-cached flag per call | — | deterministic |

R > 1 really overlaps: every feature belongs to R supergroups. At H = 100M each supergroup has
3 dX chunks, so 3·R CTAs write each (feature, token): 12 / 24 / 48 (min = max = mean). The
ownership tests (`test_overlapping_writers_and_exact_coverage`) count these from the work lists at
test sizes as well.

### 2.3 Deterministic organization

* **Forward** (`sg_fwd` with `SgArgs.part`):
  * the CTA of chunk c, token tile t stores its per-item partial into `part[item, token − w0]`,
    with no atomics; items are exclusive to one chunk;
  * `sp_sg_reduce` then sums, for each output, its items in ascending item index (= supergroup
    order) from +0.0f, and writes y once.
  * Rows = 262,144 items at H = 100M; a full-M partial would be 8 GiB. At the 1 GiB default cap
    this is 8 windows of 1024 tokens (16 launches).
* **dX** (`sg_dx` / `sgx_dx`):
  * warp-private shared copies are summed in copy order 0..7;
  * the CTA stores row (chunk·S + local feature) for its token window, including zeros, so every
    read row is written;
  * `sp_sg_reduce` sums each feature's rows in ascending chunk index (supergroup, then item range)
    and writes dX once.
  * Rows = 12,288 / 24,576 / 49,152. Workspace 384 / 768 MiB (1 window), and 1020 MiB at S = 64
    (2 windows of 5440 tokens; full M would be 1.5 GiB).
* **Order lists**: CSR (offsets, records) built once per runtime from the static work lists
  (`SgExec.det_rows`); 1.0 MiB (forward) and 52 / 100 / 196 KiB (dX) at H = 100M.
* **Token windows**: windows are multiples of the token tile T, the largest that fit
  `deterministic_workspace_bytes` (default 1 GiB per stage).
  * Each destination's sum has the same contributors in the same order in every window, so the
    cap **does not change the bits**. Tested: 6 / 12 / 64 MiB caps and NaN-poisoned workspace in
    the unit tests; 4 MiB-cap training with 4–8 windows; and the H = 100M cap sweep, where 256 MiB
    gives 32 + 7 windows.
  * A cap below one tile raises.
* **Stream / sync**: everything runs on the current stream, with no device-wide synchronization
  and no host reads. The workspace is a per-call caching-allocator tensor, so no state survives
  between calls (tested with two instances, repeated calls, and main_grad accumulation).

## 3. Correctness gate

### 3.1 Layer level

`tests/test_deterministic.py` (37 tests):
* *Shapes:* 8 shapes × split / BF16, among them R = 1, several chunks per supergroup, token tails
  past 128 / 32, ragged H, and tiny / empty buckets.
* *Repeatability:* 10 identical calls are byte-for-byte equal (y, dX, dW, db, signed zeros).
* *Reference:* within the established tolerances (2e-5 split, 2e-2 BF16; the documented S64 ReLU
  tie uses `boundary_reference`).
* *Deterministic vs fast:* numerically close, and dW / db bitwise when the dX tiles are the same.
* *Other coverage:* overlapping writers and exact coverage, cap / poison, extremes (signed zero,
  subnormal, tiny, cancellation, boundary, inf, NaN), main_grad with two instances, API resolution,
  unsupported layers, and host determinism of the chunk tables.

Layer numerics, two processes (`benchmarks/det_numerics.py`; `det_numerics_p1/p2.{json,md}`):
| Case (D, H, G, S/R, B) | prec. | det ×10 / 2 processes | fast runs ≠ first | det−fast rel L2 y / dX / dW / db | det−fast max abs dX / RMS dX | det−FP64 rel dX / dW | fast−FP64 rel dX / dW | non-finite | near-zero ref dX (n / off) |
|---|---|---|---:|---|---|---|---|---:|---|
| 1024, 300,000, 64, 16/4, 200 | split | yes / yes | 9/9 | 1.2e-07 / 1.8e-08 / 0e+00 / 0e+00 | 1.9e-09 / 7.7e-11 | 2.3e-06 / 2.4e-06 | 2.3e-06 / 2.4e-06 | 0 | 0 / 0 |
| 1024, 300,000, 64, 16/4, 200 | bf16 | yes / yes | 9/9 | 1.2e-07 / 1.8e-08 / 0e+00 / 0e+00 | 1.9e-09 / 7.5e-11 | 1.6e-03 / 1.6e-03 | 1.6e-03 / 1.6e-03 | 0 | 0 / 0 |
| 1024, 300,000, 64, 32/8, 136 | split | yes / yes | 9/9 | 1.3e-07 / 3.2e-08 / 0e+00 / 0e+00 | 3.7e-09 / 1.4e-10 | 2.3e-06 / 2.3e-06 | 2.3e-06 / 2.3e-06 | 0 | 0 / 0 |
| 1024, 300,000, 64, 32/8, 136 | bf16 | yes / yes | 9/9 | 1.3e-07 / 4.4e-08 / 0e+00 / 0e+00 | 3.7e-09 / 1.9e-10 | 1.6e-03 / 1.6e-03 | 1.6e-03 / 1.6e-03 | 0 | 0 / 0 |
| 1024, 300,000, 64, 64/16, 256 | split | yes / yes | 9/9 | 1.1e-07 / 1.1e-07 / 0e+00 / 0e+00 | 5.6e-09 / 4.7e-10 | 2.3e-06 / 2.4e-06 | 2.3e-06 / 2.4e-06 | 0 | 0 / 0 |
| 1024, 300,000, 64, 64/16, 256 | bf16 | yes / yes | 9/9 | 1.1e-07 / 9.2e-08 / 0e+00 / 0e+00 | 3.7e-09 / 3.8e-10 | 1.6e-03 / 1.6e-03 | 1.6e-03 / 1.6e-03 | 0 | 0 / 0 |
| 256, 300,000, 16, 16/4, 72 | split | yes / yes | 9/9 | 1.0e-07 / 6.9e-08 / 0e+00 / 0e+00 | 5.6e-09 / 3.9e-10 | 2.5e-06 / 2.3e-06 | 2.5e-06 / 2.3e-06 | 0 | 0 / 0 |
| 256, 300,000, 16, 16/4, 72 | bf16 | yes / yes | 9/9 | 1.0e-07 / 6.9e-08 / 0e+00 / 0e+00 | 3.7e-09 / 3.9e-10 | 1.8e-03 / 1.7e-03 | 1.8e-03 / 1.7e-03 | 0 | 0 / 0 |
| 512, 200,003, 32, 32/8, 136 | split | yes / yes | 9/9 | 1.9e-07 / 5.2e-08 / 0e+00 / 0e+00 | 3.7e-09 / 2.4e-10 | 2.3e-06 / 2.4e-06 | 2.3e-06 / 2.4e-06 | 0 | 0 / 0 |
| 512, 200,003, 32, 32/8, 136 | bf16 | yes / yes | 9/9 | 1.9e-07 / 5.1e-08 / 0e+00 / 0e+00 | 3.7e-09 / 2.4e-10 | 1.7e-03 / 1.7e-03 | 1.7e-03 / 1.7e-03 | 0 | 0 / 0 |
| 256, 20,011, 8, 16/1, 64 | split | yes / yes | 9/9 | 8.4e-08 / 0.0e+00 / 0e+00 / 0e+00 | 0.0e+00 / 0.0e+00 | 2.4e-06 / 2.5e-06 | 2.4e-06 / 2.5e-06 | 0 | 0 / 0 |
| 256, 20,011, 8, 16/1, 64 | bf16 | yes / yes | 9/9 | 8.3e-08 / 0.0e+00 / 0e+00 / 0e+00 | 0.0e+00 / 0.0e+00 | 1.9e-03 / 1.7e-03 | 1.9e-03 / 1.7e-03 | 0 | 0 / 0 |

* *Repeatability:* every case is bitwise across 10 calls and across two process launches.
* *Fast mode* differs from its own first run in 9 of 9 repeats, even at R = 1, because the
  forward atomics span the supergroups touching an output.
* *dW / db* are bitwise equal in the two modes: same tiles, and dW does not depend on y.
* *Accuracy:* deterministic and fast results are equally close to the FP64 reference.
* *Near-zero and non-finite:* these random cases have no near-zero reference dX entries (< 1e-30)
  and no non-finite values. Signed zero, subnormal, cancellation, ReLU-boundary and inf / NaN
  cases are covered by `test_extremes` (§3.1).


### 3.2 Composed-model training (small / medium)

Model: trainable projection → BF16 → GranuleLayer (C8, Mean(ReLU)) → trainable head / tanh →
nonzero-target MSE, with changing pre-generated batches.

* *Comparison method:* each case runs in two processes. The first writes complete per-step records
  (y, loss, dX, dW, db before clearing, projection / head gradients, all parameters, moments,
  master weights, step and SR counters). The second restores the identical initial state and
  compares bitwise at every step.
* *Environment:* `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`
  (`reduction="auto"` → deterministic) and TF32 off, all set before library initialization.
* *Results* (`det_training_small_run.log`, `det_training/`): **all bitwise**.
  * S (small: H = 1M, D = G = 1024, M = 1000) and M (medium: H = 20M, D = 256, G = 64, M = 1024,
    Q = 64, 2 dX chunks per supergroup) cases; 16 steps each.
  * 3 seeds × S/R (16, 4), (32, 8), (64, 16): split.
  * SR on (seed 0) and BF16 gradients for each S/R; FP32 moments + master (M S64); BF16 + SR (M S16 seed 1).
  * 4 MiB-cap token windows (M S32 / S64; 4–8 windows).
  * After the dW change: compact dW at M S16 / S32 / S64 split, S64 BF16, and S16 SR.
* *Compact vs padded:* the compact trajectories' complete records equal the padded ones (561
  tensors each; `compact_vs_padded_training.md`).

### 3.3 Production H = 100M, M = 8192

4 changing-batch steps, SR off, deterministic, two processes from the same state
(`det_training_production_run.log`, `det_training/P_*`). Full-state comparison per step, streamed
to disk, outside any timing.

| Case | Steps × tensors | Bitwise | Peak alloc / reserved |
|---|---:|---|---|
| S16 R4 seed 0 split | 165 | yes | 9.81 / 10.65 GiB |
| S32 R8 seed 0 split | 165 | yes | 9.81 / 10.65 GiB |
| S64 R16 seed 0 split | 165 | yes | 10.00 / 10.65 GiB |
| S64 R16 seed 1 split | 165 | yes | 10.00 / 10.65 GiB |
| S64 R16 seed 0 BF16 | 165 | yes | 10.00 / 10.65 GiB |
| S64 R16 seed 0 split, compact dW | 165 | yes; also equal to the padded run's records | 10.00 / 10.65 GiB |

"Steps × tensors" = 33 tensors × (initial state + 4 steps) recorded by the first process. The 33
tensors are y, loss, dX, dW, db, W, b, the layer's moments / seed / step, and the projection and
head parameters, gradients and moments. The second process compares them in two restored
trajectories (297 comparisons per case). Peaks include the model, optimizer state and the
streaming comparison buffers.

### 3.4 Fast mode, characterized separately

Same restored state and batches at H = 100M, relative L2 per step (`fast_vs_det`: fast vs the
deterministic records; `fast_repeat`: fast vs a previous fast run):

| Case | Step | y | dX | dW | db | W (after step) | opt.m |
|---|---:|---:|---:|---:|---:|---:|---:|
| S16 fast vs det | 1 / 4 | 1.5e-6 / 3.5e-6 | 1.1e-4 / 2.9e-4 | 1.2e-4 / 3.8e-4 | 3.4e-5 / 1.1e-4 | 1.1e-8 / 2.6e-8 | 1.5e-4 / 4.7e-4 |
| S32 fast vs det | 1 / 4 | 1.2e-6 / 3.5e-6 | 1.0e-4 / 3.6e-4 | 9.8e-5 / 3.8e-4 | 2.7e-5 / 1.1e-4 | 9.1e-9 / 2.4e-8 | 1.2e-4 / 4.6e-4 |
| S64 fast vs det | 1 / 4 | 1.2e-6 / 3.8e-6 | 1.3e-4 / 4.5e-4 | 1.1e-4 / 4.2e-4 | 3.0e-5 / 1.2e-4 | 8.4e-9 / 2.4e-8 | 1.3e-4 / 5.1e-4 |
| S32 fast vs fast | 1 / 4 | 6.0e-8 / 2.0e-6 | 3.0e-5 / 2.1e-4 | 1.6e-7 / 2.2e-4 | 7.3e-8 / 6.1e-5 | 2.2e-9 / 1.1e-8 | 2.0e-5 / 2.4e-4 |
| S64 fast vs fast | 1 / 4 | 6.0e-8 / 2.0e-6 | 3.0e-5 / 2.5e-4 | 1.6e-7 / 2.3e-4 | 7.4e-8 / 6.3e-5 | 2.4e-11 / 1.0e-8 | 2.0e-5 / 2.5e-4 |

* *Fast vs fast:* fast mode is not repeatable from step 1, and the two-run spread grows over steps
  to the same order as the fast vs deterministic difference.
* *Interpretation:* these are different valid reduction orders. The dX relative difference looks
  large because many dX entries are tiny sums with cancellation; the losses agree within 1.1e-7 relative at step 4.
* *Not a correctness pass/fail criterion:* no tolerance was tuned to make this "pass". The four
  categories are reported separately:
  1. deterministic repeatability: passes, §3.2–3.3;
  2. agreement of valid orders: the table above and §3.1;
  3. fast run-to-run variation: the table above;
  4. actual defects: none found, §3.5.

### 3.5 Sanitizers

`benchmarks/run_sanitizer_det.sh` → `sanitizer_det.log`. memcheck, racecheck, initcheck and
synccheck over 24 targeted cases: S = 16 / 32 / 64; split / BF16; token tails (B = 72 / 136 / 264);
multiple windows (3 MiB cap); compact dW; the feature-owner scatter; non-finite gradients.
Result: **0 errors / 0 hazards** in every tool.

## 4. Deterministic-mode cost

### 4.1 Production (H = 100M, M = 8192), paired, same session

Per-phase medians (ms); `final_100M_summary.md`; seed 0 files, padded dW:

| S / R | mode | dX plan | fwd total (stage 1 / reduce) | dX (stage 1 / reduce) | dW | optimizer | step |
|---|---|---|---|---|---:|---:|---:|
| 16 / 4 | fast | tiles T64 / 1 warp per copy | 312.5 | 1047.9 | 1046.8 | 21.4 | 2429.6 |
| 16 / 4 | det | same | 321.3 (310.6 / 10.0) | 1049.3 (1048.8 / 0.5) | 1045.7 | 21.4 | 2438.2 |
| 32 / 8 | fast | tiles T32 / 1 | 316.4 | 1251.8 | 1081.8 | 21.4 | 2673.0 |
| 32 / 8 | det | same | 323.7 (313.2 / 10.0) | 1251.9 (1250.9 / 1.0) | 1081.4 | 21.4 | 2678.5 |
| 64 / 16 | fast | private_t32 | 319.1 | 1596.7 | 1112.9 | 21.4 | 3052.4 |
| 64 / 16 | det | private_t32 | 326.5 (316.0 / 10.0) | 1595.0 (1593.1 / 1.9) | 1112.8 | 21.4 | 3058.0 |
| 64 / 16 BF16 | fast | private_t32 | 321.5 | 1478.9 | 1027.8 | 21.4 | 2850.4 |
| 64 / 16 BF16 | det | private_t32 | 328.8 (318.2 / 10.0) | 1478.9 (1476.9 / 1.9) | 1027.1 | 21.4 | 2856.5 |

Paired step differences det − fast:
* +8.14 ms (+0.34 % [−0.08, +2.37]); one round of this file had a +57 ms outlier in both modes;
* +5.64 ms (+0.21 % [+0.16, +0.32]);
* +6.16 ms (+0.20 % [+0.10, +0.25]);
* BF16: +7.14 ms (+0.25 % [+0.18, +0.31]).

With the compact dW: +0.64 / +0.30 / +0.27 / +0.22 %. Fresh-process single runs
(`mem_*`, for memory): 2413.7 → 2424.0, 2680.0 → 2688.9, 3064.3 → 3071.1 ms.

Composed model (projection + layer + head, S64; `C8_S64_R16_composed`): 3058.3 → 3066.6 ms,
**+0.26 %** [+0.24, +0.40].

* *Launches:* fast runs 1 forward + 1 dX launch. Deterministic runs 16 forward launches (8 windows
  × stage 1 + reduce) and 2 / 2 / 4 dX launches.
* *Traffic (computed):*
  * the forward partials are written once and read once: 8 GiB each way. The 10 ms reduce reads
    8 GiB, about 860 GB/s, close to the 936 GB/s peak;
  * stage 1 itself is not slower than the atomic forward (310–318 vs 312–321 ms);
  * the dX partials are 0.375 / 0.75 / 1.5 GiB each way.

### 4.2 Memory

Fresh process per mode (`final_100M/mem_*`), steady state including warm-up:

| S / R | fast alloc / reserved | det alloc / reserved | det workspace (fwd + dX) |
|---|---|---|---|
| 16 / 4 | 8.84 / 10.38 GiB | 9.66 / 10.38 GiB | 1024 + 384 MiB |
| 32 / 8 | 8.84 / 10.38 | 9.66 / 10.38 | 1024 + 768 MiB |
| 64 / 16 | 8.84 / 10.38 | 9.84 / 10.38 | 1024 + 1020 MiB |

* *Peaks:* the forward and dX workspaces are not live at the same time, so the peak grows by about
  the larger one plus allocator rounding.
* *Reserved memory is unchanged.* The caching allocator's pool, sized by the snapshot / restore and
  optimizer buffers, already covers the workspace.
* *Cap sweep (S64, paired):*

  | Cap | Windows (fwd + dX) | Step vs 1 GiB | Peak alloc |
  |---|---|---|---|
  | 256 MiB | 32 + 7 | +0.09 % | 9.09 GiB |
  | 4 GiB | 2 + 1 | −0.04 % | 12.66 GiB |

  1 GiB is a reasonable default.

### 4.3 Medium screen (H = 1M / 10M / 30M, M = 1024 / 8192; `screen_summary.md`)

Deterministic − fast (padded dW), paired step:

| | S16 R4 | S32 R8 | S64 R16 |
|---|---|---|---|
| H 1M, M 1024 | +3.2 % | +6.9 % | +23.8 % |
| H 1M, M 8192 | +3.2 % | +4.1 % | +22.0 % |
| H 10M, M 1024 | +4.3 % | +2.3 % | +11.9 % |
| H 10M, M 8192 | +2.6 % | +2.6 % | +11.1 % |
| H 30M, M 1024 | +1.4 % | +0.9 % | +0.6 % |
| H 30M, M 8192 | +1.1 % | +0.9 % | −0.2 % |

Further cases:
* ragged H 23,456,789 S32: +1.25 %;
* G 512, S64 R8: +0.11 %;
* H 30M S64 BF16: +0.93 %.

*Why S64 at low occupancy costs so much:* fast mode uses the `shared_t64` plan there, because the
nonempty mean is below the rule's thresholds. Deterministic mode cannot share copies (CAS order),
so it uses `private_t32`, which the previous phase measured to be slower at those shapes. The
reduce kernels are a few % at small H, where fixed costs dominate.

### 4.4 C64 controls (H = 100M)

| Case | fast (auto) | det | Δ step | Why |
|---|---|---|---|---|
| C64 S16 R4 | 1696.9 ms, dX tiles T64, 8 warps per copy | 1650.0 ms, T64, 1 warp per copy | **−2.8 %** (dX −7.9 %) | the fast C64 auto tile shares copies (CAS); det uses private copies |
| C64 S64 R16 | 1703.5 ms, T64 / 8 warps per copy | 1923.0 ms, T32 / 1 warp per copy | **+12.8 %** (dX +30 %) | private copies at T64 need 138 KiB of shared memory > 99 KiB, so det uses T32 |

The C64 S16 result suggests the fast C64 dX auto tile may be suboptimal at H = 100M. That is a
fast-mode dispatch question outside this phase, recorded as a lead; **no default was changed**,
since a change would need its own held-outs.

### 4.5 Fast mode vs the starting code

Separate processes, same session, unpaired; `prev_*` built from `c9a591e`:

| Case | c9a591e step | this branch, fast | Δ |
|---|---:|---:|---|
| C8 S16 | 2471.1 (rounds 2425–2505, noisy) | 2425.4–2429.6 (3 seeds) | noise |
| C8 S32 | 2672.1 | 2671.0–2673.0 | ≈ 0 |
| C8 S64 | 3049.5 | 3050.0–3052.4 | +0.02 to +0.1 % |
| C64 S16 | 1690.9 | 1696.9 | +0.35 % |
| C64 S64 | 1699.7 | 1703.5 | +0.22 % |

* *C8:* fast-mode kernel registers are unchanged (forward 73, dX Plan A 80, Plan B 78,
  S16 tile 86, dW 127).
* *C64:* the small unpaired difference is not attributed. Its forward is 265.0 vs 262.9 ms; the
  new `part` branch in the shared forward body is a candidate, and it is within cross-process
  variation of earlier phases.

## 5. Track B: canonical bucket statistics

### 5.1 Definitions

`supergroups.bucket_statistics(layout, buckets, Q, S, plan)`, schema `granule.bucket_stats/2`. It
validates the bucket table first: duplicate records, negative counts or out-of-range indices raise;
it never merges records.

| Field | Definition |
|---|---|
| `total_work_units` U | Σ positive bucket unit counts. A unit = `unit_blocks` C-blocks: 4 × 8 neurons at C = 8 (the fragment-packed parameter quantum), from the layout, not H / constant |
| `total_possible_buckets` | G · Q |
| `nonempty_buckets` Nnz, `empty_buckets` | Nnz, G·Q − Nnz (`zero_unit_records` counted separately) |
| `bucket_occupancy` | Nnz / (G·Q) |
| `mean_units_per_nonempty_bucket` | U / Nnz (None when Nnz = 0; the dispatch rules read 0.0) |
| `mean_units_per_possible_bucket` | U / (G·Q) |
| `bucket_units_min/max/quantiles` | nearest rank, q_p = x_(max(1, ⌈p n⌉)), p ∈ {0, .1, .5, .9, .99, 1} |
| chunk metrics (`chunk_statistics`) | chunk count, units per chunk (limit, min / max / mean / quantiles): distinct from bucket metrics |
| provenance | valid neurons, padded capacity, C, packing, S, R, Q, D, G, graph seed / generator / version |

Asserted for nonempty graphs: possible mean = occupancy × nonempty mean, and nonempty mean ≥ 1.
Compatibility alias: `units_per_bucket_mean` in `item_chunk_stats` = the nonempty mean, as before,
documented in the docstring.

Tests (`tests/test_bucket_stats.py`, 21):
* hand-built tables (empty + nonempty, all nonempty, zero records, zero work, malformed);
* the quantile convention;
* generated C8 / C64 graphs (ragged H, sparse occupancy);
* the historical H = 1M values;
* matched C64 → C8 conversion;
* chunk metrics distinct from bucket metrics;
* the dispatch reads the nonempty mean, including a graph where the possible and nonempty means
  straddle the 4.0 threshold, so a denominator regression flips the plan;
* runtime / benchmark fields are produced by the canonical function.

### 5.2 Old vs corrected metric definitions

| Old field / text | Where | What it actually was | Corrected field |
|---|---|---|---|
| "units/bucket" column in the S64 crossover tables, H = 1M rows (0.12) | SCALE_DISPATCH_FUSED_BACKWARD.md lines 296 (split) and 328 (BF16) | U / (G·Q), the possible-bucket mean | `mean_units_per_possible_bucket` = 0.1211; nonempty mean = 1.000; occupancy = 0.1211 |
| the same column, all other rows | lines 297–312, 329–332, 384–396 | U / Nnz (nonempty mean), 2-decimal truncation in 5 rows | `mean_units_per_nonempty_bucket` |
| `units_per_bucket_mean` (runtime info) | `item_chunk_stats`, logged JSON | U / Nnz | kept as alias of `mean_units_per_nonempty_bucket` |

Reconstruction (`benchmarks/audit_bucket_stats.py`, `historical_bucket_audit.{md,json,csv}`):
* the graphs were rebuilt with the recorded generator (balanced_permutation v1), seed, packing and
  shape;
* the reconstructed values were checked against the values logged in the original JSON: 50 files,
  0 mismatches;
* the original logs and JSON are untouched. The audit is a derived table labelled with its
  provenance; the freshly computed fields are not presented as logged.

### 5.3 Dispatch

`scale_rule` consumed the nonempty mean before and after: thresholds 4.0 (split) and 2.0 (BF16)
→ `private_t32`, unchanged.

* *H = 1M:* nonempty mean 1.0 and possible mean 0.12 are both below both thresholds → `shared_t64`
  either way.
* *Threshold-near* (H = 30M BF16: 3.58 → `private_t32`; H = 15M BF16: 1.79 → `shared_t64`) and
  *dense* (H = 100M: 11.92) cases have occupancy 1, where the two means coincide.
* *All audited rows:* "Rule (nonempty)" = "Rule (possible)".

**No dispatch decision changed.** The dispatch reason strings now name the nonempty mean, the
possible mean and the occupancy. In deterministic mode the fast thresholds are not applied (§2.1).

## 6. Track C: standalone C = 8 dW compact projection

### 6.1 Implementation

`dw_block_c8_compact<SPLIT, MASKED>` (`kernels/cuda/sparse_pool_bwd.cuh`), selected by
`PhysicalPlan.cuda_grouped_dw_projection = "compact"`; the old padded path is unchanged and is the
default. It applies to the grouped C = 8 Mean(ReLU) dW with the fast epilogue. For an 8-neuron ×
16-token block:

1. **Projection.** One `mma.m16n8k8` in dX orientation: A = the token × K fragment from the same
   `ldmatrix.trans` registers, B = the W fragment, C = the bias pair. This replaces the two padded
   neuron-row MMAs.
2. **Mask.** `tile_mask` uses `set.gt.ftz` (the production compare), then `pair_mask`, then one
   `movmatrix.trans` per token half into the dW orientation.
3. **Gradient.** The mask epilogue builds operands `hp & mq` and `lp & mq` (hi / lo split rows
   unchanged), and db accumulates `one & mask`. The m16n8k16 gradient MMA, the token order, the
   exclusive owners and the write-back are untouched.

Dispatch metadata: `grouped_dw.projection / projection_reason / kernel / projection_fallback`.
Other C, other activations and the generic epilogue keep padded; an explicit compact request there
raises.

Static, production configuration (cfg 2: 8 warps × 2 units, TS 4, ST 2), `dw_projection_sass.md`:

| Kernel | regs | stack / spills | HMMA.1688 | HMMA.16816 | MOVM | FSET(P).GT.FTZ | non-FTZ GT | PRMT | total instr. |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| split, padded | 127 | 0 / 0 | 32 | 16 | 0 | 64 | 0 | 93 | 2648 |
| split, compact | 128 | 0 / 0 | 16 | 16 | 32 | 64 | 0 | 69 | 2656 |
| BF16, padded | 127 | 0 / 0 | 32 | 16 | 0 | 64 | 0 | 29 | 2504 |
| BF16, compact | 128 | 0 / 0 | 16 | 16 | 32 | 64 | 0 | 61 | 2624 |

* *MMA and MOVM counts (static, all unrolled copies):* the projection MMAs are halved as intended,
  with 2 MOVM per block-tile added.
* *FTZ compares:* the production compares keep `.FTZ` (the previous phase's ptxas hazard is absent).
* *Instruction totals* are about equal.
* *Spills:* the unused BF16 cfg 0 compact instantiation spills 32 B (8 STL / 8 LDL). Auto never
  selects it.

Computed FLOPs at H = 100M, M = 8192, using 6.25e9 block-tiles (12.5M blocks × 512 token tiles):
* *Projection:* padded issues 2 × 2048 FLOP per block-tile (25.6 TFLOP), of which half is useful;
  compact issues 2048 FLOP, all useful (12.8 TFLOP) — a saving of 12.8 TFLOP and 6.25 G HMMA.1688.
* *Gradient MMA* (4096 FLOP per block-tile) is unchanged. All 16 rows are useful in split mode
  (rows 8..15 carry the low correction); in BF16 rows 8..15 are zero.

### 6.2 Numerics

`tests/test_dw_projection.py` (72): **bitwise dW / db equality to padded** across:
* shapes S16 / S32 / S64, ragged H, tails, split / BF16, all four dW configurations;
* extremes: random, signed zero, subnormal products and biases, cancellation, near-zero z, the
  documented ReLU tie, inf / NaN;
* reference accuracy, and pre-existing main_grad accumulation;
* selection and error behaviour;
* the mask-epilogue control (padded projection + mask epilogue = fast epilogue, bitwise).

The non-finite guard uses the generic epilogue, which has no compact variant, so exactly one path
contributes. Training trajectories are identical to padded (§3.2, §3.3).

### 6.3 Timings: old (padded) vs new (compact) dW within each mode

H = 100M, M = 8192, paired (`final_100M_summary.md`); medians [min, max] over the 4 rounds:

| S / R, seed | mode | dW padded → compact | Δ dW | Δ step |
|---|---|---|---|---|
| 16 / 4, s0 | fast | 1046.8 → 1078.4 | +3.01 % [+2.48, +3.10] | +1.22 % [+0.82, +1.27] |
| 16 / 4, s1 | fast | 1045.2 → 1077.7 | +3.14 % | +1.32 % |
| 16 / 4, s2 | fast | 1044.5 → 1077.2 | +3.13 % | +1.26 % |
| 16 / 4, s0 | det | 1045.7 → 1088.4 | +3.36 % | +1.34 % |
| 32 / 8, s0 | fast | 1081.8 → 1121.4 | +3.65 % | +1.41 % |
| 32 / 8, s1 | fast | 1081.0 → 1121.1 | +3.76 % | +1.45 % |
| 32 / 8, s2 | fast | 1081.5 → 1121.2 | +3.62 % | +1.46 % |
| 32 / 8, s0 | det | 1081.4 → 1121.8 | +3.75 % | +1.48 % |
| 64 / 16, s0 | fast | 1112.9 → 1157.4 | +4.03 % | +1.40 % |
| 64 / 16, s1 | fast | 1112.7 → 1158.7 | +4.10 % | +1.46 % |
| 64 / 16, s2 | fast | 1112.4 → 1156.9 | +4.03 % | +1.46 % |
| 64 / 16, s0 | det | 1112.8 → 1157.8 | +4.08 % | +1.48 % |
| 64 / 16 BF16, s0 | fast | 1027.8 → 1114.1 | +8.48 % | +3.00 % |
| 64 / 16 BF16, s0 | det | 1027.1 → 1114.3 | +8.45 % | +2.97 % |

* *Every round is slower.* The medium screen agrees (+1.2 to +3.3 % step at M = 8192, every shape).
* *Deterministic mode does not change the dW comparison.*
* *Decision:* this is a fully measured **negative result**. The padded default is kept;
  compact stays explicit / experimental.

### 6.4 Attribution

H = 100M, M = 8192, fast reduction, 4 rounds × 2 restored repetitions × 3 steps
(`attribution/`, `attribution_summary.md`; dev commit 9ca57de). Variants, all bitwise equal except
the ablation:

* `padded`: the default;
* `padded_mask`: the padded projection with the mask-operand epilogue, i.e. the compact path's
  gradient operands without its projection or transposes;
* `compact`;
* `compact_nomovm`: a separate `SP_ABLATE_DWC=1` build, compact without the two `movmatrix`
  transposes. Its dW is wrong; it is for timing only.

Paired dW-kernel differences, median [min, max] over rounds:

| Case | padded_mask − padded | compact − padded_mask | compact − padded | compact_nomovm − padded (ablation run) |
|---|---|---|---|---|
| S16 R4 split | +1.46 % [+0.43, +2.08] | +1.59 % [+1.38, +1.81] | +3.07 % [+2.25, +3.50] | −1.84 % [−3.97, −1.78] |
| S64 R16 split | +1.49 % [+1.27, +1.58] | +2.61 % [+2.47, +2.76] | +4.14 % [+3.77, +4.38] | −1.27 % [−1.30, −1.24] |
| S64 R16 BF16 | +6.03 % [+6.01, +6.13] | +2.05 % [+1.98, +2.16] | +8.24 % [+8.17, +8.30] | +1.38 % [+1.37, +1.49] |

Complete step, compact − padded: +1.21 / +1.46 / +2.92 %. The ablation's step is −0.75 / −0.52 /
+0.49 %.

**Interpretation** (combining files from different processes, so about ±1 %; the padded baselines
of the two files of one case differ by up to 1.3 %):

* **The removed MMA is worth about 3 % of dW** in split mode. Estimate: ablation − mask epilogue
  ≈ −3.3 % (S16) and −2.8 % (S64), and about −4.7 % in BF16.
* **The two transposes cost about 5 % (split) and 7 % (BF16)**, so the net is a loss.
* **The mask-operand epilogue itself is not free.** It costs +1.5 % (split) and +6 % (BF16) against
  the fast epilogue, whose padded projection writes the masks directly in dW orientation. The BF16
  fast epilogue is especially cheap (PRMT 29 vs 61 in the static counts), which is why BF16 loses
  most.
* **Why the instruction totals hide this.** The compact path chains the projection MMA, then the
  compare, then MOVM, then the operand AND, then the gradient MMA within one block-tile. The padded
  path issues two independent projection MMAs whose outputs feed the masks directly. Static totals
  are equal (2648 vs 2656), so the cost is latency / dependency, not instruction count. This is an
  interpretation; there are no counters to confirm it.
* **What a compact variant would need to win:** a mask transform cheaper than two MOVM per
  block-tile (for example, a projection layout whose masks need no transpose) together with the
  fast epilogue's operand construction. Not attempted, to stay within the one-factor scope.


## 7. Defaults, options, limitations

* **Defaults changed:** none.
  * `reduction="auto"` → fast unless the global deterministic flag is set.
  * The fast S64 rule and thresholds are unchanged.
  * dW projection auto → padded.
  * FusedAdamW barrier fix kept.
* **New options:**
  * `reduction="deterministic"` (production-ready, ≈ +0.2–0.34 % per step at H = 100M);
  * `deterministic_workspace_bytes`;
  * `cuda_grouped_dw_projection="compact"` (experimental, correct, slower);
  * `cuda_dw_epilogue="mask"` on the grouped C = 8 dW (attribution control).
* **Unsupported in deterministic mode** (all raise): fused backward, generic / Triton dX,
  shared-copy dX plans, non-Mean pooling, token-major input, B % 8 ≠ 0.
* **Limitations:**
  * repeatability is claimed only for a fixed environment and resolved plan;
  * NaN payload bits are not promised;
  * deterministic mode costs +11–24 % at S64 low occupancy (H ≤ 10M) and +12.8 % for C64 S64;
  * fast C64 S16 may have a better dX tile (a lead, not tested on held-outs).

## 8. Reproduction

All commands from `sparse_pooling/benchmarks/` with the `env_isaaclab` Python; GPU otherwise idle
for timings.

```bash
P=/home/evga/.conda/envs/env_isaaclab/bin/python; R=../results/determinism_metrics_dw_projection
# statistics, historical audit, ownership map
$P -m pytest -q ../tests/test_bucket_stats.py ../tests/test_deterministic.py ../tests/test_dw_projection.py
$P audit_bucket_stats.py --out $R/historical_bucket_audit
$P ownership_map.py --out $R/ownership_map
# layer numerics, two processes
$P det_numerics.py --records /tmp/numrec --role write   --out $R/det_numerics_p1
$P det_numerics.py --records /tmp/numrec --role compare --out $R/det_numerics_p2
# composed training (write + compare processes per case)
bash run_det_training.sh $P /tmp/dt small; bash run_det_training.sh $P /tmp/dt production
bash run_det_training.sh $P /tmp/dt windows; bash run_det_training.sh $P /tmp/dt after_dw
$P summarize_det_training.py $R/det_training/*.json --md $R/det_training_summary.md
# sanitizers (not concurrently with timings)
bash run_sanitizer_det.sh $P
# dW static evidence, screen, production timings, attribution
$P dw_projection_sass.py --out $R/dw_projection_sass
bash run_dm_screen.sh $P; bash run_dm_final.sh $P; bash run_dm_attribution.sh $P
$P summarize_dm.py $R/final_100M/*.json --csv $R/final_100M.csv --md $R/final_100M_summary.md
# full suite
$P -m pytest -q ../tests
```

`run_dm_final.sh` expects the starting commit built in `../../GranuleLayer-sdfb` (`prev_*` runs).
Every bench JSON records the commit, dirty flag, NVCC flags, the loaded library path, GPU clocks /
temperature / power before and after, seeds and the resolved kernels.

Test summary: full suite once on the final code (b262934): **2748 passed, 31 skipped** in 141 s (`full_suite.log`); the previous phase recorded 2618 / 31, and the difference is the 130 new tests (test_deterministic 37, test_bucket_stats 21, test_dw_projection 72).
