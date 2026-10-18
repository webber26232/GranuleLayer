# Input-supergroup connectivity and grouped CUDA kernels (RTX 3090)

Branch `input-supergroup-optimization`, created from `backward-compute-optimization` at
`fcc363b47fcac6cd9692ee1480810d81590ea4ff` (the MMA-register / forward-epilogue branch was not used).
Every number below is a CUDA-event measurement on this GPU unless labelled *static* (SASS counts),
*computed* (residency from compiler resources) or *ablation* (timing-only builds with wrong results).
Hardware counters are unavailable on this machine (admin-only), so nothing here is an Nsight counter.
Raw data: `results/input_supergroups/` (JSON per run, CSV summaries, logs).

## 1. Main result and selected plans

Primary configuration: D = G = 1024, H = 100,000,000 valid neurons, K = 8, M = 8192, BF16
parameters/inputs, FP32 MMA accumulation and `main_grad`, split-precision gradients, Mean(ReLU),
fused AdamW with BF16 moments.  Same graph, inputs and initial parameters within each row group;
4 interleaved rounds (alternating order), 2 reps × 2 steps per round; ranges are per-round medians.

| C | Graph policy | S | Q / R | Backend | Fwd total | dX total | dW total | Step | Peak VRAM¹ | Extra scratch² |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 8 | Unrestricted | 1024 | control | Existing | 387.7 | 2319.6 | 2065.6 | **4794.6** | 10.46 GiB | — |
| 8 | Constrained | 16 | 256 / 4 | Existing | 387.8 | 2369.2 | 2067.2 | **4846.6** (4827–4872) | 10.66 GiB | — |
| 8 | Same constrained graph | 16 | 256 / 4 | Grouped fwd + dX + dW | 310.9 | 1337.3 | 1045.7 | **2716.9** (2704–2730) | 10.71 GiB | +32 MiB |
| 64 | Unrestricted | 1024 | control | Existing | 300.8 | 711.1 | 729.7 | **1764.1** | 10.32 GiB | — |
| 64 | Constrained | 16 | 256 / 4 | Existing | 301.4 | 715.3 | 724.1 | **1763.3** (1753–1771) | 10.33 GiB | — |
| 64 | Same constrained graph | 16 | 256 / 4 | Grouped fwd + dX + dW | 266.7 | 704.8 | 708.3 | **1703.1** (1702–1706) | 10.37 GiB | +35 MiB |

¹ Includes the benchmark's 1.69 GiB parameter snapshot (used to restore identical parameters per
variant) and, for constrained graphs, a 190.8 / 23.9 MiB (C = 8 / 64) validation index copy the
harness kept alive (§10); `bench_kernels.py` on the unrestricted layer peaks at 8.64 / 8.79 GiB.
² Measured per-step peak minus persistent allocations (generic 208 MiB → grouped 240–243 MiB): the
FP32 coefficient transpose `c1T [G, M]` (32 MiB).  Budget target was ≤ 64 MiB.

**Three seeds** (graph = input = init seed ∈ {0, 1, 2}, 4 rounds each):

| C | S / R | Generic step (ms) | Grouped step (ms) | Change |
|---:|---|---|---|---:|
| 8 | 16 / 4 | 4846.6 / 4841.0 / 4839.0 | 2716.9 / 2711.9 / 2710.1 | **−43.9 %** (−43.9…−44.0) |
| 64 | 16 / 4 | 1763.3 / 1751.4 / 1753.6 | 1703.1 / 1693.0 / 1693.8 | **−3.4 %** (−3.4…−3.4) |

Conclusions (measured; model quality was not studied):

* **C = 8**: a bounded input working set makes the complete training step 1.78× faster on the same
  graph (4847 → 2717 ms), and 1.76× faster than the unrestricted architecture with the existing
  backend (4795 ms).  Every phase gains: forward −20 %, dX −44 %, dW −49 %.
* **C = 64**: −3.4 % (forward −12 %, dX −1.5 %, dW −2.2 %).  The C = 64 backward was already within
  ~1.4× of the measured MMA rate, so there is little left for scheduling to remove.
* **Locality alone does nothing**: the constrained graph through the existing backend runs exactly
  as fast as the unrestricted graph (C = 64: 1763 vs 1764 ms; C = 8: 4847 vs 4795 ms, i.e. +1 %).  All
  of the gain comes from kernels that exploit the bounded set (larger token tiles, compact slabs,
  contention-free accumulation).
* **C = 8 vs C = 64 gap**: 2.72× with the existing implementation (4795 / 1764) → **1.60×** with the
  best plans (2711 / 1693).  The matched-connectivity control (§7) shows the remaining 1.60× is
  entirely the C = 8 packing / per-block cost: the same C = 64 graph run through C = 8 blocks takes
  2705 ms, identical to the native C = 8 graph (2706–2717 ms).
* **Best S**: C = 8: S = 16 (R = 4) — but at S = 16 only C(16, 8) = 12,870 records exist per set and
  the realized graph repeats records (3.22M distinct for 12.5M blocks, §4); the best S without record
  repetition is S = 32 (−37.8 %, 3001 ms).  C = 64: S ≤ 64 are equivalent within ±1 %.

Selected defaults (`PhysicalPlan.cuda_input_supergroup_backend="auto"`, only for layers built with a
`ConnectivityPlan`; unrestricted connectivity stays the architectural default):

| C | S | Forward | dX | dW |
|---:|---|---|---|---|
| 8 | 8 / 16 / 32 | grouped T = 128, 8 warps | grouped, warp-private FP32 copies, T = 128·16w / 64·8w / 32·8w | grouped cfg 2 (8 warps × 2 units, 4×16-token stages, 2 CTAs/SM) |
| 8 | 64 / 128 | grouped T = 128, 8 warps | grouped, one CAS slab per 8 warps, T = 64 / 32 | grouped cfg 2 |
| 8 | 256 | grouped T = 64 | existing | grouped cfg 0 |
| 8 | ≥ 512 | existing | existing | existing |
| 64 | ≤ 64 | grouped T = 128, 4 warps (8 at S = 128) | grouped T = 64 (split, buckets ≥ 3 units) else existing | grouped cfg 2 if S ≤ 16 and buckets ≥ 3 units, else existing |
| 64 | 256 | grouped T = 64 | existing | existing |

(`kernels/cuda_ops.py: SG_AUTO`, `SG_AUTO_C64_LARGE_BUCKETS`).  Hybrid plans are intentional: a phase
keeps the existing kernel wherever the grouped one was not measured faster.

## 2. Baseline, environment, experiment families

| Item | Value |
|---|---|
| GPU | GeForce RTX 3090 (sm_86, 82 SMs, 24 GiB, 370 W cap), driver 550.67 |
| Toolchain | nvcc 12.1 `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17`; PyTorch 2.7.0+cu128, Triton 3.3.0 |
| GPU state during runs | SM 1650–1770 MHz, memory 9501 MHz, 52–55 °C, 338–349 W (power-limited) |
| Commits | implementation `03bbed7`; H = 100M finalist / seed runs used `03bbed7` plus the C = 64 auto-rule edit committed in `5870bd9` (JSON `dirty: true`); exact build hash in every JSON (`library`) |

P0 re-measurement of the starting branch (`bench_kernels.py`, unrestricted): C = 64 step 1811.6 ms
(fwd 304, dX 736, dW 748), C = 8 4895.7 ms (fwd 394, dX 2369, dW 2115) — consistent with the
historical 1.8 / 4.8 s.  The tables above use the same-session `bench_supergroups.py` runs instead
(one harness for all variants).

Experiment families (graph seed 0 unless stated):

* **Family A** (fixed Q = 256): S ∈ {16, 32, 64, 128, 256} with R = S / 4, plus the S = 8 diagnostic
  (R = 2) and the S = 1024 single-set control (Q = R = 1).
* **Family B** (fixed R = 16): S ∈ {16, 32, 64, 128} → Q ∈ {1024, 512, 256, 128}.
* **R probe**: S = 16 with R ∈ {8, 32}; S = 64 with R ∈ {8, 32}.
* Scale: tests at small/ragged sizes; screens at H = 1M / M = 1024 and H = 10M / M = 8192 (one seed,
  3 reps × 3 steps); finalists at H = 100M / M = 8192 (4 rounds); small M (256, 1024) at H = 100M.

## 3. Logical graph and physical layout

| Symbol | Meaning | Here |
|---|---|---|
| D, G, H, K, M | inputs, pooling outputs, valid neurons, fan-in, tokens | 1024, 1024, 1e8, 8, 8192 |
| C | neurons sharing one index record (Mode B) | 8, 64 |
| S | features per input supergroup | 8 … 1024 |
| R | partition rounds | 1 … 64 |
| Q = R·D/S | supergroups | 1 … 2048 |
| T | tokens per CTA tile (per phase) | 16 … 128 |
| unit | packed parameter quantum: UB blocks | C = 8: 4 blocks = 32 neurons; C = 64: 1 block = 64 neurons |

```
logical:   feat[Q, S]  (R independent seeded permutations of 0..D-1, each cut into D/S sets)
           neuron j:  output p(j) (unchanged), supergroup q(j), record of 8 distinct inputs ⊂ feat[q]
physical:  unchanged packed parameters [G, NB, ...] with GLOBAL int16 indices (the generic kernels
           run the identical tensors);  bucket (q, g) = contiguous unit range inside output g
compact:   buckets [n, 4] int32 (q, g, first unit, unit count), sorted by q        ~4 MiB
work lists (derived, cached): items = buckets, chunks (q, lo, hi), q-ordered unit list   14–21 MiB
```

Pooling outputs and supergroups are independent axes: every output draws units from all 256
supergroups (all 4–64 rounds) at S ≤ 256, Q = 256, and its Mean is the sum over all its buckets
divided by its exact valid count n_g (never a mean of bucket means).  No neuron permutation map is
stored: buckets are defined as contiguous unit ranges, so semantic ids stay offset_g + slot.
Weights and biases are drawn exactly as for the unrestricted layer with the same seed (only indices
differ; tested).  Checkpoints carry `sg_feat`, `sg_buckets` and `sg_plan` (JSON with generation
version); loading into a mismatched connectivity raises; unrestricted checkpoints are unaffected.

## 4. Generator: balance, coverage, repetition, metadata

`supergroups.generate_indices` (seeded, offline, GPU): feature sets from independent `randperm(D)`
per round; output g's valid units get quotas ⌊n/Q⌋ + 1 on a rotating window of an interleaved
supergroup order (position p → round p mod R), the partial tail unit of each output is placed
greedily on the supergroup with the fewest tails; every supergroup consumes one stream of local
records cut from successive `randperm(S).reshape(S/8, 8)` ("coordinated local rounds"), sorted
ascending (sorted local and global orders coincide, so weight/index pairing and MMA order match the
generic kernels).  `supergroups.validate` checks (raises on violation): exactly H valid neurons and
unchanged per-output counts, 8 distinct in-range indices per record, every index in its set, S
distinct features per set, every feature in exactly R sets, buckets cover each output's valid units
exactly once.  H = 100M results (`generator_stats_H100M.json`):

| C | S / R | Q | Buckets | Units/bucket | Distinct records | Blocks per distinct | Set pair coverage (expected) | Feature-degree CV | Gen. time |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | unrestricted | — | — | — | 12,500,224 | 1.00 | 1 | 3.2e-3 | — |
| 8 | 8 / 2 (diag.) | 256 | 262,144 | 11.9 | **256** | 48,829 | 0.0136 (0.0136) | 0 | 0.18 s |
| 8 | 16 / 4 | 256 | 262,144 | 11.9 | **3,220,700** | **3.88** | 0.0574 (0.0574) | 2.0e-5 | 0.17 s |
| 8 | 32 / 8 | 256 | 262,144 | 11.9 | 12,471,457 | 1.00 | 0.2182 (0.2182) | 2.4e-5 | 0.17 s |
| 8 | 64 / 16 | 256 | 262,144 | 11.9 | 12,500,159 | 1.00 | 0.6388 (0.6383) | 2.7e-5 | 0.18 s |
| 8 | 128 / 32 | 256 | 262,144 | 11.9 | 12,500,224 | 1.00 | 0.9858 (0.9856) | 2.8e-5 | 0.17 s |
| 8 | 16 / 16 (B) | 1024 | 1,048,576 | 3.0 | 8,077,784 | 1.55 | 0.2109 (0.2105) | 2.0e-5 | 0.52 s |
| 64 | unrestricted | — | — | — | 1,562,624 | 1.00 | 1 | 8.9e-3 | — |
| 64 | 16 / 4 | 256 | 262,144 | 6.0 | 1,244,736 | 1.26 | 0.0574 (0.0574) | 2.0e-5 | 0.15 s |
| 64 | 64 / 16 | 256 | 262,144 | 6.0 | 1,562,623 | 1.00 | 0.6388 (0.6383) | 2.6e-5 | 0.16 s |

* Balance: every supergroup holds exactly 390,625 valid neurons at Q = 256 (both C); every output
  touches all Q supergroups and all R rounds (per-round unit spread ≤ 1).  The tail placement was a
  measured fix: before it, H = 1M / C = 64 / S = 8 had 265–1024 valid neurons per supergroup.
* Set pair coverage matches 1 − (1 − (S−1)/(D−1))^R at every point (it is set co-membership, not
  coverage of 8-input combinations).  Realized record pair coverage equals set pair coverage at these
  sizes.  No duplicate sets across rounds occurred (D = 1024).  At D = 64, S = 32, R = 16 (test)
  sets overlap almost completely; the generator and kernels handle it.
* **Record repetition**: at S ≤ 16 the per-set record space C(S, 8) is small (S = 16: 12,870; S = 8:
  one record).  C = 8 / S = 16 / R = 4 has ~3.9 blocks per distinct record (≈31 neurons per record),
  so it does not provide C = 8 connection diversity; S = 32 (12.47M distinct) does.  This is
  reported, not corrected; the C-comparison conclusions are given for both S = 16 and S = 32.
* `independent_subsets` (diagnostic): same coverage and repetition, 100× worse feature-degree balance
  (CV 2–9e-3 vs 2–3e-5) and 4–27× slower generation (rejection sampling); not used.
* Metadata: feat + buckets 4.0–4.3 MiB at Q = 256 (16.8 MiB at Q = 1024, 33.6 MiB at Q = 2048);
  cached work lists 13.7 MiB (C = 64) / 21.2 MiB (C = 8); index tensor unchanged (23.9 / 190.8 MiB);
  generation peak 30 MiB (C = 64) / 214 MiB (C = 8) transient GPU memory, 0.1–1.0 s.

## 5. Kernels and per-phase experiments

New header `kernels/cuda/sparse_pool_sg.cuh` (+ `sg_*.cu` instantiations); existing kernels are
byte-identical.  All grouped kernels stage only feat[q]'s S rows of X (feature-major, cp.async) and
translate the global int16 indices through a D-entry shared-memory map (no extra index tensor).

**Forward** (`sg_fwd_kernel`): CTA = (chunk of buckets of one supergroup, T tokens); a warp owns whole
buckets, keeps the Mean sum in registers, quad-reduces and issues one FP32 `atomicAdd` per (token,
bucket) into `ysum [G, M]` (token-contiguous; replaces the generic `[1, M, G]` state buffer).
Finalize divides by n_g and transposes.  H = 10M, S = 64 screen (ms; generic 42.7 / 60.7):

| T | C = 64, 4 / 8 / 16 warps | C = 8, 4 / 8 / 16 warps |
|---:|---|---|
| 32 | 56.1 / 55.96 / 58.9 | 62.0 / 63.4 / 63.4 |
| 64 | 30.4 / 30.6 / 35.1 | 43.8 / 43.6 / 43.6 |
| 128 | **26.8** / 27.3 / 27.6 | 36.4 / **35.8** / 38.2 |

The gain comes from the token tile (parameters re-streamed M/T times: 64 instead of 256 passes), not
from the smaller slab: at T = 32 the grouped forward is slower than the generic one.  At H = 100M the
forward gain is smaller (C = 64 −12 %, C = 8 −20 %) because the generic kernel is relatively more
efficient there (1.56× vs 2.2× the MMA floor); the grouped forward sits at ~1.4× the floor at both
sizes.  Merge cost (ablation 3): −13 ms (C = 64) / −6 ms (C = 8) at H = 100M.  Forward S dependence
(H = 10M, C = 8): 32.8 ms at S = 8 → 35.8 at S = 64 → 38.4 at S = 128 → 51–54 at S = 256 (slab size
and CTA residency).

**dX** (`sg_dx_kernel`): CTA = (supergroup chunk, T tokens); per block the generic dZ arithmetic
(split hi/lo, mask epilogue for Mean; generic body as in-kernel non-finite fallback), Wᵀ by
`movmatrix` once per block (hoisted: asm volatile prevented CSE across token tiles), dA scattered
into FP32 slab copies `[NW / WG][S][T]`: WG = 1 warp-private copies with plain read-modify-write
(exclusive: a warp owns its copy and a record's 8 features are distinct, so lanes of one instruction
never collide — racecheck-clean), WG > 1 shared copies with the existing 64-bit CAS.  Copies are
summed and flushed with one FP32 `atomicAdd` per (feature, token) per CTA into `dXᵀ [D, M]`.

| H = 10M, C = 8 (ms; generic 233–240) | S = 8 | S = 16 | S = 32 | S = 64 |
|---|---:|---:|---:|---:|
| private, T = 16, 16 warps | 211 | 184 | 187 | 227 |
| private, T = 64, 8 warps | 166 | **140** | infeasible | infeasible |
| private, best feasible | **159** (T 128) | **140** | **174** (T 32) | 253 (T 32) |
| shared CAS slab per 8 warps, T = 64 | 252 | 217 | 205 | **198** |

* The structural opportunity is real but S-dependent: warp-private copies (NW·S·T·4 bytes) remove
  the CAS and permit large tiles only for small S; at S = 64 the copies no longer fit at T > 32 and
  one shared CAS slab with T = 64 wins.  Concentration hurts as predicted: S = 8 is slower than S = 16
  for every reduction strategy (all contributions land on 8 features).
* The fully private design is infeasible beyond the 99 KiB opt-in limit (e.g. S = 64, T = 64, 8
  warps: 138 KiB) and is rejected analytically there (`--skip-infeasible` records every case).
* Attribution at H = 100M, C = 8, S = 16 (ablations): global merge ≈ 0 ms (1331 → 1328), local scatter
  −458 ms (34 % of the grouped dX) — the dominant remaining dX cost even without CAS.  C = 64: merge
  ≈ 0, scatter −92 ms.
* C = 64: grouped dX is at parity at H = 10M (73.9–79.5 vs 74–75 ms) and −1.2…−2.2 % at H = 100M
  (693–702 vs 709–711 ms, ±2 ms).  Static SASS: 14.5 vs 17 instructions per HMMA; both are near the
  MMA/issue limit (dX ≈ 70 % of the measured MMA rate in both).

**dW** (`sg_dw_kernel`): exclusive ownership as in `k_dw_idx` (register accumulators over all M,
plain RMW write-back; no extra atomics or scratch), but the S rows of TS 16-token tiles are staged by
the whole CTA into an ST-stage ring (one barrier per stage); coefficient loads per unit (reused while
consecutive units share an output).  Same `dw_block` arithmetic in the same token order → **dW and db
bitwise identical to the generic kernels**.  H = 10M, S = 64 (ms):

| cfg (warps × units, tiles/stage, stages, min CTAs/SM) | C = 64 (generic 80.4) | C = 8 (generic 226.0) |
|---|---:|---:|
| 0: 8 × 2, 1, 3, 2 | 95.9 | 133.3 |
| 1: 8 × 4, 1, 3, 1 | 119.2 | 204.6 |
| 2: 8 × 2, 4, 2, 2 | 88.3 | **130.8** |
| 3: 8 × 4, 4, 2, 1 | 117.2 | 198.3 |

* C = 8 gains −42 % (H = 10M) / −49 % (H = 100M): the generic C = 8 dW stages the full D-wide slab (its
  32 work units × 4 blocks × 8 rows already cover all of D, so indexed staging cannot help) and runs at
  255 registers with spills, 1 CTA/SM; the grouped kernel stages S rows, uses 127 registers, no stack,
  2 CTAs/SM, and the exact `fast` dZ epilogue (bitwise equal to generic by construction).
* C = 64: the existing per-warp indexed ring without CTA barriers is better at H = 10M and at S ≥ 32;
  at H = 100M / S = 16 the grouped dW is −15 ms (706 vs 721) and is selected only there.

**Tile shapes per phase differ** (forward T = 128, C = 8 dX T = 64–128 or 32, dW 16-token MMA
reduction with 64-token stages): independent tuning was necessary.

## 6. Same-graph backend gain vs architecture (locality) change, attribution

H = 100M, seed 0, 4 rounds (fwd total / dX / dW / step, ms):

| C | S / R | Generic | Fwd only | dX only | dW only | All grouped | Auto (§1 plan) |
|---:|---|---|---|---|---|---|---|
| 8 | 16 / 4 | 386 / 2360 / 2058 / **4827** | 313 / 2363 / 2059 / 4757 | 387 / 1327 / 2061 / 3797 | 384 / 2360 / 1040 / 3806 | 309 / 1331 / 1043 / **2705** | 2706 |
| 8 | 32 / 8 | 387 / 2362 / 2049 / **4821** | 317 / 2365 / 2050 / 4755 | 387 / 1581 / 2050 / 4042 | 384 / 2361 / 1077 / 3846 | 314 / 1583 / 1079 / **2999** | 3001 |
| 8 | 64 / 16 | 387 / 2347 / 2049 / **4805** | 322 / 2351 / 2049 / 4744 | 387 / 1925 / 2051 / 4384 | 385 / 2348 / 1111 / 3866 | 319 / 1928 / 1113 / **3383** | 3382 |
| 64 | 16 / 4 | 296 / 710 / 721 / **1749** | 260 / 710 / 721 / 1714 | 296 / 702 / 719 / 1740 | 298 / 710 / 709 / 1740 | 263 / 703 / 706 / **1695** | 1703³ |
| 64 | 32 / 8 | 297 / 709 / 722 / **1750** | 262 / 709 / 722 / 1715 | 298 / 696 / 720 / 1737 | 299 / 709 / 723 / 1755 | 265 / 697 / 721 / **1705** | — |
| 64 | 64 / 16 | 298 / 709 / 723 / **1753** | 264 / 711 / 723 / 1720 | 298 / 693 / 723 / 1737 | 300 / 710 / 742 / 1776 | 266 / 694 / 739 / 1721 | 1723³ |

³ Re-measured with the final C = 64 rule (separate sessions): S = 16 1703 / 1693 / 1694 (3 seeds),
S = 64 1723 (fwd + dX grouped, dW generic).  Round-to-round spread is ±2 ms for C = 64 and ±5–20 ms
for C = 8; all reported differences exceed it.

* Phase gains are additive here (C = 8 / S = 16: −70 −1030 −1018 = −2118 ms vs −2121 measured).  One
  interaction was observed: with BF16 gradients the generic dX ran 5 ms slower after a grouped forward.
* Architecture effect (unrestricted vs constrained, both existing backend): C = 64 +0.0 %, C = 8
  +1.0 %; i.e. the locality of the constrained graph is not exploited by the existing kernels.

## 7. Native-diversity vs matched-connectivity C comparison

Matched control (`benchmarks/bench_matched.py`): the C = 64 graph, with every 64-neuron block
represented as eight C = 8 blocks carrying its record (weights, bias, slots, outputs unchanged;
`test_matched_c8_conversion_same_function` checks identical pre-activations).

| Graph (H = 100M) | Physical C | Existing backend | Best grouped plan |
|---|---:|---:|---:|
| C = 64 graph, S = 16 / R = 4 | 64 | 1763 | 1693–1703 |
| same graph as C = 8 blocks (matched) | 8 | 4803 | **2705** |
| native C = 8 graph, S = 16 / R = 4 | 8 | 4847 | 2717 |
| C = 64 graph, S = 64 / R = 16 | 64 | 1776 | 1723 |
| same graph as C = 8 blocks (matched) | 8 | 4788 | **3358** |
| native C = 8 graph, S = 64 / R = 16 | 8 | 4805 | 3382 |

The C = 8 cost is independent of connection diversity (matched ≈ native within 1 %): it is per-block
work — 8× more index records, gathers, dZ epilogue setups and slab scatters per neuron, and the
k8-wide MMAs.  The grouped plans cut the physical C = 8 / C = 64 ratio on identical mathematics from
2.72× to 1.59× (S = 16) and 1.95× (S = 64).

## 8. Resources, residency, contention (computed / static)

`results/input_supergroups/kernel_resources_S{16,64}.json`, `sass_loops.json`:

| Kernel (finalist) | Regs | Stack | Smem | CTAs/SM (warps) | Hot loop instr / HMMA |
|---|---:|---:|---:|---|---:|
| generic fwd C = 64 (T = 32, 16 warps) | 66 | 0 | 64 KiB | 1 (16) | 15.9 |
| grouped fwd C = 64 (T = 128, 4 warps), S = 16 | 108 | 0 | 6 KiB | 4 (16) | 12.2 |
| generic fwd C = 8 | 62 | 0 | 64 KiB | 1 (16) | 20.3 |
| grouped fwd C = 8 (T = 128, 8 warps) | 73 | 0 | 6 KiB | 3 (24) | 14.7 |
| generic dX C = 8 (16 warps, T = 16) | 51 | 0 | 96 KiB | 1 (16) | 44 |
| grouped dX C = 8 (T = 64, 8 warps, private), S = 16 | 79 | 0 | 36 KiB | 2 (16) | 26 |
| generic dW C = 8 (k_dw 8 × 4) | 255 | 40 B | 64 KiB | 1 (8) | 21.6 |
| grouped dW C = 8 cfg 2 | 127 | 0 | 6 KiB (S = 16) | 2 (16) | 18.3 |
| grouped dW C = 64 cfg 2 | 128 | 16 B | 6 KiB | 2 (16) | — |

Static counts are per loop body, not executed counts; residency is computed, not measured
occupancy.  Larger tiles raise registers (fwd C = 64: 56 → 108 from T = 16 to 128) without spills;
large S raises shared memory and lowers residency, which is why S ≥ 256 loses (dX at S = 256, T = 64:
98 KiB → 1 CTA/SM).

## 9. Full-scale variability, seeds, ablations, small M, BF16

* Seeds: §1 (3 seeds, spread ≤ 0.3 % of the gain).  Rounds: ±2 ms (C = 64), ≤ ±13 ms (C = 8).
* Small M (H = 100M, S = 16, 2 rounds): C = 8 M = 256 183.6 → 113.6 ms (−38 %), M = 1024 614.2 → 365.3
  (−41 %); C = 64 M = 256 81.7 → 79.8 (−2.4 %), M = 1024 242.4 → 236.7 (−2.3 %).  Work lists are built
  once and cached; there is no host synchronization in the per-step path (the bucket-size statistic
  used by the C = 64 rule is computed once on construction).
* BF16 gradients (S = 16, 2 rounds): C = 8 4752 → 2607 ms (−45 %); C = 64 1448–1457 → 1413–1414 ms
  (−2.4…−2.9 %); the C = 64 dX choice under BF16 is inconclusive within 5 ms (grouped 614 vs generic
  599–608) and auto keeps the existing dX.
* Family B and R probe (H = 10M, auto plans, `auto_screen_10M.csv`): C = 8 S = 16: R = 4 −41.8 %,
  R = 8 −40.4 %, R = 16 −39.8 %, R = 32 −39.4 %; S = 64: R = 8 −32.2 %, R = 16 −28.4 %, R = 32 −27.9 %;
  S = 32 / R = 16 −31.5 %, S = 128 / R = 16 −25.5 %.  Fewer, larger buckets (small R) help slightly.
  C = 64: −6.1…−7.1 % for all S ≤ 128 and R (forward only at H = 10M), −2.8 % at S = 256.
* Diagnostics at H = 100M: S = 8 (one record per set) C = 8 2877 ms (worse than S = 16), C = 64 1690.
  S = 1024 single-set control at H = 10M: grouped C = 8 838 ms vs generic 529 (the slab is D-wide;
  confirms the gains need a bounded set).

## 10. Memory

Hidden activations stay implicit; no buffer scales with M·H or M·Q·G.  Per step (M = 8192): forward
`ysum [G, M]` FP32 32 MiB (replaces the generic `[1, M, G]` sums, same size), backward `c1T [G, M]`
32 MiB (new), `dXᵀ [D, M]` 32 MiB (replaces `dX [M, D]`), shared-memory slabs only on chip.  Measured
step scratch 208 → 240–243 MiB.  Persistent: +4 MiB graph metadata, +14–21 MiB work lists, indices
unchanged.  Rejected by budget: `[M, Q, G]` output partials (8 GiB), `[M, Q, S]` dX partials (512 MiB
at R = 16), gathered-input caches (invalid across steps anyway).

| Plan (C = 8, H = 100M) | Step | Peak alloc | Extra scratch |
|---|---:|---:|---:|
| existing, unrestricted | 4795 ms | 10.46 GiB | — |
| grouped, S = 16 | 2717 ms | 10.71 GiB | +32 MiB |
| grouped, S = 32 | 3001 ms | 10.71 GiB | +32 MiB |

(Peak includes the 1.69 GiB benchmark snapshot.  The constrained-graph runs also include an unpacked
index copy that the harness kept alive after `--validate` (190.8 MiB at C = 8, 23.9 MiB at C = 64; fixed
in `bench_supergroups.py` after these runs); the constrained layer itself adds only the 4 MiB metadata.)

## 11. Numerics, tests, sanitizer, limitations

* dW / db: bitwise identical to the generic kernels for every tested S, C, shape, tiling and both
  precisions, including NaN / ±inf / subnormal / huge upstream gradients and signed zeros (int32 view
  compared).  Multi-step AdamW trajectories (4 steps) and main_grad accumulation over 3 backward calls
  are bitwise identical.
* Forward y: FP32 atomics per (token, bucket) change the summation order (grouped forward is not
  run-to-run deterministic; the generic forward is).  Grouped vs generic: ≤ 7 ULP, ≤ 4.1e-7 of the
  output scale (`numerics.json`, H = 1M, M = 1024).
* dX: ≤ 3.8e-7 of scale vs generic; the generic dX's own atomic run-to-run variation is ≤ 2.1e-7.
  Median 2 ULP, p99 80–132 ULP (large ULP counts only on near-cancelled small values).
* Against the PyTorch reference the existing tolerances (2e-5 split, 5e-3 BF16) hold, with one
  documented exception that is not caused by the new kernels: C = 8, S = 64, H = 300k, M = 256 has one
  neuron with true z = +3.7e-9 that rounds to ≤ 0 in the tensor-core FP32 accumulation of *both* the
  generic and grouped kernels (dW rel. error 8.2e-5 in both, bitwise equal); the test accepts it only
  when the generic kernel exceeds the tolerance identically.
* Tests: `tests/test_input_supergroups.py` (392 cases: generator/validator for all S at three sizes,
  determinism, overlapping sets, S = 8, matched conversion, grouped vs generic for C 8/64 × S 8–256 ×
  4 shapes × 2 precisions, every tiling, token tails B ∈ {8 … 264}, tiny/empty buckets, S = D,
  Mean(identity), per-phase selection, visible auto fallbacks (LME, power, softmax-weighted, Max, C = 16,
  B % 8 ≠ 0) and explicit-request errors, Mode A rejection, extremes, accumulation, AdamW, checkpoints).
  Full suite: **2061 passed, 1 skipped** (the pre-existing GELU-on-Triton skip).
* Compute Sanitizer (`run_sanitizer_sg.sh`, 22 grouped configurations per tool):
  memcheck, initcheck, synccheck **0 errors**; racecheck `--racecheck-report all` **0 hazards**.  The
  first attempt without `CUDA_MODULE_LOADING=LAZY` spent > 1 h instrumenting the 500-kernel library
  and was aborted; lazy loading instruments only launched kernels.  The pre-existing kernels were not
  re-sanitized in this phase (they are unchanged).
* Limitations: grouped kernels support Mean(ReLU / identity), packed Mode B, C ∈ {8, 64}, single-level
  pooling, feature-major input (B % 8 = 0), D ≤ 16384; everything else runs the existing kernels with
  the reason in `last_info["grouped_fallback"]` (auto) or raises (explicit).  No Mean-only merge is
  applied to other operators.  The project has no CUDA Graph path, so graph capture was not checked.
  The optimizer's stochastic-rounding seed counter is not checkpointed (pre-existing; affects resumed
  trajectories with SR on).  Model quality of constrained graphs was not studied.

## 12. Conclusions

* Bounded input working sets enable a **1.86× faster C = 8 backward** and **1.78× faster C = 8 steps** on the
  same graph, mainly through dX (warp-private, contention-free accumulation with 64–128-token tiles)
  and dW (S-row staging at half the registers, 2 CTAs/SM).  For C = 64 the same ideas give 3.4 %.
* Best S: C = 8 S = 16 (with record repetition) or S = 32 (without); C = 64 any S ≤ 64.  S ≥ 128 loses
  most of the C = 8 gain (shared memory), S = 8 is worse than S = 16 (contention, no diversity).
* C = 8 closes about two thirds of its excess cost: the C = 8 / C = 64 step ratio falls from 2.72× to
  1.60× (S = 16) / 1.76× (S = 32 vs C = 64 S = 32: 3001 / 1705).  The remainder is per-block physical
  cost, identical for matched and native graphs.

## 13. Remaining bottlenecks and next experiment

* C = 8 dX: the local slab scatter is 34 % of the grouped dX (ablation); a register-aggregated
  feature-owner scheme or packing the two m16 tiles' scatters into one 128-bit RMW per lane (private
  copies, MT ≥ 2) is the most direct next step.
* C = 8 dW / forward: per-block overhead (k8 MMAs, epilogue setup); combining the 4 blocks of a unit into
  wider MMAs is impossible with distinct records, but S ≤ 16 record repetition could be exploited
  explicitly (shared-record blocks as one wider block) — this changes the graph definition and was not
  attempted.
* C = 64: backward is at ~70 % of the measured MMA rate in both backends; forward is at ~1.4× its MMA
  floor.  Hardware counters (stall reasons) would be needed to go further.

## Reproduction

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd sparse_pooling/benchmarks
$PY -m pytest ../tests -q
bash run_sanitizer_sg.sh $PY
./run_sg_family_a.sh $PY 1000000 1024 a1M && ./run_sg_family_a.sh $PY 10000000 8192 a10M   # §5 screens
./run_sg_auto_screen.sh $PY 10000000 8192 input_supergroups/auto_a10M "16 4" "32 8" "64 16"   # §9
./run_sg_final_100M.sh $PY input_supergroups/final_100M 0 split "8 16 4" "64 16 4"           # §1, §6
./run_sg_seeds_100M.sh $PY && ./run_sg_extra_100M.sh $PY                                     # §1, §7, §9
./run_sg_ablation.sh $PY                                                                      # §5
$PY gen_stats.py --out ../results/input_supergroups/generator_stats_H100M.json               # §4
$PY numerics_sg.py --out ../results/input_supergroups/numerics.json                          # §11
$PY sg_resources.py --S 16 ; $PY sass_loops.py --kernel 'sg_dx_kernel<1, 8, 4, 8, 1, 2>'     # §8
$PY summarize_sg.py ../results/input_supergroups/final_100M --csv out.csv
```

Usage: `SparsePoolingLayer(..., shared_neurons=8, pooling="mean",
connectivity=ConnectivityPlan(input_supergroup_dim=16, partition_rounds=4, seed=0))`; execution via
`PhysicalPlan(cuda_input_supergroup_backend="auto" | "generic" | "grouped", cuda_grouped_fwd/dx/dw=…,
cuda_grouped_*_tokens / _warps, cuda_grouped_dx_reduction, cuda_grouped_dw_cfg)`; the kernels actually
used are in `layer.last_info` (`forward_kernel`, `dx_kernel`, `dw_kernel`, `grouped_*`).
