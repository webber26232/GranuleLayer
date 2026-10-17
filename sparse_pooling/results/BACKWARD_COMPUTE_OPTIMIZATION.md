# Backward compute optimization, phase 2 (RTX 3090)

Branch `backward-compute-optimization`, branched from `dw-indexed-staging` (c47f001).  Every
performance comparison below is against that commit, built and run from a separate git worktree
(its own kernel build), interleaved with this branch in the same session.  "Measured" = CUDA-event
phase timing (`benchmarks/bench_kernels.py`) or CUPTI kernel durations; "static" = SASS
instruction counts of one loop-body iteration (`benchmarks/sass_loops.py`); "computed" =
residency from compiler resource usage.  Hardware counters remain unavailable
(`RmProfilingAdminOnly: 1`, no passwordless sudo), so no number here is an Nsight counter.

Primary configuration: H = 100M, C = 64 (Mode B, packed), D = G = 1024, fan-in 8, M = 8192, BF16
parameters, FP32 accumulation and `main_grad`, Mean(ReLU), `grad_precision="split"`, fused AdamW
with BF16 moments.

## 1. Executive summary

Interleaved A/B, 4 rounds in alternating order, each 5 independent repetitions × 3 warmed-up
steps (`run_ab_phase2.sh`; `results/backward_compute/ab2_*.json`):

| | Baseline (dw-indexed-staging) | This branch (defaults) | Saved | Speedup |
|---|---:|---:|---:|---:|
| Forward kernel | 307.8 ms | 305.9 ms | — | 1.00× (unchanged code) |
| dX kernel | 831.3 ms (809.3–837.7) | 736.6 ms (736.0–737.5) | 94.7 ms | **1.129×** |
| dW kernel | 798.6 ms (780.2–799.6) | 748.0 ms (746.2–750.0) | 50.6 ms | **1.068×** |
| Backward total | 1629.7 ms | 1485.6 ms | 144.1 ms | 1.097× |
| Optimizer | 21.4 ms | 21.5 ms | — | 1.00× |
| **Full step** | **1964.0 ms (1909.0–1968.7)** | **1814.4 ms (1812.2–1816.5)** | **149.6 ms** | **1.082×** |
| Peak allocated / reserved | 8.64 / 9.88 GiB | 8.64 / 9.88 GiB | — | unchanged |

(Medians over rounds; ranges are min–max of the per-round medians.)  The slowest optimized
round (1816.5 ms) is faster than the fastest baseline round (1909.0 ms).  With BF16 dZ
gradients: step 1523.4 → 1482.8 ms (1.027×), all from dW (587.0 → 546.0 ms).

The absolute baseline is ~3 % slower than the 1903.8 ms in `MEMORY_TRAFFIC_OPTIMIZATION.md`:
the card is power-limited at ~345 W of its 370 W cap and its SM clock varied 1650–1770 MHz
between sessions (`gpu_before`/`gpu_after` in every JSON).  Only same-session interleaved
comparisons are used for conclusions.

What changed (all gradients bitwise identical for dW/db; dX up to its FP32-atomic order):

1. **Pooling-specialised dZ epilogues** (`cuda_dx_epilogue` / `cuda_dw_epilogue`):
   * *fast* (Mean, LogMeanExp, integer-p power mean): drops the per-element token-validity select
     and the multiply by the 0/1 activation derivative; Mean(ReLU) becomes `z > 0 ? c1_ftz : 0`.
   * *mask* (Mean only): dZ takes two values per token, so its BF16 hi/lo split is computed once
     per token and the per-element work is a `set.gt.ftz.u32` mask, one `PRMT` per value pair and
     one `LOP3` per operand — no per-element FP32→BF16 conversion at all.
   * Exactness for all inputs: subnormal coefficients are flushed exactly as the generic `FMUL.FTZ`
     does (`mul.ftz` per token); inf/NaN (or |c1| ≥ 2^64) coefficients, where the generic path's
     `c1 * 0` is NaN, are detected by a device-side flag (no host sync) and run the generic body.
2. **Measured defaults** (`"auto"`): dW *fast* everywhere it exists; dX *mask* for Mean with split
   gradients, *generic* for Mean with BF16 gradients (measured faster), *fast* for LME / power.
3. **dX warp dispatch** by work units per pooling unit (32 warps only if ≥ 32 work units).
4. **Explicit `cuda_dw_staging="indexed"` fallbacks are reported** (warning + `last_info`), and the
   profiler refuses runs whose requested variant was not applied.

Attribution (same A/B, same rounds): dX epilogue alone saves 101 ms of step time, dW epilogue
alone 49 ms, combined 150 ms (§12).

## 2. Hardware / software environment

| Item | Value |
|---|---|
| GPU | GeForce RTX 3090 (GA102, sm_86, 82 SMs, 24 GiB), VBIOS 94.02.59.00.5C, 370 W limit |
| Driver / CUDA (driver API) | 550.67 / 12.4 |
| nvcc | 12.1 (V12.1.66): `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17` |
| PyTorch / CUDA runtime / Triton | 2.7.0+cu128 / 12.8 / 3.3.0 |
| Python env | `/home/evga/.conda/envs/env_isaaclab` (3.11) |
| GPU state during the A/B | SM 1650–1770 MHz, memory 9501 MHz, 45–55 °C, 339–346 W |

## 3. Exact baseline

`dw-indexed-staging` defaults: indexed k_dw staging, dW cfg 3 (8 warps × 2 work units,
`__launch_bounds__(256, 2)`, 128 registers, 2 CTAs/SM), 32-warp k_dx at C = 64, generic
epilogues.  P0 re-measurement at the start of this phase (single run): forward 307.6, dX 840.8,
dW 805.9, step 1976.8 ms — 3.8 % slower than the phase-1 report at 1650–1680 MHz vs 1725–1770 MHz
SM clock; the A/B table in §1 is the controlled baseline.

## 4. dW instruction-level analysis

Static SASS of the hot token loop of `k_dw_idx` (Mean(ReLU), cfg 3).  One iteration = one
16-token tile for the warp's 2 work units (2 × 64 neurons × 16 tokens = 64 elements per lane);
static counts include both the masked (tail block) and unmasked copies of `dw_block`, so
per-tile dynamic counts are lower — the counts are for comparing variants.

| Loop body (split) | Total | HMMA | F2FP | PRMT | FADD | FMUL | FSEL | FSETP | LOP3 | MOV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generic | 1313–1325 | 64 | 128 | 128 | 256 | 128 | 128 | 0 | 29 | 91 |
| fast | 1207–1218 | 64 | 128 | 128 | 260 | 4 | 192 | 128 | 29 | 106 |
| mask | 1096–1099 | 64 | 4 | 68 | 136 | 0 | 0 | 128 | 264 | 110 |
| **generic, `grad_precision="bf16"`** | 981 | 48 | 64 | **0** | 128 | 128 | 128 | 0 | — | 94 |
| fast, bf16 | 835 | 48 | — | — | — | — | — | — | — | — |

Generic per-element sequence (from the SASS): `FSET.BF.GT.FTZ` (act′ as 1.0/0.0), `FMUL.FTZ`
(× c1), `FSEL` (token validity), then per value pair `F2FP.BF16.PACK_AB` (hi), 2 × `PRMT`
(unpack hi), 2 × `FADD` (residual), `F2FP` (lo), plus the bias `FADD`s.  The remaining
non-epilogue work per tile: 64 HMMA (2 × 8 m16n8k8 recompute + 2 × 8 m16n8k16 split gradient
per work unit), 8 `LDSM`, ~73 integer address/loop instructions, ~100 register moves (largely
re-forming the HMMA C-operand quad `{b0, b0, b1, b1}` after the 128-register cap reuses it).

## 5. BF16 packing experiments

* **Hypothesis "the non-split path wastes residual work": refuted by the compiler.**  With
  `SPLIT = false` NVCC already removes the unused residual: the bf16 dW loop has no `PRMT`, half
  the `F2FP` and no residual `FADD` (table above); dX likewise (32 → 16 F2FP, 32 → 0 PRMT).
  No `if constexpr` specialisation is needed and none was added.
* **Shorter split sequence for general dZ: none found.**  For an exact RNE hi/lo decomposition the
  generic 6 instructions per pair (F2FP, 2 PRMT, 2 FADD, F2FP) are already minimal on sm_86: a
  truncated hi needs the same unpack and changes the rounding (and loosens |lo| from ½ to 1 ulp);
  Veltkamp splitting costs 4 FP instructions per element; BF16x2 `sub` is sm_90+.  The packing is
  kept unchanged for every operator whose dZ varies per element.
* **Per-token packing (mask epilogue, Mean): implemented.**  When dZ ∈ {c1, 0} per token, hi/lo
  are computed once per token per tile (2 `split_bf16` per lane per tile in dX, 2 in dW), and
  per element only a mask is formed.  Instruction reduction (static, split): dX work-unit loop
  361 → 271 (−24.9 %), dW token loop 1313 → 1096 (−16.5 %).  The bits are exactly those of the
  per-element split (hi/lo of c1 or of 0).

## 6. dW epilogue experiments (H = 100M, C = 64; same-session screens, 3 reps × 2 steps)

| Variant | Exact? | Regs / stack | dW ms |
|---|---|---|---:|
| generic | — | 128 / 16 B | 790–799 |
| fast, `z > 0 ? c1 : 0` (screen) | **no**: keeps subnormal c1 (generic flushes), drops NaN for inf/NaN c1 | 128 / 8 B | 742–746 |
| fast, `z > 0 ? c1·1 : c1·0` per token (FTZ) | yes | 128 / 24 B | 777–785 |
| fast, `z > 0 ? mul.ftz(c1,1) : RZ` + non-finite flag, generic body in-kernel | yes | 128 / 8 B | 779 |
| **fast, same + generic fallback as a separate launch (default)** | yes | 128 / 8 B | **747–750** |
| mask, generic body in-kernel | yes | 128 / 88 B | 785 |
| mask, separate fallback launch | yes | 128 / 40 B | 770–773 |
| mask, 8 × 4 tiling (255 registers, 1 CTA/SM) | yes | 255 / 0 | 846 |

Findings:
* The exact select is as fast as the inexact screen once (a) the per-token zero register is
  replaced by `RZ` (exact because finite c1 makes the generic `c1·0` a signed zero, which every
  accumulator — initialised to +0 — absorbs) and (b) the non-finite fallback lives in a
  **separate launch** (one body per kernel: the fallback no longer competes for the 128-register
  budget).  The fallback launch costs 0.19 ms (CUPTI) when it exits immediately.
* The mask epilogue saves more instructions but spills at the 128-register cap (per-token hi/lo
  and the c1 copies for the bias sum add 8 live registers); it loses to the select in dW.
  Lifting the cap (8 × 4) loses the 2-CTA/SM residency and is slower still.  dW default: *fast*.
* Operators: LogMeanExp fast dW 904 → 876 ms (split), 720 → 681 ms (bf16); power mean p=2
  926 → 905 ms (split), 725 → 670 ms (bf16).  Softmax-weighted keeps the generic epilogue: its
  factor `fma(β ln2, a − c3, 1)` has no bound that would make the `z ≤ 0 → 0` shortcut provably
  equal to the generic `v · 0` (which is NaN if `v` overflows), so no exact variant exists.
* LME exactness argument: for valid tokens c2 = L2 ≥ max(a·scale), so `ex2(a·scale − c2)` ≤ 1 + ε
  and `c1·e` is finite whenever |c1| < 2^64 (included in the device flag); padded tokens get
  c2 = +∞ (e = 0) instead of the token select.  Power mean (p ∈ {1, 2, 4, 8}): the generic body
  already returns 0 for z ≤ 0 before multiplying; padded tokens give exactly 0.

Register pressure: every default dW variant keeps 128 registers and 2 CTAs/SM (§10).

## 7. dX compute experiments (H = 100M, C = 64)

| Variant | dX ms | Notes |
|---|---:|---|
| generic (32 warps) | 829–834 | baseline kernel |
| fast, exact | 790–800 | −4 %; 64 registers |
| **mask, generic body in-kernel (default, split)** | **730–738** | −11 %; 64 registers, no spills |
| mask, separate fallback launch | 762–779 | identical hot loops (271/718 instr.), different code placement |
| mask + `prefetch.global.L2` 2 / 4 work units ahead | 759–764 | **rejected**: parameters are already L2-resident (the hints only add LSU work) |

* The in-kernel vs separate-launch difference (~30 ms, reproduced in 2 interleaved rounds) has
  identical loop bodies; the only static differences are code placement and the outer-loop
  layout.  Without counters it cannot be attributed further (instruction-fetch behaviour is the
  likely candidate); the measured-faster structure was kept for dX, the opposite one for dW.
* BF16 gradients: dX generic 645, fast 660, mask 694 ms (2 rounds) — the short BF16 generic path
  leaves nothing for the per-token preprocessing to amortise, so `auto` selects *generic* for Mean
  with BF16 gradients.  For LME and power mean with BF16 the fast dX epilogue still wins (796 → 754
  and 758 → 718 ms).
* Parameter loading: deeper register prefetch would need a second 18-register `Unit<64>` buffer,
  above the 64-register budget of 32 warps; the L2 prefetch failed (above).  DRAM is not the limit:
  each wave of 82 CTAs shares one 128-unit parameter range (228 MB), ≈ 11 GB of DRAM reads per
  backward (≈ 14 ms at 819 GB/s, analytical).  The remaining parameter cost is L2→SM delivery of
  871 GiB of logical requests.
* Larger token tile: X slab (BF16) + dX slab (FP32) = D·BM·6 bytes = 96 KiB at BM = 16 and
  192 KiB at BM = 32 > 99 KiB per CTA; feature partitioning does not apply (random indices touch
  all features); clusters/DSMEM are sm_90-only.  Not feasible on GA102.

## 8. CAS / reduction

Correctness-breaking attribution ablations on the **current** dX (mask epilogue, 32 warps; timing
only, `SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_DX=n`, separate builds;
`backward_compute/ablation_*_exp01.json`):

| Variant | dX ms | Δ |
|---|---:|---:|
| reference | 733.2 | — |
| racy plain RMW instead of 64-bit CAS (=1) | 687.1 | −46 ms (6 %) |
| no scatter (=2) | 688.9 | −44 ms |
| no global flush (=3) | 739.8 | ≈ 0 (noise) |
| parameter loads made cache hits (=4) | 696.6 | −37 ms (5 %) |

With 32 warps hiding latency the CAS loop costs ~46 ms (down from ~164 ms in the 16-warp
kernel measured in phase 1), and the shared-memory accesses themselves ≈ 0 (racy RMW ≈ no
scatter).  Any correct replacement (warp-group ownership of feature ranges, hierarchical
reduction, feature-partitioned accumulation) adds shared-memory traffic and synchronisation of
its own and could at best recover part of 46 ms (2.5 % of the step); warp-local duplicate
aggregation has no opportunity (index records are sampled distinct, so no two lanes of one CAS
instruction target the same word) and the XOR swizzle was already bounded in phase 1.
**Decision: not implemented.**

## 9. Recomputation-sharing feasibility (dX/dW fusion)

The kernels need opposite reductions: dX sums over neurons (kept on chip by token-stationary
CTAs in a 64 KiB FP32 slab per 16 tokens), dW sums over tokens (kept in registers by
weight-stationary CTAs over all 8192 tokens).  Any fused tiling keeps one reduction on chip and
must move the other off chip.

| Candidate | Off-chip reduction | Volume (analytical) | Max intermediate state |
|---|---|---|---|
| token-stationary fused CTA (16 tok × 128 units) | dW partials via FP32 global atomics | 512 tiles × 8.0e8 weights = **4.1e11** atomics | — |
| same with partial buffers | dW partials | 512 × 3.2 GB | ≫ 24 GiB |
| weight-stationary fused CTA (16 work units, all tokens) | dX via global atomics (random features: no smem merging) | 1.56M blocks × 8 × 8192 = **1.0e11** atomics | — |
| hybrid: 512-token groups × work-unit chunks | dW atomics (16 groups) + dX atomics | 1.3e10 + 1.0e11 | — |
| share only sign(z) (all Mean(ReLU) dZ needs) | bitmask between kernels | H·M bits = **102 GB** | ≫ 24 GiB; kernel orders also mismatch |
| producer/consumer persistent kernel | needs dZ tiles in both loop orders | same as above | — |

Saving: one projection recompute = 6.4e9 m16n8k8 (13.1 TFLOP, ≥ 0.19 s at the measured
69 TFLOPS) plus one dZ epilogue — at most ~0.25–0.35 s.  Cost: ≥ 1e11 global FP32 atomics
(≥ 1 s at plausible L2 atomic throughput) or tens of GB of scratch.  **No-go**: the separate
kernels exist precisely to keep both reductions on chip; no bounded-memory tiling offers a net
benefit, so no prototype was built.

## 10. Register / shared-memory resources (computed)

`results/backward_compute/kernel_resources.json` (cuobjdump; sm_86 residency rules):

| Kernel (C = 64) | Regs | Stack | Smem | CTAs/SM | Warps/SM |
|---|---:|---:|---:|---:|---:|
| `k_dw_idx` Mean cfg 3, generic, split | 128 | 16 B | 8 KiB | 2 | 16 |
| `k_dw_idx` Mean cfg 3, fast, split (default) | 128 | 8 B | 8 KiB | 2 | 16 |
| `k_dw_idx` Mean cfg 3, mask, split | 128 | 40 B | 8 KiB | 2 | 16 |
| `k_dw_idx` Mean cfg 3, fast, bf16 (default) | 128 | 24 B | 8 KiB | 2 | 16 |
| `k_dw_idx` LME cfg 3, generic / fast | 128 / 128 | 8 / 8 B | 8 KiB | 2 | 16 |
| `k_dw_idx` power p=2 8 × 4, generic / fast | 255 / 255 | 0 / 0 | 16 KiB | 1 | 8 |
| `k_dx` Mean 32 warps, generic / fast / mask | 64 | 0 | 96 KiB | 1 | 32 |
| `k_dx` Mean 16 warps, mask (small units) | 64 | 0 | 96 KiB | 1 | 16 |
| `k_dx` LME 32 warps, fast | 64 | 0 | 96 KiB | 1 | 32 |

No default kernel lost residency.

## 11. Numerical correctness and sanitizer results

* `pytest sparse_pooling/tests`: **1669 passed, 1 skipped** (the pre-existing GELU-on-Triton skip).
  New `tests/test_backward_compute.py` (929 cases) checks every variant (fast / mask / auto, dX-only,
  dW-only) against the generic epilogue — **dW/db bitwise identical**, dX within 1e-6 relative
  (atomic order) — for Mean(ReLU), Mean(identity), LME, power p=2 / p=3.5 (generic fallback),
  softmax-weighted (generic), split and BF16, C = 8 and 64, four shapes (ragged groups, partial
  blocks, B ∈ {8, 40, 64, 136}); against the PyTorch reference with the existing tolerances
  (2e-5 split, 5e-3 BF16, not weakened); ReLU boundary z = 0 with signed upstream gradients;
  extreme magnitudes (1e30, 3e-38, 1e-36 → subnormal c1); ±inf, NaN and 3e38 upstream gradients
  (bitwise, NaN pattern included); colliding/repeated index records; accumulation over two
  backward calls; the dX warp dispatch rule; the explicit-indexed fallback warning.  Phase-1
  tests (indexed vs full staging bitwise, multi-step AdamW trajectories, Mode A / C = 8 fallbacks,
  Max) run with the new defaults and pass.
* Compute Sanitizer (`run_sanitizer.sh`, workload extended with every new variant incl. both
  non-finite fallback paths, 16- and 32-warp dX, C = 8; log `backward_compute/sanitizer.log`):
  memcheck, initcheck, synccheck **0 errors**; racecheck: only the 20 pre-existing reports of the
  Triton `bwd_dense_kernel` (triton_fused.py:441), 0 others.  Racecheck with
  `--racecheck-report all --print-limit 0` on the native phase-2 variants alone:
  **0 hazards (0 errors, 0 warnings)**; memcheck / initcheck / synccheck 0 errors
  (`sanitizer_native_phase2.log`).

## 12. Before / after timings

### 12.1 Attribution (same A/B rounds as §1, split)

| Variant | New dW epilogue | New dX epilogue | dX ms | dW ms | Step ms |
|---|---|---|---:|---:|---:|
| baseline (worktree) | — | — | 831.3 | 798.6 | 1964.0 |
| this branch, generic | off | off | 832.6 | 794.7 | 1960.3 |
| dW only | on (fast) | off | 835.2 | 747.6 | 1914.7 |
| dX only | off | on (mask) | 737.5 | 795.5 | 1863.1 |
| combined (defaults) | on | on | 736.6 | 748.0 | 1814.4 |

"Generic" on the new branch reproduces the baseline (the refactored generic path is not slower),
and the two effects add (−49 ms and −101 ms of step vs −150 ms combined).

### 12.2 dX warp dispatch (`run_dispatch_sweep.sh`, Mean, C = 64, mask/fast epilogues)

| H | work units / unit | dX 32-warp / 16-warp, M = 8192 | M = 1024 |
|---:|---:|---:|---:|
| 1M | 16 | 1.113 | 1.077 |
| 2M | 32 | 0.941 | 0.906 |
| 4M | 64 | 0.906 | 0.882 |
| 10M | 160 | 0.871 | 0.885 |
| 30M | 472 | 0.887 | 0.887 |
| 100M | 1528 | 0.875 | 0.886 |

k_dx gives each warp whole work units of a pooling unit, so with fewer than 32 work units some of
32 warps idle; the rule `32 warps iff work units per unit ≥ 32` matches the measured crossover at
both token counts (packed groups round blocks up to multiples of 4).  Explicit
`cuda_dx_warps` still overrides.

### 12.3 Configuration matrix (one run each, 3 reps × 3 steps; `matrix_*_exp03.json`, ms)

| Config | dX base → new | dW base → new | Step base → new | Speedup |
|---|---|---|---|---:|
| Mean, 100M, C 64, M 8192, split | 809.1 → 714.2 | 779.4 → 728.4 | 1907.8 → 1760.6 | 1.084 |
| Mean, 100M, C 64, M 8192, bf16 | 621.9 → 619.0 | 586.5 → 545.1 | 1526.1 → 1481.8 | 1.030 |
| Mean, 100M, C 64, M 1024, split | 100.5 → 88.4 | 100.6 → 96.1 | 259.6 → 243.5 | 1.066 |
| Mean, 100M, C 64, M 1024, bf16 | 71.7 → 71.8 | 78.1 → 73.3 | 208.6 → 203.8 | 1.023 |
| Mean, 100M, C 8, M 8192, split | 2373.5 → 2308.6 | 2121.6 → 2065.2 | 4911.9 → 4783.9 | 1.027 |
| Mean, 100M, C 8, M 1024, split | 299.6 → 291.5 | 259.4 → 254.4 | 628.8 → 615.2 | 1.022 |
| Mean, 10M, C 64, M 8192, split | 85.1 → 75.1 | 83.8 → 80.0 | 214.7 → 200.7 | 1.070 |
| Mean, 10M, C 64, M 8192, bf16 | 62.6 → 62.4 | 65.2 → 60.3 | 173.8 → 169.2 | 1.027 |
| Mean, 10M, C 64, M 1024, split | 11.1 → 9.9 | 10.1 → 9.6 | 28.7 → 27.0 | 1.063 |
| Mean, 10M, C 8, M 8192, split | 241.5 → 238.3 | 230.4 → 227.3 | 537.6 → 529.0 | 1.016 |
| Mean, 1M, C 64, M 8192, split | 13.76 → 11.33 | 8.26 → 7.91 | 45.95 → 43.67 | 1.052 |
| Mean, 1M, C 64, M 1024, split | 2.46 → 2.25 | 1.06 → 1.01 | 5.82 → 5.58 | 1.042 |
| Mean, 1M, C 8, M 8192, split | 26.4 → 26.0 | 22.8 → 22.0 | 74.9 → 74.0 | 1.013 |
| LME τ=0.5, 100M, C 64, split | 921.2 → 891.0 | 880.9 → 852.2 | 2422.6 → 2364.5 | 1.025 |
| LME τ=0.5, 100M, C 64, bf16 | 781.9 → 748.5 | 705.9 → 661.9 | 2105.9 → 2032.7 | 1.036 |
| LME τ=0.5, 100M, C 8, split | 2432.9 → 2396.6 | 2861.9 → 2864.9 | 6041.8 → 6003.0 | 1.006 |
| Power p=2, 100M, C 64, split | 896.9 → 860.6 | 901.9 → 877.9 | 2292.3 → 2230.2 | 1.028 |
| Power p=2, 100M, C 64, bf16 | 717.1 → 698.9 | 700.9 → 652.5 | 1904.5 → 1840.9 | 1.035 |
| Softmax-weighted, 100M, C 64, split | 1031.5 → 1034.0 | 1072.6 → 1072.9 | 2776.5 → 2776.4 | 1.000 (unchanged path) |
| Softmax-weighted, 100M, C 64, bf16 | 936.9 → 937.9 | 899.3 → 899.0 | 2508.7 → 2514.7 | 0.998 (unchanged path) |
| Max, 100M, C 64 / C 8 (control) | — | — | 408.4 → 407.0 / 494.6 → 493.5 | 1.00 |

Peak allocated / reserved memory is identical in every row (e.g. 8.64 / 9.88 GiB at 100M,
C = 64, M = 8192).  Single-run rows below ~1 % (C = 8 LME, softmax-weighted, Max) are within
run-to-run variation and are not claimed as changes.

## 13. Failed experiments

* Inexact fast dZ (`z > 0 ? c1 : 0`): fastest dW screen (742 ms) but changes FTZ behaviour of
  subnormal coefficients and the NaN pattern for non-finite upstream gradients; caught by the
  extreme-magnitude test and replaced by the exact form.
* Per-token `c1·1` / `c1·0` registers (exact, no flag): lost the gain (dW 777–785 ms) — the extra
  live register at the 128 cap.
* Mask epilogue for dW: fewer instructions but spills at 128 registers (40–88 B stack); 770–785 ms
  vs 748 ms for *fast*; with 255 registers (8 × 4) 846 ms.
* Mask / fast dX epilogue with BF16 gradients for Mean: 694 / 660 ms vs 645 ms generic.
* Separate fallback launch for dX: ~30 ms slower than the in-kernel fallback with identical hot loops.
* `prefetch.global.L2` of dX parameters 2 / 4 work units ahead: +25–35 ms.
* Shortening the general split-precision BF16 packing: no shorter exact RNE sequence on sm_86.
* dX/dW recompute sharing: infeasible without ≥ 1e11 atomics or ≥ 100 GB scratch (§9).

## 14. Remaining bottlenecks

* dW (748 ms) and dX (737 ms) are now about equal.  dW: ~39.3 TFLOP of MMA work (≈ 0.57 s at the
  measured MMA rate, analytical), the remaining SIMT epilogue (hi/lo split for the split pass)
  and ~100 register moves per tile caused by the 128-register cap.
* dX: projection recompute + split gradient MMAs dominate; CAS ~46 ms, parameter delivery
  ~37 ms (ablations).
* Forward (306 ms) is untouched in this phase.
* C = 8 gains only from the dX epilogue (dW at C = 8 uses the full-slab kernel).
* Softmax-weighted has no exact specialised epilogue.

## 15. Recommended future work

1. Nsight Compute counters (admin): stall reasons for `k_dw_idx` cfg 3 (MOV / register-bank
   pressure vs MMA issue) and instruction-fetch stalls for the two dX code layouts (§7).
2. dW: an MMA C-operand layout that does not need re-forming `{b0, b0, b1, b1}` every tile (bias
   added through a dedicated accumulator init), to remove most of the ~100 register moves.
3. Forward: the same per-token / per-tile specialisation ideas do not apply (no dZ), but the
   Mean(ReLU) forward epilogue (`fmaxf` + add per element) could be audited the same way.
4. Softmax-weighted: an exact specialised path needs a provable bound on `c1·pr·f`; a device-side
   overflow check of `max |a − c3|` (analogous to the |c1| flag) would make it possible.

## Reproduction

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
WT=<path>; git worktree add --detach $WT dw-indexed-staging          # the baseline
$PY -m pytest sparse_pooling/tests -q
bash sparse_pooling/benchmarks/run_sanitizer.sh $PY
bash sparse_pooling/benchmarks/run_ab_phase2.sh $PY $WT 4 split      # §1 / §12.1
bash sparse_pooling/benchmarks/run_ab_phase2.sh $PY $WT 4 bf16
bash sparse_pooling/benchmarks/run_dispatch_sweep.sh $PY mean        # §12.2
bash sparse_pooling/benchmarks/run_matrix_phase2.sh $PY $WT          # §12.3
$PY sparse_pooling/benchmarks/sass_loops.py --kernel 'k_dw_idx<2, 1, 64, 8, 2, 2, 1, 0, 2, 1>' ...
$PY sparse_pooling/benchmarks/kernel_resources.py --filter '^(k_dw_idx|k_dx)<(2|3|4|5), 1, 64'
```

Raw data in `results/backward_compute/`: `ab2_mean_H100M_C64_M8192_{split,bf16}_<variant>_r<n>.json`,
`matrix_<pool>_H<H>_C<C>_M<M>_<prec>_{baseline,optimized}_exp03.json`,
`dispatch_mean_H<H>_C64_M<M>_split_dxw<W>_exp02.json`, `ablation_*_exp01.json`,
`kernels_mean_H100M_C64_M8192_split_baseline_exp00.json` (P0), `sass_loops.json`,
`kernel_resources.json`, `matrix_summary.txt`, sanitizer logs.  Phase-1 data
(`results/traffic/`, `MEMORY_TRAFFIC_OPTIMIZATION.md`) is unchanged.
