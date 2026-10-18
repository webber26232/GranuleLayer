# MMA operand/register layout and forward epilogue optimization, phase 3 (RTX 3090)

Branch `mma-forward-epilogue-optimization`, branched from `backward-compute-optimization`
(**fcc363b47fcac6cd9692ee1480810d81590ea4ff**).  Every performance comparison is against that
commit, built and run from its own git worktree (own kernel build), interleaved with this branch in
the same session.  Vocabulary used throughout:

* **measured**: CUDA-event phase timing of real training steps (`benchmarks/bench_kernels.py`:
  forward, fused backward, fused AdamW with BF16 moments);
* **static**: instruction counts in the SASS of one copy of a loop body (`benchmarks/sass_attrib.py`,
  `benchmarks/sass_loops.py`).  Static counts are **not** dynamic instruction counts: a loop body
  contains the masked and unmasked copies of the block code and one copy per work unit of a warp,
  of which only one runs per block;
* **computed**: occupancy from compiler resource usage (`cuobjdump -res-usage`).

Hardware counters are still unavailable (`RmProfilingAdminOnly: 1`, no passwordless sudo): no number
in this report is an Nsight Compute counter.

Primary configuration: H = 100M, C = 64 (Mode B, packed), D = G = 1024, fan-in 8, M = 8192, BF16
parameters, FP32 MMA accumulation and FP32 `main_grad`, Mean(ReLU), `grad_precision="split"`,
fused AdamW with BF16 moments.

## 1. Executive summary

* **Track A (dW MMA C-operand / register layout): hypothesis disproven for exact variants.**  The
  ~100 static MOVs of the dW hot loop are real (119; 74 of them build HMMA C operands — the bias
  quads `{b0, b0, b1, b1}` — and 34 pack the dZ A operands), but ptxas re-derives the same register
  allocation from every exact source formulation tried (persistent C fragment, in-place accumulate,
  one-tile lookahead): −4 static MOVs at best, 128 registers, 2 CTAs/SM, **no measurable dW
  change** (732.2 vs 732.8 ms, 4 rounds).  An ablation that drops the bias entirely (wrong results)
  bounds the whole bias-C cost of dW at ≈ 20 ms (3 %).  Moving the bias out of the MMA
  (C = 0, then FADD) is *not* exact — a micro-test finds inputs where it differs — and is not faster.
  No dW change is retained.
* **Track B (forward Mean epilogue): one exact optimization retained.**  Forward ablations show the
  forward is latency/issue-bound: removing the bias (its loads, unpack and C operand) saves 47 ms,
  removing most of the epilogue another 48 ms.  FMNMX/FADD are already one each per element (the compiler eliminated the
  intermediate arrays and unused `PoolState` fields), so the remaining handle is *latency*: issuing
  the projection MMAs of 4 n8 tiles per epilogue chunk instead of 2 (`cuda_fwd_mean_epilogue=
  "chunk4"`, default for Mean(ReLU) at C = 64; measured slower for Mean(identity), §12.1, where
  `auto` keeps the baseline) lets ptxas build the bias C fragments ahead of the MMAs.  Every
  pooling state still receives its values in the original order: **forward outputs and all
  gradients are bitwise identical**.

Paired interleaved A/B, 10 rounds (4 attribution rounds + 6 confirmation rounds, alternating order,
each 5 repetitions × 3 warmed-up steps; `results/mma_forward/{final,confirm}_*.json`,
`paired_deltas_final_confirm.json`):

| | Baseline (fcc363b) | This branch (defaults) | Paired Δ (median, range) | |
|---|---:|---:|---:|---|
| Forward kernel | 299.1 ms | 290.5 ms | **−9.0 ms** (−10.5 … −5.3) | faster in 10/10 rounds, 1.031× |
| dX kernel | 714.9 ms | 713.2 ms | −0.7 ms (−12.5 … +4.5) | unchanged code |
| dW kernel | 727.1 ms | 727.1 ms | −0.8 ms (−8.6 … +5.9) | unchanged code |
| Backward total | 1442.9 ms | 1441.2 ms | −0.6 ms | |
| **Full step** | **1763.6 ms** | **1753.5 ms** | **−10.0 ms** (−29.6 … +4.6) | faster in 9/10 rounds, **1.0057×** |
| Peak allocated / reserved | 8.64 / 9.88 GiB | 8.64 / 9.88 GiB | 0 | |

The step saving equals the forward saving; it is small (0.57 %) but reproducible.  The GPU changed
clock state during the confirmation run (rounds 1–3 ran ~4 % slower for both builds); paired
deltas are unaffected, absolute medians mix both states.

## 2. Hardware and software environment

| Item | Value |
|---|---|
| GPU | GeForce RTX 3090 (GA102, sm_86, 82 SMs, 24 GiB), 370 W limit |
| Driver | 550.67 |
| nvcc | 12.1 (V12.1.66): `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17` (+ only `-D` switches for screening builds) |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| Python env | `/home/evga/.conda/envs/env_isaaclab` (3.11) |
| GPU state during the final A/B | SM 1650–1785 MHz, 39–54 °C, 339–347 W (`gpu_before` / `gpu_after` in every JSON) |

Every variant is its own library: the build hash covers the flags and the `#include` closure, so a
screening build can never be served from a stale object.  Objects built while headers were being
edited were deleted and rebuilt (§16).

## 3. Exact baseline

`backward-compute-optimization` @ fcc363b defaults: forward `fwd_mma_kernel<Mean, ReLU, C=64, MT=2,
NW=16, packed>` (66 registers, 64 KiB slab, 1 CTA/SM); dX `k_dx<Mean, ReLU, 64, 32 warps, split,
mask epilogue>`; dW `k_dw_idx<Mean, ReLU, 64, NW=8, WPW=2, ST=2, split, P_INT=0, MINB=2, fast
epilogue>` (128 registers, 8 B stack, 8 KiB smem, 2 CTAs/SM).  Measured in this session (10 paired
rounds, median): forward 299.1, dX 714.9, dW 727.1, backward 1442.9, step 1763.6 ms.  Within this
session the same baseline build measured 1755–1846 ms per round depending on the card's clock state,
so the historical 1814 ms of the phase-2 report (another session) is not comparable; only
same-session pairs are used.

## 4. dW MMA operand / register analysis (experiment A0)

`sass_attrib.py` follows every register move in an HMMA loop to the first instruction that reads
the moved register and classifies it by that consumer.  Default dW kernel, static counts of the
token-tile loop (4 copies of the block code: 2 work units × masked/unmasked):

| Item | Static count (loop body) | Executed path, one 64-neuron block × 16 tokens |
|---|---:|---:|
| Instructions | 1207 | 250 |
| HMMA | 64 (32 × m16n8k8 projection, 32 × m16n8k16 gradient) | 16 (8 + 8) |
| MOV / IMAD.MOV | 119 | 26 |
| … feeding an HMMA C operand (bias quads) | 74 | 21 |
| … feeding an HMMA A operand (dZ hi/lo packing) | 34 | 5 |
| … other / not resolved by the tool | 3 / 8 | 0 |
| F2FP (f32 → bf16x2) | 128 | 32 |
| PRMT | 128 | 32 |
| FSETP / FSEL (fast dZ epilogue) | 128 / 128 | 32 / 32 |
| FADD | 260 | 64 |
| Global loads (LDG; the coefficient loads) | 4 | — |
| Shared loads (LDSM) | 8 | 2 |
| Registers / stack | 128 / 8 B | |
| Local memory in the loop | 0 LDL/STL (one `STL.64` before the loop, two `LDL` in the write-back) | |

## 5. Annotated SASS findings

**Where the C-operand MOVs come from.**  The C operand of an m16n8k8 is a quad of consecutive
registers holding `{b0, b0, b1, b1}` (bias of neuron rows g and g+8, two token columns each).  The
kernel keeps `b0, b1` (2 registers per row tile and work unit, `biasT`, 16 registers), so each
projection HMMA needs the duplicates re-made; ptxas additionally gives the second token half a
private copy of the quad and writes that HMMA in place.  Representative excerpt (row tile 0 of one
block; full listing in `results/mma_forward/annotated_sass_excerpts.txt`):

```
01880  IMAD.MOV.U32  R53, RZ, RZ, R52     ; MOV -> hmma_c (HMMA @0x19b0)    b0 duplicate
018b0  IMAD.MOV.U32  R55, RZ, RZ, R54     ; MOV -> hmma_c (HMMA @0x19b0)    b1 duplicate
018e0  IMAD.MOV.U32  R96..R99 <- R52,R52,R54,R54  (4 MOVs)    private quad for token half 1
01950  IMAD.MOV.U32  R57, RZ, RZ, R56     ; MOV -> hmma_c (HMMA @0x1a10)    next row tile
01990  LDSM.16.MT88.2  R10, [R61]         ; X operand (tokens x inputs)
019b0  HMMA.1688.F32.BF16  R92, R12, R10, R52     ; z = W.x + {b0,b0,b1,b1}, token half 0
019c0  HMMA.1688.F32.BF16  R96, R12, R11, R96     ; token half 1, in place on the copy
019d0  FSETP.GT.FTZ.AND  P6, PT, R92, RZ, PT      ; ReLU'(z)
01a20  FSEL  R65, R2, RZ, P6                      ; dZ = z > 0 ? c1 : 0
01ae0  F2FP.BF16.PACK_AB  R57, R73, R65           ; dZ hi (bf16x2)
01b10  IMAD.MOV.U32  R96, RZ, RZ, R57     ; MOV -> hmma_a (HMMA.16816 @0x1bd0)  A-operand quad
01b60  PRMT  R92, RZ, 0x5410, R57                 ; hi back to f32 for the lo residual
01ba0  FADD.FTZ  R92, R65, -R92                   ; lo = dz - hi
01bd0  HMMA.16816.F32.BF16  R48, R96, R6, R48     ; dW += dZ_hi x X
```

Per executed block the C-operand moves are 21 of 250 instructions (8.4 %); the gradient A-operand
moves (5) come from F2FP results that land outside the A-operand quad.  The phase-2 report's "≈ 100
register moves" is the static count of the whole loop (119), not the executed count.

## 6. dW layout and accumulator experiments (A1–A4)

All variants are compile-time screening switches in commit 3db68de (`SP_DW_MMA`, described in the
header), built with identical flags plus one `-D`.  Static counts of the default dW instantiation:

| Variant | Description | Loop instr | MOV (C / A) | Regs / stack | CTAs/SM | Bitwise |
|---|---|---:|---:|---:|---:|---|
| A0 | current scalar `b0, b1` | 1207 | 119 (74 / 34) | 128 / 8 | 2 | — |
| A1 (`SP_DW_MMA=1`) | persistent 4-register `{b0,b0,b1,b1}` fragment per row tile (bias stored 4×) | 1203 | 115 (70 / 34) | 128 / 8 | 2 | yes (construction) |
| A2 (`=2`) | one fragment copy per row tile; token half 1 accumulates into it in place (`"+f"` tied operands, the standard PTX accumulate form) | 1203 | 115 (70 / 34) | 128 / 8 | 2 | yes |
| A4 (`=4`) | A2 + projection MMAs of row tile q+1 issued before tile q's epilogue | **SASS identical to A2** | | | | yes |
| A3 diag (`=3`) | C = 0, bias added with 4 FADD per MMA | 1243 | 30 (0 / 18) | 128 / 0 | 2 | **no** (§6.2) |
| ablation (`=5`) | C = 0, no bias (WRONG results) | — | — | 128 | 2 | no |

ptxas materialises a C quad per HMMA regardless of how the source expresses it: the persistent
fragment of A1 and the tied operands of A2 end up with the same register count and all but 4 of the
C-operand moves, and the reordering of A4 is undone by the scheduler (byte-identical SASS to A2).
Without the 128-register cap ptxas might keep A1's quads resident, but raising the cap would cost
the second resident CTA, which phase 1 showed is worth far more.  Measured (interleaved, `s2_*.json`, H = 100M,
4 rounds): **dW 732.8 (A0) vs 732.2 ms (A2)**; H = 10M screening (2 rounds): A0 80.1, A1 79.2,
A2 78.8, A3 80.3 ms — within the 10M round-to-round spread (78.9–81.2 ms for A0 alone).

### 6.1 Bound on the C-operand cost (ablation)

`SP_ABLATE_DW=4` (production tree; was `SP_DW_MMA=5` in the screening tree) removes the bias from
the projection MMA altogether: dW 732.1 → 710.4 ms in round 2; round 1 (731.5 vs 727.9 ms) ran
in a slower clock state for the ablation build (its dX, untouched code, was +29 ms), and
normalised by dX it is −3 %.  The *entire* bias-C cost of dW is therefore ≈ 20 ms (≈ 3 % of dW,
1.1 % of the step); none of it was recoverable by an exact layout change.

### 6.2 Accumulator initialisation (A3): C = b versus C = 0 then + b

* Random data (`numerics_dw_bias.py`, H = 1M, M = 8192, bias σ = 0.2 and 0.02,
  `results/mma_forward/a3_numerics_*.json`): dW, db, dX bitwise identical, 0 ReLU decision
  flips over 1,048,576 neurons × 8192 tokens.  The flip count is exact: with dy = n per unit the
  split db of a neuron is the integer number of tokens with recomputed z > 0; the control build
  without bias (`SP_ABLATE_DW=4`) shows 998,830 neurons with flips, so the method is sensitive.
* Directed micro-test (`benchmarks/mma_c_rounding.cu`, `a3_mma_c_rounding_microtest.txt`): products
  {1, 2⁻²⁴} with c = −1 give `mma(C=c) = 2⁻²⁴` (exact) but `mma(C=0) + c = 0` — the accumulator
  input enters the tensor core's internal sum before rounding.  **The variant is not exact in
  general**, random tests just do not hit the cancellation cases.
* It is not faster either (dW 80.3 vs 80.1 ms at 10M, within noise; +36 instructions — the 4 FADD
  per MMA replace the MOVs one for one and add more).  Rejected; not promoted.

## 7. Register and occupancy measurements

| Resource | dW baseline | dW best candidate (A2) | Forward baseline (FE=0) | Forward chunk4 (FE=1) |
|---|---:|---:|---:|---:|
| Registers / thread | 128 | 128 | 66 | 70 |
| Shared memory / CTA | 8 KiB | 8 KiB | 64 KiB slab (+ reduction reuse) | 64 KiB |
| Stack / local memory in the loop | 8 B / 0 LDL-STL | 8 B / 0 | 0 / 0 | 0 / 0 |
| Theoretical CTAs / SM | 2 (registers) | 2 | 1 (shared memory) | 1 |
| Resident warps / SM | 16 | 16 | 16 | 16 |
| MOV (static loop) | 119 | 115 | 60 | 52 |
| HMMA (static loop) | 64 | 64 | 16 | 16 |
| Kernel time (H = 100M) | 726.3 ms | 726.7 ms | 298.3 ms | 289.0 ms |

The forward's residency is bounded by its 64 KiB slab (1 CTA/SM at 512 threads); its 70 registers
leave headroom to 128.  (Kernel times in this table: medians of the 4-round attribution A/B, §13.)

## 8. Forward epilogue SASS analysis (B0)

`fwd_mma_kernel<Mean, ReLU, 64, MT=2, NW=16, TRACK=0, P_INT=0, packed>`, one fully-valid 64-neuron
unit × 32 tokens per loop iteration (the loop body *is* the executed path here):

| Item | Baseline (FE=0) | chunk4 (FE=1) |
|---|---:|---:|
| Instructions / unit | 255 | 245 |
| HMMA.1688 | 16 | 16 |
| FMNMX (ReLU) | 64 | 64 |
| FADD (pooling sum) | 64 | 64 |
| MOV | 60 (44 C-operand, 8 parameter-ring copy, 8 other) | 52 (36 C-operand, 8, 8) |
| Bias unpack (SHF/IMAD.U32/LOP3 of bf16x2) | 16 | 16 |
| Predicates / selects | 0 (fully valid units take the unmasked path) | 0 |
| LDG (next unit's parameters) / LDSM | 5 / 1 | 5 / 1 |
| SHFL / local memory in the loop | 0 / 0 | 0 / 0 |

Findings:

* **The epilogue is already minimal.**  64 values per lane per unit cost exactly 64 FMNMX + 64 FADD.
  NVCC has removed the `d[][][] → v[]` re-arrangement and the unused `PoolState` fields (`m, t, inv,
  id` are never touched in the loop).  There is no single-instruction fused `s += max(z, 0)` on
  sm_86 (FP32 has no packed-pair add on Ampere; a predicated FADD still needs an FSETP), so B1
  "direct fragment accumulation" and B3 "specialised Mean state" have nothing left to remove in
  the hot loop — not implemented, per the stop conditions.
* **The bias C quads are the visible overhead**: per n8 tile, 2 unpack instructions + ~5.5 MOVs build
  two quads `{bx, by, bx, by}` (one per 16-token half), immediately before the HMMAs that read them
  (excerpt in `annotated_sass_excerpts.txt`).
* **Final reduction**: 8 SHFL (sum only; the shuffles of unused fields were eliminated), 16 STS
  (12 of them store the constant `m, t, id` of a Mean state), ≤ 17 LDS, 2 STG per CTA.  It runs once
  per CTA after ~96 units × 255 instructions per warp (≈ 0.6 % of the CTA's instructions), so B4 was
  deprioritised as instructed.

### 8.1 Forward ablations (measured, WRONG results; `SP_ABLATE_FWD`, `abl_fwd_*`, `ntc_*`)

| Build (H = 100M) | Forward kernel | Δ vs base | Static loop instr |
|---|---:|---:|---:|
| base | 298.9 ms | — | 255 |
| `=4`: C = 0, bias still loaded and unpacked (+16 FADD to keep it live) | 280.1 ms | −19 | 227 |
| `=2`: C = 0, no bias (bias loads, unpack and C MOVs gone) | 251.8 ms | −47 | — |
| `=3`: as 2 + one FADD per MMA instead of 4 FMNMX + 4 FADD | 203.6 ms | −95 | — |

The forward is not tensor-bound (16 HMMA.1688 × 16 tensor cycles = 256 cycles per unit per SMSP would
put it near 185 ms at these clocks).  Building the C quads costs ≥ 19 ms (`=4` still pays 16 extra
FADD), loading and unpacking the bias another ~28 ms, the ReLU + accumulate epilogue ~48 ms.  With
only 4 warps per SMSP to hide dependent chains, both issue slots and latency matter.  Removing moves
that do not feed an HMMA does not help (ping-pong, §9); the change that helps (chunk4, §9) is the
one that gives ptxas room to build the C quads earlier — consistent with a latency explanation,
but unverified without stall-reason counters.

## 9. Forward direct-fragment, chunking and scheduling variants (B1 / B2 / P4)

| Variant | Change | Static instr / MOV / regs | Forward (H = 100M) | Verdict |
|---|---|---|---:|---|
| base | NTC = 2 n8 tiles per epilogue chunk | 255 / 60 / 66 | 301.9 (s2), 298.9 (ntc) | — |
| `SP_FWD_EPI=1` | one shared C fragment per n-tile, second token half in place | **SASS identical to base** | — | no effect (ptxas) |
| `SP_FWD_EPI=2` | ping-pong parameter buffers instead of the ring-slot copy (2 units / iteration) | 238 / 44 / 75 per unit | **306.7 (+4.8, 4/4 rounds slower)** | rejected |
| **`SP_FWD_NTC=4`** | **4 n8 tiles per epilogue chunk** | **245 / 52 / 70** | **289.1 (−9.8, both rounds)** | **retained** |
| `SP_FWD_NTC=8` | whole unit per chunk | 254 / 60 / 103 | 298.0 (±0) | rejected (register growth, scheduler gives the MOVs back) |

The ping-pong result is the instructive negative: 17 fewer instructions per unit, but the removed
copies were not on a dependency chain into an HMMA, and the two-unit loop body changes where ptxas
places the next unit's loads (a plausible but unverified explanation without stall counters); it
is consistently 1.6 % slower.

**Accumulation order (§7.1 of the brief).**  `PoolState::update` folds the `2·NTC` values of a chunk
in n-tile-then-column order and chunks follow in n-tile order, so for every row state the sequence
of FP32 additions over a block is the same for NTC = 2 and NTC = 4 (and for the masked tail block,
whose validity flags are per slot).  No reassociation, no tree reduction: outputs are bitwise
identical, verified in §14.

## 10. ReLU and accumulation experiments (B2)

* FMNMX + FADD per element is the minimum on sm_86 for `s += max(z, 0)` with the current semantics;
  a select or predicated add needs an FSETP as well (same count).  No candidate was implemented.
* `s = fmaf(z, float(z > 0), s)` is not equivalent and saves nothing: it still needs the compare
  (FSETP/FSEL + FFMA, same count), and it changes the special values — z = −inf gives
  −inf · 0 = NaN and z = NaN gives NaN, where `fmaxf` returns 0 for both (+inf, ±0, subnormals and
  finite values agree: `z · 1 + s` is a single rounding, like the FADD).  Not pursued.
* Interleaving / scheduling: the only effective scheduling lever found is the larger chunk (§9),
  which gives ptxas independent MMAs to place the C-fragment construction against.

## 11. Mean pooling reduction (B3 / B4)

Not changed: NVCC already eliminates every unused `PoolState` field in the loop and the shuffles of
unused fields in the quad merge; what remains (12 constant STS + their LDS per CTA, < 1 % of the
kernel) is below measurement resolution, and changing the shared-memory layout would also alter
the compact-state output contract (`st_m` is written for Mean).

## 12. Full A/B results and configuration regression matrix

**Primary configuration, controlled result**: the 10 paired rounds in §1 (`final_*`, `confirm_*`,
`paired_deltas_final_confirm.json`): forward −9.0 ms (10/10 pairs faster), step −10.0 ms
(9/10 pairs faster; the one slower pair +4.6 ms), backward and optimizer unchanged, peak memory
unchanged.  Per-round GPU snapshots (clock, temperature, power) are in every JSON and in the
`gpu after:` lines of `final_ab.log` / `confirm_ab.log`.

### 12.1 Configuration regression matrix

2 interleaved rounds per configuration (`run_matrix_phase3.sh`, `matrix_*.json`, `matrix_summary.json`,
`matrix.log`), baseline = fcc363b worktree, new = this branch at the time of the run (auto = chunk4
for every packed Mean at C = 64).  Mean(identity) is not selectable in the baseline commit's
benchmark; its control is this branch with `--fwd-epilogue baseline` (same kernels as fcc363b).
Max has no native backward (`dw = none`).

| H | C | M | Pooling | Precision | Forward epilogue (new) | Forward base → new | dX base → new | dW base → new | Step base → new | Step paired Δ (r1, r2) | Peak alloc |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1M | 64 | 8192 | Mean(ReLU) | split | chunk4 | 23.0 → 22.6 | 11.4 → 11.3 | 7.8 → 7.9 | 45.2 → 44.2 | -0.9, -1.1 | 0.31 → 0.31 GiB |
| 10M | 64 | 8192 | Mean(ReLU) | split | chunk4 | 42.1 → 41.7 | 77.9 → 77.8 | 82.0 → 82.8 | 206.8 → 205.7 | -0.6, -1.5 | 1.08 → 1.08 GiB |
| 100M | 64 | 8192 | Mean(ReLU) | split | chunk4 | 304.6 → 298.0 | 730.1 → 741.8 | 741.7 → 753.3 | 1801.1 → 1816.2 | -6.9, +37.0 | 8.64 → 8.64 GiB |
| 100M | 64 | 8192 | Mean(ReLU) | bf16 | chunk4 | 295.6 → 284.4 | 617.0 → 620.1 | 546.9 → 546.5 | 1483.7 → 1473.7 | -6.0, -13.8 | 8.64 → 8.64 GiB |
| 100M | 64 | 1024 | Mean(ReLU) | split | chunk4 | 36.5 → 35.2 | 87.5 → 88.1 | 95.5 → 95.7 | 241.1 → 241.3 | +0.4, -0.0 | 8.45 → 8.45 GiB |
| 100M | 8 | 8192 | Mean(ReLU) | split | baseline | 385.3 → 386.0 | 2314.5 → 2315.4 | 2062.0 → 2064.1 | 4786.6 → 4788.7 | +4.4, -0.1 | 8.79 → 8.79 GiB |
| 100M | 64 | 8192 | Mean(identity) | split | chunk4 | 250.6 → 256.6 | 441.5 → 440.7 | 422.1 → 422.5 | 1138.6 → 1143.3 | +3.8, +5.7 | 8.64 → 8.64 GiB |
| 100M | 64 | 8192 | LogMeanExp | split | baseline | 594.6 → 595.5 | 878.7 → 880.8 | 848.0 → 846.4 | 2344.7 → 2347.8 | +3.7, +2.5 | 8.70 → 8.70 GiB |
| 100M | 64 | 8192 | Softmax-weighted | split | baseline | 649.9 → 650.9 | 1033.2 → 1031.9 | 1077.1 → 1077.3 | 2784.0 → 2783.8 | -0.7, +0.3 | 8.73 → 8.73 GiB |
| 100M | 64 | 8192 | Max | split | baseline | 375.4 → 375.3 | — → — | — → — | 408.3 → 408.0 | -0.0, -0.5 | 8.65 → 8.65 GiB |

* **Mean(ReLU), C = 64**: forward faster at every size (−0.4 ms at 1M, −1.3 ms at M = 1024, −6.6
  ms at 100M split in this run, −11 ms with BF16 gradients).  The 100M split pair r2 shows step
  +37 ms with the forward −2.9 ms: the unchanged dX and dW kernels ran +21 / +22 ms slower in the
  "new" run of that pair (the before/after clock snapshots in the JSON do not explain it; they are
  instantaneous readings).  It is run-to-run variance of unchanged code; the 10-pair A/B in §1
  (9 of 10 pairs faster, forward faster in 10 of 10) is the controlled result.
* **Mean(identity), C = 64: chunk4 is slower** — forward 250.6 → 256.5 ms in both rounds (+2.4 %),
  step +3.8 / +5.7 ms.  Without the FMNMX the epilogue is short enough that the larger chunk only
  adds register pressure.  **`auto` was therefore restricted to Mean(ReLU)** (commit after the
  matrix; `test_auto_selection_by_activation`); explicit `"chunk4"` remains available and exact.
* **C = 8, LogMeanExp, softmax-weighted, Max**: forward epilogue `baseline` (not eligible / not
  Mean); all phases within ±0.2 % — no regression from the shared dispatch or the new `FwdArgs` field.
* Peak allocated memory is unchanged in every configuration (no new buffers).
* Mode A is validated by `test_mode_a_unchanged` (Mode A never reaches the packed Mode-B forward).


## 13. Attribution matrix

Interleaved, 4 rounds, H = 100M, C = 64, M = 8192, Mean(ReLU), split (`final_*.json`).  Each row is a
real build: baseline = fcc363b worktree; dW-only = screening tree 3db68de with `-DSP_DW_MMA=2`
(the best exact dW variant; no dW change was promoted); forward-only = this branch's defaults;
combined = screening tree with `-DSP_DW_MMA=2 -DSP_FWD_NTC=4`; control = this branch with
`--fwd-epilogue baseline`.

| Variant | dW MMA change | Forward change | Forward | dX | dW | Backward | Step |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | off | off | 298.3 | 712.6 | 726.3 | 1439.2 | 1760.5 |
| dW-only | on (A2) | off | 298.3 | 713.6 | 726.7 | 1441.7 | 1760.4 |
| Forward-only (this branch) | off | on | 289.0 | 712.5 | 727.1 | 1440.5 | 1750.8 |
| Combined | on (A2) | on | 288.9 | 713.0 | 727.0 | 1441.0 | 1750.9 |
| Control: this branch, `baseline` epilogue | off | off | 299.6 | 714.0 | 727.8 | 1441.5 | 1763.0 |

(Medians of the per-round medians, ms; per-round ranges in the JSON and `final_ab.log`.)  The two
changes are not additive because the dW change contributes nothing; the control row reproduces the
baseline within 1.3 ms of forward time (it differs from the baseline commit only by the new
`FwdArgs` field and dispatch code).

## 14. Correctness results

**Test suite**: `pytest` over `sparse_pooling/tests`: **1779 passed, 1 skipped** (the skip is the
pre-existing GELU case, which runs on Triton).  1669 tests existed at fcc363b; 110 are new in
`tests/test_mma_forward.py`:

| Test | Cases | Checks |
|---|---:|---|
| `test_chunk4_matches_baseline_bitwise` | 80 | Mean(ReLU) and Mean(identity) × split/bf16 × block_m 16/32 (MT = 1/2) × C = 32/64 × 5 shapes (ragged H = 20003/70001/3001, token tails B = 8/33/40/64/136, D = 64/256/1024): forward output, dW, db **bitwise**; dX rel. error < 1e-6 (atomic order) |
| `test_chunk4_against_reference` | 4 | PyTorch reference, existing tolerances (y, db 2e-5; dX/dW 2e-5 split, 5e-3 bf16) |
| `test_relu_boundary_signed_zero_and_rounding` | 2 | z exactly 0 (x = 0, bias 0), ±0, subnormal products (flushed), cancelling inputs a few ULP from 0, small bias: bitwise |
| `test_nonfinite_and_extreme_inputs` | 8 | ±inf, NaN, 3e38 and ×1e30 inputs: bitwise including NaN positions |
| `test_colliding_indices_and_accumulation` | 1 | 12 distinct features for all indices; two backward calls accumulating into `.grad` |
| `test_training_trajectory_identical` | 2 | 3 fused-AdamW steps × 2 micro-batches (`main_grad`), split and bf16: identical parameters and identical forward outputs at every step |
| `test_other_poolings_unaffected_and_fallback_reported` | 4 | Max / LME / softmax-weighted / power mean: `auto` keeps the baseline without a note; explicit `chunk4` warns once and records `fwd_mean_epilogue_fallback`; outputs unchanged |
| `test_auto_selection_by_block` | 4 | C = 8/16/32 → baseline, C = 64 → chunk4; explicit request at C < 32 warns |
| `test_auto_selection_by_activation` | 2 | Mean(ReLU) → chunk4, Mean(identity) → baseline under `auto`; explicit chunk4 honoured |
| `test_mode_a_unchanged` | 3 | Mode A (independent indices) output unchanged for every setting |

Numerical differences of the retained change: **none** — forward outputs, pooling statistics,
dW, db and the optimizer trajectory are bitwise identical to the baseline epilogue in every test;
dX differs only by the order of its global FP32 atomics (as before).  The inexact diagnostic of §6.2 is not reachable from any plan option.

## 15. Sanitizer results

`results/mma_forward/sanitizer_full_phase3.log` (`run_sanitizer.sh`: every kernel family plus the
new forward cases, print limit 20) and `sanitizer_native_phase3.log` (`SAN_ONLY=phase3`: the 7
chunk4/baseline forward+backward cases only, C = 32/64, block_m 16/32, ragged H, token tails;
print limit unlimited):

| Tool | Full workload | Phase-3 native cases |
|---|---|---|
| memcheck | 0 errors | 0 errors |
| initcheck | 0 errors | 0 errors |
| synccheck | 0 errors | 0 errors |
| racecheck | 20 reports, **all** in the Triton `bwd_dense_kernel` (`triton_fused.py:441`), the pre-existing reports documented since phase 1 | **0 hazards** (0 errors, 0 warnings) |

No new hazard.

## 16. Failed experiments and process notes

* dW A1/A2/A4 (exact C-operand layouts): no SASS change of consequence, no measurable effect.
* dW A3 (C = 0 + FADD): inexact in general (micro-test), not faster.  Kept only as `SP_ABLATE_DW=5`.
* Forward chunk4 under `auto` for Mean(identity): measured +6 ms forward (§12.1); `auto` restricted
  to Mean(ReLU).
* Forward `SP_FWD_EPI=1` (in-place shared C): identical SASS.  `SP_FWD_EPI=2` (ping-pong): +4.8 ms.
* Forward NTC = 8: 103 registers, no gain.
* B1/B3/B4: not implemented — the SASS showed nothing left to remove (§8, §11).
* Process: two screening builds were started while a header was being edited; their objects could
  have been compiled from newer text than their hash, so all objects and libraries produced after
  that edit were deleted and rebuilt before any measurement.  Later screens ran from a separate
  worktree.  The variant code of 3db68de was removed from the production tree afterwards (only the
  ablations remain); 3db68de reproduces every screening build.

## 17. Remaining bottlenecks

* **dX ≈ 713 ms and dW ≈ 727 ms (82 % of the step)** are unchanged by this phase.  dW's bias C path
  is bounded at ≈ 20 ms; the rest of its loop is the dZ split/pack chain (FSETP → FSEL → F2FP →
  PRMT → FADD → F2FP, 6 dependent instructions per value pair) feeding 8 HMMA.16816 per block.
* **Forward ≈ 290 ms**: still far above its ~185 ms tensor floor.  Remaining overhead: bias load +
  unpack + C quads (≈ 40 ms by ablation), epilogue latency (≈ 48 ms), parameter streaming (≈ 13 %
  in phase 1).

## 18. Recommended next steps

1. **Bias pre-expanded in the packed parameter layout** (the forward reads `bf16x2` and unpacks
   per n-tile): storing the forward's bias already as the FP32 quad it needs would trade 16 unpack
   instructions and the dependent MOVs for wider loads — but it doubles bias bytes (+200 MB at
   H = 100M, against this project's low-memory constraint) and is a parameter-format change;
   evaluate only as an opt-in layout, if at all.
2. **Nsight Compute with admin rights** to confirm the latency attribution (stall reasons
   "wait"/"short scoreboard" on the HMMA C operands) before any further forward restructuring.
3. **dW dZ chain**: test a dZ hi/lo formulation that shortens the 6-instruction dependency chain
   (e.g. split via integer masking of the FP32 bits for the hi part) — the dominant non-MMA work
   left in dW.

## Reproduction

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
git worktree add ../GL_base fcc363b; git worktree add ../GL_screen 3db68de
# MOV attribution of any build
$PY sparse_pooling/benchmarks/sass_attrib.py --kernel 'k_dw_idx<2, 1, 64, 8, 2, 2, 1, 0, 2, 1>' \
    --kernel 'fwd_mma_kernel<2, 1, 64, 2, 16, 0, 0, 1, 1>'
# stage-1 screening (H = 1M, 10M) of a compile-time variant (screening tree)
(cd ../GL_screen && bash sparse_pooling/benchmarks/screen_variant.sh $PY dw2 "-DSP_DW_MMA=2")
# full-scale interleaved A/B of compile-time variants / ablations
bash sparse_pooling/benchmarks/run_ab_phase3.sh $PY abl_fwd 2 "base:" "noC:-DSP_ABLATE_FWD=2"
# final attribution matrix and confirmation
bash sparse_pooling/benchmarks/run_ab_phase3_final.sh $PY final 4 "baseline|../GL_base||" \
    "dw_only|../GL_screen|-DSP_DW_MMA=2|" "fwd_only|.||" \
    "combined|../GL_screen|-DSP_DW_MMA=2 -DSP_FWD_NTC=4|" "fwd_base|.||--fwd-epilogue baseline"
bash sparse_pooling/benchmarks/run_matrix_phase3.sh $PY ../GL_base 2
# numerics of the accumulator-initialisation variant
$PY sparse_pooling/benchmarks/numerics_dw_bias.py --out base.pt
SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_DW=5 $PY sparse_pooling/benchmarks/numerics_dw_bias.py --out diag.pt
$PY sparse_pooling/benchmarks/numerics_dw_bias.py --compare base.pt diag.pt
nvcc -arch=sm_86 -O2 sparse_pooling/benchmarks/mma_c_rounding.cu -o /tmp/mma_c && /tmp/mma_c
# tests and sanitizers
(cd sparse_pooling/tests && $PY -m pytest -q)
bash sparse_pooling/benchmarks/run_sanitizer.sh $PY
```
