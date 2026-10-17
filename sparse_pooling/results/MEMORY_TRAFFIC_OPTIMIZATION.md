# Memory traffic analysis and backward kernel optimization (RTX 3090)

Branch `dw-indexed-staging`.  All timings are medians of CUDA-event intervals recorded at phase
boundaries inside the native launcher (`benchmarks/bench_kernels.py`); "measured" means one of
those, "model" means the source-derived request count of `benchmarks/traffic_model.py`, and
"computed" means derived from compiler output.  Hardware performance counters were **not**
available (§2.3), so no number below is a measured DRAM, L2 or occupancy counter.

Primary configuration: H = 100,000,000, C = 64 (Mode B, fragment-packed), D = G = 1024, fan-in 8,
M = 8192 tokens, BF16 parameters/inputs, FP32 accumulation, Mean(ReLU), `grad_precision="split"`,
FP32 `main_grad`, fused AdamW with BF16 moments.

## 1. Executive summary

| | Original | Optimized | Speedup |
|---|---:|---:|---:|
| Forward kernel | 300.1 ms | 300.0 ms | 1.00× (unchanged) |
| dX kernel (`k_dx`) | 939.3 ms | 806.2 ms | **1.165×** |
| dW kernel (`k_dw` → `k_dw_idx`) | 1096.0 ms | 774.6 ms | **1.415×** |
| Backward total | 2035.7 ms | 1581.3 ms | 1.287× |
| Fused AdamW | 21.4 ms | 21.4 ms | 1.00× |
| **Full training step** | **2357.4 ms** | **1903.8 ms** | **1.238×** |
| Peak allocated / reserved | 8.64 / 9.88 GiB | 8.64 / 9.88 GiB | unchanged |

Interleaved A/B, 4 rounds in alternating order, each round 5 independent repetitions × 3 steps;
the per-round step medians span 2353–2414 ms (original) and 1899–1912 ms (optimized), so the
ranges do not overlap.  The repository's original end-to-end benchmark (`bench_end_to_end.py`,
its own timing method) agrees: Mean(ReLU) step 2361 → 1995 ms, LogMeanExp 2910 → 2497 ms,
softmax-weighted 3279 → 2878 ms, power mean 2800 → 2369 ms (`traffic/e2e_check_optimized.json`
vs `e2e_final_100M_C64.json`).

Two changes, both C = 64 defaults now, both with the original path kept selectable:

1. **Indexed, warp-local input staging for dW (`k_dw_idx`).**  Each warp stages only the
   8 index rows of each of its work units (192 GiB of logical staging instead of 768 GiB at
   H = 100M) into a private `cp.async` ring, synchronising with `__syncwarp` instead of a CTA
   barrier.  It issues the same MMAs on the same operands in the same order, so dW/db are
   **bitwise identical** to the original kernel.  The smaller slab (8–16 KiB instead of 64 KiB)
   also makes a register-capped 2-CTA/SM tiling possible, which is the default for Mean and
   LogMeanExp.  Ablations attribute the gain to the staging copies and their per-tile CTA barrier
   (§6, §8); the 75 % byte reduction did **not** give a 4× kernel speedup: dW is now within
   ~1.35× of the measured tensor-core rate for its MMA count (analytical, §5.3).
2. **32-warp `k_dx` CTAs.**  `k_dx` is held at 1 CTA/SM by its 96 KiB shared-memory slab; at
   64 registers, 32 warps (1024 threads) exactly fill the register file and hide more latency
   than 16.  Same kernel, new instantiation; dX differs from the original only by the order of
   its global FP32 atomics (relative difference ~3e-8).

The indexed-staging hypothesis was **successful**, though for a different mechanism than the
byte count suggests: removing the redundant copies mattered mostly because each copy batch
was followed by a CTA-wide barrier, and because the reduced slab frees the residency that the
64 KiB full slab blocked.

## 2. Baseline and methodology

### 2.1 Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (GA102, sm_86, 82 SMs), 24 GiB, VBIOS 94.02.59.00.5C |
| Power limit / max clocks | 370 W (default) / 2115 MHz SM, 9751 MHz memory |
| Driver / CUDA (driver API) | 550.67 / 12.4 |
| nvcc | 12.1 (V12.1.66), flags `-O3 -arch=sm_86 --use_fast_math -lineinfo -std=c++17` |
| PyTorch / CUDA runtime / Triton | 2.7.0+cu128 / 12.8 / 3.3.0 |
| OS kernel | Linux 6.8.0-138-generic |
| Python env | `/home/evga/.conda/envs/env_isaaclab` (Python 3.11) |

GPU temperature, SM/memory clock and power were recorded with `nvidia-smi` before and after
every configuration (`gpu_before` / `gpu_after` in each JSON record).

### 2.2 Per-kernel profiler

`kernels/cuda_ops.py` gained an optional `PHASE_EVENTS` list; when a profiler installs it,
`_mark(name)` records a CUDA event on the current stream at each phase boundary (input
transpose, forward kernel, finalize, coefficient generation, backward transpose, dX buffer
init, `k_dx`, `k_dw`).  `bench_kernels.py` adds step-level marks (forward end, backward end,
optimizer end).  The computation is identical with and without the marks.  Each configuration
runs 2 warm-up steps (nvcc/Triton compilation, lazy Adam state, allocator growth), then
`--reps` independent repetitions of `--iters` full steps; reported numbers are medians over all
steps, with the spread of the per-repetition medians in the JSON.  An event interval also
contains any GPU idle time between launches; at H ≥ 10M this is negligible, at H = 1M the
smallest phases are upper bounds.

### 2.3 Hardware counters: not available

`/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1` and `sudo` requires a password,
so Nsight Compute counter collection fails with `ERR_NVGPUCTRPERM`.  Achieved occupancy, Tensor
Core utilisation, L1/L2/DRAM throughput, bank-conflict counters and stall reasons were **not
measured**.  Fallbacks used instead: CUDA-event timing, `cuobjdump -res-usage` / `ptxas -v`
register and spill counts, theoretical residency from the CUDA occupancy rules, static SASS
instruction counts, and correctness-breaking attribution ablations compiled into a separate
library.  To collect counters later: `sudo -E bash sparse_pooling/benchmarks/profile_ncu.sh
<python>`, or set `NVreg_RestrictProfilingToAdminUsers=0` and reload the driver.

### 2.4 Baseline (original implementation, this machine, today)

The historical reference in the task (forward ≈ 310 ms, backward ≈ 2217 ms, step ≈ 2556 ms)
was measured while the card was degraded (20 GiB, 744 GB/s, 300 W cap).  Re-measured today with
the original code paths (`--dw-staging full --dx-warps 16`):

| H | C | Grad | Fwd kernel | dX kernel | dW kernel | Bwd total | Optimizer | Step | Peak alloc / reserved |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1M | 8 | split | 24.57 | 26.21 | 22.66 | 49.81 | 0.23 | 74.94 | 0.31 / 0.34 GiB |
| 1M | 64 | split | 22.50 | 12.16 | 16.59 | 29.68 | 0.24 | 52.85 | 0.31 / 0.32 GiB |
| 1M | 64 | bf16 | 22.42 | 10.10 | 14.03 | 25.10 | 0.24 | 48.13 | 0.31 / 0.32 GiB |
| 10M | 8 | split | 59.74 | 241.91 | 228.72 | 472.61 | 2.15 | 534.57 | 1.08 / 1.33 GiB |
| 10M | 64 | split | 40.87 | 93.77 | 114.73 | 209.51 | 2.20 | 252.85 | 1.08 / 1.13 GiB |
| 10M | 64 | bf16 | 40.68 | 72.62 | 89.44 | 163.34 | 2.19 | 206.53 | 1.08 / 1.13 GiB |
| 100M | 8 | split | 385.44 | 2350.06 | 2097.64 | 4444.06 | 21.37 | 4850.67 | 8.79 / 10.38 GiB |
| 100M | 64 | split | 300.37 | 941.59 | 1096.73 | 2040.02 | 21.38 | 2361.20 | 8.64 / 9.88 GiB |
| 100M | 64 | bf16 | 302.55 | 698.90 | 889.80 | 1589.34 | 21.39 | 1913.26 | 8.64 / 9.88 GiB |

(ms; Mean(ReLU); 5 repetitions × 3 steps.)  Other phases at H = 100M, C = 64: input transpose
≈ 0.3 ms, forward finalize ≈ 0.2 ms, coefficients ≈ 0.1 ms, dX FP32 → BF16 conversion and
autograd bookkeeping < 1 ms (`results/traffic/TABLES.md`).  **dW was the largest single kernel
(54 % of backward)**, dX the second.  Max (sparse winner-event backward on Triton) is a control:
backward 10.6–10.7 ms at C = 8 and 64, unchanged by this work (step 493.4 → 493.1 ms and
407.4 → 408.4 ms).

## 3. Kernel mapping (confirmed from source and compiler output)

Resources from `benchmarks/kernel_resources.py` (cuobjdump of the built library; residency
computed with the sm_86 rules: 64K registers per SM in 256-register warp granules, 100 KiB shared
memory per SM with 1 KiB reserved per CTA, ≤ 48 warps, ≤ 16 CTAs).  Mean(ReLU), C = 64:

| Kernel | Threads (warps) | Tokens / CTA | Work per CTA | CTAs at H = 100M | Smem / CTA | Regs | Spills | CTAs/SM (limiter) |
|---|---:|---:|---|---:|---:|---:|---:|---|
| forward `fwd_mma_kernel<…,C=64,MT=2,NW=16,packed>` | 512 (16) | 32 | 1 unit = 1 group, all 1,528 blocks | 256 × 1024 = **262,144** | 64 KiB | 68 | 0 | 1 (registers and smem) |
| `k_dx` (original) | 512 (16) | 16 | 128 units (groups) | 512 × 8 = **4,096** | 96 KiB | 64 | 0 | 1 (smem) |
| `k_dx` (optimized, C = 64) | 1024 (32) | 16 | 128 units | 4,096 | 96 KiB | 64 | 0 | 1 (smem / registers) |
| `k_dw` 8 × 4 (original default) | 256 (8) | all 8192, 16 per tile | 32 work units = 32 blocks = 2,048 neurons | 48 × 1024 = **49,152** | 64 KiB | 255 | 0 | 1 (registers) |
| `k_dw` 8 × 2 / 16 × 1 | 256 / 512 | all | 16 work units | 98,304 | 64 KiB | 141–145 / 103–128 | 0 | 1 (registers) |
| `k_dw_idx` 8 × 4, 2-stage | 256 (8) | all | 32 work units | 49,152 | 16 KiB | 255 | 0 | 1 (registers) |
| `k_dw_idx` 8 × 2, `__launch_bounds__(256,2)` (cfg 3) | 256 (8) | all | 16 work units | 98,304 | 8 KiB | 128 | 8–16 B stack (Mean) | **2** (registers) |

The three launch shapes expected in the task (262,144 / 4,096 / 49,152 CTAs) are confirmed.
One refinement: the packed layout rounds each group to a multiple of 4 blocks, so a group has
1,528 blocks (100,139,008 slots, 0.14 % padding) and dW has ⌈1528/32⌉ = 48 chunks per unit.

Strategies (from the code, `kernels/cuda/*.cuh`):

* **Forward**: parameters arrive as 16-byte `ld.global.nc` per lane per fragment-packed unit
  (2 × weights, 2 × bias uint4 + int16 indices at C = 64), one unit prefetched per warp
  (`QPD = 1`); X tile staged once per CTA with `cp.async` (16 B chunks, swizzled `[1024][32]`
  slab), gathered with `ldmatrix.x4.trans`; `mma.m16n8k8` with the bias as accumulator input;
  pooling state in registers, quad shuffles + shared-memory merge across warps, one compact
  state per (token, unit) written.  No atomics.
* **`k_dx`**: X tile staged once (`cp.async`, `[1024][16]`), parameters streamed per warp with a
  one-unit register double buffer; z recomputed (8 × m16n8k8), dZ from c1 (Mean), `dA = dZ Wᵀ`
  (4 × m16n8k16 per pass, 2 passes for split, Wᵀ by `movmatrix`); scatter into the
  `[1024][16]` FP32 slab with two 64-bit CAS loops per lane per block (`atomic_add_f32x2`, tokens
  t and t + 8 interleaved — the existing pairing); one FP32 `atomicAdd` to global dX per
  non-zero (feature, token) per CTA.
* **`k_dw`**: the warp's WPW work units, their transposed bias and accumulators stay in
  registers for the whole kernel; X staged per 16-token tile into a double-buffered full
  `[1024][16]` slab (`cp.async` prefetch of tile t + 1 by all threads, `cp.async.wait_all` +
  `__syncthreads` per tile); per block 8 × m16n8k8 recompute (zᵀ = W X_gᵀ) and 4 m16n8k16 per pass
  (dWᵀ = dZᵀ X_g); write-back is a plain read-modify-write of `main_grad` (exclusive ownership,
  no atomics, deterministic).

## 4. Forward traffic model (H = 100M, C = 64)

Logical requests (level A), `traffic_model.py`:

| Stream | Per CTA | Total per forward | Distinct data |
|---|---:|---:|---:|
| Weights + bias + indices (`ld.global.nc`) | 1.70 GiB / 1024 = 1.70 MiB (the unit's records) | **435.7 GiB** (every one of 256 token tiles reads every record) | 1.83 GB |
| X slab (`cp.async`) | 64 KiB | 16.0 GiB | 16 MiB |
| Pooling state writes | 32 tokens × 4 B | 32 MiB | — |

L2/DRAM (analysis, not measured): the grid is token-tile-fastest, so the 256 CTAs of one group
are launched consecutively; at 1 CTA/SM, 82 resident CTAs walk the same group's 1.8 MB of
parameters in near lock-step, so each 128-byte parameter line is fetched from DRAM roughly once
per group pass and served from L2 to the other CTAs.  DRAM parameter traffic is therefore
expected near the 1.8–6 GB range (1–3 passes per group), i.e. ≤ 7 ms at 819 GB/s, while the
435.7 GiB of L2→SM requests in 300 ms correspond to ~1.6 TB/s of L2 delivery.  Measured
attribution: making every parameter load a cache hit (ablation `SP_ABLATE_FWD=1`) shortens the
forward from 305 to 264 ms (−13 %), so parameter streaming is a real but minor cost; the rest is
MMA + epilogue work.

## 5. dX and dW traffic models (H = 100M, C = 64)

### 5.1 dX

| Quantity | Model | Notes |
|---|---:|---|
| Projection recomputations | 1 per (token, neuron) | 6.41e9 m16n8k8 (13.1 TFLOP) |
| Gradient products | 6.41e9 m16n8k16 (split: 2 passes) | 26.2 TFLOP |
| Parameter requests | **871.4 GiB** | every one of 512 16-token tiles re-reads all records |
| X staging | 128 MiB | once per CTA |
| Shared-memory 64-bit CAS | **5.13e10** (2 per lane per block per tile) | CAS retries not measurable without counters |
| Global FP32 atomics (flush) | 6.7e7 (≤ 1 per (feature, token) per CTA, 256 MiB) | native L2 atomics |

Collision structure (from the code): in one CAS instruction lane L updates (feature
`idx[2(L&3)+c]`, token `L>>2`); index records are sampled distinct, so **no two lanes of an
instruction ever target the same word** — warp-local aggregation has nothing to combine, and
retries can only come from other warps of the CTA touching the same (feature, token) word
between a lane's read and its CAS.  Bank conflicts: `dxs_off(f, t)` places a feature at bank
offset 0 or 16 by its parity only; simulating real index draws gives 2.75 shared-memory
wavefronts per 16-lane phase (ideal 1.0); an XOR swizzle of the token slot with feature bits 1–3
would give 2.13 (the optimum for any fixed per-feature placement with 4 tokens per phase).

### 5.2 dW (verifying the 768 GiB estimate)

| Config | CTAs | Index rows / CTA | X staging, full | X staging, indexed | Smem full / indexed (2-stage) |
|---|---:|---:|---:|---:|---:|
| 8 warps × 4 WU (original default) | 49,152 | 256 | **768.0 GiB** | **192.0 GiB** | 64 / 16 KiB |
| 8 × 2 and 16 × 1 | 98,304 | 128 | 1536.0 GiB | 192.0 GiB | 64 / 8 KiB |

The code matches the task's model exactly: every k_dw CTA stages all 1024 feature rows of every
16-token tile (2048 × 16-byte `cp.async` per tile) although its warps read at most 256 of them.
Other dW streams: parameters 1.70 GiB (read once, exclusive ownership), `main_grad` FP32 RMW
6.71 GiB, coefficient loads 96 GiB of logical 4-byte requests (identical addresses across the 8
warps → L1-resident; removing them saves 43 ms, §8).

**Indexed staging gives no saving at C = 8**: there a work unit is 4 blocks, so 32 work units
× 4 × 8 = 1,024 rows = all of D.  This is why the new kernel is restricted to C = 64.

### 5.3 What bounds dW (analysis)

dW issues 6.41e9 m16n8k8 (13.1 TFLOP) + 6.41e9 split-precision m16n8k16 (26.2 TFLOP) = 39.3
TFLOP of MMA work.  Against the 69 TFLOPS measured for `mma.m16n8k8` BF16→FP32 on this GPU
(`REPORT.md` §1, first campaign) that is a 0.57 s floor; original dW (1.10 s) ran at ~36
effective TFLOPS, optimized dW (0.775 s) at ~51 (~74 % of the measured MMA rate).  Static SASS
of the hot loop: ~3,900 instructions per warp per 16-token tile, of which 128 HMMA and ~1,000
SIMT epilogue instructions (FADD/F2FP/FSEL/FMUL for dZ and the hi/lo split); staging is 8
`LDGSTS` per thread (full) vs 2 per lane (indexed) — the copies are cheap to issue, their cost
is in the memory pipe and the barrier that waits for them (§8).

## 6. Theoretical opportunities considered

| Opportunity | Model / expectation | Outcome |
|---|---|---|
| dW indexed staging | −75 % logical X staging (768 → 192 GiB); smem 64 → 16 KiB | **implemented**, dW 1.415× (§7) |
| dW residency (≤ 128 regs, 2 CTAs/SM) | only possible once the slab shrinks | **implemented** as cfg 3; +4–5 % for Mean/LME, not for SW/power (spills) |
| dX more warps per CTA | 1 CTA/SM by smem; 64 regs × 1024 threads fits | **implemented**, dX 1.165× |
| dX scatter bank swizzle | 2.75 → 2.13 wavefronts per phase | not implemented: ablation bounds the non-atomic scatter cost at ≤ 11 ms (§8) |
| dX warp-local aggregation | no intra-instruction duplicates exist | not applicable (§5.1) |
| dX without CAS (fixed point / reverse CSR) | CAS loop costs ~164 ms (17 %) | not implemented: fixed point weakens FP32 accumulation; reverse CSR needs ~25 MB of reverse indices plus re-recompute per feature owner, no plausible net gain |
| dX 32-token tile (halve parameter re-reads) | 128 KiB FP32 slab + 64 KiB X slab > 99 KiB | infeasible on GA102 |

## 7. Implemented changes

All code on branch `dw-indexed-staging`.

* `kernels/cuda/sparse_pool_bwd.cuh`
  * `k_dw_idx<POOL, ACT, C, NW, WPW, ST, SPLIT, P_INT, MINB>`: per-warp ring of ST compact slabs
    `[R = WPW·UB·8 rows][16 tokens]` with the same `Swz<1>` row layout as the full slab (rows
    4–7 of each 8-row group chunk-flipped, so `ldmatrix` stays conflict-free).  At kernel start
    each lane resolves its `ceil(2R/32)` copy sources by shuffling the int16 indices out of the
    already-loaded work-unit registers (local row r = (i·UB + bu)·8 + k ↔ feature
    `idx[unit i, block bu, k]`); rows of absent work units are zero-filled (`src-size 0`), so a
    garbage index is never dereferenced.  Per tile: stage tile t + ST − 1 into the slot freed by
    the previous iteration, `cp.async.commit_group`, `cp.async.wait_group ST−1`, `__syncwarp`,
    compute, `__syncwarp`.  `dw_block` is reused unchanged by passing the local row as `fidx`.
    Token tails use the existing partial-copy rule (`src-size < 16` zero-fills).
  * The write-back loop moved into `dw_writeback` (shared by both kernels; `k_dw` compiles to the
    same register counts as before).
  * `SP_ABLATE_DX` / `SP_ABLATE_DW` attribution ablations, `#define`d to 0 unless set through
    `SPARSE_POOL_NVCC_EXTRA` (separate build hash/directory; never in the default build).
* `kernels/cuda/sparse_pool_fwd.cuh`: `SP_ABLATE_FWD` (same rules).
* `kernels/cuda/bwd_dispatch.cuh`: kinds 2/3 = indexed dW with a 2-/3-stage ring (C = 64 only,
  feature-major X only, else error −2); dW cfg 3 = 8 × 2 with `__launch_bounds__(256, 2)`;
  `k_dx` with 32 warps.
* `kernels/cuda_ops.py`: `dw_staging()` selects indexed only for packed Mode B, C = 64, a
  feature-major operand and compute capability 8.6; everything else (C ≠ 64, Mode A, canonical
  layout, token-major input, B % 8 ≠ 0, other GPUs) uses the original `k_dw`.  Per-operator
  default tiling `DW_CFG_INDEXED` (cfg 3 for Mean / LME, 8 × 4 for softmax-weighted / power);
  `dx_warps = 32` at C = 64 when `cuda_dx_warps = 0`.  `last_info` reports `dw_staging`,
  `dw_cfg`, `dx_warps`.  Optional `PHASE_EVENTS` marks.
* `config.py`: `PhysicalPlan.cuda_dw_staging = "auto" | "full" | "indexed"` (default `auto`),
  `cuda_dx_warps` default `0` (auto).  **Restore the original implementation with
  `PhysicalPlan(cuda_dw_staging="full", cuda_dx_warps=16)`.**
* No new global buffers, no new atomics, no change to forward, optimizer, gradient accumulation
  or Mode A.

## 8. Before / after performance

### 8.1 Primary configuration (interleaved A/B, §1)

| | Fwd kernel | dX kernel | dW kernel | Bwd total | Optimizer | Step |
|---|---:|---:|---:|---:|---:|---:|
| original (4 rounds) | 300.1 | 939.3 (936.8–960.5) | 1096.0 (1093.4–1125.3) | 2035.7 | 21.4 | 2357.4 (2353.0–2413.5) |
| optimized (4 rounds) | 300.0 | 806.2 (803.3–806.8) | 774.6 (774.2–780.2) | 1581.3 | 21.4 | 1903.8 (1898.6–1911.6) |
| speedup | 1.000× | 1.165× | 1.415× | 1.287× | 1.000× | **1.238×** |

### 8.2 Configuration matrix (one run each, 5 reps × 3 steps; ms)

| Pooling | H | C | Grad | dX orig → opt | dW orig → opt | Step orig → opt | Step speedup |
|---|---:|---:|---|---:|---:|---:|---:|
| Mean(ReLU) | 1M | 8 | split | 26.21 → 26.16 | 22.66 → 22.59 | 74.94 → 74.89 | 1.00× (no change at C = 8) |
| Mean(ReLU) | 1M | 64 | split | 12.16 → 13.68 | 16.59 → 8.22 | 52.85 → 46.10 | 1.15× |
| Mean(ReLU) | 1M | 64 | bf16 | 10.10 → 11.46 | 14.03 → 6.29 | 48.13 → 41.87 | 1.15× |
| Mean(ReLU) | 10M | 8 | split | 241.91 → 242.21 | 228.72 → 230.05 | 534.57 → 538.17 | 0.99× (unchanged path) |
| Mean(ReLU) | 10M | 64 | split | 93.77 → 84.38 | 114.73 → 83.16 | 252.85 → 212.92 | 1.19× |
| Mean(ReLU) | 10M | 64 | bf16 | 72.62 → 61.81 | 89.44 → 64.57 | 206.53 → 172.52 | 1.20× |
| Mean(ReLU) | 100M | 8 | split | 2350.06 → 2348.08 | 2097.64 → 2097.97 | 4850.67 → 4854.71 | 1.00× (unchanged path) |
| Mean(ReLU) | 100M | 64 | split | 941.59 → 807.64 | 1096.73 → 775.05 | 2361.20 → 1904.22 | 1.24× |
| Mean(ReLU) | 100M | 64 | bf16 | 698.90 → 618.16 | 889.80 → 584.92 | 1913.26 → 1518.95 | 1.26× |
| LogMeanExp τ=0.5 | 100M | 64 | split | 1020.03 → 915.78 | 1280.79 → 874.96 | 2914.71 → 2407.11 | 1.21× |
| Softmax-weighted β=1 | 100M | 64 | split | 1145.79 → 1031.48 | 1437.02 → 1074.14 | 3251.48 → 2778.88 | 1.17× |
| Power mean p=2 | 100M | 64 | split | 1054.95 → 897.55 | 1245.58 → 902.90 | 2785.18 → 2294.53 | 1.21× |
| Max (control) | 100M | 8 / 64 | split | — | — | 493.44 → 493.06 / 407.42 → 408.42 | 1.00× |

Forward kernels are unchanged (±1 %).  Peak allocated/reserved memory is identical in every
configuration (e.g. 8.64 / 9.88 GiB at H = 100M, C = 64): the new kernel allocates nothing.
Phase-by-phase (same method as `granule_memory_audit.py`, H = 100M, C = 64, B = 32 × 256, after
a warm-up step; steady state 8.434 GiB), the peak above the phase's starting allocation is
forward +0.141 GiB, backward +0.172 GiB, optimizer 0 — identical for original and optimized.  At H = 1M the 32-warp dX is
1.2–1.5 ms slower (more warps than the 4,096 short CTAs can use), but the step still gains 6–7 ms
from dW; the dX setting is chosen for H ≥ 10M, where it is 9–14 % faster.

### 8.3 Attribution ablations (H = 100M, C = 64, 16-warp dX; wrong results, timing only)

| Variant | dW ms | Δ vs its reference |
|---|---:|---:|
| `k_dw` full slab, 8 × 4 (reference) | 1127 | — |
| … without X staging in the token loop (`SP_ABLATE_DW=1`) | 752 | −375 ms (33 %) |
| … without the per-tile `__syncthreads` (`=2`, copies kept) | 916 | −211 ms |
| … without coefficient loads (`=3`) | 1084 | −43 ms |
| `k_dw_idx` 8 × 4 (reference) | 842 | — |
| … without X staging | 735 | −107 ms |
| `k_dw_idx` cfg 3, 2 CTAs/SM (reference) | 798 | — |
| … without X staging | 690 | −108 ms |
| … without coefficient loads | 744 | −54 ms |

| Variant | dX ms | Δ |
|---|---:|---:|
| `k_dx` 16 warps (reference) | 967 | — |
| racy plain RMW instead of 64-bit CAS (`SP_ABLATE_DX=1`) | 803 | −164 ms (CAS loop) |
| no scatter at all (`=2`) | 792 | −175 ms (CAS + shared accesses) |
| no global flush (`=3`) | 935 | −32 ms (within drift) |
| parameter loads made cache hits (`=4`) | 837 | −130 ms |

Interpretation:
* Full staging + its CTA barrier cost ~375 ms of the original dW; indexed staging recovers
  ~285 ms of it (842 vs 1127 ms at the same tiling).  The barrier alone (211 ms) is a larger
  share than the copy bytes would suggest: a CTA cannot start tile t + 1 until all 8 warps
  finished tile t and the whole 32 KiB slab landed.
* What remains of staging in the indexed kernel (~108 ms) is copy latency the 2-stage ring does
  not fully hide; a 3-stage ring did not help (842 vs 845 ms at 8 × 4; 797 vs 811 at cfg 3).
* With staging gone entirely the kernel would run at 690 ms — the rest is MMA + epilogue (§5.3).
* dX: CAS loop ≈ 17 %, parameter streaming ≈ 13 %, flush ≈ 3 %, remainder (recompute, dZ,
  split-precision gradient MMAs) ≈ 65 %.  The bank conflicts of the scatter cost at most
  ~11 ms (racy RMW still has them), so the simulated XOR swizzle (§5.1) was not worth building.
* The ablation runs drifted ~3 % in forward time between processes (282–305 ms) — deltas below
  ~30 ms are not significant.

### 8.4 dW tiling and ring depth (H = 100M, C = 64, Mean, exploration runs, 3 reps × 2 steps)

| dW kernel | Regs | CTAs/SM | dW ms | Step ms |
|---|---:|---:|---:|---:|
| full 8 × 4 (original) | 255 | 1 | 1131 | 2416 |
| full 8 × 2 / 16 × 1 | 145 / 103 | 1 | 1541 / 1510 | 2837 / 2780 |
| indexed 8 × 4, 2-stage / 3-stage | 255 | 1 | 842 / 845 | 2137 / 2149 |
| indexed 16 × 1, 2-stage / 3-stage | 86–109 | 1 | 876 / 864 | 2180 / 2172 |
| indexed 8 × 2 (unconstrained registers) | 163–179 | 1 | 937 | 2243 |
| indexed 8 × 2, `__launch_bounds__(256, 2)` (cfg 3), 2-stage / 3-stage | 128 | 2 | **797** / 811 | 2084 / 2103 |

Per operator (100M, C = 64, dW ms, full → indexed 8 × 4 → cfg 3): Mean 1127 → 842 → 798,
LogMeanExp 1317 → 953 → 914, softmax-weighted 1489 → 1115 → 1130, power p=2 1297 → 942 → 937.
cfg 3 spills 8 B (LME), 32 B (power) and 96 B (softmax-weighted) of stack; it is the default only
where it was faster.  `dx_warps` 8 / 16 / 32 at 100M, C = 64: 1315 / 968 / 838 ms.

## 9. CUDA resource usage

From `results/traffic/kernel_resources.json` (computed residency, §3):

| Kernel (Mean(ReLU), C = 64, split) | Regs | Stack | Smem | CTAs/SM | Warps/SM |
|---|---:|---:|---:|---:|---:|
| `k_dw` 8 × 4 (original) | 255 | 0 | 64 KiB | 1 | 8 |
| `k_dw_idx` 8 × 4, 2-stage | 255 | 0 | 16 KiB | 1 | 8 |
| `k_dw_idx` cfg 3 (default, Mean / LME) | 128 | 16 B | 8 KiB | 2 | 16 |
| `k_dx` 16 warps (original) | 64 | 0 | 96 KiB | 1 | 16 |
| `k_dx` 32 warps (default, C = 64) | 64 | 0 | 96 KiB | 1 | 32 |

Achieved occupancy, Tensor Core utilisation and stall reasons are not available (§2.3).

## 10. Numerical correctness

* **dW / db of the indexed kernel are bitwise identical to the original kernel** in every tested
  case: 6 smooth operators (Mean(ReLU), Mean(identity), LME, power p=2 and p=3.5,
  softmax-weighted) × split/bf16 × dW tilings 8 × 2, 8 × 4, 16 × 1 × 2-/3-stage rings × 4 shapes
  (D ∈ {64, 256, 1024}, uneven group sizes, partial blocks, B ∈ {8, 40, 64, 136}).
* Against the PyTorch reference at H = 50,000, B = 264 (16 full tiles + 8-token tail;
  `results/traffic/numerics.json`), original and optimized have the same errors to three digits:

| Op | Grad | dX rel / max abs | dW rel / max abs | db rel / max abs | dX vs original (rel) |
|---|---|---|---|---|---:|
| Mean(ReLU) | split | 2.57e-6 / 7.3e-8 | 2.58e-6 / 2.4e-8 | 1.45e-7 / 1.4e-9 | 4.1e-8 |
| LogMeanExp | split | 2.59e-6 / 1.1e-5 | 2.79e-6 / 2.3e-5 | 6.29e-7 / 1.9e-6 | 3.0e-8 |
| Softmax-weighted | split | 1.96e-6 / 3.2e-6 | 2.59e-6 / 1.9e-5 | 3.69e-7 / 5.4e-7 | 3.0e-8 |
| Power p=2 | split | 1.21e-6 / 7.4e-8 | 2.96e-6 / 2.1e-7 | 1.22e-6 / 9.0e-8 | 4.2e-8 |
| Mean(ReLU) | bf16 | 1.70e-3 / 3.9e-5 | 1.63e-3 / 1.7e-5 | 1.45e-7 / 1.4e-9 | 3.8e-8 |
| LogMeanExp | bf16 | 1.58e-3 / 5.2e-3 | 1.65e-3 / 1.2e-2 | 6.29e-7 / 1.9e-6 | 2.8e-8 |
| Softmax-weighted | bf16 | 1.28e-3 / 2.2e-3 | 1.58e-3 / 1.1e-2 | 3.69e-7 / 5.4e-7 | 3.3e-8 |
| Power p=2 | bf16 | 6.72e-4 / 2.8e-5 | 1.66e-3 / 3.9e-5 | 1.22e-6 / 9.0e-8 | 4.1e-8 |

* Changed accumulation order: only dX, through the order of its global FP32 `atomicAdd`s (which
  was already non-deterministic); dW/db keep the original exclusive, deterministic order.
* Tests: `pytest sparse_pooling/tests` → **740 passed, 1 skipped** (the skip is pre-existing:
  GELU runs on the Triton path).  New `tests/test_dw_indexed.py` (364 cases) covers the bitwise
  comparison, reference agreement for split and bf16 (existing tolerances 2e-5 and 5e-3, not
  weakened), colliding/repeated indices (records drawn from 12 features), determinism, fallback
  to the original kernel (C = 8, token-major input, B % 8 ≠ 0), a two-step training trajectory
  with gradient accumulation over two backward calls and fused AdamW (bitwise-identical
  parameters), the default selection, and k_dx with 8 / 16 / 32 warps at C = 8 and 64.
* Compute Sanitizer (`run_sanitizer.sh`, workload extended with indexed variants: both rings,
  all tilings, bf16, B ∈ {8, 24, 40}): memcheck, initcheck, synccheck 0 errors.  racecheck: the
  20 reports are all the pre-existing, documented Triton `bwd_dense_kernel` (triton_fused.py:441)
  replicated-value write; racecheck with `--racecheck-report all --print-limit 0` restricted to
  the native C = 64 path (indexed and full k_dw, 32-warp k_dx, Mean and softmax-weighted) reports
  **0 hazards (0 errors, 0 warnings)**.

## 11. Failed or rejected experiments

* **3-stage ring**: no gain over 2 stages (842 vs 845 ms at 8 × 4; 797 vs 811 at cfg 3).
* **Unconstrained 8 × 2 indexed tiling**: 163–179 registers, 1 CTA/SM, 937 ms — slower than
  8 × 4.  Note: adding an explicit `minBlocks` argument to `__launch_bounds__` changed ptxas's
  register allocation for the *other* instantiations too (an early standalone build of 8 × 2 /
  3-stage landed at 128 registers by accident, 821 ms); cfg 3 pins the budget explicitly so the
  2-CTA/SM variant is reproducible.
* **cfg 3 for softmax-weighted / power mean**: 96 B / 32 B of spills, no gain (1130 vs 1115 ms,
  937 vs 942 ms) → these operators keep 8 × 4.
* **8-warp k_dx**: 1315 ms vs 968 ms (16) and 838 ms (32).
* **dX shared-memory XOR swizzle**: modelled (2.75 → 2.13 wavefronts) but rejected after the
  ablation bounded the non-atomic scatter cost at ≤ 11 ms.
* **Larger dX token tile**: does not fit in 99 KiB of shared memory.

## 12. Remaining bottlenecks

* **dX (806 ms)** is now the largest kernel: projection recompute + dZ + split gradient MMAs
  (~65 %), the 64-bit CAS scatter (~17 %), parameter re-streaming from L2 (~13 %; 871 GiB of
  logical requests because each 16-token tile re-reads all parameters).
* **dW (775 ms)**: ~74 % of the measured MMA rate for its MMA count (analytical); ~108 ms of
  remaining copy latency; the SIMT epilogue (dZ, hi/lo split) shares issue slots with the MMAs.
* **Forward (300 ms)**: unchanged; ~13 % parameter streaming.
* **C = 8** (dX 2.35 s, dW 2.10 s) is untouched: indexed staging cannot help there (§5.2), and
  the dX scatter is 8× denser.
* Counter-based confirmation of any of the above requires Nsight Compute with admin rights.

## 13. Recommended next steps

1. Collect Nsight Compute counters for `k_dw_idx` and `k_dx` (admin): stall reasons and L2
   throughput would show whether the remaining ~108 ms of dW staging is latency or bandwidth.
2. dW epilogue: fold the hi/lo split and the ReLU mask into fewer SIMT instructions (e.g. pack
   both halves with one `cvt.rn.bf16x2` pair per 4 values) — the epilogue is now the largest
   non-MMA cost.
3. dX: a feature-blocked flush layout or a two-level (warp-group) slab could cut CAS retries,
   but only the 17 % CAS share is addressable; the parameter re-reads need a different tile
   shape (e.g. splitting the D-wide FP32 slab across a cluster — not available on sm_86).
4. Keep `cuda_dw_staging="full"` as a fallback for one release cycle, then consider removing
   the full-slab path for C = 64 only (it remains required for C ≤ 32).

## Reproduction

```bash
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
$PY -m pytest sparse_pooling/tests -q                                   # correctness
bash sparse_pooling/benchmarks/run_sanitizer.sh $PY                     # sanitizer
bash sparse_pooling/benchmarks/run_ab.sh $PY 4                          # §8.1 interleaved A/B
bash sparse_pooling/benchmarks/run_traffic.sh $PY optimized             # §8.2 matrix (optimized)
bash sparse_pooling/benchmarks/run_traffic.sh $PY original              # §8.2 matrix (original)
bash sparse_pooling/benchmarks/run_ablation.sh $PY                      # §8.3 ablations
$PY sparse_pooling/benchmarks/traffic_model.py --out sparse_pooling/results/traffic/traffic_model.json
$PY sparse_pooling/benchmarks/kernel_resources.py --out sparse_pooling/results/traffic/kernel_resources.json
$PY sparse_pooling/benchmarks/make_traffic_tables.py                    # results/traffic/TABLES.md
```

Raw data: `results/traffic/*.json` (final runs: `*_ab_r*`, `*_original`, `*_optimized`,
ablations `*_ref`, `*_dw_*`, `*_dx_*`, `*_param_cached`; exploration runs in
`results/traffic/exploration/`), `traffic_model.json`, `kernel_resources.json`,
`numerics.json`, `e2e_check_optimized.json`.  The parameter-cache ablation (`*_param_cached`)
and its reference were run with the 16-warp dX of that time; all §8.3 rows use 16-warp dX.
