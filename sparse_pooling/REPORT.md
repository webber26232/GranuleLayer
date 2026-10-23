# Report: 100M-neuron sparse projection with fused pooling on one RTX 3090

All numbers are medians of CUDA-event timings on this machine (RTX 3090, **24 GiB**, 82 SMs,
370 W limit), from the full campaign re-run on 2026-10-16/17 after a hardware repair.  The
first campaign ran on the same card while it exposed only 20 GiB at 744 GB/s under a 300 W cap
(results at git tag `pre-24gib-rerun`; what changed is summarised in §9).  Run-to-run variation
at full scale reaches ~10–15 % (first campaign: the C = 8 LME backward measured 5.13 s and
5.54 s in two runs whose only difference was a dW tiling that is 0.46 s *faster* when timed in
isolation).  Full tables with IQRs, peak memory, kernel-launch counts and backends:
[`results/TABLES.md`](results/TABLES.md); raw JSON/logs in `results/`.  Unless stated otherwise:
H = 100,000,000, B = 8192, D = G = 1024, fan-in 8, BF16 parameters/inputs, FP32 accumulation,
fragment-packed layout, `grad_precision="split"`, step = forward + backward + fused AdamW.

## 1. Hardware facts that shaped the design (measured, `scratchpad` micro-benchmarks)

| Unit | Measured | Consequence |
|---|---|---|
| `mma.m16n8k8` BF16→FP32 | 69 TFLOPS (same rate as k16) | fan-in 8 maps onto a native k = 8 MMA without padding; Triton's minimum k = 16 wastes half |
| SIMT FFMA | 33 TFLOPS | tensor cores are only ~2.1× SIMT on GA102 GeForce (FP32-accumulate is half rate) |
| MUFU `ex2` | 2.17 Tops/s | one exp per hidden element (8.2e11) costs ≥ 0.38 s — more than the MMAs (0.19 s) |
| shared-memory FP32 `atomicAdd` | CAS loop (`ATOMS.CAST.SPIN`); `red.shared.*.f32/f16x2` too | there are no native FP float atomics in shared memory on sm_86 |
| random-line L2 reads | ~0.11 line requests/clk/SM (~2 TB/s) | the number of 128 B lines touched, not bytes, limits small parameter loads |
| DRAM | 819 GB/s (1 GiB device-to-device copy; 744 GB/s before the repair) | parameters (2 GB) stream from L2 when CTAs walk them in lock-step |

Useful work per forward: 2·B·H·8 = 13.1 TFLOP; the MMA floor is ~0.19 s, so a step that
recomputes once in backward has an MMA floor of ≈ 0.6–1.1 s depending on the formulation.

## 2. Final comparison (H = 100M, B = 8192)

Step = forward + backward + fused AdamW (BF16 moments).  Peak VRAM includes ~5.4 GiB of
parameters + FP32 `main_grad` and 3.35 GiB of Adam moments.  "Correct?" = this exact
kernel/policy combination passes the forward and gradient tests against the reference at
small H (tests cannot run the materialising reference at 100M).  All rows come from one run
of `run_all.sh`, after the per-operator dW tiling fix (§4).

C = 8 (8 neurons share an index record):

| Pooling | Levels | Policy | Forward ms | Backward ms | Step ms | Peak VRAM | tokens/s | Correct? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Max | 1 | save_winners | 475 | 11 | 513 | 8.80 GiB | 15,963 | yes |
| Max | 1 | minimal_stats (recompute winners) | 423 | 485 | 935 | 8.89 GiB | 8,760 | yes |
| Top-8 Mean | 1 | save_winners | 1081 | 79 | 1180 | 9.26 GiB | 6,944 | yes |
| Top-8 Mean | 1 | recompute_stats | 1081 | 1155 | 2262 | 9.29 GiB | 3,622 | yes |
| Mean(ReLU) | 1 | minimal_stats | 386 | 4473 | 4898 | 8.79 GiB | 1,672 | yes |
| LogMeanExp τ=0.5 | 1 | minimal_stats | 724 | 5345 | 6114 | 8.88 GiB | 1,340 | yes |
| Softmax-weighted β=1 | 1 | minimal_stats | 807 | 5561 | 6403 | 8.91 GiB | 1,279 | yes |
| Softmax-weighted β=1 | 1 | recompute_stats | 806 | 6383 | 7225 | 8.94 GiB | 1,134 | yes |
| Power mean p=2 | 1 | minimal_stats | 582 | 5190 | 5811 | 8.92 GiB | 1,410 | yes |
| Max → Mean | 2 | save_winners | 552 | 63 | 626 | 9.62 GiB | 13,094 | yes |
| Top-8 → LogMeanExp | 2 | save_winners | 2152 | 487 | 2668 | 12.48 GiB | 3,071 | yes |
| Mean(ReLU) → LogMeanExp | 2 | minimal_stats | 464 | 4481 | 4989 | 9.68 GiB | 1,642 | yes |
| Max + affine → Max | 2 | save_winners | 552 | 20 | 586 | 9.90 GiB | 13,984 | yes |

C = 64:

| Pooling | Levels | Policy | Forward ms | Backward ms | Step ms | Peak VRAM | tokens/s | Correct? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Max | 1 | save_winners | 378 | 11 | 410 | 8.65 GiB | 19,986 | yes |
| Max | 1 | minimal_stats (recompute winners) | 257 | 390 | 667 | 8.74 GiB | 12,274 | yes |
| Top-8 Mean | 1 | save_winners | 1076 | 77 | 1177 | 9.11 GiB | 6,958 | yes |
| Top-8 Mean | 1 | recompute_stats | 1074 | 1151 | 2255 | 9.14 GiB | 3,633 | yes |
| Mean(ReLU) | 1 | minimal_stats | 291 | 2035 | 2361 | 8.64 GiB | 3,470 | yes |
| LogMeanExp τ=0.5 | 1 | minimal_stats | 594 | 2307 | 2910 | 8.73 GiB | 2,815 | yes |
| Softmax-weighted β=1 | 1 | minimal_stats | 656 | 2580 | 3279 | 8.76 GiB | 2,499 | yes |
| Softmax-weighted β=1 | 1 | recompute_stats | 653 | 3233 | 3916 | 8.79 GiB | 2,092 | yes |
| Power mean p=2 | 1 | minimal_stats | 467 | 2304 | 2800 | 8.77 GiB | 2,926 | yes |
| Max → Mean | 2 | save_winners | 458 | 64 | 539 | 9.47 GiB | 15,202 | yes |
| Top-8 → LogMeanExp | 2 | save_winners | 3362 | 461 | 3851 | 12.32 GiB | 2,127 | yes |
| Mean(ReLU) → LogMeanExp | 2 | minimal_stats | 359 | 2057 | 2450 | 9.53 GiB | 3,343 | yes |
| Max + affine → Max | 2 | save_winners | 459 | 20 | 499 | 9.75 GiB | 16,427 | yes |

Two-level models use 16,384-neuron level-1 shards (6 shards per group, 6,144 level-1 units).
Single-BF16 dZ gradients (`grad_precision="bf16"`, standard AMP semantics) at C = 64:
Mean(ReLU) step 1905 ms, LogMeanExp 2502 ms, softmax-weighted 2861 ms, power 2313 ms.

## 3. Answers to the research questions

**Q1. Which operators fuse efficiently into a streaming reduction?**  All six.  Max,
Mean(act), Power mean and the LSE-type operators (LogMeanExp, softmax-weighted) have
constant-size associative states (1–3 floats) and cost the projection plus their epilogue.
Forward at C = 8: Mean(ReLU) 0.39 s, Max 0.42 s (0.48 s while also tracking winner ids),
Power p=2 0.58 s, LogMeanExp 0.72 s, softmax-weighted 0.81 s.  The exp-based ones are bound by
MUFU (one `ex2` per hidden element is ≥ 0.38 s on its own).  Top-K is exact and streaming but
needs per-lane lists plus a selection merge: 0.90–1.08 s for k ≤ 8.

**Q2. Saved compact statistics vs recomputation.**  Nothing needs more than O(B·G·k) state.
Mean needs no statistic; LogMeanExp recovers its log-normaliser exactly from its own output;
Power mean needs only y; softmax-weighted needs y plus the log-normaliser (32 MiB); Max/Top-K
need winner ids (32 MiB / 256 MiB int32) or a recompute pass.  Recomputing statistics costs one
extra forward-type pass (C = 64): Max backward 11 → 390 ms, Top-8 77 → 1151 ms,
softmax-weighted 2.58 → 3.23 s.  Saving winners is the right default for Max/Top-K (`auto`
does so) — unless the ids themselves become large: with 2,048-neuron semantic shards (49,152
level-1 units) Top-8 ids would be 12.9 GB, so above a 2 GiB checkpoint budget the layer falls
back to recomputing winners in 512 MiB token chunks (reported in `last_info["policy_fallback"]`).

**Q3. True overhead of backward recomputation.**  It depends on how many neurons carry gradient:
* Max/Top-K with saved winners: backward is 11 ms / 79 ms (8.4M / 67M sparse events) — 2–7 %
  of the step.
* Smooth operators: every hidden element carries gradient, the projection is recomputed twice
  (once in each gradient kernel) and two gradient products are formed.  Backward/forward =
  3.9–7.0× at C = 64 (2.0–2.6 s) and 6.9–11.6× at C = 8 (4.5–5.6 s).  §4 explains why.
* Recompute policies add one forward-type pass (above).

**Q4. Is LogMeanExp a useful trade-off vs Max?**  As a speed proposition, no: forward 1.5–1.6×
Max, step 7.1× (C = 64) to 12× (C = 8), because Max's backward is nearly free (winner events)
while LogMeanExp's is a dense recompute.  What it buys is a gradient for every one of the
~97,657 neurons in a group (softmax weights) and a smooth approximation of Max that sharpens
with τ.  Mean(ReLU) gives a dense gradient for ~20 % less step time (no exp).  The temperature
does not affect cost (τ = 0.1 … 2: 2.86–2.88 s forward + backward at C = 64; 5.93–6.02 s at
C = 8).

**Q5. Top-K vs Max.**  Forward at C = 8: Top-2/4/8/16 cost 2.1/2.3/2.5/4.3× Max without id
tracking (k = 32 runs on Triton: 12×).  Backward with saved ids: 20/40/79/159/318 ms for
k = 2…32 (linear in k).  The first CUDA design (per-lane threshold only) was slower than Triton;
the two-pass design (a Max-cost pass that yields a provable lower bound for the k-th value, then
exact insertion of the few elements above it) made Top-8 2.7× faster than the Triton kernel
(1.07 s vs 2.89 s, first campaign).

**Q6. Does hierarchical (physical) pooling improve kernel efficiency?**  Not at B = 8192.
Token tiles × groups already give 262K CTAs; splitting a group across programs only adds a
workspace round trip.  Sweep at H = 10M (TABLES.md): CUDA Max C = 8 63 ms unsplit vs 219 ms
with 8 splits (512 MiB workspace); the Triton weight-stationary schedule (each program owns a
neuron chunk and loops over all token tiles) is likewise no better than token-stationary.
Top-8 with 8 splits would need a 4 GiB workspace and is refused by the budget check.  Splits
help only at small B, where they are chosen automatically.

**Q7. Which multi-level schemes add expressiveness?**  Physical hierarchies of one associative
state never do (tests assert equality with the flat operator for Max, Sum/Mean with counts, LSE,
softmax-weighted, power and Top-K).  Max → Max with a monotone link (identity/ReLU) is exactly
global Max (the API warns).  Genuinely different models (tests compare with explicit hierarchical
references and assert ≠ flat): Max → Mean, Mean(ReLU) → Max, Top-K → LogMeanExp,
Mean(ReLU) → LogMeanExp, LogMeanExp → Mean, and anything with a learned affine/gate transform or
a learned weighted sum between levels.  Mean → Mean equals the global mean only if all shards
have equal size; with a partial last shard it is a different model (tested).

**Q8. Memory saved by recomputation.**  The hidden tensor would need 3.2 TB (FP32) at B = 8192;
what `ctx` keeps is 32–288 MiB for single-level models.  Where Strategy A is possible at all
(H = 262,144, B = 1024; trade-off table in TABLES.md), storing activations costs 4.0–6.0 GiB of
extra memory versus 37–98 MiB, and is 10–53× *slower* (150–195 ms vs 2.8–20 ms per step), because
ordinary autograd over a materialised tensor moves far more data than the fused kernels
recompute.  So there is no memory/speed trade-off to choose between A and B: B (compact
statistics) dominates A on both axes.  C (recompute statistics) costs 1.26–1.70× B's step and
only shrinks the checkpoint by 4–36 MiB; its *peak* is slightly higher than B's because the
recomputed statistics are allocated transiently in backward.  At full scale the same holds:
the Pareto set is {B with saved winners for Max/Top-K, B minimal statistics for smooth operators},
plus C only when saved ids would exceed the checkpoint budget (Q2).

The table's error columns compare against an FP64 reference on the same inputs (now seeded).
For Max, Top-K and ReLU they are ill-conditioned: an FP32-vs-FP64 near-tie (or a pre-activation
at the ReLU kink) flips one winner and moves the relative error from ~1e-7 to 1e-4–1e-3.  With
unseeded inputs this happened in a fraction of runs on every strategy, including plain autograd
(A); with the fixed seed only Mean(ReLU) A shows it (4.3e-5).  Bitwise-repeat GEMM and VRAM
pattern tests found no hardware errors.

**Q9. Best total throughput.**  Max with saved winners: 410 ms per step at C = 64 (19,986
tokens/s; 34.7 useful TFLOPS in a forward that also tracks winner ids, 50.9 TFLOPS without),
513 ms at C = 8.  Among dense-gradient operators: Mean(ReLU) at C = 64, 1.90 s per step with
BF16 dZ gradients.

**Q10. Practical for B = 8192, H = 100M on a 24 GB 3090?**  Memory — yes.  Every single-level
configuration fits with either optimizer setting (peak 8.6–9.3 GiB with BF16 Adam moments,
17.2 GiB with FP32 moments + FP32 master weights); two-level models with Top-K level-1 features
fit with BF16 moments (12.5 GiB with 6,144 level-1 units, 16.2 GiB with 49,152).  With FP32
moments + master weights the 6,144-unit model fits (19.2 GiB); the 49,152-unit variants need
~23 GiB at peak (21.45 GiB allocated plus a 1.5 GiB request) and run out of memory on the
23.7 GiB the card offers with a desktop session attached — with or without
`expandable_segments`.  Time — Max and Top-K models train at 0.41–1.18 s per 8192-token step,
which is practical; dense-gradient pooling takes 1.9–6.4 s per step, usable but dominated by
backward, and C = 8 is the expensive end (§4).

## 4. Where backward time goes (smooth operators)

* **dX kernel** (token-stationary).  At C = 8 each block contributes 8 values per token:
  8.2e11 scatter contributions per step that must be reduced on chip.  Ablation at H = 10M
  (first campaign): 376 ms with FP32 smem `atomicAdd` (CAS loop), 235 ms after pairing tokens
  into 64-bit CAS, 158 ms with racy plain RMW, 105 ms without any scatter.  The scatter is ∝ 1/C,
  which is why C = 64 is 2.2× faster.  The CTA must hold a `[1024 × 16]` FP32 slab plus the X
  slab, so the token tile is 16 and the weights stream twice as often as in forward (~820 GB of
  L2 traffic).
* **dW kernel** (weight-stationary).  All dW/db stay in registers (written once, no atomics),
  which caps neurons per CTA (~1–2K) and makes every CTA stream all of X (traffic
  H·B·2 KB / neurons-per-CTA).  At C = 8 the m16 neuron dimension of every MMA is half empty;
  we use the empty rows for the lo part of the split dZ (hi+lo for one MMA).  An attempt to
  free registers by re-reading parameters from L1 every token tile was 1.6–3× *slower* and
  was reverted.
* **Precision**: single-BF16 dZ (standard AMP semantics, ~1.5e-3 relative gradient error)
  saves 15–22 % of backward at C = 64 (Mean 2.04 → 1.59 s) and only 3–5 % at C = 8, where MMA
  is not the limit.
* **Tiling is operator-dependent** (controlled measurement at C = 8, H = 100M, dW kernel only,
  first campaign): Mean is fastest with 8 warps × 4 work units (2.1 s vs 2.6 s), while the
  exp/power epilogues spill at that size and run 1.2–1.5× faster with 8 × 2 (LME 2.96 vs
  3.42 s, softmax-weighted 3.24 vs 4.76 s, power 3.03 vs 4.68 s).  The dispatcher picks per
  operator.

## 5. Forward findings

* Feature-major input is mandatory: token-major gathers were 22–28× slower in Triton (each
  gathered BF16 is its own sector).  The CUDA kernel accepts both (transposes into the slab).
* Native m16n8k8 vs Triton k16 (same connectivity and layout, Max, C = 8, H = 10M, B = 8192):
  63 ms vs 120–140 ms over Triton batch tiles 64–256 (1.9–2.2×); LogMeanExp 86 ms vs 131–166 ms.
* Fragment-packed parameters (first campaign): Max C = 8 at H = 100M 823 → 415 ms; the
  K-contiguous layout needed three line requests per 8-neuron block (weights, bias, indices);
  the packed layout needs 3 per 4 blocks.  Index-table traffic itself is small; the request
  count was the issue.
* The smem `ldmatrix` gather has ~2.5-way bank conflicts for random rows, but removing it
  entirely changed nothing (ablation) — not a bottleneck.
* Weight reuse comes from L2: token tiles of one group run concurrently and walk the group's
  weights in lock-step (grid order token-tile-fastest).  We do not claim weights are read from
  DRAM once — that was not measured (no counters).  Consistent with this, Max forward barely
  moved when DRAM bandwidth rose 10 % with the repair (§9).
* Exp-based operators are MUFU-bound; C barely matters for them (LME 0.72 s at C = 8 vs
  0.59 s at C = 64), while Max (without id tracking) goes from 0.42 to 0.26 s.  Tracking winner
  ids costs 12 % (C = 8) to 47 % (C = 64) of the Max forward — the compare-and-select per element
  is no longer hidden behind the tensor cores at C = 64.

## 6. Numerical precision

* Projections: FP32 accumulation of exact BF16 products; reference agreement ≤ 3e-6 relative.
* Winners: kernels and reference use identical operands; winner agreement > 99.9 % is tested
  (disagreement possible only at FP32 near-ties); exact ties resolve to the smallest neuron
  id in every path (tested with all-equal neurons).
* LSE states use base-2 arithmetic with a lazy maximum (rescale only when exceeded by 2^8);
  power mean is max-scaled; both are stable at |z| ≈ 50 (tested).
* Gradients: split precision within ~3e-6 of FP32 autograd; FP64 finite differences agree to
  1e-4; bias gradients never pass through BF16.

## 7. Limitations and unresolved bottlenecks

* No counter-based profiling (admin-only counters); analytic traffic numbers are labelled as such.
* C = 8 dense-gradient backward (shared-memory FP32 atomics and dW register pressure).
* Top-K with k = 32 (Triton path, 5.2–6.4 s forward) and Top-K over small semantic shards
  (fixed per-CTA cost of the two-pass selection: Top-8 → LME with 16K-neuron shards has a
  2.2–3.4 s forward against 1.08 s for flat Top-8).
* Two-level Top-K models with 49,152 level-1 units and FP32 Adam moments + master weights need
  ~23 GiB and do not fit next to a desktop session (the 6,144-unit variant fits at 19.2 GiB).
* Exp-heavy epilogues are MUFU-bound; offloading a fraction of `ex2` to an FMA polynomial
  (FA3-style) was estimated at ≤ 20 % and not implemented.
* Mode A (independent indices) runs on Triton SIMT only.
* Sparse/lazy optimizer updates for Max/Top-K are deliberately not offered: they change AdamW
  semantics (moment decay and weight decay of non-selected parameters).

## 8. Backward kernel optimization (dX / dW)

Per-kernel timing, a source-derived traffic model and ablations are in
[MEMORY_TRAFFIC_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/MEMORY_TRAFFIC_OPTIMIZATION.md).  Two
changes are now the C = 64 defaults: indexed, warp-local input staging in `k_dw` (dW 1096 →
775 ms, bitwise-identical gradients) and 32-warp `k_dx` CTAs (939 → 806 ms).  Full step at
H = 100M, C = 64, Mean(ReLU), split: 2357 → 1904 ms (1.24×, interleaved A/B).  The tables in
§2 above predate these changes.  Phase 2 ([BACKWARD_COMPUTE_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/BACKWARD_COMPUTE_OPTIMIZATION.md)):
exact pooling-specialised dZ epilogues (per-token BF16 hi/lo + bit masks for Mean), measured
per-operator defaults and a work-unit-based dX warp dispatch take the same step from 1964 to
1814 ms in a same-session A/B against `dw-indexed-staging` (1.08×; dX 1.13×, dW 1.07×).

## 9. Re-measurement after the hardware repair

The first campaign ran while the card exposed 20 GiB at 744 GB/s under a 300 W cap; the card
now exposes 24 GiB (23.7 GiB to PyTorch) at 819 GB/s under its default 370 W limit.

* **Capacity** is the material change: the FP32-moments + master-weights two-level Top-K model
  with 6,144 level-1 units now fits (19.2 GiB); the 49,152-unit variants still do not (Q10).
* **Speed** changed little where kernels are L2- or MMA-bound: Max forward −3 % (C = 64) to
  +2 % (C = 8), Max → Mean / Max + affine → Max steps −7 to −9 %.  Dense-gradient steps are
  2–10 % faster (LogMeanExp C = 64 3.21 → 2.91 s, softmax-weighted C = 8 7.04 → 6.40 s), Top-K
  forward 8–12 % faster, and the H = 262K trade-off table 10–12 % faster.  These shifts are within
  or near the 10–15 % run-to-run spread and coincide with the higher power limit, so they are not
  attributed to DRAM bandwidth alone.
* Earlier tables mixed rows from a separate `_rerun` measurement; `make_report.py` now uses a
  re-run only if it is newer than the main run, and the obsolete `_rerun` files were removed.
