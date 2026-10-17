# Report: 100M-neuron sparse projection with fused pooling on one RTX 3090

All numbers are medians of CUDA-event timings on this machine (RTX 3090, **20 GiB**
variant, 82 SMs, ~1.7–1.9 GHz under a 300 W cap at ~84 °C; run-to-run variation at full
scale reaches ~10–15 %: the C = 8 LME backward measured 5.13 s and 5.54 s in two runs whose only
difference was a dW tiling that is 0.46 s *faster* when timed in isolation).  Full tables with IQRs, peak memory, kernel-launch counts and backends:
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
| DRAM | 744 GB/s | parameters (2 GB) stream from L2 when CTAs walk them in lock-step |

Useful work per forward: 2·B·H·8 = 13.1 TFLOP; the MMA floor is ~0.19 s, so a step that
recomputes once in backward has an MMA floor of ≈ 0.6–1.1 s depending on the formulation.

## 2. Final comparison (H = 100M, B = 8192)

Step = forward + backward + fused AdamW (BF16 moments).  Peak VRAM includes ~5.4 GiB of
parameters + FP32 `main_grad` and 3.35 GiB of Adam moments.  "Correct?" = this exact
kernel/policy combination passes the forward and gradient tests against the reference at
small H (tests cannot run the materialising reference at 100M).  Rows marked † were re-measured
after the per-operator dW tiling fix (§4); the first measurement is in the C8 log.

C = 8 (8 neurons share an index record):

| Pooling | Levels | Policy | Forward ms | Backward ms | Step ms | Peak VRAM | tokens/s | Correct? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Max | 1 | save_winners | 468 | 13 | 503 | 8.80 GiB | 16,293 | yes |
| Max | 1 | minimal_stats (recompute winners) | 398 | 482 | 906 | 8.89 GiB | 9,046 | yes |
| Top-8 Mean | 1 | save_winners | 1066 | 96 | 1191 | 9.76 GiB | 6,877 | yes |
| Top-8 Mean | 1 | recompute_stats | 1073 | 1175 | 2287 | 9.79 GiB | 3,583 | yes |
| Mean(ReLU) | 1 | minimal_stats | 392 | 4503 | 4974 | 8.79 GiB | 1,647 | yes |
| LogMeanExp τ=0.5 † | 1 | minimal_stats | 721 | 5539 | 6567 | 8.88 GiB | 1,248 | yes |
| Softmax-weighted β=1 † | 1 | minimal_stats | 863 | 6143 | 7041 | 8.91 GiB | 1,163 | yes |
| Softmax-weighted β=1 † | 1 | recompute_stats | 870 | 7043 | 7940 | 8.94 GiB | 1,032 | yes |
| Power mean p=2 † | 1 | minimal_stats | 632 | 5775 | 6424 | 8.92 GiB | 1,275 | yes |
| Max → Mean | 2 | save_winners | 571 | 75 | 687 | 9.62 GiB | 11,920 | yes |
| Top-8 → LogMeanExp † | 2 | save_winners | 2439 | 568 | 3008 | 12.48 GiB | 2,724 | yes |
| Mean(ReLU) → LogMeanExp † | 2 | minimal_stats | 555 | 5003 | 5606 | 9.68 GiB | 1,461 | yes |
| Max + affine → Max | 2 | save_winners | 575 | 24 | 629 | 9.90 GiB | 13,030 | yes |

C = 64:

| Pooling | Levels | Policy | Forward ms | Backward ms | Step ms | Peak VRAM | tokens/s | Correct? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Max | 1 | save_winners | 389 | 13 | 433 | 8.65 GiB | 18,922 | yes |
| Max | 1 | minimal_stats (recompute winners) | 276 | 417 | 720 | 8.74 GiB | 11,372 | yes |
| Top-8 Mean | 1 | save_winners | 1206 | 92 | 1318 | 9.61 GiB | 6,215 | yes |
| Mean(ReLU) | 1 | minimal_stats | 310 | 2217 | 2556 | 8.64 GiB | 3,205 | yes |
| LogMeanExp τ=0.5 | 1 | minimal_stats | 617 | 2489 | 3208 | 8.73 GiB | 2,554 | yes |
| Softmax-weighted β=1 | 1 | minimal_stats | 716 | 2818 | 3573 | 8.76 GiB | 2,293 | yes |
| Softmax-weighted β=1 | 1 | recompute_stats | 726 | 3558 | 4322 | 8.79 GiB | 1,895 | yes |
| Power mean p=2 | 1 | minimal_stats | 520 | 2523 | 3081 | 8.77 GiB | 2,659 | yes |
| Max → Mean | 2 | save_winners | 500 | 73 | 582 | 9.47 GiB | 14,082 | yes |
| Top-8 → LogMeanExp | 2 | save_winners | 3815 | 535 | 4280 | 12.32 GiB | 1,914 | yes |
| Mean(ReLU) → LogMeanExp | 2 | minimal_stats | 374 | 2202 | 2643 | 9.53 GiB | 3,099 | yes |
| Max + affine → Max | 2 | save_winners | 489 | 24 | 542 | 9.75 GiB | 15,122 | yes |

Two-level models use 16,384-neuron level-1 shards (6 shards per group, 6,144 level-1 units).
Single-BF16 dZ gradients (`grad_precision="bf16"`, standard AMP semantics) at C = 64:
Mean(ReLU) step 2063 ms, LogMeanExp 2741 ms, softmax-weighted 3204 ms, power 2584 ms.

## 3. Answers to the research questions

**Q1. Which operators fuse efficiently into a streaming reduction?**  All six.  Max,
Mean(act), Power mean and the LSE-type operators (LogMeanExp, softmax-weighted) have
constant-size associative states (1–3 floats) and cost the projection plus their epilogue.
Forward at C = 8: Mean(ReLU) 0.39 s, Max 0.40 s (0.47 s while also tracking winner ids),
Power p=2 0.63 s, LogMeanExp 0.72 s, softmax-weighted 0.86 s.  The exp-based ones are bound by
MUFU (one `ex2` per hidden element is ≥ 0.38 s on its own).  Top-K is exact and streaming but
needs per-lane lists plus a selection merge: 0.97–1.2 s for k ≤ 8.

**Q2. Saved compact statistics vs recomputation.**  Nothing needs more than O(B·G·k) state.
Mean needs no statistic; LogMeanExp recovers its log-normaliser exactly from its own output;
Power mean needs only y; softmax-weighted needs y plus the log-normaliser (32 MiB); Max/Top-K
need winner ids (32 MiB / 256 MiB int32) or a recompute pass.  Recomputing statistics costs one
extra forward-type pass: Max backward 13 → 482 ms, Top-8 96 → 1175 ms, softmax-weighted
2.82 → 3.56 s (C = 64).  Saving winners is the right default for Max/Top-K (`auto` does so) —
unless the ids themselves become large: with 2,048-neuron semantic shards (49,152 level-1
units) Top-8 ids would be 12.9 GB, so above a 2 GiB checkpoint budget the layer falls back to
recomputing winners in 512 MiB token chunks (reported in `last_info["policy_fallback"]`).

**Q3. True overhead of backward recomputation.**  It depends on how many neurons carry gradient:
* Max/Top-K with saved winners: backward is 13 ms / 96 ms (8.4M / 67M sparse events) — 2–8 %
  of the step.
* Smooth operators: every hidden element carries gradient, the projection is recomputed twice
  (once in each gradient kernel) and two gradient products are formed.  Backward/forward =
  3.9–7.1× at C = 64 (2.2–2.8 s) and 7.1–11.5× at C = 8 (4.5–6.1 s).  §4 explains why.
* Recompute policies add one forward-type pass (above).

**Q4. Is LogMeanExp a useful trade-off vs Max?**  As a speed proposition, no: forward 1.6–1.8×
Max, step 7.4× (C = 64) to 13× (C = 8), because Max's backward is nearly free (winner events)
while LogMeanExp's is a dense recompute.  What it buys is a gradient for every one of the
~97,657 neurons in a group (softmax weights) and a smooth approximation of Max that sharpens
with τ.  Mean(ReLU) gives a dense gradient for ~20 % less step time (no exp).  The temperature
does not affect cost (τ = 0.1 … 2: 3.18–3.24 s step at C = 64; 6.63–6.73 s at C = 8).

**Q5. Top-K vs Max.**  Forward at C = 8: Top-2/4/8/16 cost 2.4/2.7/3.1/5.1× Max (k = 32 runs on
Triton: 14×).  Backward with saved ids: 25/48/95/188/374 ms for k = 2…32 (linear in k).  The
first CUDA design (per-lane threshold only) was slower than Triton; the two-pass design (a
Max-cost pass that yields a provable lower bound for the k-th value, then exact insertion of the
few elements above it) made Top-8 2.7× faster than the Triton kernel (1.07 s vs 2.89 s).

**Q6. Does hierarchical (physical) pooling improve kernel efficiency?**  Not at B = 8192.
Token tiles × groups already give 262K CTAs; splitting a group across programs only adds a
workspace round trip.  Sweep at H = 10M (TABLES.md): CUDA Max C = 8 69 ms unsplit vs 272 ms
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
extra memory versus 37–102 MiB, and is 10–56× *slower* (168–218 ms vs 3–37 ms per step), because
ordinary autograd over a materialised tensor moves far more data than the fused kernels
recompute.  So there is no memory/speed trade-off to choose between A and B: B (compact
statistics) dominates A on both axes.  C (recompute statistics) costs 1.3–1.65× B's step and
only shrinks the checkpoint by 4–36 MiB; its *peak* is slightly higher than B's because the
recomputed statistics are allocated transiently in backward.  At full scale the same holds:
the Pareto set is {B with saved winners for Max/Top-K, B minimal statistics for smooth operators},
plus C only when saved ids would exceed the checkpoint budget (Q2).

**Q9. Best total throughput.**  Max with saved winners: 433 ms per step at C = 64 (18,922
tokens/s; 33.7 useful TFLOPS in a forward that also tracks winner ids, 47.5 TFLOPS without),
503 ms at C = 8.  Among dense-gradient operators: Mean(ReLU) at C = 64, 2.06 s per step with
BF16 dZ gradients.

**Q10. Practical for B = 8192, H = 100M on a 24 GB 3090?**  Memory — yes.  On this 20 GiB card
every single-level configuration fits with either optimizer setting (peak 8.6–9.8 GiB with BF16
Adam moments, 17.2 GiB with FP32 moments + FP32 master weights); two-level models with Top-K
level-1 features fit with BF16 moments (12.5 GiB with 6,144 level-1 units, 16.2 GiB with 49,152)
but exceed 20 GiB with FP32 moments + master weights.  Time — Max and Top-K models train at
0.43–1.3 s per 8192-token step, which is practical; dense-gradient pooling takes 2–7 s per step,
usable but dominated by backward, and C = 8 is the expensive end (§4).

## 4. Where backward time goes (smooth operators)

* **dX kernel** (token-stationary).  At C = 8 each block contributes 8 values per token:
  8.2e11 scatter contributions per step that must be reduced on chip.  Ablation at H = 10M:
  376 ms with FP32 smem `atomicAdd` (CAS loop), 235 ms after pairing tokens into 64-bit CAS,
  158 ms with racy plain RMW, 105 ms without any scatter.  The scatter is ∝ 1/C, which is why
  C = 64 is 2.4× faster.  The CTA must hold a `[1024 × 16]` FP32 slab plus the X slab, so the
  token tile is 16 and the weights stream twice as often as in forward (~820 GB of L2 traffic).
* **dW kernel** (weight-stationary).  All dW/db stay in registers (written once, no atomics),
  which caps neurons per CTA (~1–2K) and makes every CTA stream all of X (traffic
  H·B·2 KB / neurons-per-CTA).  At C = 8 the m16 neuron dimension of every MMA is half empty;
  we use the empty rows for the lo part of the split dZ (hi+lo for one MMA).  An attempt to
  free registers by re-reading parameters from L1 every token tile was 1.6–3× *slower* and
  was reverted.
* **Precision**: single-BF16 dZ (standard AMP semantics, ~1.5e-3 relative gradient error)
  saves 20–25 % at C = 64 (Mean backward 2.22 → 1.70 s) and nothing at C = 8, where MMA is not
  the limit.
* **Tiling is operator-dependent** (controlled measurement at C = 8, H = 100M, dW kernel only):
  Mean is fastest with 8 warps × 4 work units (2.1 s vs 2.6 s), while the exp/power epilogues
  spill at that size and run 1.2–1.5× faster with 8 × 2 (LME 2.96 vs 3.42 s, softmax-weighted
  3.24 vs 4.76 s, power 3.03 vs 4.68 s).  The dispatcher picks per operator.

## 5. Forward findings

* Feature-major input is mandatory: token-major gathers were 38× slower in Triton (each
  gathered BF16 is its own sector).  The CUDA kernel accepts both (transposes into the slab).
* Native m16n8k8 vs Triton k16 (same connectivity and layout, Max, C = 8, H = 10M, B = 8192):
  69 ms vs 138–161 ms over Triton batch tiles 64–256 (2.0–2.3×); LogMeanExp 89 ms vs 163 ms.
* Fragment-packed parameters: Max C = 8 at H = 100M 823 → 415 ms; the K-contiguous layout
  needed three line requests per 8-neuron block (weights, bias, indices); the packed layout
  needs 3 per 4 blocks.  Index-table traffic itself is small; the request count was the issue.
* The smem `ldmatrix` gather has ~2.5-way bank conflicts for random rows, but removing it
  entirely changed nothing (ablation) — not a bottleneck.
* Weight reuse comes from L2: token tiles of one group run concurrently and walk the group's
  weights in lock-step (grid order token-tile-fastest).  We do not claim weights are read from
  DRAM once — that was not measured (no counters).
* Exp-based operators are MUFU-bound; C barely matters for them (LME 0.72 s at C = 8 vs
  0.62 s at C = 64), while Max (without id tracking) goes from 0.40 to 0.28 s.  Tracking winner
  ids costs 17 % (C = 8) to 41 % (C = 64) of the Max forward — the compare-and-select per element
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
* Top-K with k = 32 (Triton path, 5.8–7.1 s forward) and Top-K over small semantic shards
  (fixed per-CTA cost of the two-pass selection: Top-8 → LME with 16K-neuron shards has a
  2.4–3.8 s forward against 1.1–1.2 s for flat Top-8).
* Two-level Top-K models with FP32 Adam moments + master weights do not fit in 20 GiB.
* Exp-heavy epilogues are MUFU-bound; offloading a fraction of `ex2` to an FMA polynomial
  (FA3-style) was estimated at ≤ 20 % and not implemented.
* Mode A (independent indices) runs on Triton SIMT only.
* Sparse/lazy optimizer updates for Max/Top-K are deliberately not offered: they change AdamW
  semantics (moment decay and weight decay of non-selected parameters).
