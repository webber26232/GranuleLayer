# Learned sliding-window reduction: experiment report

**Status: experimental. Nothing here changes production defaults or `main`.** Base commit `4ed74df`. All numbers are from
synthetic teachers on one RTX 3090; they don't establish downstream application accuracy.

Generated tables with every per-seed value: [`tables.md`](../../sparse_pooling/results/learned_sliding_window_reduction/tables.md).
Baseline audit: [`learned_sliding_window_baseline_audit.md`](learned_sliding_window_baseline_audit.md).

## Recommendation

1. **Don't adopt the exact proposal (A4, signed random init).** With the same parameters and the same data, starting from
   `P = 1/L` (A4mean) is 25-32 % better on compose and 5-12 % better on interact. Random init is the part of the proposal that
   costs the most.
2. **Carry `A3mean` forward as the leading candidate:** non-overlapping learned windows of 2,048 neurons, mean init, then a
   dense readout. With plentiful data it beats A0 by 22.7 % (compose) and 23.7 % (interact) at equal steps, and by 17 % and 23 % at
   equal GPU time. It beats A4mean by 5.4 % and 5.3 % with 3.7 times fewer added weights. It costs 1.35 times A0 per step and
   +0.49 GiB of peak memory.
3. **Don't pay for overlap.** With init matched, overlap (A4mean) and the parameter-matched non-overlapping control (MCmean)
   can't be told apart: +0.2 % and +0.4 %, both intervals spanning zero.
4. **Use `A2` (fixed mean over 2,048, then dense) if 1.35x is too expensive and the task resembles interact.** It costs 1.03x and
   gains 12.8 % on interact with plentiful data, but gains nothing on compose and is 9.7 % worse there on limited data.
5. **Keep A0 if training data is scarce or the task is support-local, until shown otherwise.** On the 256-batch pool, A3mean
   doesn't reliably beat A0 on interact (-1.1 %, interval spans zero). On the local task (screening only, one seed), A0 was best
   of all 15 models.
6. **Not ready for production.** Before integrating, run the three checks in [Next steps](#next-steps), above all A3mean at a longer
   horizon: A4mean's compose advantage over A0 disappeared by 8192 updates, and A3mean hasn't been tested there.

## How to read these results

| | |
|---|---|
| **Declared campaign** | Rules committed before any screening result. Commit `8bc8fd9`. 701 training jobs, 61 performance jobs. |
| **Supplementary round 1** | Commit `34ffa02`, **declared after the campaign's test losses were seen**. Controls, longer horizon, fresh data, secondary-width leader, real M = 8,192 training. |
| **Supplementary round 2** | Commit `1811b3c`, **declared after round 1's results**. Remaining matrix on fresh data, fresh data with longer horizon. |
| **Pool data** | 256 training batches of 512 tokens that repeat (131,072 tokens). My declared choice, and small next to tens of millions of weights. |
| **Fresh data** | No training batch repeats. **Learning rates were selected on pool data and not re-tuned.** |

Why the additions were needed: the declared finalist rule compared A3mean with plain A4, which confounds init with overlap; the
brief requires a control in that case. It also left A2 and A5 without full-width confirmation, and chose the secondary-width
reducer from short tuning runs. A0 was also still improving at 4096 updates, and the large models were overfitting a pool whose
size I had picked.

Selection used validation loss only throughout, and test losses never fed a selection rule. But the *decision to run more
experiments* was made after seeing test losses, so the supplementary results are not blind. Intervals are 95 % paired t with n = 3
and aren't adjusted for the 40+ contrasts, so treat borderline ones as weak.

## Quality versus cost

H = 10,485,760. Test loss, mean of 3 paired seeds, 4096 updates, lower is better. Cost is the measured median full training step at
M = 8,192, relative to A0 at 520.6 ms.

| Model | Added weights | Compose, fresh | Interact, fresh | Compose, pool | Interact, pool | Step cost | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 mean | 0 | 0.0427 | 0.7193 | 0.0456 | 0.7118 | 1.00 | 2.20 |
| A1 A0 + dense | 1,048,576 | 0.0402 | 0.7995 | 0.0465 | 0.7669 | 1.00 | 2.26 |
| A2 mean of 2048 + dense | 5,242,880 | 0.0425 | 0.6275 | 0.0500 | 0.6769 | 1.03 | 2.66 |
| A5 fixed overlap + dense | 5,238,784 | 0.0484 | 0.6557 | 0.0527 | 0.6891 | 1.04 | 2.69 |
| A3 learned, random init | 15,728,640 | not run | not run | 0.0493 | 0.8574 | 1.35 | 2.69 |
| **A3mean learned, mean init** | 15,728,640 | **0.0330** | **0.5487** | **0.0406** | 0.7041 | 1.35* | 2.69* |
| A4 exact, random init | 57,626,624 | 0.0512 | 0.6110 | 0.0548 | 0.7783 | 1.41 | 3.24 |
| A4mean exact, mean init | 57,626,624 | 0.0349 | 0.5796 | 0.0412 | 0.6823 | 1.41* | 3.24* |
| MC multi-channel, random init | 57,671,680 | 0.0447 | 0.6438 | 0.0501 | 0.7344 | 1.41 | 3.16 |
| MCmean multi-channel, mean init | 57,671,680 | 0.0349 | 0.5774 | not run | not run | 1.41* | 3.16* |

\* Timed with the same-shape random-init kernels (A3, A4, MC). Training `ms/step` confirms this: 52.0 vs 52.1 for A3mean / A3 and
56.8 for A4mean / A4.

**Which data regime applies is the largest open question.** The ranking on interact changes with it: A3mean is clearly best on fresh
data, while on the pool A2, A4mean, A5 and A3mean can't be separated.

## Direct answers

### Which reducer has the best held-out quality at equal steps? At equal wall-clock?

**Equal steps (4096 updates):** A3mean on fresh data, on both tasks, by a margin the intervals resolve against every other model
run there (against MCmean only on interact; compose -5.2 %, interval spans zero). On the pool, A3mean is best on compose, tied with
A4mean. On interact no winner is resolved among A2 (0.677), A4mean (0.682), A5 (0.689) and A3mean (0.704).

**Equal GPU time (164 s, when A0 finishes 4096 updates; validation loss):**

| Model | Compose, fresh | Interact, fresh | Compose, pool | Interact, pool |
|---|---:|---:|---:|---:|
| A0 | 0.0429 | 0.7029 | 0.0457 | 0.6957 |
| A1 | 0.0402 | 0.7840 | 0.0466 | 0.7512 |
| A2 | 0.0431 | 0.6118 | 0.0488 | **0.6566** |
| A3mean | **0.0358** | **0.5428** | **0.0421** | 0.6786 |
| A4mean | 0.0386 | 0.5948 | 0.0449 | 0.6672 |
| A5 | 0.0476 | 0.6405 | 0.0530 | 0.6704 |
| MCmean | 0.0386 | 0.5905 | not run | not run |

The slower A3mean still wins at equal time on fresh data and on pool compose. On pool interact, the cheap A2 wins.

**Longer horizon (8192 updates; A0 against the validation-selected leader, A4mean; A3mean wasn't run):**

| Data | Task | A0 test | A4mean test | A4mean vs A0, 95 % CI | Val at 328 s, A0 / A4mean |
|---|---|---:|---:|---|---:|
| Fresh | compose | 0.0303 | 0.0300 | -1.0 % [-3.3, +1.3] | **0.0304** / 0.0315 |
| Fresh | interact | 0.6011 | 0.5432 | **-9.6 %** [-11.5, -7.7] | 0.5850 / **0.5465** |
| Pool | compose | 0.0404 | 0.0394 | -2.5 % [-13.6, +8.6] | 0.0404 / 0.0394 |
| Pool | interact | 0.6423 | 0.6965 | **+8.4 %** [+3.9, +13.0] | **0.6271** / 0.6580 |

So on compose the reducer mostly converges faster, and A0 catches up. On interact with plentiful data the gain persists, shrinking
from -19.4 % to -9.6 %. On pool interact the reducer overfits and A0 overtakes it.

### Is most of the gain explained by adding the dense mixing layer?

**No.** A1 (A0 + identity-initialized dense) is no better on compose (+2.1 % pool, -6.0 % fresh, both spanning zero) and **worse** on
interact (+7.7 % pool, +11.2 % fresh, both resolved).

Retaining about 5,120 intermediate features does help, but only on some tasks. A2 vs A1 on interact is -11.7 % (pool) and -21.5 %
(fresh), while on compose it is +7.5 % (pool) and +5.9 % (fresh, spans zero).

### Does learned weighting beat fixed mean at similar intermediate width?

**Yes with mean init and plentiful data. Not reliably otherwise.**

| Contrast (same width) | Compose, pool | Interact, pool | Compose, fresh | Interact, fresh |
|---|---:|---:|---:|---:|
| A3mean - A2 (non-overlap) | **-18.7 %** | +4.0 % | **-22.3 %** | **-12.6 %** |
| A4mean - A5 (overlap) | **-21.9 %** | -1.0 % | **-27.8 %** | **-11.6 %** |
| A3 - A2, random init | -1.3 % | **+26.7 %** (worse) | not run | not run |
| A4 - A5, random init | +3.9 % | **+13.0 %** (worse) | not run | not run |

Bold: the interval excludes zero. The frozen-readout screening agrees that learned weights recover far more information than fixed
means (compose: A4mean 0.066 and A3mean 0.070, against A2 0.219 and A5 0.260), but frozen supports are a different problem from
end-to-end training.

### Does overlap beat the nearly parameter-matched non-overlap multi-channel control?

**No.** With matched init on fresh data, A4mean - MCmean is +0.2 % (compose, [-6, +7]) and +0.4 % (interact, [-0.9, +1.7]).

- With random init the sign depends on the task: A4 - MC is +14.5 % on compose and -5.1 % on interact on fresh data, and +9.4 % and
  +6.0 % on the pool.
- The non-overlapping A3mean, with 3.7 times fewer weights, beats MCmean on interact (-5.0 %) and is not separable on compose
  (-5.2 %, spans zero), so the extra budget buys nothing here.
- Overlap alone with fixed weights is no help either: A5 is worse than A2 on fresh data (+13.7 %, +4.5 %).
- MCmean wasn't screened; it borrows A4mean's learning rates, which if anything favours A4mean, and it wasn't run on the pool.

### Is GELU helpful, and is the benefit stable across seeds/tasks?

**Not end to end; the evidence is one seed at screening width (H = 2,621,440).** A4gelu beat A4 on one task of three, so the declared
gate didn't promote it. Beyond the 256-update smoke, GELU has no full-width quality run.

| Test loss, end to end | Compose | Interact | Local |
|---|---:|---:|---:|
| A4 / A4gelu | 0.0508 / 0.0517 | 0.7278 / 0.7980 | 0.2211 / 0.2124 |
| A3 / A3gelu | 0.0509 / 0.0545 | 0.8421 / 0.8671 | 0.2398 / 0.2256 |
| A5 / A5gelu (fixed control) | 0.0443 / 0.0482 | 0.6988 / 0.6919 | 0.1702 / 0.1735 |
| A2 / A2gelu (fixed control) | 0.0452 / 0.0457 | 0.6945 / 0.6789 | 0.1747 / 0.1953 |

With a frozen projection the picture differs: A4gelu reaches 0.712 on interact against A4's 0.887, the largest GELU effect seen, and
A4gelu 0.071 vs 0.079 on compose. So nonlinearity helps when W can't adapt, and that benefit doesn't carry over when W trains.
GELU also hurts robustness under the scale shift for fixed reducers (A2gelu 0.182 vs A2 0.123 on compose). GELU uses the exact
function. GELU combined with mean init wasn't tested.

### Does the three-stage pilot warrant further work?

**Not on this evidence; it wasn't promoted.** At screening width, TS_id and TS_gelu are worse than A0 on all three tasks, at the
highest cost of the screened models (17.9 vs 13.4 ms per step at M = 512). TS_id is slightly ahead of A4 on compose and local and
slightly behind on interact; TS_gelu doesn't beat A4 on any task.

| Test loss, end to end | Compose | Interact | Local |
|---|---:|---:|---:|
| A0 | 0.0460 | 0.6660 | 0.1631 |
| A4 | 0.0508 | 0.7278 | 0.2211 |
| TS_id | 0.0490 | 0.7346 | 0.2156 |
| TS_gelu | 0.0547 | 0.8535 | 0.2214 |

Two caveats: its second stage is a narrower 256-feature bottleneck at screening width (1,216 at full width), and it used random init
only, which later turned out to be a major handicap for the other learned models. A mean-init three-stage model is untested.

### What is the incremental training-step cost and peak-memory cost versus the current Cover-TopK/16 baseline?

Median step in ms and peak GiB, at H = 10,485,760. Dispersion (IQR / median) is at most 1.6 %, and most cells are below 1 %.

| Model | M = 64 | M = 512 | M = 8,192 | vs A0 at 8,192 | Peak GiB at 8,192 | vs A0 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 11.5 | 39.8 | 520.6 | 1.00 | 2.20 | |
| A1 | 12.0 | 40.1 | 523.1 | 1.00 | 2.26 | +0.06 |
| A2 | 12.0 | 40.8 | 536.6 | 1.03 | 2.66 | +0.46 |
| A5 | 12.2 | 41.0 | 540.8 | 1.04 | 2.69 | +0.49 |
| A3 | 13.3 | 51.6 | 702.2 | 1.35 | 2.69 | +0.49 |
| MC | 15.2 | 56.0 | 732.7 | 1.41 | 3.16 | +0.96 |
| A4 | 15.9 | 56.2 | 734.5 | 1.41 | 3.24 | +1.04 |

**Baseline definition:** A0 here uses `mt2_nw4_wg2` for dX, which is 3 % faster than the validated `mt4_nw8_wg4` (537.0 ms). Against
that validated tiling, A4 is 1.37x and A3 is 1.31x. `mt2_nw4_wg2` is **not validated for production**. Legacy dX is 757.0 ms.

**Secondary width, H = 12,582,912, M = 8,192:** A0 587.1 ms and 2.61 GiB; A3 794.7 ms (1.35x) and 3.21 GiB; A4 828.1 ms (1.41x) and
3.87 GiB. Quality, on the pool, one seed, compose, 4096 updates: A4mean 0.0441 vs A0 0.0469 (-5.9 %). A3mean wasn't run at this width,
because the validation rule selected A4mean as leader.

**Memory breakdown, A4, M = 8,192:**

| Item | Size |
|---|---:|
| Sparse W, BF16 | 320 MiB |
| Sparse b | 20 MiB |
| Sparse main_grad, FP32 | 680 MiB |
| P, BF16 | 100 MiB |
| P main_grad, FP32 | 200 MiB |
| V, c, d, FP32 | 20 MiB |
| Metadata | 26.9 MiB |
| Reducer work lists | 11.1 MiB |
| FusedAdamW state (A0: 0.66 GiB) | 0.86 GiB |
| **Steady state with optimizer** | **2.78 GiB** |
| Peak: forward / backward / optimizer | 2.26 / 2.34 / 2.82 GiB |
| **Peak: full step** / reserved | **3.24** / 3.44 GiB |

Cold start, excluded from step timing: load base 0.39 s, reducer model and work lists 0.82 s, first step 0.78 s.

Physical padding for A4: 40,960 of 52,428,800 P slots (0.08 %). Accounting reference, not a measurement: forward reduction 0.944
TFLOPs (P 0.858, V 0.086) and sparse projection 2.749 TFLOPs at 2 FLOPs per multiply-add, without padding or recomputation.

### Which kernels dominate, and did recomputation, atomic contention or register pressure change the bottleneck?

**dX stays the largest kernel, so the bottleneck didn't move.** A4's fused dW + dP is what grew most.

| ms, M = 8,192 | Forward | Finalize + readout | Dense bwd | dX | dW / db (+ dP) | Optimizers | **Step** |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 63.8 | 0.3 | 0.5 | 309.6 (59 %) | 141.3 | 4.2 | 520.6 |
| A3 | 121.3 | 5.2 | 8.2 | 327.6 (47 %) | 232.9 | 4.9 | 702.2 |
| A4 | 136.5 | 5.1 | 8.2 | 333.4 (45 %) | 242.7 | 5.9 | 734.5 |

Of A4's extra 214 ms, about 101 ms is fused dW + dP, 73 ms forward, 24 ms dX and 13 ms dense.

**Register pressure was the most important kernel finding.** It sets how many CTAs fit per SM:

| Kernel | Variant | Registers | CTAs / SM | Measured effect |
|---|---|---:|---:|---|
| dX | MT4 / NW8 / WG4 | 171 | 1 | 617 ms, step 1014.5 ms |
| dX | MT2 / NW4 / WG2 (default) | 112 | 4 | 333 ms, step 734.5 ms |
| dW + dP fused | WPT4 | 191 | 1 | 291.7 ms |
| dW + dP fused | WPT2 (default) | 112 | 2 | 242.7 ms |
| dW, dP separate | WPT2 | 80 / 64 | 3 / 4 | 175.9 + 141.5 = 317.4 ms |
| Forward | MMA split (default) | 87 | 2 | 136.5 ms |

- **Fusing dP was worth it:** 243 ms, against 317 ms separately, because it avoids one more recomputation. Hidden activations are
  computed three times per step (forward, dX, dW / dP).
- **Forward epilogue:** MMA BF16 is fastest (111.2 ms) but is a declared numerical change. MMA split is 136.5 ms and SIMT is 230.2 ms.
- **Dense readout is cheap:** about 13 ms in total at M = 8,192. TF32 changes the A4 step only from 734.5 to 726.8 ms.
- **Atomic contention wasn't measured directly.** I have phase timings and compiled attributes, not GPU traces. The forward bound
  is 1.68 billion atomics per call, but its share of the time is unknown.

### Can the exact proposed model train at H = 10,485,760, M = 8,192 on the 3090 without materializing M*H tensors?

**Yes.** A4 ran 128 real optimizer updates at M = 8,192, each batch built from 16 distinct pool batches: 735.4 ms per step, 2.98 GiB
peak in training and 3.24 GiB in the isolated memory pass, validation loss 0.735 to 0.108. An M*H BF16 tensor would need 160 GiB. A test
asserts that no such tensor is allocated or saved in forward or backward. This shows feasibility only: A0 reached 0.102 over the same
128 updates, and M = 8,192 was not run to convergence.

### What should remain experimental, and what would justify a later production integration?

**Everything here stays experimental**: the `lrd` kernels, `LearnedReducer`, and the `mt2_nw4_wg2` dX default in this path. Kernel
status is recorded as `experimental, unvalidated`. It is not marked validated, and it doesn't inherit `mt4_nw8_wg4`'s validation.

Integration of A3mean would be justified by:
1. A3mean keeping a resolved advantage over A0 at a long horizon with plentiful data.
2. The advantage appearing on a real task, not only on synthetic teachers.
3. A dedicated dispatch and hardware validation record for its exact shape.

## What explains the gain

| Factor | Effect | Evidence |
|---|---|---|
| **Init of learned weights** | **Largest single factor.** A4mean vs A4: -24.8 % / -12.3 % (pool), -31.7 % / -5.1 % (fresh). A3mean vs A3: -17.6 % / -17.9 % (pool) | Same architecture, separately tuned learning rates |
| **Amount of training data** | Large, especially for high-capacity models. A4mean on interact: 0.682 (pool), 0.580 (fresh). Gap to A0 moves from -4.2 % to -19.4 % | LR not re-tuned for fresh data |
| **Learned weighting** | Strong with mean init on fresh data: -12 to -28 % | A3mean vs A2, A4mean vs A5 |
| **Extra intermediate features** | Task-dependent: helps interact, not compose | A2 vs A1 |
| **Dense mixing alone** | None, or harmful on interact | A1 vs A0 |
| **Overlap** | None resolved | A4mean vs MCmean, A5 vs A2 |
| **Nonlinear depth** | Not end to end; only with frozen W; one seed | GELU table above |
| **Third stage** | Not competitive as piloted | Pilot table above |

**Mean init changes the starting point:** on the primary compose task, A0 starts at validation loss 0.453, random-init A4 at 0.735.
Why it then stays better for thousands of updates is not something these experiments isolate.

**Linear-depth reminder:** with identity activations the reduction is a structured linear map over h, so identity variants add no
nonlinear expressivity, and A3mean's advantage can't come from that. These runs don't separate how much of it comes from signed
coefficients, multiple outgoing paths, or easier optimization.

## Correctness and gates

| Gate | Result |
|---|---|
| Full test suite on the training code (`1811b3c`) | **3139 passed, 32 skipped.** All skips are in existing tests, for reasons unrelated to this work: two-GPU hardware, shapes that don't fit, path-specific variants |
| `test_learned_reduction.py` | 63 tests, included above: FP64 oracle, finite differences, routing, padding, optimizer and checkpoint round-trip, M*H absence, rejections, deterministic-request rejection |
| Sanitizer, `lrd` | 4 of 4 pass at `2984cba`: memcheck, racecheck, initcheck, synccheck. Kernels unchanged since |
| Sanitizer, `cover` and `kgen` | 8 of 8 pass |
| **Not covered by sanitizer** | The cuBLAS dense GEMM: its initialization trips memcheck on `cuCtxGetLimit`, in library code, so the workload omits it |
| Historical replication | Support fingerprint `e4465520b9cc` matches; `mt4_nw8_wg4` hardware-validated; val 0.1699, test 0.1688 (historical development 0.1713 +- 0.0042, research about 0.1697) |

**Numerical error against the FP64 same-operand oracle.** Tolerances were set before evaluating the kernels. Values are the worst
case over 27 cases: 9 geometries and M = 8, 40, 136. Raw: `numeric_errors.json`.

| Quantity | Relative L2 | Tolerance | Max absolute | Max normalized (tolerance 5e-3) |
|---|---:|---:|---:|---:|
| U | 2.6e-6 | 3e-5 | 1.5e-5 | 3.5e-6 |
| dx | 4.5e-6 | 1e-4 | 2.0e-5 | 5.4e-6 |
| dW | 4.7e-6 | 1e-4 | 6.0e-5 | 8.8e-6 |
| db | 7.5e-6 | 1e-4 | 4.3e-5 | 7.2e-6 |
| dP | 7.5e-6 | 1e-4 | 2.5e-4 | 1.6e-5 |

Reductions use FP32 atomics and are **not bitwise deterministic**. A strict deterministic request is rejected explicitly.

## Run, skipped, blocked, failed

**Failed:** 0 of 831 training jobs and 0 of 61 performance jobs in the final records. One earlier launch was killed by systemd-oomd, in
`from_checkpoint`'s mask check, a pre-existing `main` bug (about 25.6 GB, or 23.8 GiB, of host indices). Replication now uses
`from_envelope`. The bug was fixed separately on `fix/cover-topk16-mask-check-memory`, not in this branch.

| Cell | State | Reason |
|---|---|---|
| Core matrix A0-A5 + MC, 3 seeds, compose + interact, H = 10.5M, pool | **Run** | |
| A3mean, A4mean, 3 seeds, pool | **Run** | A3mean by the declared rule, A4mean as a control |
| A0, A1, A2, A3mean, A4, A4mean, A5, MC, MCmean, fresh data | **Run** | Supplementary |
| A3 random init, fresh data | Not run | Not declared: round 1 chose A0, the leader and A4; round 2 took A3mean as the non-overlap candidate |
| MCmean, pool data | Not run | Added only for the fresh-data overlap question |
| **A3mean, longer horizon** | **Not run** | Validation rule chose A4mean as leader. Most important gap; see next steps |
| **Local task at full width** | **Not run** | Screening only, where A0 wins. Limits generality |
| A3 / A4 GELU beyond 256-update full-width smoke | Skipped by gate | A4gelu won 1 of 3 tasks; A3gelu not the strongest non-overlap reducer |
| Three-stage at full width | Skipped by gate | Not competitive at screening |
| CUDA Graph timing | **Blocked** | Per-call scratch tensors have fresh addresses, so these layers don't support capture. All timing is eager |
| GPU traces / direct atomic-contention profile | Not done | Phase events and compiled attributes only |
| H = 100,663,296 probe; 8 / 16-output windows; `append_last`, circular boundaries | Optional, skipped | Budget priority; narrow-output MMA wasn't shown to be limiting |

**GPU use:** campaign services ran 13.7 hours end to end: 6.0 (declared), 4.2 and 3.5 (supplementary), including 12.7 hours of training
and evaluation. Pilots and gates from before the campaign weren't logged in total and add a few hours. Soft budget: 24.

## Limitations

- **Three seeds.** Intervals are rough and unadjusted for multiple comparisons. Graph, init and training seeds move together in
  each pair.
- **The learning-rate grid is limiting at its edges.** A0, A1, A2 and A5 selected the top value (0.003) and A4 and A4mean the bottom
  (1e-4) on compose, so none of them may be at their true optimum. Selection used 512-update runs at full width, which doesn't
  necessarily predict 4096 updates.
- **Fresh-data learning rates were inherited from pool selection**, and MCmean's from A4mean.
- **Screening rank doesn't predict full width reliably.** For example, A4mean led screening on compose while A3mean led at full width.
  I didn't extrapolate quality between widths.
- **Secondary width** has one seed, pool data, compose only.
- **Synthetic teachers only.**
- **Negative train-validation gaps** on fresh interact reflect sampling differences between small evaluation sets, not a real effect.
  Test exceeds validation by about 0.016 on interact for every model for the same reason.
- **Commit message of `8bc8fd9`** says "25.6 GiB"; the correct figure is 25.6 GB (23.8 GiB). History wasn't rewritten.

## Next steps

In order of value:
1. **A3mean vs A0 at 8192+ updates on fresh data, 3 seeds, both tasks.** This is about 1.4 GPU-hours and decides whether its lead lasts.
2. **Run A0, A2 and A3mean on a real task with its real data volume.** The pool/fresh reversal means synthetic results can't say which
   regime applies.
3. **Local task at full width on fresh data**, since A0 won it at screening.
4. Re-tune learning rates on fresh data, widening the grid past the current edges.

## Reproduction

Branch `experiment/learned-sliding-window-reduction`. Measurements ran from a clean detached checkout (`env.provenance` in every
record gives its commit and source-tree hash). Raw per-job JSON: `../lsw_runs/run`, not committed. Compact raw results, CSV and tables:
`sparse_pooling/results/learned_sliding_window_reduction/`, with checksums in `SHA256SUMS`.

```bash
PY=python; OUT=/path/to/results
S=sparse_pooling/benchmarks/run_learned_reduction.sh
# Run detached: this machine's systemd-oomd kills large jobs started from a terminal scope
systemd-run --user --unit=lsw-campaign -p MemoryHigh=20G -p MemoryMax=24G --working-directory=$PWD --setenv=PATH="$PATH" \
    --setenv=CKPT_G0=/path/to/C256_R256_prox_a0_step128.pt \
    bash $S $PY $OUT correctness sanitizer replicate screen frozen primary secondary perf supplementary supplementary2
python sparse_pooling/benchmarks/learned_reduction_report.py --out $OUT --dest sparse_pooling/results/learned_sliding_window_reduction
python sparse_pooling/benchmarks/learned_reduction_numeric.py --out sparse_pooling/results/learned_sliding_window_reduction/numeric_errors.json
```

Every stage resumes from per-job atomic JSON. The learned-model API lives in `sparse_pooling/experimental/learned_reduction.py`
(`ReductionModel`, `ReducerConfig`, `core_configs`). Results can't reproduce bitwise, because reductions use FP32 atomics.
