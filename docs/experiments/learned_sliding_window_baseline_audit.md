# Baseline audit: learned sliding-window reduction

Pinned to `origin/main` = `4ed74df51b526af9491019407e26e59af70706ac` (fetched 2026-11-06; identical to the SHA named in the
experiment brief, no intervening commits). No `AGENTS.md` / `CLAUDE.md` exists in the repository. Everything in sections 1-3
is **verified behaviour** of that commit, read from the code and measured on the local RTX 3090
(`GPU-74c27f13-7dc6-68bb-e77a-e2ff53220236`, driver 550.67, 24 GiB, nvcc 12.1, torch 2.7.0+cu128). Section 4 lists the
**planned changes**, which are additive and behind explicit experimental API.

Raw record: `sparse_pooling/results/learned_sliding_window_reduction/stage0/audit_shapes.json`.

## 1. The six facts, checked against the code

| # | Claim | Verdict | Where |
|---|---|---|---|
| 1 | `SparsePoolingLayer` is the fan-in-8 API with several pooling operators and an optional within-output semantic hierarchy | Confirmed | `layout.py:55` (`fan_in != 8` rejected), `config.py:22` `POOL_TYPES`, `config.py:206` `is_semantic_hierarchy`, `layer.py:190` |
| 2 | `CoverTopK16Layer` uses frozen supports selected from a K64 envelope; TopK is connectivity selection, not activation pooling | Confirmed | `cover_topk16.py:13-22,111`, `experimental/local_supports.py` |
| 3 | The named path computes Mean(ReLU), BF16 operands, FP32 accumulation / main_grad, backward recomputation, split BF16 gradient products | Confirmed | `kgen.cuh:31-35`, `kgen_layer.py:6-8,99-127` |
| 4 | Forward accumulates compact output sums; dX and dW consume a coefficient shared per output group, `c1T[g, token]` | Confirmed | `kgen.cuh:193` (`ysum`), `:233`, `:455`; `kgen_layer.py:99` |
| 5 | Fast reductions are not bitwise deterministic; no deterministic kgen path | Confirmed | `kgen.cuh:35`, `dispatch.py:571`, `dispatch.py:692` |
| 6 | `mt4_nw8_wg4` was validated for H = 12,500,000 and M in {64, 512, 8192}; nothing else inherits it | Confirmed | `profiles/rtx3090.py:104-108,207-216`. Measured: `auto` at each aligned H resolves to legacy, reason `outside the validated scope: H = ... (validated 12500000)` |

## 2. Actual shapes (real construction, graph 0)

Canonical hidden-neuron map: global neuron `i = offset_g + slot`, physical slot = `g * NB * C + slot`.
With these widths physical order equals global order (`padding_fraction = 0`).

| | H = 10 x 2^18 | H = 40 x 2^18 | H = 48 x 2^18 |
|---|---:|---:|---:|
| H | 2,621,440 | 10,485,760 | 12,582,912 |
| H / G (min = max count) | 2,560 | 10,240 | 12,288 |
| K64 envelope weight | [1024, 10, 256, 64] | [1024, 40, 256, 64] | [1024, 48, 256, 64] |
| C256 records (per output) | 10,240 (10) | 40,960 (40) | 49,152 (48) |
| executed `weight` BF16 | [1024, 160, 16, 16] | [1024, 640, 16, 16] | [1024, 768, 16, 16] |
| executed `indices` int16 | [1024, 160, 16] | [1024, 640, 16] | [1024, 768, 16] |
| `bias` BF16 | [1024, 160, 16] | [1024, 640, 16] | [1024, 768, 16] |
| C16 records, all valid | 163,840 | 655,360 | 786,432 |
| valid weight slots, biases | 41,943,040, 2,621,440 | 167,772,160, 10,485,760 | 201,326,592, 12,582,912 |
| padding | 0 | 0 | 0 |
| `sg_feat` int16 | [4096, 64] | [4096, 64] | [4096, 64] |
| records per input set (min / max) | 2 / 3 | 10 / 10 | 12 / 12 |
| buckets; blocks per bucket | 10,240; 16 | 40,960; 16 | 49,152; 16 |
| buckets crossing an s = 2048 boundary | **0** | **0** | **0** |
| Cover-TopK feasibility (`partial_groups`, `valid_slots == 16 H`) | 0, yes | 0, yes | 0, yes |
| Support fingerprint | `d94978172843…` | `b8b6c9f9da5e…` | `a17c465c658e…` |
| Indices / buckets fingerprint | `2ae4b5a3…` / `64967c19…` | `d10c620e…` / `55ee99b4…` | `997b6e70…` / `6468f8f2…` |

Sets `sg_feat` are identical across widths (`6f53164b…`): they depend only on D, S, R and the graph seed.

The arithmetic alignment was therefore confirmed in the real lowering metadata: every bucket is one C256 record of 16
C16 blocks, and no record straddles a 2,048-neuron stride block.

## 3. Ownership, dispatch, state dict

* **Ownership.** dW / db: exclusive per 16-neuron task, plain read-modify-write of FP32 `main_grad`. dX: FP32 slab copies in shared
  memory, one FP32 atomic per (feature, token) per CTA. Forward: one FP32 atomic per (token, item).
* **Dispatch.** `sp_kg_config(16)`: fwd MT 4 / NW 8; dX MT 4 / NW 8; dW 4 tasks per warp, NW 8, 2 tiles per stage, 3 stages.
  `kgen_dx_group_warps` = 2 for D 1024 / S 64 on the 3090. `dx_variant="auto"` falls back to legacy at all three aligned widths.
* **State dict.** Persistent: `weight`, `bias` (BF16), `indices`, `sg_feat` (int16), `kg_buckets` (int32), `kg_spec` (uint8 JSON), `env`
  (uint8, canonical identity only). Non-persistent: `counts`, `counts_f`. `load_state_dict` rejects another graph spec.
* **Memory (unique storage).** H = 10,485,760: 395.2 MB of persistent tensors before gradients.
* **Replication of the accepted result.** Historical graph-0 checkpoint, energy restoration, 256 updates, data seed 12: support fingerprint
  `e4465520b9cc…` equals the validation record; `auto` executed `mt4_nw8_wg4`, status `hardware-validated`; validation
  0.1699 (start 0.2591), test `icdx39000` 0.1688 against a historical development mean of 0.1713 +- 0.0042 and about 0.1697 on test.

## 4. Pre-existing defect found (not changed here)

`CoverTopK16Layer.from_checkpoint` checks its mask with `m[valid[..., None].expand_as(m)]` (`cover_topk16.py:167`). Boolean indexing over
805 M elements materialises 4 x int64 x 805 M = about **25.6 GiB of host memory** before `.all()`. On this 31 GiB machine that gets the
process killed by the OOM killer or systemd-oomd. The fix is one line, `bool((m != 0).logical_or(~valid[..., None]).all())`. It is out of scope for this
branch and left for review; the replication goes through `from_envelope`.

## 5. Planned changes (all additive)

* New experimental modules `experimental/reduction_layout.py`, `reduction_reference.py`, `learned_reduction.py`; new kernel bundle `lrd`
  (`kernels/cuda/experimental/lrd/`, bundle id 5).
* `backends/cuda/build.py` lists the new bundle; `benchmarks/run_sanitizer.sh` accepts the `lrd` workload.
* No change to production kernels, defaults, the RTX 3090 profile's table, `kgen`, `CoverTopK16Layer` or `SparsePoolingLayer`.
