# Cover-TopK/16 (`sparse_pooling.cover_topk16`)

`CoverTopK16Layer` is an opt-in input-supergroup layer.  It is separate from the fan-in-8 `SparsePoolingLayer`, whose
API and defaults are unchanged.  For every 16-neuron row group it keeps a frozen 16-input support, selected once from
an original K64 envelope, and executes compactly at K16 / C16.

## Notation: original vs executed

| | K (fan-in) | C (neurons sharing one index record) | S (candidate set) | records |
|---|---:|---:|---:|---|
| original envelope | 64 | 256 | 64 inputs per record (`feat[q]`) | (output g, record j), 64 inputs each |
| executed (compact) | 16 | 16 | -- | one per valid 16-neuron row group u = (g, j, c // 16) |

Validated model:

| quantity | value |
|---|---|
| D = G | 1024 |
| H | 12,500,000 |
| envelope | K64 / C256, S64 / R256, Q = 4096 |
| original records | 49,152 (48,128 full and 1,024 tails of 175 / 176 neurons) |
| executed records | 781,312 compact K16 / C16 records |
| valid weight slots | exactly 200,000,000 |
| biases | 12,500,000 |

`layer.metadata` records:
- `envelope` (K, C, S, R, Q, generation, seed, spec);
- `executed` (K, C, records);
- `valid_slots` / `biases`;
- `support_fingerprint`;
- `policy`, plus `policy_version` / `rng_schema`;
- `normalization`.

## Semantics

1. **Importance.** For each row group u and local input j:
   `a[u, j] = sum over u's valid neurons c of W_pre[u, c, j]^2`.
   It is computed in FP64 with sequential adds, so it is independent of chunking and evaluation order.
2. **TopK.** Each valid row group keeps the 16 largest `a[u, j]` among its record's 64 candidates. Ties go to the lower
   local coordinate.
3. **Cover repair.** Inside every original record, each of the 64 candidates must be kept by at least one valid row
   group. This is a union constraint, not equal-frequency quotas.
   * The repair repeatedly applies the swap of minimum energy loss, taking from a donor that is still covered at least
     twice.
   * A record with m valid row groups is infeasible if 16 m < 64. That raises `ValueError`; there is no fallback.
4. **What each neuron keeps.**
   * Every valid neuron keeps exactly 16 trainable weights: its pre-pruning values at the kept coordinates, as sorted
     index / value pairs. It also keeps its own bias.
   * Neuron, output, record, input-set (q) identity and the original per-output valid-neuron count (the Mean
     denominator) are unchanged.
5. **Optional energy restoration.** `normalization="energy"` applies a one-time energy restoration to the canonical
   K64 values before lowering.
   * Per neuron: `s_i = min(8, sqrt(E_dense_i / max(E_kept_i, 1e-6 * E_dense_i)))`.
   * It scales weights only; bias scaling is off.
   * The result is rounded to BF16 once.
   * It is recorded in `metadata["normalization"]` and is never re-applied.
     * The packed layer keeps no pre-pruning values.
     * `from_envelope` refuses an envelope that is already marked as normalized.

Selection is one-time preprocessing. After it, the support is frozen: the packed layer stores only the 16 kept values
per neuron, so training cannot change the support.

## Quickstart

The example is small, so it runs anywhere CUDA does. For the validated model, use `input_dim=1024`,
`hidden_dim=12_500_000`, `output_dim=1024`, `ConnectivityPlan(64, 256, seed=graph_seed)`, and the K64 / C256 envelope
values of your checkpoint. Alternatively, call `CoverTopK16Layer.from_checkpoint(path)` for a K64 envelope checkpoint
in the research format (`granule.k64_checkpoint/1`). Its weight / bias sha256 are verified, and the path is yours:
nothing is hard-coded.

<!-- quickstart -->
```python
import torch
from sparse_pooling import CoverTopK16Layer
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan

D, H, G = 256, 8 * 431 + 4, 8                      # small demo; validated model: 1024, 12_500_000, 1024
env = KLayout(D, H, G, 64, 256)                    # the original K64 / C256 envelope layout
w = (torch.randn(env.weight_shape, device="cuda") / 8).bfloat16()   # your pre-pruning K64 values [G, NB, 256, 64]
b = torch.zeros(env.bias_shape, device="cuda", dtype=torch.bfloat16)
zero_padding_(env, w, b)

layer = CoverTopK16Layer.from_envelope(
    w, b, input_dim=D, hidden_dim=H, output_dim=G,
    connectivity=ConnectivityPlan(64, 2, seed=0),  # S64 candidate sets; validated model: R = 256
    normalization="energy")                        # one-time energy restoration ("none" to skip)
print(layer)                                       # envelope K64/C256 S64/R2, executed K16/C16, dx_variant='auto'

layer.enable_main_grad()                           # FP32 main_grad, consumed and zeroed by FusedAdamW
opt = FusedAdamW([layer.weight, layer.bias], lr=1e-4, moments="bf16", stochastic_rounding=True)

x = torch.randn(4, 32, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)  # [batch, time, D]
y = layer(x)                                       # [4, 32, G]: leading dims flattened (4 * 32 % 8 == 0)
y.float().square().mean().backward()               # x.grad: [4, 32, D] BF16
opt.step()

print(layer.explain())                             # profile, build identity; dX "pending" until a batch is known
print(layer.last_info["kgen_dx"]["variant"])       # what the backward actually launched, and why

layer.save_packed("cover_topk16_packed.pt")        # frozen support + values + graph / metadata
restored = CoverTopK16Layer.load_packed("cover_topk16_packed.pt")  # no reselection, no envelope needed
```

`forward` accepts `[..., D]` and returns `[..., G]`. The flattened token count must be a multiple of 8, because the
kernels read the input feature-major. Input gradients come back in the input's shape and dtype.

## Execution and the K16 dX choice

* **Numerics.**
  * Mean(ReLU), with BF16 inputs and parameters.
  * FP32 accumulation and FP32 `main_grad`.
  * The backward recomputes z, and dZ is split into BF16 hi + lo parts.
  * Cross-CTA sums use FP32 atomics, so results are not bitwise reproducible run to run.
  * `reduction="deterministic"` is rejected with an error: no deterministic kgen path exists.
* **No M x H intermediate.** No M x H activation, mask or product is materialized. This is tested.
* **dX selection.** `dx_variant` is resolved at call time by `backends.dispatch.kgen_dx_resolve`, in this order:
  1. numerical requirements;
  2. legality: compiled, WG divides NW, shared memory within the device's opt-in limit;
  3. an explicit request;
  4. the selected tuning profile's measured choice;
  5. the legacy launch.
* **Requests.**
  * `"auto"` (the default of this layer): the profile's measured tiling inside its validated scope.
    * The RTX 3090 profile (`backends/profiles/rtx3090.py`, `KGEN_DX_AUTO`) selects `mt4_nw8_wg4` for:
      * the primary model: D = G = 1024, H = 12.5M, envelope K64 / C256, S64 / R256, executed K16 / C16;
      * Mean(ReLU), split dZ, fast reduction;
      * the measured batches M = 64, 512 and 8192.
    * Everything else gets the legacy launch, with the first out-of-scope field as the reason. That includes other
      batches, other shapes or precisions, other GPUs, and a forced `generic_cuda` profile (`PhysicalPlan(
      tuning_profile="generic_cuda")` or `SPARSE_POOL_PROFILE=generic_cuda`).
  * `"legacy"`: the original launch. Use it for A/B comparisons and troubleshooting.
  * `"mt<MT>_nw<NW>_wg<WG>"`: an explicit tiling.
    * It is capability-checked, and an illegal request raises with the reason.
    * Its status is `explicit, unvalidated` unless the selected profile is hardware-validated on this device and
      measured exactly that tiling for this call.
* **The selected tiling `mt4_nw8_wg4`.**
  * 16 MT = 64 tokens per CTA, 8 warps = 256 threads.
  * WG 4 means two FP32 shared slab copies, each shared by 4 warps through a 64-bit CAS.
  * Shared memory at D 1024 / S 64 is 42 KiB, against 74 KiB for the legacy launch. That fits 2 CTAs / SM on the
    RTX 3090 instead of 1.
  * Forward, dW and the final dX cast are unchanged; only the reduction order of the FP32 dX sums differs.
* **Records and caching.**
  * `layer.last_info["kgen_dx"]` records what actually ran after each backward:
    * entry point, MT / NW / WG, threads, shared bytes, token tiles / CTAs;
    * the decision's source / status / reason / profile + version;
    * the validated scope;
    * the kgen library hash.
  * `layer.explain(B)` reports the decision a batch-B call would take (labelled `planned`). Without B it reports
    `pending`.
  * Decisions are cached per device identity, profile, request, full call context (incl. batch), compiled variants
    and library hash.

## Serialization and runtime requirements

* **`save_packed` stores:**
  * compact `indices` (int16 [G, NB16, 16]);
  * `weight` (BF16 [G, NB16, 16, 16]) and `bias`;
  * `env` (uint8 local positions, used for canonical identity only);
  * `sg_feat` (the S64 candidate sets) and `kg_buckets`;
  * the graph spec, connectivity and metadata;
  * fingerprints of the immutable buffers.
* **`load_packed`** restores the layer without the envelope or the research checkpoint, and does not run selection
  again. It is a `weights_only` load, and it checks the immutable buffers against their fingerprints. Save the
  optimizer state separately with `opt.state_dict()`.
* **Caches.** Device moves (`.to()`) and `load_state_dict` invalidate the per-device runtime (work lists, kernel
  attributes). The per-output denominators are non-persistent buffers that move with the module.
* **Runtime needs** an NVIDIA GPU of the native kernel family (sm_80+, see `docs/gpu_portability.md`), CUDA
  `nvcc` (12.x) and PyTorch with BF16.
  * The `kgen` library is compiled on first use into `SPARSE_POOL_CACHE_DIR`.
  * Importing `sparse_pooling` and host-side construction touch neither CUDA nor nvcc.

## Memory and one-time costs

Full model, graph 0, with steady-state persistent tensors counted by unique storage (`layer.memory_report()`,
`sparse_pooling.memory_accounting`):

| category | MiB |
|---|---:|
| weights (BF16, 200M valid + padding) | 381.5 |
| bias | 23.8 |
| FP32 main_grad (weight + bias) | 810.7 |
| BF16 Adam moments (FusedAdamW, m + v) | 810.7 |
| indices + envelope map + sets + buckets + work lists + denominators | 43.9 |

An earlier research table reported 67.7 MiB of metadata for this layer (and 392.9 MiB for U16 wide). It double counted
aliased storage and added a tile count as bytes. The corrected values are 43.87 / 10.65 MiB (`docs/evidence.md`,
erratum).

One-time costs (graph 0, energy, RTX 3090):
- support selection: 2.8 s;
- normalization + lowering: 3.9 s;
- `from_checkpoint` total: 18.9 s, including the checkpoint load and envelope graph generation;
- `save_packed`: 0.5 s; `load_packed`: 0.4 s; packed file: 464 MB;
- construction peak allocation: 6.7 GiB.

## Evidence and limitations

See `docs/validation/cover_topk16_rtx3090_dx.md`. It holds the integration-head checks and the previous research
evidence, pinned to commit `f4c538a` of `interaction-coverage-k16-dx` and kept separate from the new checks.

Limitations:
- Quality results come from synthetic fine-tuning / diagnostic tasks on one GPU. They are not downstream-accuracy
  claims.
- The auto tiling is measured only on the RTX 3090 and only for the scope above; nothing is claimed for other GPUs.
- Atomic reductions make training not bitwise deterministic.
