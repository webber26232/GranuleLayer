# 100M-neuron sparse projection with fused (hierarchical) pooling and recompute backward — RTX 3090

A PyTorch layer that maps `x [B, 1024]` through **H = 100,000,000** logical neurons
(fan-in 8, every neuron evaluated every step) and pools them to `y [B, 1024]`, without
ever materialising the `[B, H]` hidden tensor (1.6 TB in BF16 at B = 8192) — neither in
forward nor in backward.  Neuron activations live only in registers of the kernel that
consumes them; backward recomputes them.

Everything in this document was measured on the machine it was developed on:
**RTX 3090 (GA102, sm_86, 82 SMs, 5 MB L2)** — note this card exposes **20 GiB**, not 24 GB —
driver 550.67, CUDA 12.1 `nvcc`, PyTorch 2.7 (CUDA 12.8 runtime), Triton 3.3.
Measured tables: [`results/TABLES.md`](results/TABLES.md) (generated from `results/*.json`).
The analysis and answers to the research questions are in [REPORT.md](REPORT.md).

## Quick start

```python
import torch
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.optim import FusedAdamW

layer = SparsePoolingLayer(
    input_dim=1024, hidden_dim=100_000_000, output_dim=1024, fan_in=8,
    shared_neurons=8,                      # Mode B (C neurons share 8 indices); None -> Mode A
    pooling={"levels": [
        {"type": "topk_mean", "k": 8, "neurons_per_shard": 16384, "activation": "relu"},
        {"type": "logmeanexp", "temperature": 0.5},
    ]},
    checkpoint_policy="auto",               # minimal_stats | recompute_stats | save_winners | auto
    dtype=torch.bfloat16, device="cuda",
)
layer.enable_main_grad()                    # FP32 grads accumulated in-kernel (no transient copy)
opt = FusedAdamW([layer.weight, layer.bias], lr=1e-4, moments="bf16")

x = torch.randn(8192, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = layer(x)                                # [8192, 1024] float32
y.float().square().mean().backward()
opt.step()                                  # also zeroes main_grad
print(layer.last_info)                      # backends, policy actually used, workspace, checkpoint bytes
```

Single level: `pooling="max"` (or `"topk_mean"`, `"mean"`, `"logmeanexp"`, `"power_mean"`,
`"softmax_weighted"`, or a dict with parameters).  Input `[batch, seq, 1024]` is flattened.

Requirements: CUDA GPU (sm_80+; tuned for sm_86), PyTorch ≥ 2.3, Triton ≥ 3.0, and `nvcc`
for the native kernels (compiled on first use, ~2 min, cached in
`kernels/cuda/build/`; set `NVCC=` if not on `PATH`).  Without `nvcc` everything runs on the
Triton path (`backend="triton"`), which is functionally complete but slower.

### Running the full 100M configuration

```bash
# tests (≈ 1 min, 377 tests)
cd sparse_pooling/tests && python -m pytest -q
# one full-scale training-step benchmark (≈ 1–2 min per case)
python sparse_pooling/benchmarks/bench_end_to_end.py --hidden 100000000 --batch 8192 --C 8 --suite final --optimizer
# everything that produced results/ (≈ 3 h)
bash sparse_pooling/benchmarks/run_all.sh python && python sparse_pooling/benchmarks/make_report.py
```

Memory at H = 100M, B = 8192 (C = 8): BF16 weights 1.6 GB + bias/indices 0.4 GB, FP32
`main_grad` 3.6 GB, Adam moments 3.6 GB (BF16) or 7.2 GB (FP32), optional FP32 master
weights 3.6 GB; checkpoints are 32–288 MiB for single-level models.  Measured peaks
(`results/TABLES.md`): 8.6–9.8 GiB with BF16 moments, 17.2 GiB with FP32 moments + master
weights.  Two-level models with Top-K level-1 features need 12.5–16.2 GiB with BF16 moments
and do not fit the 20 GiB card with FP32 moments + master weights.  If saved winner ids would
exceed 2 GiB (`autograd.CHECKPOINT_BUDGET`) the layer recomputes winners in token chunks.

## Layout of the repository

```
sparse_pooling/
  config.py            LevelSpec / PoolingConfig (semantic hierarchy) / PhysicalPlan (execution only)
  layout.py            group partition, padding, global neuron ids, init, fragment-packed storage
  layer.py             SparsePoolingLayer (nn.Module)
  autograd.py          Level1Runtime + SparsePoolFunction / SemanticPoolFunction, checkpoint policies
  optim.py             FusedAdamW (reads/zeroes FP32 main_grad, BF16|FP32 moments, master weights or SR)
  memory.py            analytic memory model
  kernels/
    triton_fused.py    Triton: fused forward (all operators, Mode A SIMT + Mode B tl.dot),
                       generic recompute backward, sparse winner-event backward
    ops.py             launches, state merge/finalize, backward coefficients, workspace budget
    cuda_ops.py        ctypes bridge + incremental nvcc build of the native kernels
    cuda/              native sm_86 kernels (m16n8k8 forward, Top-K, dX and dW recompute backward)
  reference/pytorch_reference.py   materialising reference (small H), tie rules identical
  tests/               projection, operators, hierarchy, backward, recomputation, precision, optimizer
  benchmarks/          end-to-end, sweep, trade-off, memory, SASS inspection, sanitizer, ncu script
  results/             JSON + logs + TABLES.md
```

## Design

### Neuron layout and semantics (`layout.py`)

* `G = 1024` output groups; group `g` owns `n_g = H//G + (g < H%G)` valid neurons
  (97,656 or 97,657 at H = 100M).  Valid slot `s` of group `g` is global neuron
  `g*(H//G) + min(g, H%G) + s`, so valid neurons enumerate exactly `range(H)`.
* Canonical storage (spec): indices `[G, NB, 8]` int16, weights `[G, NB, C, 8]` BF16
  (K-contiguous = the mma B-fragment order), bias `[G, NB, C]`; `NB = 12208` at C = 8.
  Padded slots have zero weights and are masked in every kernel (forward and backward).
* **Fragment-packed storage** (default with the CUDA backend): the same values permuted so
  that each lane fetches the B fragments of 4 consecutive (block, n-tile) fragments with one
  16-byte load (weights `[G, NB·NT/4, 32, 4, 2]`, bias and indices interleaved likewise).
  Measured 2.0× faster forward at C = 8 (the canonical layout is limited by the number of
  L2 line requests: ~0.11 lines/clk/SM, not by bytes).  The dW accumulator fragment of the
  backward lands exactly on this layout, so gradient write-back is a coalesced plain store.
  Pack/unpack are pure permutations (`layout.pack_* / unpack_*`), so the reference consumes
  either layout and its autograd returns gradients in the parameter's layout.
* Indices are distinct and sorted inside each 8-record; Mode A stores 8 indices per neuron.

### Forward

Native CUDA (`kernels/cuda/sparse_pool_fwd.cuh`, Mode B): CTA = (token tile of 16 or 32 tokens,
pooling unit, optional split).  The token tile of X is staged as a swizzled **feature-major
slab in shared memory** (`cp.async`), so the 8-feature gather is one `ldmatrix.x4.trans` per
block for 32 tokens.  `mma.sync.m16n8k8` BF16→FP32 computes `Z = X_g W + b` with the bias
entering as the accumulator input; a templated epilogue folds the 4 accumulator values per
lane into the per-row pooling state (registers).  Quad shuffles + one shared-memory pass
merge the per-lane states; one compact state per (token, unit, split) is written.  Weights are
prefetched one or two work units ahead.  Top-K uses a separate kernel (below).

Triton (`kernels/triton_fused.py`): every operator, Mode A (SIMT gather/FMA) and Mode B
(`tl.dot` with K = 16: two index blocks per dot with a block-diagonal weight tile — half of
each MMA is structural zeros, the minimum K Triton supports for BF16).  Used for Mode A,
Top-K with k = 32, GELU neuron activations, the canonical layout, and as the reference
implementation of the streaming states.

Streaming states (identical semantics in both backends, merged across splits on the host):

| Pooling | state | merge |
|---|---|---|
| Max | max (+ winner slot) | max; tie → smaller id (exact) |
| Top-K mean | k (value, id) keys (int64 order-preserving) | top-k of union (exact) |
| Mean(act) | sum (count static) | sum |
| LogMeanExp | m (base-2, lazy max with 2^8 slack), s = Σ 2^(u−m) | rescale to common m |
| Power mean | m = max a, s = Σ (a/m)^p | rescale by (m_i/M)^p |
| Softmax-weighted | m, s, t = Σ a·2^(u−m) | rescale |

Top-K (exact streaming selection, CUDA, k ≤ 16): pass 1 computes per-lane row maxima at Max
cost; lanes cover disjoint neurons, so the k-th largest of a row's lane maxima is a valid
lower bound of the row's k-th largest value.  Pass 2 keeps an exact per-lane top-k list
(branch-free insertion — data-dependent branches made the compiler demote the lists to
local memory) and inserts only elements ≥ the shared bound (`atomicMax` on an
order-preserving int encoding).  The final merge keeps only entries ≥ the bound
(compact per-row buffers, exhaustive fallback on overflow).  Ties: smaller neuron id.

### Backward (recompute; nothing of size `[B, H]` exists)

`ctx` keeps x, W, b (references), the pooled pre-activation `[B, U]` and, by policy,
winner slots (Max/Top-K, `save_winners`) or the log2-normaliser (softmax-weighted,
`minimal_stats`); `recompute_stats` keeps nothing and re-runs the streaming reduction.
LogMeanExp recovers its normaliser from the saved output; power mean needs only y.

* **Max / Top-K** — sparse winner events (Triton): one event per (token, unit, winner);
  z of the winner is recomputed (8 MACs) for `act'(z)`; gradients by FP32 atomics
  (8.4M events at B = 8192 → ~13 ms).
* **Smooth operators** (CUDA): two kernels, because the two reductions want opposite loop
  orders and both must stay on chip:
  * `k_dx` token-stationary: CTA = (16 tokens, ~128 groups).  Recompute z (m16n8k8),
    dZ from compact coefficients, `dA = dZ Wᵀ` on tensor cores (Wᵀ via `movmatrix`), scatter
    into a `[1024 × 16]` FP32 slab in shared memory, one global flush per CTA.  GA102 has no
    native FP32 shared-memory atomics (`atomicAdd` compiles to a CAS loop); tokens t and t+8
    are interleaved so each lane updates its 4 values with two 64-bit CAS.
  * `k_dw` weight-stationary: CTA = chunk of work units of one unit, loops over all tokens
    with a double-buffered slab.  Recomputes `zᵀ = W X_gᵀ` (the packed W fragment is directly
    the A operand, the forward's `ldmatrix.trans` registers the B operand), so `dZᵀ` comes
    out in exactly the A layout of `dWᵀ = dZᵀ X_g` — no transposes.  At C = 8 the structurally
    empty neuron rows 8–15 carry the lo half of the split dZ, giving hi+lo precision for the
    price of one MMA.  dW/db live in registers for the whole kernel and are written once
    (exclusive ownership: no atomics, deterministic).
* dZ enters the tensor-core products as hi + lo BF16 pair (`grad_precision="split"`,
  ~16-bit mantissa, gradients within ~3e-6 of FP32 autograd) or a single BF16
  (`"bf16"`, the usual mixed-precision semantics, ~1.5e-3).  Accumulation is FP32 always.
* Mode A / canonical layout / GELU: generic Triton recompute backward (FP32 atomics).
* Gradients go to persistent FP32 `main_grad` buffers when enabled (returned `None` to
  autograd), otherwise to temporaries returned in the parameter dtype.

### Hierarchies (`config.py`)

* `PhysicalPlan` — execution only: token tile, warps, input layout, number of **splits**
  (programs sharing one unit, merged by state; chosen automatically under a workspace
  budget, default 512 MiB: `B·U·splits·state_bytes` is computed before allocating and
  exceeding it raises `WorkspaceBudgetError`), Triton token-/weight-stationary scheduling.
  Never changes the function.
* `PoolingConfig.levels` — semantic: level 1 pools shards of `neurons_per_shard` neurons
  into features `[B, G, S]`; an optional transform (`activation`, learned `affine`, learned
  `gate`) and a level-2 operator (any pooling or learned `weighted_sum`) produce `y`.
  Level 2 runs in PyTorch on the compact tensor.  `level1_checkpoint="save"` keeps the
  level-1 statistics, `"recompute"` re-runs the level-1 kernel in backward.
  Max → Max without a non-monotone transform raises a warning (it is global Max).

## Verification

* `tests/` — 377 tests: layout/partition (exactly `range(H)`), pack/unpack, raw projection,
  every operator × both modes × C × batch sizes × splits × backends vs the reference, winner and
  Top-K index agreement, deterministic ties, state merges, the hierarchy matrix (physical ≡ flat,
  semantic ≡ explicit reference and ≠ flat), gradients of x / W / b / transform parameters vs
  autograd for all operators and policies, saved vs recomputed statistics, bitwise-reproducible
  recomputation, no `[B, H]` allocation (peak-memory bound), in-place parameter modification
  rejected, FP64 finite differences, numerical stability at large magnitudes, LME limits,
  optimizer vs `torch.optim.AdamW`.
* Compute Sanitizer (`benchmarks/run_sanitizer.sh`, log in `results/sanitizer.log`): memcheck,
  initcheck and synccheck report 0 errors.  racecheck reports write-write hazards only inside a
  Triton-generated cross-warp reduction of the Mode-A backward (`tl.sum` before an atomic; threads
  holding replicated values); the native kernels are race-free.
* Tensor cores: `benchmarks/inspect_sass.py` (results/sass_inspection.json) — `HMMA.1688.F32.BF16`
  in the forward/Top-K kernels, `HMMA.16816.F32.BF16` in the gradient products, Triton PTX
  `mma.sync.aligned.m16n8k16`.
* Nsight Compute: hardware counters are admin-only on this machine (`ERR_NVGPUCTRPERM`,
  `RmProfilingAdminOnly=1`, no sudo), so no counter-based profiles were collected.
  `benchmarks/profile_ncu.sh` profiles the hot kernels with the relevant metrics when run with
  sudo.  Bottlenecks in this work were established with micro-benchmarks and ablation builds
  (see REPORT.md); DRAM/L2 byte counts quoted are analytic.
