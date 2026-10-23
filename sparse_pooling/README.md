# 100M-neuron sparse projection with fused (hierarchical) pooling and recompute backward

A PyTorch layer that maps `x [B, 1024]` through **H = 100,000,000** logical neurons
(fan-in 8, every neuron evaluated every step) and pools them to `y [B, 1024]`, without
ever materialising the `[B, H]` hidden tensor (1.6 TB in BF16 at B = 8192) — neither in
forward nor in backward.  Neuron activations live only in registers of the kernel that
consumes them; backward recomputes them.

GPU support: the native CUDA kernels need an NVIDIA GPU with sm_80 or newer. They are tuned and
hardware-validated on an **RTX 3090**; every other GPU runs a conservative capability-based policy whose
performance is unmeasured (support matrix: [docs/gpu_portability.md](../docs/gpu_portability.md)).

Everything in this document was measured on the machine it was developed on:
**RTX 3090 (GA102, sm_86, 82 SMs, 5 MB L2, 24 GiB, 370 W limit)** — results were re-measured
after a hardware repair (the card previously exposed only 20 GiB at 744 GB/s; now 23.7 GiB
usable to PyTorch, 819 GB/s device-to-device copy) —
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

Requirements: an NVIDIA GPU with compute capability ≥ 8.0, PyTorch ≥ 2.3, Triton ≥ 3.0, and a CUDA
toolkit whose `nvcc` can target the GPU (12.1 covers sm_80 – sm_90) and whose CUDA version the driver
supports. The native library is compiled for the GPU's own target on first use (~3–4 min with 6 CPU
cores) and cached under `~/.cache/granulelayer/cuda` (`SPARSE_POOL_CACHE_DIR`; `NVCC=` selects the
compiler). It is loaded through ctypes, so no PyTorch extension build is needed. Importing the package
never compiles or touches the GPU. Without `nvcc`, or on a GPU older than sm_80, `backend="auto"` runs
the Triton path (functionally complete but slower) and records why in `last_info["backend_fallback"]`.
Tested combination: driver 550.67, nvcc 12.1.66, PyTorch 2.7.0+cu128, Triton 3.3.0.

### Execution policy: inspecting, forcing, deterministic

```python
info = layer.explain(8192)        # what would run: device, build target, profile, per-phase kernels
info["profile"]                   # {'name': 'rtx3090', 'status': 'hardware-validated', ...}
layer.last_info["execution"]      # the same record for the last call that actually ran

# debugging / other GPUs: the conservative capability-based policy (legality is always checked)
layer.plan = PhysicalPlan(tuning_profile="generic_cuda")      # or SPARSE_POOL_PROFILE=generic_cuda
# bitwise-repeatable training (input-supergroup layers; see "Deterministic execution" below)
layer.plan = PhysicalPlan(reduction="deterministic")          # or torch.use_deterministic_algorithms(True)
```

Adding a profile for a new GPU: [docs/adding_a_gpu.md](../docs/adding_a_gpu.md). Module responsibilities:
[docs/architecture.md](../docs/architecture.md). Migration from the experiment branches:
[docs/migration.md](../docs/migration.md).

### Running the full 100M configuration

```bash
# tests (≈ 7 min on the RTX 3090 after the first build; CPU-only host tests run without a GPU)
cd sparse_pooling/tests && python -m pytest -q
# one full-scale training-step benchmark (≈ 1–2 min per case)
python sparse_pooling/benchmarks/bench_end_to_end.py --hidden 100000000 --batch 8192 --C 8 --suite final --optimizer
# everything that produced results/ (≈ 3 h)
bash sparse_pooling/benchmarks/run_all.sh python && python sparse_pooling/benchmarks/make_report.py
# deterministic-training repeatability (independent processes; exits nonzero on any failure)
bash sparse_pooling/benchmarks/run_det_training.sh python /scratch/rec small /scratch/out
# Compute Sanitizer over the native kernels (parallel runs: --jobs N, default CPUs - 1; per-run logs and
# summary.tsv in --out DIR; exits nonzero if any run fails, see the header of the script)
bash sparse_pooling/benchmarks/run_sanitizer.sh --out /scratch/sanitizer python
```

Memory at H = 100M, B = 8192 (C = 8): BF16 weights 1.6 GB + bias/indices 0.4 GB, FP32
`main_grad` 3.6 GB, Adam moments 3.6 GB (BF16) or 7.2 GB (FP32), optional FP32 master
weights 3.6 GB; checkpoints are 32–288 MiB for single-level models.  Measured peaks
(`results/TABLES.md`): 8.6–9.3 GiB with BF16 moments, 17.2 GiB with FP32 moments + master
weights.  Two-level models with Top-K level-1 features need 12.5–16.2 GiB with BF16 moments;
with FP32 moments + master weights the 6,144-unit variant fits (19.2 GiB) but the 49,152-unit
variants need ~23 GiB and run out of memory on this 24 GiB card with a desktop attached.  If saved winner ids would
exceed 2 GiB (`autograd.CHECKPOINT_BUDGET`) the layer recomputes winners in token chunks.

## Layout of the repository

```
sparse_pooling/
  config.py            LevelSpec / PoolingConfig (semantic hierarchy) / PhysicalPlan (execution only)
  layout.py            group partition, padding, global neuron ids, init, fragment-packed storage
  layer.py             SparsePoolingLayer (nn.Module), explain()
  autograd.py          Level1Runtime + SparsePoolFunction / SemanticPoolFunction, checkpoint policies
  supergroups.py       input-supergroup connectivity: generator, validator, bucket statistics, work lists
  optim.py             FusedAdamW (reads/zeroes FP32 main_grad, BF16|FP32 moments, master weights or SR)
  memory.py            analytic memory model
  backends/
    capabilities.py    device / toolchain facts, ISA feature table (no tuning)
    profiles/          rtx3090.py (measured defaults + provenance), generic_cuda.py (conservative)
    dispatch.py        requested PhysicalPlan -> ResolvedPlan (legality, precedence, reasons)
    cuda/              variants.py (kernel family), build.py (bundles, cache), runtime.py (ctypes, launches)
  kernels/
    triton_fused.py    Triton: fused forward (all operators, Mode A SIMT + Mode B tl.dot),
                       generic recompute backward, sparse winner-event backward
    ops.py             launches, state merge/finalize, backward coefficients, workspace budget
    cuda_ops.py        compatibility facade of the native backend
    cuda/              native kernels (sm_80+): forward, Top-K, dX / dW recompute backward,
                       input-supergroup kernels, deterministic reduction; experimental/, diagnostics/
  reference/pytorch_reference.py   materialising reference (small H), tie rules identical
  tests/               semantics, kernels, dispatch / profiles / build (CPU), harness, codegen
  benchmarks/          maintained tools: timing, repeatability, sanitizer, SASS, decision matrix
  results/             JSON + logs + TABLES.md of the first implementation's tables (re-measured)
docs/                  architecture, GPU portability, adding a GPU, migration, evidence index, validation
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
  (8.4M events at B = 8192 → ~11 ms).
* **Smooth operators** (CUDA): two kernels, because the two reductions want opposite loop
  orders and both must stay on chip:
  * `k_dx` token-stationary: CTA = (16 tokens, ~128 groups).  Recompute z (m16n8k8),
    dZ from compact coefficients, `dA = dZ Wᵀ` on tensor cores (Wᵀ via `movmatrix`), scatter
    into a `[1024 × 16]` FP32 slab in shared memory, one global flush per CTA.  GA102 has no
    native FP32 shared-memory atomics (`atomicAdd` compiles to a CAS loop); tokens t and t+8
    are interleaved so each lane updates its 4 values with two 64-bit CAS.  The 96 KiB of
    shared memory hold it at 1 CTA/SM; at C = 64 it runs 32 warps per CTA (64 registers fill
    the register file), 9–14 % faster than 16 (`PhysicalPlan.cuda_dx_warps`).
  * `k_dw` weight-stationary: CTA = chunk of work units of one unit, loops over all tokens
    with a double-buffered slab.  Recomputes `zᵀ = W X_gᵀ` (the packed W fragment is directly
    the A operand, the forward's `ldmatrix.trans` registers the B operand), so `dZᵀ` comes
    out in exactly the A layout of `dWᵀ = dZᵀ X_g` — no transposes.  At C = 8 the structurally
    empty neuron rows 8–15 carry the lo half of the split dZ, giving hi+lo precision for the
    price of one MMA.  dW/db live in registers for the whole kernel and are written once
    (exclusive ownership: no atomics, deterministic).  At C = 64 (packed, feature-major X,
    sm_86) the default `k_dw_idx` stages only the 8 index rows of each work unit into a
    per-warp `cp.async` ring instead of all 1024 features (192 vs 768 GiB of logical staging
    at H = 100M), with warp-level instead of CTA-level synchronisation; results are bitwise
    identical to `k_dw` and 1.4× faster (`PhysicalPlan.cuda_dw_staging = "full"` restores
    the original kernel).  Details: [MEMORY_TRAFFIC_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/MEMORY_TRAFFIC_OPTIMIZATION.md).
  * dZ epilogues (`PhysicalPlan.cuda_{dx,dw}_epilogue`, default `"auto"`): for Mean, LME and
    integer-p power mean the token-validity select and the multiply by act'(z) are dropped; for
    Mean, dZ is one of two values per token, so its BF16 hi/lo split is computed per token and
    each element needs only a bit mask.  Gradients are bitwise identical to the generic path
    (FTZ and non-finite coefficients included, via a device-side fallback flag).  Details:
    [BACKWARD_COMPUTE_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/BACKWARD_COMPUTE_OPTIMIZATION.md).
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

* `tests/` — 2,800+ tests: layout/partition (exactly `range(H)`), pack/unpack, raw projection,
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

## Input-supergroup connectivity (optional architecture)

`SparsePoolingLayer(..., connectivity=ConnectivityPlan(input_supergroup_dim=S, partition_rounds=R))`
restricts every index record to one of Q = R·D/S seeded input sets (`supergroups.py`: generator,
validator, compact metadata saved in the state dict).  The default stays unrestricted.  With
`PhysicalPlan(cuda_input_supergroup_backend="auto")` the measured grouped Mean kernels run where they
are faster (H = 100M, M = 8192, C = 8, S = 16: step 4.85 s → 2.72 s; C = 64: 1.76 s → 1.70 s); see
[INPUT_SUPERGROUP_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/INPUT_SUPERGROUP_OPTIMIZATION.md).

The grouped C = 8 dX local accumulation has conflict-reduced slab layouts
(`PhysicalPlan.cuda_grouped_dx_scatter`; `auto` selects the measured variant, the baseline elsewhere):
at H = 100M, M = 8192 the C = 8 step drops another 10–11 % (S = 16: 2.80 s → 2.51 s, S = 32:
3.10 s → 2.76 s, same-session A/B) with bitwise-identical per-CTA arithmetic; see
[C8_DX_SCATTER_OPTIMIZATION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/C8_DX_SCATTER_OPTIMIZATION.md).  Benchmarks restore the complete training state
(`benchmarks/train_state.py`).

## Deterministic execution (`PhysicalPlan.reduction`)

The fast grouped kernels sum across CTAs with FP32 atomics (forward: one per (token, bucket);
dX: one per (feature, token) per CTA, plus a shared-memory CAS in the S = 64 `shared_t64` dX plan),
so repeated runs differ in the last bits.  `PhysicalPlan(reduction="deterministic")` gives every
reduction a fixed owner and order: the grouped forward and dX store each CTA's partial into
workspace rows only that CTA writes, and a second kernel (`kernels/cuda/sg_det.cu`) sums each
output / input feature's contributors in a static order (built once from the work lists); dX uses
warp-private shared-memory copies; dW / db already have exclusive owners.  Repeated runs with the
same graph, inputs, parameters, optimizer and RNG state, resolved plan, device and toolchain are
bitwise identical (outputs, dX, dW / db, optimizer updates).  No claim is made across GPU models,
compiler / driver versions, precision modes or different reduction trees (deterministic and fast
results differ in the last bits, as any two summation orders do).

```python
import torch
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.supergroups import ConnectivityPlan

lay = SparsePoolingLayer(1024, 1_000_000, 1024, shared_neurons=8, pooling="mean", device="cuda",
                         connectivity=ConnectivityPlan(64, 16),
                         plan=PhysicalPlan(reduction="deterministic"))   # or "fast" / "auto"
x = torch.randn(1024, 1024, device="cuda").bfloat16()
y = lay(x); y.float().square().sum().backward()
print(lay.last_info["reduction"], lay.last_info["grouped_dx"]["reduction"])
```

`reduction="auto"` (the default) runs `"deterministic"` under
`torch.use_deterministic_algorithms(True)` and `"fast"` otherwise; an explicit `"fast"` under the
global flag raises (warns with `warn_only=True`).  The partial workspace per stage is capped by
`deterministic_workspace_bytes` (default 1 GiB, `config.DET_WORKSPACE_DEFAULT`): larger problems run
in token windows.  The window does not change the bits (each destination's sum has the same
contributors in the same order); a cap below one token tile raises — there is never an atomic
fallback.

| Layer / plan | `reduction="deterministic"` |
|---|---|
| input-supergroup layers, packed Mode B C = 8 or 64, Mean(ReLU) / Mean(identity), split or BF16 gradients, `input_layout="feature_major"`, B % 8 == 0 | supported (forward grouped or generic, dX grouped, dW grouped or generic) |
| grouped dX `auto` | warp-private copies; C = 8, S = 64 → `private_t32` (no fast-mode threshold); other S: the automatic tiles with `cuda_grouped_dx_reduction = 1` |
| explicit `cuda_grouped_dx_plan="private_t32"`, explicit private tiles / scatters | honoured |
| explicit `shared_t64`, `cuda_grouped_dx_reduction > 1` | `ValueError` (shared-copy CAS) |
| `cuda_grouped_dx="generic"` / layers without connectivity (k_dx), Triton backend | `ValueError` (FP32 atomics) |
| `cuda_grouped_backward="fused" / "fused_recompute"` | `ValueError` (atomic dX flush); `auto` is separate |
| other pooling operators, semantic hierarchies, token-major input, B % 8 != 0 | `ValueError` |
| non-finite upstream gradients | the in-kernel generic epilogue of the same kernels (one path; finite elements bitwise repeatable; inf / NaN positions reproduced, NaN payload bits not promised) |

Cost (RTX 3090, C = 8, H = 100M, M = 8192, paired complete step): +0.34 / +0.21 / +0.20 % for
S / R = 16 / 4, 32 / 8, 64 / 16 (+0.25 % BF16 gradients), almost all of it the forward reduce
(about 10 ms: 8 GiB of partials in 8 token windows); peak allocated memory +0.8 / +0.8 / +1.0 GiB.
Smaller S = 64 graphs whose fast plan is `shared_t64` cost more (+11 to +24 % at H <= 10M), as
does C = 64, S = 64 (+12.8 %: private copies need 32-token tiles).

`PhysicalPlan.cuda_grouped_dw_projection` (`"auto"` = `"padded"`, `"compact"`): the compact C = 8
Mean(ReLU) dW projection computes z with one `mma.m16n8k8` in dX orientation and transposes the
ReLU mask with `movmatrix`.  It is bitwise equal to the padded projection but 3-4 % slower in dW
(8 % with BF16 gradients) on the RTX 3090, so it is explicit / experimental;
`last_info["grouped_dw"]["projection"]` records the choice.

See [DETERMINISM_METRICS_DW_PROJECTION.md](https://github.com/webber26232/GranuleLayer/blob/2b9a84d66c239934fd58ec45d7736804c78ccba2/sparse_pooling/results/DETERMINISM_METRICS_DW_PROJECTION.md) for the ownership map, overheads and the
production H = 100M repeatability evidence.
