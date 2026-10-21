# Architecture

```
sparse_pooling/
  config.py, layer.py, layout.py, supergroups.py, autograd.py, optim.py, memory.py   semantic core
  reference/                      materialising PyTorch reference (tests)
  kernels/
    ops.py, triton_fused.py       Triton kernels + host helpers (state merges, coefficients, splits)
    cuda/                         native CUDA sources (see "Native sources")
    cuda_ops.py                   compatibility facade -> backends (no policy of its own)
  backends/
    capabilities.py               DeviceCapabilities (per device), ToolchainInfo (nvcc), ISA feature table
    profiles/                     TuningProfile; rtx3090.py (measured), generic_cuda.py (conservative)
    dispatch.py                   Resolver: requested PhysicalPlan -> immutable ResolvedPlan + reasons
    cuda/
      variants.py                 kernel family: instantiations, resource formulas, variant status
      build.py                    bundles, target selection, build identity, cache, locking
      runtime.py                  ctypes ABI, library registry, launches (CudaBackend)
  benchmarks/                     maintained tools (timing, repeatability, sanitizer, SASS, matrices)
  tests/
docs/                             this documentation, evidence index, validation summaries
```

## Responsibilities

| Layer | Owns | Must not contain |
|---|---|---|
| **Semantic core** (`config`, `layer`, `layout`, `supergroups`, `autograd`, `optim`) | public API, pooling semantics, connectivity / graph generation, canonical bucket statistics, logical shapes, state-dict format, optimizer correctness, the determinism contract (`resolve_reduction`) | GPU names, measured tables, shared-memory constants |
| **Capabilities** (`backends/capabilities.py`) | facts of the device the tensors live on (compute capability, SMs, warp size, shared-memory limits, registers, memory) and of the toolchain (nvcc version, emit-able targets, host compiler); the ISA feature table | preferences |
| **Kernel family** (`kernels/cuda/*.cu[h]`, `backends/cuda/variants.py`) | CUDA implementations, ISA primitives (`isa_sm80.cuh`), compiler-sensitive numerics (`numerics.cuh`), the launch guard (`launch.cuh`), compiled instantiations, shared-memory formulas, production / experimental / diagnostic status | which tiling is fastest on a device |
| **Tuning profiles** (`backends/profiles/`) | measured choices with provenance and scope (rtx3090) or conservative choices (generic_cuda); a matcher; validation status | legality checks, numerics |
| **Dispatch** (`backends/dispatch.py`) | precedence (numerics > legality > explicit overrides > profile > fallback), fitting to the device's limits, reasons, `ResolvedPlan` | kernel launches |
| **Build / runtime** (`backends/cuda/build.py`, `runtime.py`) | lazy per-target JIT builds, identity hashing, cache, ABI checks, the registry of loaded libraries, device guards and streams, executing a `ResolvedPlan` | decisions |

## Call flow

```
SparsePoolingLayer.forward
  -> layer._runtime()            once per (device, indices, plan object, graph):
       backend_for(device)       capabilities.query(device) -> build.resolve_target -> build (cached)
                                 -> runtime.NativeLibrary (ABI checked) -> CudaBackend (per target)
       Level1Runtime(...)        holds caps / profile selection lazily, SgExec work lists
  -> SparsePoolFunction / SemanticPoolFunction (autograd)
       rt.resolve_reduction(B)   PyTorch deterministic flag read on EVERY call (never cached)
       rt.resolve(B, "fwd")      Resolver.resolve -> ResolvedPlan (cached by B, mode, plan fields)
                                 -> last_info["execution"]
       backend.forward           under torch.cuda.device(x.device), stream of that device
       ... backward: rt.resolve(B, "bwd") -> backend.backward
```

`ResolvedPlan.as_dict()` and `SparsePoolingLayer.explain(batch)` contain: the backend, the device
(index, name, UUID, arch, SM count, opt-in shared memory, source), the build (target and why, kind,
library hash, flags, nvcc, host compiler, ABI, loaded bundles), the profile (name, version, forced,
match reason, validation status), the reduction mode and its reason, and per phase: the kernel,
tiles, warps, copies, chunk sizes, scatter, epilogues, staging, projection, shared memory and fallback
reasons. The legacy `last_info` keys (`grouped_dx`, `dx_epilogue`, …) are unchanged.

## Native sources and bundles

| Bundle | Units | Built |
|---|---|---|
| production | `abi.cu`, `sparse_pool.cu`, `fwd_*.cu`, `bwd_*.cu`, `sg_fwd_*`, `sg_dx_*`, `sg_dxs_c8_*` (sc 9 / 10), `sg_dw_*`, `sg_det.cu` | first CUDA call per target |
| experimental | `experimental/`: other scatter layouts, owner variant, fused backward, compact dW projection, mask dW control | first explicit request |
| diagnostics | `diagnostics/`: slab layout dumps, projection / ReLU-decision diagnostic | tests / tools only |

Headers: `common.cuh` (activations, power, BF16 helpers) includes `isa_sm80.cuh` (mma, ldmatrix,
movmatrix, cp.async; `#error` below sm_80) and `numerics.cuh` (FTZ compare / multiply, ex2). Kernel
templates live in `sparse_pool_{fwd,bwd,sg,sg_dxs,sg_bwdf}.cuh`; launch templates in `*_dispatch.cuh`,
all going through `launch.cuh`.

Wrong-result ablations (`SP_ABLATE_*` via `SPARSE_POOL_NVCC_EXTRA`) produce builds of kind
`ablation`: separate cache entries, a load-time warning, and `build.kind` recorded in every
`last_info["execution"]`. `det_training.py` reports the build kind of the implementation it validates.
