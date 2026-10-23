# GPU portability: what is algorithm, what is CUDA, what is RTX 3090

GranuleLayer's speed comes from a campaign of optimizations measured on one RTX 3090 (sm_86).
This document separates them so that another GPU can reuse what is reusable without inheriting
what was only measured. Module boundaries are described in [architecture.md](architecture.md),
and the contributor recipe is [adding_a_gpu.md](adding_a_gpu.md).

## 1. Four statuses that are never implied by each other

| Status | Meaning | Where it is established |
|---|---|---|
| **compiles** | nvcc accepts the sources for a target | `backends/cuda/build.py`, `tests/test_isa_requirements.py` (ptxas probes), `tests/test_build.py` |
| **legal** | the device has the instructions and the launch fits its limits | `backends/capabilities.py` (features), `backends/dispatch.py` (resource formulas against the device's own opt-in shared memory), `kernels/cuda/launch.cuh` (the driver's kernel attributes, checked before each first launch per device) |
| **numerically tested** | the test suite / training repeatability passed on that hardware | profile `validation.numerics`, the evidence index |
| **performance-measured** | the tuning choices were measured on that hardware | profile `validation.performance`, the evidence index |

### Profile matching

"sm_86" is not "RTX 3090": an RTX 3080 or A10 shares the compute capability, but it has a different SM
count, memory system and power limit, so the RTX 3090 measurements say nothing about it. Under
`tuning_profile="auto"` (`PhysicalPlan.tuning_profile`, or `SPARSE_POOL_PROFILE` when the plan says
`auto`), `backends/profiles.select` takes the first registered measured profile whose matcher accepts
the device, else `generic_cuda`. The RTX 3090 matcher (`profiles/rtx3090.is_rtx3090`) requires all of:
the exact board name, compute capability 8.6, 82 SMs and 99 KiB opt-in shared memory per block. Naming a
profile forces it; forcing `rtx3090` on another device is reported as `user-forced-unvalidated`. A profile
only chooses among legal options: the resolver checks every choice against the device's own limits.
The selected profile, why it matched, and its status are in `layer.explain(batch)["profile"]` and in
every `last_info["execution"]`.

## 2. Instruction requirements (verified)

These come from the PTX ISA "Target ISA Notes", and the installed toolkit (nvcc / ptxas 12.1.66) checks
them: `tests/test_isa_requirements.py` compiles each primitive at its declared minimum target and checks
that ptxas rejects it one architecture below.

| Primitive | Used by | Minimum target |
|---|---|---|
| `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` | forward, recompute, dW projection | sm_80 (PTX 7.0) |
| `mma.sync.aligned.m16n8k16 ... bf16 ... f32` | gradient products | sm_80 (PTX 7.0) |
| `cp.async.cg.shared.global`, `commit_group`, `wait_group`, `wait_all` | slab staging, rings | sm_80 (PTX 7.0) |
| `ldmatrix.sync.aligned.m8n8.{x2,x4}[.trans].shared.b16` | X gathers | sm_75 (PTX 6.5) |
| `movmatrix.sync.aligned.m8n8.trans.b16` | Wᵀ fragments, compact dW mask | sm_75 (PTX 7.8) |
| `ex2.approx.ftz.f32`, `set.gt.ftz.u32.f32`, `mul.ftz.f32` | streaming states, FTZ masks | all targets |
| FP32 `atomicAdd` (global), 64-bit `atomicCAS` (shared), `__shfl_sync` | fast-mode reductions, slab CAS, transposes | all supported targets |

The kernel family therefore needs **sm_80 or newer** and 32-lane warps. `isa_sm80.cuh` refuses to compile
below sm_80. `mma.sync` remains available on sm_89 / sm_90 / sm_100-class targets. Hopper `wgmma` and
Blackwell `tcgen05` are different instruction families: they would be new kernel families with their own
(architecture-specific, `sm_90a`-style) targets, which these kernels do not use.

## 3. Optimization inventory

Categories: **A** algorithm / semantics, **I** CUDA ISA-family implementation, **R** device resource
constraint, **P** measured performance policy, **N** correctness- or compiler-sensitive behaviour.
"Home" is where the code lives after the refactor. "Validated" means hardware-tested on the RTX 3090
unless stated otherwise.

| # | Optimization / dependency | Source | Effect | Cat. | HW / ISA | Resources | Compiler assumptions | Evidence | Home | Fallback | Validated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mode B structured connectivity (C neurons share 8 indices) | `layout.py` | one index record per C neurons; MMA-friendly | A | none | — | — | REPORT.md | core | Mode A (Triton) | yes |
| 2 | Input supergroups (`ConnectivityPlan`), seeded generator, validator, compact metadata in state dict | `supergroups.py` | restricts records to Q supergroups; enables grouped kernels | A | none | host / device memory for generation | — | INPUT_SUPERGROUP | core | unrestricted graph | yes |
| 3 | Canonical bucket statistics (`bucket_statistics`, schema `granule.bucket_stats/2`, nonempty-bucket mean) | `supergroups.py` | descriptor of the scale rule; the denominator is authoritative | A | none | — | — | DETERMINISM §4 | core | — | yes (CPU) |
| 4 | Recompute backward, no `[M,H]` tensor in forward or backward | `autograd.py`, all kernels | bounded memory | A | — | — | — | REPORT.md | core | — | yes (peak-memory tests) |
| 5 | Streaming pooling states + split merges (Max, Top-K, Mean, LME, power, softmax-weighted) | `kernels/ops.py`, `sparse_pool_fwd.cuh`, Triton | single pass over neurons | A + I | — | registers per state | `ex2.approx.ftz` | REPORT.md | core + kernels | Triton | yes |
| 6 | Fragment-packed parameter storage (`pack_*`/`unpack_*`) | `layout.py` | 16-byte B-fragment loads; dW accumulator lands on the layout | I (physical layout of m16n8k8 fragments, 32 lanes) | sm_80 MMA layout | — | — | REPORT.md (2.0× fwd) | layout (state_dict format) | canonical layout (Triton) | yes |
| 7 | BF16 m16n8k8 forward, bias as accumulator input, swizzled feature-major X slab, `ldmatrix.trans` gather, `cp.async` staging | `sparse_pool_fwd.cuh` | tensor-core projection | I | mma bf16, cp.async (sm_80) | slab `D·16·MT·2` B | — | REPORT.md, SASS | kernels | Triton | yes |
| 8 | Weight prefetch depth per C (`Prefetch<C>::PD`), token tile / warps of the forward (`PhysicalPlan.cuda_block_m=32`, `cuda_warps=16` defaults) | `sparse_pool_fwd.cuh`, `config.py` | latency hiding | P (compile-time / plan default) | — | registers | — | REPORT.md | kernels / plan defaults | explicit plan fields | 3090 |
| 9 | Top-K exact two-pass streaming selection (lane-max bound, branch-free lists), k ≤ 16 | `sparse_pool_fwd.cuh` | exact Top-K at near-Max cost | A + I + R | sm_80 | static smem ≤ 256 B + slab; registers ∝ k | branch-free insertion (data-dependent branches demoted lists to local memory) | REPORT.md | kernels | Triton (k = 32, canonical) | yes |
| 10 | `k_dx` token-stationary dX with FP32 shared slab + 64-bit CAS scatter | `sparse_pool_bwd.cuh` | on-chip dX reduction | I + R | — | `D·96` B (96 KiB at D = 1024) → 1 CTA/SM | GA102 has no native FP32 shared atomics (CAS) | REPORT.md | kernels | Triton | yes |
| 11 | dX warps 32 at C = 64 when ≥ 32 work units per unit | `profiles/rtx3090.dx_warps` | +9–13 % dX at C = 64 | P | — | register file | — | MEMORY_TRAFFIC | rtx3090 profile | 16 warps (generic) | 3090 |
| 12 | `k_dw` weight-stationary, exclusive dW / db ownership, registers until one write | `sparse_pool_bwd.cuh` | deterministic dW without atomics | A + I | sm_80 | registers | — | REPORT.md | kernels | Triton (atomics) | yes |
| 13 | Split hi/lo dZ (BF16 pair), C = 8 low half in the structural rows 8–15 | `sparse_pool_bwd.cuh` (`split_bf16`, `dw_block`) | ~16-bit-mantissa gradients at one MMA | N + I | — | — | FP32 subtraction exact in FTZ mode | REPORT.md | kernels | `grad_precision="bf16"` | yes |
| 14 | Indexed k_dw staging (cp.async ring of the 8 index rows), C = 64, 2/3 stages; cfg 3 = 2 CTAs/SM register cap | `sparse_pool_bwd.cuh`, `profiles/rtx3090` | 1.4× dW at C = 64 | I (cp.async) + P (staging choice, stages, cfg) | sm_80 | ring `NW·ST·WPW·UB·8·16·2` B | `__launch_bounds__(256, 2)` | MEMORY_TRAFFIC | kernels + rtx3090 profile | full staging (generic) | 3090 |
| 15 | dZ epilogues `fast` / `mask` (per-token hi/lo, bit masks), device non-finite flag + guarded generic launch | `sparse_pool_bwd.cuh`, `numerics.cuh` | bitwise-equal faster epilogues | N (bitwise contract, FTZ, non-finite fallback) + P (`auto` choice) | — | — | `set.gt.ftz` mask consumed as data; `mul.ftz` not foldable | BACKWARD_COMPUTE | kernels + profile `epilogue` | generic epilogue | 3090 |
| 16 | FTZ compare hazard: a mask only tested for non-zero compiled to `FSETP.GT` without `.FTZ` (ptxas 12.1) | `numerics.cuh`, `diagnostics/diag_proj_c8.cu` | identical ReLU decisions in every path | N (compiler-sensitive) | — | — | ptxas 12.1 behaviour | SCALE_DISPATCH §6 | numerics.cuh + `tests/test_codegen.py` | — | 3090 / ptxas 12.1 |
| 17 | `--use_fast_math` (FTZ, approximate transcendentals) | `backends/cuda/build.py` | numerical contract of every kernel | N | — | — | part of the build identity | — | build | — | yes |
| 18 | Grouped (supergroup) forward / dX / dW kernels: S-feature slabs, work lists (items / chunks / units), chunk sizes 2048 / 4096 | `sparse_pool_sg.cuh`, `supergroups.SgExec` | 1.8× C = 8 step at H = 100M | A (work lists) + I + R (`sg_smem` formulas) + P (chunk sizes) | sm_80 | `D·2 + S·T·2 (+ copies·S·T·4)` B | — | INPUT_SUPERGROUP | kernels + variants + profile | generic kernels | 3090 |
| 19 | `SG_AUTO` per-(C, S) tiles and hybrid generic/grouped phases | `profiles/rtx3090.SG_AUTO` | measured fastest per phase | P | — | — | — | INPUT_SUPERGROUP | rtx3090 profile | generic profile (grouped, T = 64, fitted) | 3090 |
| 20 | C = 64 large-bucket rule (≥ 3 units / nonempty bucket: grouped dX S ≤ 64 split, dW S ≤ 16) | `profiles/rtx3090.SG_AUTO_C64_LARGE_BUCKETS` | −15 ms each at H = 100M | P | — | — | — | INPUT_SUPERGROUP | rtx3090 profile | — | 3090 |
| 21 | dX scatter layouts (swizzle4, float4 records, shuffle transpose, hoisted addresses, owner) | `sparse_pool_sg_dxs.cuh` | fewer bank conflicts, same order | I (warp-32 shuffles, 128-bit smem) | sm_80 | per-variant | — | C8_DX_SCATTER | kernels (production: sc 9, 10; others experimental) | baseline | 3090 |
| 22 | `swizzle4_capped`: ≤ 80 registers via `__launch_bounds__(256, 3)` | `sparse_pool_sg_dxs.cuh` (SC_SWZ4M) | 3 CTAs/SM for shared copies | P (occupancy target as a named instantiation) | — | 80 regs | launch bound | C8_DX_SCATTER | kernels (named instantiation) + rtx3090 table | uncapped `swizzle4` (experimental) | 3090 |
| 23 | `SG_AUTO_DX_SCATTER` choices | `profiles/rtx3090` | −10–11 % C = 8 step | P | — | — | — | C8_DX_SCATTER | rtx3090 profile | baseline scatter | 3090 |
| 24 | Whole dX plans `shared_t64` / `private_t32`; S = 64 rule (split ≥ 4.0, BF16 ≥ 2.0 units per nonempty bucket, M ≥ 1024) | `config.DX_PLANS`, `profiles/rtx3090.SG_AUTO_DX_PLAN` | scale-aware dX | P (thresholds) + A (plans) | — | — | — | SCALE_DISPATCH §5 | plans: config; rule: rtx3090 profile | `tiles` | 3090 |
| 25 | Deterministic reductions: per-CTA partials in exclusive rows, fixed-order `sp_sg_reduce`, warp-private dX copies, token windows under a 1 GiB cap | `sg_det.cu`, `runtime._sg_det_run`, `dispatch.det_reason` | bitwise repeatability | A (fixed-order partial reductions, exclusive ownership) + N (contract) + I | sm_80 | workspace ≤ cap | — | DETERMINISM §3, §5 | kernels + dispatch (precedence 1) | raises (never atomics) | 3090 |
| 26 | Compact C = 8 dW projection (`movmatrix` mask transposes) | `sparse_pool_bwd.cuh`, `experimental/sg_dw_c8_exp.cu` | bitwise = padded, 3–4 % slower dW | I (movmatrix) | sm_75+ / sm_80 | — | — | DETERMINISM §6.4 | experimental bundle | padded | 3090 (correct, slower) |
| 27 | Fused shared-projection backward | `sparse_pool_sg_bwdf.cuh`, `experimental/sg_bwdf_c8.cu` | correct, not faster | I | sm_80 | `sgf_smem` | — | SCALE_DISPATCH §7 | experimental bundle | separate | 3090 (correct, slower) |
| 28 | FusedAdamW (Triton): FP32 `main_grad` read / clear, BF16 / FP32 moments, SR counter, master weights; **load-before-store barrier** | `optim.py` | one pass optimizer | N (barrier: a race seen with Triton 3.3, a correctness rule for all devices) + Triton | Triton-supported GPU | — | `tl.debug_barrier` ordering | SCALE_DISPATCH (eab9f59) | core (optimizer) | `torch.optim.AdamW` | yes |
| 29 | Triton kernels (Mode A, GELU, canonical layout, Top-K k = 32, winner events) | `kernels/triton_fused.py` | functional completeness | I (Triton) | Triton target (`tl.dot` bf16: sm_80 for MMA) | — | Triton codegen | REPORT.md | kernels | — | 3090 |
| 30 | Physical splits fill the GPU (`8 × SMs` programs) under a workspace budget | `kernels/ops.choose_splits` | occupancy | R (SM count from the device) | — | budget 512 MiB | — | REPORT.md | ops (device SM count) | explicit `phys_splits` | 3090 (82 SMs) |
| 31 | Checkpoint policies, 2 GiB winner budget, chunked recompute | `autograd.py` | memory bounds | A | — | device memory | — | REPORT.md | core | recompute | yes |
| 32 | Shared-memory opt-in per kernel and device; static + dynamic accounting; thread limits | `kernels/cuda/launch.cuh` | legal launches | R | — | device attributes | — | this refactor | launch guard | error code before launch | 3090 |
| 33 | ctypes ABI (no PyTorch extension); struct layout + ABI version + bundle / kind checks | `abi.cu`, `backends/cuda/runtime.py` | toolchain-independent binding | I (host) | — | — | ctypes = C layout | this refactor | runtime | refuse load | yes |
| 34 | Per-target JIT build cache keyed by sources, flags, target, nvcc and host compiler | `backends/cuda/build.py` | reproducible builds | build | nvcc supporting the target | — | — | this refactor | build | actionable error / Triton | yes |

## 4. Support matrix

| Device class | compiles | legal | numerics tested | performance measured | Profile under `auto` |
|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 99 KiB opt-in) | yes (nvcc 12.1) | yes | **yes** | **yes** | `rtx3090` (hardware-validated) |
| RTX 3090 with `tuning_profile="generic_cuda"` | yes | yes | **yes** (`tests/test_device_runtime.py`; the whole suite under `SPARSE_POOL_PROFILE=generic_cuda`: no numerical failure, only assertions of RTX 3090 auto choices fail; medium training repeatability) | no | `generic_cuda` (forced) |
| Other sm_86 (RTX 3080/3070, A10, A40, …) | yes | yes (capability rules) | no | no | `generic_cuda` (capability-eligible, hardware-unvalidated) |
| sm_80 (A100, A30), sm_87, sm_89 (Ada), sm_90 (H100, non-`a` target) | production bundle compile-tested for sm_80 / sm_89 / sm_90 with nvcc 12.1 (all 1,881 kernels; docs/validation) | yes by capability (synthetic resolver tests) | **no** | no | `generic_cuda` |
| sm_100 / sm_103 / sm_120 / sm_121 (Blackwell) | not with nvcc 12.1 (actionable error); needs a newer toolkit. Target parsing and selection (native target, same-major fallback, overrides) are covered by CPU resolver tests with synthetic toolkits only; no build for these targets was run | — (capability rules only) | no | no | `generic_cuda` once built |
| < sm_80 (Turing, Volta, …) | the production sources refuse to compile | no | — | — | native backend unavailable → Triton (`backend="auto"`) or an error (`backend="cuda"`) |
| AMD / ROCm, CPU | no native backend | — | CPU: reference and host tests only | — | Triton where it supports the device; otherwise unsupported |

No support for A100, RTX 4090, H100 or Blackwell is claimed: compiling for them is not a validation.

## 5. Build targets, overrides and the build cache

Distribution is per-target JIT (`backends/cuda/build.py`): nothing is compiled at import; the first CUDA
operation on a device compiles the needed bundle for that device's target and caches it.

* **Target grammar.** Targets are plain `sm_<major><minor>` tokens, matched as a whole string. CUDA writes
  compute capability X.Y as `XY`, so the minor is the last digit and the major everything before it:
  `sm_86` = 8.6, `sm_90` = 9.0, `sm_100` = 10.0, `sm_120` = 12.0, `sm_121` = 12.1 (`parse_arch`).
  Comparisons use integer tuples. Malformed or partial tokens (`sm_100junk`, `sm_8`, `sm_86\n`, `10.0`)
  are rejected. Architecture-specific / family targets (`sm_90a`, `sm_100a`, `sm_100f`) are valid
  compiler targets, but their cubins carry their own compatibility rules and these kernels do not use
  them, so they are refused by name rather than stripped to the plain target. The toolkit's lists
  (`nvcc --list-gpu-code` / `--list-gpu-arch`) are kept as exact tokens (`capabilities.parse_gpu_list`).
* **Selection** (`resolve_target`): a device without the kernel family's instructions (pre-sm_80) gets no
  target. Otherwise the device's own target if nvcc lists it; else the newest listed plain target with
  the same major and a lower or equal minor (a cubin for X.Y runs on X.Z with Z ≥ Y); else an error that
  names the supported targets. There is no cross-major fallback and no PTX JIT path: the embedded
  `compute_XY` PTX of a build is never used to run on a newer architecture.
* **Override**: `SPARSE_POOL_ARCH=sm_<major><minor>` (or `--target` of
  `python -m sparse_pooling.backends.cuda.build`) must parse, be binary compatible with the device, be at
  least sm_80, and be listed by nvcc; otherwise it raises. A target that parses is not a promise that
  the kernels were tested there (§1).
* **Identity and cache.** A library's cache key hashes the build schema, bundle, kind, the exact target
  string, the effective flags (`-O3 --use_fast_math -lineinfo -std=c++17`, `-arch=<target>`), the
  `nvcc --version` text, the host compiler and the content of every unit and transitively included
  header. Different targets, toolkits or flags never share an artifact
  (`<cache>/<target>/lib/<bundle>-<kind>-<hash>/`; cache root `SPARSE_POOL_CACHE_DIR`, else
  `$XDG_CACHE_HOME/granulelayer/cuda`, else `~/.cache/granulelayer/cuda`). Extra flags
  (`SPARSE_POOL_NVCC_EXTRA`) make a `development` or `ablation` build with its own key, and may not set
  the target.

Tests: `tests/test_build.py` (grammar, toolkit-list parsing through a fake nvcc, selection and
diagnostics for sm_80 – sm_121 with synthetic devices and toolkits, cache identity per target). These
are resolver tests: no sm_100 / sm_120 hardware or Blackwell-capable toolkit was used.

## 6. Fallbacks and their limits

| Fallback | When | Correctness | Memory | Performance |
|---|---|---|---|---|
| `generic_cuda` profile (production kernels, simple choices) | any CUDA device without a matching measured profile; or forced | same kernels and numerical contracts as the RTX 3090 profile; the full suite passes with it forced on the RTX 3090 | same bounded-memory design (no `[M, H]` tensor); token tiles shrink to fit the device | unmeasured on every device; it skips the measured rules (scatter layouts, S = 64 plan, C = 64 large buckets, fast / mask epilogues, indexed dW staging) |
| Triton (`backend="auto"` → `triton`) | no nvcc, pre-sm_80 device, or a native phase that is illegal on the device (e.g. the generic `k_dx` needs `D × 96` B shared memory) | functionally complete (the only path for Mode A connectivity, GELU, Top-K k = 32), tested against the PyTorch reference by the suite on the RTX 3090; it reduces with FP32 atomics, so it is not bitwise repeatable: `reduction="deterministic"` raises instead of running it (under PyTorch `warn_only` it warns and runs fast) | no `[M, H]` tensor either | measured slower on the RTX 3090 (e.g. 1.9–2.2× slower forward for Max at C = 8, H = 10M; `sparse_pooling/REPORT.md` §5); unmeasured elsewhere |
| PyTorch reference (`reference/pytorch_reference.py`) | tests only | the definition the kernels are tested against | materialises the hidden activations: small H only | not a runtime path |

`last_info["backend_fallback"]` records why `backend="auto"` did not run the native kernels.

## 7. Numerical limits and determinism

* Determinism (`reduction="deterministic"`): when the environment, the graph, the inputs, the
  parameters, the optimizer and RNG state, the **resolved plan**, the device and the toolchain are all
  unchanged, every run gives the same bits. Different profiles can resolve different (legal) reduction
  trees. Examples are the generic profile's tiles, or another device's tile fitting. Such runs are
  numerically correct, and each is repeatable within its own profile. They are not bitwise equal to each
  other, and no cross-device bitwise identity is promised.
* Fast mode sums with FP32 atomics, so run-to-run bits vary.
* `grad_precision="split"` gives ~16-bit-mantissa dZ (within ~3e-6 of FP32 autograd); `"bf16"` gives
  the usual mixed precision (~1.5e-3).
* The compiler-sensitive FTZ rules (inventory #15–17) are part of the kernels and of the build flags,
  not of a profile: no profile may change them.

## 8. Known limitations

* The native backward needs the generic `k_dx` (`D × 96` bytes of shared memory) to be legal even when
  the grouped kernels would cover dX. On a device with a smaller opt-in limit (or for D > 1056 on the
  RTX 3090) the whole backward runs on Triton, and an explicit `grouped` request raises. Splitting
  backward legality per phase is a possible follow-up; it would change RTX 3090 decisions for D > 1056
  and was therefore not done here.
* Per-target JIT only: there is no prebuilt multi-target fatbin. The embedded `compute_XY` PTX is not used
  to run on other architectures (the runtime always builds for the device's own target, or a
  binary-compatible same-major one).
* Only one GPU was available, so the real two-device routing test (`tests/test_device_runtime.py`) skips
  here. Per-device code paths (launch guard slots, `torch.cuda.device` guards, per-device streams, the
  runtime rebuild on device migration) are covered by single-device and mocked tests.
