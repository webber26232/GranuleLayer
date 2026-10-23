# GPU-portability review fixes: report

| | |
|---|---|
| Reviewed snapshot | `gpu-portability-cleanup` @ `34a6433` (= `origin/gpu-portability-cleanup` when this work started; `origin/main` @ `300a268`, not merged in) |
| Fix commit (tested) | `fd99c2d` — every result below was produced from a fresh clone of exactly this commit |
| Evidence commit | the commit that adds this report and [gpu_portability_review_fixes/](gpu_portability_review_fixes/): records, logs, docs only; it changes no implementation file (§9) |
| Hardware / toolchain | one NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 99 KiB opt-in shared memory, 23.68 GiB usable, 370 W limit), driver 550.67, nvcc 12.1.66, g++ 11.4.0, Compute Sanitizer 2023.1.0, Python 3.11.17, PyTorch 2.7.0+cu128 (CUDA runtime 12.8), Triton 3.3.0, Ubuntu 22.04.5, bash 5.1.16, 6 CPUs, 31 GiB RAM ([environment.txt](gpu_portability_review_fixes/environment.txt)) |
| Build | `production` bundle, target `sm_86` (`device NVIDIA GeForce RTX 3090 (sm_86)`), flags `-O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fPIC -arch=sm_86`, kind `production`, ABI 3, library `e55eb569…`; no `SPARSE_POOL_ARCH` / `SPARSE_POOL_NVCC_EXTRA` / `SPARSE_POOL_PROFILE` set except where a run says so |
| Profile | `auto` → `rtx3090` (`hardware-validated`; matched on name, sm_86, 82 SMs, 99 KiB) ([identity.json](gpu_portability_review_fixes/build/identity.json)) |

**Result: every gate passed on the fix commit.** CPU/shell gate, both full GPU suites, Compute Sanitizer
(16/16 under the repaired runner), deterministic training regression (26/26 runs, bitwise against the
pre-refactor implementation), SASS identity (2,321/2,321 kernels) and the paired performance A/B (no
slowdown above 0.32 % at production scale). Limitations: §10.

## 1. Findings and fixes

| Finding | Before (reproduced on `34a6433`) | Fix | Regression tests |
|---|---|---|---|
| **Documentation absent from a clean checkout** | `docs/` existed only as untracked files in the developer worktree; nothing ignored it (`.gitignore` lists only the build dir and `__pycache__`; there is no packaging config). A clean tree fails `test_dispatch.py::test_documented_profile_template_is_usable_and_unregistered` ([log](gpu_portability_review_fixes/repro/cpu_suite_34a6433_no_gpu.log): 52 failures, this one plus the 51 below) | `docs/` (gpu_portability, adding_a_gpu, evidence, architecture, migration, migration manifest, validation) and the top-level `README.md` are tracked. The documents were checked against the shipped code; corrected: target grammar, override syntax, the decision-matrix recipe (`plan_matrix.py` has no CLI for other boards: the documented Python adapter is used instead), fallbacks and their limits, profile matching. The historical refactor records the validation page cites are committed ([gpu_portability_cleanup/](gpu_portability_cleanup/README.md)); every relative link in every Markdown file resolves | the existing template test; new `test_documented_unknown_device_snippet_runs_as_written` (executes the resource-constrained-device example of `adding_a_gpu.md` §5 and checks its documented output) |
| **Architecture parsing assumed a one-digit major** | `parse_arch("sm_100") = (1, 0)`, `"sm_120" = (1, 20)`, `"sm_121" = (1, 21)`. Explicit overrides `sm_100` / `sm_120` on matching devices were rejected as "cannot run", and a 12.1 device with a toolkit listing `sm_120` got no target; automatic selection worked only when the toolkit listed the native target. `"sm_86\n"` passed the regex (`$` before a newline) ([repro](gpu_portability_review_fixes/repro/repro_arch_before.txt)) | `build.parse_arch`: complete-token `sm_([1-9]\d*)(\d)` (major = all leading digits, minor = last digit); `a` / `f` targets refused by name; `arch_name` for the inverse. `resolve_target` uses integer tuples for overrides and the same-major fallback (plain targets only, never across majors, never PTX), and refuses capability records lacking the kernel family's features. `make_spec` validates the target before it gets a cache identity. Toolkit lists parsed as exact tokens (`capabilities.parse_gpu_list`, used by `toolchain()`) | 59 new tests in `tests/test_build.py` (81 in total) (grammar table sm_80 – sm_121, numeric ordering, 15 malformed tokens, 4 suffixed targets, toolkit-list parsing, the public `toolchain()` query on a fake nvcc listing Blackwell targets, explicit and environment overrides, native and same-major selection, 14 rejection cases incl. cross-major / newer minor / not emitted / missing feature / PTX-only, per-target cache identity, CLI). 23 of them fail on `34a6433` ([list](gpu_portability_review_fixes/repro/arch_tests_on_34a6433.txt)). Resolver tests only: no sm_100 / sm_120 hardware or toolkit |
| **Sanitizer runner could report success after a failed child** | the runner never read job exit statuses, accepted any log containing a zero summary line, let racecheck warnings pass, and deleted all logs on exit. With a fake sanitizer: "0 errors" + exit 42, a failure line + zero summary, another tool's summary, a workload exiting 1 and an incomplete workload all exited **0** ([before / after](gpu_portability_review_fixes/repro/sanitizer_runner_before_after.txt)). Compute Sanitizer 2023.1 itself returns 0 on detected errors unless `--error-exitcode` is set, and passes the workload's exit status through (both verified on this machine) | `run_sanitizer.sh` rewritten: per-job PID → status table (polled with `wait <pid>`, no `set -e`, no pipelines on the status path); `--error-exitcode 97`; per-tool grammar: exactly one completed summary of that tool with all counts 0 (racecheck warnings fail), no other `=========` diagnostic, no traceback, and the workload's new `WORKLOAD COMPLETE <name>` marker; logs truncated before each run and kept (`--out`, `summary.tsv`); `--timeout` (default 4 h) via coreutils `timeout`; INT / TERM / HUP terminate every job's process group; missing sanitizer / workload → exit 2; CLI unchanged apart from new options | `tests/test_sanitizer_runner.py` (24 tests, real runner, fake sanitizer + minimal workloads, ~1 min): all-pass for all four tools, exit 42, sanitizer crash, workload exit 1, traceback, errors with exit 0, errors with the error exit code, racecheck hazards, racecheck warnings, truncated log, failure + zero summary (two tools), two summaries, other tool's summary (two tools), bad start (255), incomplete workload, one failure among 16 jobs (all 16 statuses collected), sanitizer missing / not executable / not on PATH, usage errors, stale successful log, per-run timeout (process group killed), SIGINT / SIGTERM (exit 130 / 143, all 8 child processes gone) |

The strict deterministic-record comparator is unchanged and still strict: `test_det_records.py` (19 passed,
1 skipped on CPU; the skip needs a GPU and runs in the GPU suite) and the end-to-end self-test (§5).

## 2. CPU and shell gate (fresh clone of `fd99c2d`, no GPU visible)

`git clone --no-local` of the branch, `git checkout fd99c2d`, `git status` empty, 428 tracked files, no
untracked files copied in; empty `SPARSE_POOL_CACHE_DIR` / `TRITON_CACHE_DIR`.

| Command | Exit | Result |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES= python -m pytest -q -rfs --tb=short -p no:cacheprovider` (in `sparse_pooling/tests`) | 1 | **174 passed, 2,738 skipped, 51 failed** (75 s; [log](gpu_portability_review_fixes/cpu/cpu_suite_no_gpu.log), [per file](gpu_portability_review_fixes/cpu/cpu_suite_by_file.md)) |
| same on `34a6433` | 1 | 89 passed, 2,738 skipped, 52 failed ([log](gpu_portability_review_fixes/repro/cpu_suite_34a6433_no_gpu.log)) |
| `flake8 --max-line-length 120` on every changed Python file | — | no new warnings (the 6 remaining predate the review) |
| `bash -n` on every `benchmarks/*.sh` | 0 | ok (`shellcheck` is not installed) |

Every CPU-capable test passes: `test_build.py` 81, `test_sanitizer_runner.py` 24, `test_bucket_stats.py` 19,
`test_det_records.py` 19, `test_dispatch.py` 15, `test_isa_requirements.py` 6 (ptxas probes, compile only),
`test_cpu_import.py` 2, and single host tests elsewhere. Skips: tests marked as needing a GPU or the native
kernels ("CUDA GPU required", "native CUDA kernels unavailable"). **The 51 failures are not CPU tests**: 50
input-supergroup generator tests (the generator runs on the GPU) and one train-state fingerprint test create
CUDA tensors without a skip guard; they fail identically on `34a6433` and pass in the GPU suite (§3). This is
a pre-existing gap in the suite's CPU guards, left unchanged (out of scope). Skipped GPU tests are not counted
as CPU validation.

## 3. GPU suites (fresh clone, RTX 3090)

| Command | Exit | Result |
|---|---:|---|
| `python -m pytest -q -rfs --tb=short -p no:cacheprovider` | 0 | **2,931 passed, 32 skipped, 0 failed** (14 min 18 s, cold Triton cache and first builds of the experimental / diagnostics bundles included; [log](gpu_portability_review_fixes/gpu_suites/suite_auto.log)) |
| same with `SPARSE_POOL_PROFILE=generic_cuda` | 0 | **2,913 passed, 50 skipped, 0 failed** (6 min 15 s; [log](gpu_portability_review_fixes/gpu_suites/suite_generic.log)) |

Skips (auto): 25 scatter variants whose shared memory does not fit, 4 `feature_owner` variants limited to
S ≤ 32, `mean_identity` not a constructible pooling type, GELU on the Triton path, and the real two-device
routing test (one GPU). Under `generic_cuda` 18 more tests skip: they assert the RTX 3090 profile's measured
automatic choices; every numerical test runs under both profiles.

## 4. Build, target, profile and cache identity

| Check | Result |
|---|---|
| Cold build of the production bundle into an empty cache (`python -m sparse_pooling.backends.cuda.build --build`, 6 jobs) | 186 s ([log](gpu_portability_review_fixes/build/build_cold.log)) — compile overhead, reported separately from training time |
| Warm load in a new process | 1.21 s ([log](gpu_portability_review_fixes/build/load_warm.log)) |
| Independent cold build of `34a6433`, own cache, 3 bundles | 255 s ([log](gpu_portability_review_fixes/build/build_base.log)) |
| Library identity (sources + headers + flags + nvcc text + host compiler) | production `e55eb569…`, experimental `ec72e621…`, diagnostics `3f1bf046…` — identical for `34a6433` and `fd99c2d` (and `2f55d71`) |
| Selected profile / target on the RTX 3090 | `rtx3090`, `hardware-validated`; forced `generic_cuda`: `numerics-validated, performance-unmeasured`; target `sm_86` |
| Mocked devices (synthetic records, control flow only) | RTX 3080 (sm_86, 68 SMs), unknown sm_89 and sm_120 boards → `generic_cuda`, `capability-eligible, hardware-unvalidated`; forcing `rtx3090` on them → `user-forced-unvalidated` |

## 5. Deterministic training regression

`benchmarks/run_det_regression.sh <py> <2b9a84d checkout> <scratch> <out> all` from the fresh clone: for each
case the pre-refactor implementation (`c8-deterministic-reduction-dw-projection` @ `2b9a84d`, clean) writes the
records of both trajectories in one process and `fd99c2d` compares against them in an independent process
— output, loss, dX, `main_grad`, parameters, optimizer moments / master weights / SR state after every step,
byte for byte with the expected record counts — plus identical resolved plans and state restoration. `G_`
cases: two processes of `fd99c2d` under `generic_cuda`. Model: Mean(ReLU) layer with real R > 1 supergroup
connectivity, tanh head with MSE loss (nonlinear), FusedAdamW, `torch.use_deterministic_algorithms(True)`
(`reduction="auto"` → deterministic). Production cases: H = 100M, M = 8192, R = 4 / 8 / 16 (S = 16 / 32 / 64),
4 steps; medium cases: several dX chunks per supergroup, BF16 / split / FP32 optimizer, C = 64 controls, a
64 MiB workspace cap with multiple token windows. Bitwise here means same device, toolchain, plan and
initial state; it is not a cross-device claim, and the fast (atomic) path is not part of it.

| Case | Kind | det_training case | optimizer | dZ | plan | steps | compared steps | tensors compared | differing | status | same plan | state restored | tested | reference | peak GiB |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|
| G_M64_S16_R4_sr_split | new vs new (generic_cuda) | M64_S16_R4 | sr | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | fd99c2d | 2.11 |
| G_M_S64_R16_det_split | new vs new (generic_cuda) | M_S64_R16 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | fd99c2d | 2.14 |
| M64_S16_R4_det_split | old vs new | M64_S16_R4 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 2.11 |
| M64_S64_R16_det_bf16 | old vs new | M64_S64_R16 | det | bf16 | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 2.10 |
| M_S16_R4_sr_split | old vs new | M_S16_R4 | sr | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 2.14 |
| M_S32_R8_det_split_cap64M | old vs new | M_S32_R8 | det | split | deterministic_workspace_bytes=67108864,cuda_grouped_dx_tokens=64,cuda_grouped_dx_scatter=baseline | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 2.14 |
| M_S64_R16_det_bf16 | old vs new | M_S64_R16 | det | bf16 | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 2.14 |
| M_S64_R16_fp32_split | old vs new | M_S64_R16 | fp32 | split | - | 16 | 32 | 1120 | 0 | pass | True | True | fd99c2d | 2b9a84d | 3.51 |
| P_S16_R4_det_split | old vs new | P_S16_R4 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | fd99c2d | 2b9a84d | 9.81 |
| P_S32_R8_det_split | old vs new | P_S32_R8 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | fd99c2d | 2b9a84d | 9.81 |
| P_S64_R16_det_split | old vs new | P_S64_R16 | det | split | - | 4 | 8 | 264 | 0 | pass | True | True | fd99c2d | 2b9a84d | 10.00 |
| S_S16_R4_det_split | old vs new | S_S16_R4 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 0.30 |
| S_S64_R16_det_split | old vs new | S_S64_R16 | det | split | - | 16 | 32 | 1056 | 0 | pass | True | True | fd99c2d | 2b9a84d | 0.30 |

13 comparisons, 13 pass, 0 differing tensors

**26 of 26 runs passed** (13 comparisons, 0 differing tensors). Raw: [det_regression/](gpu_portability_review_fixes/det_regression/).
The runs were split by two infrastructure events, both logged in [incident_notes.txt](gpu_portability_review_fixes/incident_notes.txt):
systemd-oomd killed the first pipeline (det regression in parallel with the sanitizer) after 20 runs had
passed, during the first production write ([part 1](gpu_portability_review_fixes/det_regression/run_part1_killed_by_oomd.log)); the
three production cases were rerun alone ([part 2](gpu_portability_review_fixes/det_regression/run_part2_production.log)) with a copy of
the runner that skips the passed cases ([tools/run_det_regression_resume.sh](gpu_portability_review_fixes/tools/run_det_regression_resume.sh)).
A first version of that copy leaked `SPARSE_POOL_PROFILE=generic_cuda` from a skipped generic case into the
production cases; the comparator failed `P_S16_R4` on the resolved plan (all 264 tensors were bitwise equal)
and that run was discarded — kept in [incidents/](gpu_portability_review_fixes/incidents/det_resume_profile_env_leak/) as a live example
of the strict plan check.

Harness self-test (`run_det_training.sh <py> <scratch> selftest <out>`, exit 0): the strict pair passes and
every negative control is detected — existing record directory, flipped byte, truncated / extended / missing
record files, structure and count mismatches, incomplete records, changed plan, other seed, fast-mode
characterization ([log](gpu_portability_review_fixes/det_regression/selftest/selftest.log)).

## 6. Compute Sanitizer (repaired runner)

`bash sparse_pooling/benchmarks/run_sanitizer.sh --jobs 3 --out <dir> <py>`: exit **0**, **16/16 PASS**, 78 min
([summary.tsv](gpu_portability_review_fixes/sanitizer/summary.tsv), [runner output](gpu_portability_review_fixes/sanitizer/runner_output.log), per-run logs in
[sanitizer/](gpu_portability_review_fixes/sanitizer/)).

| Workload | memcheck | racecheck | initcheck | synccheck |
|---|---|---|---|---|
| generic (`sanitize_workload.py`) | PASS, 0 errors | PASS, 0 hazards (0 errors, 0 warnings) | PASS, 0 errors | PASS, 0 errors |
| sg | PASS, 0 errors | PASS, 0 hazards (0 errors, 0 warnings) | PASS, 0 errors | PASS, 0 errors |
| dx_scatter | PASS, 0 errors | PASS, 0 hazards (0 errors, 0 warnings) | PASS, 0 errors | PASS, 0 errors |
| det | PASS, 0 errors | PASS, 0 hazards (0 errors, 0 warnings) | PASS, 0 errors | PASS, 0 errors |

Every run exited 0 with `--error-exitcode 97` in force, and each log contains only the banner, one zero summary
of its tool and the workload's completion marker. Policy: racecheck excludes exactly one kernel, the Triton
`bwd_dense_kernel` (`triton_fused.py:441`, a `tl.sum` writing replicated values; ~37k WAW reports identically
at `2b9a84d`, [historical logs](gpu_portability_cleanup/sanitizer/)); memcheck / initcheck / synccheck still cover
it; any other hazard or warning fails. Two earlier attempts in this session (with the det regression in
parallel, and with `--jobs 5` under a 16 GiB soft memory limit) were killed by systemd-oomd before completing;
they wrote no summary, are not counted, and are kept in [incidents/](gpu_portability_review_fixes/incidents/).

## 7. Performance A/B (fix-only baseline)

A = `34a6433` (reviewed snapshot), B = `fd99c2d`, both clean clones, B's harness:
`bash benchmarks/run_impl_ab.sh <py> <A> <B> <out> 4 all` then `summarize_impl_ab.py` — complete training
steps (forward, backward, FusedAdamW) of `bench_supergroups.py`, identical graphs / inputs / initial state,
weights, bias, optimizer moments, step, RNG and SR state restored per repetition (`--reps 2 --iters 3`), a
fresh process per (implementation, round), order alternating A B / B A, GPU otherwise idle (29 °C, 0 %
utilisation before the run; a GNOME remote-desktop process held 333 MiB). Paired statistic: per-round
difference B − A of the median step (and of each phase). Both implementations loaded the same cached library
(`e55eb569…`, built by B; A's independent build has identical SASS, §8), so the A/B isolates host-side
changes. Raw records: [ab/](gpu_portability_review_fixes/ab/), 8-round rerun [ab_h1m_8rounds/](gpu_portability_review_fixes/ab_h1m_8rounds/).

| Config | Variant | rounds | step A ms | step B ms | step paired Δ ms [min, max] | step % | fwd % | dX % | dW % | bwd % | optimizer % | same plan | peak alloc A / B GiB |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| C64_S16_R4 | det | 4 | 1649.2 | 1649.1 | -0.70 [-1.87, -0.06] | -0.04 | -0.17 | +0.07 | -0.06 | -0.02 | +0.01 | yes | 9.50 / 9.50 |
| C64_S16_R4 | fast | 4 | 1694.6 | 1694.5 | +0.04 [-35.78, +0.95] | +0.00 | -0.30 | +0.07 | -0.10 | +0.00 | -0.03 | yes | 8.69 / 8.69 |
| C64_S64_R16 | fast | 4 | 1704.4 | 1705.8 | -0.59 [-2.55, +2.75] | -0.03 | +0.13 | +0.07 | -0.05 | -0.12 | +0.00 | yes | 8.68 / 8.68 |
| C8_S16_R4 | det | 4 | 2437.1 | 2444.9 | +4.54 [-19.09, +16.29] | +0.19 | +0.16 | +0.11 | +0.23 | +0.19 | -0.02 | yes | 9.66 / 9.66 |
| C8_S16_R4 | fast | 4 | 2428.9 | 2438.2 | +7.68 [-2.54, +16.76] | +0.32 | +0.37 | +0.16 | +0.21 | +0.15 | -0.01 | yes | 8.84 / 8.84 |
| C8_S32_R8 | det | 4 | 2685.6 | 2685.5 | -0.03 [-28.16, +2.31] | -0.00 | -0.21 | +0.01 | +0.04 | -0.00 | -0.00 | yes | 9.66 / 9.66 |
| C8_S32_R8 | fast | 4 | 2678.3 | 2677.1 | -0.75 [-25.75, +0.02] | -0.03 | -0.15 | +0.06 | -0.03 | +0.03 | +0.04 | yes | 8.84 / 8.84 |
| C8_S64_R16 | det | 4 | 3064.7 | 3063.1 | -1.60 [-4.66, +1.71] | -0.05 | -0.18 | -0.03 | -0.03 | -0.05 | -0.01 | yes | 9.84 / 9.84 |
| C8_S64_R16 | fast | 4 | 3054.6 | 3057.0 | +1.71 [+0.07, +3.70] | +0.06 | +0.31 | -0.01 | -0.03 | -0.01 | -0.01 | yes | 8.84 / 8.84 |
| C8_S64_R16_bf16 | fast | 4 | 2856.1 | 2857.1 | +0.76 [+0.36, +2.98] | +0.03 | -0.07 | +0.11 | +0.04 | +0.06 | +0.03 | yes | 8.84 / 8.84 |
| H10M_C8_S16_R4_M1000 | det | 2 | 38.2 | 37.6 | -0.61 [-1.06, -0.16] | -1.59 | -1.52 | -1.21 | -0.88 | -1.60 | +0.15 | yes | 1.88 / 1.88 |
| H10M_C8_S16_R4_M1000 | fast | 2 | 36.8 | 36.5 | -0.31 [-0.40, -0.22] | -0.85 | -0.49 | -0.26 | -0.86 | -0.80 | -0.34 | yes | 0.91 / 0.91 |
| H10M_C8_S64_R16 | fast | 2 | 359.6 | 359.4 | -0.25 [-1.07, +0.58] | -0.07 | -0.54 | +0.13 | +0.29 | +0.02 | -0.09 | yes | 1.12 / 1.12 |
| H1M_C8_S64_R16 | fast | 2 | 39.6 | 38.4 | -1.21 [-2.27, -0.15] | -3.06 | -1.26 | -1.70 | -2.20 | -3.09 | -0.17 | yes | 0.34 / 0.34 |
| H1M_C8_S64_R16 (8-round rerun) | fast | 8 | 37.9 | 38.0 | -0.03 [-1.22, +0.20] | -0.07 | -0.27 | -0.08 | -0.10 | -0.09 | -0.11 | yes | 0.34 / 0.34 |
| H1M_C8_S64_R16_M1000 | fast | 2 | 5.2 | 5.1 | -0.03 [-0.04, -0.01] | -0.52 | -1.42 | -0.46 | -0.02 | -0.38 | +0.25 | yes | 0.12 / 0.12 |

Conclusion: **no regression.** Every production-scale row (C = 8 S = 16 / 32 / 64 split in fast and
deterministic mode, C = 8 S = 64 BF16, C = 64 controls) is within −0.05 % … +0.32 % (paired median), no phase
moves by more than 0.4 %, and resolved plans and peak memory are identical in every row. The smaller-scale
points all favour B; `H1M_C8_S64_R16` (−3.1 % in 2 rounds) was rerun with 8 rounds: −0.07 %. Nothing reaches
the 3 % slowdown investigation trigger. The multi-seed protocol of the campaign was not repeated: with
identical SASS and plans, run-to-run variance is the only expected difference.

## 8. SASS identity

`benchmarks/compare_sass.py --old <34a6433 libs> --new <fd99c2d libs>` on two independent cold builds (separate
caches, same toolchain / flags / target): per kernel (mangled name), the `cuobjdump -sass` instruction and
encoding lines with line-info comments removed and whitespace normalised, plus registers / stack / static
shared / local memory. **2,321 kernels on each side, 2,321 identical, 0 resource differences, none only on one
side**; combined digest `9e33bbb2…`, the same digest the refactor's historical comparison recorded for
`2f55d71` ([sass_compare.json](gpu_portability_review_fixes/sass/sass_compare.json)). Expected: the fixes touch no device source, and the
build identities are equal (§4).

## 9. Provenance and reproduction

* Commands, start times, exit codes and durations of every stage: [stage_status.txt](gpu_portability_review_fixes/stage_status.txt);
  the exact driver scripts: [tools/](gpu_portability_review_fixes/tools/) (`gates.sh` first pipeline, `gates2.sh` / `gates4.sh` the
  sequential resumes, probes, the resume copy of the det runner, the 8-round A/B copy of `run_impl_ab.sh`).
  Paths in them point into the session's scratch directory.
* Every det-regression and A/B JSON records its implementation commit and `dirty: false`
  (`fd99c2d` / `34a6433` / `2b9a84d`).
* [MANIFEST.sha256](gpu_portability_review_fixes/MANIFEST.sha256) hashes every file of the package (`sha256sum -c MANIFEST.sha256` in
  that directory).
* The evidence commit changes no implementation file: `git diff --stat fd99c2d <evidence commit> --
  sparse_pooling ':!sparse_pooling/results'` is empty.

## 10. Limitations

* One GPU model was available. **Only the RTX 3090 is hardware-validated.** sm_80 / sm_89 / sm_90 are
  compile-tested (historical) and capability-eligible; sm_100 / sm_103 / sm_120 / sm_121 are covered only by
  CPU resolver tests with synthetic devices and toolkits — no Blackwell toolkit build, no hardware run. The
  generic profile's performance is unmeasured everywhere.
* The two-device routing test skips (one GPU).
* 51 GPU-only tests lack a CPU skip guard (§2), pre-existing.
* `architecture-specific` / family targets (`sm_90a`, `sm_100f`) are refused, not supported; a future kernel
  family that needs them must extend the grammar together with their exact-device compatibility rule.
* The SASS identity and A/B are for this toolchain (nvcc 12.1.66, sm_86). Bitwise determinism is a
  same-device, same-toolchain, same-plan property.
* Historical sanitizer results in [gpu_portability_cleanup/](gpu_portability_cleanup/README.md) came from the
  defective runner and are superseded by §6.
