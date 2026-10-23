# Raw records: validation of the GPU-portability refactor (historical)

These records were produced **before** the review of `gpu-portability-cleanup` @ `34a6433`, by the
sessions that wrote the refactor, and are committed here unchanged (only copied and, for two large
logs, gzip-compressed). They are the evidence behind
[docs/validation/gpu_portability_cleanup.md](../../../docs/validation/gpu_portability_cleanup.md).
Fresh evidence for the review fixes is in [../gpu_portability_review_fixes/](../gpu_portability_review_fixes/)
(see [../GPU_PORTABILITY_REVIEW_FIXES.md](../GPU_PORTABILITY_REVIEW_FIXES.md)); nothing here is a
substitute for it.

Hardware / software of every record: one NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 23.68 GiB usable),
driver 550.67, nvcc 12.1.66, PyTorch 2.7.0+cu128 (CUDA runtime 12.8), Triton 3.3.0. Baseline (A,
"reference", "old"): `c8-deterministic-reduction-dw-projection` @ `2b9a84d` (clean checkout).

| Path | What | Source revision(s) | Command |
|---|---|---|---|
| `ab/` | performance A/B: one JSON + log per (config, round, implementation); `ab_summary.{json,md}` from `summarize_impl_ab.py`; `ab_run.log` the runner output | A `2b9a84d`, B `2f55d71` (both clean, recorded in every JSON) | `benchmarks/run_impl_ab.sh <py> <A> <B> <dir> 4 all` |
| `ab_h1m_8rounds/` | 8-round rerun of `H1M_C8_S64_R16` (16 JSONs + `summary.{json,md}`). The runner's own log of this rerun was not kept | A `2b9a84d`, B `2f55d71` | `run_impl_ab.sh` with that one config and 8 rounds |
| `sass/sass_all_bundles.json`, `sass/sass_production.json` | kernel-by-kernel SASS / resource comparison (`benchmarks/compare_sass.py`): the baseline's single library against the production / experimental / diagnostics bundles | old: the `2b9a84d` library; new: the cached `production` / `experimental` / `diagnostics` libraries `e55eb569…` / `ec72e621…` / `3f1bf046…`, which are the build identities of `2f55d71` and `34a6433` (identical inputs) | `compare_sass.py --old <lib> --new <libs> --out ...` |
| `sass/compile_only_sm80_sm89_sm90.log` | production bundle built for sm_80 / sm_89 / sm_90 (compile only, kernel counts via cuobjdump) | `2f55d71` | `python -m sparse_pooling.backends.cuda.build --target sm_XY --build` |
| `sass/build_sm86.log` | cold build of the sm_86 production bundle | `2f55d71` | same, default target |
| `suites/suite_auto.log`, `suites/suite_generic_cuda.log` | full test suite, auto profile and `SPARSE_POOL_PROFILE=generic_cuda` | `c75c402` | `python -m pytest -q -rfs --tb=short` in `sparse_pooling/tests` |
| `det_regression/` | `benchmarks/run_det_regression.sh ... all`: one JSON + log per side; `run_part1.log` + `run_part2_resumed.log` (the first run was killed by systemd-oomd after 10 passing runs and resumed from the next case); `selftest.log` = `run_det_training.sh selftest` | new `34a6433` (`c75c402` for the first case; the two differ only in `run_sanitizer.sh`); the JSONs record `dirty: true` because of uncommitted documentation files; reference `2b9a84d` | see the runner header |
| `sanitizer/sequential_runner_c75c402.txt` | output of the previous `run_sanitizer.sh` (sequential) | `c75c402` | `run_sanitizer.sh <py>` |
| `sanitizer/parallel_runner_34a6433.txt` | output of `run_sanitizer.sh --jobs 5` as committed at `34a6433` (racecheck excludes the Triton `bwd_dense_kernel`) | `34a6433` | `run_sanitizer.sh --jobs 5 <py>` |
| `sanitizer/racecheck_generic_*.log*` | racecheck of `sanitize_workload.py` at the baseline and at the refactor (all hazards), and at the refactor with `bwd_dense_kernel` excluded | `2b9a84d`, `c75c402` | `compute-sanitizer --tool racecheck --racecheck-report all --print-limit 200` (excluded: `--kernel-regex-exclude kns=bwd_dense_kernel --print-limit 20`) |
| `gates_status.txt` | stage log of the gate pipeline that produced `suites/`, `det_regression/` and `sanitizer/` | as listed in it | — |

Known limitations of these records:

* The runner used for `sanitizer/*runner*.txt` is the one the review found defective: it did not check
  the sanitizer's exit status, deleted its per-run logs, and its printed excerpts are all that remain.
  The summaries in them are genuine sanitizer output, but these runs are **not** accepted as evidence
  under the repaired runner's rules; the fresh runs in `../gpu_portability_review_fixes/` are.
* The det-regression records (`*_ref_write` / `*_new_compare`) were deleted after each comparison by
  the runner; the JSONs keep the per-step comparison counts and verdicts.
