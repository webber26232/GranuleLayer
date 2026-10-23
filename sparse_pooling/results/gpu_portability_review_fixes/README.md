# Raw records: GPU-portability review fixes

Fresh evidence for [../GPU_PORTABILITY_REVIEW_FIXES.md](../GPU_PORTABILITY_REVIEW_FIXES.md), produced from a fresh
clone of `fd99c2d` (baseline clone `34a6433`, det reference `2b9a84d`) on one RTX 3090. Historical refactor records
are separate: [../gpu_portability_cleanup/](../gpu_portability_cleanup/README.md).

| Path | Contents |
|---|---|
| `environment.txt` | hardware / driver / toolchain / library versions (GPU UUID redacted) |
| `stage_status.txt`, `incident_notes.txt` | every stage's command, start, exit code, duration; the oomd kills and the discarded runs |
| `repro/` | before-fix reproductions on `34a6433`: architecture parsing, the new build tests run against it, the sanitizer runner under injected faults, the CPU suite of a clean `34a6433` tree |
| `cpu/` | CPU-only suite of the fresh clone (log, per-file counts, JUnit XML) |
| `gpu_suites/` | full suite, auto and `generic_cuda` profiles (logs, JUnit XML) |
| `build/` | cold / warm build logs, the baseline's independent build, `identity.json` (profile, target, build identity, mocked selections) |
| `sass/` | `compare_sass.py` result: 34a6433 vs fd99c2d, all bundles |
| `ab/`, `ab_h1m_8rounds/` | performance A/B raw JSON + logs per (config, round, implementation), summaries |
| `det_regression/` | `run_det_regression.sh all` (JSON + log per side, runner logs), `selftest/` |
| `sanitizer/` | repaired runner: 16 per-run logs, `summary.tsv`, runner output |
| `incidents/` | runs that did not count: two sanitizer runs killed by systemd-oomd, the det resume with the leaked profile variable |
| `tools/` | the exact driver scripts and probes used |
| `MANIFEST.sha256` | SHA-256 of every other file here (`sha256sum -c MANIFEST.sha256`) |
