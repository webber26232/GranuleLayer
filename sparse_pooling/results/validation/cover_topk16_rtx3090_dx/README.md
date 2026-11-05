# Cover-TopK/16 integration: compact acceptance records

Report: [docs/validation/cover_topk16_rtx3090_dx.md](../../../../docs/validation/cover_topk16_rtx3090_dx.md).

All records were produced by `sparse_pooling/benchmarks/run_cover_topk16_validation.sh` from a clean frozen checkout of
`ad3a15c8baca4746d481db42ceaf8ca89f9a979b` (`source_head.txt`). `stages.tsv` lists each stage, its exit code and its
seconds.

| file | content |
|---|---|
| `stages.tsv`, `source_head.txt`, `environment.txt` | stage outcomes, source head, software and GPU |
| `timing/summary.json`, `timing/plan.json`, `timing/status.json`, `timing/job*.json` | same-session timing: summary, randomized order, per-job exit, all 24 samples per job, clocks / power / temperature per round, peaks, executed variant, fingerprints |
| `identity_g{0,1,2}.json` | full-size structure, coverage, identities, memory, round trip |
| `train_smoke.json` | 32-update legacy / auto trajectories, seeds 0 / 1 / 2 |
| `optimizer_equivalence.json` | FusedAdamW vs the research optimizer, 8 configurations |
| `research_contrasts.json` | research contrasts recomputed with the corrected cohort validation |
| `compat_smoke.json` | fan-in-8 paths against main |
| `sanitizer/` | runner summary and one log per workload and tool (24 runs) |
| `logs/` | fresh-cache focused tests, complete suite, timing, training and sanitizer stage logs |

Edits after the run, all to these copies:
- Machine paths were replaced by placeholders: `<OUT>`, `<RUNS>`, `<CHECKOUT>`, `<BASELINE>`, `<RESEARCH_CHECKOUT>`,
  `<CKPT_DIR>`, `<PY>`, `<SITE>`. No measured value was changed.
- `nvcc_version_recorded_after_run.txt` and `compute_sanitizer_version_recorded_after_run.txt` were added after the
  run, on the same machine. `environment.txt` captured only nvcc's banner.

Not kept here: packed states, checkpoints, the CUDA cache, compiled libraries and the raw research campaign.
