#!/usr/bin/env bash
# Random-support study, stage 2 (frozen checkout ../GranuleLayer-rsn-meas2 @ e904796): one GPU job at a time.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/rsn_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-rsn-meas2/sparse_pooling/benchmarks
while pgrep -f "bash stage1.sh" >/dev/null; do sleep 30; done
R=$GRANULE_RESULTS_ROOT
mv $R/refresh $R/refresh_oom_attempt_2f95faa 2>/dev/null
$PY rsn_jobs.py refresh --out refresh --resume
$PY rsn_jobs.py confirm --out confirm --selection $R/confirm_selection.json --resume
$PY rsn_jobs.py coverage_sel --out coverage_confirm --selection $R/coverage_confirm_selection.json --resume
$PY rsn_jobs.py k8 --out k8 --resume
$PY rsn_jobs.py norm --out norm --selection $R/norm_selection.json --resume
$PY rsn_jobs.py interaction --out interaction --selection $R/interaction_selection.json --resume
echo "=== stage2 done $(date)"
