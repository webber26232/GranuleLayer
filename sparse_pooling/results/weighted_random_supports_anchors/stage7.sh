#!/usr/bin/env bash
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas3/sparse_pooling/benchmarks
git -C .. rev-parse HEAD > $R/stage7_commit.txt; git -C .. status --short >> $R/stage7_commit.txt
echo "=== interaction_cont $(date)"; $PY kwa_jobs.py interaction --out interaction_cont --selection $R/interaction_cont_selection.json --resume
echo "=== sanitizer $(date)"; bash run_sanitizer.sh --out $R/sanitizer $PY kwa rsn kgen > $R/sanitizer_driver.out 2>&1; echo "sanitizer exit $?"
echo "=== suite $(date)"; cd ..; $PY -m pytest -q tests > $R/full_suite.txt 2>&1; echo "suite exit $?"; tail -n 3 $R/full_suite.txt
echo "=== stage7 done $(date)"
