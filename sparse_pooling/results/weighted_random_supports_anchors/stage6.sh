#!/usr/bin/env bash
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas2/sparse_pooling/benchmarks
until grep -q "=== stage5 done" $R/stage5_driver.out; do sleep 30; done
echo "=== interaction_extra $(date)"; $PY kwa_jobs.py interaction --out interaction_extra --selection $R/interaction_extra_selection.json --resume
echo "=== stage6 done $(date)"
