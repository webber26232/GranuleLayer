#!/usr/bin/env bash
# Random-support study, stage 3 (frozen checkout ../GranuleLayer-rsn-meas3 @ 52820b7)
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/rsn_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-rsn-meas3/sparse_pooling/benchmarks
$PY rsn_jobs.py cap16 --out cap16 --selection $GRANULE_RESULTS_ROOT/cap16_selection.json --resume
echo "=== stage3 done $(date)"
