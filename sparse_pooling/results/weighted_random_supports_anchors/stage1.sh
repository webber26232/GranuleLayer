#!/usr/bin/env bash
# K8 / weighted / anchor study, stages 1-2 (frozen checkout ../GranuleLayer-kwa-meas1 @ a0a7b4b)
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas1/sparse_pooling/benchmarks
git -C .. rev-parse HEAD > $GRANULE_RESULTS_ROOT/stage1_commit.txt
git -C .. status --short >> $GRANULE_RESULTS_ROOT/stage1_commit.txt
for st in errata_cov qcheck k8_cov k8_timing k8_q32 k8_q256; do
  echo "=== $st $(date)"
  $PY kwa_jobs.py $st --out $st --resume
done
echo "=== stage1 done $(date)"
