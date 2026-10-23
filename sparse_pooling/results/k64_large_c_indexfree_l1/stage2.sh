#!/bin/bash
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/k64l1_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
CK=/home/evga/Projects/pamachine/k64l1_ckpt
cd /home/evga/Projects/pamachine/GranuleLayer-k64l1-meas/sparse_pooling/benchmarks
$PY k64_jobs.py anchors --out sparse_anchors --ckpt-dir $CK --densities 0.25,0.1,0.05
$PY k64_jobs.py refresh --out refresh --ckpt-dir $CK --C 256
mkdir -p $GRANULE_RESULTS_ROOT/semi24 && $PY k64_semi24.py --out $GRANULE_RESULTS_ROOT/semi24/panel.json > $GRANULE_RESULTS_ROOT/semi24/panel.log 2>&1; echo "semi24 rc=$?"
$PY k64_campaign.py coverage --out coverage
$PY k64_jobs.py maskcov --out maskcov --ckpt-dir $CK
echo STAGE2 DONE
