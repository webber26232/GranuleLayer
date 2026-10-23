#!/bin/bash
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/k64l1_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-k64l1-meas/sparse_pooling/benchmarks
$PY k64_campaign.py confirm --out confirm --selection $GRANULE_RESULTS_ROOT/confirm_selection.json
$PY k64_jobs.py plan --plan-file $GRANULE_RESULTS_ROOT/confirm_sparse_plan.json --out confirm_sparse --ckpt-dir /home/evga/Projects/pamachine/k64l1_ckpt
echo STAGE3 DONE
