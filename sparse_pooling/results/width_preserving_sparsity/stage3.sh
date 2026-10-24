#!/usr/bin/env bash
# Width-preserving study, stage 3 (frozen checkout ../GranuleLayer-wps-meas3): interaction capacity variant
# (16 terms per record; the dense control first, as a learnability gate) and the fine-tune run-to-run noise.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/wps_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-wps-meas3/sparse_pooling/benchmarks
echo "=== interaction_capacity $(date)"; $PY wps_jobs.py interaction_capacity --out interaction_capacity --resume
echo "=== finetune_repeat $(date)";      $PY wps_jobs.py finetune_repeat --out finetune_repeat --resume
echo "=== stage3 done $(date)"
