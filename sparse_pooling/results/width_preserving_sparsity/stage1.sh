#!/usr/bin/env bash
# Width-preserving study, stage 1 (frozen checkout ../GranuleLayer-wps-meas): one GPU job at a time.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/wps_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-wps-meas/sparse_pooling/benchmarks
$PY wps_checkpoints.py --out $GRANULE_RESULTS_ROOT/checkpoints/manifest.json
for st in screen c2048 ablation reject inference coverage finetune coverage_ft interaction regen; do
  echo "=== stage $st $(date)"
  $PY wps_jobs.py $st --out $st --resume
done
echo "=== stage1 done $(date)"
