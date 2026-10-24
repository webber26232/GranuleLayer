#!/usr/bin/env bash
# Random-support study, stage 1 (frozen checkout ../GranuleLayer-rsn-meas @ 2f95faa): one GPU job at a time.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/rsn_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-rsn-meas/sparse_pooling/benchmarks
echo "=== sanitizer $(date)"
bash run_sanitizer.sh --out $GRANULE_RESULTS_ROOT/sanitizer --timeout 3600 $PY rsn kgen kgw
 wps_checkpoints.py --out $GRANULE_RESULTS_ROOT/checkpoints/manifest.json
for st in refresh coverage screen norm_first; do
  echo "=== stage $st $(date)"
  $PY rsn_jobs.py $st --out $st --resume
done
echo "=== stage1 done $(date)"
