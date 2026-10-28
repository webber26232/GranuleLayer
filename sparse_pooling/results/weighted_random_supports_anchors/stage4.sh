#!/usr/bin/env bash
# K8 / weighted / anchor study, stage 5: locked final test + interaction screen (frozen ../GranuleLayer-kwa-meas2)
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas2/sparse_pooling/benchmarks
until grep -q "=== stage3 done" $R/stage3_driver.out; do sleep 30; done
echo "=== select final_methods $(date)"
$PY kwa_select.py final_methods > $R/final_select.out 2>&1 || { echo "selection failed"; exit 1; }
$PY $R/make_interaction_sel.py $R
echo "=== final $(date)"; $PY kwa_jobs.py final --out final --selection $R/final_selection.json --resume
echo "=== interaction $(date)"; $PY kwa_jobs.py interaction --out interaction --selection $R/interaction_selection.json --resume
echo "=== stage4 done $(date)"
