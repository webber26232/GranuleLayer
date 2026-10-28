#!/usr/bin/env bash
# K8 / weighted / anchor study, stage 2b-3 (frozen checkout ../GranuleLayer-kwa-meas1 @ a0a7b4b)
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas1/sparse_pooling/benchmarks
until grep -q "=== stage1 done" $GRANULE_RESULTS_ROOT/stage1_driver.out; do sleep 30; done
echo "=== select k8_graph $(date)"
SEL=/home/evga/Projects/pamachine/GranuleLayer-kwa-tools/sparse_pooling/benchmarks; git -C $SEL rev-parse HEAD > $GRANULE_RESULTS_ROOT/selection_tools_commit.txt
$PY $SEL/kwa_select.py k8_graph > $GRANULE_RESULTS_ROOT/k8_graph_select.out 2>&1 || { echo "selection failed"; exit 1; }
$PY $SEL/kwa_select.py parent_treatments > $GRANULE_RESULTS_ROOT/parent_treatments.out 2>&1
for st in w8_screen; do echo "=== $st $(date)"; $PY kwa_jobs.py $st --out $st --resume; done
echo "=== k8_graph $(date)"
$PY kwa_jobs.py k8_graph --out k8_graph --selection $GRANULE_RESULTS_ROOT/k8_graph_selection.json --resume
echo "=== stage2 done $(date)"
