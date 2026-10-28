#!/usr/bin/env bash
# K8 / weighted / anchor study, stages 3-4 (frozen checkout ../GranuleLayer-kwa-meas2 @ 3269277); selections by
# kwa_select.py (predeclared rules, same checkout)
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas2/sparse_pooling/benchmarks
until grep -q "=== stage2 done" $R/stage2_driver.out; do sleep 30; done
git -C .. rev-parse HEAD > $R/stage3_commit.txt; git -C .. status --short >> $R/stage3_commit.txt
sel() { echo "=== select $1 $(date)"; $PY kwa_select.py $1 > $R/$1_select.out 2>&1 || { echo "selection $1 failed"; cat $R/$1_select.out; exit 1; }; }
run() { echo "=== $1 $(date)"; $PY kwa_jobs.py $1 --out $1 ${2:+--selection $R/$2} --resume; }
sel w8_advance
run w8_dev w8_advance_selection.json
sel w8_final
run m25_screen w8_final_selection.json
run w8_confirm w8_final_selection.json
sel m25_advance
run m25_dev m25_advance_selection.json
sel m25_final
run m25_confirm m25_final_selection.json
echo "=== stage3 done $(date)"
