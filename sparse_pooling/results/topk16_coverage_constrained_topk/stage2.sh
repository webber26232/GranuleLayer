#!/usr/bin/env bash
# TopK16 / Cover-TopK round, stage 2 (frozen checkout ../GranuleLayer-tkc-run): quiet construction rerun, launch
# counts, coverage errata reruns, the NEW locked final cohort (tkc29000), graph checks, reduced interaction diagnostic
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/tkc_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-tkc-run/sparse_pooling/benchmarks
until grep -q "=== stage1 done" $R/stage1_driver.out; do
  grep -q "selection .* failed" $R/stage1_driver.out && { echo "stage 1 selection failed: not starting"; exit 1; }
  sleep 60
done
git -C .. rev-parse HEAD > $R/stage2_commit.txt; git -C .. status --short >> $R/stage2_commit.txt
run() { echo "=== $1 -> $2 $(date)"; $PY tkc_jobs.py $1 --out $2 ${3:+--selection $R/$3} --resume; }
sel() { echo "=== select $1 $(date)"; $PY tkc_select.py $1 > $R/$1_select.out 2>&1 || { echo "selection $1 failed"; cat $R/$1_select.out; exit 1; }; }
run scripts construct_quiet construct_quiet_selection.json
run scripts launch launch_selection.json
run scripts errata_cov errata_cov_selection.json
run final final final_selection.json
run graph_checks graph_checks graph_selection.json
run interaction interaction
sel interaction_states
run interaction_cont interaction_cont interaction_cont_selection.json
echo "=== stage2 done $(date)"
