#!/usr/bin/env bash
# TopK16 / Cover-TopK round, stage 1 (frozen checkout ../GranuleLayer-tkc-run; the construct jobs ran at ../GranuleLayer-tkc-meas1 = 637b97c before a restart): construction, performance screen /
# confirmation, coverage, development learning; then the mechanical selections of the preregistered rules
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/tkc_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-tkc-run/sparse_pooling/benchmarks
git -C .. rev-parse HEAD > $R/stage1_commit.txt; git -C .. status --short >> $R/stage1_commit.txt
run() { echo "=== $1 $(date)"; $PY tkc_jobs.py $1 --out $1 ${2:+--selection $R/$2} --resume; }
sel() { echo "=== select $1 $(date)"; $PY tkc_select.py $1 > $R/$1_select.out 2>&1 || { echo "selection $1 failed"; cat $R/$1_select.out; exit 1; }; }
run construct
# the CPU-load signal stress tests must not overlap any timing session
until grep -q "MATRIX DONE" /tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/59d0cc1a-a1b0-4b68-9369-4a46240cb8ce/scratchpad/sig/matrix.out; do sleep 30; done
echo "=== stress matrix finished $(date)"
run perf_screen
run perf_confirm
run coverage
run dev32
run dev256
sel continuity
sel treatments
sel final_methods
sel graph_methods
echo "=== stage1 done $(date)"
