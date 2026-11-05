#!/usr/bin/env bash
# Interaction-coverage / K16 dX round, stage 1 (frozen checkout ../GranuleLayer-icdx-run): preregistration, dX
# correctness gate, construction, profiling, dX screen, coverage, development, reduced task (no test), ANOVA bound,
# backend confirmation, selections and the freeze.  GPU work strictly serialized; timing sessions run with nothing
# else on the GPU.  Stops (exit 1) when a selection cannot be made.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/icdx_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-icdx-run/sparse_pooling/benchmarks
git -C .. rev-parse HEAD > $R/stage1_commit.txt; git -C .. status --short >> $R/stage1_commit.txt
run() { echo "=== $1 -> $2 $(date)"; $PY icdx_jobs.py $1 --out $2 ${3:+--selection $R/$3} --resume; }
sel() { echo "=== select $1 $(date)"; $PY icdx_select.py $1 > $R/$1_select.out 2>&1 || { echo "selection $1 failed"; cat $R/$1_select.out; exit 1; }; }
sel rules
echo "=== dX correctness gate $(date)"
(cd .. && $PY -m pytest -q tests/test_kgen_dx_variants.py) > $R/dx_correctness.txt 2>&1; echo "dx gate exit $?: $(tail -1 $R/dx_correctness.txt)"
run construct construct
run profile profile
run dx_screen dx_screen
run coverage coverage
run dev32 dev32
sel continuity
run dev256 dev256
if [ "$($PY -c "import json; print(len(json.load(open('$R/continuity_selection.json'))['rerun']))")" != "0" ]; then
  run dev256_rerun dev256_rerun continuity_selection.json
fi
sel treatments
sel dx_screen
run interaction interaction
sel interaction_states
run anova anova
run confirm confirm confirm_selection.json
sel backend
run guard guard backend_selection.json
run replicates replicates backend_selection.json
sel final
echo "=== stage1 done $(date)"
