#!/usr/bin/env bash
# Interaction-coverage / K16 dX round, stage 2 (frozen checkout ../GranuleLayer-icdx-run): starts only after the
# stage-1 freeze (final_selection.json); the NEW locked tests are opened here for the first time: full-size cohort
# icdx39000 (rows 1-6 dense-masked, rows 7 / 8 compact backend), graph checks, reduced-task endpoints icdx89000 at
# 1000 updates (all eight) and 4000 updates (continued four), launch counts.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/icdx_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-icdx-run/sparse_pooling/benchmarks
until grep -q "=== stage1 done" $R/stage1_driver.out; do
  grep -q "selection .* failed" $R/stage1_driver.out && { echo "stage 1 selection failed: not starting"; exit 1; }
  systemctl --user is-active -q icdx-stage1 || { grep -q "=== stage1 done" $R/stage1_driver.out || { echo "stage 1 ended without freeze"; exit 1; }; }
  sleep 60
done
[ -f $R/final_selection.json ] || { echo "no final_selection.json"; exit 1; }
git -C .. rev-parse HEAD > $R/stage2_commit.txt; git -C .. status --short >> $R/stage2_commit.txt
run() { echo "=== $1 -> $2 $(date)"; $PY icdx_jobs.py $1 --out $2 ${3:+--selection $R/$3} --resume; }
run final final final_selection.json
run graph_checks graph_checks final_selection.json
run interaction_eval interaction_eval interaction_states_selection.json
run interaction_cont interaction_cont continuation_selection.json
$PY - <<PYEOF
import json
s = json.load(open("$R/final_selection.json"))
v, m = s["backend"]["chosen"], s["mixed"]
specs = ["Cover-TopK/16:indexed:legacy", "Balanced-TopK/16#0:indexed:legacy", "BP-row#0:indexed:legacy",
         f"{m}:grouped:legacy", "Mix-TopK#0:grouped:legacy", "U16@0.25:wide:legacy", "dense:dense:legacy"]
if v != "legacy":
    specs += [f"Cover-TopK/16:indexed:{v}", f"{m}:grouped:{v}"]
json.dump(dict(jobs=[dict(kind="script", tag="launch_counts_g0", script="icdx_launch_count.py",
                          args=["--specs", ",".join(specs), "--graph-seed", "0"])]), open("$R/launch_selection.json", "w"))
PYEOF
run scripts launch launch_selection.json
echo "=== stage2 done $(date)"
