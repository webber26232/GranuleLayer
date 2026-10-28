#!/usr/bin/env bash
# Stage 5: graph / layout confirmations that failed in stage 3 on a selection-key mismatch ('rule' vs 'rule_spec');
# rerun from derived selection files (rule <- rule_spec), frozen ../GranuleLayer-kwa-meas2
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/kwa_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-kwa-meas2/sparse_pooling/benchmarks
until grep -q "=== stage4 done" $R/stage4_driver.out; do sleep 30; done
$PY - <<'PY'
import json
R = "/home/evga/Projects/pamachine/kwa_runs"
j = json.load(open(f"{R}/m25_final_selection.json"))
json.dump(dict(j, rule=j["rule_spec"], derived_from="m25_final_selection.json (rule <- rule_spec)"),
          open(f"{R}/m25_confirm_selection.json", "w"), indent=1)
PY
for st in w8_confirm m25_confirm; do
  echo "=== $st $(date)"
  mv $R/$st $R/${st}_failed_selection_key 2>/dev/null
  $PY kwa_jobs.py $st --out $st --selection $R/${st}_selection.json --resume
done
echo "=== stage5 done $(date)"
