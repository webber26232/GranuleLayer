#!/usr/bin/env bash
# Cross-session check (after stage 3): the PREVIOUS round's frozen checkout (../GranuleLayer-tkc-run @ fcf6925) and
# this round's frozen checkout time the same dense / Cover-TopK/16 legacy jobs in one session, alternating, to
# separate a code change from an environment / session change in absolute milliseconds.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/icdx_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
until grep -q "=== stage3 done" $R/stage3_driver.out 2>/dev/null; do
  systemctl --user is-active -q icdx-stage3 || { grep -q "=== stage3 done" $R/stage3_driver.out || { echo "stage 3 ended without completion"; exit 1; }; }
  sleep 60
done
nvidia-smi > $R/cross_session_nvidia_smi.txt
echo '{"sessions": [{"pairs": [["Cover-TopK/16", "indexed"]], "graph_seed": 0, "rounds": [2, 2, 2]}]}' > $R/cross_session_sel.json
for rep in 1 2; do
  echo "=== previous checkout rep $rep $(date)"
  (cd /home/evga/Projects/pamachine/GranuleLayer-tkc-run/sparse_pooling/benchmarks && git -C .. rev-parse HEAD && \
   $PY tkc_jobs.py timing --out cross_session_prev$rep --selection $R/cross_session_sel.json --resume)
  echo "=== current checkout rep $rep $(date)"
  (cd /home/evga/Projects/pamachine/GranuleLayer-icdx-run/sparse_pooling/benchmarks && git -C .. rev-parse HEAD && $PY - <<PYEOF
import icdx_jobs as J, icdx_cases as IC
J.run_plan(dict(stage="cross_session", jobs=J.session([("Cover-TopK/16", "indexed", "legacy")], 0, IC.SCREEN_ROUNDS, 7)),
           "cross_session_cur$rep", 3600, True)
PYEOF
)
done
echo "=== stage4 done $(date)"
