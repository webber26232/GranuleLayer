#!/usr/bin/env bash
# Width-preserving study, stage 2 (frozen checkout ../GranuleLayer-wps-meas2): confirmation over graph seeds 0/1/2,
# winner checks at C64 / C2048, the dense C256/S128/R256 reference over seeds 0/1/2, refresh costs.
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/wps_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
R=$GRANULE_RESULTS_ROOT
cd /home/evga/Projects/pamachine/GranuleLayer-wps-meas2/sparse_pooling/benchmarks
echo "=== confirm $(date)";      $PY wps_jobs.py confirm --out confirm --selection $R/confirm_selection.json --resume
echo "=== winners $(date)";      $PY wps_jobs.py winners --out winners --selection $R/winners_selection.json --resume
echo "=== confirm_s128 $(date)"; $PY k64_campaign.py confirm --out confirm_s128 --selection $R/confirm_s128_selection.json --resume
echo "=== refresh_cost $(date)"; $PY wps_jobs.py refresh_cost --out refresh_cost --resume
if [ -f $R/refresh_selection.json ]; then
  echo "=== refresh $(date)";    $PY wps_jobs.py refresh --out refresh --selection $R/refresh_selection.json --resume
fi
echo "=== stage2 done $(date)"
