#!/usr/bin/env bash
# Interaction-coverage / K16 dX round, stage 3 (frozen checkout ../GranuleLayer-icdx-run): Compute Sanitizer over the
# new dX paths / K32 grouped execution and the related existing workloads, then the complete required suite.
set -u
R=/home/evga/Projects/pamachine/icdx_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-icdx-run/sparse_pooling/benchmarks
until grep -q "=== stage2 done" $R/stage2_driver.out 2>/dev/null; do
  systemctl --user is-active -q icdx-stage2 || { grep -q "=== stage2 done" $R/stage2_driver.out || { echo "stage 2 ended without completion"; exit 1; }; }
  sleep 60
done
git -C .. rev-parse HEAD > $R/stage3_commit.txt; git -C .. status --short >> $R/stage3_commit.txt
echo "=== sanitizer $(date)"; bash run_sanitizer.sh --jobs 4 --out $R/sanitizer $PY icdx kgen tkc kwa rsn > $R/sanitizer_driver.out 2>&1; echo "sanitizer exit $?"
cd ..
echo "=== suite $(date)"; env --default-signal=INT $PY -m pytest -q tests > $R/full_suite.txt 2>&1; echo "suite exit $?"; tail -n 3 $R/full_suite.txt
echo "=== stage3 done $(date)"
