#!/usr/bin/env bash
# TopK16 / Cover-TopK round, stage 4 (frozen checkout ../GranuleLayer-tkc-final2 = final code): Compute Sanitizer on
# the tkc / rsn / kgen workloads after the rsn-workload fix, and the full suite on that code
set -u
R=/home/evga/Projects/pamachine/tkc_runs
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-tkc-final2/sparse_pooling/benchmarks
git -C .. rev-parse HEAD > $R/stage4_commit.txt; git -C .. status --short >> $R/stage4_commit.txt
echo "=== sanitizer $(date)"; env --default-signal=INT bash run_sanitizer.sh --out $R/sanitizer $PY tkc rsn kgen > $R/sanitizer_driver.out 2>&1; echo "sanitizer exit $?"; tail -1 $R/sanitizer_driver.out
cd ..
echo "=== suite $(date)"; env --default-signal=INT $PY -m pytest -q tests > $R/full_suite.txt 2>&1; echo "suite exit $?"; tail -n 2 $R/full_suite.txt
echo "=== stage4 done $(date)"
