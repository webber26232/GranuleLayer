#!/usr/bin/env bash
# TopK16 / Cover-TopK round, stage 3 (frozen checkout ../GranuleLayer-tkc-final = final code): launch counts v2,
# Compute Sanitizer, runner regressions under several launch conditions, full suite
set -u
export GRANULE_RESULTS_ROOT=/home/evga/Projects/pamachine/tkc_runs
R=$GRANULE_RESULTS_ROOT
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
cd /home/evga/Projects/pamachine/GranuleLayer-tkc-final/sparse_pooling/benchmarks
until grep -qE "=== stage2 done" $R/stage2_driver.out; do sleep 60; done
git -C .. rev-parse HEAD > $R/stage3_commit.txt; git -C .. status --short >> $R/stage3_commit.txt
echo "=== launch2 $(date)"; $PY tkc_jobs.py scripts --out launch2 --selection $R/launch2_selection.json --resume
echo "=== sanitizer $(date)"; bash run_sanitizer.sh --out $R/sanitizer $PY tkc rsn kgen > $R/sanitizer_driver.out 2>&1; echo "sanitizer exit $?"
cd ..
echo "=== runner tests $(date)"
# this driver is started with '&' (SIGINT ignored by inheritance); env --default-signal=INT restores SIG_DFL
grep SigIgn /proc/$$/status | sed 's/^/driver /'
for i in 1 2 3; do env --default-signal=INT $PY -m pytest -q tests/test_sanitizer_runner.py > $R/runner_dfl_$i.txt 2>&1; echo "SIGINT default $i exit $?: $(tail -1 $R/runner_dfl_$i.txt)"; done
for i in 1 2 3; do $PY -m pytest -q tests/test_sanitizer_runner.py > $R/runner_ign_$i.txt 2>&1; echo "SIGINT ignored by inheritance $i exit $?: $(tail -1 $R/runner_ign_$i.txt)"; done
hogs=(); for c in $(seq 1 $(nproc)); do $PY -c "while True: pass" & hogs+=($!); done
for i in 1 2; do env --default-signal=INT $PY -m pytest -q tests/test_sanitizer_runner.py > $R/runner_load_$i.txt 2>&1; echo "under full CPU load $i exit $?: $(tail -1 $R/runner_load_$i.txt)"; done
kill "${hogs[@]}"; wait "${hogs[@]}" 2>/dev/null
echo "=== suite $(date)"; env --default-signal=INT $PY -m pytest -q tests > $R/full_suite.txt 2>&1; echo "suite exit $?"; tail -n 3 $R/full_suite.txt
echo "=== stage3 done $(date)"
