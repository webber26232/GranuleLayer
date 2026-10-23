#!/usr/bin/env bash
# GPU gates for the review fixes, from the fresh clone (src = fd99c2d) and the baseline clone
# (base = 34a6433).  Every stage appends "<stage> rc=<exit> <seconds>s" and its command to $O/status.
set -uo pipefail
F=/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh
O=$F/out
PY=/home/evga/.conda/envs/env_isaaclab/bin/python
SRC=$F/src
BASE=$F/base
DETREF=/home/evga/Projects/pamachine/GranuleLayer-det   # 2b9a84d, clean
export SPARSE_POOL_CACHE_DIR=$F/cache TRITON_CACHE_DIR=$F/triton
unset SPARSE_POOL_PROFILE SPARSE_POOL_ARCH SPARSE_POOL_NVCC_EXTRA
mkdir -p $O
stage() {  # name, command...
  local name=$1; shift
  echo "$name START $(date -Is) :: $*" >> $O/status
  local t0=$(date +%s)
  "$@"
  local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $O/status
  return $rc
}
cd $SRC
echo "src $(git -C $SRC rev-parse HEAD) dirty=$(git -C $SRC status --porcelain | wc -l); base $(git -C $BASE rev-parse HEAD) dirty=$(git -C $BASE status --porcelain | wc -l); detref $(git -C $DETREF rev-parse HEAD) dirty=$(git -C $DETREF status --porcelain | wc -l)" >> $O/status

# 1. cold build of the production bundle into the empty cache, then a warm load in a new process
stage build_cold bash -c "$PY -m sparse_pooling.backends.cuda.build --build --verbose > $O/build_cold.log 2>&1"
stage load_warm bash -c "$PY /tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh/load_warm.py > $O/load_warm.log 2>&1"
# 2. full suites
stage suite_auto bash -c "cd $SRC/sparse_pooling/tests && $PY -m pytest -q -rfs --tb=short -p no:cacheprovider --junitxml=$O/suite_auto.junit.xml > $O/suite_auto.log 2>&1"
stage suite_generic bash -c "cd $SRC/sparse_pooling/tests && SPARSE_POOL_PROFILE=generic_cuda $PY -m pytest -q -rfs --tb=short -p no:cacheprovider --junitxml=$O/suite_generic.junit.xml > $O/suite_generic.log 2>&1"
# 3. profile / target / cache identity on the RTX 3090 and mocked selections
stage identity bash -c "$PY $F/identity_probe.py > $O/identity.json 2> $O/identity.err"
# 4. independent cold build of the baseline (own cache) and SASS comparison of all bundles
stage build_base bash -c "cd $BASE && for b in production experimental diagnostics; do SPARSE_POOL_CACHE_DIR=$F/cache_base $PY -m sparse_pooling.backends.cuda.build --bundle \$b --build; done > $O/build_base.log 2>&1"
stage build_src_bundles bash -c "for b in experimental diagnostics; do $PY -m sparse_pooling.backends.cuda.build --bundle \$b --build; done > $O/build_src_bundles.log 2>&1"
stage sass bash -c "cd $SRC/sparse_pooling/benchmarks && $PY compare_sass.py --old $F/cache_base/sm_86/lib/*/libgranule_*.so --new $F/cache/sm_86/lib/*/libgranule_*.so --out $O/sass_compare.json > $O/sass_compare.log 2>&1"
echo STAGE_A_DONE >> $O/status

# 5. performance A/B (GPU otherwise idle): A = 34a6433, B = fd99c2d, same harness (B's)
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu --format=csv > $O/gpu_before_ab.csv
stage ab bash -c "bash $SRC/sparse_pooling/benchmarks/run_impl_ab.sh $PY $BASE $SRC $O/ab 4 all > $O/ab_run.log 2>&1"
stage ab_summary bash -c "cd $SRC/sparse_pooling/benchmarks && $PY summarize_impl_ab.py $O/ab --json $O/ab_summary.json --md $O/ab_summary.md > $O/ab_summary.log 2>&1"
echo STAGE_B_DONE >> $O/status

# 6. det regression (old 2b9a84d vs new fd99c2d) + harness self-test, concurrently with the sanitizer
( stage detreg bash -c "bash $SRC/sparse_pooling/benchmarks/run_det_regression.sh $PY $DETREF $F/detreg_scr $O/detreg all > $O/detreg_run.log 2>&1"
  stage selftest bash -c "bash $SRC/sparse_pooling/benchmarks/run_det_training.sh $PY $F/selftest_scr selftest $O/selftest > $O/selftest.log 2>&1" ) &
DPID=$!
stage sanitizer bash -c "bash $SRC/sparse_pooling/benchmarks/run_sanitizer.sh --jobs 4 --out $O/sanitizer $PY > $O/sanitizer_run.log 2>&1"
wait $DPID
echo ALLDONE >> $O/status
