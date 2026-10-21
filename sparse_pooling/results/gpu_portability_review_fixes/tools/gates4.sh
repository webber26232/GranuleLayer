#!/usr/bin/env bash
# Sequential, one systemd scope per stage.
set -uo pipefail
F=/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh; O=$F/out; PY=/home/evga/.conda/envs/env_isaaclab/bin/python; SRC=$F/src
export SPARSE_POOL_CACHE_DIR=$F/cache TRITON_CACHE_DIR=$F/triton
unset SPARSE_POOL_PROFILE SPARSE_POOL_ARCH SPARSE_POOL_NVCC_EXTRA
stage() { local name=$1; shift; echo "$name START $(date -Is) :: $*" >> $O/status; local t0=$(date +%s); "$@"; local rc=$?; echo "$name rc=$rc $(( $(date +%s) - t0 ))s" >> $O/status; return $rc; }
scope() { local unit=$1; shift; systemd-run --user --scope --quiet --unit="$unit" "$@"; }
cd $SRC
echo "RESUME4 src $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)" >> $O/status
stage sanitizer scope granule-san -p MemoryMax=24G bash -c "bash $SRC/sparse_pooling/benchmarks/run_sanitizer.sh --jobs 3 --out $O/sanitizer $PY > $O/sanitizer_run.log 2>&1"
stage ab_h1m_8 scope granule-ab -p MemoryHigh=16G -p MemoryMax=20G bash -c "bash $F/run_ab_h1m8.sh > $O/ab_h1m8_run.log 2>&1"
stage ab_h1m_8_summary bash -c "cd $SRC/sparse_pooling/benchmarks && $PY summarize_impl_ab.py $O/ab_h1m8 --json $O/ab_h1m8_summary.json --md $O/ab_h1m8_summary.md > /dev/null 2>&1"
rm -rf $F/detreg_scr/rec_P_*
stage detreg_production scope granule-det -p MemoryHigh=16G -p MemoryMax=20G bash -c "SKIP='S_S16_R4_det_split S_S64_R16_det_split M_S16_R4_sr_split M_S32_R8_det_split_cap64M M_S64_R16_det_bf16 M_S64_R16_fp32_split M64_S16_R4_det_split M64_S64_R16_det_bf16 G_M_S64_R16_det_split G_M64_S16_R4_sr_split' bash $F/run_det_regression_resume.sh $PY /home/evga/Projects/pamachine/GranuleLayer-det $F/detreg_scr $O/detreg all > $O/detreg_run3_production.log 2>&1"
echo ALLDONE4 >> $O/status
