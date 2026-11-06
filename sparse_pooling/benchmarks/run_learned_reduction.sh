#!/usr/bin/env bash
# One-command reproduction of the learned-reduction study (docs/experiments/learned_sliding_window_reduction.md).
#   bash sparse_pooling/benchmarks/run_learned_reduction.sh <python> <results-root> <stage> [stage ...]
# stages: correctness sanitizer replicate screen frozen primary secondary perf supplementary report
# Every stage is resumable (per-job atomic JSON); run it detached, serialized, one workload per GPU:
#   systemd-run --user --unit=lsw -p MemoryHigh=20G -p MemoryMax=24G --working-directory=$PWD --setenv=PATH="$PATH" \
#       bash sparse_pooling/benchmarks/run_learned_reduction.sh python /path/results screen frozen primary perf
set -uo pipefail
PY=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export SPARSE_POOL_CACHE_DIR=${SPARSE_POOL_CACHE_DIR:-$OUT/cuda_cache}
PACKED=$OUT/packed
mkdir -p "$OUT/manifests" "$PACKED"
cd "$HERE"
stage() {  # campaign stage -> manifest -> training session
  "$PY" learned_reduction_campaign.py "$1" --out "$OUT" --write "$OUT/manifests/$1.json" || return 1
  "$PY" learned_reduction_train.py session --manifest "$OUT/manifests/$1.json" --root "$PACKED" --out "$OUT/$1"
}
for s in "$@"; do
  echo "=== $s $(date -Is)"
  case $s in
    correctness) "$PY" -m pytest "$HERE/../tests/test_learned_reduction.py" -q ;;
    sanitizer) bash run_sanitizer.sh --jobs 4 --out "$OUT/sanitizer" "$PY" lrd ;;
    replicate) "$PY" learned_reduction_train.py replicate --checkpoint "${CKPT_G0:?K64 checkpoint of graph 0}" --out "$OUT/stage0/replicate_g0_s12.json" ;;
    screen) stage screen_tune1; stage screen_tune2; stage screen_final ;;
    frozen) stage frozen_tune; stage frozen_final ;;
    primary) stage primary_smoke; stage primary_tune; stage primary_final ;;
    secondary) stage secondary ;;
    supplementary) for x in control_tune control_final horizon fresh secondary_leader m8192; do stage $x; done ;;
    perf) "$PY" learned_reduction_perf.py session --plan "$HERE/../../configs/learned_reduction/perf_plan.json" --root "$PACKED" --out "$OUT/perf" --order-seed 20261106 ;;
    report) "$PY" learned_reduction_report.py --out "$OUT" ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
