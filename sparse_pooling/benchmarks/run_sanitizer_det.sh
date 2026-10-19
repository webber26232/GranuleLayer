#!/usr/bin/env bash
# Compute Sanitizer over the deterministic reduction and compact dW kernels.
#   bash sparse_pooling/benchmarks/run_sanitizer_det.sh [python]
set -uo pipefail
PY=${1:-python}
export CUDA_MODULE_LOADING=LAZY  # instrument only the kernels that run (the library holds ~500)
HERE=$(cd "$(dirname "$0")" && pwd)
for tool in memcheck racecheck initcheck synccheck; do
  echo "=== compute-sanitizer --tool $tool"
  extra=""
  [ "$tool" = racecheck ] && extra="--racecheck-report all"
  compute-sanitizer --tool "$tool" $extra --target-processes all --print-limit 20 \
      "$PY" "$HERE/sanitize_det_workload.py" 2>&1 | grep -E "ERROR SUMMARY|RACECHECK SUMMARY|^ok|^skip|Invalid|Race|Uninitialized|Barrier|hazard" | sort | uniq -c
done
