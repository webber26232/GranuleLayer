#!/usr/bin/env bash
# Run NVIDIA Compute Sanitizer over a small workload touching every kernel.
#   bash sparse_pooling/benchmarks/run_sanitizer.sh [python]
set -euo pipefail
PY=${1:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
for tool in memcheck racecheck initcheck synccheck; do
  echo "=== compute-sanitizer --tool $tool"
  compute-sanitizer --tool "$tool" --target-processes all --print-limit 20 \
      "$PY" "$HERE/sanitize_workload.py" 2>&1 | grep -E "ERROR SUMMARY|^ok|Invalid|Race|Uninitialized|Barrier" | sort | uniq -c
done
