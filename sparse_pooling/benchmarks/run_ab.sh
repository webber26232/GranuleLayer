#!/usr/bin/env bash
# Interleaved A/B of the primary configuration (H = 100M, C = 64, Mean(ReLU), split):
#   original = full-slab k_dw + 16-warp k_dx, optimized = library defaults.
#   bash sparse_pooling/benchmarks/run_ab.sh [python] [rounds] [extra bench_kernels args...]
set -uo pipefail
PY=${1:-python}
ROUNDS=${2:-4}
shift 2 || true
B="$(cd "$(dirname "$0")" && pwd)/bench_kernels.py"
orig() { "$PY" "$B" --hidden 100000000 --C 64 --dw-staging full --dx-warps 16 --tag "ab_r$1" "${@:2}" 2>&1 | grep -v Warning; }
opt()  { "$PY" "$B" --hidden 100000000 --C 64 --tag "ab_r$1" "${@:2}" 2>&1 | grep -v Warning; }
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then orig "$r" "$@"; opt "$r" "$@"; else opt "$r" "$@"; orig "$r" "$@"; fi
done
