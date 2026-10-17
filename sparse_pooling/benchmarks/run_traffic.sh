#!/usr/bin/env bash
# Per-kernel measurement matrix for results/MEMORY_TRAFFIC_OPTIMIZATION.md.
#   bash sparse_pooling/benchmarks/run_traffic.sh [python] [original|optimized]
#   original  = full-slab k_dw, 16-warp k_dx (the implementation before this work)
#   optimized = library defaults (indexed k_dw staging + 32-warp k_dx at C = 64)
set -uo pipefail
PY=${1:-python}
MODE=${2:-optimized}
HERE=$(cd "$(dirname "$0")" && pwd)
B="$HERE/bench_kernels.py"
if [ "$MODE" = original ]; then FLAGS=(--dw-staging full --dx-warps 16); else FLAGS=(); fi
run() { "$PY" "$B" "${FLAGS[@]}" --tag "$MODE" "$@" 2>&1 | grep -v Warning; }
for H in 1000000 10000000 100000000; do
  for C in 8 64; do
    run --hidden $H --C $C --pooling mean --grad-precision split
  done
  run --hidden $H --C 64 --pooling mean --grad-precision bf16
done
run --hidden 100000000 --C 8 --pooling max
run --hidden 100000000 --C 64 --pooling max
for op in lme sw pow2; do
  run --hidden 100000000 --C 64 --pooling $op --grad-precision split
done
