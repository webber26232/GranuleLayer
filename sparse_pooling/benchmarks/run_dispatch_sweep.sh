#!/usr/bin/env bash
# dX warp-count crossover (P6): H x M x {16, 32} warps, Mean(ReLU), C = 64, mask/fast epilogues.
#   bash sparse_pooling/benchmarks/run_dispatch_sweep.sh [python] [pooling]
set -uo pipefail
PY=${1:-python}
POOL=${2:-mean}
B="$(cd "$(dirname "$0")" && pwd)/bench_kernels.py"
EPI=(--dx-epilogue mask --dw-epilogue fast)
[ "$POOL" != mean ] && EPI=(--dx-epilogue fast --dw-epilogue fast)
for M in 8192 1024; do
  for H in 1000000 2000000 4000000 10000000 30000000 100000000; do
    for W in 16 32; do
      "$PY" "$B" --hidden $H --C 64 --batch $M --pooling "$POOL" "${EPI[@]}" --dx-warps $W --reps 3 --iters 3 \
        --tag dispatch --out "backward_compute/dispatch_${POOL}_H${H}_C64_M${M}_split_dxw${W}_exp02" 2>&1 | grep -v Warning
    done
  done
done
