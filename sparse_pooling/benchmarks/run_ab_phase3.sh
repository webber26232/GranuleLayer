#!/usr/bin/env bash
# Phase-3 interleaved A/B at H = 100M, C = 64, M = 8192, Mean(ReLU) of compile-time
# variants (each its own library, identical flags apart from the -D switch).
#   bash sparse_pooling/benchmarks/run_ab_phase3.sh <python> <exp> <rounds> name:"flags" ...
# The order of the variants is reversed in every even round.
set -uo pipefail
PY=$1; EXP=$2; ROUNDS=$3; shift 3
VARS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
PREC=${PREC:-split}; HID=${HID:-100000000}; C=${C:-64}; M=${M:-8192}; POOL=${POOL:-mean}
run() {  # $1 round, $2 name:flags
  local name=${2%%:*} flags=${2#*:}
  SPARSE_POOL_NVCC_EXTRA="$flags" "$PY" "$HERE/bench_kernels.py" --hidden "$HID" --C "$C" --batch "$M" --pooling "$POOL" \
    --grad-precision "$PREC" --reps 5 --iters 3 --tag "${EXP}_r$1" \
    --out "mma_forward/${EXP}_${POOL}_H${HID}_C${C}_M${M}_${PREC}_${name}_r$1" 2>&1 | grep -v Warning \
    | sed "s/^/$(printf '%-10s' "$name") /"
}
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then for (( i=${#VARS[@]}-1; i>=0; i-- )); do run "$r" "${VARS[$i]}"; done
  else for v in "${VARS[@]}"; do run "$r" "$v"; done; fi
done
