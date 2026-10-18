#!/usr/bin/env bash
# Phase-3 final interleaved A/B + attribution matrix.  Each entry runs bench_kernels.py
# from its own source tree (git worktree) with its own build:
#   name|tree|nvcc extra flags|extra bench args
#   bash run_ab_phase3_final.sh <python> <exp> <rounds> 'baseline|../GL_base||' 'fwd_only|.||' ...
# Results: <this tree>/sparse_pooling/results/mma_forward/<exp>_<config>_<name>_r<k>.json
set -uo pipefail
PY=$1; EXP=$2; ROUNDS=$3; shift 3
VARS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$ROOT/sparse_pooling/results/mma_forward"
PREC=${PREC:-split}; HID=${HID:-100000000}; C=${C:-64}; M=${M:-8192}; POOL=${POOL:-mean}
CFG="${POOL}_H${HID}_C${C}_M${M}_${PREC}"
run() {  # $1 round, $2 entry
  IFS='|' read -r name tree flags extra <<< "$2"
  local t; t=$(cd "$ROOT" && cd "$tree" && pwd)
  local rel="mma_forward/${EXP}_${CFG}_${name}_r$1"
  (cd "$t" && SPARSE_POOL_NVCC_EXTRA="$flags" "$PY" sparse_pooling/benchmarks/bench_kernels.py --hidden "$HID" --C "$C" \
     --batch "$M" --pooling "$POOL" --grad-precision "$PREC" --reps 5 --iters 3 --tag "${EXP}_r$1" $extra --out "$rel" 2>&1 \
     | grep -v Warning | sed "s/^/$(printf '%-10s' "$name") /")
  [ "$t" != "$ROOT" ] && mkdir -p "$OUT" && cp "$t/sparse_pooling/results/$rel.json" "$OUT/" 2>/dev/null
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader | sed "s/^/$(printf '%-10s' "$name") gpu after: /"
}
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then for (( i=${#VARS[@]}-1; i>=0; i-- )); do run "$r" "${VARS[$i]}"; done
  else for v in "${VARS[@]}"; do run "$r" "$v"; done; fi
done
