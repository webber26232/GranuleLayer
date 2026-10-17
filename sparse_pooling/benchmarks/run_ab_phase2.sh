#!/usr/bin/env bash
# Phase-2 interleaved A/B + attribution (H = 100M, C = 64, M = 8192, Mean(ReLU)).
#   baseline = the dw-indexed-staging commit in a git worktree (its own build)
#   new_*    = this branch with the epilogue switches of the attribution matrix
#   bash sparse_pooling/benchmarks/run_ab_phase2.sh <python> <baseline-worktree> [rounds] [precision]
set -uo pipefail
PY=$1; WT=$2; ROUNDS=${3:-4}; PREC=${4:-split}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../results/backward_compute"
COMMON=(--hidden 100000000 --C 64 --batch 8192 --pooling mean --grad-precision "$PREC" --reps 5 --iters 3)
base() {
  (cd "$WT" && "$PY" sparse_pooling/benchmarks/bench_kernels.py "${COMMON[@]}" --tag "ab2_r$1" \
     --out "backward_compute/ab2_mean_H100M_C64_M8192_${PREC}_baseline_r$1" 2>&1 | grep -v Warning | sed 's/^/baseline     /')
  cp "$WT/sparse_pooling/results/backward_compute/ab2_mean_H100M_C64_M8192_${PREC}_baseline_r$1.json" "$OUT/"
}
new() {  # $1 round, $2 name, $3 dx epilogue, $4 dw epilogue
  "$PY" "$HERE/bench_kernels.py" "${COMMON[@]}" --dx-epilogue "$3" --dw-epilogue "$4" --tag "ab2_r$1" \
     --out "backward_compute/ab2_mean_H100M_C64_M8192_${PREC}_$2_r$1" 2>&1 | grep -v Warning | sed "s/^/$(printf '%-12s' "$2") /"
}
for r in $(seq 1 "$ROUNDS"); do
  if [ "$PREC" = split ]; then
    runs=("base $r" "new $r generic generic generic" "new $r dw_only generic fast" "new $r dx_only mask generic" "new $r combined auto auto")
  else
    runs=("base $r" "new $r combined auto auto")
  fi
  if (( r % 2 == 0 )); then for (( i=${#runs[@]}-1; i>=0; i-- )); do eval "${runs[$i]}"; done
  else for run in "${runs[@]}"; do eval "$run"; done; fi
done
