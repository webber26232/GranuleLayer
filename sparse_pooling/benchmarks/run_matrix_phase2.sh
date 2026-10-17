#!/usr/bin/env bash
# Phase-2 configuration matrix: baseline worktree vs this branch (defaults), one run each.
#   bash sparse_pooling/benchmarks/run_matrix_phase2.sh <python> <baseline-worktree>
set -uo pipefail
PY=$1; WT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../results/backward_compute"
one() {  # $1 name suffix, rest: bench args
  local name=$1; shift
  (cd "$WT" && "$PY" sparse_pooling/benchmarks/bench_kernels.py "$@" --reps 3 --iters 3 --tag matrix \
     --out "backward_compute/matrix_${name}_baseline_exp03" 2>&1 | grep -v Warning | sed 's/^/baseline  /')
  cp "$WT/sparse_pooling/results/backward_compute/matrix_${name}_baseline_exp03.json" "$OUT/"
  "$PY" "$HERE/bench_kernels.py" "$@" --reps 3 --iters 3 --tag matrix \
     --out "backward_compute/matrix_${name}_optimized_exp03" 2>&1 | grep -v Warning | sed 's/^/optimized /'
}
for M in 8192 1024; do
  for H in 1000000 10000000 100000000; do
    for C in 8 64; do
      one "mean_H${H}_C${C}_M${M}_split" --hidden $H --C $C --batch $M --pooling mean --grad-precision split
    done
    one "mean_H${H}_C64_M${M}_bf16" --hidden $H --C 64 --batch $M --pooling mean --grad-precision bf16
  done
done
for op in lme sw pow2; do for prec in split bf16; do
  one "${op}_H100000000_C64_M8192_${prec}" --hidden 100000000 --C 64 --batch 8192 --pooling $op --grad-precision $prec
done; done
one "lme_H100000000_C8_M8192_split" --hidden 100000000 --C 8 --batch 8192 --pooling lme --grad-precision split
for C in 8 64; do
  one "max_H100000000_C${C}_M8192_split" --hidden 100000000 --C $C --batch 8192 --pooling max --grad-precision split
done
