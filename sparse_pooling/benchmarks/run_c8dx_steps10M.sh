#!/bin/bash
# Stage-2 complete training steps (corrected harness) at H = 10M, M = 8192: scatter finalists
# against the unchanged grouped baseline (same grouped forward / dW, auto tiles).
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/screen
d() { echo "$1:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=$2,cuda_grouped_dx_warps=$3,cuda_grouped_dx_reduction=$4,cuda_grouped_dx_scatter=$5"; }
$PY bench_supergroups.py --hidden 10000000 --C 8 --S 16 --R 4 --rounds 4 --reps 2 --iters 3 \
  --variant baseline:cuda_input_supergroup_backend=auto \
  --variant "$(d vectorized 64 8 1 vectorized)" --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)" \
  --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" \
  --variant "$(d feature_owner 64 8 1 feature_owner)" --out $OUT/steps10M_S16 | tee $OUT/steps10M_S16.log
$PY bench_supergroups.py --hidden 10000000 --C 8 --S 32 --R 8 --rounds 4 --reps 2 --iters 3 \
  --variant baseline:cuda_input_supergroup_backend=auto \
  --variant "$(d vectorized 32 8 1 vectorized)" --variant "$(d vectorized_hoisted 32 8 1 vectorized_hoisted)" \
  --variant "$(d vectorized_transpose_hoisted 32 8 1 vectorized_transpose_hoisted)" \
  --out $OUT/steps10M_S32 | tee $OUT/steps10M_S32.log
$PY bench_supergroups.py --hidden 10000000 --C 8 --S 64 --R 16 --rounds 4 --reps 2 --iters 3 \
  --variant baseline:cuda_input_supergroup_backend=auto \
  --variant "$(d swizzle4_T64 64 8 8 swizzle4)" --variant "$(d baseline_T128W16 128 16 8 baseline)" \
  --variant "$(d swizzle4_T128W16 128 16 8 swizzle4)" --out $OUT/steps10M_S64 | tee $OUT/steps10M_S64.log
