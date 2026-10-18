#!/bin/bash
# Re-run of the more-overlapping graph (C = 8, S = 16, R = 16, H = 10M) with an explicit baseline
# scatter (the first run_c8dx_extra.sh run used the auto plan as "baseline" after the auto rule
# selected vectorized_transpose_hoisted there: both variants were the same kernel).
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/extra
d() { echo "$1:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=$2,cuda_grouped_dx_warps=$3,cuda_grouped_dx_reduction=$4,cuda_grouped_dx_scatter=$5"; }
$PY bench_supergroups.py --hidden 10000000 --C 8 --S 16 --R 16 --rounds 4 --reps 2 --iters 3 \
  --variant baseline:cuda_input_supergroup_backend=auto,cuda_grouped_dx_scatter=baseline \
  --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" \
  --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)" --variant selected:cuda_input_supergroup_backend=auto \
  --out $OUT/H10M_C8_S16_R16_explicit 2>&1 | tee $OUT/H10M_C8_S16_R16_explicit.log
