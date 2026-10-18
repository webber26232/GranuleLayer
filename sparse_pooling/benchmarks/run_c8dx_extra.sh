#!/bin/bash
# (1) a more-overlapping graph (C = 8, S = 16, R = 16) at H = 10M, M = 8192: finalists vs baseline;
# (2) secondary C = 64 dimension check: S = 16 / 64 / 128 (R = 4 / 16 / 32) at H = 100M, M = 8192 in
#     one session with the existing supported per-phase plans (auto) and the generic backend.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/extra
mkdir -p $OUT
d() { echo "$1:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=$2,cuda_grouped_dx_warps=$3,cuda_grouped_dx_reduction=$4,cuda_grouped_dx_scatter=$5"; }
B=baseline:cuda_input_supergroup_backend=auto
$PY bench_supergroups.py --hidden 10000000 --C 8 --S 16 --R 16 --rounds 4 --reps 2 --iters 3 --variant $B \
  --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" \
  --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)" --out $OUT/H10M_C8_S16_R16 2>&1 | tee $OUT/H10M_C8_S16_R16.log
for SR in "16 4" "64 16" "128 32"; do
  read -r S R <<< "$SR"
  $PY bench_supergroups.py --hidden 100000000 --C 64 --S $S --R $R --rounds 4 --reps 2 --iters 3 --variant $B \
    --variant generic:cuda_input_supergroup_backend=generic --out $OUT/H100M_C64_S${S}_R${R} 2>&1 | tee $OUT/H100M_C64_S${S}_R${R}.log
done
