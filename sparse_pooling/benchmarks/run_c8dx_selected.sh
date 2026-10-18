#!/bin/bash
# Selected plan (cuda_grouped_dx_scatter = auto) against the explicit baseline scatter, both with the
# auto backend / tiles, H = 100M, M = 8192, split, corrected harness.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/selected_100M
mkdir -p $OUT
for SR in "16 4" "32 8" "64 16"; do
  read -r S R <<< "$SR"
  $PY bench_supergroups.py --hidden 100000000 --C 8 --S $S --R $R --rounds 4 --reps 2 --iters 3 \
    --variant baseline:cuda_input_supergroup_backend=auto,cuda_grouped_dx_scatter=baseline \
    --variant selected:cuda_input_supergroup_backend=auto --out $OUT/C8_S${S}_R${R} 2>&1 | tee $OUT/C8_S${S}_R${R}.log
done
