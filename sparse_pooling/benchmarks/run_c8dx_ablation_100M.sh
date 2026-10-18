#!/bin/bash
# Timing-only attribution (WRONG dX: SP_ABLATE_SG=2 removes the baseline's local slab update and
# keeps the products live; separate build): baseline sg_dx at H = 100M, S = 16 / 32, auto tiles.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/ablation_100M
mkdir -p $OUT
for SR in "16 4" "32 8"; do
  read -r S R <<< "$SR"
  SPARSE_POOL_NVCC_EXTRA="-DSP_ABLATE_SG=2" $PY bench_supergroups.py --hidden 100000000 --C 8 --S $S --R $R \
    --rounds 2 --reps 2 --iters 3 --variant noscatter_ablation:cuda_input_supergroup_backend=auto,cuda_grouped_dx_scatter=baseline \
    --out $OUT/C8_S${S}_R${R}_ablate2 2>&1 | tee $OUT/C8_S${S}_R${R}_ablate2.log
done
