#!/bin/bash
# merge / scatter attribution at H = 100M, S = 16 (wrong results, timing only; separate builds)
PY=$1; OUT=input_supergroups/ablation
for AB in 0 1 2 3; do
  for C in 8 64; do
    SPARSE_POOL_NVCC_EXTRA="-DSP_ABLATE_SG=$AB" $PY bench_supergroups.py --hidden 100000000 --C $C --S 16 --R 4 \
      --reps 2 --iters 2 --rounds 1 --variant grouped:cuda_input_supergroup_backend=grouped \
      --out $OUT/H100M_C${C}_S16_R4_ablate$AB
  done
done
