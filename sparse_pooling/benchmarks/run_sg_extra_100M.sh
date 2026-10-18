#!/bin/bash
PY=$1; OUT=input_supergroups/final_100M; G=cuda_input_supergroup_backend
for C in 8 64; do
  for M in 256 1024; do
    $PY bench_supergroups.py --hidden 100000000 --batch $M --C $C --S 16 --R 4 --reps 3 --iters 3 --rounds 2 \
      --variant generic:$G=generic --variant auto:$G=auto --out $OUT/H100M_M${M}_C${C}_S16_R4_seed0_split
  done
done
$PY bench_supergroups.py --hidden 100000000 --C 64 --S 16 --R 4 --grad-precision bf16 --reps 2 --iters 2 --rounds 2 \
  --variant generic:$G=generic --variant auto:$G=auto --out $OUT/H100M_C64_S16_R4_seed0_bf16_auto2
$PY bench_supergroups.py --hidden 100000000 --C 64 --S 64 --R 16 --reps 2 --iters 2 --rounds 2 \
  --variant generic:$G=generic --variant auto:$G=auto --out $OUT/H100M_C64_S64_R16_seed0_split_auto2
