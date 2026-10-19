#!/bin/bash
# Track D: BF16 gradients at S = 64: the current automatic plan (per-field tiles; the original scatter,
# since SG_AUTO_DX_SCATTER has no BF16 S = 64 entry) vs Plan A vs Plan B.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/crossover_bf16
mkdir -p $OUT
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
T=auto_tiles:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=tiles
for H in 10000000 20000000 30000000 100000000; do
  $PY bench_supergroups.py --C 8 --S 64 --R 16 --grad-precision bf16 --rounds 4 --reps 2 --iters 3 --hidden $H \
    --variant $T --variant $A --variant $B --out $OUT/H$H 2>&1 | tee $OUT/H$H.log
done
