#!/bin/bash
# Track D stage 1: S = 64 dX crossover screen (C = 8, R = 16, D = G = 1024, M = 8192, split, seed 0).
# Plan A (auto today: shared_t64) vs Plan B (private_t32) vs the original grouped scatter
# (explicit), complete-state harness, 4 alternating rounds x 2 restored repetitions x 3 steps.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/crossover_screen
mkdir -p $OUT
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
O=baseline:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=64,cuda_grouped_dx_warps=8,cuda_grouped_dx_reduction=8,cuda_grouped_dx_scatter=baseline
for H in ${HS:-1000000 10000000 30000000 50000000 100000000}; do
  $PY bench_supergroups.py --hidden $H --C 8 --S 64 --R 16 --rounds 4 --reps 2 --iters 3 \
    --variant $A --variant $B --variant $O "$@" --out $OUT/H$H 2>&1 | tee $OUT/H$H.log
done
