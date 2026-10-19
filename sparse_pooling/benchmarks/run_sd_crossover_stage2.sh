#!/bin/bash
# Track D stage 2 (one seed): intermediate H around the crossover, bucket length (G), overlap /
# supergroup count (R), token count (M), BF16 gradients.  Plan A (shared_t64) vs Plan B
# (private_t32); the original grouped scatter as context on one shape.  Same harness as stage 1.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/crossover_stage2
mkdir -p $OUT
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
O=baseline:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=64,cuda_grouped_dx_warps=8,cuda_grouped_dx_reduction=8,cuda_grouped_dx_scatter=baseline
run() { local name=$1; shift; $PY bench_supergroups.py --C 8 --S 64 --rounds 4 --reps 2 --iters 3 "$@" --out $OUT/$name 2>&1 | tee $OUT/$name.log; }
for H in 20000000 40000000 70000000; do run H${H}_R16 --hidden $H --R 16 --variant $A --variant $B; done
run H10000000_G256 --hidden 10000000 --R 16 --groups 256 --variant $A --variant $B
run H30000000_G256 --hidden 30000000 --R 16 --groups 256 --variant $A --variant $B --variant $O
for R in 8 32; do for H in 30000000 50000000; do run H${H}_R$R --hidden $H --R $R --variant $A --variant $B; done; done
for M in 1024 4096; do for H in 30000000 100000000; do run H${H}_M$M --hidden $H --R 16 --batch $M --variant $A --variant $B; done; done
for H in 30000000 100000000; do run H${H}_bf16 --hidden $H --R 16 --grad-precision bf16 --variant $A --variant $B; done
