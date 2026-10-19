#!/bin/bash
# Track D held-out validation of the scale rule (shapes NOT used to choose the thresholds): Plan A,
# Plan B and the new automatic choice (auto), plus the previous BF16 default (tiles) for BF16.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/holdout
mkdir -p $OUT
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
U=auto:cuda_input_supergroup_backend=auto
T=prev_bf16_default:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=tiles
run() { local name=$1; shift; $PY bench_supergroups.py --C 8 --S 64 --rounds 4 --reps 2 --iters 3 "$@" --out $OUT/$name 2>&1 | tee $OUT/$name.log; }
run a_H45M --hidden 45000000 --R 16 --variant $A --variant $B --variant $U
run b_H33M --hidden 33000000 --R 16 --variant $A --variant $B --variant $U
run c_H77777777 --hidden 77777777 --R 16 --variant $A --variant $B --variant $U
run d_H12M --hidden 12000000 --R 16 --variant $A --variant $B --variant $U
run e_H25M_G512 --hidden 25000000 --R 16 --groups 512 --variant $A --variant $B --variant $U
run f_H60M_M2048 --hidden 60000000 --R 16 --batch 2048 --variant $A --variant $B --variant $U
run g_H20M_R8 --hidden 20000000 --R 8 --variant $A --variant $B --variant $U
run h_H100M_R32 --hidden 100000000 --R 32 --variant $A --variant $B --variant $U
run i_H100M_seed1 --hidden 100000000 --R 16 --graph-seed 1 --init-seed 1 --input-seed 1 --variant $A --variant $B --variant $U
run j_H15M_bf16 --hidden 15000000 --R 16 --grad-precision bf16 --variant $A --variant $B --variant $U --variant $T
run k_H50M_bf16 --hidden 50000000 --R 16 --grad-precision bf16 --variant $A --variant $B --variant $U --variant $T
run l_H30M_M1024_bf16 --hidden 30000000 --R 16 --batch 1024 --grad-precision bf16 --variant $A --variant $B --variant $U --variant $T
run m_H5M_M1024_bf16 --hidden 5000000 --R 16 --batch 1024 --grad-precision bf16 --variant $A --variant $B --variant $U --variant $T
