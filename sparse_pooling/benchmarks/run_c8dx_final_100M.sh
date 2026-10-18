#!/bin/bash
# Stage 3, H = 100M, M = 8192 (corrected harness: complete training-state restore, pinned host
# snapshot, 4 alternating-order rounds x 2 restored repetitions x 3 steps).  Explicit dX variants
# against the unchanged grouped baseline (auto plan); grouped forward / dW identical.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/final_100M
mkdir -p $OUT
d() { echo "$1:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=$2,cuda_grouped_dx_warps=$3,cuda_grouped_dx_reduction=$4,cuda_grouped_dx_scatter=$5"; }
B=baseline:cuda_input_supergroup_backend=auto
run() { local name=$1; shift; $PY bench_supergroups.py --hidden 100000000 --rounds 4 --reps 2 --iters 3 "$@" --out $OUT/$name 2>&1 | tee $OUT/$name.log; }
# --- C = 8, S = 16, three graph / input / init seeds (seed 0 adds candidate B and the generic reference)
run C8_S16_R4_seed0 --C 8 --S 16 --R 4 --variant $B --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" \
  --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)" --variant "$(d feature_owner 64 8 1 feature_owner)" \
  --variant generic:cuda_input_supergroup_backend=generic
for s in 1 2; do
  run C8_S16_R4_seed$s --C 8 --S 16 --R 4 --graph-seed $s --init-seed $s --input-seed $s --variant $B \
    --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)"
done
# --- C = 8, S = 32
for s in 0 1 2; do
  run C8_S32_R8_seed$s --C 8 --S 32 --R 8 --graph-seed $s --init-seed $s --input-seed $s --variant $B \
    --variant "$(d vectorized_transpose_hoisted 32 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 32 8 1 vectorized_hoisted)"
done
# --- C = 8, S = 64: regression / transferability (shared CAS copies)
run C8_S64_R16_seed0 --C 8 --S 64 --R 16 --variant $B --variant "$(d swizzle4_capped 64 8 8 swizzle4_capped)" \
  --variant "$(d swizzle4_T128W16 128 16 8 swizzle4)" --variant "$(d vectorized_transpose_hoisted_T32 32 8 1 vectorized_transpose_hoisted)"
# --- unchanged C = 64 controls
run C64_S16_R4_seed0 --C 64 --S 16 --R 4 --variant $B
run C64_S64_R16_seed0 --C 64 --S 64 --R 16 --variant $B
# --- historical-protocol diagnostic (weights / bias reset only) for the unchanged baselines
run C8_S16_R4_seed0_historical --protocol historical --C 8 --S 16 --R 4 --variant $B --variant generic:cuda_input_supergroup_backend=generic
run C64_S16_R4_seed0_historical --protocol historical --C 64 --S 16 --R 4 --variant $B
# --- finalists: BF16 gradients and M = 1024
run C8_S16_R4_bf16 --C 8 --S 16 --R 4 --grad-precision bf16 --variant $B \
  --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)"
run C8_S32_R8_bf16 --C 8 --S 32 --R 8 --grad-precision bf16 --variant $B \
  --variant "$(d vectorized_transpose_hoisted 32 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 32 8 1 vectorized_hoisted)"
run C8_S16_R4_M1024 --C 8 --S 16 --R 4 --batch 1024 --variant $B \
  --variant "$(d vectorized_transpose_hoisted 64 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 64 8 1 vectorized_hoisted)"
run C8_S32_R8_M1024 --C 8 --S 32 --R 8 --batch 1024 --variant $B \
  --variant "$(d vectorized_transpose_hoisted 32 8 1 vectorized_transpose_hoisted)" --variant "$(d vectorized_hoisted 32 8 1 vectorized_hoisted)"
