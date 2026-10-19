#!/usr/bin/env bash
# Held-out evaluation of a frozen default change (holdout_reserved.txt, reserved before any
# dW-projection timing): padded vs compact dW, and auto, in fast mode; 4 rounds x 2 x 3.
#   bash run_dm_holdout.sh <python>
set -euo pipefail
PY=${1:-/home/evga/.conda/envs/env_isaaclab/bin/python}
cd "$(dirname "$0")"
R=../results/determinism_metrics_dw_projection/holdout
mkdir -p "$R"
V="--variant padded:reduction=fast,cuda_grouped_dw_projection=padded
   --variant compact:reduction=fast,cuda_grouped_dw_projection=compact
   --variant auto:"
P="--C 8 --rounds 4 --reps 2 --iters 3"
run() {
  local name=$1; shift
  $PY bench_supergroups.py $P "$@" $V --out "$R/$name" > "$R/$name.log" 2>&1
  grep "^\[" "$R/$name.log"
}
run h1_S32_R8_H45M      --S 32 --R 8  --hidden 45000000 --batch 8192 --groups 1024 --graph-seed 3 --init-seed 3 --input-seed 3
run h2_S16_R4_H77777777 --S 16 --R 4  --hidden 77777777 --batch 8192 --groups 1024 --graph-seed 4 --init-seed 4 --input-seed 4
run h3_S64_R8_H60M_G512_M2048 --S 64 --R 8 --hidden 60000000 --batch 2048 --groups 512 --graph-seed 5 --init-seed 5 --input-seed 5
run h4_S64_R16_H5M_M1024_bf16 --S 64 --R 16 --hidden 5000000 --batch 1024 --groups 1024 --grad-precision bf16 --graph-seed 6 --init-seed 6 --input-seed 6
run h5_S16_R4_H15M_M4096_G256 --S 16 --R 4 --hidden 15000000 --batch 4096 --groups 256 --graph-seed 7 --init-seed 7 --input-seed 7
run h6_S32_R8_H2500000  --S 32 --R 8  --hidden 2500000 --batch 8192 --groups 1024 --graph-seed 8 --init-seed 8 --input-seed 8
echo holdout-done
