#!/usr/bin/env bash
# Stage 2 medium screen (results/DETERMINISM_METRICS_DW_PROJECTION.md): fast vs deterministic
# reduction x padded vs compact dW projection, same graph / init / input per file; H = 1M / 10M /
# 30M, M = 1024 / 8192, S / R = (16, 4), (32, 8), (64, 16); a ragged H and a changed G / R case;
# the deterministic workspace cap at H = 30M, S = 64.  4 alternating rounds x 2 restored
# repetitions x 3 steps.  Single idle GPU, nothing else running.
#   bash run_dm_screen.sh <python>
set -euo pipefail
PY=${1:-/home/evga/.conda/envs/env_isaaclab/bin/python}
cd "$(dirname "$0")"
R=../results/determinism_metrics_dw_projection/screen
mkdir -p "$R"
V="--variant fast_padded:reduction=fast,cuda_grouped_dw_projection=padded
   --variant fast_compact:reduction=fast,cuda_grouped_dw_projection=compact
   --variant det_padded:reduction=deterministic,cuda_grouped_dw_projection=padded
   --variant det_compact:reduction=deterministic,cuda_grouped_dw_projection=compact"
run() {  # name, bench args...
  local name=$1; shift
  $PY bench_supergroups.py --C 8 --rounds 4 --reps 2 --iters 3 "$@" --out "$R/$name" > "$R/$name.log" 2>&1
  grep "^\[" "$R/$name.log"
}
for H in 1000000 10000000 30000000; do
  for M in 1024 8192; do
    for SR in "16 4" "32 8" "64 16"; do
      set -- $SR
      run "H${H}_M${M}_S$1_R$2" --hidden $H --batch $M --S $1 --R $2 $V
    done
  done
done
run "H23456789_M8192_S32_R8_ragged" --hidden 23456789 --batch 8192 --S 32 --R 8 $V
run "H30000000_M8192_G512_S64_R8" --hidden 30000000 --batch 8192 --groups 512 --S 64 --R 8 $V
run "H30000000_M8192_S64_R16_bf16" --hidden 30000000 --batch 8192 --S 64 --R 16 --grad-precision bf16 $V
run "H30000000_M8192_S64_R16_caps" --hidden 30000000 --batch 8192 --S 64 --R 16 \
  --variant cap256M:reduction=deterministic,deterministic_workspace_bytes=268435456 \
  --variant cap1G:reduction=deterministic \
  --variant cap4G:reduction=deterministic,deterministic_workspace_bytes=4294967296 \
  --variant fast:reduction=fast
echo screen-done
