#!/usr/bin/env bash
# Final paired timings at H = 100M, M = 8192 (results/DETERMINISM_METRICS_DW_PROJECTION.md):
#   * C = 8 split, S / R = (16, 4), (32, 8), (64, 16): fast padded vs fast compact dW, 3 seeds; seed 0
#     also deterministic padded vs compact (fast vs deterministic overhead per S / R);
#   * S = 64 BF16 gradients: the same four variants;
#   * deterministic workspace cap at S = 64 (256 MiB / 1 GiB / 4 GiB);
#   * composed model (projection -> layer -> head) at S = 64: fast vs deterministic;
#   * C = 64 fast controls (S = 16 / 64) and the previous phase's code (c9a591e, ../GranuleLayer-sdfb)
#     for C = 8 auto and C = 64 in the same session (regression check of the fast path);
#   * fresh-process memory: one variant per process (allocator state not shared between variants).
# 4 alternating rounds x 2 restored repetitions x 3 steps; single idle GPU.
#   bash run_dm_final.sh <python>
set -euo pipefail
PY=${1:-/home/evga/.conda/envs/env_isaaclab/bin/python}
cd "$(dirname "$0")"
R=../results/determinism_metrics_dw_projection/final_100M
mkdir -p "$R"
FP="--variant fast_padded:reduction=fast,cuda_grouped_dw_projection=padded
    --variant fast_compact:reduction=fast,cuda_grouped_dw_projection=compact"
DP="--variant det_padded:reduction=deterministic,cuda_grouped_dw_projection=padded
    --variant det_compact:reduction=deterministic,cuda_grouped_dw_projection=compact"
run() {
  local name=$1; shift
  $PY bench_supergroups.py --hidden 100000000 --batch 8192 "$@" --out "$R/$name" > "$R/$name.log" 2>&1
  grep "^\[" "$R/$name.log"
}
P="--rounds 4 --reps 2 --iters 3"
for SR in "16 4" "32 8" "64 16"; do
  set -- $SR
  run "C8_S$1_R$2_seed0" --C 8 --S $1 --R $2 $P $FP $DP
  for s in 1 2; do
    run "C8_S$1_R$2_seed$s" --C 8 --S $1 --R $2 --graph-seed $s --init-seed $s --input-seed $s $P $FP
  done
done
run "C8_S64_R16_bf16" --C 8 --S 64 --R 16 --grad-precision bf16 $P $FP $DP
run "C8_S64_R16_caps" --C 8 --S 64 --R 16 $P \
  --variant cap256M:reduction=deterministic,deterministic_workspace_bytes=268435456 \
  --variant cap1G:reduction=deterministic \
  --variant cap4G:reduction=deterministic,deterministic_workspace_bytes=4294967296
run "C8_S64_R16_composed" --C 8 --S 64 --R 16 --composed 256,256 $P \
  --variant fast:reduction=fast --variant det:reduction=deterministic
for SR in "16 4" "64 16"; do
  set -- $SR
  run "C64_S$1_R$2" --C 64 --S $1 --R $2 $P --variant fast_auto:reduction=fast --variant det:reduction=deterministic
done
# previous phase's code, same session (its own worktree / library; default plans = its auto)
PREV=../../../GranuleLayer-sdfb/sparse_pooling/benchmarks
for spec in "8 16 4" "8 32 8" "8 64 16" "64 16 4" "64 64 16"; do
  set -- $spec
  (cd "$PREV" && $PY bench_supergroups.py --hidden 100000000 --batch 8192 --C $1 --S $2 --R $3 $P \
     --variant prev_auto: --out "$OLDPWD/$R/prev_C$1_S$2_R$3" > "$OLDPWD/$R/prev_C$1_S$2_R$3.log" 2>&1)
  grep "^\[" "$R/prev_C$1_S$2_R$3.log"
done
# fresh-process memory (one variant per process)
for SR in "16 4" "32 8" "64 16"; do
  set -- $SR
  for v in fast:reduction=fast det:reduction=deterministic; do
    run "mem_C8_S$1_R$2_${v%%:*}" --C 8 --S $1 --R $2 --rounds 1 --reps 1 --iters 2 --variant $v
  done
done
echo final-done
