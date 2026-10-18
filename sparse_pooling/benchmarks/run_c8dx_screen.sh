#!/bin/bash
# Stage-2 screening of the grouped C = 8 dX scatter variants (H = 10M, M = 8192):
# constant-upstream dX microbenchmarks (bench_dx_micro.py) for S = 16 / 32 / 64 over
# variants x tiles; infeasible shared-memory configurations are skipped and recorded.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/c8_dx_scatter/screen
mkdir -p $OUT
SCAT="baseline swizzle4 vectorized vectorized_transpose vectorized_parity feature_owner"
for SR in "16 4" "32 8" "64 16"; do
  read -r S R <<< "$SR"
  V=()
  for T in 32 64 128; do for NW in 8 16; do for sc in $SCAT; do
    V+=(--variant "${sc}_T${T}_W${NW}:cuda_grouped_dx_tokens=$T,cuda_grouped_dx_warps=$NW,cuda_grouped_dx_reduction=1,cuda_grouped_dx_scatter=$sc")
  done; done; done
  if [ "$S" = 64 ]; then  # shared FP32 copies (64-bit CAS): baseline / swizzle4 layouts
    for T in 32 64 128; do for NW in 8 16; do for WG in 2 4 8 16; do for sc in baseline swizzle4; do
      [ "$WG" -le "$NW" ] && V+=(--variant "${sc}_T${T}_W${NW}_G${WG}:cuda_grouped_dx_tokens=$T,cuda_grouped_dx_warps=$NW,cuda_grouped_dx_reduction=$WG,cuda_grouped_dx_scatter=$sc")
    done; done; done; done
  fi
  $PY bench_dx_micro.py --hidden 10000000 --S $S --R $R --rounds 4 --iters 3 --skip-infeasible "${V[@]}" \
    --out $OUT/micro_S${S} 2>&1 | tee $OUT/micro_S${S}.log
done
