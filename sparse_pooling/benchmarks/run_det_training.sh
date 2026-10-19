#!/usr/bin/env bash
# Composed-model training repeatability (det_training.py): every case runs in two independent
# processes (write, then compare against the first process's records); records are streamed to a
# scratch directory and deleted after the pair.
#   bash run_det_training.sh <python> <scratch dir> [list: small | production | fast]
set -euo pipefail
PY=${1:-/home/evga/.conda/envs/env_isaaclab/bin/python}
SCR=${2:?scratch dir}
LIST=${3:-small}
cd "$(dirname "$0")"
R=../results/determinism_metrics_dw_projection/det_training
mkdir -p "$R"
pair() {  # name, then det_training args
  local name=$1; shift
  local rec="$SCR/rec_$name"
  rm -rf "$rec"
  $PY det_training.py "$@" --records "$rec" --role write --out "$R/${name}_p1" > "$R/${name}_p1.log" 2>&1
  $PY det_training.py "$@" --records "$rec" --role compare --out "$R/${name}_p2" > "$R/${name}_p2.log" 2>&1
  tail -1 "$R/${name}_p1.log"; tail -1 "$R/${name}_p2.log"
  rm -rf "$rec"
}
if [ "$LIST" = small ]; then
  for c in S_S16_R4 S_S32_R8 S_S64_R16 M_S16_R4 M_S32_R8 M_S64_R16; do
    for s in 0 1 2; do
      pair "${c}_seed${s}_det_split" --case $c --seed $s --steps 16 --opt-mode det
    done
    pair "${c}_seed0_sr_split" --case $c --seed 0 --steps 16 --opt-mode sr
    pair "${c}_seed0_det_bf16" --case $c --seed 0 --steps 16 --opt-mode det --precision bf16
  done
  pair "M_S64_R16_seed0_fp32_split" --case M_S64_R16 --seed 0 --steps 16 --opt-mode fp32
  pair "M_S16_R4_seed1_sr_bf16" --case M_S16_R4 --seed 1 --steps 16 --opt-mode sr --precision bf16
  # an explicit private fast-tile override in deterministic mode (API precedence), several windows
  pair "M_S32_R8_seed0_det_split_cap64M" --case M_S32_R8 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=67108864,cuda_grouped_dx_tokens=64,cuda_grouped_dx_scatter=baseline
elif [ "$LIST" = production ]; then
  # per S / R: deterministic repeatability (two processes), then from the same construction the fast
  # mode against the deterministic records (order deviation) and against itself (run-to-run)
  for c in P_S16_R4 P_S32_R8 P_S64_R16; do
    name="${c}_seed0_det_split"
    rec="$SCR/rec_$name"
    rm -rf "$rec" "$SCR/rec_fast"
    $PY det_training.py --case $c --seed 0 --steps 4 --records "$rec" --role write --out "$R/${name}_p1" > "$R/${name}_p1.log" 2>&1
    $PY det_training.py --case $c --seed 0 --steps 4 --records "$rec" --role compare --out "$R/${name}_p2" > "$R/${name}_p2.log" 2>&1
    $PY det_training.py --case $c --seed 0 --steps 4 --reduction fast --warn-only --characterize --records "$rec" \
      --role compare --out "$R/${c}_seed0_fast_vs_det" > "$R/${c}_seed0_fast_vs_det.log" 2>&1
    $PY det_training.py --case $c --seed 0 --steps 4 --reduction fast --warn-only --characterize --records "$SCR/rec_fast" \
      --role write --out "$R/${c}_seed0_fast_repeat" > "$R/${c}_seed0_fast_repeat.log" 2>&1
    for f in "${name}_p1" "${name}_p2" "${c}_seed0_fast_vs_det" "${c}_seed0_fast_repeat"; do tail -1 "$R/$f.log"; done
    rm -rf "$rec" "$SCR/rec_fast"
  done
  pair "P_S64_R16_seed1_det_split" --case P_S64_R16 --seed 1 --steps 4 --opt-mode det
  pair "P_S64_R16_seed0_det_bf16" --case P_S64_R16 --seed 0 --steps 4 --opt-mode det --precision bf16
elif [ "$LIST" = windows ]; then
  # several token windows (small workspace cap) and an explicit private fast-tile override
  pair "M_S32_R8_seed0_det_split_cap4M" --case M_S32_R8 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=4194304
  pair "M_S64_R16_seed0_det_split_cap4M_T16" --case M_S64_R16 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=4194304,cuda_grouped_dx_tokens=16,cuda_grouped_dx_scatter=baseline
elif [ "$LIST" = after_dw ]; then  # representative composed training after the dW projection change
  for c in M_S16_R4 M_S32_R8 M_S64_R16; do
    pair "${c}_seed0_det_split_compact" --case $c --seed 0 --steps 16 --opt-mode det --plan cuda_grouped_dw_projection=compact
  done
  pair "M_S64_R16_seed0_det_bf16_compact" --case M_S64_R16 --seed 0 --steps 16 --opt-mode det --precision bf16 \
    --plan cuda_grouped_dw_projection=compact
  pair "M_S16_R4_seed1_sr_split_compact" --case M_S16_R4 --seed 1 --steps 16 --opt-mode sr --plan cuda_grouped_dw_projection=compact
  pair "P_S64_R16_seed0_det_split_compact" --case P_S64_R16 --seed 0 --steps 4 --opt-mode det \
    --plan cuda_grouped_dw_projection=compact
fi
echo det-training-$LIST-done
