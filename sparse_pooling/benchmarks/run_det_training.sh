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
  for c in P_S16_R4 P_S32_R8 P_S64_R16; do
    pair "${c}_seed0_det_split" --case $c --seed 0 --steps 4 --opt-mode det
  done
  pair "P_S64_R16_seed1_det_split" --case P_S64_R16 --seed 1 --steps 4 --opt-mode det
  pair "P_S64_R16_seed0_det_bf16" --case P_S64_R16 --seed 0 --steps 4 --opt-mode det --precision bf16
fi
echo det-training-$LIST-done
