#!/usr/bin/env bash
# Old-versus-new deterministic training regression: the REFERENCE implementation (another checkout,
# GRANULE_IMPL_ROOT) writes the records of each case in one process, the implementation of THIS
# checkout compares both of its trajectories against them in an independent process (strict:
# bitwise outputs, gradients, optimizer state; same resolved plan).  Same harness for both sides.
# Also: within-profile repeatability of the generic profile (this checkout, two processes).
#
#   bash run_det_regression.sh <python> <reference checkout> <scratch dir> <output dir> [small|production|all]
set -uo pipefail
PY=${1:?python}
REF=${2:?reference checkout}
SCR=${3:?scratch dir}
R=${4:?output dir}
LIST=${5:-all}
HERE=/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh/src/sparse_pooling/benchmarks
NEW=$(cd "$HERE/../.." && pwd)
mkdir -p "$R" "$SCR"
FAILED=()
SIDE_ENV=""  # extra environment of the current side (e.g. SPARSE_POOL_PROFILE=generic_cuda)
side() {  # implementation root, name, args...
  local root=$1 name=$2; shift 2
  env $SIDE_ENV GRANULE_IMPL_ROOT="$root" "$PY" "$HERE/det_training.py" "$@" --out "$R/$name" > "$R/$name.log" 2>&1
  local rc=$?
  echo "[$rc] $name: $(tail -1 "$R/$name.log")"
  [ $rc -eq 0 ] || FAILED+=("$name (exit $rc)")
}
oldnew() {  # name, det_training args
  local name=$1; shift
  case " ${SKIP:-} " in *" $name "*) SIDE_ENV=""; echo "[skip] $name (passed before the oomd kill)"; return;; esac
  local rec="$SCR/rec_$name"
  rm -rf "$rec"
  side "$REF" "${name}_ref_write" "$@" --records "$rec" --role write
  [ -f "$rec/manifest.json" ] && side "$NEW" "${name}_new_compare" "$@" --records "$rec" --role compare
  rm -rf "$rec"
}
newnew() {  # name, environment, det_training args (within-profile repeatability of this checkout)
  local name=$1; SIDE_ENV=$2; shift 2
  case " ${SKIP:-} " in *" $name "*) SIDE_ENV=""; echo "[skip] $name (passed before the oomd kill)"; return;; esac
  local rec="$SCR/rec_$name"
  rm -rf "$rec"
  side "$NEW" "${name}_p1" "$@" --records "$rec" --role write
  [ -f "$rec/manifest.json" ] && side "$NEW" "${name}_p2" "$@" --records "$rec" --role compare
  SIDE_ENV=""
  rm -rf "$rec"
}
if [ "$LIST" = small ] || [ "$LIST" = all ]; then
  oldnew S_S16_R4_det_split --case S_S16_R4 --seed 0 --steps 16 --opt-mode det
  oldnew S_S64_R16_det_split --case S_S64_R16 --seed 1 --steps 16 --opt-mode det
  oldnew M_S16_R4_sr_split --case M_S16_R4 --seed 0 --steps 16 --opt-mode sr
  oldnew M_S32_R8_det_split_cap64M --case M_S32_R8 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=67108864,cuda_grouped_dx_tokens=64,cuda_grouped_dx_scatter=baseline
  oldnew M_S64_R16_det_bf16 --case M_S64_R16 --seed 0 --steps 16 --opt-mode det --precision bf16
  oldnew M_S64_R16_fp32_split --case M_S64_R16 --seed 0 --steps 16 --opt-mode fp32
  oldnew M64_S16_R4_det_split --case M64_S16_R4 --seed 0 --steps 16 --opt-mode det
  oldnew M64_S64_R16_det_bf16 --case M64_S64_R16 --seed 0 --steps 16 --opt-mode det --precision bf16
  newnew G_M_S64_R16_det_split "SPARSE_POOL_PROFILE=generic_cuda" --case M_S64_R16 --seed 0 --steps 16 --opt-mode det
  newnew G_M64_S16_R4_sr_split "SPARSE_POOL_PROFILE=generic_cuda" --case M64_S16_R4 --seed 0 --steps 16 --opt-mode sr
fi
if [ "$LIST" = production ] || [ "$LIST" = all ]; then
  for c in P_S16_R4 P_S32_R8 P_S64_R16; do
    oldnew "${c}_det_split" --case $c --seed 0 --steps 4 --opt-mode det
  done
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "det-regression-$LIST: FAILED (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "det-regression-$LIST: all strict comparisons passed"
