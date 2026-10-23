#!/usr/bin/env bash
# Composed-model training repeatability (det_training.py).  Every case runs in two independent
# processes: the first writes its records (published atomically), the second compares against them.
# Records are streamed to a scratch directory and deleted after the pair.
#
#   bash run_det_training.sh <python> <scratch dir> <list> <output dir>
#     list: selftest | small | windows | production | after_dw
#
# Exit status: 0 only if every strict pair passed (characterization runs are reported, never counted
# as a pass); 1 if any strict pair failed or a selftest negative control was not detected.  A
# failing case does not stop the list; the summary at the end names every failure.
set -uo pipefail
PY=${1:?python interpreter}
SCR=${2:?scratch dir}
LIST=${3:?list}
R=${4:?output dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$R" "$SCR"
FAILED=()
run() {  # name, then det_training args; returns det_training's exit code
  local name=$1; shift
  "$PY" "$HERE/det_training.py" "$@" --out "$R/$name" > "$R/$name.log" 2>&1
  local rc=$?
  echo "[$rc] $(tail -1 "$R/$name.log")"
  return $rc
}
expect() {  # name, expected outcome (pass | fail), rc
  local name=$1 want=$2 rc=$3
  if { [ "$want" = pass ] && [ "$rc" -ne 0 ]; } || { [ "$want" = fail ] && [ "$rc" -eq 0 ]; }; then
    FAILED+=("$name (expected $want, exit $rc)")
  fi
}
pair() {  # name, then det_training args
  local name=$1; shift
  local rec="$SCR/rec_$name"
  rm -rf "$rec"
  run "${name}_p1" "$@" --records "$rec" --role write; expect "${name}_p1" pass $?
  if [ -f "$rec/manifest.json" ]; then
    run "${name}_p2" "$@" --records "$rec" --role compare; expect "${name}_p2" pass $?
  else
    FAILED+=("${name}_p2 (no published records from p1)")
  fi
  rm -rf "$rec"
}
characterize() {  # name, records dir, role, then args: reported, never counted as a pass
  local name=$1 rec=$2 role=$3; shift 3
  run "$name" "$@" --reduction fast --warn-only --characterize --records "$rec" --role "$role"
  local rc=$?
  [ $rc -eq 0 ] || FAILED+=("$name (characterization run failed structurally, exit $rc)")
}

if [ "$LIST" = selftest ]; then
  # the harness must pass a repeatable pair and detect each corruption (negative controls)
  T="--case T_S16_R4 --seed 0 --steps 3 --opt-mode det"
  rec="$SCR/rec_selftest"
  rm -rf "$rec" "$rec.bad"
  run selftest_write $T --records "$rec" --role write; expect selftest_write pass $?
  run selftest_compare $T --records "$rec" --role compare; expect selftest_compare pass $?
  run selftest_overwrite $T --records "$rec" --role write; expect selftest_overwrite fail $?
  corrupt() {  # label, python snippet editing a copy of the records (d = its Path)
    rm -rf "$rec.bad"; cp -r "$rec" "$rec.bad"
    "$PY" -c "import json, sys; from pathlib import Path; d = Path(sys.argv[1]); $2" "$rec.bad"
    run "selftest_$1" $T --records "$rec.bad" --role compare; expect "selftest_$1" fail $?
  }
  M="m = json.loads((d / 'manifest.json').read_text())"
  W="(d / 'manifest.json').write_text(json.dumps(m))"
  corrupt flipped_byte "f = d / 's2_dW.bin'; b = bytearray(f.read_bytes()); b[7] ^= 1; f.write_bytes(bytes(b))"
  corrupt truncated "f = d / 's1_W.bin'; f.write_bytes(f.read_bytes()[:-2])"
  corrupt extra_bytes "f = d / 's1_db.bin'; f.write_bytes(f.read_bytes() + bytes(4))"
  corrupt missing_file "(d / 's3_y.bin').unlink()"
  corrupt missing_key "$M; m['records'].pop('s1_dX'); (d / 's1_dX.bin').unlink(); $W"
  corrupt extra_key "$M; m['records']['s2_ghost'] = dict(m['records']['s2_y']); (d / 's2_ghost.bin').write_bytes((d / 's2_y.bin').read_bytes()); $W"
  corrupt shape "$M; m['records']['s1_y']['shape'] = [1] + m['records']['s1_y']['shape']; $W"
  corrupt dtype "$M; m['records']['s1_y']['dtype'] = 'torch.int32'; $W"
  corrupt incomplete "$M; m['complete'] = False; $W"
  corrupt bad_schema "$M; m['schema'] = 'granule.det_records/1'; $W"
  corrupt no_manifest "(d / 'manifest.json').unlink()"
  corrupt other_experiment "$M; m['identity']['seed'] = 1; $W"
  corrupt other_plan "$M; m['resolved_plan']['grouped_dx']['tokens'] = 1; $W"
  # fewer steps than the reference (identity differs) and another seed through the CLI
  run selftest_short_trajectory --case T_S16_R4 --seed 0 --steps 2 --opt-mode det --records "$rec" --role compare
  expect selftest_short_trajectory fail $?
  run selftest_other_seed --case T_S16_R4 --seed 1 --steps 3 --opt-mode det --records "$rec" --role compare
  expect selftest_other_seed fail $?
  # characterization (fast mode against the deterministic records) is reported, never a pass
  characterize selftest_characterize "$rec" compare $T --ignore-identity reduction,warn_only
  grep -q '"status": "characterization"' "$R/selftest_characterize.json" \
    || FAILED+=("selftest_characterize (status is not 'characterization')")
  rm -rf "$rec" "$rec.bad"
elif [ "$LIST" = small ]; then
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
elif [ "$LIST" = windows ]; then
  # several token windows (small workspace cap) and an explicit private fast-tile override
  pair "M_S32_R8_seed0_det_split_cap4M" --case M_S32_R8 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=4194304
  pair "M_S64_R16_seed0_det_split_cap4M_T16" --case M_S64_R16 --seed 0 --steps 16 --opt-mode det \
    --plan deterministic_workspace_bytes=4194304,cuda_grouped_dx_tokens=16,cuda_grouped_dx_scatter=baseline
elif [ "$LIST" = production ]; then
  # per S / R: deterministic repeatability (two processes), then from the same construction the fast
  # mode against the deterministic records (order deviation) and against itself (run-to-run), both
  # characterization only
  for c in P_S16_R4 P_S32_R8 P_S64_R16; do
    name="${c}_seed0_det_split"
    rec="$SCR/rec_$name"
    rm -rf "$rec" "$SCR/rec_fast"
    run "${name}_p1" --case $c --seed 0 --steps 4 --records "$rec" --role write; expect "${name}_p1" pass $?
    run "${name}_p2" --case $c --seed 0 --steps 4 --records "$rec" --role compare; expect "${name}_p2" pass $?
    characterize "${c}_seed0_fast_vs_det" "$rec" compare --case $c --seed 0 --steps 4 --ignore-identity reduction,warn_only
    characterize "${c}_seed0_fast_repeat" "$SCR/rec_fast" write --case $c --seed 0 --steps 4
    rm -rf "$rec" "$SCR/rec_fast"
  done
  pair "P_S64_R16_seed1_det_split" --case P_S64_R16 --seed 1 --steps 4 --opt-mode det
  pair "P_S64_R16_seed0_det_bf16" --case P_S64_R16 --seed 0 --steps 4 --opt-mode det --precision bf16
elif [ "$LIST" = after_dw ]; then  # representative composed training with the compact dW projection
  for c in M_S16_R4 M_S32_R8 M_S64_R16; do
    pair "${c}_seed0_det_split_compact" --case $c --seed 0 --steps 16 --opt-mode det --plan cuda_grouped_dw_projection=compact
  done
  pair "M_S64_R16_seed0_det_bf16_compact" --case M_S64_R16 --seed 0 --steps 16 --opt-mode det --precision bf16 \
    --plan cuda_grouped_dw_projection=compact
  pair "M_S16_R4_seed1_sr_split_compact" --case M_S16_R4 --seed 1 --steps 16 --opt-mode sr --plan cuda_grouped_dw_projection=compact
  pair "P_S64_R16_seed0_det_split_compact" --case P_S64_R16 --seed 0 --steps 4 --opt-mode det \
    --plan cuda_grouped_dw_projection=compact
else
  echo "unknown list $LIST" >&2
  exit 2
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "det-training-$LIST: FAILED (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "det-training-$LIST: all strict pairs passed"
