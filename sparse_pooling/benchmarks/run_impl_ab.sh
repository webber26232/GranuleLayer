#!/usr/bin/env bash
# Performance A/B of two implementations (checkouts) with the SAME harness (bench_supergroups.py of
# this checkout; GRANULE_IMPL_ROOT selects the implementation): complete training steps, identical
# graphs / initial state / inputs (same seeds), restored training state per repetition
# (--reps 2 --iters 3), fresh process per (implementation, round), rounds alternating A B / B A.
# The GPU must be idle (no other benchmark, test or sanitizer).  Summary: summarize_impl_ab.py.
#
#   bash run_impl_ab.sh <python> <checkout A (reference)> <checkout B (candidate)> <output dir> [rounds] [set]
#   set: production | scale | all (default all)
set -uo pipefail
PY=${1:?python}
A=${2:?checkout A}
B=${3:?checkout B}
R=${4:?output dir}
ROUNDS=${5:-4}
SET=${6:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$R"
FAST="--variant fast:reduction=fast"
DET="--variant det:reduction=deterministic"
# name | rounds override | bench_supergroups args
CONFIGS=()
if [ "$SET" = production ] || [ "$SET" = all ]; then
  CONFIGS+=(
    "C8_S16_R4|$ROUNDS|--hidden 100000000 --batch 8192 --C 8 --S 16 --R 4 $FAST $DET"
    "C8_S32_R8|$ROUNDS|--hidden 100000000 --batch 8192 --C 8 --S 32 --R 8 $FAST $DET"
    "C8_S64_R16|$ROUNDS|--hidden 100000000 --batch 8192 --C 8 --S 64 --R 16 $FAST $DET"
    "C8_S64_R16_bf16|$ROUNDS|--hidden 100000000 --batch 8192 --C 8 --S 64 --R 16 --grad-precision bf16 $FAST"
    "C64_S16_R4|$ROUNDS|--hidden 100000000 --batch 8192 --C 64 --S 16 --R 4 $FAST $DET"
    "C64_S64_R16|$ROUNDS|--hidden 100000000 --batch 8192 --C 64 --S 64 --R 16 $FAST"
  )
fi
if [ "$SET" = scale ] || [ "$SET" = all ]; then
  CONFIGS+=(
    "H1M_C8_S64_R16|2|--hidden 1000000 --batch 8192 --C 8 --S 64 --R 16 $FAST"
    "H10M_C8_S64_R16|2|--hidden 10000000 --batch 8192 --C 8 --S 64 --R 16 $FAST"
    "H10M_C8_S16_R4_M1000|2|--hidden 10000000 --batch 1000 --C 8 --S 16 --R 4 $FAST $DET"
    "H1M_C8_S64_R16_M1000|2|--hidden 1000000 --batch 1000 --C 8 --S 64 --R 16 $FAST"
  )
fi
run() {  # impl label, root, config name, round, args
  local label=$1 root=$2 name=$3 r=$4; shift 4
  GRANULE_IMPL_ROOT="$root" "$PY" "$HERE/bench_supergroups.py" "$@" --rounds 1 --reps 2 --iters 3 \
    --out "$R/${name}_r${r}_${label}" > "$R/${name}_r${r}_${label}.log" 2>&1 || echo "FAILED $name r$r $label"
  grep "^\[" "$R/${name}_r${r}_${label}.log" | sed "s/^/  $label r$r /"
}
for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r name n args <<< "$spec"
  echo "== $name ($n rounds)"
  for ((r = 0; r < n; r++)); do
    if ((r % 2 == 0)); then
      run A "$A" "$name" "$r" $args; run B "$B" "$name" "$r" $args
    else
      run B "$B" "$name" "$r" $args; run A "$A" "$name" "$r" $args
    fi
  done
done
echo impl-ab-done
