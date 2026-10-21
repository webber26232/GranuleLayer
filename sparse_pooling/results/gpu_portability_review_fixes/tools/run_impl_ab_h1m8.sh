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
HERE=/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh/src/sparse_pooling/benchmarks
mkdir -p "$R"
FAST="--variant fast:reduction=fast"
DET="--variant det:reduction=deterministic"
# name | rounds override | bench_supergroups args
CONFIGS=("H1M_C8_S64_R16|8|--hidden 1000000 --batch 8192 --C 8 --S 64 --R 16 $FAST")
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
