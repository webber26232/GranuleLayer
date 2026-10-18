#!/usr/bin/env bash
# Phase-3 configuration regression matrix: baseline (backward-compute-optimization worktree)
# vs this branch (defaults), ROUNDS interleaved rounds per configuration.  Mean(identity)
# is not selectable in the baseline benchmark; its control is this branch with
# --fwd-epilogue baseline (same kernels as the baseline commit: dW/dX are unchanged).
#   bash run_matrix_phase3.sh <python> <baseline-worktree> [rounds]
set -uo pipefail
PY=$1; WT=$2; ROUNDS=${3:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIGS=(  # H C M pooling precision
  "1000000 64 8192 mean split" "10000000 64 8192 mean split" "100000000 64 8192 mean split"
  "100000000 64 8192 mean bf16" "100000000 64 1024 mean split" "100000000 8 8192 mean split"
  "100000000 64 8192 mean_id split" "100000000 64 8192 lme split" "100000000 64 8192 sw split"
  "100000000 64 8192 max split")
for cfg in "${CONFIGS[@]}"; do
  read -r H C M P PR <<< "$cfg"
  base="baseline|$WT||"
  [ "$P" = mean_id ] && base="fwd_base|.||--fwd-epilogue baseline"
  HID=$H C=$C M=$M POOL=$P PREC=$PR bash "$HERE/run_ab_phase3_final.sh" "$PY" matrix "$ROUNDS" "$base" "new|.||"
done
