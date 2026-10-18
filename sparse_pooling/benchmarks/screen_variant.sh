#!/usr/bin/env bash
# Stage-1 screening of a compile-time variant (SPARSE_POOL_NVCC_EXTRA): build, record
# register/stack usage and MOV attribution, then time H = 1M and 10M (C = 64, M = 8192,
# Mean(ReLU), split).  Every variant gets its own library (the flags are part of the build hash).
#   bash sparse_pooling/benchmarks/screen_variant.sh <python> <name> "<nvcc extra flags>" [hidden...]
set -euo pipefail
PY=$1; NAME=$2; EXTRA=$3; shift 3
HS=("${@:-1000000 10000000}")
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../results/mma_forward/screen"
mkdir -p "$OUT"
export SPARSE_POOL_NVCC_EXTRA="$EXTRA"
LIB=$("$PY" -c "import sys; sys.path.insert(0, '$HERE/../..'); from sparse_pooling.kernels.cuda_ops import build_library; print(build_library())")
DW='k_dw_idx<2, 1, 64, 8, 2, 2, 1, 0, 2, 1>'
FW='fwd_mma_kernel<2, 1, 64, 2, 16, 0, 0, 1>'
"$PY" "$HERE/sass_attrib.py" --lib "$LIB" --kernel "$DW" --kernel "$FW" --out "$OUT/sass_$NAME.json"
cuobjdump -res-usage "$LIB" 2>/dev/null | grep -A1 -E '_Z8k_dw_idxILi2ELi1ELi64ELi8ELi2ELi2ELb1ELi0ELi2ELi1E|_Z14fwd_mma_kernelILi2ELi1ELi64ELi2ELi16ELb0ELi0ELb1E' \
  | grep REG | tee "$OUT/res_$NAME.txt"
for H in ${HS[@]}; do
  "$PY" "$HERE/bench_kernels.py" --hidden "$H" --C 64 --batch 8192 --pooling mean --grad-precision split \
    --reps 5 --iters 3 --tag "$NAME" --out "mma_forward/screen/s1_mean_H${H}_C64_M8192_split_$NAME" 2>&1 | grep -v Warning
done
