#!/bin/bash
# Family A screen (fixed Q = 256 where defined) + diagnostics, one seed, both C.
# usage: run_sg_family_a.sh PY H M [tag]
PY=$1; H=$2; M=$3; TAG=${4:-a}
OUT=input_supergroups/screen_${TAG}
G="cuda_input_supergroup_backend=grouped"
VARS=(--variant generic:cuda_input_supergroup_backend=generic
      --variant gAuto:$G
      --variant gB:$G,cuda_grouped_fwd_tokens=64,cuda_grouped_fwd_warps=8,cuda_grouped_dx_tokens=128,cuda_grouped_dx_warps=16,cuda_grouped_dx_reduction=8,cuda_grouped_dw_cfg=0
      --variant gC:$G,cuda_grouped_fwd_tokens=128,cuda_grouped_fwd_warps=8,cuda_grouped_dx_tokens=32,cuda_grouped_dx_warps=8,cuda_grouped_dx_reduction=4,cuda_grouped_dw_cfg=3
      --variant gD:$G,cuda_grouped_fwd_tokens=32,cuda_grouped_fwd_warps=8,cuda_grouped_dx_tokens=16,cuda_grouped_dx_warps=16,cuda_grouped_dx_reduction=1,cuda_grouped_dw_cfg=2)
for C in 64 8; do
  $PY bench_supergroups.py --hidden $H --batch $M --C $C --S 0 --reps 3 --iters 3 --out $OUT/H${H}_M${M}_C${C}_unrestricted
  for SR in "8 2" "16 4" "32 8" "64 16" "128 32" "256 64" "1024 1"; do
    set -- $SR
    $PY bench_supergroups.py --hidden $H --batch $M --C $C --S $1 --R $2 --reps 3 --iters 3 --validate --skip-infeasible \
      "${VARS[@]}" --out $OUT/H${H}_M${M}_C${C}_S$1_R$2
  done
done
