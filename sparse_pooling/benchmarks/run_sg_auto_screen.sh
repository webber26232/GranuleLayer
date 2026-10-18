#!/bin/bash
# generic vs measured auto (hybrid) plan on the same graph, for a list of "S R" points
# usage: run_sg_auto_screen.sh PY H M OUTDIR "S R" ...
PY=$1; H=$2; M=$3; OUT=$4; shift 4
POINTS=("$@")
for C in ${CS:-64 8}; do
  for SR in "${POINTS[@]}"; do
    set -- $SR
    $PY bench_supergroups.py --hidden $H --batch $M --C $C --S $1 --R $2 --reps 3 --iters 3 --validate \
      --variant generic:cuda_input_supergroup_backend=generic --variant auto:cuda_input_supergroup_backend=auto \
      --out $OUT/H${H}_M${M}_C${C}_S$1_R$2
  done
done
