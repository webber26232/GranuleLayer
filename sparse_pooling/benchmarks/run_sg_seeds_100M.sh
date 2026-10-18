#!/bin/bash
# H = 100M: seeds for the preferred plans, BF16 gradients, matched-connectivity controls, S = 8.
PY=$1; OUT=input_supergroups/final_100M
G=cuda_input_supergroup_backend
run() {  # C S R seed prec rounds extra... out
  local C=$1 S=$2 R=$3 SEED=$4 PREC=$5 ROUNDS=$6 TAG=$7; shift 7
  $PY bench_supergroups.py --hidden 100000000 --C $C --S $S --R $R --graph-seed $SEED --input-seed $SEED --init-seed $SEED \
    --grad-precision $PREC --reps 2 --iters 2 --rounds $ROUNDS --validate "$@" \
    --variant generic:$G=generic --variant auto:$G=auto --out $OUT/H100M_C${C}_S${S}_R${R}_seed${SEED}_${PREC}${TAG}
}
for SEED in 0 1 2; do
  run 64 16 4 $SEED split 4 _auto2
  run 8 16 4 $SEED split 4 _auto2
done
run 64 16 4 0 bf16 2 ""
run 8 16 4 0 bf16 2 ""
run 64 16 4 0 split 4 _matchedC8 --matched-c8
run 64 64 16 0 split 4 _matchedC8 --matched-c8
run 8 8 2 0 split 2 "_diag"
run 64 8 2 0 split 2 "_diag"
