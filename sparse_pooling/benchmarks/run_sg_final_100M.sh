#!/bin/bash
# H = 100M, M = 8192 interleaved attribution (4 rounds, alternating order) on the same graph:
# generic / fwd-only / dX-only / dW-only / all grouped / auto (best validated hybrid), plus the
# unrestricted architecture reference (separate process, before and after).
# usage: run_sg_final_100M.sh PY OUTDIR [graph_seed] [precision] "C S R" ...
PY=$1; OUT=$2; SEED=$3; PREC=$4; shift 4
POINTS=("$@")
G=cuda_input_supergroup_backend
for C in $(for p in "${POINTS[@]}"; do set -- $p; echo $1; done | sort -u); do
  $PY bench_supergroups.py --hidden 100000000 --C $C --S 0 --grad-precision $PREC --reps 2 --iters 2 --rounds 2 \
    --out $OUT/H100M_C${C}_unrestricted_${PREC}_pre
done
for p in "${POINTS[@]}"; do
  set -- $p
  $PY bench_supergroups.py --hidden 100000000 --C $1 --S $2 --R $3 --graph-seed $SEED --input-seed $SEED --init-seed $SEED \
    --grad-precision $PREC --reps 2 --iters 2 --rounds 4 --validate \
    --variant generic:$G=generic \
    --variant fwd_only:cuda_grouped_fwd=grouped,cuda_grouped_dx=generic,cuda_grouped_dw=generic \
    --variant dx_only:cuda_grouped_fwd=generic,cuda_grouped_dx=grouped,cuda_grouped_dw=generic \
    --variant dw_only:cuda_grouped_fwd=generic,cuda_grouped_dx=generic,cuda_grouped_dw=grouped \
    --variant all_grouped:$G=grouped \
    --variant auto:$G=auto \
    --out $OUT/H100M_C$1_S$2_R$3_seed${SEED}_${PREC}
done
for C in $(for p in "${POINTS[@]}"; do set -- $p; echo $1; done | sort -u); do
  $PY bench_supergroups.py --hidden 100000000 --C $C --S 0 --grad-precision $PREC --reps 2 --iters 2 --rounds 2 \
    --out $OUT/H100M_C${C}_unrestricted_${PREC}_post
done
