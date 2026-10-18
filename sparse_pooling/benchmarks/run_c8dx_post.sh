#!/bin/bash
# Post-acceptance evidence: full-scale ablation, training validation of the selected variants,
# fixed-dY numerics, memory re-check with the chunked fingerprint (d97b69e), full test suite.
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
R=../results/c8_dx_scatter
./run_c8dx_ablation_100M.sh
$PY numerics_training.py --graphs C8_S16_R4,C8_S32_R8,C8_S16_R16,C8_S64_R16_shared \
  --candidates vectorized_transpose_hoisted,vectorized_hoisted,swizzle4_capped,feature_owner \
  --out $R/training_validation_selected > $R/training_validation_selected.log 2>&1
$PY numerics_scatter.py --out $R/numerics_fixed_dy > $R/numerics_fixed_dy.log 2>&1
mkdir -p $R/memory_recheck
$PY bench_supergroups.py --hidden 100000000 --C 8 --S 16 --R 4 --rounds 2 --reps 2 --iters 2 \
  --variant baseline:cuda_input_supergroup_backend=auto,cuda_grouped_dx_scatter=baseline \
  --variant selected:cuda_input_supergroup_backend=auto --out $R/memory_recheck/C8_S16_R4 > $R/memory_recheck/C8_S16_R4.log 2>&1
cd ../tests && $PY -m pytest -q -p no:cacheprovider . > ../results/c8_dx_scatter/test_suite.log 2>&1
echo "post done"
