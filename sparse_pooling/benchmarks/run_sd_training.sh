#!/bin/bash
# Production R > 1 nonlinear / multi-step characterization with the fixed optimizer (null controls,
# predeclared checks C1-C3 of training_validation.py; x10 for whole-plan / fused comparisons).
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
R=../results/scale_dispatch_fused_backward
$PY numerics_training.py --graphs C8_S16_R4,C8_S64_R16,C8_S64_R16_planB_vs_planA,C8_S16_R4_fused_vs_separate \
  --candidates vectorized_transpose_hoisted,swizzle4_capped --models mse,compose --opt-modes det,sr \
  --out $R/training_validation_production > $R/training_validation_production.log 2>&1
