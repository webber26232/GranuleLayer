#!/bin/bash
# Reruns of the stage-2 shapes that overlapped a concurrent sanitizer process (NOTE_concurrent_sanitizer.txt)
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/crossover_stage2
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
for H in 20000000 40000000 70000000; do
  $PY bench_supergroups.py --C 8 --S 64 --rounds 4 --reps 2 --iters 3 --hidden $H --R 16 --variant $A --variant $B \
    --out $OUT/H${H}_R16_rerun 2>&1 | tee $OUT/H${H}_R16_rerun.log
done
