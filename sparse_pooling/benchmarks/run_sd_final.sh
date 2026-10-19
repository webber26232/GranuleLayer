#!/bin/bash
# Final same-session attribution (clean committed sources), H = 100M, M = 8192, D = G = 1024,
# complete-state harness, 4 alternating rounds x 2 restored repetitions x 3 steps.
#  * S = 64 dispatch: Plan A (previous automatic choice), Plan B, the new automatic rule; three
#    graph / input / init seeds; the original grouped scatter as a context row (seed 0);
#  * S = 64 BF16: previous BF16 default (per-field tiles, original scatter), A, B, new auto;
#  * S = 16 / 32 fusion: the selected separate backward vs the fused prototype (v1 cfg 3, v2 cfg 6,
#    v2 cfg 7) and the recompute-preserving control of cfg 7 (same schedule, two projections);
#  * C = 64 S = 16 / 64 unchanged controls; the generic backend (existing fallback) at C = 8, S = 16.
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/evga/.conda/envs/env_isaaclab/bin/python}
OUT=../results/scale_dispatch_fused_backward/final_100M
mkdir -p $OUT
A=planA:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=shared_t64
B=planB:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=private_t32
U=auto:cuda_input_supergroup_backend=auto
O=orig_scatter:cuda_input_supergroup_backend=auto,cuda_grouped_dx_tokens=64,cuda_grouped_dx_warps=8,cuda_grouped_dx_reduction=8,cuda_grouped_dx_scatter=baseline
T=prev_bf16_default:cuda_input_supergroup_backend=auto,cuda_grouped_dx_plan=tiles
F() { echo "$1:cuda_input_supergroup_backend=auto,cuda_grouped_backward=$2,cuda_grouped_fused_cfg=$3"; }
run() { local name=$1; shift; $PY bench_supergroups.py --hidden 100000000 --rounds 4 --reps 2 --iters 3 "$@" --out $OUT/$name 2>&1 | tee $OUT/$name.log; }
run C8_S64_R16_seed0 --C 8 --S 64 --R 16 --variant $A --variant $B --variant $U --variant $O
for s in 1 2; do
  run C8_S64_R16_seed$s --C 8 --S 64 --R 16 --graph-seed $s --init-seed $s --input-seed $s --variant $A --variant $B --variant $U
done
run C8_S64_R16_bf16 --C 8 --S 64 --R 16 --grad-precision bf16 --variant $T --variant $A --variant $B --variant $U
run C8_S16_R4_fusion --C 8 --S 16 --R 4 --variant separate:cuda_input_supergroup_backend=auto --variant "$(F fused_v1c3 fused 3)" \
  --variant "$(F fused_v2c6 fused 6)" --variant "$(F fused_v2c7 fused 7)" --variant "$(F recompute_v2c7 fused_recompute 7)"
run C8_S32_R8_fusion --C 8 --S 32 --R 8 --variant separate:cuda_input_supergroup_backend=auto --variant "$(F fused_v1c3 fused 3)" \
  --variant "$(F fused_v2c6 fused 6)" --variant "$(F fused_v2c7 fused 7)" --variant "$(F recompute_v2c7 fused_recompute 7)"
run C64_S16_R4 --C 64 --S 16 --R 4 --variant $U
run C64_S64_R16 --C 64 --S 64 --R 16 --variant $U
run C8_S16_R4_generic --C 8 --S 16 --R 4 --variant separate:cuda_input_supergroup_backend=auto --variant generic:cuda_input_supergroup_backend=generic
