#!/usr/bin/env bash
# Track C attribution (results/DETERMINISM_METRICS_DW_PROJECTION.md): where the compact C = 8 dW
# projection loses time.  H = 100M, M = 8192, split (and bf16 at S = 64), fast reduction:
#   padded (fast epilogue) | padded_mask (mask epilogue of the padded path: the mask-operand
#   epilogue without the compact projection) | compact;
# and a SP_ABLATE_DWC = 1 build (compact WITHOUT the two movmatrix transposes: wrong dW, timing
# only) with padded | compact_nomovm.  4 alternating rounds x 2 restored repetitions x 3 steps.
# Single idle GPU, nothing else running.
#   bash run_dm_attribution.sh <python>
set -euo pipefail
PY=${1:-/home/evga/.conda/envs/env_isaaclab/bin/python}
cd "$(dirname "$0")"
R=../results/determinism_metrics_dw_projection/attribution
mkdir -p "$R"
run() {  # name, bench args...
  local name=$1; shift
  $PY bench_supergroups.py --C 8 --rounds 4 --reps 2 --iters 3 --hidden 100000000 --batch 8192 "$@" \
    --out "$R/$name" > "$R/$name.log" 2>&1
  grep "^\[" "$R/$name.log"
}
V="--variant padded:reduction=fast,cuda_grouped_dw_projection=padded
   --variant padded_mask:reduction=fast,cuda_grouped_dw_projection=padded,cuda_dw_epilogue=mask
   --variant compact:reduction=fast,cuda_grouped_dw_projection=compact"
VA="--variant padded:reduction=fast,cuda_grouped_dw_projection=padded
    --variant compact_nomovm:reduction=fast,cuda_grouped_dw_projection=compact"
for SR in "16 4" "64 16"; do
  set -- $SR
  run "S$1_R$2_epilogue" --S $1 --R $2 $V
  SPARSE_POOL_NVCC_EXTRA="-DSP_ABLATE_DWC=1" run "S$1_R$2_ablate_movm" --S $1 --R $2 $VA
done
run "S64_R16_bf16_epilogue" --S 64 --R 16 --grad-precision bf16 $V
SPARSE_POOL_NVCC_EXTRA="-DSP_ABLATE_DWC=1" run "S64_R16_bf16_ablate_movm" --S 64 --R 16 --grad-precision bf16 $VA
echo attribution-done
