#!/usr/bin/env bash
# Attribution ablations (WRONG gradients by construction; timing only), H = 100M and 10M, C = 64.
#   bash sparse_pooling/benchmarks/run_ablation.sh [python]
set -uo pipefail
PY=${1:-python}
B="$(cd "$(dirname "$0")" && pwd)/bench_kernels.py"
run() { local tag=$1 extra=$2; shift 2
  SPARSE_POOL_NVCC_EXTRA="$extra" "$PY" "$B" --C 64 --reps 3 --iters 2 --tag "$tag" "$@" 2>&1 | grep -v Warning; }
for H in 100000000 10000000; do
  run ref           ""                 --hidden $H --dw-staging full
  run ref           ""                 --hidden $H --dw-staging indexed --dw-cfg 1
  run ref           ""                 --hidden $H --dw-staging indexed --dw-cfg 3
  run dw_nostage    "-DSP_ABLATE_DW=1" --hidden $H --dw-staging full
  run dw_nostage    "-DSP_ABLATE_DW=1" --hidden $H --dw-staging indexed --dw-cfg 1
  run dw_nostage    "-DSP_ABLATE_DW=1" --hidden $H --dw-staging indexed --dw-cfg 3
  run dw_nobarrier  "-DSP_ABLATE_DW=2" --hidden $H --dw-staging full
  run dw_nocoef     "-DSP_ABLATE_DW=3" --hidden $H --dw-staging full
  run dw_nocoef     "-DSP_ABLATE_DW=3" --hidden $H --dw-staging indexed --dw-cfg 3
  run dx_racyrmw    "-DSP_ABLATE_DX=1" --hidden $H --dw-staging full
  run dx_noscatter  "-DSP_ABLATE_DX=2" --hidden $H --dw-staging full
  run dx_noflush    "-DSP_ABLATE_DX=3" --hidden $H --dw-staging full
done
