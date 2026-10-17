#!/usr/bin/env bash
# Nsight Compute profiles of the hot kernels.
#
# GPU performance counters are restricted to administrators by default
# (ERR_NVGPUCTRPERM).  Either run this script with sudo, or allow user access:
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvprof.conf
#   (then reboot / reload the nvidia module)
#
# Usage:  sudo -E bash sparse_pooling/benchmarks/profile_ncu.sh [python] [hidden] [batch]
set -euo pipefail
PY=${1:-python}
H=${2:-10000000}
B=${3:-8192}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../results/ncu
mkdir -p "$OUT"

SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
 --section ComputeWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats \
 --section SchedulerStats --section SourceCounters --section InstructionStats"
# Tensor-core / DRAM / L2 / smem metrics answering the questions in the report
METRICS="--metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,smsp__inst_executed_op_shared_atom.sum,\
smsp__sass_inst_executed_op_global_red.sum,launch__registers_per_thread,launch__occupancy_limit_registers"

for case in "max 8" "logmeanexp 8" "mean 8" "logmeanexp 64" "topk_mean 8"; do
  set -- $case
  pool=$1; C=$2
  echo "== $pool C=$C"
  ncu --target-processes all --kernel-name regex:'fwd_mma_kernel|k_dx|k_dw|fwd_pool_kernel|bwd_events_kernel' \
      --launch-skip 0 --launch-count 6 $SECTIONS $METRICS \
      -o "$OUT/${pool}_C${C}_H${H}" -f \
      "$PY" "$HERE/profile_workload.py" --pool "$pool" --C "$C" --hidden "$H" --batch "$B"
done
echo "reports in $OUT (open with ncu-ui, or: ncu --import <file>.ncu-rep --page details)"
