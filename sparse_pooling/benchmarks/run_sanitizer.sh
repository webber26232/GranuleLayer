#!/usr/bin/env bash
# NVIDIA Compute Sanitizer (memcheck, racecheck, initcheck, synccheck) over small workloads that
# touch the native kernels.  Separate from any timing run.
#   bash sparse_pooling/benchmarks/run_sanitizer.sh [--jobs N] <python> [workload ...]
# workloads: generic (sanitize_workload.py: every generic kernel), sg (grouped kernels),
#            dx_scatter (scatter variants), det (deterministic reductions, compact dW);
#            default: all four.  Exit status 1 if any tool reports an error.
# --jobs N: (workload, tool) runs in parallel (default: CPUs - 1).  Each instrumented process is
#           bound by one CPU core and uses little GPU memory; reports print in the fixed order.
set -uo pipefail
JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
if [ "${1:-}" = "--jobs" ]; then
  JOBS=${2:?--jobs needs a number}
  shift 2
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs: positive integer expected, got '$JOBS'" >&2; exit 2; }
PY=${1:?python interpreter}
shift
WL=("$@")
[ ${#WL[@]} -eq 0 ] && WL=(generic sg dx_scatter det)
HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_MODULE_LOADING=LAZY  # instrument only the kernels that run (the libraries hold ~2000)
TOOLS=(memcheck racecheck initcheck synccheck)
OUT=$(mktemp -d "${TMPDIR:-/tmp}/granule-sanitizer.XXXXXX")
trap 'rm -rf "$OUT"' EXIT

run_one() {  # workload, tool -> $OUT/<workload>.<tool>.log
  local wl=$1 tool=$2 script extra=""
  case $wl in
    generic) script=sanitize_workload.py ;;
    *) script=sanitize_${wl}_workload.py ;;
  esac
  # racecheck skips the Triton bwd_dense_kernel (triton_fused.py:441): its tl.sum reduction writes
  # replicated values from several lanes, reported as ~37k WAW hazards identically at 2b9a84d
  # (documented since MEMORY_TRAFFIC / BACKWARD_COMPUTE).  memcheck / initcheck / synccheck cover it.
  [ "$tool" = racecheck ] && extra="--racecheck-report all --kernel-regex-exclude kns=bwd_dense_kernel"
  compute-sanitizer --tool "$tool" $extra --target-processes all --print-limit 20 "$PY" "$HERE/$script" \
    > "$OUT/$wl.$tool.log" 2>&1
}

for wl in "${WL[@]}"; do
  for tool in "${TOOLS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run_one "$wl" "$tool" &
  done
done
wait

status=0
for wl in "${WL[@]}"; do
  for tool in "${TOOLS[@]}"; do
    out=$(cat "$OUT/$wl.$tool.log")
    echo "=== $wl: compute-sanitizer --tool $tool"
    echo "$out" | grep -E "ERROR SUMMARY|RACECHECK SUMMARY|^ok|^skip|Invalid|Race|Uninitialized|Barrier|hazard|Traceback|Error" \
      | sort | uniq -c
    if ! echo "$out" | grep -qE "(ERROR|RACECHECK) SUMMARY: 0 (errors|hazards)"; then status=1; fi
    if echo "$out" | grep -q "Traceback"; then status=1; fi
  done
done
exit $status
