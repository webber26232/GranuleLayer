#!/usr/bin/env bash
# NVIDIA Compute Sanitizer (memcheck, racecheck, initcheck, synccheck) over small workloads that
# touch the native kernels.  Separate from any timing run.
#   bash sparse_pooling/benchmarks/run_sanitizer.sh <python> [workload ...]
# workloads: generic (sanitize_workload.py: every generic kernel), sg (grouped kernels),
#            dx_scatter (scatter variants), det (deterministic reductions, compact dW);
#            default: all four.  Exit status 1 if any tool reports an error.
set -uo pipefail
PY=${1:?python interpreter}
shift
WL=("$@")
[ ${#WL[@]} -eq 0 ] && WL=(generic sg dx_scatter det)
HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_MODULE_LOADING=LAZY  # instrument only the kernels that run (the libraries hold ~2000)
status=0
for wl in "${WL[@]}"; do
  case $wl in
    generic) script=sanitize_workload.py ;;
    *) script=sanitize_${wl}_workload.py ;;
  esac
  for tool in memcheck racecheck initcheck synccheck; do
    echo "=== $wl: compute-sanitizer --tool $tool"
    extra=""
    [ "$tool" = racecheck ] && extra="--racecheck-report all"
    out=$(compute-sanitizer --tool "$tool" $extra --target-processes all --print-limit 20 "$PY" "$HERE/$script" 2>&1)
    echo "$out" | grep -E "ERROR SUMMARY|RACECHECK SUMMARY|^ok|^skip|Invalid|Race|Uninitialized|Barrier|hazard|Traceback|Error" \
      | sort | uniq -c
    if ! echo "$out" | grep -qE "(ERROR|RACECHECK) SUMMARY: 0 (errors|hazards)"; then status=1; fi
    if echo "$out" | grep -q "Traceback"; then status=1; fi
  done
done
exit $status
