#!/usr/bin/env bash
# NVIDIA Compute Sanitizer (memcheck, racecheck, initcheck, synccheck) over small workloads that
# touch the native kernels.  Separate from any timing run.
#   bash sparse_pooling/benchmarks/run_sanitizer.sh [--jobs N] [--out DIR] [--timeout SEC] <python> [workload ...]
# workloads: generic (sanitize_workload.py: every generic kernel), sg (grouped kernels),
#            dx_scatter (scatter variants), det (deterministic reductions, compact dW);
#            default: all four.  kgen (EXPERIMENTAL generalized fan-in kernels, sanitize_kgen_workload.py)
#            kgx (EXPERIMENTAL K = 64 kernels + L1AdamW, sanitize_kgx_workload.py) and kgw (width-preserving
#            16 x 8 / backbone + tile kernels, sanitize_kgw_workload.py) and rsn (indexed local supports of the
#            random-support study, sanitize_rsn_workload.py) run only when named.
# --jobs N      (workload, tool) runs in parallel (default: CPUs - 1).  Each instrumented process is
#               bound by one CPU core and uses little GPU memory.
# --out DIR     keep the per-run logs and summary.tsv there (default: a new directory under $TMPDIR;
#               logs are never deleted).  Each run's log is truncated before it starts, so output of
#               an earlier run can never satisfy a later check.
# --timeout SEC per-run wall-clock limit (default 14400; 0 = none).  A timed-out run fails.
# COMPUTE_SANITIZER  the sanitizer executable (default: compute-sanitizer on PATH).
# GRANULE_SANITIZER_WORKLOADS  directory of the workload scripts (default: this directory; tests
#               point it at minimal fake workloads).
#
# A run PASSES only if ALL of these hold (anything else, including a missing log, FAILS):
#   * the sanitizer process exited 0 (it runs with --error-exitcode, so detected errors exit
#     nonzero; the workload's own exit status, a signal or a timeout also arrive here);
#   * its log has exactly one completed summary of THAT tool, and every count in it is 0:
#       memcheck / initcheck / synccheck  "========= ERROR SUMMARY: 0 errors"
#       racecheck  "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
#     Racecheck WARNINGS fail as well: no hazard class is accepted silently.  The single
#     documented exclusion is the Triton bwd_dense_kernel below, which racecheck does not check;
#   * no other sanitizer diagnostic line ("========= ..." other than the banner and that summary)
#     and no Python traceback;
#   * the workload printed its completion marker "WORKLOAD COMPLETE <workload>".
# Exit status: 0 every run passed; 1 at least one run failed; 2 usage / setup error (e.g. no
# sanitizer); 129 / 130 / 143 interrupted (every child process group is terminated first).
set -uo pipefail

JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
OUT=""
TIMEOUT=14400
usage() { sed -n '4p' "$0" | sed 's/^# *//' >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --jobs|--out|--timeout)
      [ $# -ge 2 ] || usage
      case $1 in --jobs) JOBS=$2 ;; --out) OUT=$2 ;; --timeout) TIMEOUT=$2 ;; esac
      shift 2 ;;
    --help|-h) usage ;;
    --*) echo "unknown option $1" >&2; usage ;;
    *) break ;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs: positive integer expected, got '$JOBS'" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "--timeout: seconds expected, got '$TIMEOUT'" >&2; exit 2; }
PY=${1:-}
[ -n "$PY" ] || usage
shift
WL=("$@")
[ ${#WL[@]} -eq 0 ] && WL=(generic sg dx_scatter det)
HERE=$(cd "$(dirname "$0")" && pwd)
WDIR=${GRANULE_SANITIZER_WORKLOADS:-$HERE}
SAN=${COMPUTE_SANITIZER:-compute-sanitizer}
if ! SAN_PATH=$(command -v "$SAN"); then
  echo "sanitizer not found: '$SAN' (install the CUDA toolkit's compute-sanitizer or set COMPUTE_SANITIZER)" >&2
  exit 2
fi
command -v timeout > /dev/null || { echo "coreutils 'timeout' not found" >&2; exit 2; }
TOOLS=(memcheck racecheck initcheck synccheck)
ERR_EXIT=97  # --error-exitcode: sanitizer-detected errors
script_of() {
  case $1 in
    generic) echo sanitize_workload.py ;;
    sg|dx_scatter|det|kgen|kgx|kgw|rsn) echo "sanitize_$1_workload.py" ;;
    *) return 1 ;;
  esac
}
for wl in "${WL[@]}"; do
  script=$(script_of "$wl") || { echo "unknown workload '$wl' (generic sg dx_scatter det kgen kgx kgw rsn)" >&2; exit 2; }
  [ -f "$WDIR/$script" ] || { echo "workload script missing: $WDIR/$script" >&2; exit 2; }
done
if [ -z "$OUT" ]; then
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/granule-sanitizer.XXXXXX") || exit 2
fi
mkdir -p "$OUT" || exit 2
SUMMARY=$OUT/summary.tsv
rm -f "$SUMMARY"
export CUDA_MODULE_LOADING=LAZY  # instrument only the kernels that run (the libraries hold ~2000)
echo "sanitizer: $SAN_PATH ($("$SAN_PATH" --version 2>/dev/null | tail -1)); logs: $OUT; jobs $JOBS; timeout ${TIMEOUT}s"

declare -A PID_KEY=()  # running job pid -> "<workload>.<tool>"
declare -A RC=()       # "<workload>.<tool>" -> exit status

launch() {  # workload, tool: the job runs in its own process group (timeout is the group leader)
  local wl=$1 tool=$2 extra=() log kill_after=()
  # racecheck skips the Triton bwd_dense_kernel (triton_fused.py:441): its tl.sum reduction writes
  # replicated values from several lanes, reported as ~37k WAW hazards identically at 2b9a84d
  # (documented since MEMORY_TRAFFIC / BACKWARD_COMPUTE).  memcheck / initcheck / synccheck cover it.
  [ "$tool" = racecheck ] && extra=(--racecheck-report all --kernel-regex-exclude kns=bwd_dense_kernel)
  log=$OUT/$wl.$tool.log
  : > "$log"
  [ "$TIMEOUT" -gt 0 ] && kill_after=(--kill-after=30)
  timeout "${kill_after[@]}" "$TIMEOUT" "$SAN_PATH" --tool "$tool" "${extra[@]}" --error-exitcode "$ERR_EXIT" \
    --target-processes all --print-limit 20 "$PY" "$WDIR/$(script_of "$wl")" > "$log" 2>&1 < /dev/null &
  PID_KEY[$!]="$wl.$tool"
}

reap() {  # collect the exit status of every launched job that has finished
  local pid running
  running=" $(jobs -rp | tr '\n' ' ') "
  for pid in "${!PID_KEY[@]}"; do
    if [[ "$running" != *" $pid "* ]]; then
      wait "$pid"
      RC[${PID_KEY[$pid]}]=$?
      unset "PID_KEY[$pid]"
    fi
  done
}

on_signal() {  # terminate every running job's process group, then exit nonzero
  local sig=$1 code=$2 pid
  trap '' INT TERM HUP
  echo "interrupted (SIG$sig): terminating ${#PID_KEY[@]} running sanitizer job(s)" >&2
  for pid in "${!PID_KEY[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -z "$(jobs -rp)" ] && break
    sleep 0.5
  done
  for pid in "${!PID_KEY[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; done
  wait 2>/dev/null
  echo "sanitizer: INTERRUPTED; logs kept in $OUT" >&2
  exit "$code"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

for wl in "${WL[@]}"; do
  for tool in "${TOOLS[@]}"; do
    while [ ${#PID_KEY[@]} -ge "$JOBS" ]; do
      reap
      [ ${#PID_KEY[@]} -ge "$JOBS" ] && sleep 1
    done
    launch "$wl" "$tool"
  done
done
while [ ${#PID_KEY[@]} -gt 0 ]; do
  reap
  [ ${#PID_KEY[@]} -gt 0 ] && sleep 1
done

check() {  # workload, tool, exit status, log -> prints "" (pass) or the failure reason
  local wl=$1 tool=$2 rc=$3 log=$4 n bad pat other zero
  if [ ! -s "$log" ]; then echo "empty or missing log"; return; fi
  case $rc in
    0) ;;
    "$ERR_EXIT") echo "sanitizer reported errors (exit $rc)"; return ;;
    124|137) echo "timed out after ${TIMEOUT}s (exit $rc)"; return ;;
    126|127) echo "could not start the sanitizer / workload (exit $rc)"; return ;;
    *) if [ "$rc" -gt 128 ] && [ "$rc" -le 192 ]; then echo "terminated by signal $((rc - 128)) (exit $rc)"
       else echo "nonzero exit $rc"; fi
       return ;;
  esac
  if [ "$tool" = racecheck ]; then
    pat='^========= RACECHECK SUMMARY: [0-9]+ hazards? displayed \([0-9]+ errors?, [0-9]+ warnings?\)$'
    zero='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
    other='^========= ERROR SUMMARY:'
  else
    pat='^========= ERROR SUMMARY: [0-9]+ errors?$'
    zero='========= ERROR SUMMARY: 0 errors'
    other='^========= RACECHECK SUMMARY:'
  fi
  n=$(grep -cE "$pat" "$log")
  if [ "$n" -ne 1 ]; then echo "expected one completed $tool summary, found $n"; return; fi
  if grep -qE "$other" "$log"; then echo "summary of another tool in the log"; return; fi
  if ! grep -qxF "$zero" "$log"; then echo "$(grep -E "$pat" "$log" | sed 's/^========= //')"; return; fi
  bad=$(grep -E '^=========' "$log" | grep -vxF -e '========= COMPUTE-SANITIZER' -e "$zero" | head -1)
  if [ -n "$bad" ]; then echo "sanitizer diagnostic: ${bad:0:160}"; return; fi
  if grep -q 'Traceback (most recent call last)' "$log"; then echo "Python traceback"; return; fi
  if ! grep -qx "WORKLOAD COMPLETE $wl" "$log"; then echo "workload did not complete (no 'WORKLOAD COMPLETE $wl')"; return; fi
  echo ""
}

status=0
npass=0
nrun=0
printf 'workload\ttool\tresult\texit\treason\tlog\n' > "$SUMMARY"
for wl in "${WL[@]}"; do
  for tool in "${TOOLS[@]}"; do
    key=$wl.$tool
    log=$OUT/$key.log
    rc=${RC[$key]:-255}
    reason=$(check "$wl" "$tool" "$rc" "$log")
    nrun=$((nrun + 1))
    if [ -z "$reason" ]; then
      result=PASS
      npass=$((npass + 1))
    else
      result=FAIL
      status=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$wl" "$tool" "$result" "$rc" "$reason" "$log" >> "$SUMMARY"
    printf '%-4s %-10s %-9s exit=%-3s %s%s\n' "$result" "$wl" "$tool" "$rc" "$log" "${reason:+  ($reason)}"
    if [ "$result" = FAIL ]; then
      grep -E '^=========|Traceback|Error' "$log" 2>/dev/null | grep -v 'COMPUTE-SANITIZER' | head -5 | sed 's/^/       | /'
    fi
  done
done
echo "sanitizer: $npass/$nrun runs passed; summary $SUMMARY"
exit $status
