#!/usr/bin/env bash
# Native (unchanged-implementation) screen of the K/C/S/R sweep: E01-E04 and the X01 control, one
# fresh process per case, against the implementation in GRANULE_IMPL_ROOT (default: the pristine
# main checkout).  Order: X01 first and last (session-drift baseline), the cases in a seeded order.
#   bash run_kcsr_native.sh <python> <outdir-under-results> [impl_root]
set -euo pipefail
PY=${1:?python}
OUT=${2:-random_kcsr_coverage_performance/screen/native}
IMPL=${3:-/home/evga/Projects/pamachine/GranuleLayer-kcsr-base}
HERE=$(cd "$(dirname "$0")" && pwd)
RES="$HERE/../results/$OUT"
mkdir -p "$RES"
ORDER=$("$PY" -c "import random; c=['E01','E02','E03','E04']; print(' '.join(['X01']+random.Random(20261023).sample(c,len(c))+['X01']))")
echo "order: $ORDER" | tee "$RES/order.txt"
declare -A K C S R
while read -r id k c s r; do K[$id]=$k; C[$id]=$c; S[$id]=$s; R[$id]=$r; done < <("$PY" -c "
import sys; sys.path.insert(0, '$HERE'); import kcsr_cases as KC
for c in KC.cases()[:4] + [KC.case('X01')]: print(c['id'], c['K'], c['C'], c['S'], c['R'])")
n=0
for id in $ORDER; do
  n=$((n+1))
  tag="${id}_native_s0_run$n"
  echo "[$(date +%T)] $tag" | tee -a "$RES/progress.log"
  (cd "$HERE" && GRANULE_IMPL_ROOT="$IMPL" timeout 3600 "$PY" bench_supergroups.py --hidden $((800000000 / K[$id])) \
      --C "${C[$id]}" --S "${S[$id]}" --R "${R[$id]}" --graph-seed 0 --init-seed 17 --input-seed 23 \
      --rounds 2 --reps 2 --iters 2 --validate --kcsr --out "$OUT/$tag") > "$RES/$tag.log" 2>&1 \
    || echo "FAILED $tag (exit $?)" | tee -a "$RES/progress.log"
  tail -1 "$RES/$tag.log" | tee -a "$RES/progress.log"
done
echo "[$(date +%T)] done" | tee -a "$RES/progress.log"
