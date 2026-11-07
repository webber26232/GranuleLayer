#!/usr/bin/env bash
# Reproduction of the A3mean follow-up (docs/experiments/a3mean_long_horizon_batch_scaling.md).
#   bash sparse_pooling/benchmarks/run_lrf.sh <python> <run-root> <packed-root> <checkpoint-root> <stage> [stage ...]
# stages: tests sanitizer gate perf A B C D_tune D_direct D_accum anchor secondary B_ext unlock test
# Every training stage resumes (per-job JSON, full checkpoints).  Run it detached, one workload on the GPU:
#   systemd-run --user --unit=lrf -p MemoryHigh=20G -p MemoryMax=24G --working-directory=$PWD --setenv=PATH="$PATH" \
#       bash sparse_pooling/benchmarks/run_lrf.sh python RUN PACKED CKPT gate perf A B C D_tune D_direct D_accum
set -uo pipefail
PY=$1; OUT=$2; PACKED=$3; CKPT=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
CFG="$HERE/../../configs/learned_reduction_followup"
export SPARSE_POOL_CACHE_DIR=${SPARSE_POOL_CACHE_DIR:-$OUT/cuda_cache}
mkdir -p "$OUT/manifests" "$PACKED" "$CKPT"
cd "$HERE"
stage() {
  "$PY" lrf_campaign.py "$1" --out "$OUT" --lrs "$CFG/recovered_lrs.json" --ckpt-root "$CKPT" --write "$OUT/manifests/$1.json" || return 1
  "$PY" lrf_train.py session --manifest "$OUT/manifests/$1.json" --root "$PACKED" --ckpt-root "$CKPT" --out "$OUT/$1"
}
for s in "$@"; do
  echo "=== $s $(date -Is)"
  t0=$(date +%s)
  case $s in
    tests) "$PY" -m pytest "$HERE/../tests/test_lrf_followup.py" "$HERE/../tests/test_learned_reduction.py" -q ;;
    sanitizer) bash run_sanitizer.sh --jobs 4 --out "$OUT/sanitizer" "$PY" lrd ;;
    gate) "$PY" lrf_gate.py --root "$PACKED" --tmp "$OUT/gate_tmp" --out "$OUT/gate/gate.json" || exit 1 ;;
    perf) "$PY" lrf_perf.py session --plan "$CFG/perf_plan.json" --root "$PACKED" --out "$OUT/perf" ;;
    A) stage A_tune; stage A_ext ;;
    C) stage C_tune; stage C_ext; stage C ;;
    B|D_tune|D_direct|D_accum|anchor|secondary|B_ext) stage "$s" ;;
    extension_gate) "$PY" lrf_campaign.py extension_gate --out "$OUT" --write "$OUT/extension_gate.json" ;;
    unlock) "$PY" - "$OUT" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
stages = sorted(p.name for p in out.iterdir() if (p / "failures.jsonl").exists() or list(p.glob("*.json")))
c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
(out / "test_unlock.json").write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), commit=c, stages_present=stages,
    statement="all LR selections and the extension decision are final; the test split was not read before this file"), indent=1))
print((out / "test_unlock.json").read_text())
PY
      ;;
    test)
      for st in B C D_direct D_accum anchor secondary B_ext; do
        [ -d "$OUT/$st" ] || continue
        mkdir -p "$OUT/test_eval/$st"
        for f in "$OUT/$st"/*.json; do
          g="$OUT/test_eval/$st/$(basename "$f")"
          [ -f "$g" ] || "$PY" lrf_train.py evaluate --record "$f" --root "$PACKED" --unlock "$OUT/test_unlock.json" --out "$g"
        done
      done ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
  echo "=== $s done rc $? $(( $(date +%s) - t0 ))s $(date -Is)"
done
