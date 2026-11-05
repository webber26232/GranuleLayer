#!/usr/bin/env bash
# Integration-head validation of the named Cover-TopK/16 layer (docs/validation/cover_topk16_rtx3090_dx.md).
# GPU work is strictly serialized; profilers / sanitizer never overlap a timing session.  Every result goes to $OUT.
#
#   PY=python OUT=/path/results CKPT_G0=g0.pt CKPT_G1=g1.pt CKPT_G2=g2.pt BASELINE=/path/checkout-of-main \
#   [RESEARCH_L1=/path/research/sparse_pooling/experimental/l1.py] [RESEARCH_FINAL=/path/research-results/final] \
#   [RESEARCH_PRIMARY_CSV=/path/final_primary.csv] [FRESH_CACHE=$OUT/cuda_cache] \
#   bash sparse_pooling/benchmarks/run_cover_topk16_validation.sh
#
# Stages: fresh-cache build + focused tests; same-session timing (graphs 0 / 1 / 2 at M 8192, M 512 / 64 guards);
# full-size identity / budget / coverage / round trip (graphs 0 / 1 / 2); compact training smoke (legacy vs auto, seeds
# 0 / 1 / 2); optimizer equivalence and research contrast recomputation (when the research inputs are given);
# fan-in-8 compatibility smoke against main; Compute Sanitizer; complete test suite.  Exit status: number of failed
# stages (a failed stage never stops the later ones; each is reported in $OUT/stages.tsv).
set -u
: "${PY:?}" "${OUT:?}" "${CKPT_G0:?}" "${CKPT_G1:?}" "${CKPT_G2:?}" "${BASELINE:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT"
FRESH_CACHE=${FRESH_CACHE:-$OUT/cuda_cache}
STAGES=$OUT/stages.tsv
: > "$STAGES"
FAILED=0
git -C "$ROOT" rev-parse HEAD > "$OUT/source_head.txt"
git -C "$ROOT" status --short >> "$OUT/source_head.txt"
"$PY" -c "import torch,triton,platform;print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,'python',platform.python_version())" > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv >> "$OUT/environment.txt" 2>&1
"${NVCC:-nvcc}" --version 2>&1 | head -2 >> "$OUT/environment.txt"
stage() {  # name, command...
  local name=$1 t0 rc; shift
  echo "=== $name $(date)"
  t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$STAGES"
  echo "=== $name exit $rc ($(( $(date +%s) - t0 )) s)"
  [ "$rc" -eq 0 ] || FAILED=$((FAILED + 1))
}
cd "$ROOT/sparse_pooling"
# a fresh task-specific cache once: catches missing translation units / headers / registrations
stage fresh_cache_focused env --default-signal=INT SPARSE_POOL_CACHE_DIR="$FRESH_CACHE" "$PY" -m pytest -q \
  tests/test_cover_topk16.py tests/test_kgen_dx_dispatch.py tests/test_kgen.py tests/test_memory_accounting.py \
  tests/test_paired_stats.py tests/test_quota_repair_accounting.py tests/test_cpu_import.py tests/test_dispatch.py \
  tests/test_build.py
export SPARSE_POOL_CACHE_DIR=$FRESH_CACHE
cd "$HERE"
stage timing "$PY" cover_topk16_timing.py session --checkpoints "$CKPT_G0,$CKPT_G1,$CKPT_G2" --out "$OUT/timing"
stage identity_g0 "$PY" cover_topk16_identity.py --checkpoint "$CKPT_G0" --normalization energy --out "$OUT/identity_g0.json" \
  ${EXPECT_G0:-}
stage identity_g1 "$PY" cover_topk16_identity.py --checkpoint "$CKPT_G1" --normalization energy --out "$OUT/identity_g1.json" \
  ${EXPECT_G1:-}
stage identity_g2 "$PY" cover_topk16_identity.py --checkpoint "$CKPT_G2" --normalization energy --out "$OUT/identity_g2.json" \
  ${EXPECT_G2:-}
stage train_smoke "$PY" cover_topk16_train_smoke.py --checkpoint "$CKPT_G0" --out "$OUT/train_smoke.json"
[ -n "${RESEARCH_L1:-}" ] && stage optimizer_equivalence "$PY" optimizer_equivalence.py --l1 "$RESEARCH_L1" \
  --out "$OUT/optimizer_equivalence.json"
[ -n "${RESEARCH_FINAL:-}" ] && stage research_contrasts env CUDA_VISIBLE_DEVICES="" "$PY" verify_research_contrasts.py \
  --final-dir "$RESEARCH_FINAL" ${RESEARCH_PRIMARY_CSV:+--committed-csv "$RESEARCH_PRIMARY_CSV"} \
  --out "$OUT/research_contrasts.json"
stage compat_smoke "$PY" cover_topk16_compat_smoke.py compare --baseline "$BASELINE" --out "$OUT/compat_smoke.json"
stage sanitizer bash run_sanitizer.sh --jobs "${SANITIZER_JOBS:-4}" --out "$OUT/sanitizer" "$PY" cover kgen generic sg dx_scatter det
cd "$ROOT/sparse_pooling"
stage full_suite env --default-signal=INT "$PY" -m pytest -q tests
echo "=== done; $FAILED failed stage(s) $(date)"
exit "$FAILED"
