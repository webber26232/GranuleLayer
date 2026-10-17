#!/usr/bin/env bash
# Full measurement campaign (RTX 3090).  ~2-3 h.  Logs + JSON in sparse_pooling/results/.
#   bash sparse_pooling/benchmarks/run_all.sh [python]
set -uo pipefail
PY=${1:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HERE/../results
mkdir -p "$R"
run() { local name=$1; shift; echo "=== $name: $*"; "$PY" "$@" 2>&1 | grep -v Warning | tee "$R/$name.log"; }

# 1. final comparison table, full scale (H = 100M, B = 8192), both C extremes
run e2e_final_100M_C8   "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 8  --suite final --optimizer --out e2e_final_100M_C8
run e2e_final_100M_C64  "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 64 --suite final --optimizer --out e2e_final_100M_C64
# 2. gradient precision (single BF16 dZ = standard mixed-precision semantics)
run e2e_bf16_100M_C8    "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 8  --suite final --optimizer --grad-precision bf16 --only "Mean(ReLU),LogMeanExp t=0.5,Softmax-weighted b=1,Power mean p=2" --out e2e_bf16_100M_C8
run e2e_bf16_100M_C64   "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 64 --suite final --optimizer --grad-precision bf16 --only "Mean(ReLU),LogMeanExp t=0.5,Softmax-weighted b=1,Power mean p=2" --out e2e_bf16_100M_C64
# 3. operator parameter sweeps
run e2e_lme_temps_100M  "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 8,64 --suite lme_temps --out e2e_lme_temps_100M
run e2e_topk_100M       "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 8,64 --suite topk --out e2e_topk_100M
run e2e_power_100M      "$HERE/bench_end_to_end.py" --hidden 100000000 --batch 8192 --C 8 --suite power --out e2e_power_100M
# 4. H x B scaling (shared C = 16, three operators)
for H in 1000000 10000000 100000000; do
  for B in 256 1024 4096 8192; do
    run "e2e_scale_H${H}_B${B}" "$HERE/bench_end_to_end.py" --hidden $H --batch $B --C 16 --suite final \
        --only "Max,Top-8 Mean,LogMeanExp t=0.5" --iters 3 --out "e2e_scale_H${H}_B${B}"
  done
done
# 5. configuration sweep (C, batch tile, splits, input layout, scheduling)
run sweep_H10M          "$HERE/sweep_configs.py" --hidden 10000000 --batch 8192 --quick --out sweep_H10M_B8192
# 6. memory-compute trade-off (strategy A only feasible at small H)
run tradeoff_H262k      "$HERE/bench_tradeoff.py" --hidden 262144 --batch 1024 --out tradeoff_H262144_B1024
# 7. memory accounting at full scale
run memory_100M         "$HERE/bench_memory.py" --hidden 100000000 --batch 8192 --out memory_H100M_B8192
run memory_100M_master  "$HERE/bench_memory.py" --hidden 100000000 --batch 8192 --moments fp32 --master --out memory_H100M_B8192_fp32master
# 8. machine-code inspection and sanitizer
run sass                "$HERE/inspect_sass.py"
bash "$HERE/run_sanitizer.sh" "$PY" 2>&1 | tee "$R/sanitizer.log"
echo "done"
