"""End-to-end training step: forward + backward (+ optimizer) at a given scale.

Example (full configuration):
    python sparse_pooling/benchmarks/bench_end_to_end.py --hidden 100000000 --batch 8192 \
        --suite final --out final_100M

Every measured number is printed and written to results/<out>.json.  The
timed region includes the feature-major transpose of x, all fused kernels,
state merges, coefficient preparation, dX/dW kernels and the dX cast.
"""
from __future__ import annotations

import argparse
import gc
import math

import torch

from common import KernelCounter, fmt_gib, save, time_cuda, useful_tflops
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.optim import FusedAdamW

SUITES = {
    # (name, pooling spec, checkpoint policy)
    "final": [
        ("Max", "max", "save_winners"),
        ("Max (recompute winners)", "max", "minimal_stats"),
        ("Top-8 Mean", {"type": "topk_mean", "k": 8}, "save_winners"),
        ("Top-8 Mean (recompute)", {"type": "topk_mean", "k": 8}, "recompute_stats"),
        ("Mean(ReLU)", "mean", "minimal_stats"),
        ("LogMeanExp t=0.5", {"type": "logmeanexp", "temperature": 0.5}, "minimal_stats"),
        ("Softmax-weighted b=1", {"type": "softmax_weighted", "beta": 1.0}, "minimal_stats"),
        ("Softmax-weighted b=1 (recompute)", {"type": "softmax_weighted", "beta": 1.0}, "recompute_stats"),
        ("Power mean p=2", {"type": "power_mean", "p": 2}, "minimal_stats"),
        ("Max -> Mean", {"levels": [{"type": "max", "neurons_per_shard": 16384, "out_activation": "identity"},
                                    {"type": "mean", "activation": "identity"}]}, "save_winners"),
        ("Top-8 -> LogMeanExp", {"levels": [{"type": "topk_mean", "k": 8, "neurons_per_shard": 16384},
                                            {"type": "logmeanexp", "temperature": 0.5}]}, "save_winners"),
        ("Mean(ReLU) -> LogMeanExp", {"levels": [{"type": "mean", "neurons_per_shard": 16384},
                                                 {"type": "logmeanexp", "temperature": 0.5}]}, "minimal_stats"),
        ("Max+affine -> Max", {"levels": [{"type": "max", "neurons_per_shard": 16384, "transform": "affine"},
                                          {"type": "max"}]}, "save_winners"),
    ],
    "lme_temps": [(f"LogMeanExp t={t}", {"type": "logmeanexp", "temperature": t}, "minimal_stats")
                  for t in (0.1, 0.25, 0.5, 1.0, 2.0)],
    "topk": [(f"Top-{k} Mean", {"type": "topk_mean", "k": k}, "save_winners") for k in (2, 4, 8, 16, 32)],
    "power": [(f"Power mean p={p}", {"type": "power_mean", "p": p}, "minimal_stats") for p in (1, 2, 4, 8)],
    "policies": [
        ("Max save_winners", "max", "save_winners"),
        ("Max minimal_stats", "max", "minimal_stats"),
        ("LME minimal_stats", {"type": "logmeanexp", "temperature": 0.5}, "minimal_stats"),
        ("LME recompute_stats", {"type": "logmeanexp", "temperature": 0.5}, "recompute_stats"),
        ("SW minimal_stats", {"type": "softmax_weighted", "beta": 1.0}, "minimal_stats"),
        ("SW recompute_stats", {"type": "softmax_weighted", "beta": 1.0}, "recompute_stats"),
        ("Top-8 save_winners", {"type": "topk_mean", "k": 8}, "save_winners"),
        ("Top-8 recompute_stats", {"type": "topk_mean", "k": 8}, "recompute_stats"),
    ],
}


def run_case(name, pooling, policy, args, C):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    plan = PhysicalPlan(grad_precision=args.grad_precision, input_layout=args.input_layout)
    lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=C, pooling=pooling, checkpoint_policy=policy,
                             device="cuda", plan=plan, backend=args.backend, seed=0)
    lay.enable_main_grad()
    opt = None
    if args.optimizer:
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments=args.moments, master_weights=args.master)
    x = torch.randn(args.batch, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()

    holder = {}

    def fwd():
        holder["y"] = lay(x)

    def bwd():
        holder["y"].float().square().mean().backward()
        x.grad = None

    def step():
        fwd()
        bwd()
        if opt is not None:
            opt.step()

    # forward alone, then backward alone (fresh forward each iteration, untimed)
    tf = time_cuda(fwd, warmup=1, iters=args.iters)
    bt = []
    for _ in range(args.iters + 1):
        fwd()
        torch.cuda.synchronize()
        bt.append(time_cuda(bwd, warmup=0, iters=1)["median_ms"])
    bt = sorted(bt[1:])
    tb = {"median_ms": bt[len(bt) // 2], "min_ms": bt[0], "iqr_ms": bt[(3 * len(bt)) // 4] - bt[len(bt) // 4]}
    torch.cuda.reset_peak_memory_stats()
    ts = time_cuda(step, warmup=1, iters=args.iters)
    peak = torch.cuda.max_memory_allocated()
    with KernelCounter() as kc:
        step()
    info = lay.last_info
    rec = {
        "name": name, "pooling": lay.cfg.describe(), "levels": len(lay.cfg.levels), "C": C, "policy": policy,
        "effective_policy": info.get("policy"), "recomputed_stats": info.get("recomputed_stats"),
        "forward_backend": info.get("forward_backend"), "backward_backend": info.get("backward_backend"),
        "forward_ms": tf["median_ms"], "forward_iqr_ms": tf["iqr_ms"],
        "backward_ms": tb["median_ms"], "backward_iqr_ms": tb["iqr_ms"],
        "step_ms": ts["median_ms"], "step_iqr_ms": ts["iqr_ms"],
        "peak_bytes": peak, "static_bytes": static, "checkpoint_bytes": info.get("checkpoint_bytes"),
        "workspace_bytes": info.get("workspace_bytes"), "kernel_launches_per_step": kc.launches,
        "tokens_per_s": args.batch / (ts["median_ms"] * 1e-3),
        "fwd_useful_TFLOPs": useful_tflops(args.batch, args.hidden, tf["median_ms"]),
        "optimizer": bool(opt), "grad_precision": args.grad_precision,
    }
    print(f"{name:34s} C={C:2d} {lay.last_info.get('policy', ''):14s} fwd {tf['median_ms']:8.1f} ms  "
          f"bwd {tb['median_ms']:8.1f} ms  step {ts['median_ms']:8.1f} ms  peak {fmt_gib(peak)}  "
          f"launches {kc.launches}  [{rec['forward_backend']}/{rec['backward_backend']}]", flush=True)
    del lay, opt, x, holder
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--C", type=str, default="8")
    ap.add_argument("--suite", default="final", choices=list(SUITES))
    ap.add_argument("--only", default="", help="comma-separated exact case names")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--grad-precision", default="split", choices=["split", "bf16"])
    ap.add_argument("--input-layout", default="feature_major")
    ap.add_argument("--optimizer", action="store_true")
    ap.add_argument("--moments", default="bf16")
    ap.add_argument("--master", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = []
    for C in [int(c) for c in args.C.split(",")]:
        for name, pooling, policy in SUITES[args.suite]:
            if args.only and name not in args.only.split(","):
                continue
            try:
                recs.append(run_case(name, pooling, policy, args, C))
            except torch.cuda.OutOfMemoryError as e:
                print(f"{name}: OOM ({e})", flush=True)
                recs.append({"name": name, "C": C, "error": "OOM"})
    out = args.out or f"e2e_{args.suite}_H{args.hidden}_B{args.batch}"
    print("saved", save(out, {"args": vars(args), "records": recs}))


if __name__ == "__main__":
    main()
