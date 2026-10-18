"""Constant-upstream microbenchmark of the grouped dX kernel alone (one layer, one graph): the
forward / coefficient preparation is done once (c1T [G, M], X^T [D, M], nonfinite flag), then
every variant launches only its dX kernel (dXT zeroing outside the timed interval) in
interleaved rounds with alternating order.  Complete-step costs are measured by
bench_supergroups.py; this isolates the kernel.

    python bench_dx_micro.py --hidden 10000000 --S 16 --R 4 \\
        --variant base:cuda_grouped_dx_tokens=64,cuda_grouped_dx_warps=8,cuda_grouped_dx_reduction=1 ...
"""
from __future__ import annotations

import argparse
import statistics

import torch

from bench_supergroups import git_dirty, git_sha, parse_variant
from common import save
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.kernels import cuda_ops, ops
from sparse_pooling.supergroups import ConnectivityPlan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=10_000_000)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-precision", default="split")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skip-infeasible", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=args.C, pooling="mean", device="cuda",
                             seed=args.seed, connectivity=ConnectivityPlan(args.S, args.R, seed=args.seed),
                             plan=PhysicalPlan(grad_precision=args.grad_precision))
    g = torch.Generator(device="cuda")
    g.manual_seed(args.seed)
    x = torch.randn(args.batch, 1024, device="cuda", generator=g).bfloat16()
    dy = torch.randn(args.batch, 1024, device="cuda", generator=g)
    rt = lay._runtime()
    be = rt.backend
    B = args.batch
    # coefficients of Mean for upstream dy (as backward(): c1 [B, G] -> c1T [G, B])
    c1, _, _ = ops.dense_coefficients(rt.level, dy, None, rt.units.counts, None)
    c1T = c1.t().contiguous()
    xop = x.t().contiguous()
    ok = torch.linalg.vector_norm(c1, float("inf")) < 2.0**64
    nonfinite = (~ok).to(torch.uint8)
    dxt = torch.zeros((1024, B), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    import ctypes

    variants = []
    for spec in args.variant:
        name, kv = parse_variant(spec)
        kv.setdefault("cuda_input_supergroup_backend", "grouped")
        variants.append((name, PhysicalPlan(grad_precision=args.grad_precision, **kv)))

    def launch(plan):
        lay.plan = plan
        r = lay._runtime()
        a = be._sg_args(r, xop, lay.weight, lay.bias, B)
        a.nonfinite = cuda_ops._ptr(nonfinite)
        epi = cuda_ops._epilogue(plan.cuda_dx_epilogue, r.level, plan.grad_precision, "dx")
        be._sg_dx(r, a, dxt, c1T, B, int(plan.grad_precision == "split"), epi, ctypes.c_void_p(stream))
        return r

    infeasible, info, samples, rounds = {}, {}, {n: [] for n, _ in variants}, {n: [] for n, _ in variants}
    for name, plan in list(variants):
        lay.plan = plan
        why = be.sg_reason(lay._runtime(), "dx", B)
        if not why:
            try:
                be.sg_dx_scatter(lay._runtime(), *be.sg_dx_cfg(lay._runtime())[:3])
            except ValueError as e:
                why = str(e)
        if why:
            if not args.skip_infeasible:
                raise SystemExit(f"{name}: {why}")
            infeasible[name] = why
            variants.remove((name, plan))
    for name, plan in variants:  # warm-up (work lists, smem attribute)
        launch(plan)
    torch.cuda.synchronize()
    for rr in range(args.rounds):
        for name, plan in (variants if rr % 2 == 0 else variants[::-1]):
            r = launch(plan)  # rebuild the runtime outside the timed launches
            ts = []
            for _ in range(args.iters):
                dxt.zero_()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                launch(plan)
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            samples[name] += ts
            rounds[name].append(statistics.median(ts))
            info[name] = dict(r.last_info["grouped_dx"])
    out = {"commit": git_sha(), "dirty": git_dirty(), "library": str(cuda_ops.build_library()), "args": vars(args),
           "infeasible": infeasible, "variants": {}}
    for name, _ in variants:
        out["variants"][name] = {"median_ms": statistics.median(samples[name]), "min_ms": min(samples[name]),
                                 "round_medians_ms": rounds[name], "info": info[name]}
        gd = info[name]
        print(f"[{name}] S={args.S} T={gd['tokens']} NW={gd['warps']} WG={gd['reduction_group_warps']} "
              f"{gd['scatter']:20s} dX kernel {statistics.median(samples[name]):8.2f} ms "
              f"(rounds {min(rounds[name]):.2f}-{max(rounds[name]):.2f})", flush=True)
    save(args.out, out)


if __name__ == "__main__":
    main()
