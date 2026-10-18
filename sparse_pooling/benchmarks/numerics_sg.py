"""Numerical differences of the grouped kernels vs the generic kernels on the same graph:
max abs / rel difference and ULP-distance distribution of y and dX (FP32 reduction order
changes), bitwise check of dW / db, and the run-to-run variability of the generic dX (its
own FP32 atomics) as the baseline.

    python numerics_sg.py --out results/input_supergroups/numerics.json
"""
from __future__ import annotations

import argparse
import json

import torch

import common  # noqa: F401
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.supergroups import ConnectivityPlan


def ulp(a, b):
    ai = a.float().contiguous().view(torch.int32).to(torch.int64)
    bi = b.float().contiguous().view(torch.int32).to(torch.int64)
    ai = torch.where(ai < 0, -(ai & 0x7FFFFFFF), ai)  # monotone integer map of the float order
    bi = torch.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
    return (ai - bi).abs()


def stats(a, b):
    d = (a.double() - b.double()).abs()
    u = ulp(a, b)
    rel = d / b.double().abs().clamp_min(1e-30)
    q = torch.quantile(u.double().flatten()[:2_000_000], torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64,
                                                                     device=u.device))
    return {"max_abs": float(d.max()), "max_rel_vs_scale": float(d.max() / b.double().abs().max()),
            "median_rel": float(rel.median()), "bitwise_equal_fraction": float((u == 0).double().mean()),
            "ulp_p50_p90_p99_max": [float(x) for x in q]}


def grads(lay, x, dy):
    lay.weight.grad = lay.bias.grad = None
    xx = x.detach().clone().requires_grad_(True)
    y = lay(xx)
    (y.float() * dy).sum().backward()
    return y.detach().float(), xx.grad.float(), lay.weight.grad.float(), lay.bias.grad.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = []
    for C in (8, 64):
        for S, R in ((16, 4), (64, 16)):
            for prec in ("split", "bf16"):
                lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=C, pooling="mean", device="cuda",
                                         plan=PhysicalPlan(cuda_input_supergroup_backend="generic", grad_precision=prec),
                                         connectivity=ConnectivityPlan(S, R, seed=0), bias_std=0.1)
                lay.weight = torch.nn.Parameter(lay.weight.detach().float())
                lay.bias = torch.nn.Parameter(lay.bias.detach().float())
                g = torch.Generator(device="cuda")
                g.manual_seed(0)
                x = torch.randn(args.batch, 1024, device="cuda", generator=g).bfloat16().float()
                dy = torch.randn(args.batch, 1024, device="cuda", generator=g)
                g1 = grads(lay, x, dy)
                g2 = grads(lay, x, dy)  # generic run-to-run (dX atomics)
                lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped", grad_precision=prec)
                s1 = grads(lay, x, dy)
                s2 = grads(lay, x, dy)
                row = {"C": C, "S": S, "R": R, "precision": prec, "H": args.hidden, "M": args.batch,
                       "y_grouped_vs_generic": stats(s1[0], g1[0]), "y_grouped_run_to_run": stats(s2[0], s1[0]),
                       "y_generic_run_to_run": stats(g2[0], g1[0]),
                       "dx_grouped_vs_generic": stats(s1[1], g1[1]), "dx_generic_run_to_run": stats(g2[1], g1[1]),
                       "dx_grouped_run_to_run": stats(s2[1], s1[1]),
                       "dw_bitwise": bool(torch.equal(s1[2], g1[2])), "db_bitwise": bool(torch.equal(s1[3], g1[3]))}
                res.append(row)
                print(C, S, prec, "y", row["y_grouped_vs_generic"]["ulp_p50_p90_p99_max"],
                      "dx", row["dx_grouped_vs_generic"]["ulp_p50_p90_p99_max"],
                      "dx generic r2r", row["dx_generic_run_to_run"]["ulp_p50_p90_p99_max"],
                      "dw/db bitwise", row["dw_bitwise"], row["db_bitwise"], flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
