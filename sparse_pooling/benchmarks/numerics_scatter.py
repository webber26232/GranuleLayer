"""Fixed-upstream numerics of the grouped dX scatter variants on one graph: dX of every variant
against the baseline grouped dX (same grouped forward / dW) next to the baseline's own
run-to-run variation (global FP32 dX atomics), ULP distributions, bitwise dW / db, and the
deterministic setting (R = 1, one chunk per supergroup: one CTA per (feature, token tile)), in
which the order-preserving variants must be bitwise equal to the baseline.

    python numerics_scatter.py --out results/c8_dx_scatter/numerics_fixed_dy
"""
from __future__ import annotations

import argparse

import torch

import common  # noqa: F401
from common import save
from numerics_sg import stats
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels.cuda_ops import SG_DX_SCATTERS
from sparse_pooling.supergroups import ConnectivityPlan

CASES = [(16, 4, 64, 8, 1), (32, 8, 32, 8, 1), (64, 16, 64, 8, 8), (16, 1, 64, 8, 1), (32, 1, 32, 8, 1)]
VARIANTS = ["vectorized_transpose_hoisted", "vectorized_hoisted", "vectorized", "vectorized_transpose", "swizzle4",
            "feature_owner", "swizzle4_capped"]


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
    for S, R, T, NW, WG in CASES:
        for prec in ("split", "bf16"):
            det = R == 1
            base_kw = dict(cuda_input_supergroup_backend="grouped", grad_precision=prec, cuda_grouped_dx_tokens=T,
                           cuda_grouped_dx_warps=NW, cuda_grouped_dx_reduction=WG)
            if det:
                base_kw["cuda_grouped_dx_chunk_units"] = 1 << 30
            lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=8, pooling="mean", device="cuda",
                                     plan=PhysicalPlan(**base_kw), connectivity=ConnectivityPlan(S, R, seed=0),
                                     bias_std=0.1)
            lay.weight = torch.nn.Parameter(lay.weight.detach().float())
            lay.bias = torch.nn.Parameter(lay.bias.detach().float())
            g = torch.Generator(device="cuda")
            g.manual_seed(0)
            x = torch.randn(args.batch, 1024, device="cuda", generator=g).bfloat16().float()
            dy = torch.randn(args.batch, 1024, device="cuda", generator=g)
            lay.plan = PhysicalPlan(**base_kw, cuda_grouped_dx_scatter="baseline")
            b1 = grads(lay, x, dy)
            b2 = grads(lay, x, dy)
            row = {"S": S, "R": R, "T": T, "NW": NW, "WG": WG, "precision": prec, "deterministic": det,
                   "H": args.hidden, "M": args.batch, "dx_baseline_run_to_run": stats(b2[1], b1[1]), "variants": {}}
            for sc in VARIANTS:
                lay.plan = PhysicalPlan(**base_kw, cuda_grouped_dx_scatter=sc)
                try:
                    c = grads(lay, x, dy)
                except ValueError as e:
                    row["variants"][sc] = {"unavailable": str(e)}
                    continue
                row["variants"][sc] = {
                    "order": SG_DX_SCATTERS[sc]["order"], "dx_vs_baseline": stats(c[1], b1[1]),
                    "dx_bitwise": bool(torch.equal(c[1].view(torch.int32), b1[1].view(torch.int32))),
                    "dw_bitwise": bool(torch.equal(c[2], b1[2])), "db_bitwise": bool(torch.equal(c[3], b1[3])),
                    "y_vs_baseline": stats(c[0], b1[0])}
                v = row["variants"][sc]
                print(f"S={S} R={R} T={T} WG={WG} {prec:5s} {sc:30s} dX ulp p50/p99/max "
                      f"{v['dx_vs_baseline']['ulp_p50_p90_p99_max'][0]:.0f}/{v['dx_vs_baseline']['ulp_p50_p90_p99_max'][2]:.0f}/"
                      f"{v['dx_vs_baseline']['ulp_p50_p90_p99_max'][3]:.0f} rel {v['dx_vs_baseline']['max_rel_vs_scale']:.1e} "
                      f"bitwise {v['dx_bitwise']} | baseline r2r max ulp "
                      f"{row['dx_baseline_run_to_run']['ulp_p50_p90_p99_max'][3]:.0f} | dW/db bitwise {v['dw_bitwise']}/{v['db_bitwise']}",
                      flush=True)
            res.append(row)
            del lay
            torch.cuda.empty_cache()
    save(args.out, {"args": vars(args), "results": res})


if __name__ == "__main__":
    main()
