"""Numerics of the dW projection accumulator initialisation (experiment A3).

Runs one Mean(ReLU) backward with the library selected by SPARSE_POOL_NVCC_EXTRA and
saves dW / db / dX.  With dy[b, u] = n_u (neurons of unit u), c1 = 1 exactly, so the
split-precision db of a neuron is the exact number of tokens whose recomputed
pre-activation is > 0: comparing two builds counts ReLU decision flips directly.

    SPARSE_POOL_NVCC_EXTRA=-DSP_ABLATE_DW=5 python numerics_dw_bias.py --out diag.pt
    python numerics_dw_bias.py --out base.pt
    python numerics_dw_bias.py --compare base.pt diag.pt
"""
from __future__ import annotations

import argparse
import json

import torch

import common  # noqa: F401
from sparse_pooling import PhysicalPlan, SparsePoolingLayer


def run(args):
    lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=64, pooling="mean", device="cuda", seed=0,
                             bias_std=args.bias_std, plan=PhysicalPlan(grad_precision="split"), backend="auto")
    lay.enable_main_grad()  # FP32 gradient buffers: token counts stay exact
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    x = torch.randn(args.batch, 1024, device="cuda", generator=g, dtype=torch.bfloat16).requires_grad_(True)
    y = lay(x)
    n = lay.layout.counts_tensor("cuda").float()
    (y.float() * n[None, :]).sum().backward()
    torch.save({"dw": lay.weight.main_grad.cpu(), "db": lay.bias.main_grad.cpu(), "dx": x.grad.float().cpu(),
                "info": {k: str(v) for k, v in lay.last_info.items()}}, args.out)


def ulp(a, b):
    ia = a.view(torch.int32).long()
    ib = b.view(torch.int32).long()
    ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ia - ib).abs()


def compare(pa, pb):
    a, b = torch.load(pa), torch.load(pb)
    rep = {}
    for k in ("dw", "db", "dx"):
        x, y = a[k], b[k]
        d = (x - y).abs()
        u = ulp(x.contiguous(), y.contiguous())
        rep[k] = {"bitwise_equal": bool(torch.equal(x, y)), "max_abs": float(d.max()),
                  "rel_l2": float(d.norm() / (x.norm() + 1e-30)), "frac_differs": float((u > 0).float().mean()),
                  "ulp_max": int(u.max()), "ulp_p99": float(torch.quantile(u.flatten()[:1 << 24].float(), 0.99))}
    flips = (a["db"] - b["db"]).abs()
    rep["relu_flips"] = {"neurons_with_flip": int((flips > 0).sum()), "neurons": int(flips.numel()),
                         "abs_count_diff_total": float(flips.sum()),
                         "db_is_token_count": bool(torch.equal(a["db"], a["db"].round()))}
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--bias-std", type=float, default=0.2)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    if args.compare:
        rep = compare(*args.compare)
        if args.save:
            with open(args.save, "w") as f:
                json.dump(rep, f, indent=1)
    else:
        run(args)


if __name__ == "__main__":
    main()
