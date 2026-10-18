"""The one reference-tolerance exceedance of the grouped / generic C = 8 kernels
(tests/test_input_supergroups.py::test_grouped_matches_generic, split, D = 1024, H = 300,000,
G = 64, B = 256, S = 64, R = 16): locate the ReLU decisions at which the kernels and an exact
(FP64) reference disagree, and show that the exceedance is exactly those decisions.

The kernels recompute z = x W + b with BF16 operands and FP32 tensor-core accumulation; the
reference evaluates the same BF16-exact operands in FP64.  For every (token, neuron) with
|z| below --window the script records z (FP64 and the FP32 reference), the decision the
kernels took (inferred from their dW: the gradient with and without that element's
contribution), and the error of every kernel against the FP64 reference before and after
adopting the kernels' decisions at those elements.

    python relu_boundary_case.py --out results/c8_dx_scatter/relu_boundary_case
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import common  # noqa: F401
from common import save

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_input_supergroups import inputs, plan_for, run, sg_layer  # noqa: E402
from conftest import rel_err  # noqa: E402
from sparse_pooling.reference import pytorch_reference as ref  # noqa: E402

CASE = dict(D=1024, H=300_000, G=64, B=256, C=8, S=64, R=16, precision="split")


def reference_grads(lay, x, dy, dtype, flip=None):
    """Reference y, dx, dW, db in `dtype`; flip = boolean [B, G, slots] mask of elements whose
    ReLU decision is replaced by the opposite one (forward value and derivative)."""
    w = lay.weight.detach().to(dtype).clone().requires_grad_(True)
    b = lay.bias.detach().to(dtype).clone().requires_grad_(True)
    xr = x.to(dtype).clone().requires_grad_(True)
    L = lay.layout
    z = ref.hidden_preactivations(xr, lay.indices, w, b, L)
    if flip is not None:  # a = z where active (z > 0 xor flip), else 0
        act = (z > 0) ^ flip
        a = torch.where(act, z, torch.zeros_like(z))
    else:
        a = torch.relu(z)
    mask = L.valid_mask(z.device)
    y = (a * mask).sum(-1) / mask.sum(-1).to(dtype)  # Mean(ReLU) over valid slots
    (y * dy.to(dtype)).sum().backward()
    return y.detach(), xr.grad, w.grad, b.grad, z.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = CASE
    lay = sg_layer(c["D"], c["H"], c["G"], c["C"], c["S"], c["R"], plan=plan_for("generic", c["precision"]))
    x, dy = inputs(c["B"], c["D"], c["G"])
    kern = {}
    for name, backend, kw in (("generic", "generic", {}), ("grouped", "grouped", {}),
                              ("grouped_vectorized", "grouped", dict(cuda_grouped_dx_tokens=32, cuda_grouped_dx_warps=8,
                                                                      cuda_grouped_dx_reduction=1,
                                                                      cuda_grouped_dx_scatter="vectorized"))):
        lay.plan = plan_for(backend, c["precision"], **kw)
        y, dx, dw, db, info = run(lay, x, dy)
        kern[name] = dict(y=y, dx=dx, dw=dw, db=db, kernels=(info["forward_kernel"], info["dx_kernel"],
                                                             info["dw_kernel"]))
    y32, dx32, dw32, db32, z32 = reference_grads(lay, x, dy, torch.float32)
    y64, dx64, dw64, db64, z64 = reference_grads(lay, x, dy, torch.float64)
    valid = lay.layout.valid_mask(z64.device)
    near = (z64.abs() < args.window) & valid
    cand = near.nonzero().tolist()
    # kernel decision per candidate: compare the kernel dW against FP64 references with the
    # candidate's decision flipped (only that candidate) -> the closer one is the kernel's
    rows = []
    flips = torch.zeros_like(near)
    for (t, g, s) in cand:
        f = torch.zeros_like(near)
        f[t, g, s] = True
        _, _, dwf, _, _ = reference_grads(lay, x, dy, torch.float64, flip=f)
        e_keep = rel_err(kern["generic"]["dw"], dw64)
        e_flip = rel_err(kern["generic"]["dw"], dwf)
        kernel_active = bool((z64[t, g, s] > 0).item()) ^ (e_flip < e_keep)
        if e_flip < e_keep:
            flips[t, g, s] = True
        rows.append(dict(token=t, group=g, slot=s, z_fp64=float(z64[t, g, s]), z_fp32_reference=float(z32[t, g, s]),
                         fp64_active=bool((z64[t, g, s] > 0).item()), fp32_reference_active=bool((z32[t, g, s] > 0).item()),
                         kernel_active=kernel_active, dw_err_fp64=e_keep, dw_err_fp64_with_kernel_decision=e_flip))
    _, dxk, dwk, dbk, _ = reference_grads(lay, x, dy, torch.float64, flip=flips)
    errs = {}
    for name, k in kern.items():
        errs[name] = {
            "kernels": k["kernels"],
            "vs_fp32_reference": {g: rel_err(k[g], r) for g, r in (("y", y32), ("dx", dx32), ("dw", dw32), ("db", db32))},
            "vs_fp64_reference": {g: rel_err(k[g], r) for g, r in (("y", y64), ("dx", dx64), ("dw", dw64), ("db", db64))},
            "vs_fp64_with_kernel_decisions": {g: rel_err(k[g], r) for g, r in (("dx", dxk), ("dw", dwk), ("db", dbk))},
        }
    out = {"case": c, "window": args.window, "candidates": rows, "n_flipped": int(flips.sum()), "errors": errs,
           "fp32_reference_vs_fp64": {g: rel_err(a, b) for g, a, b in (("dx", dx32, dx64), ("dw", dw32, dw64),
                                                                        ("db", db32, db64))}}
    for r in rows:
        print(r)
    for name, e in errs.items():
        print(name, e)
    print("fp32 reference vs fp64", out["fp32_reference_vs_fp64"])
    save(args.out, out)


if __name__ == "__main__":
    main()
