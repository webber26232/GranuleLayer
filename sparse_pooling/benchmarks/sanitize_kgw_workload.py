"""Small workload over the EXPERIMENTAL width-preserving kgx kernels (kgx_wide.cuh: 16 x 8 tiles, 16 x 16 tiles
on the new kernels, two-input backbone + 16 x 16 tiles) and the K16 / K8 lowered one-tile control for NVIDIA
Compute Sanitizer (run_sanitizer.sh ... kgw): every policy of width_masks (U / P / A / H2 / H2A, magnitude and
random scores), b16 / b8 tile metadata incl. partial row groups and partial logical tails (C = 64 / 128 / 512),
backbone-only groups (no extra tile), bias-only rows (U8 with dropped groups), tiles overlapping backbone
coordinates, split / BF16 dZ, token tails (B = 8, 40, 136), non-finite upstream gradients, main_grad
accumulation and the L1AdamW update with the tile value mask."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental import width_masks as WM  # noqa: E402
from sparse_pooling.experimental.kgx_layer import KgxLayer  # noqa: E402
from sparse_pooling.experimental.kgx_onetile import lower_one_tile  # noqa: E402
from sparse_pooling.experimental.kgx_wide import KgxWideLayer  # noqa: E402
from sparse_pooling.experimental.l1 import L1AdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

D, G = 256, 8
n = 0


def call(lay, B, nonfinite=False, steps=1):
    global n
    lay.enable_main_grad()
    for _ in range(steps):
        x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = lay(x)
        g = torch.randn_like(y)
        if nonfinite:
            g[0, 0] = float("inf")
            g[min(1, B - 1), 1] = float("nan")
        (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    n += 1


CASES = (("U16", 0.3), ("P16", 0.3), ("A16", 0.45), ("U8", 0.1), ("U8", 0.25), ("P8", 0.2), ("A8", 0.25), ("H2", None),
         ("H2", 0.07), ("H2A", 0.2), ("P16", 0.25), ("P8", 0.125))
for C, H, R in ((128, 3003, 2), (64, 2999, 2), (512, 9001, 1)):
    for prec, B, nonfinite in (("split", 40, False), ("bf16", 8, False), ("split", 136, True)):
        base = KgxLayer(D, H, G, C, ConnectivityPlan(64, R, seed=1), variant="idf", tile=128, device="cuda", seed=2,
                        plan=PhysicalPlan(grad_precision=prec), keep_indices=False)
        with torch.no_grad():
            base.bias.normal_(0, 0.2)
        L = base.layout
        for pol, rho in CASES:
            if rho is None:  # exactly the backbone: no extra tile, every group backbone-only
                sk = WM.skeleton(WM.POLICIES[pol], L)
                rho = sk["skeleton_slots"] / sk["W0"]
            for score in ("magnitude", "random"):
                try:
                    ms = WM.build(pol, rho, base.weight.detach(), L, score=score, seed=3)
                except WM.Infeasible:
                    continue
                em = ms.element_mask(L)
                dm = KgxLayer(D, H, G, C, base.connectivity, variant="idf", tile=128, device="cuda",
                              plan=PhysicalPlan(grad_precision=prec), weights=(base.weight.detach() * em,
                                                                                base.bias.detach().clone()),
                              keep_indices=False)
                sp = KgxWideLayer(dm, ms)
                call(sp, B, nonfinite, steps=2 if score == "magnitude" else 1)
                if score == "magnitude" and B == 40:
                    ps = [sp.weight, sp.bias] + ([sp.bb] if sp.backbone else [])
                    o = L1AdamW(ps, lr=1e-3)
                    if sp.tvmask is not None:
                        o.set_mask(sp.weight, sp.tvmask)
                    o.step()
                    torch.cuda.synchronize()
                    n += 1
                if pol in ("P16", "P8") and rho in (0.25, 0.125) and C <= 256:
                    call(lower_one_tile(dm, ms), B, nonfinite)
                del sp, dm
torch.cuda.synchronize()
print(f"{n} kgw forward + backward / optimizer calls", flush=True)
print("WORKLOAD COMPLETE kgw", flush=True)
