"""Small workload over the EXPERIMENTAL generalized-fan-in kernels (bundle "kgen") for NVIDIA Compute Sanitizer
(run_sanitizer.sh ... kgen): K = 8 / 16 / 32 / 64, C = 8 / 24 / 136 / 256 (one n-tile, C % 16 == 8 tails, C <= 256),
S = 16 - 128, split / BF16, ReLU / identity, token tails (B = 8, 40, 136), warp-private and shared (CAS) dX slab
copies, partial C blocks, non-finite coefficients; plus logical C = 128 / 256 lowered to the production C = 64
kernels (grouped and generic phases)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling.backends.cuda import variants as V  # noqa: E402
from sparse_pooling.backends.dispatch import smem_limit  # noqa: E402
from sparse_pooling.config import PhysicalPlan  # noqa: E402
from sparse_pooling.experimental.kgen_layer import KgenLayer  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

D, G = 512, 8
n = 0
for K, C, S, R, H in ((8, 8, 16, 2, 3001), (8, 136, 32, 2, 9000), (16, 24, 16, 4, 4001), (16, 256, 128, 2, 9001),
                      (32, 8, 64, 2, 2003), (32, 136, 128, 2, 6000), (64, 24, 64, 2, 2501), (64, 256, 128, 2, 9000)):
    for prec, act, B, nonfinite in (("split", "relu", 40, False), ("bf16", "identity", 8, False),
                                    ("split", "relu", 136, True)):
        pool = "mean" if act == "relu" else {"type": "mean", "activation": "identity"}
        lay = KgenLayer(D, H, G, K, C, ConnectivityPlan(S, R, seed=1), pooling=pool, device="cuda", seed=2,
                        bias_std=0.1, plan=PhysicalPlan(grad_precision=prec))
        lay.enable_main_grad()
        rt = lay._runtime()
        for wg in (1, 2, rt.cfg[3]):  # private copies / pairs / one CTA-shared copy, where they fit
            if V.kgen_smem("dx", D, S, rt.cfg, wg) > smem_limit(rt.caps):
                continue
            rt.dx_group_warps = wg
            x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            y = lay(x)
            g = torch.randn_like(y)
            if nonfinite:
                g[0, 0] = float("inf")
                g[min(1, B - 1), 1] = float("nan")
            (y.float() * g).sum().backward()
            torch.cuda.synchronize()
            n += 1
# logical C = 128 / 256 lowered to the production C = 64 kernels (grouped and generic phases)
from sparse_pooling.experimental.lowering import build_logical, lower  # noqa: E402

for C, S, R, H in ((128, 32, 2, 9001), (256, 64, 2, 9000)):
    L, idx, w, b, bk, feat, spec, _ = build_logical(D, H, G, C, ConnectivityPlan(S, R, seed=1), "cuda", seed=2, bias_std=0.1)
    for be in ("grouped", "generic"):
        lay = lower(L, idx, w, b, bk, feat, spec, plan=PhysicalPlan(cuda_input_supergroup_backend=be), device="cuda")
        lay.enable_main_grad()
        x = torch.randn(40, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = lay(x)
        (y.float() * torch.randn_like(y)).sum().backward()
        torch.cuda.synchronize()
        n += 1
print(f"{n} kgen / lowered forward + backward calls", flush=True)
print("WORKLOAD COMPLETE kgen", flush=True)
