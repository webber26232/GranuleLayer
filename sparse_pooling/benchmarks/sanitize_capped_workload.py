"""Compute Sanitizer workload for swizzle4_capped (SC_SWZ4M, added after the main sanitizer run):
S = 64 / 16 with shared copies (WG = 8, 2) and warp-private copies, T = 64 / 128, split / BF16,
ReLU / identity, token tails, the non-finite fallback body."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

for S, R, H, T, WG, prec, act, B, nonfinite in [
        (64, 16, 30011, 64, 8, "split", "relu", 136, False), (64, 16, 30011, 64, 2, "bf16", "identity", 72, True),
        (16, 4, 20011, 128, 8, "split", "relu", 200, True), (16, 4, 20011, 64, 1, "split", "relu", 40, False)]:
    lay = SparsePoolingLayer(1024, H, 16, shared_neurons=8, pooling="mean", device="cuda", bias_std=0.1,
                             connectivity=ConnectivityPlan(S, R, seed=1))
    lay.cfg.levels[0].activation = act
    lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped", grad_precision=prec, cuda_grouped_dx_tokens=T,
                            cuda_grouped_dx_warps=8, cuda_grouped_dx_reduction=WG, cuda_grouped_dx_scatter="swizzle4_capped")
    x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    g = torch.randn_like(y)
    if nonfinite:
        g[0, 0] = float("inf")
    (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    print("ok", S, R, T, WG, prec, act, B, nonfinite, lay.last_info["grouped_dx"]["kernel"], flush=True)
