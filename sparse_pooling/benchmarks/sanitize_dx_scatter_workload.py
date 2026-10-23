"""Small workload over the grouped C = 8 dX scatter variants (sparse_pool_sg_dxs.cuh) for NVIDIA
Compute Sanitizer (run_sanitizer_dx_scatter.sh): every variant, S = 8 / 16 / 32 / 64, token
tiles 32 / 64 / 128, 8 / 16 warps, warp-private and (swizzle4) shared copies, split / BF16,
ReLU / identity, token tails, tiny buckets, the non-finite fallback body."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

PRIVATE = ["baseline_private", "swizzle4", "vectorized", "vectorized_transpose", "vectorized_hoisted",
           "vectorized_transpose_hoisted", "vectorized_parity", "feature_owner"]
CASES = [  # (S, R, H, T, NW, WG, precision, activation, B, nonfinite)
    (16, 4, 20011, 64, 8, 1, "split", "relu", 136, False), (16, 4, 20011, 32, 16, 1, "bf16", "relu", 40, True),
    (16, 16, 3001, 128, 8, 1, "split", "identity", 200, False), (8, 2, 9001, 128, 16, 1, "split", "relu", 72, True),
    (32, 8, 30011, 32, 8, 1, "split", "relu", 264, False), (32, 8, 30011, 64, 8, 1, "bf16", "identity", 24, False),
    (64, 16, 30011, 32, 8, 1, "split", "relu", 120, True)]
for S, R, H, T, NW, WG, prec, act, B, nonfinite in CASES:
    lay = SparsePoolingLayer(1024, H, 16, shared_neurons=8, pooling="mean", device="cuda", bias_std=0.1,
                             connectivity=ConnectivityPlan(S, R, seed=1))
    lay.cfg.levels[0].activation = act
    variants = [(sc, WG) for sc in PRIVATE if not (sc == "feature_owner" and S > 32)]
    if S == 64:
        variants += [("swizzle4", 8), ("swizzle4", 2)]
    for sc, wg in variants:
        lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped", grad_precision=prec,
                                cuda_grouped_dx_tokens=T if wg == 1 else 64, cuda_grouped_dx_warps=NW,
                                cuda_grouped_dx_reduction=wg, cuda_grouped_dx_scatter=sc)
        rt = lay._runtime()
        why = rt.backend.sg_reason(rt, "dx", B)
        if why:
            print("skip", S, T, NW, wg, sc, why, flush=True)
            continue
        x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = lay(x)
        g = torch.randn_like(y)
        if nonfinite:
            g[0, 0] = float("inf")
        (y.float() * g).sum().backward()
        torch.cuda.synchronize()
        print("ok", S, R, T, NW, wg, sc, prec, act, B, nonfinite, lay.last_info["grouped_dx"]["kernel"], flush=True)
torch.cuda.synchronize()
print("WORKLOAD COMPLETE dx_scatter", flush=True)  # run_sanitizer.sh requires this line
