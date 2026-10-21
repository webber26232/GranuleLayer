"""Compute Sanitizer workload of this phase's new kernels (results/DETERMINISM_METRICS_DW_PROJECTION.md):
the deterministic first stages (sg_fwd / sg_dx / sgx_dx with partial workspaces, token windows
under a small cap, token tails, ragged outputs, empty buckets, the in-kernel non-finite path) and
the fixed-order reduction sp_sg_reduce; the compact C = 8 dW projection (cfg 0-3, split / BF16,
ragged blocks, tails, the non-finite generic fallback launch); C = 64 deterministic control."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402


def step(S, R, H, B, plan_kw, nonfinite=False, G=16, prec="split", C=8, D=1024):
    lay = SparsePoolingLayer(D, H, G, shared_neurons=C, pooling="mean", device="cuda", bias_std=0.1,
                             connectivity=ConnectivityPlan(S, R, seed=1),
                             plan=PhysicalPlan(grad_precision=prec, **plan_kw))
    lay.enable_main_grad()
    x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    g = torch.randn_like(y.float())
    if nonfinite:
        g[0, 0] = float("inf")
    (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    i = lay.last_info
    print("ok", C, S, R, H, B, prec, plan_kw, "nonfinite" if nonfinite else "", i["reduction"]["mode"],
          (i.get("grouped_dx") or {}).get("kernel"), (i.get("grouped_dw") or {}).get("kernel"),
          (i.get("grouped_dx") or {}).get("reduction"), flush=True)


DET = dict(reduction="deterministic")
SMALL = dict(reduction="deterministic", deterministic_workspace_bytes=3 << 20)  # several token windows
for S, R in ((16, 4), (32, 8), (64, 16)):
    step(S, R, 30011, 136, DET)
    step(S, R, 30011, 264, SMALL)                    # windows + tail past the last window
    step(S, R, 30011, 136, dict(DET, cuda_grouped_dw_projection="compact"))
    step(S, R, 30011, 72, dict(DET, cuda_grouped_dw_projection="compact"), prec="bf16")
step(16, 4, 30011, 136, DET, nonfinite=True)          # in-kernel generic epilogue of the dX / dW fallback launch
step(64, 16, 30011, 136, dict(DET, cuda_grouped_dw_projection="compact"), nonfinite=True)
step(16, 4, 2000, 136, DET, G=64)                     # tiny / empty buckets
step(16, 4, 300000, 136, dict(DET, cuda_grouped_dx_chunk_units=48, cuda_grouped_fwd_chunk_units=40), D=256)  # chunks
step(16, 4, 30011, 136, dict(DET, cuda_grouped_dx_scatter="feature_owner"))  # owner variant, partial flush
for cfg in (0, 1, 2, 3):                              # compact dW tilings, fast mode, ragged H
    step(16, 4, 20011, 136, dict(cuda_grouped_dw_projection="compact", cuda_grouped_dw_cfg=cfg), G=8)
step(16, 4, 30011, 136, DET, C=64)                    # C = 64 deterministic control
step(64, 16, 30011, 136, DET, C=64)
print("ok all", flush=True)
torch.cuda.synchronize()
print("WORKLOAD COMPLETE det", flush=True)  # run_sanitizer.sh requires this line
