"""Small workload over every grouped (input-supergroup) kernel variant for NVIDIA Compute
Sanitizer (run_sanitizer_sg.sh): C = 8 / 64, S = 16 / 64 / 256, all compiled token tiles,
warp-private / grouped / CTA-shared dX copies, every dW tiling, split / BF16, ReLU / identity,
token tails, tiny buckets, the non-finite fallback launches."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402

G = "grouped"
for C in (8, 64):
    for S, R, H in ((16, 4, 20011), (64, 16, 30011), (256, 4, 5003)):
        lay = SparsePoolingLayer(1024, H, 16, shared_neurons=C, pooling="mean", device="cuda", bias_std=0.1,
                                 connectivity=ConnectivityPlan(S, R, seed=1))
        for ft, dx, dwc, prec, act, B, nonfinite in [
            (16, (16, 16, 1), 0, "split", "relu", 40, False), (32, (32, 8, 2), 1, "bf16", "relu", 72, False),
            (64, (64, 8, 8), 2, "split", "identity", 136, False), (128, (128, 16, 16), 3, "split", "relu", 200, True),
            (128, (64, 16, 4), 2, "bf16", "relu", 8, True)]:
            plan = PhysicalPlan(cuda_input_supergroup_backend=G, grad_precision=prec, cuda_grouped_fwd_tokens=ft,
                                cuda_grouped_dx_tokens=dx[0], cuda_grouped_dx_warps=dx[1],
                                cuda_grouped_dx_reduction=dx[2], cuda_grouped_dw_cfg=dwc)
            rt_ok = True
            lay.plan = plan
            lay.cfg.levels[0].activation = act
            rt = lay._runtime()
            reasons = {ph: rt.backend.sg_reason(rt, ph, B) for ph in ("fwd", "dx", "dw")}
            if any(reasons.values()):
                print("skip", C, S, ft, dx, dwc, reasons, flush=True)
                continue
            x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            y = lay(x)
            g = torch.randn_like(y)
            if nonfinite:
                g[0, 0] = float("inf")
            (y.float() * g).sum().backward()
            torch.cuda.synchronize()
            print("ok", C, S, ft, dx, dwc, prec, act, B, nonfinite, lay.last_info["forward_kernel"],
                  lay.last_info["dx_kernel"], lay.last_info["dw_kernel"], flush=True)
