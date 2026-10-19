"""Compute Sanitizer workload of this phase's new kernels (results/SCALE_DISPATCH_FUSED_BACKWARD.md):
the fused shared-projection backward (v1 cfg 0 / 3, v2 cfg 6 / 7; shared projection and the
recompute control) with token tails, M below one window, ragged outputs and the non-finite
fallback launches; the S = 64 whole plans (named and auto-selected, split / BF16, tails); the
projection diagnostic and the fusion screening kernels."""
import ctypes
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch  # noqa: E402

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.kernels.cuda_ops import CudaBackend, _ptr  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402


def step(S, R, H, B, plan_kw, nonfinite=False, G=16, prec="split"):
    lay = SparsePoolingLayer(1024, H, G, shared_neurons=8, pooling="mean", device="cuda", bias_std=0.1,
                             connectivity=ConnectivityPlan(S, R, seed=1),
                             plan=PhysicalPlan(cuda_input_supergroup_backend="grouped", grad_precision=prec, **plan_kw))
    lay.enable_main_grad()
    x = torch.randn(B, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = lay(x)
    g = torch.randn_like(y.float())
    if nonfinite:
        g[0, 0] = float("inf")
    (y.float() * g).sum().backward()
    torch.cuda.synchronize()
    info = lay.last_info
    k = (info.get("grouped_fused") or {}).get("kernel") or (info.get("grouped_dx") or {}).get("kernel")
    print("ok", S, R, H, B, prec, plan_kw, "nonfinite" if nonfinite else "", info.get("backward_organization"), k, flush=True)


for org in ("fused", "fused_recompute"):
    for cfg in (0, 3, 6, 7):
        step(16, 4, 20011, 136, dict(cuda_grouped_backward=org, cuda_grouped_fused_cfg=cfg))
    step(16, 16, 20011, 16, dict(cuda_grouped_backward=org, cuda_grouped_fused_cfg=6))  # M < one window
    step(32, 8, 20011, 72, dict(cuda_grouped_backward=org, cuda_grouped_fused_cfg=7))
    step(16, 4, 20011, 72, dict(cuda_grouped_backward=org, cuda_grouped_fused_cfg=6), nonfinite=True)  # fallbacks
for plan in ("shared_t64", "private_t32"):
    for prec in ("split", "bf16"):
        step(64, 16, 30011, 136, dict(cuda_grouped_dx_plan=plan), prec=prec)
step(64, 16, 300000, 1024, {}, prec="bf16", G=8)  # auto rule selection (units / bucket >= 2 -> private_t32)
step(64, 16, 30011, 1024, {}, prec="bf16")        # below the threshold -> shared_t64
step(64, 16, 300000, 1032, {}, G=4, nonfinite=True)  # split auto, tail, non-finite in-kernel fallback
# diagnostic / screening kernels
be = CudaBackend.get(strict=True)
lay = SparsePoolingLayer(256, 20011, 8, shared_neurons=8, pooling="mean", device="cuda",
                         plan=PhysicalPlan(cuda_input_supergroup_backend="grouped"), connectivity=ConnectivityPlan(16, 4, seed=1))
rt = lay._runtime()
x = torch.randn(72, 256, device="cuda").bfloat16()
xop = x.t().contiguous()
a = be._sg_args(rt, xop, lay.weight, lay.bias, 72)
items = torch.tensor([[0, 0, 2000], [5, 4, 2000], [17, 3, 1300]], dtype=torch.int32, device="cuda")
z = torch.empty((3, 4, 16, 8), device="cuda")
z2, m = torch.empty_like(z), torch.empty((6, 3, 4, 16, 8), dtype=torch.int32, device="cuda")
be.lib.sp_diag_proj_c8.restype = ctypes.c_int
assert be.lib.sp_diag_proj_c8(ctypes.byref(a), _ptr(items), 3, _ptr(z), _ptr(z2), _ptr(m), ctypes.c_void_p(0)) == 0
feat = rt.sg.feat.contiguous()
cq = torch.tensor([0, 1, 1, 5], dtype=torch.int32, device="cuda")
dXT = torch.zeros((256, 72), device="cuda")
be.lib.sp_screen_flush.restype = ctypes.c_int
assert be.lib.sp_screen_flush(_ptr(feat), _ptr(cq), 4, ctypes.c_int(16), ctypes.c_int(72), ctypes.c_int(32), ctypes.c_int(3),
                              ctypes.c_int(256), _ptr(dXT), ctypes.c_float(1.0), ctypes.c_void_p(0)) == 0
buf = torch.zeros(4 * 1000, device="cuda")
be.lib.sp_screen_rmw.restype = ctypes.c_int
assert be.lib.sp_screen_rmw(_ptr(buf), ctypes.c_longlong(1000), ctypes.c_float(1.0), ctypes.c_void_p(0)) == 0
torch.cuda.synchronize()
print("ok diag / screen", flush=True)
