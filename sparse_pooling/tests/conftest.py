import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.reference import pytorch_reference as ref  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def cuda_backend_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sparse_pooling.kernels.cuda_ops import CudaBackend

        CudaBackend.get()
        return True
    except Exception:
        return False


requires_cuda_backend = pytest.mark.skipif(not cuda_backend_available(), reason="native CUDA kernels unavailable")


def active_profile():
    """Tuning profile the current CUDA device resolves to under "auto" (SPARSE_POOL_PROFILE included);
    None without CUDA."""
    if not torch.cuda.is_available():
        return None
    from sparse_pooling.backends.capabilities import query
    from sparse_pooling.backends.profiles import select

    return select(query(torch.device("cuda"))).profile


def active_profile_name():
    prof = active_profile()
    return prof.name if prof is not None else None


# Tests asserting the RTX 3090 profile's measured automatic choices (tables, thresholds, warps,
# staging).  They do not apply when another profile is active (SPARSE_POOL_PROFILE=generic_cuda or
# another device); the numerical tests still run under every profile.
rtx3090_policy = pytest.mark.skipif(active_profile_name() not in (None, "rtx3090"),
                                    reason="asserts the RTX 3090 profile's measured auto policy; another profile is active")


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def make_layer(D, H, G, C, pooling, policy="auto", backend="triton", seed=0, bias_std=0.2, plan=None,
               weight_layout="auto"):
    """Layer whose parameters are FP32 copies of BF16-exact values, so the
    kernel gradients can be compared to FP32 reference autograd without the
    final BF16 rounding of the returned gradient.  Returns (layer, w_ref, b_ref)."""
    lay = SparsePoolingLayer(D, H, G, shared_neurons=C, pooling=pooling, checkpoint_policy=policy,
                             device="cuda", seed=seed, bias_std=bias_std, plan=plan or PhysicalPlan(block_m=32),
                             backend=backend, weight_layout=weight_layout)
    w = lay.weight.detach().float()
    b = lay.bias.detach().float()
    lay.weight = torch.nn.Parameter(w.clone())
    lay.bias = torch.nn.Parameter(b.clone())
    return lay, w.clone().requires_grad_(True), b.clone().requires_grad_(True)


def run_both(lay, w, b, B, D, seed=0):
    """Forward/backward through the fused layer and the reference with the
    same upstream gradient.  Returns dict of (kernel, reference) pairs."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float().requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    y = lay(x)
    dy = torch.randn(y.shape, device="cuda", generator=g)
    (y * dy).sum().backward()
    tp = {n: getattr(lay, "t_" + n).detach().clone().requires_grad_(True) for n in lay._tnames}
    yr = ref.reference_forward(xr, lay.indices, w, b, lay.layout, lay.cfg, tp)
    (yr * dy).sum().backward()
    out = {"y": (y.detach(), yr.detach()), "dx": (x.grad, xr.grad), "dw": (lay.weight.grad, w.grad),
           "db": (lay.bias.grad, b.grad)}
    for n in lay._tnames:
        out["d" + n] = (getattr(lay, "t_" + n).grad, tp[n].grad)
    return out
