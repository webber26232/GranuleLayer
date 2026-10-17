"""Gradients of the recompute backward against PyTorch autograd on the reference."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan

OPS = [
    ("max", "save_winners"),
    ("max", "minimal_stats"),
    ("max", "recompute_stats"),
    ({"type": "max", "activation": "gelu"}, "save_winners"),
    ({"type": "topk_mean", "k": 4}, "save_winners"),
    ({"type": "topk_mean", "k": 16, "activation": "relu"}, "recompute_stats"),
    ("mean", "minimal_stats"),
    ({"type": "mean", "activation": "gelu"}, "minimal_stats"),
    ({"type": "logmeanexp", "temperature": 0.25}, "minimal_stats"),
    ({"type": "logmeanexp", "temperature": 2.0}, "recompute_stats"),
    ({"type": "power_mean", "p": 2}, "minimal_stats"),
    ({"type": "power_mean", "p": 3.5}, "minimal_stats"),
    ({"type": "softmax_weighted", "beta": 2.0}, "minimal_stats"),
    ({"type": "softmax_weighted", "beta": 0.5}, "recompute_stats"),
]


def _assert_close(r, tol):
    for k, (a, e) in r.items():
        err = rel_err(a, e)
        assert err < tol, f"{k}: rel err {err:.2e} >= {tol}"


@requires_cuda
@pytest.mark.parametrize("pooling,policy", OPS)
@pytest.mark.parametrize("C", [8, 16])
@pytest.mark.parametrize("shared", [True, False])
def test_triton_backward(pooling, policy, C, shared):
    lay, w, b = make_layer(64, 3000, 8, C if shared else None, pooling, policy, "triton")
    _assert_close(run_both(lay, w, b, 64, 64), 2e-5)


@requires_cuda_backend
@pytest.mark.parametrize("pooling,policy", OPS)
@pytest.mark.parametrize("C", [8, 16, 32, 64])
@pytest.mark.parametrize("D,B", [(64, 64), (1024, 40)])
def test_cuda_backward_split(pooling, policy, C, D, B):
    lay, w, b = make_layer(D, 20000, 8, C, pooling, policy, "cuda",
                           plan=PhysicalPlan(block_m=32, grad_precision="split"))
    _assert_close(run_both(lay, w, b, B, D), 2e-5)


@requires_cuda_backend
@pytest.mark.parametrize("pooling", ["mean", {"type": "logmeanexp", "temperature": 0.5},
                                     {"type": "softmax_weighted", "beta": 1.0}, {"type": "power_mean", "p": 4}])
@pytest.mark.parametrize("C", [8, 64])
def test_cuda_backward_bf16_precision(pooling, C):
    """Single-BF16 dZ operand: expect ~2^-9 relative error, not FP32 accuracy."""
    lay, w, b = make_layer(64, 20000, 8, C, pooling, "minimal_stats", "cuda",
                           plan=PhysicalPlan(block_m=32, grad_precision="bf16"))
    r = run_both(lay, w, b, 64, 64)
    assert rel_err(*r["db"]) < 2e-5  # bias gradient never goes through BF16
    assert rel_err(*r["dx"]) < 5e-3
    assert rel_err(*r["dw"]) < 5e-3


SEMANTIC = [
    {"levels": [{"type": "topk_mean", "k": 4, "neurons_per_shard": 256, "activation": "relu"},
                {"type": "logmeanexp", "temperature": 0.5}]},
    {"levels": [{"type": "max", "neurons_per_shard": 256, "transform": "affine"}, {"type": "max"}]},
    {"levels": [{"type": "mean", "neurons_per_shard": 512}, {"type": "weighted_sum"}]},
    {"levels": [{"type": "logmeanexp", "temperature": 0.5, "neurons_per_shard": 512, "transform": "gate"},
                {"type": "mean", "activation": "identity"}]},
]


@requires_cuda
@pytest.mark.parametrize("cfg", SEMANTIC)
@pytest.mark.parametrize("l1ckpt", ["save", "recompute"])
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_semantic_backward(cfg, l1ckpt, backend):
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
    cfg = dict(cfg, level1_checkpoint=l1ckpt)
    lay, w, b = make_layer(64, 4000, 8, 8, cfg, "auto", backend)
    with torch.no_grad():
        for n in lay._tnames:
            getattr(lay, "t_" + n).add_(0.2 * torch.randn_like(getattr(lay, "t_" + n)))
    _assert_close(run_both(lay, w, b, 48, 64), 2e-5)


@requires_cuda
def test_main_grad_accumulates_across_steps():
    lay, w, b = make_layer(64, 3000, 8, 8, "mean", "minimal_stats", "triton")
    lay.enable_main_grad()
    x = torch.randn(16, 64, device="cuda").bfloat16().float()
    lay(x).sum().backward()
    g1 = lay.weight.main_grad.clone()
    assert lay.weight.grad is None  # routed to main_grad, not autograd
    lay(x).sum().backward()
    assert torch.allclose(lay.weight.main_grad, 2 * g1, rtol=1e-5, atol=1e-9)
    lay.zero_main_grad()
    assert float(lay.weight.main_grad.abs().max()) == 0.0
