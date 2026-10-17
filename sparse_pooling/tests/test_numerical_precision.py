"""Finite differences, numerical stability and mixed-precision winner drift."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda, requires_cuda_backend
from sparse_pooling.config import LevelSpec
from sparse_pooling.reference import pytorch_reference as ref


@requires_cuda
@pytest.mark.parametrize("backend", ["triton", "cuda"])
@pytest.mark.parametrize("pooling", [
    {"type": "logmeanexp", "temperature": 0.5},
    {"type": "softmax_weighted", "beta": 1.0},
    {"type": "mean", "activation": "gelu"},
    {"type": "power_mean", "p": 2},
])
def test_directional_derivative_vs_finite_differences(backend, pooling):
    """<grad, v> from the fused kernel vs a central difference of the FP64
    reference (smooth regions only; no Max/Top-K/ReLU kinks)."""
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
        if pooling.get("activation") == "gelu":
            pytest.skip("gelu runs on the Triton path")
    D = 1024 if backend == "cuda" else 64
    lay, w, b = make_layer(D, 6000, 8, 8, pooling, "minimal_stats", backend, bias_std=0.3)
    torch.manual_seed(3)
    x = torch.randn(8, D, device="cuda").bfloat16().float().requires_grad_(True)
    y = lay(x)
    dy = torch.randn_like(y)
    (y * dy).sum().backward()
    v = torch.randn_like(x)
    analytic = (x.grad * v).sum().double()

    def f(xx):
        return (ref.reference_forward(xx, lay.indices, w.double(), b.double(), lay.layout, lay.cfg) * dy.double()).sum()

    h = 1e-4
    with torch.no_grad():
        xd = x.detach().double()
        fd = (f(xd + h * v.double()) - f(xd - h * v.double())) / (2 * h)
    assert abs((analytic - fd) / fd).item() < 1e-4


@requires_cuda
@pytest.mark.parametrize("backend", ["triton", "cuda"])
@pytest.mark.parametrize("spec", [LevelSpec("logmeanexp", temperature=0.1), LevelSpec("softmax_weighted", beta=20.0),
                                  LevelSpec("power_mean", p=8)])
def test_large_magnitudes_are_stable(backend, spec):
    """Big pre-activations (|z| ~ 50) must not overflow the streaming states."""
    if backend == "cuda":
        from conftest import cuda_backend_available

        if not cuda_backend_available():
            pytest.skip("no CUDA backend")
    import dataclasses

    lay, w, b = make_layer(1024, 20000, 8, 8, dataclasses.asdict(spec), "minimal_stats", backend)
    with torch.no_grad():
        lay.weight.mul_(16)
        w.mul_(16)
    x = (torch.randn(16, 1024, device="cuda") * 4).bfloat16().float().requires_grad_(True)
    y = lay(x)
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(lay.weight.grad).all()
    z = ref.hidden_preactivations(x.detach(), lay.indices, w, b, lay.layout)
    yr = ref.pool(z, lay.layout.valid_mask("cuda"), spec)
    assert rel_err(y, yr) < 1e-5


@requires_cuda
def test_lme_limits():
    """tau -> 0 approaches Max, large tau approaches Mean (identity)."""
    lay, w, b = make_layer(64, 3000, 8, 8, {"type": "logmeanexp", "temperature": 0.01}, "minimal_stats", "triton")
    x = torch.randn(8, 64, device="cuda").bfloat16().float()
    z = ref.hidden_preactivations(x, lay.indices, w, b, lay.layout)
    mask = lay.layout.valid_mask("cuda")
    mx = ref.pool(z, mask, LevelSpec("max", out_activation="identity"))
    n = mask.sum(-1).float()
    assert (lay(x) - mx).abs().max() < 0.01 * torch.log(n).max() + 1e-3
    lay2, w2, b2 = make_layer(64, 3000, 8, 8, {"type": "logmeanexp", "temperature": 1000.0}, "minimal_stats",
                              "triton")
    mean = ref.pool(ref.hidden_preactivations(x, lay2.indices, w2, b2, lay2.layout), mask,
                    LevelSpec("mean", activation="identity"))
    assert (lay2(x) - mean).abs().max() < 1e-2


@requires_cuda
def test_winner_agreement_bf16_operands_vs_fp32_reference():
    """Kernels and reference use the same BF16 operands and FP32 accumulation;
    winner disagreement can only come from summation-order rounding at
    near-ties.  Measure it."""
    lay, w, b = make_layer(1024, 200_000, 64, 8, "max", "save_winners", "auto")
    x = torch.randn(64, 1024, device="cuda").bfloat16()
    rt = lay._runtime()
    out = rt.forward(x, lay.weight.detach().bfloat16(), lay.bias.detach().bfloat16(), True)
    ex = {}
    ref.pool(ref.hidden_preactivations(x, lay.indices, w, b, lay.layout), lay.layout.valid_mask("cuda"),
             LevelSpec("max"), ex)
    agree = (out["winner"].long() == ex["winner"]).float().mean().item()
    assert agree > 0.999
