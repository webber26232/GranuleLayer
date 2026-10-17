"""Backward compute variants of branch backward-compute-optimization.

Every variant must reproduce the dw-indexed-staging kernels: dW/db bitwise,
dX up to the (already non-deterministic) order of its global FP32 atomics."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan
from sparse_pooling.kernels.cuda_ops import _epilogue
from test_dw_indexed import SMOOTH, _grads, _inputs

BASE = dict(block_m=32, cuda_dx_epilogue="generic", cuda_dw_epilogue="generic")
SHAPES = [(1024, 20000, 8, 40), (64, 20003, 8, 64), (1024, 9000, 4, 8), (256, 70000, 16, 136)]


def _variant(**kw):
    return dict(BASE, **kw)


VARIANTS = {
    "fast_epilogue": _variant(cuda_dx_epilogue="fast", cuda_dw_epilogue="fast"),
    "fast_dx_only": _variant(cuda_dx_epilogue="fast"),
    "fast_dw_only": _variant(cuda_dw_epilogue="fast"),
    "mask_epilogue": _variant(cuda_dx_epilogue="mask", cuda_dw_epilogue="mask"),
    "mask_dx_only": _variant(cuda_dx_epilogue="mask"),
    "mask_dw_only": _variant(cuda_dw_epilogue="mask"),
}


def _compare(lay, x, dy, base_kw, new_kw):
    lay.plan = PhysicalPlan(**base_kw)
    dx0, dw0, db0, _ = _grads(lay, x, dy)
    lay.plan = PhysicalPlan(**new_kw)
    dx1, dw1, db1, info = _grads(lay, x, dy)
    assert torch.equal(dw1, dw0) and torch.equal(db1, db0), "dW/db must be bitwise identical"
    assert rel_err(dx1, dx0) < 1e-6  # global atomics: order only
    return info


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pooling", SMOOTH, ids=str)
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("D,H,G,B", SHAPES)
@pytest.mark.parametrize("C", [8, 64])
def test_variant_matches_baseline(variant, pooling, precision, D, H, G, B, C):
    lay, _, _ = make_layer(D, H, G, C, pooling, "auto", "cuda", plan=PhysicalPlan(block_m=32))
    x, dy = _inputs(B, D, G)
    kw = dict(VARIANTS[variant], grad_precision=precision)
    info = _compare(lay, x, dy, dict(BASE, grad_precision=precision), kw)
    lv = lay.cfg.levels[0]
    want_dx = _epilogue(kw["cuda_dx_epilogue"], lv)
    want_dw = _epilogue(kw["cuda_dw_epilogue"], lv) if C == 64 else "generic"
    assert (info["dx_epilogue"], info["dw_epilogue"]) == (want_dx, want_dw)
    # which operators actually get a specialised epilogue
    special = {"mean": True, "logmeanexp": lv.activation in ("relu", "identity"),
               "power_mean": float(lv.p) in (1.0, 2.0, 4.0, 8.0), "softmax_weighted": False}[lv.type]
    if kw["cuda_dx_epilogue"] != "generic":
        assert (want_dx != "generic") == special


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pooling", ["mean", {"type": "mean", "activation": "identity"}], ids=str)
@pytest.mark.parametrize("precision,tol", [("split", 2e-5), ("bf16", 5e-3)])
def test_variant_against_reference(variant, pooling, precision, tol):
    lay, w, b = make_layer(1024, 20000, 8, 64, pooling, "auto", "cuda",
                           plan=PhysicalPlan(**dict(VARIANTS[variant], grad_precision=precision)))
    r = run_both(lay, w, b, 40, 1024)
    for k, (a, e) in r.items():  # same tolerances as test_backward.py
        assert rel_err(a, e) < (2e-5 if k in ("y", "db") else tol), k


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_relu_boundary_and_signed_zero(variant):
    """Pre-activations exactly 0 (x = 0 and b = 0 for half the tokens) and
    upstream gradients of both signs: ReLU'(0) = 0 in every path, -0 vs +0
    dZ must not change any accumulated bit."""
    D, B = 1024, 48
    lay, _, _ = make_layer(D, 20000, 8, 64, "mean", "auto", "cuda", plan=PhysicalPlan(block_m=32), bias_std=0.0)
    x, dy = _inputs(B, D, 8)
    x[: B // 2] = 0.0
    dy[::3] = -dy[::3].abs()
    dy[1::3] = 0.0
    _compare(lay, x, dy, BASE, VARIANTS[variant])


EXACT_POOLS = ["mean", {"type": "mean", "activation": "identity"}, {"type": "logmeanexp", "temperature": 0.5},
               {"type": "logmeanexp", "temperature": 0.5, "activation": "identity"}, {"type": "power_mean", "p": 2},
               {"type": "power_mean", "p": 4}]


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pooling", EXACT_POOLS, ids=str)
def test_extreme_magnitudes(variant, pooling):
    """Large and tiny upstream gradients through the split path; c1 = dy / n becomes
    subnormal, which the generic path flushes (FTZ) and every variant must too."""
    D, B = 1024, 40
    lay, _, _ = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=PhysicalPlan(block_m=32))
    x, dy = _inputs(B, D, 8)
    scale = torch.tensor([1e30, 1e-30, 1e-36, 3e-38, 1.0, -1e30, -1e-36, 65504.0], device="cuda")  # c1 = dy / n: subnormal for 1e-36
    dy = dy * scale.repeat(B // 8)[:, None]
    _compare(lay, x, dy, BASE, VARIANTS[variant])


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_colliding_indices_and_accumulation(variant):
    D, B = 1024, 72
    lay, _, _ = make_layer(D, 20000, 8, 64, "mean", "auto", "cuda", plan=PhysicalPlan(block_m=32))
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    feats = torch.randperm(D, device="cuda", generator=g)[:12]
    lay.indices.copy_(feats[torch.randint(0, 12, lay.indices.shape, device="cuda", generator=g)].to(lay.indices.dtype))
    x, dy = _inputs(B, D, 8)
    _compare(lay, x, dy, BASE, VARIANTS[variant])
    # accumulation over two backward calls into existing .grad
    for kw in (BASE, VARIANTS[variant]):
        lay.plan = PhysicalPlan(**kw)
        lay.weight.grad = lay.bias.grad = None
        for _ in range(2):
            xx = x.detach().clone().requires_grad_(True)
            (lay(xx) * dy).sum().backward()
        if kw is BASE:
            w2, b2 = lay.weight.grad.clone(), lay.bias.grad.clone()
    assert torch.equal(lay.weight.grad, w2) and torch.equal(lay.bias.grad, b2)


@requires_cuda_backend
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pooling", EXACT_POOLS, ids=str)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 3e38])
def test_nonfinite_upstream_gradient(variant, pooling, bad):
    """A non-finite upstream gradient for one token: the generic path spreads NaN through
    c1 * act'(z) (inf * 0 = NaN); every variant must produce the same bits."""
    D, B = 1024, 40
    lay, _, _ = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=PhysicalPlan(block_m=32))
    x, dy = _inputs(B, D, 8)
    dy[5, 3] = bad
    lay.plan = PhysicalPlan(**BASE)
    _, dw0, db0, _ = _grads(lay, x, dy)
    lay.plan = PhysicalPlan(**VARIANTS[variant])
    _, dw1, db1, _ = _grads(lay, x, dy)
    same = lambda a, b: torch.equal(torch.nan_to_num(a, 1.0, 2.0, 3.0), torch.nan_to_num(b, 1.0, 2.0, 3.0)) \
        and torch.equal(a.isnan(), b.isnan())  # noqa: E731
    assert same(dw1, dw0) and same(db1, db0)


@requires_cuda_backend
def test_explicit_indexed_fallback_is_reported():
    """An explicit cuda_dw_staging='indexed' that cannot be honoured warns and records why."""
    lay, w, b = make_layer(1024, 20000, 8, 8, "mean", "auto", "cuda",
                           plan=PhysicalPlan(block_m=32, cuda_dw_staging="indexed"))
    from sparse_pooling.kernels import cuda_ops
    cuda_ops._warned.clear()
    with pytest.warns(UserWarning, match="C = 64 only"):
        run_both(lay, w, b, 40, 1024)
    assert lay.last_info["dw_staging"] == "full" and "C = 64" in lay.last_info["dw_staging_fallback"]
    # "auto" falls back silently (that is its job) and leaves no fallback note
    lay.plan = PhysicalPlan(block_m=32)
    run_both(lay, w, b, 40, 1024)
    assert "dw_staging_fallback" not in lay.last_info
