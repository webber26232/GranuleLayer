"""Forward Mean epilogue variants of branch mma-forward-epilogue-optimization.

"chunk4" only regroups the projection MMAs of process_block (4 n8 tiles per epilogue
chunk instead of 2); every pooling state receives the same values in the same order,
so forward outputs and all gradients must be bitwise identical to "baseline"
(dX: up to the order of its global FP32 atomics, as in every backward test)."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.optim import FusedAdamW
from test_dw_indexed import _inputs

MEANS = ["mean", {"type": "mean", "activation": "identity"}]
SHAPES = [(1024, 20000, 8, 40), (64, 20003, 8, 64), (1024, 9000, 4, 8), (256, 70001, 16, 136), (1024, 3001, 8, 33)]


def _plan(fe, block_m=32, precision="split"):
    return PhysicalPlan(block_m=block_m, grad_precision=precision, cuda_fwd_mean_epilogue=fe)


def _run(lay, x, dy):
    lay.weight.grad = lay.bias.grad = None
    xx = x.detach().clone().requires_grad_(True)
    y = lay(xx)
    (y * dy).sum().backward()
    return y.detach().clone(), xx.grad.clone(), lay.weight.grad.clone(), lay.bias.grad.clone(), dict(lay.last_info)


def _same(a, b):  # bitwise, NaN positions included
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(torch.nan_to_num(a, 1.0, 2.0, 3.0),
                                                              torch.nan_to_num(b, 1.0, 2.0, 3.0))


def _compare(lay, x, dy, block_m=32, precision="split"):
    lay.plan = _plan("baseline", block_m, precision)
    y0, dx0, dw0, db0, i0 = _run(lay, x, dy)
    lay.plan = _plan("chunk4", block_m, precision)
    y1, dx1, dw1, db1, i1 = _run(lay, x, dy)
    assert i0["fwd_mean_epilogue"] == "baseline" and i1["fwd_mean_epilogue"] == "chunk4"
    assert _same(y1, y0), "forward output must be bitwise identical"
    assert _same(dw1, dw0) and _same(db1, db0), "dW/db must be bitwise identical"
    if torch.isfinite(dx0).all():
        assert rel_err(dx1, dx0) < 1e-6  # global atomics: order only
    return y1


@requires_cuda_backend
@pytest.mark.parametrize("pooling", MEANS, ids=str)
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("block_m", [16, 32])
@pytest.mark.parametrize("D,H,G,B", SHAPES)
@pytest.mark.parametrize("C", [32, 64])
def test_chunk4_matches_baseline_bitwise(pooling, precision, block_m, D, H, G, B, C):
    lay, _, _ = make_layer(D, H, G, C, pooling, "auto", "cuda", plan=_plan("baseline", block_m))
    x, dy = _inputs(B, D, G)
    _compare(lay, x, dy, block_m, precision)


@requires_cuda_backend
@pytest.mark.parametrize("pooling", MEANS, ids=str)
@pytest.mark.parametrize("precision,tol", [("split", 2e-5), ("bf16", 5e-3)])
def test_chunk4_against_reference(pooling, precision, tol):
    lay, w, b = make_layer(1024, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("chunk4", precision=precision))
    r = run_both(lay, w, b, 40, 1024)
    assert lay.last_info["fwd_mean_epilogue"] == "chunk4"
    for k, (a, e) in r.items():  # same tolerances as test_backward.py
        assert rel_err(a, e) < (2e-5 if k in ("y", "db") else tol), k


@requires_cuda_backend
@pytest.mark.parametrize("pooling", MEANS, ids=str)
def test_relu_boundary_signed_zero_and_rounding(pooling):
    """Pre-activations exactly 0 (x = 0, bias 0), tiny inputs whose products are
    subnormal (flushed), and cancelling inputs that leave values a few ULP from 0."""
    D, B = 1024, 48
    lay, _, _ = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("baseline"), bias_std=0.0)
    x, dy = _inputs(B, D, 8)
    x[: B // 4] = 0.0
    x[B // 4: B // 2] *= 1e-38
    x[B // 2: 3 * B // 4] = torch.where(torch.arange(D, device="cuda") % 2 == 0, 1.0, -1.0) * 0.5
    _compare(lay, x, dy)
    lay2, _, _ = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("baseline"), bias_std=1e-3)
    _compare(lay2, x, dy)


@requires_cuda_backend
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 3e38])
@pytest.mark.parametrize("pooling", MEANS, ids=str)
def test_nonfinite_and_extreme_inputs(bad, pooling):
    D, B = 1024, 40
    lay, _, _ = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("baseline"))
    x, dy = _inputs(B, D, 8)
    x[3, :: 7] = bad
    x[11] *= 1e30
    _compare(lay, x, dy)


@requires_cuda_backend
def test_colliding_indices_and_accumulation():
    D, B = 1024, 72
    lay, _, _ = make_layer(D, 20000, 8, 64, "mean", "auto", "cuda", plan=_plan("baseline"))
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    feats = torch.randperm(D, device="cuda", generator=g)[:12]
    lay.indices.copy_(feats[torch.randint(0, 12, lay.indices.shape, device="cuda", generator=g)].to(lay.indices.dtype))
    x, dy = _inputs(B, D, 8)
    _compare(lay, x, dy)
    grads = {}
    for fe in ("baseline", "chunk4"):  # two backward calls accumulating into .grad
        lay.plan = _plan(fe)
        lay.weight.grad = lay.bias.grad = None
        for _ in range(2):
            (lay(x.detach().clone().requires_grad_(True)) * dy).sum().backward()
        grads[fe] = (lay.weight.grad.clone(), lay.bias.grad.clone())
    assert all(torch.equal(a, b) for a, b in zip(grads["baseline"], grads["chunk4"]))


@requires_cuda_backend
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_training_trajectory_identical(precision):
    """main_grad accumulation over micro-batches + fused AdamW steps: identical parameters."""
    def train(fe):
        lay, _, _ = make_layer(1024, 30000, 8, 64, "mean", "auto", "cuda", plan=_plan(fe, precision=precision))
        lay.weight = torch.nn.Parameter(lay.weight.detach().bfloat16())
        lay.bias = torch.nn.Parameter(lay.bias.detach().bfloat16())
        lay.enable_main_grad()
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, moments="bf16")
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        ys = []
        for _ in range(3):
            for _ in range(2):
                x = torch.randn(64, 1024, device="cuda", generator=g, dtype=torch.bfloat16)
                y = lay(x)
                ys.append(y.detach().clone())
                y.float().square().mean().backward()
            opt.step()
            lay.zero_main_grad()
        assert lay.last_info["fwd_mean_epilogue"] == fe
        return lay.weight.detach().clone(), lay.bias.detach().clone(), ys

    w0, b0, y0 = train("baseline")
    w1, b1, y1 = train("chunk4")
    assert torch.equal(w1, w0) and torch.equal(b1, b0)
    assert all(torch.equal(a, b) for a, b in zip(y0, y1))


OTHER = [("max", "max"), ({"type": "logmeanexp", "temperature": 0.5}, "logmeanexp"),
         ({"type": "softmax_weighted", "beta": 1.0}, "softmax_weighted"), ({"type": "power_mean", "p": 2}, "power_mean")]


@requires_cuda_backend
@pytest.mark.parametrize("pooling,name", OTHER, ids=str)
def test_other_poolings_unaffected_and_fallback_reported(pooling, name):
    lay, w, b = make_layer(1024, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("auto"))
    x, dy = _inputs(40, 1024, 8)
    y0, *_, info = _run(lay, x, dy)
    assert info["fwd_mean_epilogue"] == "baseline" and "fwd_mean_epilogue_fallback" not in info
    cuda_ops._warned.clear()
    lay.plan = _plan("chunk4")
    with pytest.warns(UserWarning, match="Mean pooling only"):
        y1, *_, info = _run(lay, x, dy)
    assert info["fwd_mean_epilogue"] == "baseline" and name in info["fwd_mean_epilogue_fallback"]
    assert torch.equal(y0, y1)


@requires_cuda_backend
@pytest.mark.parametrize("C,want", [(8, "baseline"), (16, "baseline"), (32, "baseline"), (64, "chunk4")])
def test_auto_selection_by_block(C, want):
    lay, w, b = make_layer(1024, 20000, 8, C, "mean", "auto", "cuda", plan=_plan("auto"))
    r = run_both(lay, w, b, 40, 1024)
    assert lay.last_info["fwd_mean_epilogue"] == want and "fwd_mean_epilogue_fallback" not in lay.last_info
    assert rel_err(*r["y"]) < 2e-5
    if C < 32:
        cuda_ops._warned.clear()
        lay.plan = _plan("chunk4")
        with pytest.warns(UserWarning, match="C >= 32"):
            run_both(lay, w, b, 40, 1024)
        assert "C = " in lay.last_info["fwd_mean_epilogue_fallback"]


@requires_cuda_backend
@pytest.mark.parametrize("fe", ["baseline", "chunk4", "auto"])
def test_mode_a_unchanged(fe):
    """Mode A (independent indices) never reaches the packed Mode-B kernels."""
    lay = SparsePoolingLayer(256, 3000, 8, shared_neurons=None, pooling="mean", device="cuda", seed=0,
                             plan=_plan(fe), backend="auto")
    x = torch.randn(40, 256, device="cuda")
    y = lay(x)
    lay.plan = _plan("baseline")
    assert torch.equal(lay(x), y)
