"""Indexed (warp-local) input staging of the native k_dw kernel.

The indexed kernel issues the same MMAs in the same order on the same operands
as the full-slab kernel, only from a different shared-memory slab, so dW/db
must be bitwise identical to it; both are also checked against the reference."""
import pytest
import torch

from conftest import make_layer, rel_err, requires_cuda_backend, run_both
from sparse_pooling import PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.optim import FusedAdamW

SMOOTH = [
    "mean",
    {"type": "mean", "activation": "identity"},
    {"type": "logmeanexp", "temperature": 0.5},
    {"type": "power_mean", "p": 2},
    {"type": "power_mean", "p": 3.5},
    {"type": "softmax_weighted", "beta": 1.0},
]


def _plan(staging, precision="split", cfg=-1):
    return PhysicalPlan(block_m=32, grad_precision=precision, cuda_dw_cfg=cfg, cuda_dw_staging=staging)


def _grads(lay, x, dy):
    """dX, dW, db of one backward through the current lay.plan (fresh grads)."""
    lay.weight.grad = lay.bias.grad = None
    xx = x.detach().clone().requires_grad_(True)
    (lay(xx) * dy).sum().backward()
    return xx.grad.clone(), lay.weight.grad.clone(), lay.bias.grad.clone(), lay.last_info


def _inputs(B, D, G, seed=1):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(B, G, device="cuda", generator=g)
    return x, dy


@pytest.fixture(params=[2, 3], ids=["2stage", "3stage"])
def stages(request, monkeypatch):
    monkeypatch.setattr(cuda_ops, "DW_INDEXED_STAGES", request.param)
    return request.param


@requires_cuda_backend
@pytest.mark.parametrize("pooling", SMOOTH, ids=str)
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("cfg", [0, 1, 2])
@pytest.mark.parametrize("D,H,G,B", [(1024, 20000, 8, 40), (64, 20003, 8, 64), (1024, 9000, 4, 8), (256, 70000, 16, 136)])
def test_indexed_matches_full_bitwise(pooling, precision, cfg, D, H, G, B, stages):
    lay, _, _ = make_layer(D, H, G, 64, pooling, "auto", "cuda", plan=_plan("full", precision, cfg))
    x, dy = _inputs(B, D, G)
    dx_f, dw_f, db_f, info_f = _grads(lay, x, dy)
    lay.plan = _plan("indexed", precision, cfg)
    dx_i, dw_i, db_i, info_i = _grads(lay, x, dy)
    assert info_f["dw_staging"] == "full" and info_i["dw_staging"] == "indexed"
    assert torch.equal(dw_i, dw_f) and torch.equal(db_i, db_f)
    # k_dx is shared; dX differs only by the order of its global float atomics
    assert rel_err(dx_i, dx_f) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("pooling", SMOOTH, ids=str)
@pytest.mark.parametrize("D,B", [(64, 64), (1024, 40)])
def test_indexed_against_reference_split(pooling, D, B, stages):
    lay, w, b = make_layer(D, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("indexed"))
    r = run_both(lay, w, b, B, D)
    assert lay.last_info["dw_staging"] == "indexed"
    for k, (a, e) in r.items():  # same tolerance as test_backward.test_cuda_backward_split
        assert rel_err(a, e) < 2e-5, k


@requires_cuda_backend
@pytest.mark.parametrize("pooling", ["mean", {"type": "logmeanexp", "temperature": 0.5}], ids=str)
def test_indexed_against_reference_bf16(pooling, stages):
    lay, w, b = make_layer(1024, 20000, 8, 64, pooling, "auto", "cuda", plan=_plan("indexed", "bf16"))
    r = run_both(lay, w, b, 40, 1024)
    assert lay.last_info["dw_staging"] == "indexed"
    # same tolerances as test_backward.test_cuda_backward_bf16_precision
    assert rel_err(*r["db"]) < 2e-5
    assert rel_err(*r["dx"]) < 5e-3
    assert rel_err(*r["dw"]) < 5e-3


@requires_cuda_backend
def test_indexed_repeated_and_colliding_indices(stages):
    """Every record drawn from 12 features: the same feature row appears in many
    records of a warp, and (with repeats inside a record) several times per record."""
    D, B = 1024, 72
    lay, w, b = make_layer(D, 20000, 8, 64, "mean", "auto", "cuda", plan=_plan("full"))
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    feats = torch.randperm(D, device="cuda", generator=g)[:12]
    sel = torch.randint(0, 12, lay.indices.shape, device="cuda", generator=g)
    lay.indices.copy_(feats[sel].to(lay.indices.dtype))
    x, dy = _inputs(B, D, 8)
    _, dw_f, db_f, _ = _grads(lay, x, dy)
    lay.plan = _plan("indexed")
    _, dw_i, db_i, _ = _grads(lay, x, dy)
    assert torch.equal(dw_i, dw_f) and torch.equal(db_i, db_f)
    lay.weight.grad = lay.bias.grad = None  # run_both accumulates into .grad
    r = run_both(lay, w, b, B, D)
    for k, (a, e) in r.items():
        assert rel_err(a, e) < 2e-5, k


@requires_cuda_backend
def test_indexed_is_deterministic():
    lay, _, _ = make_layer(1024, 50000, 8, 64, "mean", "auto", "cuda", plan=_plan("indexed"))
    x, dy = _inputs(256, 1024, 8)
    _, dw1, db1, _ = _grads(lay, x, dy)
    _, dw2, db2, _ = _grads(lay, x, dy)
    assert torch.equal(dw1, dw2) and torch.equal(db1, db2)


@requires_cuda_backend
@pytest.mark.parametrize("why,C,B,layout", [
    ("C=8", 8, 40, "feature_major"),
    ("token-major input", 64, 40, "feature_major_off"),
    ("B not a multiple of 8", 64, 37, "feature_major"),
])
def test_indexed_falls_back_to_full(why, C, B, layout):
    plan = _plan("indexed")
    if layout == "feature_major_off":
        plan = PhysicalPlan(block_m=32, input_layout="token_major", cuda_dw_staging="indexed")
    lay, w, b = make_layer(1024, 20000, 8, C, "mean", "auto", "cuda", plan=plan)
    r = run_both(lay, w, b, B, 1024)
    assert lay.last_info["dw_staging"] == "full", why
    for k, (a, e) in r.items():
        assert rel_err(a, e) < 2e-5, k


@requires_cuda_backend
def test_indexed_training_trajectory_matches_full():
    """main_grad accumulation over two backward calls + fused AdamW steps: the
    parameters after training are identical with either staging."""
    def train(staging):
        lay, _, _ = make_layer(1024, 30000, 8, 64, "mean", "auto", "cuda", plan=_plan(staging))
        lay.weight = torch.nn.Parameter(lay.weight.detach().bfloat16())
        lay.bias = torch.nn.Parameter(lay.bias.detach().bfloat16())
        lay.enable_main_grad()
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, moments="bf16")
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        for _ in range(2):
            for _ in range(2):  # gradient accumulation: two micro-batches per step
                x = torch.randn(64, 1024, device="cuda", generator=g, dtype=torch.bfloat16)
                lay(x).float().square().mean().backward()
            opt.step()
            lay.zero_main_grad()
        assert lay.last_info["dw_staging"] == staging
        return lay.weight.detach().clone(), lay.bias.detach().clone()

    wf, bf = train("full")
    wi, bi = train("indexed")
    assert torch.equal(wi, wf) and torch.equal(bi, bf)


@requires_cuda_backend
@pytest.mark.parametrize("pooling,cfg", [("mean", 3), ({"type": "logmeanexp", "temperature": 0.5}, 3),
                                         ({"type": "softmax_weighted", "beta": 1.0}, 1),
                                         ({"type": "power_mean", "p": 2}, 1)], ids=str)
def test_default_plan_uses_indexed_staging(pooling, cfg):
    lay, w, b = make_layer(1024, 20000, 8, 64, pooling, "auto", "cuda", plan=PhysicalPlan(block_m=32))
    r = run_both(lay, w, b, 40, 1024)
    assert lay.last_info["dw_staging"] == "indexed" and lay.last_info["dw_cfg"] == cfg
    for k, (a, e) in r.items():
        assert rel_err(a, e) < 2e-5, k
    lay.plan = PhysicalPlan(block_m=32, cuda_dw_staging="full")
    run_both(lay, w, b, 40, 1024)
    assert lay.last_info["dw_staging"] == "full"


@requires_cuda_backend
@pytest.mark.parametrize("pooling", SMOOTH, ids=str)
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("warps", [8, 16, 32])
def test_dx_warp_counts_against_reference(pooling, C, warps):
    """k_dx is NW-generic; the 32-warp CTA (default at C = 64) must match the reference."""
    lay, w, b = make_layer(1024, 20000, 8, C, pooling, "auto", "cuda",
                           plan=PhysicalPlan(block_m=32, cuda_dx_warps=warps))
    r = run_both(lay, w, b, 40, 1024)
    assert lay.last_info["dx_warps"] == warps
    for k, (a, e) in r.items():
        assert rel_err(a, e) < 2e-5, k


@requires_cuda_backend
def test_default_dx_warps():
    for C, want in ((64, 32), (8, 16)):
        lay, w, b = make_layer(1024, 20000, 8, C, "mean", "auto", "cuda", plan=PhysicalPlan(block_m=32))
        run_both(lay, w, b, 40, 1024)
        assert lay.last_info["dx_warps"] == want
