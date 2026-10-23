"""Input-supergroup connectivity (supergroups.py) and the grouped kernels (sparse_pool_sg.cuh).

Strict comparators: the grouped dW / db run the arithmetic of k_dw_idx in the same token
order, so they must be bitwise identical to the generic kernels on the same graph (both
use the same dZ epilogue values: generic / fast are bitwise equal by construction).
Forward (FP32 atomics per (token, bucket)) and dX (FP32 atomics per (feature, token) per
CTA) change the FP32 summation order: compared to the generic kernels with a 1e-6
relative bound and to the PyTorch reference with the existing tolerances (not weakened).
"""
import io
import math

import pytest
import torch

from conftest import rel_err, requires_cuda_backend, rtx3090_policy
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.layout import SparseLayout
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.reference import pytorch_reference as ref
from sparse_pooling.supergroups import (ConnectivityPlan, SupergroupGraph, generate_indices, matched_c8_from_c64,
                                        validate)

GEN, GROUP = "generic", "grouped"


def sg_layer(D, H, G, C, S, R, pooling="mean", plan=None, seed=0, graph_seed=3, gen="balanced_permutation",
             fp32=True):
    lay = SparsePoolingLayer(D, H, G, shared_neurons=C, pooling=pooling, device="cuda", seed=seed, bias_std=0.2,
                             plan=plan or PhysicalPlan(), connectivity=ConnectivityPlan(S, R, gen, graph_seed))
    if fp32:  # FP32 copies of BF16-exact values (as conftest.make_layer)
        lay.weight = torch.nn.Parameter(lay.weight.detach().float())
        lay.bias = torch.nn.Parameter(lay.bias.detach().float())
    return lay


def plan_for(backend, precision="split", **kw):
    return PhysicalPlan(cuda_input_supergroup_backend=backend, grad_precision=precision, **kw)


def inputs(B, D, G, seed=1):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(B, G, device="cuda", generator=g)
    return x, dy


def run(lay, x, dy):
    lay.weight.grad = lay.bias.grad = None
    xx = x.detach().clone().requires_grad_(True)
    y = lay(xx)
    (y * dy).sum().backward()
    return y.detach(), xx.grad.clone(), lay.weight.grad.clone(), lay.bias.grad.clone(), dict(lay.last_info)


def reference(lay, x, dy, dtype=torch.float32, flip=None):
    """Reference y, dX, dW, db (Mean(ReLU / identity) layers).  flip: boolean [B, G, slots]
    elements whose ReLU decision is replaced by the opposite one (value and derivative)."""
    w = lay.weight.detach().to(dtype).clone().requires_grad_(True)
    b = lay.bias.detach().to(dtype).clone().requires_grad_(True)
    xr = x.to(dtype).clone().requires_grad_(True)
    if flip is None:
        yr = ref.reference_forward(xr, lay.indices, w, b, lay.layout, lay.cfg, {})
    else:
        assert lay.cfg.levels[0].type == "mean" and lay.cfg.levels[0].activation == "relu"
        z = ref.hidden_preactivations(xr, lay.indices, w, b, lay.layout)
        a = torch.where((z > 0) ^ flip, z, torch.zeros_like(z))
        mask = lay.layout.valid_mask(z.device)
        yr = (a * mask).sum(-1) / mask.sum(-1).to(dtype)
    (yr * dy.to(dtype)).sum().backward()
    return yr.detach(), xr.grad, w.grad, b.grad


# The one known reference-tolerance exceedance of this file (benchmarks/relu_boundary_case.py,
# results/c8_dx_scatter/relu_boundary_case.json): for (token 206, output 30, slot 923) the exact
# pre-activation is z = +2^-28 (FP64 and the FP32 reference agree: active), but the FP32 tensor-core
# accumulation of BOTH the generic and the grouped kernels yields z <= 0 (inactive).  This single
# decision moves dX / dW / db by ~3e-5 / 8e-5 / 9e-5 relative; with it adopted every kernel is
# within ~2.4e-6 of the FP64 reference.  No other case may exceed the tolerances.
BOUNDARY_CASE = dict(C=8, S=64, R=16, shape=(1024, 300_000, 64, 256), precision="split")
BOUNDARY_ELEMENT = (206, 30, 923)
BOUNDARY_Z = 2.0 ** -28


def boundary_reference(lay, x, dy):
    """FP64 reference with the kernels' decision at the known boundary element (asserting
    that element is the documented one)."""
    t, g, s = BOUNDARY_ELEMENT
    z = ref.hidden_preactivations(x.double(), lay.indices, lay.weight.detach().double(), lay.bias.detach().double(),
                                  lay.layout)
    assert z[t, g, s].item() == BOUNDARY_Z, z[t, g, s].item()
    flip = torch.zeros(z.shape, dtype=torch.bool, device=z.device)
    flip[t, g, s] = True
    return reference(lay, x, dy, torch.float64, flip)


def check_kernels(info, want):
    assert (info["forward_kernel"], info["dx_kernel"], info["dw_kernel"]) == want, info


# --------------------------------------------------------------------------------------
# generator / validator
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("S,R", [(8, 2), (16, 4), (32, 8), (64, 16), (128, 32), (256, 64), (1024, 1)])
@pytest.mark.parametrize("H,G", [(20003, 8), (1_000_000, 1024), (5000, 64)])
def test_generator_validates(C, S, R, H, G):
    L = SparseLayout(1024, H, G, 8, C, True, packed=True).canonical
    idx, graph, _ = generate_indices(L, ConnectivityPlan(S, R, seed=5), "cuda")
    rep = validate(L, idx, graph, pair_coverage=False)  # raises on any structural violation
    assert rep["valid_neurons_check"] == H
    assert rep["Q"] == R * 1024 // S
    # per-supergroup unit totals differ by at most one
    assert rep["per_supergroup"]["units_max"] - rep["per_supergroup"]["units_min"] <= 1


def test_generator_deterministic_and_seeded():
    L = SparseLayout(1024, 100_003, 64, 8, 8, True)
    a = generate_indices(L, ConnectivityPlan(64, 16, seed=1), "cuda")
    b = generate_indices(L, ConnectivityPlan(64, 16, seed=1), "cuda")
    c = generate_indices(L, ConnectivityPlan(64, 16, seed=2), "cuda")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1].buckets, b[1].buckets) and torch.equal(a[1].feat, b[1].feat)
    assert not torch.equal(a[0], c[0])


def test_independent_subsets_generation():
    L = SparseLayout(1024, 200_000, 64, 8, 64, True)
    idx, graph, _ = generate_indices(L, ConnectivityPlan(32, 8, "independent_subsets", 0), "cuda")
    validate(L, idx, graph)


def test_overlapping_sets_and_duplicates():
    # D = 64, S = 32, R = 16: only C(64, 32) / 2 partitions -> sets overlap heavily
    L = SparseLayout(64, 30_000, 16, 8, 8, True)
    idx, graph, _ = generate_indices(L, ConnectivityPlan(32, 16, seed=0), "cuda")
    rep = validate(L, idx, graph)
    assert rep["set_pair_coverage"] > 0.99


def test_balanced_local_rounds_use_each_feature_once():
    # one bucket per (q, g) here; each complete local round of q uses every local feature once
    L = SparseLayout(1024, 1024 * 64 * 64, 1024, 8, 64, True)
    idx, graph, _ = generate_indices(L, ConnectivityPlan(64, 16, seed=0), "cuda")
    rep = validate(L, idx, graph, pair_coverage=False)
    # perfectly balanced: every feature has the same valid edge degree
    assert rep["feature_degree"]["min"] == rep["feature_degree"]["max"]


def test_s8_records_are_the_sets():
    L = SparseLayout(1024, 50_000, 64, 8, 64, True)
    idx, graph, _ = generate_indices(L, ConnectivityPlan(8, 2, seed=0), "cuda")
    rep = validate(L, idx, graph)
    assert rep["distinct_records"] <= graph.Q  # S = 8: one record per supergroup (labelled diagnostic)


def test_matched_c8_conversion_same_function():
    D, H, G, B = 256, 70_001, 16, 64
    lay64 = sg_layer(D, H, G, 64, 32, 4, plan=plan_for(GEN))
    L = lay64.layout
    idx, w, b = L.to_canonical(lay64.indices, lay64.weight.detach(), lay64.bias.detach())
    graph = SupergroupGraph(lay64.connectivity, lay64.sg_feat, lay64.sg_buckets)
    L8, idx8, w8, b8, g8 = matched_c8_from_c64(L.canonical, idx, w, b, graph)
    validate(L8.canonical, idx8, g8)
    x, _ = inputs(B, D, G)
    z64 = ref.hidden_preactivations(x, idx, w, b, L.canonical)
    z8 = ref.hidden_preactivations(x, idx8, w8, b8, L8.canonical)
    n = z8.shape[-1]
    valid = L8.canonical.valid_mask("cuda")
    assert torch.equal(z64[..., :n][:, valid], z8[:, valid])


# --------------------------------------------------------------------------------------
# grouped vs generic on the same graph
# --------------------------------------------------------------------------------------
SHAPES = [(1024, 20_000, 8, 40), (256, 70_001, 16, 136), (1024, 300_000, 64, 256), (1024, 5_000, 64, 8)]


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("S,R", [(8, 2), (16, 4), (32, 8), (64, 16), (128, 32), (256, 64)])
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_grouped_matches_generic(C, S, R, shape, precision):
    D, H, G, B = shape
    if S > D:
        pytest.skip("S > D")
    R = 1 if S == D else max(1, min(R, 4 * D // S))
    lay = sg_layer(D, H, G, C, S, R, plan=plan_for(GEN, precision))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, ig = run(lay, x, dy)
    check_kernels(ig, ("generic", "generic", "generic"))
    lay.plan = plan_for(GROUP, precision)
    ys, dxs, dws, dbs, i2 = run(lay, x, dy)
    check_kernels(i2, ("grouped", "grouped", "grouped"))
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6
    tol = 2e-5 if precision == "split" else 5e-3
    yr, dxr, dwr, dbr = reference(lay, x, dy)
    assert rel_err(ys, yr) < 2e-5
    if dict(C=C, S=S, R=R, shape=shape, precision=precision) == BOUNDARY_CASE:
        _, dxr, dwr, dbr = boundary_reference(lay, x, dy)  # the documented decision, nothing else
    for got, gen, want in ((dxs, dxg, dxr), (dws, dwg, dwr), (dbs, dbg, dbr)):
        assert rel_err(got, want) < tol and rel_err(gen, want) < tol, (rel_err(got, want), rel_err(gen, want))


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("ftok", [16, 32, 64, 128])
@pytest.mark.parametrize("dx", [(16, 16, 1), (32, 8, 2), (64, 8, 8), (128, 16, 8), (64, 16, 16)])
@pytest.mark.parametrize("dwcfg", [0, 1, 2, 3])
def test_grouped_tilings(C, ftok, dx, dwcfg):
    D, H, G, B = 1024, 60_000, 16, 200  # token tails around 16/32/64/128
    lay = sg_layer(D, H, G, C, 64, 16, plan=plan_for(GEN))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP, cuda_grouped_fwd_tokens=ftok, cuda_grouped_dx_tokens=dx[0], cuda_grouped_dx_warps=dx[1],
                        cuda_grouped_dx_reduction=dx[2], cuda_grouped_dw_cfg=dwcfg)
    ys, dxs, dws, dbs, info = run(lay, x, dy)
    assert info["grouped_dx"]["tokens"] == dx[0] and info["grouped_dw"]["cfg"] == dwcfg
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("B", [8, 16, 24, 32, 48, 64, 72, 120, 128, 136, 256, 264])
def test_token_tails(C, B):
    D, H, G = 512, 33_333, 8
    lay = sg_layer(D, H, G, C, 32, 8, plan=plan_for(GEN))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP)
    ys, dxs, dws, dbs, _ = run(lay, x, dy)
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("H,G", [(1000, 64), (64 * 8 + 3, 8), (999_999, 1000), (100, 100)])
def test_tiny_and_empty_buckets(C, H, G):
    # many empty (output, supergroup) intersections, one-unit buckets, partial C blocks
    D, B = 1024, 64
    lay = sg_layer(D, H, G, C, 16, 16, plan=plan_for(GEN))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP)
    ys, dxs, dws, dbs, _ = run(lay, x, dy)
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6
    yr, dxr, dwr, _ = reference(lay, x, dy)
    assert rel_err(ys, yr) < 2e-5 and rel_err(dxs, dxr) < 2e-5 and rel_err(dws, dwr) < 2e-5


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
def test_full_width_control_s_equals_d(C):
    D, H, G, B = 1024, 50_000, 16, 64
    lay = sg_layer(D, H, G, C, 1024, 1, plan=plan_for(GEN))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP, cuda_grouped_dx_tokens=16, cuda_grouped_dx_warps=16, cuda_grouped_dx_reduction=16,
                        cuda_grouped_dw_cfg=0)
    ys, dxs, dws, dbs, info = run(lay, x, dy)
    check_kernels(info, ("grouped", "grouped", "grouped"))
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_mean_identity(C, precision):
    D, H, G, B = 256, 40_000, 8, 64
    lay = sg_layer(D, H, G, C, 32, 8, pooling={"type": "mean", "activation": "identity"}, plan=plan_for(GEN, precision))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP, precision)
    ys, dxs, dws, dbs, info = run(lay, x, dy)
    check_kernels(info, ("grouped", "grouped", "grouped"))
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(ys, yg) < 1e-6 and rel_err(dxs, dxg) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("pooling", [{"type": "logmeanexp", "temperature": 0.5}, {"type": "power_mean", "p": 2},
                                     {"type": "softmax_weighted", "beta": 1.0}, "max"], ids=str)
def test_unsupported_pooling_falls_back_visibly(pooling):
    D, H, G, B = 256, 20_000, 8, 64
    lay = sg_layer(D, H, G, 64, 32, 8, pooling=pooling, plan=plan_for("auto"))
    x, dy = inputs(B, D, G)
    _, _, _, _, info = run(lay, x, dy)
    assert info["forward_kernel"] == "generic" and "fwd" in info["grouped_fallback"]
    lay.plan = plan_for(GROUP)
    with pytest.raises(ValueError, match="grouped fwd requested but unavailable"):
        run(lay, x, dy)


@requires_cuda_backend
def test_unsupported_shapes_fall_back_visibly():
    lay = sg_layer(256, 20_000, 8, 16, 32, 8, plan=plan_for("auto"))  # C = 16
    x, dy = inputs(64, 256, 8)
    _, _, _, _, info = run(lay, x, dy)
    assert "C = 8 and 64" in info["grouped_fallback"]["fwd"]
    lay = sg_layer(256, 20_000, 8, 64, 32, 8, plan=plan_for("auto"))
    x, dy = inputs(36, 256, 8)  # B % 8 != 0: no feature-major operand
    _, _, _, _, info = run(lay, x, dy)
    assert info["forward_kernel"] == "generic" and "feature-major" in info["grouped_fallback"]["dx"]
    lay.plan = plan_for(GROUP, cuda_grouped_dx_tokens=128, cuda_grouped_dx_warps=16, cuda_grouped_dx_reduction=1)
    with pytest.raises(ValueError, match="shared memory"):
        run(lay, *inputs(64, 256, 8))
    lay.plan = plan_for(GROUP, cuda_dx_epilogue="fast")
    with pytest.raises(ValueError, match="generic and mask"):
        run(lay, *inputs(64, 256, 8))


def test_mode_a_rejected():
    with pytest.raises(ValueError, match="Mode B"):
        SparsePoolingLayer(256, 2000, 8, shared_neurons=None, pooling="mean", device="cuda",
                           connectivity=ConnectivityPlan(32, 8))


@requires_cuda_backend
@pytest.mark.parametrize("phases", [("grouped", "generic", "generic"), ("generic", "grouped", "generic"),
                                    ("generic", "generic", "grouped"), ("grouped", "grouped", "generic")])
@pytest.mark.parametrize("C", [8, 64])
def test_per_phase_selection(phases, C):
    D, H, G, B = 1024, 50_000, 16, 128
    lay = sg_layer(D, H, G, C, 64, 16, plan=plan_for(GEN))
    x, dy = inputs(B, D, G)
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = PhysicalPlan(cuda_grouped_fwd=phases[0], cuda_grouped_dx=phases[1], cuda_grouped_dw=phases[2])
    ys, dxs, dws, dbs, info = run(lay, x, dy)
    check_kernels(info, phases)
    if phases[0] == "generic":
        assert torch.equal(ys, yg)
    assert torch.equal(dws, dwg) and torch.equal(dbs, dbg)
    assert rel_err(dxs, dxg) < 1e-6


@rtx3090_policy
@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
def test_auto_selection_reports_kernels(C):
    lay = sg_layer(1024, 50_000, 16, C, 64, 16, plan=plan_for("auto"))
    _, _, _, _, info = run(lay, *inputs(128, 1024, 16))
    from sparse_pooling.kernels import cuda_ops
    auto = cuda_ops.sg_auto(C, 64)
    for i, key in enumerate(("forward_kernel", "dx_kernel", "dw_kernel")):
        assert info[key] == ("grouped" if auto[i] is not None else "generic")


# --------------------------------------------------------------------------------------
# numerics: ReLU boundary, extremes, non-finite upstream gradients
# --------------------------------------------------------------------------------------
@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("case", ["zero_bias_zero_x", "huge", "tiny", "subnormal", "inf", "nan", "neg_zero"])
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_extreme_values(C, case, precision):
    D, H, G, B = 256, 20_000, 8, 64
    lay = sg_layer(D, H, G, C, 32, 8, plan=plan_for(GEN, precision))
    x, dy = inputs(B, D, G)
    if case == "zero_bias_zero_x":  # z == 0 exactly at the ReLU boundary
        with torch.no_grad():
            lay.bias.zero_()
        x[:, : D // 2] = 0.0
        x[:, D // 2:] = 0.0
        x[::3] = inputs(B, D, G)[0][::3]
    elif case == "huge":
        dy = dy * 1e30
    elif case == "tiny":
        dy = dy * 3e-38
    elif case == "subnormal":
        dy = dy * 1e-36  # c1 = dy / n subnormal -> flushed exactly as the generic FTZ path
    elif case == "inf":
        dy[3, 2] = float("inf")
        dy[5, 1] = -float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    elif case == "neg_zero":
        dy = torch.where(dy > 0, dy, torch.full_like(dy, -0.0))
    yg, dxg, dwg, dbg, _ = run(lay, x, dy)
    lay.plan = plan_for(GROUP, precision)
    ys, dxs, dws, dbs, _ = run(lay, x, dy)
    # bitwise incl. NaN patterns, signed zeros and FTZ
    assert torch.equal(dws.view(torch.int32), dwg.view(torch.int32))
    assert torch.equal(dbs.view(torch.int32), dbg.view(torch.int32))
    assert torch.equal(torch.isnan(dxs), torch.isnan(dxg)) and torch.equal(torch.isinf(dxs), torch.isinf(dxg))
    fin = torch.isfinite(dxg)
    assert rel_err(dxs[fin], dxg[fin]) < 1e-6
    assert rel_err(ys, yg) < 1e-6


# --------------------------------------------------------------------------------------
# training: gradient accumulation, AdamW, checkpoints
# --------------------------------------------------------------------------------------
@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
def test_main_grad_accumulates_over_backward_calls(C):
    D, H, G, B = 1024, 60_000, 16, 64
    out = {}
    for be in (GEN, GROUP):
        lay = sg_layer(D, H, G, C, 64, 16, plan=plan_for(be), fp32=False)
        lay.enable_main_grad()
        for s in (1, 2, 3):
            x, dy = inputs(B, D, G, seed=s)
            (lay(x.bfloat16()).float() * dy).sum().backward()
        out[be] = (lay.weight.main_grad.clone(), lay.bias.main_grad.clone())
    assert torch.equal(out[GEN][0], out[GROUP][0]) and torch.equal(out[GEN][1], out[GROUP][1])


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
def test_adamw_trajectory(C):
    D, H, G, B = 1024, 60_000, 16, 64
    res = {}
    for be in (GEN, GROUP):
        lay = sg_layer(D, H, G, C, 64, 16, plan=plan_for(be), fp32=False)
        lay.enable_main_grad()
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, moments="bf16", stochastic_rounding=False)
        for s in range(4):
            x, dy = inputs(B, D, G, seed=10 + s)
            (lay(x.bfloat16()).float() * dy).sum().backward()
            opt.step()
        res[be] = (lay.weight.detach().float(), lay.bias.detach().float())
    # forward rounding order differs (atomics) -> dy identical here (fixed), dW bitwise per step;
    # the only difference source is none: trajectories must agree exactly
    assert torch.equal(res[GEN][0], res[GROUP][0]) and torch.equal(res[GEN][1], res[GROUP][1])


@requires_cuda_backend
def test_checkpoint_round_trip(tmp_path):
    D, H, G, B = 1024, 60_000, 16, 64
    lay = sg_layer(D, H, G, 8, 64, 16, plan=plan_for(GROUP), fp32=False)
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-3, stochastic_rounding=False)  # the SR seed counter is not checkpointed (pre-existing)
    x, dy = inputs(B, D, G)
    (lay(x.bfloat16()).float() * dy).sum().backward()
    opt.step()
    buf = io.BytesIO()
    torch.save({"model": lay.state_dict(), "opt": opt.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    assert {"sg_feat", "sg_buckets", "sg_plan", "indices"} <= set(ck["model"])
    lay2 = sg_layer(D, H, G, 8, 64, 16, plan=plan_for(GROUP), fp32=False, seed=99)  # different init
    lay2.load_state_dict(ck["model"])
    lay2.enable_main_grad()
    opt2 = FusedAdamW([lay2.weight, lay2.bias], lr=1e-3, stochastic_rounding=False)
    opt2.load_state_dict(ck["opt"])
    for a, b in ((lay, lay2),):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.indices, b.indices)
    # one more identical step on both
    for m, o in ((lay, opt), (lay2, opt2)):
        (m(x.bfloat16()).float() * dy).sum().backward()
        o.step()
    assert torch.equal(lay.weight, lay2.weight) and torch.equal(lay.bias, lay2.bias)
    # mismatched connectivity is refused; unrestricted checkpoints keep loading as before
    other = sg_layer(D, H, G, 8, 32, 8, fp32=False)
    with pytest.raises(ValueError, match="connectivity"):
        other.load_state_dict(ck["model"])
    plain = SparsePoolingLayer(D, H, G, shared_neurons=8, pooling="mean", device="cuda")
    with pytest.raises(ValueError, match="disagree"):
        plain.load_state_dict(ck["model"])
    plain2 = SparsePoolingLayer(D, H, G, shared_neurons=8, pooling="mean", device="cuda", seed=4)
    plain2.load_state_dict(plain.state_dict())
    assert torch.equal(plain2.indices, plain.indices)


def test_same_weights_as_unrestricted_layer():
    """The connectivity changes only the indices: weights / bias equal the unrestricted
    layer's for the same seed (architectural reference uses identical parameters)."""
    a = SparsePoolingLayer(256, 20_000, 8, shared_neurons=64, pooling="mean", device="cuda", seed=7)
    b = SparsePoolingLayer(256, 20_000, 8, shared_neurons=64, pooling="mean", device="cuda", seed=7,
                           connectivity=ConnectivityPlan(32, 8))
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert not torch.equal(a.indices, b.indices)
