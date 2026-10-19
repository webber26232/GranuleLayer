"""Experimental fused shared-projection grouped backward (PhysicalPlan.cuda_grouped_backward,
kernels/cuda/sparse_pool_sg_bwdf.cuh): against the selected separate backward and the reference.

* dW / db BITWISE equal to the separate grouped dW (same projection bits, same mask decisions,
  same operands, token order and unit ownership) for every shape, tail and extreme;
* dX within the strict reference tolerance and within 1e-6 of the separate dX (different FP32
  summation order: one partial per 16-unit CTA);
* the recompute-preserving control (same schedule, second projection) gives the same dW / db;
* non-finite coefficients: the fused kernel exits and the guarded generic fallbacks produce the
  gradients (bitwise equal dW / db to the separate path's generic fallback: one path only);
* main-gradient accumulation over repeated backward calls; explicit-unsupported errors.
"""
import pytest
import torch

from conftest import rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from test_input_supergroups import inputs, reference, run, sg_layer


def plan(org="separate", cfg=0, precision="split", deterministic=False, **kw):
    kw = dict(cuda_input_supergroup_backend="grouped", grad_precision=precision, cuda_grouped_backward=org,
              cuda_grouped_fused_cfg=cfg, **kw)
    if deterministic:
        kw.update(cuda_grouped_dx_chunk_units=1 << 30)
    return PhysicalPlan(**kw)


SHAPES = [  # (D, H, G, S, R, B)
    (1024, 300_000, 64, 16, 4, 200),
    (1024, 300_000, 64, 32, 8, 136),
    (256, 20_011, 8, 16, 4, 64),     # ragged outputs (partial units / blocks)
    (256, 2_000, 64, 16, 4, 136),    # tiny / empty buckets
    (1024, 100_000, 16, 16, 16, 16),  # M smaller than one window, heavy overlap
    (512, 50_000, 32, 16, 1, 72),     # token tail 8 past a 64-token window
]


@requires_cuda_backend
@pytest.mark.parametrize("D,H,G,S,R,B", SHAPES)
@pytest.mark.parametrize("cfg", [0, 3, 6, 7, 8])
def test_fused_matches_separate(D, H, G, S, R, B, cfg):
    lay = sg_layer(D, H, G, 8, S, R, plan=plan("separate"))
    x, dy = inputs(B, D, G)
    ys, dxs, dws, dbs, is_ = run(lay, x, dy)
    for org in ("fused", "fused_recompute"):
        lay.plan = plan(org, cfg)
        try:
            yf, dxf, dwf, dbf, info = run(lay, x, dy)
        except ValueError as e:
            assert "shared memory" in str(e)
            pytest.skip(f"cfg {cfg} does not fit at S = {S}")
        assert info["backward_organization"] == org and info["grouped_fused"]["cfg"] == cfg
        assert info["grouped_fused"]["share"] == (org == "fused")
        assert torch.equal(dwf.view(torch.int32), dws.view(torch.int32)), org
        assert torch.equal(dbf.view(torch.int32), dbs.view(torch.int32)), org
        assert rel_err(dxf, dxs) < 1e-6 and rel_err(yf, ys) < 1e-6
    _, dxr, dwr, dbr = reference(lay, x, dy)
    assert rel_err(dxf, dxr) < 2e-5 and rel_err(dwf, dwr) < 2e-5 and rel_err(dbf, dbr) < 2e-5


@requires_cuda_backend
@pytest.mark.parametrize("case", ["zero_bias_zero_x", "huge", "tiny", "subnormal", "neg_zero", "inf", "nan", "w_inf"])
def test_extremes_and_fallback(case):
    D, H, G, B = 256, 20_000, 8, 72
    lay = sg_layer(D, H, G, 8, 16, 1, plan=plan("separate", deterministic=True))
    x, dy = inputs(B, D, G)
    if case == "zero_bias_zero_x":
        with torch.no_grad():
            lay.bias.zero_()
        x.zero_()
        x[::3] = inputs(B, D, G)[0][::3]
    elif case == "huge":
        dy = dy * 1e30
    elif case == "tiny":
        dy = dy * 3e-38
    elif case == "subnormal":
        dy = dy * 1e-36
    elif case == "neg_zero":
        dy = torch.where(dy > 0, dy, torch.full_like(dy, -0.0))
    elif case == "inf":
        dy[3, 2] = float("inf")
        dy[5, 1] = -float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    elif case == "w_inf":
        with torch.no_grad():
            lay.weight.view(-1)[1234] = float("inf")
    _, dxs, dws, dbs, _ = run(lay, x, dy)
    lay.plan = plan("fused", deterministic=True)
    _, dxf, dwf, dbf, info = run(lay, x, dy)
    assert info["backward_organization"] == "fused"
    assert torch.equal(dwf.view(torch.int32), dws.view(torch.int32))
    assert torch.equal(dbf.view(torch.int32), dbs.view(torch.int32))
    assert torch.equal(torch.isnan(dxf), torch.isnan(dxs)) and torch.equal(torch.isinf(dxf), torch.isinf(dxs))
    fin = torch.isfinite(dxs)
    if case in ("inf", "nan"):  # fallback: the original grouped dX, generic epilogue (same order at R = 1)
        assert torch.equal(dxf[fin].view(torch.int32), dxs[fin].view(torch.int32))
    else:
        assert rel_err(dxf[fin], dxs[fin]) < 1e-6
        assert torch.equal(torch.signbit(dxf[dxs == 0]), torch.signbit(dxs[dxs == 0]))


@requires_cuda_backend
def test_main_grad_accumulates_over_backward_calls():
    D, H, G, B = 1024, 100_000, 32, 136
    lay = sg_layer(D, H, G, 8, 16, 4, plan=plan("fused"), fp32=False)
    lay.enable_main_grad()
    x, dy = inputs(B, D, G)
    xb = x.bfloat16()
    for _ in range(3):
        (lay(xb).float() * dy).sum().backward()
    acc_f = lay.weight.main_grad.clone(), lay.bias.main_grad.clone()
    lay.weight.main_grad.zero_()
    lay.bias.main_grad.zero_()
    lay.plan = plan("separate")
    for _ in range(3):
        (lay(xb).float() * dy).sum().backward()
    assert torch.equal(acc_f[0], lay.weight.main_grad) and torch.equal(acc_f[1], lay.bias.main_grad)


@requires_cuda_backend
@pytest.mark.parametrize("what", ["bf16", "identity", "C64", "generic_dx", "S64_cfg0"])
def test_explicit_unsupported_raises(what):
    kw = dict(D=1024, H=100_000, G=16, C=8, S=16, R=4, pooling="mean")
    p = plan("fused")
    if what == "bf16":
        p = plan("fused", precision="bf16")
    elif what == "identity":
        kw["pooling"] = "mean_identity"
    elif what == "C64":
        kw["C"] = 64
    elif what == "generic_dx":
        p = plan("fused", cuda_grouped_dx="generic")
    elif what == "S64_cfg0":
        kw["S"], kw["R"] = 64, 16
    try:
        lay = sg_layer(kw["D"], kw["H"], kw["G"], kw["C"], kw["S"], kw["R"], pooling=kw["pooling"], plan=p)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"layer construction: {e}")
    x, dy = inputs(64, kw["D"], kw["G"])
    with pytest.raises(ValueError, match="cuda_grouped_backward='fused' (unavailable|needs)"):
        run(lay, x, dy)


@requires_cuda_backend
def test_auto_is_separate():
    lay = sg_layer(1024, 100_000, 16, 8, 16, 4, plan=PhysicalPlan(cuda_input_supergroup_backend="grouped"))
    x, dy = inputs(64, 1024, 16)
    _, _, _, _, info = run(lay, x, dy)
    assert info["backward_organization"] == "separate" and "grouped_fused" not in info
    assert cuda_ops.SGF_CFGS  # tilings exported for the benchmarks
