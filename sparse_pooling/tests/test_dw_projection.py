"""Compact C = 8 dW projection (PhysicalPlan.cuda_grouped_dw_projection = "compact",
sparse_pool_bwd.cuh dw_block_c8_compact) against the padded original: dW / db BITWISE equal for
every shape, tail, ragged output, precision, extreme and reduction mode; dX untouched; non-finite
coefficients take the generic padded launch (one path); explicit unsupported requests raise."""
import pytest
import torch

from conftest import rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan
from test_input_supergroups import inputs, reference, run, sg_layer


def plan(proj, precision="split", **kw):
    return PhysicalPlan(grad_precision=precision, cuda_input_supergroup_backend="grouped",
                        cuda_grouped_dw_projection=proj, **kw)


def ibits(t):
    return t.contiguous().view(torch.int32)


SHAPES = [  # (D, H, G, S, R, B)
    (1024, 300_000, 64, 16, 4, 200),
    (1024, 300_000, 64, 32, 8, 136),
    (1024, 300_000, 64, 64, 16, 256),
    (256, 20_011, 8, 16, 4, 64),      # ragged outputs: partial units / blocks
    (256, 2_000, 64, 16, 4, 136),     # tiny / empty buckets
    (512, 50_000, 32, 16, 1, 72),     # R = 1, token tail 8
    (1024, 100_003, 16, 64, 16, 24),  # B < one 16-token tile pair, ragged H
]


@requires_cuda_backend
@pytest.mark.parametrize("D,H,G,S,R,B", SHAPES)
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("cfg", [0, 1, 2, 3])
def test_compact_bitwise(D, H, G, S, R, B, precision, cfg):
    lay = sg_layer(D, H, G, 8, S, R, plan=plan("padded", precision, cuda_grouped_dw_cfg=cfg, reduction="deterministic"))
    x, dy = inputs(B, D, G)
    try:
        y0, dx0, dw0, db0, i0 = run(lay, x, dy)
    except ValueError as e:
        pytest.skip(str(e))
    lay.plan = plan("compact", precision, cuda_grouped_dw_cfg=cfg, reduction="deterministic")
    y1, dx1, dw1, db1, i1 = run(lay, x, dy)
    assert i0["grouped_dw"]["projection"] == "padded" and i1["grouped_dw"]["projection"] == "compact"
    assert i1["grouped_dw"]["kernel"].endswith(", 1, 1>") and i0["grouped_dw"]["kernel"].endswith(", 1, 0>")
    assert torch.equal(ibits(dw1), ibits(dw0)) and torch.equal(ibits(db1), ibits(db0))
    assert torch.equal(ibits(y1), ibits(y0)) and torch.equal(ibits(dx1), ibits(dx0))  # deterministic mode: all bits


@requires_cuda_backend
@pytest.mark.parametrize("case", ["zero_bias_zero_x", "neg_zero", "subnormal", "tiny", "huge", "cancel", "boundary",
                                  "inf", "nan", "w_inf"])
def test_compact_extremes(case):
    D, H, G, B = 1024, 300_000, 64, 136
    lay = sg_layer(D, H, G, 8, 32, 8, plan=plan("padded", reduction="deterministic"))
    x, dy = inputs(B, D, G)
    if case == "zero_bias_zero_x":
        with torch.no_grad():
            lay.bias.zero_()
        x.zero_()
        x[::3] = inputs(B, D, G)[0][::3]
    elif case == "neg_zero":
        dy = torch.where(dy > 0, dy, torch.full_like(dy, -0.0))
    elif case == "subnormal":
        dy = dy * 1e-36
    elif case == "tiny":
        dy = dy * 3e-38
    elif case == "huge":
        dy = dy * 1e30
    elif case == "cancel":
        dy[1::2] = -dy[0::2]
        x[1::2] = x[0::2]
    elif case == "boundary":  # many pre-activations at / near 0 (signed zeros, subnormal z)
        with torch.no_grad():
            lay.bias.mul_(1e-30)
        x = x * 1e-30
    elif case == "inf":
        dy[3, 2] = float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    elif case == "w_inf":
        with torch.no_grad():
            lay.weight.view(-1)[1234] = float("inf")
    _, _, dw0, db0, _ = run(lay, x, dy)
    lay.plan = plan("compact", reduction="deterministic")
    _, _, dw1, db1, info = run(lay, x, dy)
    assert info["grouped_dw"]["projection"] == "compact"
    assert torch.equal(ibits(dw1), ibits(dw0)) and torch.equal(ibits(db1), ibits(db0))  # NaN payloads included


@requires_cuda_backend
def test_compact_reference_and_main_grad():
    D, H, G, B = 1024, 300_000, 64, 200
    lay = sg_layer(D, H, G, 8, 16, 4, plan=plan("compact"))
    x, dy = inputs(B, D, G)
    _, dx, dw, db, _ = run(lay, x, dy)
    _, dxr, dwr, dbr = reference(lay, x, dy)
    assert rel_err(dw, dwr) < 2e-5 and rel_err(db, dbr) < 2e-5 and rel_err(dx, dxr) < 2e-5
    # accumulation into pre-existing main gradients
    lay = sg_layer(D, H, G, 8, 16, 4, plan=plan("padded", reduction="deterministic"), fp32=False)
    lay.enable_main_grad()
    xb = x.bfloat16()
    out = {}
    for proj in ("padded", "compact"):
        lay.plan = plan(proj, reduction="deterministic")
        lay.zero_main_grad()
        with torch.no_grad():
            lay.weight.main_grad.fill_(0.25)
        for _ in range(3):
            (lay(xb).float() * dy).sum().backward()
        out[proj] = (lay.weight.main_grad.clone(), lay.bias.main_grad.clone())
    assert torch.equal(ibits(out["padded"][0]), ibits(out["compact"][0]))
    assert torch.equal(ibits(out["padded"][1]), ibits(out["compact"][1]))


@requires_cuda_backend
def test_compact_selection_and_errors():
    x, dy = inputs(64, 1024, 64)
    lay = sg_layer(1024, 300_000, 64, 8, 16, 4, pooling={"type": "mean", "activation": "identity"},
                   plan=plan("auto"))
    info = run(lay, x, dy)[4]
    assert info["grouped_dw"]["projection"] == "padded" and "ReLU" in info["grouped_dw"]["projection_reason"]
    lay.plan = plan("compact")
    with pytest.raises(ValueError, match="compact"):
        run(lay, x, dy)
    lay = sg_layer(1024, 300_000, 64, 64, 16, 4, plan=plan("compact"))
    with pytest.raises(ValueError, match="C = 8"):
        run(lay, x, dy)
    lay = sg_layer(1024, 300_000, 64, 8, 16, 4, plan=plan("compact", cuda_dw_epilogue="generic"))
    with pytest.raises(ValueError, match="fast dW epilogue"):
        run(lay, x, dy)
