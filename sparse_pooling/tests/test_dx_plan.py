"""Whole grouped dX plan selection (PhysicalPlan.cuda_grouped_dx_plan, cuda_ops.SG_AUTO_DX_PLAN):
named plans, the scale-aware auto rule, precedence of explicit fields, device / precision
scoping, last_info, and numerical equivalence of the plans."""
import pytest
import torch

from conftest import rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan
from sparse_pooling.config import DX_PLANS
from sparse_pooling.kernels import cuda_ops
from test_input_supergroups import inputs, reference, run, sg_layer


def test_named_plan_conflicts_raise():
    with pytest.raises(ValueError, match="conflicting"):
        PhysicalPlan(cuda_grouped_dx_plan="private_t32", cuda_grouped_dx_tokens=64)
    with pytest.raises(ValueError, match="conflicting"):
        PhysicalPlan(cuda_grouped_dx_plan="shared_t64", cuda_grouped_dx_scatter="vectorized")
    with pytest.raises(AssertionError):
        PhysicalPlan(cuda_grouped_dx_plan="private_t64")
    # consistent explicit fields are accepted
    PhysicalPlan(cuda_grouped_dx_plan="private_t32", cuda_grouped_dx_tokens=32, cuda_grouped_dx_reduction=1,
                 cuda_grouped_dx_scatter="vectorized_transpose_hoisted")


def rule(**kw):
    r = dict(C=8, S=64, precisions=("split",), act="relu", cc=torch.cuda.get_device_capability(),
             min_units_per_chunk=0, min_tokens=0, plan="private_t32")
    r.update(kw)
    return [r]


def s64_layer(plan, H=200_000, G=64):
    return sg_layer(1024, H, G, 8, 64, 16, plan=plan)


def ran_plan(info):
    gd = info["grouped_dx"]
    return gd["plan"], (gd["tokens"], gd["warps"], gd["reduction_group_warps"], gd["scatter"]), gd["plan_reason"]


@requires_cuda_backend
@pytest.mark.parametrize("name", list(DX_PLANS))
def test_named_plans_run_their_whole_plan(name):
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped", cuda_grouped_dx_plan=name))
    x, dy = inputs(200, 1024, 64)
    _, dx, _, _, info = run(lay, x, dy)
    plan, cfg, why = ran_plan(info)
    d = DX_PLANS[name]
    assert plan == name and cfg == (d["tokens"], d["warps"], d["reduction"], d["scatter"]) and "explicit" in why
    assert info["grouped_dx"]["scatter_fallback"] == ""
    _, dxr, _, _ = reference(lay, x, dy)
    assert rel_err(dx, dxr) < 2e-5


@requires_cuda_backend
def test_plans_agree_and_dw_is_unchanged():
    """The two plans differ only in the dX local reduction (order-changing: shared CAS vs
    private copies): dW / db bitwise, dX within the baseline's own tolerance."""
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped", cuda_grouped_dx_plan="shared_t64"))
    x, dy = inputs(136, 1024, 64)
    ya, dxa, dwa, dba, _ = run(lay, x, dy)
    lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped", cuda_grouped_dx_plan="private_t32")
    yb, dxb, dwb, dbb, _ = run(lay, x, dy)
    assert torch.equal(dwa, dwb) and torch.equal(dba, dbb)
    assert rel_err(dxa, dxb) < 1e-6 and rel_err(ya, yb) < 1e-6  # grouped forward: FP32 atomics


@requires_cuda_backend
def test_auto_rule_selects_and_reports(monkeypatch):
    monkeypatch.setattr(cuda_ops, "SG_AUTO_DX_PLAN", rule())
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped"))
    x, dy = inputs(64, 1024, 64)
    _, _, _, _, info = run(lay, x, dy)
    plan, cfg, why = ran_plan(info)
    assert plan == "private_t32" and cfg == (32, 8, 1, "vectorized_transpose_hoisted") and "scale rule" in why
    assert info["grouped_dx"]["kernel"] == "sgx_dx_kernel<1, 8, 2, 8, 1, 2, 9, 1>"
    # below the work threshold: the per-field automatic tiles (shared_t64 at C = 8, S = 64)
    monkeypatch.setattr(cuda_ops, "SG_AUTO_DX_PLAN", rule(min_units_per_chunk=1e9))
    lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped")
    _, _, _, _, info = run(lay, x, dy)
    plan, cfg, why = ran_plan(info)
    assert plan == "tiles" and cfg == (64, 8, 8, "swizzle4_capped") and "threshold" in why
    # below the token threshold
    monkeypatch.setattr(cuda_ops, "SG_AUTO_DX_PLAN", rule(min_tokens=65))
    lay.plan = PhysicalPlan(cuda_input_supergroup_backend="grouped")
    _, _, _, _, info = run(lay, x, dy)
    assert ran_plan(info)[0] == "tiles"


@requires_cuda_backend
@pytest.mark.parametrize("field,value", [("cuda_grouped_dx_tokens", 64), ("cuda_grouped_dx_reduction", 8),
                                         ("cuda_grouped_dx_warps", 8), ("cuda_grouped_dx_scatter", "baseline"),
                                         ("cuda_grouped_dx_chunk_units", 4096)])
def test_explicit_fields_take_precedence(monkeypatch, field, value):
    monkeypatch.setattr(cuda_ops, "SG_AUTO_DX_PLAN", rule())
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped", **{field: value}))
    x, dy = inputs(64, 1024, 64)
    _, _, _, _, info = run(lay, x, dy)
    plan, cfg, why = ran_plan(info)
    assert plan == "tiles" and "explicit grouped dX fields" in why
    assert cfg[0] == 64 and cfg[2] == 8  # SG_AUTO tiles (+ the explicit field), never private_t32


@requires_cuda_backend
@pytest.mark.parametrize("what", ["device", "precision", "activation", "S"])
def test_rule_scope(monkeypatch, what):
    kw = {"device": dict(cc=(9, 0)), "precision": dict(precisions=("bf16",)), "activation": dict(act="identity"),
          "S": dict(S=32)}[what]
    monkeypatch.setattr(cuda_ops, "SG_AUTO_DX_PLAN", rule(**kw))
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped"))
    x, dy = inputs(64, 1024, 64)
    _, _, _, _, info = run(lay, x, dy)
    plan, cfg, why = ran_plan(info)
    assert plan == "tiles" and cfg == (64, 8, 8, "swizzle4_capped")
    if what == "device":
        assert "validated on sm_90 only" in why


@requires_cuda_backend
def test_rule_descriptor_is_cached_host_side():
    lay = s64_layer(PhysicalPlan(cuda_input_supergroup_backend="grouped"))
    x, dy = inputs(64, 1024, 64)
    run(lay, x, dy)
    sg = lay._runtime().sg
    st = sg.item_chunk_stats(cuda_ops.SG_CHUNK_UNITS["dx"])
    assert st is sg.item_chunk_stats(cuda_ops.SG_CHUNK_UNITS["dx"])  # cached object, no recomputation
    items, ch = sg.item_chunks(cuda_ops.SG_CHUNK_UNITS["dx"])
    assert st["chunks"] == ch.shape[0] and st["units"] == int(items[:, 2].sum())
    assert abs(st["units_per_chunk_mean"] * st["chunks"] - st["units"]) < 1e-6 * st["units"] + 1
