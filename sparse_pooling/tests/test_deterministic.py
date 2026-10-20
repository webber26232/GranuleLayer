"""PhysicalPlan.reduction = "deterministic" (kernels/cuda/sg_det.cu): production R > 1 grouped
execution with fixed-owner, fixed-order reductions.

* repeated identical forward / backward calls are byte-for-byte equal (outputs, dX, dW, db,
  signed zeros included) for S / R = (16, 4), (32, 8), (64, 16), R = 1, one and several chunks
  per supergroup, token tails, ragged H, empty buckets, split and BF16 gradients, C = 8 and 64;
* the R > 1 graphs really have overlapping writers in the fast organization and several chunks
  in the deterministic one (counted from the work lists), and every (chunk, local feature)
  partial reaches its global feature exactly once;
* deterministic vs the FP32 reference (same tolerances as test_input_supergroups) and vs fast
  (different FP32 order: close, not bitwise);
* the workspace cap changes the token windows but not the bits; a NaN-poisoned workspace leaves
  the results unchanged (every read row is written);
* API: explicit / auto / global torch.use_deterministic_algorithms resolution, conflicting
  explicit plans and unsupported layers raise, warn_only warns;
* non-finite coefficients take the in-kernel generic path (one path, still ordered);
* main_grad accumulation, two layer instances, repeated calls (no workspace state leaks).
"""
import warnings

import pytest
import torch

from conftest import active_profile_name, rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.supergroups import chunk_table
from test_input_supergroups import BOUNDARY_CASE, boundary_reference, inputs, reference, run, sg_layer


def det_plan(precision="split", **kw):
    return PhysicalPlan(grad_precision=precision, reduction="deterministic", **kw)


def bits(t):
    return t.detach().contiguous().view(torch.int32) if t.dtype == torch.float32 else t.detach().contiguous().view(
        torch.int16)


def assert_same(r0, r1):
    for a, b in zip(r0[:4], r1[:4]):
        assert torch.equal(bits(a), bits(b))


def run_n(lay, x, dy, n):
    out = [run(lay, x, dy) for _ in range(n)]
    for r in out[1:]:
        assert_same(out[0], r)
    return out[0]


# (D, H, G, S, R, B, extra plan fields)
SHAPES = [
    (1024, 300_000, 64, 16, 4, 200, {}),
    (1024, 300_000, 64, 32, 8, 136, {}),
    (1024, 300_000, 64, 64, 16, 256, {}),
    (256, 300_000, 16, 16, 4, 72, dict(cuda_grouped_dx_chunk_units=48, cuda_grouped_fwd_chunk_units=40)),  # several chunks
    (512, 200_003, 32, 32, 8, 136, dict(cuda_grouped_dx_chunk_units=64, cuda_grouped_fwd_chunk_units=24)),  # ragged H
    (1024, 400_000, 16, 64, 16, 264, dict(cuda_grouped_fwd_chunk_units=8)),  # tails past 128 / 32 tokens
    (256, 20_011, 8, 16, 1, 64, {}),  # R = 1 regression
    (256, 2_000, 64, 16, 4, 136, {}),  # tiny / empty buckets
]


@requires_cuda_backend
@pytest.mark.parametrize("D,H,G,S,R,B,kw", SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_repeatable_and_correct(D, H, G, S, R, B, kw, precision):
    lay = sg_layer(D, H, G, 8, S, R, plan=det_plan(precision, **kw))
    x, dy = inputs(B, D, G)
    y, dx, dw, db, info = run_n(lay, x, dy, 10)
    assert info["reduction"]["mode"] == "deterministic"
    assert info["forward_kernel"] == "grouped" and info["dx_kernel"] == "grouped"
    assert info["grouped_dx"]["reduction_group_warps"] == 1
    assert info["grouped_dx"]["reduction"]["mode"] == "ordered_partials"
    assert info["grouped_fwd"]["reduction"]["mode"] == "ordered_partials"
    boundary = (S, R, (D, H, G, B), precision) == (BOUNDARY_CASE["S"], BOUNDARY_CASE["R"], BOUNDARY_CASE["shape"],
                                                   BOUNDARY_CASE["precision"])
    _, dxr, dwr, dbr = (boundary_reference if boundary else reference)(lay, x, dy)  # the documented ReLU tie
    tol = 2e-5 if precision == "split" else 2e-2
    assert rel_err(dx, dxr) < tol and rel_err(dw, dwr) < tol and rel_err(db, dbr) < tol
    # fast mode from the same tensors: a different FP32 order, numerically equivalent
    lay.plan = PhysicalPlan(grad_precision=precision, reduction="fast", **kw)
    yf, dxf, dwf, dbf, infof = run(lay, x, dy)
    assert infof["reduction"]["mode"] == "fast"
    assert rel_err(y, yf) < 1e-6 and rel_err(dx, dxf) < 1e-5
    if infof["grouped_dx"]["reduction_group_warps"] == 1 and infof["grouped_dx"]["plan"] == info["grouped_dx"]["plan"]:
        # same dX tiles / private copies: dW / db have the same operands and order
        assert torch.equal(bits(dw), bits(dwf)) and torch.equal(bits(db), bits(dbf))


def ownership(lay, B):
    """Writers per destination from the actual work lists: fast = CTAs adding atomically into a
    dX feature row (any token tile) / FP32 atomics per (output, token); deterministic = partial
    rows per destination and coverage of every (chunk, local feature) row."""
    sg, info = lay._rt.sg, lay.last_info
    cu_x, cu_f = info["grouped_dx"]["chunk_units"], info["grouped_fwd"]["chunk_units"]
    items, ch = sg.item_chunks(cu_x)
    feat = sg.feat.cpu().long()
    writers = torch.zeros(lay.layout.input_dim, dtype=torch.long)
    writers.index_add_(0, feat[ch[:, 0].cpu().long()].reshape(-1), torch.ones(ch.shape[0] * sg.S, dtype=torch.long))
    off, rec = sg.det_rows("dx", cu_x)
    counts = (off[1:] - off[:-1]).cpu().long()
    covered = torch.bincount(rec.cpu().long(), minlength=ch.shape[0] * sg.S)
    fi, _ = sg.item_chunks(cu_f)
    foff, frec = sg.det_rows("fwd", cu_f)
    fcount = (foff[1:] - foff[:-1]).cpu().long()
    return dict(dx_writers=writers, dx_rows=counts, dx_covered=covered, chunks=ch.shape[0], Q=sg.Q,
                fwd_items_per_output=torch.bincount(fi[:, 0].cpu().long(), minlength=lay.layout.groups),
                fwd_rows=fcount, fwd_covered=torch.bincount(frec.cpu().long(), minlength=fi.shape[0]))


@requires_cuda_backend
@pytest.mark.parametrize("S,R,kw", [(16, 4, dict(cuda_grouped_dx_chunk_units=48)), (32, 8, {}), (64, 16, {})])
def test_overlapping_writers_and_exact_coverage(S, R, kw):
    lay = sg_layer(256 if kw else 1024, 300_000, 16 if kw else 64, 8, S, R, plan=det_plan(**kw))
    x, dy = inputs(72, lay.layout.input_dim, lay.layout.groups)
    run(lay, x, dy)
    o = ownership(lay, 72)
    # fast organization: every feature is written by >= R CTAs per token tile (R supergroups)
    assert int(o["dx_writers"].min()) >= R > 1
    # deterministic: the reduction of feature d reads exactly those partial rows, each once
    assert torch.equal(o["dx_rows"], o["dx_writers"])
    assert bool((o["dx_covered"] == 1).all())
    assert int(o["fwd_items_per_output"].max()) > 1  # several supergroups add into one output
    assert torch.equal(o["fwd_rows"], o["fwd_items_per_output"]) and bool((o["fwd_covered"] == 1).all())
    if kw:
        assert o["chunks"] > o["Q"]  # several chunks per supergroup


@requires_cuda_backend
@pytest.mark.parametrize("cap", [6 << 20, 12 << 20, 64 << 20])
def test_workspace_cap_and_poison(cap):
    D, H, G, S, R, B = 1024, 300_000, 64, 64, 16, 264
    lay = sg_layer(D, H, G, 8, S, R, plan=det_plan())
    x, dy = inputs(B, D, G)
    r0 = run(lay, x, dy)
    lay.plan = det_plan(deterministic_workspace_bytes=cap)
    try:
        cuda_ops.DET_WORKSPACE_FILL = float("nan")
        r1 = run(lay, x, dy)
    finally:
        cuda_ops.DET_WORKSPACE_FILL = None
    assert_same(r0, r1)
    ws = r1[4]["grouped_dx"]["reduction"]
    assert ws["workspace_bytes"] <= cap
    if cap < (1 << 24):
        assert ws["windows"] > 1
        if active_profile_name() == "rtx3090":  # forward windows depend on the profile's tiles
            assert r1[4]["grouped_fwd"]["reduction"]["windows"] > 1


@requires_cuda_backend
def test_cap_too_small_raises():
    lay = sg_layer(1024, 300_000, 64, 8, 64, 16, plan=det_plan(deterministic_workspace_bytes=4096))
    x, dy = inputs(64, 1024, 64)
    with pytest.raises(RuntimeError, match="no atomic fallback"):
        run(lay, x, dy)


@requires_cuda_backend
@pytest.mark.parametrize("C", [8, 64])
def test_c64_and_mean_identity(C):
    for pooling in ("mean", {"type": "mean", "activation": "identity"}):
        lay = sg_layer(1024, 300_000, 64, C, 16, 4, pooling=pooling, plan=det_plan())
        x, dy = inputs(136, 1024, 64)
        _, dx, dw, db, info = run_n(lay, x, dy, 3)
        assert info["dx_kernel"] == "grouped" and info["grouped_dx"]["reduction_group_warps"] == 1
        _, dxr, dwr, dbr = reference(lay, x, dy)
        assert rel_err(dx, dxr) < 2e-5 and rel_err(dw, dwr) < 2e-5 and rel_err(db, dbr) < 2e-5


@requires_cuda_backend
@pytest.mark.parametrize("case", ["zero_bias_zero_x", "neg_zero", "subnormal", "tiny", "cancel", "boundary", "inf", "nan"])
def test_extremes(case):
    D, H, G, B = 1024, 300_000, 64, 136
    lay = sg_layer(D, H, G, 8, 32, 8, plan=det_plan())
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
    elif case == "cancel":  # pairs of tokens with opposite upstream gradients: cancellation in dW
        dy[1::2] = -dy[0::2]
        x[1::2] = x[0::2]
    elif case == "boundary":  # biases near -z: pre-activations near the ReLU boundary
        with torch.no_grad():
            lay.bias.mul_(1e-3)
        x = x * 1e-3
    elif case == "inf":
        dy[3, 2] = float("inf")
        dy[5, 1] = -float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    r = [run(lay, x, dy) for _ in range(4)]
    for a in r[1:]:
        for t0, t1 in zip(r[0][:4], a[:4]):
            assert torch.equal(torch.isnan(t0), torch.isnan(t1))
            fin = torch.isfinite(t0)
            assert torch.equal(torch.isfinite(t1), fin)
            assert torch.equal(bits(t0)[fin], bits(t1)[fin])  # signed zeros included
    lay.plan = PhysicalPlan(reduction="fast")
    f = run(lay, x, dy)
    for t0, t1 in zip(r[0][:4], f[:4]):  # same non-finite pattern as the fast path
        assert torch.equal(torch.isnan(t0), torch.isnan(t1)) and torch.equal(torch.isinf(t0), torch.isinf(t1))


@requires_cuda_backend
def test_main_grad_two_instances_and_repeats():
    D, H, G, B = 1024, 300_000, 64, 136
    mk = lambda: sg_layer(D, H, G, 8, 64, 16, plan=det_plan(), fp32=False)
    a, b = mk(), mk()
    with torch.no_grad():
        b.weight.mul_(0.5)
    for lay in (a, b):
        lay.enable_main_grad()
    x, dy = inputs(B, D, G)
    xb = x.bfloat16()

    def acc(lay, n=3):
        lay.zero_main_grad()
        for _ in range(n):
            (lay(xb).float() * dy).sum().backward()
        return lay.weight.main_grad.clone(), lay.bias.main_grad.clone()

    a0 = acc(a)
    b0 = acc(b)
    a1 = acc(a)  # after another instance used its own (different-size) workspaces
    assert torch.equal(bits(a0[0]), bits(a1[0])) and torch.equal(bits(a0[1]), bits(a1[1]))
    assert not torch.equal(a0[0], b0[0])
    # accumulation honours the pre-existing main gradient: 3 calls on top of 3 calls
    a.zero_main_grad()
    for _ in range(6):
        (a(xb).float() * dy).sum().backward()
    six = a.weight.main_grad.clone()
    a.zero_main_grad()
    for _ in range(3):
        (a(xb).float() * dy).sum().backward()
    for _ in range(3):
        (a(xb).float() * dy).sum().backward()
    assert torch.equal(bits(six), bits(a.weight.main_grad))


# ---------------------------------------------------------------- API resolution
@requires_cuda_backend
def test_api_resolution():
    D, H, G, B = 1024, 300_000, 64, 72
    x, dy = inputs(B, D, G)
    lay = sg_layer(D, H, G, 8, 64, 16, plan=PhysicalPlan())
    assert run(lay, x, dy)[4]["reduction"]["mode"] == "fast"  # default: fast
    try:
        torch.use_deterministic_algorithms(True)
        info = run(lay, x, dy)[4]
        assert info["reduction"]["mode"] == "deterministic" and "torch" in info["reduction"]["reason"]
        # never the unordered shared-CAS plan; private_t32 is the RTX 3090 profile's deterministic choice
        assert not info["grouped_dx"]["plan"].startswith("shared")
        if active_profile_name() == "rtx3090":
            assert info["grouped_dx"]["plan"] == "private_t32"
        lay.plan = PhysicalPlan(reduction="fast")
        with pytest.raises(RuntimeError, match="use_deterministic_algorithms"):
            run(lay, x, dy)
        # an unsupported layer under the global flag raises (no silent atomics)
        lay.plan = PhysicalPlan(cuda_grouped_dx="generic")
        with pytest.raises(ValueError, match="explicit generic dX"):
            run(lay, x, dy)
    finally:
        torch.use_deterministic_algorithms(False)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        lay.plan = PhysicalPlan(cuda_grouped_dx="generic")
        with pytest.warns(UserWarning, match="running reduction='fast'"):
            info = run(lay, x, dy)[4]
        assert info["reduction"]["mode"] == "fast"
    finally:
        torch.use_deterministic_algorithms(False)
    # explicit conflicts
    for kw, msg in [(dict(cuda_grouped_dx_plan="shared_t64"), "shared_t64"),
                    (dict(cuda_grouped_dx_reduction=8), "cuda_grouped_dx_reduction=8"),
                    (dict(cuda_grouped_backward="fused"), "fused"),
                    (dict(cuda_grouped_dx="generic"), "generic dX")]:
        lay.plan = det_plan(**kw)
        with pytest.raises(ValueError, match=msg):
            run(lay, x, dy)
    # an explicit private plan / private fields are honoured
    lay.plan = det_plan(cuda_grouped_dx_tokens=32, cuda_grouped_dx_reduction=1, cuda_grouped_dx_scatter="baseline")
    info = run(lay, x, dy)[4]
    assert (info["grouped_dx"]["plan"], info["grouped_dx"]["tokens"], info["grouped_dx"]["scatter"]) == (
        "tiles", 32, "baseline")
    with pytest.raises(ValueError, match="reduction must be"):
        PhysicalPlan(reduction="ordered")


@requires_cuda_backend
def test_unsupported_layers_raise():
    lay = SparsePoolingLayer(256, 20_000, 8, shared_neurons=8, pooling="mean", device="cuda",
                             plan=det_plan())  # no connectivity: generic k_dx (atomics)
    with pytest.raises(ValueError, match="input-supergroup"):
        lay(torch.randn(16, 256, device="cuda").bfloat16()).sum().backward()
    lay = sg_layer(256, 20_000, 8, 8, 16, 4, pooling={"type": "logmeanexp", "temperature": 0.5}, plan=det_plan())
    with pytest.raises(ValueError, match="Mean"):
        lay(torch.randn(16, 256, device="cuda")).sum().backward()
    lay = sg_layer(256, 20_000, 8, 8, 16, 4, plan=det_plan())
    with pytest.raises(ValueError, match="feature-major"):
        lay(torch.randn(12, 256, device="cuda")).sum().backward()  # B % 8 != 0


def test_chunk_table_is_host_deterministic():
    q = torch.repeat_interleave(torch.arange(4), torch.tensor([3, 0, 5, 2]))
    w = torch.tensor([4, 1, 2, 7, 1, 1, 1, 9, 2, 2])
    assert torch.equal(chunk_table(q, w, 4, 4), chunk_table(q.clone(), w.clone(), 4, 4))
