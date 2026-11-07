"""Correctness gates of the learned sliding-window reducers (docs/experiments/learned_sliding_window_reduction.md,
section "Numerical tolerances").  The tolerances below were fixed BEFORE the candidate kernels were evaluated; every
mixed-precision comparison is against the FP64 oracle fed the same quantized BF16 operands.
"""
import copy
import math

import pytest
import torch

from sparse_pooling import CoverTopK16Layer
from sparse_pooling.experimental import reduction_reference as RR
from sparse_pooling.experimental.kgen_layout import KLayout, zero_padding_
from sparse_pooling.experimental.learned_reduction import (LearnedReducer, ReducerConfig, ReductionModel, clone_base,
                                                           core_configs, stride_view)
from sparse_pooling.experimental.reduction_layout import (ReductionSpec, Routing, aligned_constraints, assert_aligned,
                                                          parameter_counts, static_estimate, three_stage_counts)
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan

from conftest import requires_cuda_backend

# declared tolerances (relative L2 vs the same-operand FP64 oracle)
TOL_FWD_SPLIT = 3e-5   # h as a BF16 hi + lo pair; FP32 accumulation; FP32 atomic order
TOL_FWD_SIMT = 3e-5    # FP32 h, FP32 accumulation, FP32 atomic order
TOL_GRAD = 1e-4        # Delta and h as hi + lo pairs, FP32 accumulation over many terms
TOL_MAX_NORM = 5e-3    # largest element error / largest element


def relerr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def maxerr(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def build_base(H, G=8, D=256, seed=1, device="cuda"):
    L = KLayout(D, H, G, 64, 256)
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(L.weight_shape, device=device, generator=g) / 8).bfloat16()
    b = (0.1 * torch.randn(L.bias_shape, device=device, generator=g)).bfloat16()
    zero_padding_(L, w, b)
    return CoverTopK16Layer.from_envelope(w, b, input_dim=D, hidden_dim=H, output_dim=G,
                                          connectivity=ConnectivityPlan(64, 2, seed=seed), normalization="energy")


def oracle_grads(model, x, gy):
    """FP64 ordinary-autograd oracle over the same quantized operands (U-level, explicit windows)."""
    red = model.reducer
    sp = model.sparse
    xd = x.detach().to(torch.bfloat16).double().requires_grad_(True)
    W = sp.weight.detach().double().requires_grad_(True)
    b = sp.bias.detach().double().requires_grad_(True)
    Pl = red.routing.unpack(red.P.detach()).double().requires_grad_(True)
    spec = red.spec
    starts = (torch.arange(spec.J) // spec.channels) * spec.stride
    h = RR.hidden(xd, sp.indices, W, b, sp.counts)
    U = RR.window_outputs(h, Pl, starts)
    dx, dw, db, dp = torch.autograd.grad(U, [xd, W, b, Pl], gy.double())
    return U.detach(), dx, dw, db, red.routing.pack(dp)


GEOMS = [  # H, G, L, s, channels, description
    (4096, 8, 64, 64, 1, "no overlap L = s"),
    (4096, 8, 320, 64, 1, "overlap L = 5 s"),
    (4096, 8, 512, 512, 1, "L = s = output width"),
    (3840, 8, 320, 320, 5, "multi-channel"),
    (4096, 8, 4096, 4096, 1, "one window"),
    (4004, 8, 100, 64, 1, "uneven H / G, non-multiple L / s, windows crossing records"),
    (4004, 8, 164, 64, 1, "uneven H / G, non-multiple, ragged C16 tails"),
    (4020, 8, 372, 64, 1, "uneven H / G, 5.8 s windows"),
    (4096, 8, 40, 8, 1, "stride below the record width"),
]


@requires_cuda_backend
@pytest.mark.parametrize("H,G,L,s,ch,desc", GEOMS, ids=[g[-1].replace(" ", "_") for g in GEOMS])
@pytest.mark.parametrize("M", [8, 40, 136])
def test_forward_and_gradients_match_oracle(H, G, L, s, ch, desc, M):
    # Seeded: about 1 draw in 300 puts a (row, neuron) within FP32 rounding of z = 0 (seen: |z| = 1.9e-9).  FP32 and the FP64 oracle then
    # take different sides of the ReLU kink: one neuron holds 99.98 % of the dW residual, and dP is unaffected.
    torch.manual_seed(H + 7 * G + 11 * L + 13 * s + 17 * ch + 19 * M)
    base = build_base(H, G)
    cfg = ReducerConfig("t", "learned", L=L, s=s, channels=ch)
    m = ReductionModel(base, cfg, output_dim=32)
    red = m.reducer
    assert red.routing.uncovered_neurons == 0
    x = torch.randn(M, 256, device="cuda", dtype=torch.float32).requires_grad_(True)
    U = red(x)
    gy = torch.randn_like(U)
    Uo, dxo, dwo, dbo, dpo = oracle_grads(m, x, gy.cpu().to("cuda"))
    assert relerr(U, Uo) < TOL_FWD_SPLIT
    assert maxerr(U, Uo) < TOL_MAX_NORM
    m.zero_main_grad()
    torch.autograd.backward(U, gy)
    assert relerr(x.grad, dxo) < TOL_GRAD
    assert relerr(m.sparse.weight.main_grad, dwo) < TOL_GRAD
    assert relerr(m.sparse.bias.main_grad, dbo) < TOL_GRAD
    assert relerr(red.P.main_grad, dpo) < TOL_GRAD
    # no gradient anywhere but on member slots; padded neurons untouched
    member = red.routing.member_mask("cuda")
    assert float(red.P.main_grad[~member].abs().max() if (~member).any() else 0) == 0.0
    assert float(red.P.detach()[~member].float().abs().max() if (~member).any() else 0) == 0.0
    valid = m.sparse.layout.valid_mask("cuda")
    assert float(m.sparse.weight.main_grad[~valid].abs().max() if (~valid).any() else 0) == 0.0


@requires_cuda_backend
@pytest.mark.parametrize("epi,tol", [("mma_split", TOL_FWD_SPLIT), ("simt", TOL_FWD_SIMT)])
@pytest.mark.parametrize("L,s", [(320, 64), (64, 64), (100, 64)])
def test_epilogues(epi, tol, L, s):
    H = 4096 if (4096 - L) % s == 0 else 4004
    base = build_base(H)
    m = ReductionModel(base, ReducerConfig("t", "learned", L=L, s=s, epilogue=epi), output_dim=32)
    x = torch.randn(40, 256, device="cuda")
    U = m.reducer(x)
    Uo = oracle_grads(m, x, torch.ones_like(U))[0]
    assert relerr(U, Uo) < tol


@requires_cuda_backend
def test_bf16_epilogue_is_the_declared_numerical_change():
    """mma_bf16 rounds h to BF16: it matches an oracle with that quantization, and differs from the FP32-h oracle."""
    base = build_base(4096)
    m = ReductionModel(base, ReducerConfig("t", "learned", L=320, s=64, epilogue="mma_bf16"), output_dim=32)
    red, sp = m.reducer, m.sparse
    x = torch.randn(40, 256, device="cuda")
    U = red(x)
    xd = x.to(torch.bfloat16).double()
    h = RR.hidden(xd, sp.indices, sp.weight.detach().double(), sp.bias.detach().double(), sp.counts)
    Pl = red.routing.unpack(red.P.detach()).double()
    st = torch.arange(red.spec.J) * 64
    Uq = RR.window_outputs(h.float().to(torch.bfloat16).double(), Pl, st)
    Uf = RR.window_outputs(h, Pl, st)
    assert relerr(U, Uq) < TOL_FWD_SIMT
    assert relerr(U, Uf) > 10 * relerr(U, Uq)


@requires_cuda_backend
def test_block_equals_window_in_fp64():
    H, L, s = 2048, 320, 64
    h = torch.randn(16, H, dtype=torch.float64)
    spec = ReductionSpec(H, L, s)
    P = torch.randn(spec.J, L, dtype=torch.float64)
    a = RR.window_outputs(h, P, torch.arange(spec.J) * s)
    b = RR.stride_block_outputs(h, P, s)
    assert relerr(b, a) < 1e-13
    spec = ReductionSpec(H, 320, 320, channels=5)
    P = torch.randn(spec.J, 320, dtype=torch.float64)
    a = RR.window_outputs(h, P, (torch.arange(spec.J) // 5) * 320)
    b = RR.stride_block_outputs(h, P, 320, ch=5)
    assert relerr(b, a) < 1e-13


def test_counts_and_constraints():
    """The exact tables of the experiment, recomputed."""
    H1, H2, H3 = 40 * 262_144, 48 * 262_144, 10 * 262_144
    assert (H1, H2, H3) == (10_485_760, 12_582_912, 2_621_440)
    a4 = ReductionSpec(H1, 10_240, 2_048)
    assert a4.J == 5_116 and a4.weights == 52_387_840 and a4.J * 1024 == 5_238_784 and a4.uncovered_tail == 0
    assert_aligned(a4)
    pc = parameter_counts(H1)["models"]
    assert pc["A1"]["added_weights"] == 1_048_576 and pc["A2"]["added_weights"] == 5_242_880
    assert pc["A3"]["added_weights"] == 15_728_640 and pc["A4"]["added_weights"] == 57_626_624
    assert pc["A5"]["added_weights"] == 5_238_784 and pc["MC"]["added_weights"] == 57_671_680
    assert pc["A4"]["added_biases"] == 6_140
    assert pc["MC"]["added_weights"] - pc["A4"]["added_weights"] == 45_056
    b4 = ReductionSpec(H2, 10_240, 2_048)
    assert b4.J == 6_140 and b4.weights == 62_873_600 and b4.J * 1024 == 6_287_360
    assert not aligned_constraints(b4, multi_channel=False).get("H % L == 0", False)
    assert H2 % 10_240 != 0
    c4 = ReductionSpec(H3, 10_240, 2_048)
    assert c4.J == 1_276 and H3 % 10_240 == 0
    mc = ReductionSpec(H3, 10_240, 10_240, channels=5)
    assert mc.windows == 256 and mc.J == 1_280
    ts = three_stage_counts(H1)
    assert ts["J2"] == 1_216 and ts["uncovered_second_stage"] == 0 and ts["weights"] == 53_944_320
    ts = three_stage_counts(H3)
    assert ts["J1"] == 1_276 and ts["J2"] == 256 and ts["uncovered_second_stage"] == 0
    assert 8 * 2 ** 10 * 52_387_840 == 8_192 * 52_387_840
    est = static_estimate(H1, 8_192, a4)
    assert abs(est["flops"]["total_reduction_forward"] / 1e12 - 0.94415) < 1e-4
    assert abs(est["forbidden"]["h_bf16_gib"] - 160.0) < 1e-9 and abs(est["forbidden"]["unfolded_bf16_gib"] - 799.375) < 1e-9
    assert abs(est["compact"]["u_bf16_mib"] - 79.9375) < 1e-9


def test_routing_multiplicity_and_rejections():
    H = 2048 * 20
    spec = ReductionSpec(H, 10_240, 2_048)
    mult = spec.multiplicity()
    assert int(mult.max()) == 5 and int(mult.min()) == 1 and bool((mult > 0).all())
    assert mult[:2048].eq(1).all() and mult[2048:4096].eq(2).all() and mult[-2048:].eq(1).all()
    assert mult[4 * 2048:16 * 2048].eq(5).all()
    assert int(mult.sum()) == spec.weights
    for bad in (dict(hidden_dim=100, window=200, stride=10), dict(hidden_dim=100, window=5, stride=10),
                dict(hidden_dim=100, window=10, stride=5, channels=2), dict(hidden_dim=100, window=10, stride=10, boundary="circular")):
        with pytest.raises(ValueError):
            ReductionSpec(**bad)
    tail = ReductionSpec(1000, 300, 100, 1)
    assert tail.uncovered_tail == 0
    tail = ReductionSpec(1030, 300, 100, 1)
    assert tail.uncovered_tail == 30
    counts = torch.full((10,), 103, dtype=torch.int32)
    r = Routing(tail, counts, 7, 16)
    assert r.uncovered_neurons == 30 and r.needs_grad_mask


def test_pack_unpack_round_trip():
    for H, L, s, ch, G in ((4096, 320, 64, 1, 8), (4004, 100, 64, 1, 8), (3840, 320, 320, 5, 8)):
        counts = KLayout(256, H, G, 16, 16).counts_tensor("cpu")
        NB = KLayout(256, H, G, 16, 16).blocks_per_group
        r = Routing(ReductionSpec(H, L, s, ch), counts, NB, 16)
        assert r.packed_valid == r.spec.weights
        lg = torch.randn(r.spec.J, L)
        p = r.pack(lg)
        assert p.shape == (r.slots, r.F)
        assert torch.equal(r.unpack(p), lg)
        assert float(p[~r.member_mask()].abs().max()) == 0 if (~r.member_mask()).any() else True
        p_, f, j, rr = r.slot_table()
        assert len(j) == r.spec.weights and torch.unique(j * L + rr).numel() == r.spec.weights


def test_reference_size_guard():
    with pytest.raises(ValueError, match="materialises"):
        RR._guard(8192, 10_485_760)


@requires_cuda_backend
def test_oversized_span_is_rejected():
    base = build_base(4096)
    with pytest.raises(ValueError, match="consecutive outputs"):
        ReductionModel(base, ReducerConfig("t", "learned", L=256, s=8), output_dim=32)


@requires_cuda_backend
def test_uncovered_tail_is_explicit():
    base = build_base(4096)
    with pytest.raises(ValueError, match="in no window"):
        ReductionModel(base, ReducerConfig("t", "learned", L=100, s=64), output_dim=32)
    m = ReductionModel(base, ReducerConfig("t", "learned", L=100, s=64), output_dim=32, allow_uncovered=True)
    assert m.reducer.routing.uncovered_neurons == 4096 - (m.reducer.spec.J - 1) * 64 - 100


@requires_cuda_backend
def test_mean_like_oracles():
    """Mean-like learned reducers equal their fixed controls; A5 windows j = 5 g are the original group means."""
    H, G, s, L = 8 * 320 * 2, 8, 64, 320  # H / G = 640 is 2 L; use G = 16 so that H / G = L
    H, G = 16 * 320, 16
    base = build_base(H, G)
    x = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16)
    a5 = ReductionModel(clone_base(base), ReducerConfig("A5", "block_mean", L=L, s=s), output_dim=32, seed=3)
    a2 = ReductionModel(clone_base(base), ReducerConfig("A2", "block_mean", L=s, s=s), output_dim=32, seed=3)
    a4 = ReductionModel(clone_base(base), ReducerConfig("A4", "learned", L=L, s=s, init="mean"), output_dim=32, seed=3)
    a3 = ReductionModel(clone_base(base), ReducerConfig("A3", "learned", L=s, s=s, init="mean"), output_dim=32, seed=3)
    a0 = ReductionModel(clone_base(base), ReducerConfig("A0", "group_mean", dense=False), output_dim=32, seed=3)
    q = lambda P, n: 1.0 / (n * float(P.detach().view(-1)[P.detach().view(-1) != 0][0].float()))
    with torch.no_grad():
        U5, U2, U4, U3, U0 = a5.features(x), a2.features(x), a4.features(x), a3.features(x), a0.features(x)
    # BF16(1 / L) is not exactly 1 / L: the declared scale of the quantized coefficient
    assert relerr(U4 * q(a4.reducer.P, L), U5) < TOL_FWD_SPLIT
    assert relerr(U3 * q(a3.reducer.P, s), U2) < TOL_FWD_SPLIT
    assert relerr(U5[:, ::5], U0) < TOL_FWD_SPLIT
    assert relerr(U5[:, 0:1], U0[:, 0:1]) < TOL_FWD_SPLIT
    # same dense V and bias -> same output (A3 / A2 have equal J)
    a3.V.data.copy_(a2.V.data)
    a3.gamma.fill_(q(a3.reducer.P, s))
    with torch.no_grad():
        assert relerr(a3(x), a2(x)) < TOL_FWD_SPLIT


@requires_cuda_backend
def test_identity_dense_preserves_function():
    base = build_base(4096)
    x = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16)
    a0 = ReductionModel(clone_base(base), ReducerConfig("A0", "group_mean", dense=False), output_dim=8)
    a1 = ReductionModel(clone_base(base), ReducerConfig("A1", "group_mean", dense=True, dense_init="identity"), output_dim=8)
    with torch.no_grad():
        assert relerr(a1(x), a0(x)) < 1e-5


@requires_cuda_backend
def test_stride_view_shares_storage_and_matches_reference():
    H, G, s = 16 * 320, 16, 64
    base = build_base(H, G)
    v = stride_view(base, s)
    assert v.weight.data_ptr() == base.weight.data_ptr() and v.indices.data_ptr() == base.indices.data_ptr()
    x = torch.randn(40, 256, device="cuda", dtype=torch.float32).requires_grad_(True)
    v.enable_main_grad()
    y = v(x)
    gy = torch.randn_like(y)
    xd = x.detach().to(torch.bfloat16).double().requires_grad_(True)
    W = base.weight.detach().double().requires_grad_(True)
    b = base.bias.detach().double().requires_grad_(True)
    h = RR.hidden(xd, base.indices, W, b, base.counts)
    yo = RR.window_outputs(h, torch.full((H // s, s), 1.0 / s, dtype=torch.float64, device="cuda"), torch.arange(H // s) * s)
    assert relerr(y, yo) < TOL_FWD_SPLIT
    dx, dw, db = torch.autograd.grad(yo, [xd, W, b], gy.double())
    torch.autograd.backward(y, gy)
    assert relerr(x.grad, dx) < TOL_GRAD
    assert relerr(v.weight.main_grad.reshape(-1), dw.reshape(-1)) < TOL_GRAD
    assert relerr(v.bias.main_grad.reshape(-1), db.reshape(-1)) < TOL_GRAD
    with pytest.raises(ValueError):
        stride_view(build_base(4004), 64)


@requires_cuda_backend
@pytest.mark.parametrize("mode", ["separate", "fused"])
def test_dp_scheduling_and_accumulation(mode):
    """Separate and fused dP agree with the oracle; repeated backward accumulates exactly once per call."""
    base = build_base(4096)
    m = ReductionModel(base, ReducerConfig("t", "learned", L=320, s=64, dp_mode=mode), output_dim=32)
    x = torch.randn(40, 256, device="cuda", requires_grad=True)
    U = m.reducer(x)
    gy = torch.randn_like(U)
    Uo, _, dwo, dbo, dpo = oracle_grads(m, x, gy)
    m.zero_main_grad()
    torch.autograd.backward(U, gy, retain_graph=True)
    assert relerr(m.reducer.P.main_grad, dpo) < TOL_GRAD
    assert relerr(m.sparse.weight.main_grad, dwo) < TOL_GRAD
    torch.autograd.backward(U, gy)
    assert relerr(m.reducer.P.main_grad, 2 * dpo) < TOL_GRAD
    assert relerr(m.sparse.weight.main_grad, 2 * dwo) < TOL_GRAD
    assert relerr(m.sparse.bias.main_grad, 2 * dbo) < TOL_GRAD
    assert m.sparse.last_info["reduction_backward"]["recomputations_of_z"] == (2 if mode == "fused" else 3)


@requires_cuda_backend
def test_finite_differences_fp64_reference_agrees_with_kernels():
    """Reference gradcheck in FP64 away from ReLU kinks, then kernels vs reference are covered above."""
    torch.manual_seed(0)
    idx = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int16).expand(2, 2, 8).clone()
    W = (torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.3).requires_grad_(True)
    b = (torch.ones(2, 2, 8, dtype=torch.float64) * 3.0).requires_grad_(True)  # all neurons active: away from the kink
    x = (torch.randn(4, 8, dtype=torch.float64) * 0.5).requires_grad_(True)
    counts = torch.tensor([16, 16])
    P = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    c = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    starts = torch.tensor([0, 8, 16])
    ref = RR.ReferenceModel(idx, counts, starts)

    def f(x_, W_, b_, P_, c_):
        return ref.forward(x_, W_, b_, P_, c_)

    assert torch.autograd.gradcheck(f, (x, W, b, P, c), eps=1e-6, atol=1e-5, rtol=1e-4)


@requires_cuda_backend
@pytest.mark.parametrize("cfg_id", ["A0", "A1", "A2", "A5", "A3", "A4", "A4gelu"])
def test_full_gradients_against_oracle_model(cfg_id):
    """Readout parameters: dV, dd, dc, dU through psi, against ordinary autograd of the materialising model."""
    H, G, L, s = 16 * 320, 16, 320, 64
    base = build_base(H, G)
    cfg = core_configs(H, L=L, s=s)[cfg_id]
    m = ReductionModel(clone_base(base), cfg, output_dim=32 if cfg.dense and cfg.dense_init != "identity" else 16, seed=5, gamma=0.7)
    x = torch.randn(40, 256, device="cuda", requires_grad=True)
    y = m(x)
    gy = torch.randn_like(y)
    torch.autograd.backward(y, gy)
    # oracle
    sp = m.sparse
    xd = x.detach().to(torch.bfloat16).double().requires_grad_(True)
    n = 16
    W = sp.weight.detach().double().reshape(-1, n, 16).requires_grad_(True)
    bb = sp.bias.detach().double().reshape(-1, n).requires_grad_(True)
    idx = sp.indices.reshape(-1, 16).long()
    z = torch.einsum("mbk,bnk->mbn", xd[:, idx], W) + bb[None]
    h = torch.relu(z).reshape(40, -1)
    if cfg.kind == "learned":
        Pl = m.reducer.routing.unpack(m.reducer.P.detach()).double().requires_grad_(True)
        U = RR.window_outputs(h, Pl, torch.arange(m.reducer.spec.J) * cfg.s)
    elif cfg.kind == "block_mean":
        Pl = None
        U = RR.window_outputs(h, torch.full(((H - cfg.L) // cfg.s + 1, cfg.L), 1.0 / cfg.L, dtype=torch.float64, device="cuda"),
                              torch.arange((H - cfg.L) // cfg.s + 1) * cfg.s)
    else:
        Pl = None
        U = h.reshape(40, G, -1).mean(2)
    ps = []
    if m.c is not None:
        cd = m.c.detach().double().requires_grad_(True)
        U = U + cd
        ps.append(cd)
    A = torch.nn.functional.gelu(U, approximate="none") if cfg.psi == "gelu" else U
    A = A * float(m.gamma)
    if m.V is not None:
        Vd, dd = m.V.detach().double().requires_grad_(True), m.d.detach().double().requires_grad_(True)
        yo = A @ Vd + dd
        ps += [Vd, dd]
    else:
        yo = A
    assert relerr(y, yo) < 2e-4
    inputs = [xd, W, bb] + ([Pl] if Pl is not None else []) + ps
    gr = torch.autograd.grad(yo, inputs, gy.double(), allow_unused=True)
    assert relerr(x.grad, gr[0]) < 3 * TOL_GRAD
    assert relerr(sp.weight.main_grad.reshape(-1), gr[1].reshape(-1)) < 3 * TOL_GRAD
    assert relerr(sp.bias.main_grad.reshape(-1), gr[2].reshape(-1)) < 3 * TOL_GRAD
    k = 3
    if Pl is not None:
        assert relerr(m.reducer.P.main_grad, m.reducer.routing.pack(gr[3])) < 3 * TOL_GRAD
        k = 4
    for p, g in zip(([m.c] if m.c is not None else []) + ([m.V, m.d] if m.V is not None else []), gr[k:]):
        assert relerr(p.grad, g) < 3 * TOL_GRAD


@requires_cuda_backend
def test_frozen_mode_trains_only_readout():
    H, G, L, s = 16 * 320, 16, 320, 64
    base = build_base(H, G)
    m = ReductionModel(base, ReducerConfig("A4", "learned", L=L, s=s), output_dim=32, train_sparse=False)
    w0 = m.sparse.weight.detach().clone()
    x = torch.randn(40, 256, device="cuda")
    y = m(x)
    y.square().mean().backward()
    assert m.sparse.weight.main_grad.abs().max() == 0 and m.sparse.bias.main_grad.abs().max() == 0
    assert m.reducer.P.main_grad.abs().max() > 0
    info = m.sparse.last_info["reduction_backward"]
    assert info == dict(info, dx=False, dw=False, dp=True, recomputations_of_z=1)
    assert torch.equal(m.sparse.weight.detach(), w0)
    assert m.fingerprints()["indices"]


@requires_cuda_backend
def test_leading_shape_restoration_and_token_constraint():
    H, G, L, s = 16 * 320, 16, 320, 64
    m = ReductionModel(build_base(H, G), ReducerConfig("A4", "learned", L=L, s=s), output_dim=32)
    x = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 16, 32)
    y.float().square().mean().backward()
    assert x.grad.shape == x.shape and x.grad.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="B % 8"):
        m(torch.randn(13, 256, device="cuda"))


@requires_cuda_backend
def test_deterministic_request_is_rejected():
    from sparse_pooling.config import PhysicalPlan

    H, G, L, s = 16 * 320, 16, 320, 64
    m = ReductionModel(build_base(H, G), ReducerConfig("A4", "learned", L=L, s=s), output_dim=32)
    m.sparse.plan = PhysicalPlan(grad_precision="split", reduction="deterministic")
    with pytest.raises(ValueError, match="deterministic"):
        m(torch.randn(8, 256, device="cuda"))


@requires_cuda_backend
def test_optimizer_checkpoint_resume_and_identity(tmp_path):
    H, G, L, s = 16 * 320, 16, 320, 64
    base = build_base(H, G)
    cfg = ReducerConfig("A4", "learned", L=L, s=s)

    def make():
        return ReductionModel(clone_base(base), cfg, output_dim=32, seed=9)

    def opts(m):
        o = FusedAdamW(m.sparse_parameters() + m.reducer_parameters(), lr=1e-3, moments="bf16", stochastic_rounding=False)
        o2 = torch.optim.AdamW(m.dense_parameters(), lr=1e-3, weight_decay=0.0)
        return o, o2

    def step(m, o, o2, x):
        o2.zero_grad(set_to_none=True)
        m(x).square().mean().backward()
        o.step()
        o2.step()

    x = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16)
    a = make()
    assert a.reducer.P.main_grad.abs().max() == 0
    oa, oa2 = opts(a)
    fp0 = a.fingerprints()
    p0 = a.reducer.P.detach().clone()
    step(a, oa, oa2, x)
    assert a.reducer.P.main_grad.abs().max() == 0, "FusedAdamW must consume and zero main_grad"
    assert not torch.equal(a.reducer.P.detach(), p0), "P must move"
    member = a.reducer.routing.member_mask("cuda")
    if (~member).any():
        assert a.reducer.P.detach()[~member].float().abs().max() == 0, "padded slots stay zero"
    assert a.fingerprints() == fp0, "the graph is immutable"
    assert sum(1 for _ in a.reducer_parameters()) == 1
    path = tmp_path / "state.pt"
    torch.save(dict(model=a.state_dict(), o=oa.state_dict(), o2=oa2.state_dict(), seed=oa._seed), path)
    b = make()
    blob = torch.load(path, weights_only=False)
    b.load_state_dict(blob["model"])
    ob, ob2 = opts(b)
    ob.load_state_dict(blob["o"])
    ob2.load_state_dict(blob["o2"])
    ob._seed = blob["seed"]
    assert torch.equal(b.reducer.P.detach(), a.reducer.P.detach())
    step(a, oa, oa2, x)
    step(b, ob, ob2, x)
    assert relerr(b.reducer.P.detach().float(), a.reducer.P.detach().float()) < 1e-2
    assert relerr(b.V.detach(), a.V.detach()) < 1e-4
    # incompatible routing is rejected
    other = ReductionModel(clone_base(base), ReducerConfig("A3", "learned", L=s, s=s), output_dim=32, seed=9)
    with pytest.raises((ValueError, RuntimeError)):
        other.load_state_dict(a.state_dict())


@requires_cuda_backend
def test_second_stage_matches_reference():
    H, G, L, s = 16 * 320, 16, 320, 64
    base = build_base(H, G)
    J1 = (H - L) // s + 1
    L2 = 16
    s2 = (J1 - L2) % 4 == 0 and 4 or 1
    cfg = ReducerConfig("TS", "learned", L=L, s=s, psi="gelu", second=(L2, s2, "gelu"))
    m = ReductionModel(base, cfg, output_dim=32, seed=1)
    x = torch.randn(40, 256, device="cuda", requires_grad=True)
    y = m(x)
    gy = torch.randn_like(y)
    torch.autograd.backward(y, gy)
    a = torch.randn(40, J1, device="cuda", dtype=torch.float64, requires_grad=True)
    st = m.second
    P2 = st.P2.detach().double().requires_grad_(True)
    from sparse_pooling.experimental.learned_reduction import _Stage2Function

    u = _Stage2Function.apply(a, P2, st.s2, 7)
    u0 = RR.window_outputs(a, P2, torch.arange(st.J2) * st.s2)
    assert relerr(u, u0) < 1e-12
    g2 = torch.randn_like(u)
    da, dp = torch.autograd.grad(u, [a, P2], g2)
    da0, dp0 = torch.autograd.grad(u0, [a, P2], g2)
    assert relerr(da, da0) < 1e-12 and relerr(dp, dp0) < 1e-12
    assert st.P2.grad is not None and st.c2.grad is not None


@requires_cuda_backend
@pytest.mark.parametrize("cfg_id", ["A4", "MC", "A3"])
def test_numerical_edge_cases(cfg_id):
    """Zero inputs, all-zero hidden activations, near-kink preactivations and large finite values stay finite and
    match the oracle where the oracle is finite."""
    H, G, L, s = 16 * 320, 16, 320, 64
    base = build_base(H, G)
    cfg = core_configs(H, L=L, s=s)[cfg_id]
    m = ReductionModel(base, cfg, output_dim=32)
    # zero input with zero bias: z = 0 exactly (the ReLU kink: inactive by the derivative convention)
    with torch.no_grad():
        m.sparse.bias.zero_()
    x = torch.zeros(16, 256, device="cuda")
    U = m.reducer(x)
    assert U.abs().max() == 0
    torch.autograd.backward(U, torch.ones_like(U))
    assert m.reducer.P.main_grad.abs().max() == 0
    # all-zero hidden: negative bias
    with torch.no_grad():
        m.sparse.bias.fill_(-1e4)
    x = torch.randn(16, 256, device="cuda")
    U = m.reducer(x)
    assert U.abs().max() == 0
    # large finite values
    with torch.no_grad():
        m.sparse.bias.zero_()
    x = torch.randn(16, 256, device="cuda") * 1e6
    U = m.reducer(x)
    assert torch.isfinite(U).all()
    assert relerr(U, oracle_grads(m, x, torch.ones_like(U))[0]) < 10 * TOL_FWD_SPLIT


@requires_cuda_backend
def test_no_m_by_h_tensor_in_forward_or_backward():
    """Peak allocation over a step stays far below any M x H tensor (bytes), including autograd-saved tensors."""
    H, G, L, s = 1024 * 2560, 1024, 10_240, 2_048
    from sparse_pooling.experimental.kgen_layout import init_parameters_by_neuron

    LL = KLayout(1024, H, G, 64, 256)
    w, b = init_parameters_by_neuron(LL, "cuda", seed=3)
    base = CoverTopK16Layer.from_envelope(w, b, input_dim=1024, hidden_dim=H, output_dim=G,
                                          connectivity=ConnectivityPlan(64, 256, seed=0), normalization="energy")
    del w, b
    m = ReductionModel(base, ReducerConfig("A4", "learned", L=L, s=s), output_dim=1024)
    M = 2048
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.empty_cache()
    m.reducer._exec()
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = m(x)
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base_alloc
    assert peak < 0.05 * (M * H * 2), f"step scratch {peak / 2**30:.2f} GiB vs M x H BF16 {M * H * 2 / 2**30:.1f} GiB"
