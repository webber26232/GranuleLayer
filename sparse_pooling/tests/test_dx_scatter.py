"""Grouped C = 8 dX local-accumulation variants (kernels/cuda/sparse_pool_sg_dxs.cuh,
PhysicalPlan.cuda_grouped_dx_scatter).

* layout diagnostics on the device offsets: bijection, record alignment / contiguity, lane
  ownership, computed bank-conflict degree of every shared access phase;
* fixed upstream gradient: dW / db bitwise equal to the baseline grouped dX run (the dW kernel
  is unchanged), dX within 1e-6 of the baseline and within the strict reference tolerances;
* deterministic setting (R = 1 and one chunk per supergroup: one CTA per (feature, token
  tile), so the global dX flush is a single atomic onto zero): the order-preserving variants
  are BITWISE equal to the baseline, including NaN / inf / signed-zero / subnormal cases;
  feature_owner (summation inside the MMA) is characterised, not bitwise;
* ragged / tiny / empty buckets, token tails, Mean(identity), BF16 gradients, repeated
  backward accumulation, explicit-unsupported errors and auto fallbacks.
"""
import itertools

import pytest
import torch

from conftest import active_profile_name, rel_err, requires_cuda_backend
from sparse_pooling import PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from test_input_supergroups import inputs, reference, run, sg_layer

SAME = ["baseline_private", "swizzle4", "vectorized", "vectorized_transpose", "vectorized_hoisted",
        "vectorized_transpose_hoisted", "vectorized_parity"]
ALL = SAME + ["feature_owner"]
SC = {k: v["sc"] for k, v in cuda_ops.SG_DX_SCATTERS.items() if v["sc"] != 4}
SC["swizzle4_private"] = 6


def dx_plan(scatter, T=64, nw=8, wg=1, precision="split", deterministic=False, **kw):
    kw = dict(cuda_input_supergroup_backend="grouped", grad_precision=precision, cuda_grouped_dx_tokens=T,
              cuda_grouped_dx_warps=nw, cuda_grouped_dx_reduction=wg, cuda_grouped_dx_scatter=scatter, **kw)
    if deterministic:
        kw["cuda_grouped_dx_chunk_units"] = 1 << 30  # one chunk (CTA column) per supergroup
    return PhysicalPlan(**kw)


def backend():
    return cuda_ops.CudaBackend.get()


# --------------------------------------------------------------------------------------
# layouts (device offsets)
# --------------------------------------------------------------------------------------
def lane_records(name, tokens, mp_or_mt, rec):
    """(lane -> list of (local feature, local token) words) of one shared access instruction.
    rec: the block's 8 distinct local features by fan-in position."""
    out = {}
    for lane in range(32):
        t, c = lane >> 2, lane & 3
        words = []
        for e in (0, 1):  # p0 / p1 instructions are separate; take both column parities
            if name.startswith("vectorized_transpose"):
                f, tok = rec[2 * (lane >> 3) + e], lane & 7
            else:
                f, tok = rec[2 * c + e], t
            if name.startswith("vectorized"):
                base = 32 * mp_or_mt
                words.append([(f, base + tok), (f, base + tok + 8), (f, base + 16 + tok), (f, base + 24 + tok)])
            else:
                base = 16 * mp_or_mt
                words.append([(f, base + tok), (f, base + tok + 8)])
        out[lane] = words
    return out


@requires_cuda_backend
@pytest.mark.parametrize("name", ["baseline_private", "swizzle4_private", "vectorized", "vectorized_transpose",
                                  "vectorized_parity", "vectorized_hoisted", "vectorized_transpose_hoisted"])
@pytest.mark.parametrize("tokens", [32, 64, 128])
@pytest.mark.parametrize("S", [8, 16, 32, 64])
def test_layout_bijection_alignment_ownership(name, tokens, S):
    off = backend().sgx_layout(SC[name], tokens, S).cpu()
    n = S * tokens
    assert sorted(off.flatten().tolist()) == list(range(n))  # bijection onto the copy
    gen = torch.Generator().manual_seed(S * tokens)
    vec = name.startswith("vectorized")
    width = 4 if vec else 2
    phase = 8 if vec else 16  # lanes per shared-memory wavefront phase (128-bit / 64-bit)
    for _ in range(64):  # random records
        rec = torch.randperm(S, generator=gen)[:8].tolist() if S >= 8 else None
        for blk in range(tokens // (32 if vec else 16)):
            lr = lane_records(name, tokens, blk, rec)
            for e in (0, 1):
                seen = set()
                for lane in range(32):
                    ws = [int(off[f, t]) for f, t in lr[lane][e]]
                    assert ws[0] % width == 0 and ws == list(range(ws[0], ws[0] + width)), (lane, ws)  # aligned record
                    assert not (set(ws) & seen)  # no two lanes touch the same word: race-free
                    seen |= set(ws)
                # computed conflict degree of each phase (distinct addresses per bank)
                for p0 in range(0, 32, phase):
                    banks = {}
                    for lane in range(p0, p0 + phase):
                        for w in (int(off[f, t]) for f, t in lr[lane][e]):
                            banks.setdefault(w % 32, set()).add(w)
                    deg = max(len(v) for v in banks.values())
                    feats = [f for lane in range(p0, p0 + phase) for f, _ in lr[lane][e][:1]]
                    if name.startswith("vectorized_transpose"):
                        assert deg == 1
                    elif name in ("vectorized", "vectorized_hoisted", "swizzle4_private"):
                        assert deg == max(sum(1 for f in set(feats) if f % 4 == g) for g in range(4))
                    else:  # 2 bank groups (feature parity)
                        assert deg == max(sum(1 for f in set(feats) if f % 2 == g) for g in range(2))


# --------------------------------------------------------------------------------------
# fixed upstream gradient
# --------------------------------------------------------------------------------------
CASES = [  # (S, R, dX tokens) with warp-private copies that fit
    (8, 2, 128), (16, 4, 64), (16, 16, 64), (32, 8, 32), (32, 8, 64), (64, 16, 32)]


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("S,R,T", CASES)
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_fixed_dy_matches_baseline(scatter, S, R, T, precision):
    if scatter == "feature_owner" and S > 32:
        pytest.skip("feature_owner: S <= 32")
    D, H, G, B = 1024, 300_000, 64, 200
    lay = sg_layer(D, H, G, 8, S, R, plan=dx_plan("baseline", T, precision=precision))
    x, dy = inputs(B, D, G)
    yb, dxb, dwb, dbb, ib = run(lay, x, dy)
    lay.plan = dx_plan(scatter, T, precision=precision)
    yc, dxc, dwc, dbc, ic = run(lay, x, dy)
    gd = ic["grouped_dx"]
    assert ib["grouped_dx"]["scatter"] == "baseline" and gd["scatter"] == scatter and ic["dx_kernel"] == "grouped"
    assert gd["scatter_order"] == cuda_ops.SG_DX_SCATTERS[scatter]["order"] and gd["tokens"] == T
    assert torch.equal(dwc, dwb) and torch.equal(dbc, dbb)
    assert rel_err(dxc, dxb) < 1e-6 and rel_err(yc, yb) < 1e-6
    _, dxr, _, _ = reference(lay, x, dy)
    assert rel_err(dxc, dxr) < (2e-5 if precision == "split" else 5e-3)


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("S,T", [(8, 64), (16, 32), (16, 64), (16, 128), (32, 32), (32, 64), (64, 32)])
@pytest.mark.parametrize("nw", [8, 16])
def test_deterministic_bitwise(scatter, S, T, nw):
    if scatter == "feature_owner" and S > 32:
        pytest.skip("feature_owner: S <= 32")
    D, H, G, B = 1024, 200_000, 64, 136  # token tail inside the last tile
    lay = sg_layer(D, H, G, 8, S, 1, plan=dx_plan("baseline", T, nw, deterministic=True))
    x, dy = inputs(B, D, G)
    try:
        _, dxb, dwb, dbb, _ = run(lay, x, dy)
    except ValueError as e:
        assert "shared memory" in str(e)
        pytest.skip("does not fit")
    lay.plan = dx_plan(scatter, T, nw, deterministic=True)
    _, dxc, dwc, dbc, _ = run(lay, x, dy)
    assert torch.equal(dwc, dwb) and torch.equal(dbc, dbb)
    if scatter == "feature_owner":
        assert rel_err(dxc, dxb) < 1e-6 and not torch.equal(dxc, dxb)  # different FP32 order
    else:
        assert torch.equal(dxc.view(torch.int32), dxb.view(torch.int32))


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("case", ["zero_bias_zero_x", "huge", "tiny", "subnormal", "inf", "nan", "neg_zero",
                                  "w_inf"])
def test_extremes_deterministic(scatter, case):
    D, H, G, B = 256, 20_000, 8, 64
    lay = sg_layer(D, H, G, 8, 16, 1, plan=dx_plan("baseline", 64, deterministic=True))
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
    elif case == "inf":
        dy[3, 2] = float("inf")
        dy[5, 1] = -float("inf")
    elif case == "nan":
        dy[7, 0] = float("nan")
    elif case == "neg_zero":
        dy = torch.where(dy > 0, dy, torch.full_like(dy, -0.0))
    elif case == "w_inf":  # non-finite weight: inf * 0 products must not reach absent features
        with torch.no_grad():
            lay.weight.view(-1)[1234] = float("inf")
    _, dxb, dwb, dbb, _ = run(lay, x, dy)
    lay.plan = dx_plan(scatter, 64, deterministic=True)
    _, dxc, dwc, dbc, _ = run(lay, x, dy)
    assert torch.equal(dwc.view(torch.int32), dwb.view(torch.int32))
    assert torch.equal(dbc.view(torch.int32), dbb.view(torch.int32))
    assert torch.equal(torch.isnan(dxc), torch.isnan(dxb)) and torch.equal(torch.isinf(dxc), torch.isinf(dxb))
    if scatter == "feature_owner":
        fin = torch.isfinite(dxb)
        assert rel_err(dxc[fin], dxb[fin]) < 1e-6
        assert torch.equal(torch.signbit(dxc[dxb == 0]), torch.signbit(dxb[dxb == 0]))
    else:
        assert torch.equal(dxc.view(torch.int32), dxb.view(torch.int32))


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("B", [16, 24, 40, 64, 72, 120, 136, 264])
def test_token_tails(scatter, B):
    D, H, G = 512, 33_333, 8
    lay = sg_layer(D, H, G, 8, 16, 4, plan=dx_plan("baseline", 64))
    x, dy = inputs(B, D, G)
    _, dxb, dwb, dbb, _ = run(lay, x, dy)
    lay.plan = dx_plan(scatter, 64)
    _, dxc, dwc, dbc, _ = run(lay, x, dy)
    assert torch.equal(dwc, dwb) and torch.equal(dbc, dbb) and rel_err(dxc, dxb) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("H,G", [(1000, 64), (64 * 8 + 3, 8), (999_999, 1000), (100, 100), (8 * 7 + 5, 3)])
def test_tiny_empty_buckets_ragged(scatter, H, G):
    D, B = 1024, 64
    lay = sg_layer(D, H, G, 8, 16, 16, plan=dx_plan("baseline", 32))
    x, dy = inputs(B, D, G)
    _, dxb, dwb, dbb, _ = run(lay, x, dy)
    lay.plan = dx_plan(scatter, 32)
    _, dxc, dwc, dbc, _ = run(lay, x, dy)
    assert torch.equal(dwc, dwb) and torch.equal(dbc, dbb) and rel_err(dxc, dxb) < 1e-6
    _, dxr, _, _ = reference(lay, x, dy)
    assert rel_err(dxc, dxr) < 2e-5


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ALL)
@pytest.mark.parametrize("precision", ["split", "bf16"])
def test_mean_identity(scatter, precision):
    D, H, G, B = 256, 40_000, 8, 64
    lay = sg_layer(D, H, G, 8, 32, 8, pooling={"type": "mean", "activation": "identity"},
                   plan=dx_plan("baseline", 32, precision=precision))
    x, dy = inputs(B, D, G)
    _, dxb, dwb, dbb, _ = run(lay, x, dy)
    lay.plan = dx_plan(scatter, 32, precision=precision)
    _, dxc, dwc, dbc, info = run(lay, x, dy)
    assert info["grouped_dx"]["scatter"] == scatter
    assert torch.equal(dwc, dwb) and torch.equal(dbc, dbb) and rel_err(dxc, dxb) < 1e-6


@requires_cuda_backend
@pytest.mark.parametrize("scatter", ["swizzle4", "swizzle4_capped"])
@pytest.mark.parametrize("wg", [1, 2, 4, 8])
def test_shared_copy_swizzle(scatter, wg):
    D, H, G, B = 1024, 200_000, 64, 128
    lay = sg_layer(D, H, G, 8, 64, 16, plan=dx_plan("baseline", 64, wg=wg))
    x, dy = inputs(B, D, G)
    try:
        _, dxb, dwb, _, _ = run(lay, x, dy)
    except ValueError as e:
        assert "shared memory" in str(e)
        pytest.skip("does not fit")
    lay.plan = dx_plan(scatter, 64, wg=wg)
    _, dxc, dwc, _, info = run(lay, x, dy)
    assert info["grouped_dx"]["reduction_group_warps"] == wg and info["grouped_dx"]["scatter"] == scatter
    sc = {"swizzle4": 6 if wg == 1 else 1, "swizzle4_capped": 10}[scatter]  # SC_SWZ4P only for private copies
    assert info["grouped_dx"]["kernel"].endswith(f", {sc}, 1>")
    assert torch.equal(dwc, dwb) and rel_err(dxc, dxb) < 1e-6


@requires_cuda_backend
def test_repeated_backward_accumulation():
    D, H, G, B = 1024, 60_000, 16, 64
    out = {}
    for sc in ("baseline", "vectorized", "feature_owner"):
        lay = sg_layer(D, H, G, 8, 16, 4, plan=dx_plan(sc, 64), fp32=False)
        lay.enable_main_grad()
        x = inputs(B, D, G)[0].bfloat16().requires_grad_(True)
        for s in (1, 2, 3):
            (lay(x).float() * inputs(B, D, G, seed=s)[1]).sum().backward()
        out[sc] = (lay.weight.main_grad.clone(), lay.bias.main_grad.clone(), x.grad.float().clone())
    for sc in ("vectorized", "feature_owner"):
        assert torch.equal(out[sc][0], out["baseline"][0]) and torch.equal(out[sc][1], out["baseline"][1])
        assert rel_err(out[sc][2], out["baseline"][2]) < 1e-2  # BF16 dX accumulated by autograd


# --------------------------------------------------------------------------------------
# selection, explicit errors, auto fallbacks
# --------------------------------------------------------------------------------------
@requires_cuda_backend
def test_explicit_unsupported_raise():
    x, dy = inputs(64, 1024, 16)
    lay = sg_layer(1024, 50_000, 16, 8, 64, 16, plan=dx_plan("vectorized", 64, wg=8))
    with pytest.raises(ValueError, match="warp-private"):
        run(lay, x, dy)
    lay.plan = dx_plan("feature_owner", 32)
    with pytest.raises(ValueError, match="S <= 32"):
        run(lay, x, dy)
    lay.plan = dx_plan("vectorized", 16, nw=16)
    with pytest.raises(ValueError, match=">= 2 m16 token tiles"):
        run(lay, x, dy)
    lay64 = sg_layer(1024, 50_000, 16, 64, 16, 4, plan=dx_plan("vectorized", 64))
    with pytest.raises(ValueError, match="C = 8 only"):
        run(lay64, x, dy)


AUTO_EXPECTED = {(8, 16, "split", "relu"): "vectorized_transpose_hoisted", (8, 16, "bf16", "relu"): "vectorized_transpose_hoisted",
                 (8, 32, "split", "relu"): "vectorized_transpose_hoisted", (8, 32, "bf16", "relu"): "vectorized_transpose_hoisted",
                 (8, 64, "split", "relu"): "swizzle4_capped"}


@requires_cuda_backend
@pytest.mark.parametrize("C,S,R", [(8, 8, 2), (8, 16, 4), (8, 32, 8), (8, 64, 16), (64, 16, 4), (8, 128, 32)])
@pytest.mark.parametrize("precision", ["split", "bf16"])
@pytest.mark.parametrize("act", ["relu", "identity"])
def test_auto_scatter(C, S, R, precision, act):
    """auto: the measured variant exactly where it was measured (auto tiles, Mean(ReLU)),
    the baseline everywhere else; the selection is recorded with the kernel identity."""
    lay = sg_layer(1024, 50_000, 16, C, S, R, pooling={"type": "mean", "activation": act},
                   plan=PhysicalPlan(cuda_input_supergroup_backend="auto", grad_precision=precision))
    x, dy = inputs(128, 1024, 16)
    _, dxa, dwa, _, info = run(lay, x, dy)
    if info["dx_kernel"] != "grouped":
        return
    gd = info["grouped_dx"]
    measured = AUTO_EXPECTED if active_profile_name() == "rtx3090" else {}  # other profiles: baseline
    want = measured.get((C, S, precision, act), "baseline")
    assert gd["scatter"] == want and gd["scatter_fallback"] == ""
    assert gd["kernel"].startswith("sg_dx_kernel" if want == "baseline" else "sgx_dx_kernel")
    lay.plan = PhysicalPlan(cuda_input_supergroup_backend="auto", grad_precision=precision,
                            cuda_grouped_dx_scatter="baseline")
    _, dxb, dwb, _, ib = run(lay, x, dy)
    assert ib["grouped_dx"]["scatter"] == "baseline" and torch.equal(dwa, dwb) and rel_err(dxa, dxb) < 1e-6
