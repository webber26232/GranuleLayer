"""EXPERIMENTAL width-preserving sparsity (experimental/width_masks.py, kgx_wide.py, kgx_onetile.py,
canonical.py; results/WIDTH_PRESERVING_SPARSITY.md).

Correctness gates before any full-size timing: policy budgets / constraints / feasibility (incl. the
assignment's skeleton numbers and the explicit infeasibility rejections), the new 16 x 8 / 16 x 16 /
backbone + tile kernels against a float64 reference of the SAME masked dense function and against the dense
masked kgx kernels (full-z activation, dX, active dW, db, main_grad accumulation, partial row / C / token
tails, all-zero / all-one / one-tile / anchor / empty-extra supports, overlapping backbone coordinates),
the K16 / K8 lowered control (identical function), canonical identity across every representation
(signed zero, padding, optimizer moments), optimizer updates of backbone / tile / bias parameters with
structural zeros kept, complete snapshot and checkpoint round trips.
"""
import math

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.supergroups import ConnectivityPlan

pytestmark = requires_cuda_backend


def _m():
    from sparse_pooling.experimental import (canonical, kgen_layout, kgx_layer, kgx_onetile, kgx_sparse, kgx_wide,
                                             l1, reference, width_masks)

    return dict(C=canonical, KL=kgen_layout, KX=kgx_layer, OT=kgx_onetile, SP=kgx_sparse, W=kgx_wide, L1=l1,
                REF=reference, WM=width_masks)


def base_layer(D=256, H=3003, G=8, C=64, R=2, seed=1, init_seed=3, bias_std=0.3, prec="split"):
    m = _m()
    lay = m["KX"].KgxLayer(D, H, G, C, ConnectivityPlan(64, R, seed=seed), variant="idf", tile=128, device="cuda",
                           seed=init_seed, plan=PhysicalPlan(grad_precision=prec), keep_indices=True)
    g = torch.Generator(device="cuda").manual_seed(init_seed + 1)
    with torch.no_grad():
        lay.bias.copy_((bias_std * torch.randn(lay.bias.shape, device="cuda", generator=g)).bfloat16())
        lay.bias.mul_(torch.where(torch.rand(lay.bias.shape, device="cuda", generator=g) < 0.5, -1.0, 1.0).bfloat16())
    m["KL"].zero_padding_(lay.layout, lay.weight.data, lay.bias.data)
    return lay


def masked_dense(base, em, prec="split"):
    m = _m()
    L = base.layout
    d = m["KX"].KgxLayer(L.input_dim, L.hidden_dim, L.groups, L.block, base.connectivity, variant="idf", tile=128,
                         device="cuda", plan=PhysicalPlan(grad_precision=prec),
                         weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=True)
    d.mask = em
    return d


def inputs(B, D, G, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(B, D, device="cuda", generator=g).bfloat16().float(),
            torch.randn(B, G, device="cuda", generator=g))


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


def run(lay, x, dy):
    lay.enable_main_grad()
    lay.zero_main_grad()
    xx = x.detach().clone().requires_grad_(True)
    y = lay(xx)
    y.backward(dy)
    return y.detach(), xx.grad.detach()


def canon_grads(lay):
    """(dW rows [H, 64], db [H]) of main_grad in canonical order."""
    C = _m()["C"]
    src = C.sources(lay, "main_grad")
    g, n = C.neurons(lay.counts, 0, lay.counts.numel())
    return lay.canon_rows(src, g, n)


def ref_grads(dm, x, dy):
    """FP64 (y, dX, dW rows, db) of the dense masked model, dW restricted to its enabled support."""
    m = _m()
    w = dm.weight.detach().double()
    y, dx, dw, db = m["REF"].grads(x.double(), dm.logical_indices().long(), w, dm.bias.detach().double(), dm.counts,
                                   dy.double())
    dw = dw * dm.mask.double()
    g, n = m["C"].neurons(dm.counts, 0, dm.counts.numel())
    s = m["C"].dense_slots(dm, g, n)
    return y, dx, dw.reshape(-1, 64)[s], db.reshape(-1)[s]


POLICY_CASES = [("U16", 0.3), ("P16", 0.3), ("A16", 0.45), ("U8", 0.2), ("P8", 0.2), ("A8", 0.25), ("H2", 0.06),
                ("H2", 0.15), ("H2A", 0.2), ("P16", 0.25), ("P8", 0.125)]


# ----------------------------------------------------------------------------------------------- policies
@pytest.mark.parametrize("pol,rho", POLICY_CASES)
@pytest.mark.parametrize("score", ["magnitude", "random"])
def test_policy_budget_and_constraints(pol, rho, score):
    m = _m()
    WM = m["WM"]
    lay = base_layer(H=3003, C=128)  # n_g = 375 / 376: partial row groups and partial C128 tails
    L = lay.layout
    ms = WM.build(pol, rho, lay.weight.detach(), L, score=score, seed=4)
    st = ms.stats
    P = WM.POLICIES[pol]
    nv = ms.nv
    gv = nv > 0
    assert st["W0"] == 64 * L.hidden_dim
    assert st["budget_slots"] == round(rho * st["W0"])
    assert 0 <= st["budget_shortfall"] < 16 * P.tb  # within one tile below the exact budget
    em = ms.element_mask(L)
    assert int(em.sum()) == st["actual_slots"] == ms.enabled_slots()
    assert not bool(ms.tiles[~gv].any()) and not bool(ms.anchor[~gv].any())
    if P.protect:
        assert bool((ms.tiles[gv].sum(1) >= 1).all())
    if P.anchors:
        per = ms.anchor.view(-1, L.block // 16)
        blk_valid = nv.view(-1, L.block // 16).sum(1) > 0
        assert bool((per.sum(1) == blk_valid.to(torch.int64)).all())  # exactly one anchor per valid record
        full = nv.view(-1, L.block // 16) == 16
        has_full = full.any(1)
        assert bool((per[has_full] <= full[has_full]).all())  # among full groups where available
        assert bool(ms.tiles[ms.anchor].all())
        assert st["anchor_neurons"] == st["skeleton"]["anchor_neurons"]
    if P.backbone:
        assert ms.bb is not None and bool((ms.bb[:, 0] < ms.bb[:, 1]).all())
        deg = em.reshape(-1, 16, 64).sum(2)
        valid = L.valid_mask(em.device).reshape(-1, 16)
        assert bool((deg[valid] >= 2).all())
    deg = em.reshape(-1, 64).sum(1)[L.valid_mask(em.device).reshape(-1)]
    assert torch.bincount(deg, minlength=65).tolist() == st["mask_degree_hist"]
    assert st["structural_bias_only_neurons"] == int((deg == 0).sum())
    # deterministic
    ms2 = WM.build(pol, rho, lay.weight.detach(), L, score=score, seed=4)
    assert torch.equal(ms.tiles, ms2.tiles) and torch.equal(ms.anchor, ms2.anchor)


def test_magnitude_protection_picks_the_best_tile_and_anchor():
    WM = _m()["WM"]
    lay = base_layer(H=2048, G=8, C=128)
    L = lay.layout
    ms = WM.build("A8", 0.25, lay.weight.detach(), L)
    n16 = ms.nv.numel()
    a = lay.weight.detach().float().abs().view(n16, 16, 8, 8).sum((1, 3))
    nonanchor = (ms.nv > 0) & ~ms.anchor
    assert bool(ms.mandatory[nonanchor, :].sum(1).eq(1).all())
    pick = torch.argmax(ms.mandatory[nonanchor].to(torch.int8), 1)
    assert torch.equal(pick, torch.argmax(a[nonanchor], 1))
    tot = lay.weight.detach().float().abs().view(-1, L.block // 16, 16 * 64).sum(2)
    full = ms.nv.view(-1, L.block // 16) == 16
    best = torch.argmax(torch.where(full, tot, torch.full_like(tot, -1.0)), 1)
    assert torch.equal(torch.argmax(ms.anchor.view(-1, L.block // 16).to(torch.int8), 1), best)


def test_full_size_skeletons_and_infeasible_requests():
    WM, KL = _m()["WM"], _m()["KL"]
    L256 = KL.KLayout(1024, 12_500_000, 1024, 64, 256)
    c = WM.layout_counts(L256, "cpu")
    assert c["records"] == 49_152 and c["anchor_neurons"] == 786_432 and c["W0"] == 800_000_000
    sk = {p: WM.skeleton(WM.POLICIES[p], L256, counts=c)["skeleton_slots"] for p in ("A16", "A8", "H2A", "P16", "P8",
                                                                                     "H2")}
    assert sk["A16"] == 237_748_736 and sk["A8"] == 144_040_192 and sk["H2A"] == 73_758_784
    assert sk["P16"] == 200_000_000 and sk["P8"] == 100_000_000 and sk["H2"] == 25_000_000
    for pol, rho in (("P16", 0.10), ("P8", 0.05), ("H2A", 0.05), ("A16", 0.25), ("A8", 0.125)):
        f = WM.feasibility(WM.POLICIES[pol], rho, L256, counts=c)
        assert not f["feasible"] and "budget" in f["reason"]
    for pol, rho in (("P16", 0.25), ("P8", 0.125), ("A8", 0.1875), ("A16", 0.375), ("H2", 0.05), ("H2A", 0.10)):
        assert WM.feasibility(WM.POLICIES[pol], rho, L256, counts=c)["feasible"]
    L2048 = KL.KLayout(1024, 12_500_000, 1024, 64, 2048)
    c2 = WM.layout_counts(L2048, "cpu")
    f = WM.feasibility(WM.POLICIES["H2A"], 0.05, L2048, counts=c2)
    assert f["feasible"] and f["skeleton_slots"] == 31_094_848
    lay = base_layer(H=3003, C=128)
    with pytest.raises(WM.Infeasible):
        WM.build("P16", 0.10, lay.weight.detach(), lay.layout)
    with pytest.raises(WM.Infeasible):
        WM.build("P8", 0.05, lay.weight.detach(), lay.layout)
    b = WM.describe_bounds(4096, 16)
    assert abs(b[3] - 0.051420) < 1e-5 and abs(b[4] - 0.0006548) < 1e-6
    b8 = WM.describe_bounds(4096, 8)
    assert abs(b8[3] - 0.010285) < 1e-5 and abs(b8[4] - 0.0000504) < 1e-7


# ----------------------------------------------------------------------------------------------- kernels
def _formats(m, base, ms):
    """(compact representations of the support) for a mask set: wide always; KgxTileLayer for b16 w/o backbone."""
    out = [m["W"].KgxWideLayer(base, ms)]
    if ms.tb == 16 and ms.bb is None:
        out.append(m["SP"].KgxTileLayer(base, ms.tiles))
    return out


@pytest.mark.parametrize("pol,rho", POLICY_CASES)
@pytest.mark.parametrize("prec", ["split", "bf16"])
def test_wide_formats_against_float64_and_the_dense_masked_kernels(pol, rho, prec):
    m = _m()
    base = base_layer(H=3003, C=128, prec=prec)
    L = base.layout
    ms = m["WM"].build(pol, rho, base.weight.detach(), L)
    em = ms.element_mask(L)
    dm = masked_dense(base, em, prec)
    x, dy = inputs(40, L.input_dim, L.groups)  # B = 40: token tails of every tiling
    yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
    yd, dxd = run(dm, x, dy)
    tol = 2e-5 if prec == "split" else 2e-2
    for sp in _formats(m, dm, ms):
        w_, b_ = sp.dense_params()
        assert torch.equal(w_.view(torch.int16) & 0x7FFF, dm.weight.detach().view(torch.int16) & 0x7FFF)
        assert torch.equal(sp.element_mask().bool(), em.bool())
        y, dx = run(sp, x, dy)
        dw, db = canon_grads(sp)
        dwm = dw.double() * m["C"].dense_mask_rows(dm, *m["C"].neurons(dm.counts, 0, dm.counts.numel())).double()
        assert maxrel(y, yr) < 1e-6, (pol, type(sp).__name__)
        assert maxrel(dx, dxr) < tol and maxrel(dwm, dwr) < tol
        assert maxrel(db, dbr) < 1e-6
        assert maxrel(y, yd) < 1e-6 and maxrel(dx, dxd) < tol


def test_main_grad_accumulates_and_backbone_only_groups():
    m = _m()
    base = base_layer(H=3003, C=128)
    L = base.layout
    sk = m["WM"].skeleton(m["WM"].POLICIES["H2"], L)
    rho = sk["skeleton_slots"] / sk["W0"]  # exactly the backbone: no extra tile at all
    ms = m["WM"].build("H2", rho, base.weight.detach(), L)
    assert ms.stats["extra_tiles"] == 0 and not bool(ms.tiles.any())
    sp = m["W"].KgxWideLayer(base, ms)
    assert sp.weight.shape[0] == 0 or sp.weight.numel() == 0
    x, dy = inputs(24, L.input_dim, L.groups)
    dm = masked_dense(base, ms.element_mask(L))
    yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
    sp.enable_main_grad()
    sp.zero_main_grad()
    for _ in range(2):  # accumulate twice
        xx = x.clone().requires_grad_(True)
        sp(xx).backward(dy)
    dw, db = canon_grads(sp)
    assert maxrel(dw, 2 * dwr) < 2e-6 and maxrel(db, 2 * dbr) < 1e-6
    assert maxrel(sp(x).detach(), yr) < 1e-6


def test_all_one_all_zero_and_one_tile_supports():
    m = _m()
    WM = m["WM"]
    base = base_layer(H=3003, C=128)
    L = base.layout
    n16 = L.groups * L.blocks_per_group * L.block // 16
    x, dy = inputs(32, L.input_dim, L.groups)
    nv = WM.layout_counts(L, "cuda")["nv"]
    for tb in (16, 8):
        P = WM.POLICIES["U16" if tb == 16 else "U8"]
        for kind in ("all", "none", "one"):
            nkt = 64 // tb
            if kind == "all":
                tiles = (nv > 0)[:, None].expand(n16, nkt).clone()
            elif kind == "none":
                tiles = torch.zeros((n16, nkt), dtype=torch.bool, device="cuda")
                tiles[::5, 1] = True  # most groups bias-only
                tiles &= (nv > 0)[:, None]
            else:
                tiles = torch.zeros((n16, nkt), dtype=torch.bool, device="cuda")
                tiles[torch.arange(n16, device="cuda"), torch.arange(n16, device="cuda") % nkt] = True
                tiles &= (nv > 0)[:, None]
            ms = WM.MaskSet(P, tiles, torch.zeros(n16, dtype=torch.bool, device="cuda"), torch.zeros_like(tiles), None,
                            nv, {})
            em = ms.element_mask(L)
            dm = masked_dense(base, em)
            yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
            sp = m["W"].KgxWideLayer(dm, ms)
            y, dx = run(sp, x, dy)
            dw, db = canon_grads(sp)
            assert maxrel(y, yr) < 1e-6 and maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5
            assert maxrel(db, dbr) < 1e-6, (tb, kind)


def test_cancellation_near_zero_and_nonfinite_backbone():
    """Full-z decision with backbone + tile terms cancelling (z = 0 exactly -> inactive), and an inf input
    propagating like the dense masked kernels."""
    m = _m()
    base = base_layer(D=256, H=640, G=4, C=64, R=2, bias_std=0.0)
    L = base.layout
    ms = m["WM"].build("H2", 0.3, base.weight.detach(), L)
    em = ms.element_mask(L)
    with torch.no_grad():  # weights +-1 on the enabled support: integer z, many exact zeros / cancellations
        sgn = torch.where(torch.rand(base.weight.shape, device="cuda") < 0.5, -1.0, 1.0).bfloat16()
        base.weight.copy_(sgn * em)
    dm = masked_dense(base, em)
    x = torch.randint(-2, 3, (32, 256), device="cuda").float()
    dy = torch.randn(32, 4, device="cuda")
    yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
    sp = m["W"].KgxWideLayer(dm, ms)
    y, dx = run(sp, x, dy)
    dw, db = canon_grads(sp)
    assert torch.equal(y.double(), yr) or maxrel(y, yr) < 1e-7  # integer z: exact decisions at z = 0
    assert maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6  # split dZ (tiles)
    x2 = x.clone()
    x2[3, 7] = float("inf")
    yd = dm(x2).detach()
    ys = sp(x2).detach()
    assert torch.equal(torch.isnan(yd), torch.isnan(ys)) and torch.equal(torch.isinf(yd), torch.isinf(ys))


def test_lowered_one_tile_control_is_the_same_function():
    m = _m()
    base = base_layer(H=3003, C=128)
    L = base.layout
    x, dy = inputs(40, L.input_dim, L.groups)
    for pol, rho in (("P16", 0.25), ("P8", 0.125)):
        ms = m["WM"].build(pol, rho, base.weight.detach(), L)
        assert bool((ms.tiles.sum(1)[ms.nv > 0] == 1).all())
        dm = masked_dense(base, ms.element_mask(L))
        lo = m["OT"].lower_one_tile(dm, ms)
        lo.kcsr_validate()
        sp = m["W"].KgxWideLayer(dm, ms)
        assert m["C"].digest(lo)["sha"] == m["C"].digest(sp)["sha"] == m["C"].digest(dm)["sha"]
        assert m["C"].mask_digest(lo)["sha"] == m["C"].mask_digest(sp)["sha"] == m["C"].mask_digest(dm)["sha"]
        yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
        y, dx = run(lo, x, dy)
        dw, db = canon_grads(lo)
        assert maxrel(y, yr) < 1e-6 and maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6


# ----------------------------------------------------------------------------------------------- identity
def test_canonical_identity_across_representations():
    m = _m()
    C = m["C"]
    base = base_layer(H=3003, C=128)
    L = base.layout
    for pol, rho in (("U16", 0.3), ("P8", 0.2), ("H2A", 0.2), ("P16", 0.25)):
        ms = m["WM"].build(pol, rho, base.weight.detach(), L)
        em = ms.element_mask(L)
        dm = masked_dense(base, em)  # W * mask: negative weights give -0 at inactive slots
        assert bool((dm.weight.detach().view(torch.int16) == -32768).any())
        reps = [dm, m["W"].KgxWideLayer(dm, ms)]
        if ms.tb == 16 and ms.bb is None:
            reps.append(m["SP"].KgxTileLayer(dm, ms.tiles))
            reps.append(m["SP"].KgxCsrLayer(dm, em))
        ids = [C.digest(r)["sha"] for r in reps]
        assert len(set(ids)) == 1, (pol, ids)
        assert len({C.mask_digest(r)["sha"] for r in reps}) == 1
        sto = [C.storage_digest(r) for r in reps]
        assert sto[0] not in sto[1:]  # dense storage differs (tile16 on both kernels: the same compact layout)
        if len(reps) == 4:
            assert sto[1] == sto[2] and sto[3] not in sto[:3]
        # a one-ulp change of one valid enabled weight changes the canonical hash; padding does not enter it
        sp = reps[1]
        with torch.no_grad():
            sp.weight.view(-1)[0:1].view(torch.int16).add_(1)
        assert C.digest(sp)["sha"] != ids[0]
    # padding values do not enter the canonical hash
    lay = base_layer(H=3003, C=128)
    h0 = C.digest(lay)["sha"]
    valid = lay.layout.valid_mask("cuda")
    with torch.no_grad():
        lay.weight.view(-1, 64)[~valid.reshape(-1)] = 1.0
    assert C.digest(lay)["sha"] == h0


def test_canonical_identity_of_the_lowered_c512_route_and_optimizer_moments():
    m = _m()
    C = m["C"]
    lay = m["KX"].KgxLayer(256, 4100, 4, 512, ConnectivityPlan(64, 2, seed=2), variant="idf", tile=128, device="cuda",
                           seed=5, keep_indices=True)
    h = C.digest(lay)["sha"]
    lay.select_variant("lowered")  # the kgen reference on physical C256 views of the same tensors
    assert C.digest(lay)["sha"] == h
    # moments: dense masked vs compact after the same optimizer steps (stochastic rounding off)
    base = base_layer(H=3003, C=128)
    L = base.layout
    ms = m["WM"].build("H2", 0.15, base.weight.detach(), L)
    em = ms.element_mask(L)
    dm = masked_dense(base, em)
    sp = m["W"].KgxWideLayer(dm, ms)
    x, dy = inputs(32, L.input_dim, L.groups)
    o1 = m["L1"].L1AdamW([dm.weight, dm.bias], lr=1e-3, stochastic_rounding=False)
    m["L1"].apply_mask_(o1, dm.weight, em)
    o2 = m["L1"].L1AdamW([sp.weight, sp.bias, sp.bb], lr=1e-3, stochastic_rounding=False)
    if sp.tvmask is not None:
        o2.set_mask(sp.weight, sp.tvmask)
    assert C.digest(dm, "m", o1)["sha"] == C.digest(sp, "m", o2)["sha"]  # (zero state)
    for lay_, opt in ((dm, o1), (sp, o2)):
        lay_.enable_main_grad()
        for _ in range(3):
            xx = x.clone().requires_grad_(True)
            lay_(xx).backward(dy)
            opt.step()
    # values / moments agree to rounding (FP32 atomics differ in order); structural zeros stay zero
    g, n = C.neurons(dm.counts, 0, dm.counts.numel())
    for kind in ("param", "m", "v"):
        a = dm.canon_rows(C.sources(dm, kind, o1), g, n)[0].double()
        b = sp.canon_rows(C.sources(sp, kind, o2), g, n)[0].double()
        assert maxrel(b, a) < 2e-2, kind
    if sp.tvmask is not None:
        assert bool((sp.weight.detach()[sp.tvmask == 0] == 0).all())
    em_rows = C.dense_mask_rows(dm, g, n)
    assert bool((dm.canon_rows(C.sources(dm, "param"), g, n)[0][~em_rows] == 0).all())


# ----------------------------------------------------------------------------------------------- state
def test_snapshot_and_checkpoint_round_trip_of_a_hybrid_layer(tmp_path):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    from train_state import TrainState

    m = _m()
    base = base_layer(H=3003, C=128)
    L = base.layout
    ms = m["WM"].build("H2A", 0.2, base.weight.detach(), L)
    sp = m["W"].KgxWideLayer(masked_dense(base, ms.element_mask(L)), ms)
    sp.enable_main_grad()
    opt = m["L1"].L1AdamW([sp.weight, sp.bias, sp.bb], lr=1e-3)
    opt.set_mask(sp.weight, sp.tvmask)
    x, dy = inputs(32, L.input_dim, L.groups)
    sp(x).backward(dy)
    opt.step()
    st = TrainState(list(sp.parameters()), opt).capture()
    fp0, meta0 = st.fingerprint(), sp.immutable_fingerprints()
    id0 = m["C"].identity(sp, opt)
    for _ in range(2):
        sp(x).backward(dy)
        opt.step()
    assert st.fingerprint() != fp0
    st.restore()
    assert st.fingerprint() == fp0 and sp.immutable_fingerprints() == meta0
    assert m["C"].identity(sp, opt) == id0
    # checkpoint (dense W_eff + element mask) -> rebuilt hybrid layer: same canonical parameters
    p = tmp_path / "h.pt"
    m["SP"].save_checkpoint(p, sp, sp.element_mask(), dict(tag="t"))
    rec, w, b, mask = m["SP"].load_checkpoint(p)
    rebuilt = m["KX"].KgxLayer(L.input_dim, L.hidden_dim, L.groups, L.block, base.connectivity, variant="idf",
                               device="cuda", weights=(w, b), keep_indices=True)
    rebuilt.mask = mask
    assert m["C"].digest(rebuilt)["sha"] == id0["params"]["sha"]
    assert m["C"].mask_digest(rebuilt)["sha"] == id0["mask"]["sha"]


# ----------------------------------------------------------------------------------------------- audit fixes
def test_csr_reports_its_executed_fp32_dz_and_rejects_bf16_requests():
    m = _m()
    base = base_layer(H=3003, C=128)
    em = m["SP"].random_mask(m["SP"].valid_slots(base), 0.3, 1)
    dm = masked_dense(base, em)
    sp = m["SP"].KgxCsrLayer(dm, em)
    x, dy = inputs(16, 256, 8)
    run(sp, x, dy)
    assert sp.last_info["grad_precision"] == "fp32_scalar" and sp.last_info["grad_precision_requested"] == "split"
    assert "dz in FP32" in sp.numerics() and "hi + lo" not in sp.numerics()
    assert sp.last_info["execution"]["grad_precision"] == "fp32_scalar"
    sp.plan = PhysicalPlan(grad_precision="bf16")
    with pytest.raises(ValueError, match="FP32"):
        sp(x)
    # the backbone layer labels its mixed scheme; plain tile layers keep the requested split scheme
    ms = m["WM"].build("H2", 0.15, base.weight.detach(), base.layout)
    hy = m["W"].KgxWideLayer(masked_dense(base, ms.element_mask(base.layout)), ms)
    run(hy, x, dy)
    assert hy.last_info["grad_precision"].startswith("mixed") and "FP32" in hy.numerics()


def test_kcsr_params_sha_is_the_canonical_hash_for_every_format():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import kcsr_graph

    m = _m()
    base = base_layer(H=3003, C=128)
    L = base.layout
    ms = m["WM"].build("U16", 0.3, base.weight.detach(), L)
    em = ms.element_mask(L)
    dm = masked_dense(base, em)
    reps = [dm, m["SP"].KgxTileLayer(dm, ms.tiles), m["SP"].KgxCsrLayer(dm, em), m["W"].KgxWideLayer(dm, ms)]
    shas = {kcsr_graph.params_sha(r) for r in reps}
    assert len(shas) == 1 and all(kcsr_graph.params_sha_schema(r) == "canonical" for r in reps)
    w, b = kcsr_graph.logical_params(reps[1])  # compact formats: dense logical values (not the compact tensor)
    assert w.shape == dm.weight.shape


def test_executed_work_counts():
    m = _m()
    base = base_layer(H=3008, G=8, C=128)  # n_g = 376: full row groups only
    L = base.layout
    ew = m["W"].executed_work(base, 16)
    assert ew["mma_products_per_token"]["fwd"] == 64 * 3008  # every valid slot once
    ms = m["WM"].build("P16", 0.25, base.weight.detach(), L)
    sp = m["W"].KgxWideLayer(masked_dense(base, ms.element_mask(L)), ms)
    ew = sp.executed_work(16)
    assert ew["mma_products_per_token"]["fwd"] == 16 * 3008 and ew["scalar_fma"]["fwd"] == 0
    ms = m["WM"].build("H2", 0.1, base.weight.detach(), L)
    hy = m["W"].KgxWideLayer(masked_dense(base, ms.element_mask(L)), ms)
    ew = hy.executed_work(16)
    assert ew["scalar_fma"]["fwd"] == 2 * 3008 * 16


@pytest.mark.parametrize("C", [64, 512])
def test_lowered_one_tile_control_at_other_c(C):
    m = _m()
    base = base_layer(H=4100, G=8, C=C, R=1)
    L = base.layout
    x, dy = inputs(24, L.input_dim, L.groups)
    ms = m["WM"].build("P16", 0.25, base.weight.detach(), L)
    dm = masked_dense(base, ms.element_mask(L))
    lo = m["OT"].lower_one_tile(dm, ms)
    lo.kcsr_validate()
    assert m["C"].digest(lo)["sha"] == m["C"].digest(dm)["sha"]
    yr, dxr, dwr, dbr = ref_grads(dm, x, dy)
    y, dx = run(lo, x, dy)
    dw, db = canon_grads(lo)
    assert maxrel(y, yr) < 1e-6 and maxrel(dx, dxr) < 2e-5 and maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6
