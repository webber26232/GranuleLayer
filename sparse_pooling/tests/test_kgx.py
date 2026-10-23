"""EXPERIMENTAL K = 64 large-C / index-free / sparse kernels (experimental/kgx_*) and the L1 treatments
(experimental/l1.py) of the K64 study (results/K64_LARGE_C_INDEXFREE_L1.md).

Correctness gate before any full-scale timing: generator / graph invariants (R prefix rounds, identical sets
across C, index-free records == sets, by-neuron init independent of C), every dense execution variant
(kgen / lowered reference, direct, idf_rmw, idf; physical tiles) against a float64 reference with ragged H,
partial logical tails, token tails and BF16 / split operands; the lowering view and its round trip;
cancellation / full-K decisions; non-finite propagation; unsupported requests; main_grad accumulation and
checkpoints; the frozen-mask formats (tile, CSR) against the SAME dense masked model incl. all-zero /
all-one masks, empty tiles, bias-only rows and partial groups; L1AdamW (dense equivalence with FusedAdamW,
coupled / proximal rules against FP64, exact-zero preservation, masks); a short composed training run with
dynamic regrowth (proximal) separated from frozen-mask updates.
"""
import math

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan, make_feature_sets

pytestmark = requires_cuda_backend


def _m():
    from sparse_pooling.experimental import kgx_layer, kgx_lowering, kgx_sparse, l1, reference
    from sparse_pooling.experimental import kgen_layout

    return kgx_layer, kgx_lowering, kgx_sparse, l1, reference, kgen_layout


def layer(D, H, G, C, S, R, variant="direct", tile=256, seed=1, init_seed=3, bias_std=0.3, prec="split"):
    KX, _, _, _, _, KL = _m()
    lay = KX.KgxLayer(D, H, G, C, ConnectivityPlan(S, R, seed=seed), variant=variant, tile=tile, device="cuda",
                      seed=init_seed, plan=PhysicalPlan(grad_precision=prec), keep_indices=True)
    if bias_std:
        g = torch.Generator(device="cuda").manual_seed(init_seed + 1)
        with torch.no_grad():
            lay.bias.copy_((bias_std * torch.randn(lay.bias.shape, device="cuda", generator=g)).bfloat16())
        KL.zero_padding_(lay.layout, lay.weight.data, lay.bias.data)
    return lay


def run(lay, x, dy):
    lay.enable_main_grad()
    lay.zero_main_grad()
    x = x.detach().clone().requires_grad_(True)
    y = lay(x)
    y.backward(dy)
    return y.detach(), x.grad.detach(), lay.weight.main_grad.clone(), lay.bias.main_grad.clone()


def reference(lay, x, dy, w=None):
    _, _, _, _, REF, _ = _m()
    w = lay.weight.detach() if w is None else w
    return REF.grads(x.double(), lay.logical_indices().long(), w.double(), lay.bias.detach().double(), lay.counts,
                     dy.double())


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


def inputs(B, D, G, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, D, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(B, G, device="cuda", generator=g)
    return x, dy


def check(out, ref, prec="split"):
    tol = 2e-5 if prec == "split" else 2e-2
    y, dx, dw, db = out
    yr, dxr, dwr, dbr = ref
    assert maxrel(y, yr) < 1e-6
    assert maxrel(dx, dxr) < tol and maxrel(dw, dwr) < tol
    assert maxrel(db, dbr) < 1e-6


# --------------------------------------------------------------------------------- graph invariants
def test_feature_sets_prefix_rounds_and_identical_across_c():
    D = 1024
    f64 = make_feature_sets(D, ConnectivityPlan(64, 64, seed=0))
    f256 = make_feature_sets(D, ConnectivityPlan(64, 256, seed=0))
    assert torch.equal(f64, f256[: f64.shape[0]])  # R = 64 rounds are the first rounds of R = 256
    a = layer(256, 5000, 8, 64, 64, 3)
    b = layer(256, 5000, 8, 512, 64, 3)
    assert torch.equal(a.sg_feat, b.sg_feat)  # identical sets for every C (graph seed fixed)


def test_index_free_records_are_the_sets_and_s128_is_not():
    KX, _, _, _, _, _ = _m()
    lay = layer(256, 20011, 16, 256, 64, 3)
    assert lay.graph_stats["index_free_check"]["records"] > 0
    rec = KX.records_from_feat(lay.layout, lay.sg_feat, lay.kg_buckets)
    nvb = lay.layout.valid_blocks("cuda")
    valid = torch.arange(lay.layout.blocks_per_group, device="cuda")[None] < nvb[:, None]
    assert torch.equal(rec[valid], lay.indices[valid])
    s128 = layer(512, 9001, 8, 128, 128, 2)
    assert not s128.index_free_ok
    with pytest.raises(ValueError, match="S = K"):
        s128.select_variant("idf")


def test_init_by_neuron_identical_valid_draws_across_c():
    _, _, _, _, _, KL = _m()
    H, G = 30011, 16
    ws = []
    for C in (64, 128, 512, 2048):
        L = KL.KLayout(256, H, G, 64, C)
        w, b = KL.init_parameters_by_neuron(L, "cuda", seed=17, chunk_neurons=4096)
        valid = L.valid_mask("cuda")
        assert bool((w[~valid] == 0).all()) and bool((b == 0).all())
        ws.append(w.view(G, -1, 64)[:, : int(L.counts_tensor("cpu").max())])
        cnt = L.counts_tensor("cuda").long()
        ws[-1] = torch.cat([ws[-1][g, : int(cnt[g])] for g in range(G)])
    for w in ws[1:]:
        assert torch.equal(w, ws[0])
    assert abs(float(ws[0].float().std()) - 1 / 8) < 2e-3


# --------------------------------------------------------------------------------- dense variants
DENSE = [  # (D, H, G, C, S, R, B, tile)
    (256, 5000, 16, 64, 64, 4, 64, 64),
    (256, 20011, 16, 512, 64, 3, 72, 128),
    (256, 20011, 16, 2048, 64, 2, 40, 256),
    (256, 30011, 16, 1024, 64, 2, 40, 64),
    (256, 10011, 16, 256, 64, 2, 48, 128),
    (512, 9001, 8, 128, 128, 2, 48, 64),
    (512, 30000, 8, 1024, 128, 3, 64, 256),
    (512, 7001, 8, 512, 256, 1, 16, 128),
]


@pytest.mark.parametrize("D,H,G,C,S,R,B,tile", DENSE)
@pytest.mark.parametrize("prec", ["split", "bf16"])
def test_dense_variants_against_float64_reference(D, H, G, C, S, R, B, tile, prec):
    lay = layer(D, H, G, C, S, R, tile=tile, prec=prec)
    x, dy = inputs(B, D, G)
    ref = reference(lay, x, dy)
    for v in lay.bench_variants():
        lay.select_variant(v)
        out = run(lay, x, dy)
        check(out, ref, prec)
        inv = ~lay.layout.valid_mask("cuda")
        assert bool((out[2][inv] == 0).all()) and bool((out[3][inv] == 0).all()), v


def test_token_tails():
    lay = layer(256, 7001, 8, 256, 64, 2, tile=128)
    for B in (8, 24, 40, 136, 520):
        x, dy = inputs(B, 256, 8, seed=B)
        ref = reference(lay, x, dy)
        for v in ("direct", "idf_rmw", "idf", "kgen"):
            lay.select_variant(v)
            check(run(lay, x, dy), ref)


def test_cancellation_and_full_k_decision():
    lay = layer(256, 4096, 8, 64, 64, 2, bias_std=0.0)
    with torch.no_grad():  # partial sums of opposite sign to the full sum: +8 x on k 0..7, -1.25 x on 8..15
        w = torch.zeros(lay.weight.shape, device="cuda")
        w[..., :8] = 1.0
        w[..., 8:16] = -1.25
        lay.weight.copy_(w.bfloat16() * lay.layout.valid_mask("cuda")[..., None])
    x = torch.ones(32, 256, device="cuda")
    dy = torch.randn(32, 8, device="cuda")
    for v in ("direct", "idf", "kgen"):
        lay.select_variant(v)
        y, dx, dw, db = run(lay, x, dy)
        assert torch.equal(y, torch.zeros_like(y)) and torch.equal(dx, torch.zeros_like(dx))
        assert torch.equal(dw, torch.zeros_like(dw)) and torch.equal(db, torch.zeros_like(db))
    with torch.no_grad():  # exact cancellation to z = 0 (inactive), near-zero rows
        a = torch.randn(lay.weight.shape[:-1], device="cuda").abs()[..., None].bfloat16()
        sign = torch.ones(64, device="cuda").bfloat16()
        sign[32:] = -1
        lay.weight.copy_(a * sign * lay.layout.valid_mask("cuda")[..., None])
    x = torch.ones(64, 256, device="cuda")
    x[32:] = (torch.randn(32, 256, device="cuda") * 1e-3).bfloat16().float()
    dy = torch.randn(64, 8, device="cuda")
    ref = reference(lay, x, dy)
    for v in ("direct", "idf"):
        lay.select_variant(v)
        out = run(lay, x, dy)
        assert torch.equal(out[0][:32], torch.zeros_like(out[0][:32]))
        check(out, ref)


def test_nonfinite_propagates_like_the_kgen_reference():
    lay = layer(256, 5003, 8, 128, 64, 2)
    x, dy = inputs(64, 256, 8)
    dy[3, 2] = float("inf")
    dy[7, 5] = float("nan")
    lay.select_variant("kgen")
    want = run(lay, x, dy)
    for v in ("direct", "idf_rmw", "idf"):
        lay.select_variant(v)
        got = run(lay, x, dy)
        for a, r in zip(got[1:], want[1:]):
            assert torch.equal(torch.isnan(a), torch.isnan(r)) and torch.equal(torch.isinf(a), torch.isinf(r)), v


def test_lowering_view_round_trip_and_duplication():
    _, LW, _, _, _, _ = _m()
    lay = layer(256, 20011, 16, 1024, 64, 2)
    NBp, idx_p, bk_p, f = LW.physical_graph(lay.layout, lay.logical_indices(), lay.kg_buckets)
    assert f == 4 and NBp == 4 * lay.layout.blocks_per_group
    w_p = lay.weight.detach().view(16, NBp, 256, 64)  # a view: same storage
    assert w_p.data_ptr() == lay.weight.data_ptr()
    w_l, b_l = LW.logical_from_physical(lay.layout, w_p, lay.bias.detach().view(16, NBp, 256))
    assert torch.equal(w_l, lay.weight.detach()) and torch.equal(b_l, lay.bias.detach())
    # every physical copy carries its logical record; physical buckets cover exactly the valid physical blocks
    ip = idx_p.view(16, -1, 4, 64)
    assert bool((ip == ip[:, :, :1]).all())
    cnt = lay.layout.counts_tensor("cuda").long()
    nvp = (cnt + 255) // 256
    per_g = torch.zeros(16, dtype=torch.int64, device="cuda").index_add_(0, bk_p[:, 1].long(), bk_p[:, 3].long())
    assert torch.equal(per_g, nvp)


def test_unsupported_requests_raise():
    KX, _, _, _, _, _ = _m()
    with pytest.raises(ValueError, match="C % 16"):
        KX.KgxLayer(256, 2000, 8, 72, ConnectivityPlan(64, 2), device="cuda")
    with pytest.raises(ValueError):
        KX.KgxLayer(256, 2000, 8, 4096, ConnectivityPlan(64, 2), device="cuda")  # C > 2048
    lay = layer(256, 3000, 8, 64, 64, 2)
    with pytest.raises(ValueError, match="lowered"):
        lay.select_variant("lowered")  # C <= 256: use kgen
    x, dy = inputs(64, 256, 8)
    lay.plan = PhysicalPlan(reduction="deterministic")
    with pytest.raises(ValueError, match="deterministic"):
        lay(x)
    lay.plan = PhysicalPlan()
    with pytest.raises(ValueError, match="feature-major"):
        lay(x[:12])
    from sparse_pooling.experimental.kgen_layer import KgenLayer

    with pytest.raises(ValueError, match="C <= 256"):
        KgenLayer(256, 3000, 8, 64, 512, ConnectivityPlan(64, 2), device="cuda")
    nodx = KX.KgxLayer(256, 3000, 8, 64, ConnectivityPlan(64, 2), variant="idf", device="cuda", keep_indices=False)
    assert nodx.indices is None and nodx.index_bytes()["record_indices_needed"] == 0
    nodx.select_variant("direct")
    with pytest.raises(ValueError, match="record indices"):
        nodx(x)


def test_main_grad_accumulates_and_checkpoint_round_trip():
    lay = layer(256, 5000, 8, 128, 64, 2, variant="idf")
    x, dy = inputs(32, 256, 8)
    lay.enable_main_grad()
    lay.zero_main_grad()
    lay(x).backward(dy)
    g1 = lay.weight.main_grad.clone()
    lay(x).backward(dy)
    assert maxrel(lay.weight.main_grad, 2 * g1) < 1e-6
    sd = lay.state_dict()
    other = layer(256, 5000, 8, 128, 64, 2, variant="idf", init_seed=9)
    other.load_state_dict(sd)
    assert torch.equal(other.weight, lay.weight) and torch.equal(other.bias, lay.bias)
    wrong = layer(256, 5000, 8, 128, 64, 3)
    with pytest.raises(ValueError, match="graph spec"):
        wrong.load_state_dict(sd)


# --------------------------------------------------------------------------------- sparse formats
SPARSE = [(256, 5000, 16, 64, 4, 64, 0.3), (256, 20011, 16, 512, 3, 72, 0.1), (256, 20011, 16, 2048, 2, 40, 0.05),
          (256, 10011, 16, 256, 2, 48, 0.6), (256, 7001, 8, 128, 2, 24, 0.0), (256, 7001, 8, 128, 2, 24, 1.0)]


def _sparse_grads_dense(sp, out, shape):
    """Scatter compact gradients of a tile / CSR layer back to the dense [G, NB, C, 64] layout."""
    _, _, SP, _, _, _ = _m()
    if isinstance(sp, SP.KgxCsrLayer):
        mg = torch.zeros(shape, device="cuda").view(-1, 64)
        rp = sp.rowptr.long()
        rows = torch.repeat_interleave(torch.arange(rp.numel() - 1, device="cuda"), rp[1:] - rp[:-1])
        mg[rows, sp.kix.long()] = out[2]
        return mg.view(shape)
    n16 = math.prod(shape) // 1024
    gd = torch.zeros((n16, 4, 16, 16), device="cuda")
    gd[sp.tile_mask_bool] = out[2]
    return gd.permute(0, 2, 1, 3).reshape(shape)


@pytest.mark.parametrize("D,H,G,C,R,B,dens", SPARSE)
def test_sparse_formats_against_the_same_dense_masked_model(D, H, G, C, R, B, dens):
    KX, _, SP, _, _, _ = _m()
    base = layer(D, H, G, C, 64, R, variant="idf", tile=128)
    with torch.no_grad():  # bias of both signs: bias-only (all-zero) rows contribute relu(b) and need db
        base.bias.mul_(torch.where(torch.rand(base.bias.shape, device="cuda") < 0.5, -1.0, 1.0).bfloat16())
    valid = SP.valid_slots(base)
    x, dy = inputs(B, D, G)
    for kind in ("random", "tile"):
        if kind == "random":
            em, tm = SP.random_mask(valid, dens, 5), None
        else:
            tm = SP.tile_mask(base.weight.detach(), valid, dens)
            em = SP.tiles_to_elements(tm, valid)
        dm = KX.KgxLayer(D, H, G, C, ConnectivityPlan(64, R, seed=1), variant="idf", tile=128, device="cuda",
                         weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=True)
        ref = reference(dm, x, dy)
        ref = (ref[0], ref[1], ref[2] * em.double(), ref[3])
        od = run(dm, x, dy)
        check((od[0], od[1], od[2] * em, od[3]), ref)
        fmts = [SP.KgxCsrLayer(dm, em)] + ([SP.KgxTileLayer(dm, tm)] if tm is not None else [])
        for sp in fmts:
            w_, b_ = sp.dense_params()
            assert torch.equal(w_, dm.weight.detach()) and torch.equal(b_, dm.bias.detach())
            assert torch.equal(sp.element_mask().bool(), em.bool())
            o = run(sp, x, dy)
            gd = _sparse_grads_dense(sp, o, dm.weight.shape)
            check((o[0], o[1], gd, o[3]), ref)
            # vs the dense masked kernels directly (same values, mask, state)
            assert maxrel(o[0], od[0]) < 1e-6 and maxrel(o[3], od[3]) < 1e-5


def test_bias_only_rows_and_partial_groups_exact():
    KX, _, SP, _, _, _ = _m()
    base = layer(256, 3003, 8, 64, 64, 2, variant="idf")  # n_g = 375: partial last 16-group of each output
    valid = SP.valid_slots(base)
    tm = torch.zeros((base.weight.numel() // 1024, 4), dtype=torch.bool, device="cuda")
    tm[::3, 1] = True  # a third of the groups keep one tile; the rest are bias-only
    em = SP.tiles_to_elements(tm & valid.reshape(-1, 16).any(1)[:, None], valid)
    dm = KX.KgxLayer(256, 3003, 8, 64, ConnectivityPlan(64, 2, seed=1), variant="idf", device="cuda",
                     weights=(base.weight.detach() * em, base.bias.detach().clone()), keep_indices=True)
    x, dy = inputs(48, 256, 8)
    ref = reference(dm, x, dy)
    for sp in (SP.KgxTileLayer(dm, tm & valid.reshape(-1, 16).any(1)[:, None]), SP.KgxCsrLayer(dm, em)):
        rt = sp._runtime()
        assert rt.sg.bias_only is not None and rt.sg.stats["bias_only_neurons"] > 0
        o = run(sp, x, dy)
        assert maxrel(o[0], ref[0]) < 1e-6 and maxrel(o[1], ref[1]) < 2e-5 and maxrel(o[3], ref[3]) < 1e-6


def test_checkpoint_masks_round_trip(tmp_path):
    _, _, SP, _, _, _ = _m()
    lay = layer(256, 5000, 8, 128, 64, 2, variant="idf")
    valid = SP.valid_slots(lay)
    m = SP.magnitude_mask(lay.weight.detach(), valid, 0.25)
    W0 = int(valid.sum()) * 64
    assert int(m.sum()) == round(0.25 * W0)  # exact kept count (ties broken by index)
    kept = lay.weight.detach()[m.bool()].float().abs()
    dropped = lay.weight.detach()[(~m.bool()) & valid[..., None]].float().abs()
    assert float(kept.min()) >= float(dropped.max())
    rec = SP.save_checkpoint(tmp_path / "c.pt", lay, m, dict(test=True))
    r2, w, b, m2 = SP.load_checkpoint(tmp_path / "c.pt")
    assert torch.equal(w, lay.weight.detach()) and torch.equal(m2, m) and r2["mask_sha256"] == rec["mask_sha256"]


# --------------------------------------------------------------------------------- L1AdamW
def test_l1adamw_none_is_fused_adamw_bitwise():
    _, _, _, L1, _, _ = _m()
    torch.manual_seed(0)
    w0 = (torch.randn(100_003, device="cuda") * 0.125).bfloat16()
    g0 = torch.randn(100_003, device="cuda") * 1e-3
    pa, pb = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    pa.main_grad, pb.main_grad = g0.clone(), g0.clone()
    oa, ob = FusedAdamW([pa], lr=1e-4), L1.L1AdamW([pb], lr=1e-4)
    for i in range(3):
        oa.step()
        ob.step()
        pa.main_grad.copy_(g0 * (i + 2))
        pb.main_grad.copy_(g0 * (i + 2))
    assert torch.equal(pa.data, pb.data) and torch.equal(oa.state[pa]["v"], ob.state[pb]["v"])
    assert bool((pa.main_grad == g0 * 4).all())  # (main_grad zeroed by step, then re-filled)


@pytest.mark.parametrize("mode,mu", [("prox", 3e-4), ("coupled", 1e-3), ("prox", 0.0), ("none", 0.0)])
def test_l1adamw_rules_against_fp64(mode, mu):
    _, _, _, L1, _, _ = _m()
    n = 200_003
    torch.manual_seed(1)
    w0 = torch.randn(n, device="cuda") * 0.125
    w0[:500] = 0
    p = torch.nn.Parameter(w0.clone())
    p.main_grad = torch.randn(n, device="cuda") * 1e-3
    o = L1.L1AdamW([p], lr=1e-3, moments="fp32", l1=mode, mu=mu, stochastic_rounding=False)
    st = o._state(p)
    st["m"].copy_(torch.randn(n, device="cuda") * 1e-4)
    st["v"].copy_(torch.rand(n, device="cuda") * 1e-6)
    st["step"] = 4
    m0, v0, pp, gg = st["m"].clone(), st["v"].clone(), p.data.clone(), p.main_grad.clone()
    o.step()
    u, m2, v2, thr = L1.reference_update(pp, gg, m0, v0, 5, 1e-3, 0.9, 0.999, 1e-8, 0.0, mode, mu)
    assert float((p.data.double() - u).abs().max()) < 1e-7
    if thr is not None:  # zero pattern equal except where |candidate| is within FP32 rounding of the threshold
        u_pre = L1.reference_update(pp, gg, m0, v0, 5, 1e-3, 0.9, 0.999, 1e-8, 0.0, "none", 0.0)[0]
        border = (u_pre.abs() - thr).abs() <= 1e-6 * thr + 1e-12
        assert int((((p.data == 0) != (u == 0)) & ~border).sum()) == 0
        assert int((u == 0).sum()) > 0 or mu == 0
    assert float((st["m"].double() - m2).abs().max()) < 1e-9
    assert bool((p.main_grad == 0).all())


def test_prox_exact_zeros_survive_bf16_rounding_and_masks_freeze_state():
    _, _, _, L1, _, _ = _m()
    n = 300_007
    torch.manual_seed(2)
    p = torch.nn.Parameter((torch.randn(n, device="cuda") * 0.125).bfloat16())
    p.main_grad = torch.randn(n, device="cuda") * 1e-3
    o = L1.L1AdamW([p], lr=1e-4, l1="prox", mu=1e-2)
    o.step()
    z = int((p.data == 0).sum())
    assert z > 0
    # frozen mask: inactive values, moments and updates stay exactly zero under further gradients
    mask = (torch.rand(n, device="cuda") < 0.3).to(torch.uint8)
    o.param_groups[0]["l1"] = "none"
    L1.apply_mask_(o, p, mask)
    for _ in range(3):
        p.main_grad.normal_(0, 1e-3)
        o.step()
    inact = mask == 0
    assert bool((p.data[inact] == 0).all()) and bool((o.state[p]["m"][inact] == 0).all())
    assert bool((o.state[p]["v"][inact] == 0).all())


def test_calibrate_mu_and_fallback():
    _, _, SP, L1, _, _ = _m()
    lay = layer(256, 5000, 8, 64, 64, 2)
    valid = SP.valid_slots(lay)
    pos = L1.sample_positions(valid, 10_000, seed=4)
    assert torch.equal(pos, L1.sample_positions(valid, 10_000, seed=4))
    flat_valid = valid[..., None].expand(lay.weight.shape).reshape(-1)
    assert bool(flat_valid[pos].all())
    g = torch.zeros(lay.weight.shape, device="cuda")
    rec = L1.calibrate_mu(g, pos)
    assert rec["fallback"] and rec["g_ref"] == 0.0
    g.view(-1)[pos[:10]] = 1e-3
    rec = L1.calibrate_mu(g, pos)
    assert rec["fallback"] and abs(rec["g_ref"] - 1e-3) < 1e-9
    g.normal_(0, 1e-4)
    rec = L1.calibrate_mu(g, pos)
    assert not rec["fallback"] and 0.5e-4 < rec["g_ref"] < 1e-4


def test_composed_training_dynamic_regrowth_vs_frozen_mask():
    """Upstream projection -> layer -> head; proximal L1 may let a zero weight regrow (dynamic support); a
    frozen mask keeps it zero.  The two are different algorithms (not claimed equivalent)."""
    _, _, SP, L1, _, _ = _m()
    torch.manual_seed(0)
    lay = layer(256, 5000, 8, 64, 64, 2, variant="idf", bias_std=0.0)
    lay.enable_main_grad()
    proj = torch.nn.Linear(32, 256, device="cuda")
    head = torch.nn.Linear(8, 4, device="cuda")
    o_small = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=1e-3)
    opt = L1.L1AdamW([{"params": [lay.weight], "l1": "prox", "mu": 0.0}, {"params": [lay.bias]}], lr=1e-3)
    x0 = torch.randn(64, 32, device="cuda")
    tgt = torch.randn(64, 4, device="cuda") * 0.5
    losses = []
    for step in range(30):
        opt.param_groups[0]["mu"] = 5e-4 if step >= 10 else 0.0
        loss = ((torch.tanh(head(lay(proj(x0).bfloat16()).float())) - tgt) ** 2).mean()
        loss.backward()
        o_small.step()
        o_small.zero_grad(set_to_none=True)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    zeros = lay.weight.detach() == 0
    valid = SP.valid_slots(lay)[..., None].expand_as(zeros)
    assert int((zeros & valid).sum()) > 0
    # frozen mask from the support: further updates keep every inactive weight at zero
    mask = SP.learned_mask(lay.weight.detach(), SP.valid_slots(lay))
    opt.param_groups[0]["l1"] = "none"
    L1.apply_mask_(opt, lay.weight, mask)
    for _ in range(5):
        loss = ((torch.tanh(head(lay(proj(x0).bfloat16()).float())) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    assert bool((lay.weight.detach()[mask == 0] == 0).all())


def test_mask_refresh_matches_a_fresh_dense_masked_reference():
    """Train a tile layer, densify, reselect a new tile mask, recompress: the new compact layer computes exactly
    what a dense masked layer with the new mask and the same values computes."""
    KX, _, SP, L1, _, _ = _m()
    base = layer(256, 9001, 8, 128, 64, 2, variant="idf", tile=64)
    valid = SP.valid_slots(base)
    tm = SP.tile_mask(base.weight.detach(), valid, 0.4)
    with torch.no_grad():
        base.weight.mul_(SP.tiles_to_elements(tm, valid).bfloat16())
    sp = SP.KgxTileLayer(base, tm)
    sp.enable_main_grad()
    opt = L1.L1AdamW([sp.weight, sp.bias], lr=1e-3)
    x, dy = inputs(40, 256, 8)
    for _ in range(3):
        sp(x).backward(dy)
        opt.step()
    w, b = sp.dense_params()
    dense = KX.KgxLayer(256, 9001, 8, 128, ConnectivityPlan(64, 2, seed=1), variant="idf", device="cuda",
                        weights=(w.clone(), b.clone()), keep_indices=True)
    tm2 = SP.tile_mask(dense.weight.detach(), valid, 0.2)  # refresh: reselect among the trained values
    em2 = SP.tiles_to_elements(tm2, valid)
    with torch.no_grad():
        dense.weight.mul_(em2.bfloat16())
    sp2 = SP.KgxTileLayer(dense, tm2)
    ref = reference(dense, x, dy)
    o = run(sp2, x, dy)
    assert maxrel(o[0], ref[0]) < 1e-6 and maxrel(o[1], ref[1]) < 2e-5 and maxrel(o[3], ref[3]) < 1e-6
    assert bool((tm2 <= tm).all())  # a refresh from compact values can only drop tiles (no regrowth without dense grads)


def test_complete_snapshot_restore_of_a_compact_layer():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    from train_state import TrainState

    KX, _, SP, L1, _, _ = _m()
    base = layer(256, 5000, 8, 64, 64, 2, variant="idf")
    valid = SP.valid_slots(base)
    em = SP.random_mask(valid, 0.3, 2)
    with torch.no_grad():
        base.weight.mul_(em.bfloat16())
    sp = SP.KgxCsrLayer(base, em)
    sp.enable_main_grad()
    opt = L1.L1AdamW([sp.weight, sp.bias], lr=1e-3)
    x, dy = inputs(32, 256, 8)
    sp(x).backward(dy)
    opt.step()
    st = TrainState(list(sp.parameters()), opt).capture()
    fp0 = st.fingerprint()
    meta0 = sp.immutable_fingerprints()
    for _ in range(2):
        sp(x).backward(dy)
        opt.step()
    assert st.fingerprint() != fp0
    st.restore()
    assert st.fingerprint() == fp0 and sp.immutable_fingerprints() == meta0


def test_proximal_support_is_dynamic():
    """With dense gradients, a weight zeroed by the proximal step can become nonzero again (regrowth): the dynamic
    treatment is not a frozen mask."""
    _, _, SP, L1, _, _ = _m()
    torch.manual_seed(3)
    lay = layer(256, 5000, 8, 64, 64, 2, variant="idf", bias_std=0.0)
    lay.enable_main_grad()
    opt = L1.L1AdamW([{"params": [lay.weight], "l1": "prox", "mu": 2e-3}, {"params": [lay.bias]}], lr=1e-2)
    x, dy = inputs(64, 256, 8)
    for _ in range(4):
        lay(x).backward(dy)
        opt.step()
    zero = (lay.weight.detach() == 0) & SP.valid_slots(lay)[..., None]
    assert int(zero.sum()) > 0
    opt.param_groups[0]["mu"] = 0.0
    for _ in range(3):
        lay(x).backward(-dy)
        opt.step()
    regrown = zero & (lay.weight.detach() != 0)
    assert int(regrown.sum()) > 0


def test_two_four_masks_both_orientations():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import k64_semi24 as S24

    w = torch.randn(256, 64, device="cuda")
    m = S24.two_four_mask(w)
    assert S24.is_24(m, -1) and int(m.sum()) == m.numel() // 2
    b = S24.bidirectional_mask(w)
    assert S24.is_24(b, -1) and S24.is_24(b, 0) and int(b.sum()) == b.numel() // 2
    assert S24.patterns_2of4_rows_cols().shape[0] == 90
