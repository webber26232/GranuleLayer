"""Fixed / randomized local supports inside the K64 envelope (experimental/local_supports.py, kgx_onetile.lower_local;
results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md sections 3-4, 7).

Gates: row identity / original set q / k distinct sorted in-envelope coordinates for every valid row group
(partial groups included, padded groups none), exact budget k H, balanced allocation rules (FP-balanced /
RP-record: per-record spread <= 1 and all subsets present when the record has >= nkt groups; RP-record and BP-row
subsets disjoint per permutation pass), reproducibility and key independence of the seeds, uniformity /
independence statistics, the coverage models against a tiny exhaustive oracle (Monte-Carlo average of the exact
realized coverage over layout seeds vs the refined per-set model), the coverage pipeline vs brute force, and the
indexed K16 / K8 lowering against FP64 and the dense masked model (forward, dX, dW, db, canonical parameter /
mask / main_grad / moment identity, optimizer steps with stochastic rounding off).
"""
import itertools
import math
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend
from sparse_pooling.supergroups import ConnectivityPlan

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))


def _m():
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgen_layout import KLayout

    return LS, KG, KLayout


def graph(D=256, H=3003, G=8, C=64, R=2, seed=1, dev="cpu"):
    LS, KG, KLayout = _m()
    L = KLayout(D, H, G, 64, C)
    counts = L.counts_tensor(dev)
    idx, feat, ubk, _ = KG.generate(D, counts, C, 64, L.blocks_per_group, 1, ConnectivityPlan(64, R, seed=seed), dev)
    return L, idx, feat, KG.unit_to_block_buckets(ubk, counts, C, 1)


RANDOM = ("FP-balanced", "RP-record", "RP-row", "BP-row")


@pytest.mark.parametrize("k", [16, 8])
@pytest.mark.parametrize("C,H", [(64, 3003), (256, 8 * 300 + 5), (128, 8 * 700 - 3)])
def test_structure_budget_and_balance(k, C, H):
    LS, KG, _ = _m()
    L, idx, feat, bk = graph(H=H, C=C)
    nkt = 64 // k
    geo = LS.geometry(L, bk)
    for pol in RANDOM:
        s = LS.build(pol, k, L, bk, graph_seed=1, layout_seed=0)
        gv = s.nv > 0
        assert s.enabled_slots() == k * H and int(s.element_mask(L).sum()) == k * H
        env = s.env[gv]
        assert bool((env[:, 1:] > env[:, :-1]).all()) and int(env.min()) >= 0 and int(env.max()) < 64
        assert bool((s.q[gv] == geo["q"][gv]).all()) and bool((s.q[~gv] == -1).all())
        assert int(s.group_mask()[~gv].sum()) == 0
        st = s.stats
        assert st["rows_sorted_distinct_in_envelope"]
        if pol in ("FP-balanced", "RP-record"):
            assert st["per_record_subset_spread_max"] <= 1
            assert st["all_subsets_in_every_record_with_nkt_groups"]
        if pol in ("RP-record", "BP-row"):  # the subsets of one permutation pass are disjoint and cover 64
            key = geo["rec"] if pol == "RP-record" else None
            uv = geo["uv"]
            if pol == "RP-record":
                grp = geo["rec"]
                pas = grp * 64 + (s.subset[uv] // nkt) * 0  # one pass per record
            else:
                pas = geo["q"][uv] * 100000 + geo["rank_set"] // nkt
            for p in torch.unique(pas)[:200].tolist():
                rows = env[(pas == p)]
                sub = s.subset[uv][pas == p]
                u = torch.unique(sub)
                cols = torch.cat([rows[sub == v][0] for v in u.tolist()])
                assert torch.unique(cols).numel() == cols.numel()  # disjoint subsets
                if u.numel() == nkt:
                    assert torch.equal(cols.sort().values, torch.arange(64))
                if pol == "RP-record":  # rows sharing a subset id in one record share the subset
                    for v in u.tolist():
                        assert bool((rows[sub == v] == rows[sub == v][0]).all())


def test_fp_mag_is_the_p16_floor_mask():
    LS, KG, KLayout = _m()
    from sparse_pooling.experimental import width_masks as WM

    L, idx, feat, bk = graph(H=3003, C=64)
    w = torch.randn(L.weight_shape, generator=torch.Generator().manual_seed(3)).bfloat16()
    s = LS.build("FP-mag", 16, L, bk, w=w)
    ms = WM.build("P16", 0.25, w, L)
    assert torch.equal(s.element_mask(L), ms.element_mask(L))


def test_seeds_reproduce_and_are_independent_keys():
    LS, KG, _ = _m()
    L, idx, feat, bk = graph(H=8 * 700 - 3, C=128)
    for pol in RANDOM:
        a = LS.build(pol, 16, L, bk, graph_seed=1, layout_seed=0)
        b = LS.build(pol, 16, L, bk, graph_seed=1, layout_seed=0)
        c = LS.build(pol, 16, L, bk, graph_seed=1, layout_seed=1)
        d = LS.build(pol, 16, L, bk, graph_seed=2, layout_seed=0)
        assert a.fingerprint() == b.fingerprint() and torch.equal(a.env, b.env)
        assert a.fingerprint() != c.fingerprint() and a.fingerprint() != d.fingerprint()
    # different policies use different key streams (RP-row vs BP-row draw from distinct keys)
    r = LS.build("RP-row", 16, L, bk, layout_seed=0)
    p = LS.build("BP-row", 16, L, bk, layout_seed=0)
    gv = r.nv > 0
    assert not torch.equal(r.env[gv][:8], p.env[gv][:8])


def test_uniformity_and_independence_statistics():
    """Large CPU geometry: every local position is retained with frequency k / 64; in RP-row two distinct
    positions co-occur with C(62, 14) / C(64, 16) = (16 * 15) / (64 * 63); the subsets of DIFFERENT records
    (RP-record) / rows (RP-row) are independent: the overlap of consecutive rows has the hypergeometric mean 4."""
    LS, KG, _ = _m()
    L, idx, feat, bk = graph(D=1024, H=200_000, G=64, C=256, R=8)
    for pol in ("RP-record", "RP-row", "BP-row"):
        s = LS.build(pol, 16, L, bk, layout_seed=5)
        gm = s.group_mask()[s.nv > 0].double()
        n = gm.shape[0]
        freq = gm.mean(0)
        assert float((freq - 0.25).abs().max()) < 6 * math.sqrt(0.25 * 0.75 / n)
        if pol == "RP-row":
            co = (gm.t() @ gm) / n
            off = co[~torch.eye(64, dtype=torch.bool)]
            want = 16 * 15 / (64 * 63)
            assert abs(float(off.mean()) - want) < 2e-3 and float((off - want).abs().max()) < 6 * math.sqrt(want / n)
    s = LS.build("RP-record", 16, L, bk, layout_seed=5)
    geo = LS.geometry(L, bk)
    uv = geo["uv"]
    first = uv[geo["rank_rec"] == 0]  # row group 0 of every record: independent records
    gm = s.group_mask()[first].double()
    ov = (gm[1:] * gm[:-1]).sum(1)
    assert abs(float(ov.mean()) - 4.0) < 0.15


def exact_cov(sup, feat, t, D):
    """Exact layer-wide coverage fraction of t-tuples by brute force over the valid row groups' supports."""
    gv = torch.nonzero(sup.nv > 0).squeeze(1)
    glob = feat.to(torch.int64)[sup.q[gv][:, None], sup.env[gv]]  # [n, k] sorted global ids
    k = glob.shape[1]
    comb = torch.tensor(list(itertools.combinations(range(k), t)))
    tup = glob[:, comb]  # [n, C(k, t), t]
    code = torch.zeros(tup.shape[:2], dtype=torch.int64)
    for i in range(t):
        code = code * D + tup[..., i]
    return torch.unique(code).numel() / math.comb(D, t)


def model_given_graph(LS, sup, L, bk, feat, t, D):
    """Expected coverage over the layout randomness for the FIXED graph: mean over all t-tuples of
    1 - prod_{sets containing the tuple} (1 - h_q) (h_q of the refined per-set model)."""
    ri = LS.refined_inputs(sup, L, bk)
    h = LS.set_hit(sup.policy, t, sup.k, ri["m_q"], ri["b_q"], ri["rec_groups"], ri["rec_set"], ri["used_tiles"])
    Q = feat.shape[0]
    member = torch.zeros((Q, D), dtype=torch.bool)
    member.scatter_(1, feat.to(torch.int64), True)
    tot = 0.0
    for tup in itertools.combinations(range(D), t):
        inq = member[:, list(tup)].all(1)
        tot += 1.0 - float(torch.prod(1 - h[inq]))
    return tot / math.comb(D, t), h


@pytest.mark.parametrize("pol", ["RP-record", "RP-row", "BP-row"])
def test_coverage_models_match_a_tiny_exhaustive_oracle(pol):
    """D = 128, S = 64, R = 3 (Q = 6 sets, two per round), C = 64: 4 groups per full record, partial tails with
    fewer groups (RP-record refinement min(m_r, nkt)), partial BP-row passes.  The Monte-Carlo mean of the EXACT
    realized pair / triple coverage over 160 layout seeds matches the refined model's expectation for the fixed
    graph (5 standard errors), and the independent-round formula is close to it."""
    LS, KG, KLayout = _m()
    D = 128
    L, idx, feat, bk = graph(D=D, H=6 * 225, G=6, C=64, R=3, seed=4)
    for t in (2, 3):
        vals = []
        for ls in range(160):
            s = LS.build(pol, 16, L, bk, graph_seed=4, layout_seed=ls)
            vals.append(exact_cov(s, feat, t, D))
        v = torch.tensor(vals, dtype=torch.float64)
        want, h = model_given_graph(LS, s, L, bk, feat, t, D)
        se = float(v.std()) / math.sqrt(len(vals))
        assert abs(float(v.mean()) - want) < 5 * se + 1e-12, (pol, t, float(v.mean()), want, se)
        ir = LS.layer_expectation(h, t, D // 64, 64, D)
        assert abs(ir - want) < 0.1 * want + 1e-6


def test_fixed_partition_coverage_is_deterministic_and_within_the_one_tile_bound():
    """FP-balanced: the layout seed only moves the per-record phases, so the realized coverage depends only on the
    tiles a set uses (all of them here); it never exceeds the one-tile union bound Q nkt C(k, t) / C(D, t).  The
    per-set model (used / nkt) a_t is NOT a valid expectation for it: the four tiles of a set are ranges of its
    SORTED feature list, so the same nearby-id tuples share a tile in every set (rank correlation) -- the model is
    reported only as an approximation."""
    LS, KG, KLayout = _m()
    D = 128
    L, idx, feat, bk = graph(D=D, H=6 * 300, G=6, C=64, R=3, seed=4)
    for t in (2, 3):
        vals = {exact_cov(LS.build("FP-balanced", 16, L, bk, graph_seed=4, layout_seed=ls), feat, t, D)
                for ls in range(8)}
        assert len(vals) == 1
        Q = feat.shape[0]
        assert vals.pop() <= Q * 4 * math.comb(16, t) / math.comb(D, t) + 1e-12


@requires_cuda_backend
def test_coverage_pipeline_matches_brute_force_on_local_supports():
    import k64_coverage as KC64

    LS, KG, KLayout = _m()
    D = 256
    L, idx, feat, bk = graph(D=D, H=3003, G=8, C=64, R=2, dev="cuda")
    for pol in ("RP-record", "BP-row"):
        s = LS.build(pol, 16, L, bk, layout_seed=2, device="cuda")
        em = s.element_mask(L)
        sp = KC64.supports_from_dense(L, feat, bk, em)
        ex = KC64.exact_support_coverage(sp, (2, 3))
        sc = LS.LocalSupport(s.policy, s.k, s.env.cpu(), s.nv.cpu(), s.q.cpu(), s.subset.cpu())
        for t in (2, 3):
            assert abs(ex[t]["fraction"] - exact_cov(sc, feat.cpu(), t, D)) < 1e-12


# ------------------------------------------------------------------------------------ indexed lowering
def _layers():
    from sparse_pooling.experimental import canonical, kgx_layer, kgx_onetile, l1, reference

    return canonical, kgx_layer, kgx_onetile, l1, reference


def base_layer(H, C, G=8, D=256, R=2, seed=1):
    C_, KX, _, _, _ = _layers()
    from sparse_pooling.config import PhysicalPlan
    from sparse_pooling.experimental.kgen_layout import zero_padding_

    lay = KX.KgxLayer(D, H, G, C, ConnectivityPlan(64, R, seed=seed), variant="idf", tile=128, device="cuda", seed=3,
                      plan=PhysicalPlan(grad_precision="split"), keep_indices=True)
    g = torch.Generator(device="cuda").manual_seed(4)
    with torch.no_grad():
        lay.bias.copy_((0.3 * torch.randn(lay.bias.shape, device="cuda", generator=g)).bfloat16())
    zero_padding_(lay.layout, lay.weight.data, lay.bias.data)
    return lay


def masked(base, em):
    _, KX, _, _, _ = _layers()
    L = base.layout
    d = KX.KgxLayer(L.input_dim, L.hidden_dim, L.groups, L.block, base.connectivity, variant="idf", tile=128,
                    device="cuda", plan=base.plan, weights=(base.weight.detach() * em, base.bias.detach().clone()),
                    keep_indices=True)
    d.mask = em
    return d


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


@requires_cuda_backend
@pytest.mark.parametrize("k", [16, 8])
@pytest.mark.parametrize("C,H", [(256, 8 * 300 + 5), (64, 3003)])
def test_indexed_lowering_is_the_masked_k64_function(k, C, H):
    Cn, KX, OT, L1, REF = _layers()
    LS = _m()[0]
    base = base_layer(H, C)
    L = base.layout
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(40, L.input_dim, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(40, L.groups, device="cuda", generator=g)
    for pol in ("FP-mag",) + RANDOM:
        s = LS.build(pol, k, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, layout_seed=3, device="cuda")
        em = s.element_mask(L)
        dm = masked(base, em)
        lo = OT.lower_local(dm, s)
        lo.kcsr_validate()  # sorted distinct records inside their supergroup, buckets exact
        assert Cn.digest(lo)["sha"] == Cn.digest(dm)["sha"] and Cn.mask_digest(lo)["sha"] == Cn.mask_digest(dm)["sha"]
        assert Cn.mask_digest(lo)["enabled_slots"] == k * H
        # FP64 reference of the masked dense function (full-z ReLU decisions, enabled dW)
        w64 = dm.weight.detach().double()
        yr, dxr, dwr, dbr = REF.grads(x.double(), dm.logical_indices().long(), w64, dm.bias.detach().double(),
                                      dm.counts, dy.double())
        gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
        sl = Cn.dense_slots(dm, gg, nn)
        dwr = (dwr * em.double()).reshape(-1, 64)[sl]
        dbr = dbr.reshape(-1)[sl]
        lo.enable_main_grad()
        lo.zero_main_grad()
        xx = x.clone().requires_grad_(True)
        y = lo(xx)
        y.backward(dy)
        dw, db = lo.canon_rows(Cn.sources(lo, "main_grad"), gg, nn)
        assert maxrel(y, yr) < 1e-6 and maxrel(xx.grad, dxr) < 2e-5
        assert maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6, pol
        assert bool((dw[~Cn.dense_mask_rows(dm, gg, nn)] == 0).all())


@requires_cuda_backend
def test_indexed_lowering_optimizer_state_identity_and_restore(tmp_path):
    """Dense masked vs indexed K16 after the same 3 optimizer steps (stochastic rounding off): canonical values and
    both moments agree to accumulation-order rounding; the zero start state hashes identically; a complete
    snapshot restore reproduces the canonical identity exactly."""
    Cn, KX, OT, L1, REF = _layers()
    LS = _m()[0]
    from train_state import TrainState

    base = base_layer(8 * 300 + 5, 256)
    L = base.layout
    s = LS.build("BP-row", 16, L, base.kg_buckets, layout_seed=1, device="cuda")
    em = s.element_mask(L)
    dm = masked(base, em)
    lo = OT.lower_local(dm, s)
    o1 = L1.L1AdamW([dm.weight, dm.bias], lr=1e-3, stochastic_rounding=False)
    L1.apply_mask_(o1, dm.weight, em)
    o2 = L1.L1AdamW([lo.weight, lo.bias], lr=1e-3, stochastic_rounding=False)
    assert Cn.identity(dm, o1)["opt_m"]["sha"] == Cn.identity(lo, o2)["opt_m"]["sha"]
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(32, L.input_dim, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(32, L.groups, device="cuda", generator=g)
    for lay, opt in ((dm, o1), (lo, o2)):
        lay.enable_main_grad()
        for _ in range(3):
            lay.zero_main_grad()
            xx = x.clone().requires_grad_(True)
            lay(xx).backward(dy)
            opt.step()
    gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
    for kind in ("param", "m", "v"):
        a = dm.canon_rows(Cn.sources(dm, kind, o1), gg, nn)[0].double()
        b = lo.canon_rows(Cn.sources(lo, kind, o2), gg, nn)[0].double()
        assert maxrel(b, a) < 2e-2, kind
    assert bool((dm.canon_rows(Cn.sources(dm, "param"), gg, nn)[0][~Cn.dense_mask_rows(dm, gg, nn)] == 0).all())
    ident = Cn.identity(lo, o2)
    meta = [t.clone() for t in (lo.indices, lo.env, lo.kg_buckets, lo.sg_feat)]
    st = TrainState([lo.weight, lo.bias], o2).capture()
    fp0 = st.fingerprint()
    for _ in range(2):
        xx = x.clone().requires_grad_(True)
        lo(xx).backward(dy)
        o2.step()
    assert st.fingerprint() != fp0
    st.restore()
    assert st.fingerprint() == fp0 and Cn.identity(lo, o2) == ident
    assert all(torch.equal(a, b) for a, b in zip(meta, (lo.indices, lo.env, lo.kg_buckets, lo.sg_feat)))
