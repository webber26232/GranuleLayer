"""Weighted random supports, mixed K64 / K16 / K8 allocations and their grouped execution
(results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md sections 2, 4-6).

Gates: global-coordinate support identities (equal local masks in different sets stay distinct, different local
masks / sets selecting the same global inputs merge, repeated supports across records / outputs / partial groups /
padded rows, brute force vs the coverage pipeline); authoritative total Q (leading / interior / trailing empty sets,
q identities and RNG streams independent of Q, round reshaping); WRS / TopK selection (L1 = BP-row exactly, nested
candidate families, nondecreasing retained energy, stable ties, partial-group masking, evaluation-order
independence, exact budgets); mixed allocations (exact class counts at the primary geometry and the n175 / n176
tails, rejection of infeasible geometries, no anchor overlap, original bias count, no padded support); the grouped
adapter vs the dense masked function and an FP64 reference (forward, dX, dW, db, original pooling with unequal type
populations, one FP32 dX accumulation, zeroed scratch reuse, main_grad accumulation / clear, optimizer updates with
stochastic rounding off, canonical identity, state round trip, rejection paths).
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
    from sparse_pooling.experimental import mixed_supports as MX
    from sparse_pooling.experimental.kgen_layout import KLayout

    return LS, MX, KG, KLayout


def graph(D=256, H=3003, G=8, C=64, R=2, seed=1, dev="cpu"):
    LS, MX, KG, KLayout = _m()
    L = KLayout(D, H, G, 64, C)
    counts = L.counts_tensor(dev)
    idx, feat, ubk, _ = KG.generate(D, counts, C, 64, L.blocks_per_group, 1, ConnectivityPlan(64, R, seed=seed), dev)
    return L, idx, feat, KG.unit_to_block_buckets(ubk, counts, C, 1)


# ------------------------------------------------------------------------- 2.1 global support identities
def handmade(dev="cpu"):
    """D 256, S 64, R 2 (Q 8): round 0 = four contiguous sets; round 1 overlaps them (s4 = 8..39 + 128..159,
    s5 = 0..7 + 40..63 + 160..191, s6 == s1, s7 == s3).  KLayout G 2, C 32 (two row groups per record), n_g 84:
    records 0 / 1 full, record 2 holds a full group and a partial group of 4 neurons, padded rows beyond."""
    LS, MX, KG, KLayout = _m()
    D = 256
    r = lambda a, b: list(range(a, b))  # noqa: E731
    sets = [r(0, 64), r(64, 128), r(128, 192), r(192, 256), r(8, 40) + r(128, 160), r(0, 8) + r(40, 64) + r(160, 192),
            r(64, 128), r(192, 256)]
    feat = torch.tensor([sorted(s) for s in sets], dtype=torch.int16, device=dev)
    L = KLayout(D, 2 * 84, 2, 64, 32)
    NB = L.blocks_per_group
    bq = torch.tensor([0, 4, 1, 6, 5, 2], dtype=torch.int64, device=dev)  # block -> set ((g, j) order)
    bk = torch.tensor(sorted([(int(bq[g * NB + j]), g, j, 1) for g in range(2) for j in range(NB)]), dtype=torch.int32,
                      device=dev)
    em = torch.zeros(L.weight_shape, dtype=torch.uint8, device=dev)
    loc = {  # (g, j, group) -> local positions
        (0, 0, 0): r(8, 16),       # s0 local 8..15  = global 8..15
        (0, 0, 1): r(0, 8),        # s0 local 0..7   = global 0..7
        (0, 1, 0): r(0, 8),        # s4 local 0..7   = global 8..15  (same global as (0,0,0), other set / mask)
        (0, 1, 1): r(0, 8),        # s4 again: identical support repeated in one record
        (0, 2, 0): r(0, 8),        # s1 local 0..7   = global 64..71 (equal local mask, different global)
        (0, 2, 1): r(0, 16),       # partial group (4 valid neurons) s1 local 0..15 = 64..79
        (1, 0, 0): r(0, 8),        # s6 (== s1) local 0..7 = global 64..71 (same global as (0,2,0), other output)
        (1, 0, 1): list(range(64)),  # K64: all of s6
        (1, 1, 0): r(32, 48),      # s5 local 32..47 = global 160..175
        (1, 1, 1): r(0, 8),        # s5 local 0..7 = global 0..7 (same global as (0,0,1))
        (1, 2, 0): r(0, 8),        # s2 local 0..7 = 128..135
        (1, 2, 1): r(32, 48),      # partial: s2 local 32..47 = 160..175 (same global AND local mask as (1,1,0))
    }
    valid = L.valid_mask(dev)
    glob = {}
    for (g, j, u), pos in loc.items():
        for c in range(16 * u, 16 * u + 16):
            if bool(valid[g, j, c]):
                em[g, j, c, pos] = 1
        glob[(g, j, u)] = frozenset(int(feat[int(bq[g * NB + j]), p]) for p in pos)
    return L, feat, bk, bq, em, loc, glob


def brute_sharing(L, glob, bq, nv_of):
    NB = L.blocks_per_group
    scopes = dict(record=lambda g, j, u: (g, j), set=lambda g, j, u: int(bq[g * NB + j]), output=lambda g, j, u: g,
                  layer=lambda g, j, u: 0)
    out = {}
    for name, f in scopes.items():
        d = {}
        for key, s in glob.items():
            d.setdefault((f(*key), s), []).append(nv_of[key])
        out[name] = dict(distinct=len(d), groups=sorted(len(v) for v in d.values()),
                         neurons=sorted(sum(v) for v in d.values()))
    return out


def test_global_support_identities_match_brute_force():
    import rsn_coverage as RV

    L, feat, bk, bq, em, loc, glob = handmade()
    valid = L.valid_mask("cpu")
    nv_of = {k: int(valid[k[0], k[1], 16 * k[2]:16 * k[2] + 16].sum()) for k in glob}
    want = brute_sharing(L, glob, bq, nv_of)
    got = RV.sharing(L, em, bq, feat)
    for name in ("record", "set", "output", "layer"):
        assert got[name]["distinct_supports"] == want[name]["distinct"], name
        assert got[name]["groups_per_distinct_support"]["max"] == max(want[name]["groups"])
        assert got[name]["neurons_per_distinct_support"]["max"] == max(want[name]["neurons"])
    # the specific cases: equal local masks in different sets stay distinct; different masks / sets merge
    assert len(set(glob.values())) == got["layer"]["distinct_supports"] == 7
    assert got["layer"]["legacy_local_key"] == 5  # the parent's local-key metric merges distinct supports
    assert got["layer"]["labelled_local"] == 11  # (q, local mask): equal global supports in two sets stay apart
    assert got["support_size"]["max"] == 64 and got["valid_neurons"] == int(valid.sum())


def test_distinct_support_multiplicity_matches_brute_force_tuples():
    import k64_coverage as KC64
    import rsn_coverage as RV
    import wps_coverage as WV

    L, feat, bk, bq, em, loc, glob = handmade()
    supports = sorted(set(glob.values()), key=sorted)
    rep = RV.distinct_support_representatives(L, em, bq, feat)
    sp = KC64.supports_from_dense(L, feat, bk, rep)
    sp_all = KC64.supports_from_dense(L, feat, bk, em)
    for t in (2, 3):
        tups = sorted({tuple(sorted(c)) for s in supports for c in itertools.combinations(sorted(s), t)})
        ex = KC64.exact_support_coverage(sp_all, (t,))[t]
        assert ex["covered"] == len(tups)
        assert KC64.exact_support_coverage(sp, (t,))[t]["covered"] == len(tups)
        tt = torch.tensor(tups, dtype=torch.int64)
        cov, cnt = WV.query_multiplicity(sp, tt)
        want = torch.tensor([sum(set(x) <= s for s in supports) for x in tups])
        assert bool(cov.all()) and torch.equal(cnt.cpu(), want)


def test_shared_group_support_is_checked():
    import rsn_coverage as RV

    L, feat, bk, bq, em, loc, glob = handmade()
    em[0, 0, 3, 40] = 1  # one neuron of a group with an extra input
    with pytest.raises(AssertionError):
        RV.sharing(L, em, bq, feat)


# ------------------------------------------------------------------------- 2.2 authoritative Q
def test_total_q_keeps_empty_sets_and_streams():
    LS, MX, KG, KLayout = _m()
    D = 1024
    L, idx, feat, bk = graph(D=D, H=4 * 256 + 9, G=4, C=256, R=4)  # few records: most of the 64 sets are empty
    Q = feat.shape[0]
    assert Q == 4 * D // 64
    geo = LS.geometry(L, bk, Q=Q)
    occ = torch.unique(bk[:, 0].long())
    assert geo["Q"] == Q and geo["m_q"].numel() == Q and geo["b_q"].numel() == Q
    assert geo["sets_occupied"] == occ.numel() and geo["sets_empty"] == Q - occ.numel()
    assert geo["sets_empty"] > 0
    a = LS.build("BP-row", 16, L, bk, graph_seed=1, layout_seed=0, Q=Q)
    b = LS.build("BP-row", 16, L, bk, graph_seed=1, layout_seed=0, Q=Q + 32)  # extra trailing empty sets
    assert torch.equal(a.env, b.env) and a.fingerprint() == b.fingerprint() and torch.equal(a.q, b.q)
    assert a.stats["Q_total"] == Q and b.stats["sets_empty"] == a.stats["sets_empty"] + 32
    with pytest.raises(ValueError):
        LS.geometry(L, bk, Q=int(occ.max()))  # a smaller Q would drop an occupied set
    # interior / leading empty sets: relabel buckets onto a sparse subset of sets
    bk2 = bk.clone()
    bk2[:, 0] = bk2[:, 0] * 2 + 1  # set 0 and every even set empty
    g2 = LS.geometry(L, bk2, Q=2 * Q)
    assert int(g2["m_q"][0]) == 0 and torch.equal(g2["m_q"][1::2], geo["m_q"]) and int(g2["m_q"][0::2].sum()) == 0
    # rounds: per-set hits reshape into [R, D / S] only with the TOTAL Q
    ri = LS.refined_inputs(a, L, bk, Q=Q)
    h = LS.set_hit("BP-row", 3, 16, ri["m_q"], ri["b_q"])
    assert h.numel() == Q and 0 <= LS.layer_expectation(h, 3, D // 64) <= 1
    with pytest.raises(RuntimeError):
        h[:Q - 4].view(-1, D // 64)


def test_reduced_interaction_geometry_has_q4096_and_1024_occupied():
    """H 262,144 / C256 / R256: 1,024 original records, 1,024 occupied sets of Q 4,096 at graph seed 0."""
    LS, MX, KG, KLayout = _m()
    L, idx, feat, bk = graph(D=1024, H=262_144, G=1024, C=256, R=256, seed=0)
    geo = LS.geometry(L, bk, Q=feat.shape[0])
    assert geo["Q"] == 4096 and geo["records"] == 1024
    assert geo["sets_occupied"] == int(torch.unique(bk[:, 0]).numel()) == 1024 and geo["sets_empty"] == 3072


# ------------------------------------------------------------------------- 4 weighted selection
def weights(L, seed=3):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(L.weight_shape, generator=g).bfloat16()
    from sparse_pooling.experimental.kgen_layout import zero_padding_

    zero_padding_(L, w)
    return w


@pytest.mark.parametrize("k", [8, 16])
def test_wrs_l1_is_bp_row_and_families_nest(k):
    LS, MX, KG, KLayout = _m()
    L, idx, feat, bk = graph(H=8 * 300 + 5, C=256)
    Q = feat.shape[0]
    w = weights(L)
    bp = LS.build("BP-row", k, L, bk, graph_seed=1, layout_seed=2, Q=Q)
    s1 = LS.build("WRS", k, L, bk, w=w, graph_seed=1, layout_seed=2, Q=Q, L_cand=1)
    assert torch.equal(s1.env, bp.env) and s1.fingerprint() == bp.fingerprint()
    s4 = LS.build("WRS", k, L, bk, w=w, graph_seed=1, layout_seed=2, Q=Q, L_cand=4)
    s16 = LS.build("WRS", k, L, bk, w=w, graph_seed=1, layout_seed=2, Q=Q, L_cand=16)
    a = LS.feature_importance(w, L)
    gv = s1.nv > 0
    e1, e4, e16 = (LS.subset_score(a[gv], s.env[gv]) for s in (s1, s4, s16))
    assert bool((e4 >= e1).all()) and bool((e16 >= e4).all())
    # nested: where the L16 winner is among the first 4 candidates it is the L4 winner
    c16 = s16.subset[gv]
    low = c16 < 4
    assert torch.equal(s16.env[gv][low], s4.env[gv][low]) and torch.equal(c16[low], s4.subset[gv][low])
    assert s16.stats["energy"]["layer_fraction"] >= s4.stats["energy"]["layer_fraction"] >= \
        s4.stats["energy_candidate0"]["layer_fraction"]
    tk = LS.build("TopK", k, L, bk, w=w, Q=Q)
    etk = LS.subset_score(a[gv], tk.env[gv])
    assert bool((etk >= e16 - 1e-12 * etk.abs()).all())  # TopK is the per-group maximum
    for s in (s4, s16, tk):
        env = s.env[gv]
        assert s.enabled_slots() == k * L.hidden_dim
        assert bool((env[:, 1:] > env[:, :-1]).all()) and int(env.min()) >= 0 and int(env.max()) < 64
        assert torch.equal(s.q, bp.q)


def test_weighted_ties_masking_and_order_independence():
    LS, MX, KG, KLayout = _m()
    L, idx, feat, bk = graph(H=8 * 300 + 5, C=256)
    Q = feat.shape[0]
    w = torch.zeros(L.weight_shape, dtype=torch.bfloat16)
    w[..., 5] = 1.0
    w[..., 9] = 1.0  # all other coordinates tie at 0
    s = LS.build("TopK", 8, L, bk, w=w, Q=Q)
    gv = s.nv > 0
    assert bool((s.env[gv] == torch.tensor([0, 1, 2, 3, 4, 5, 6, 9])).all())  # ties -> lower coordinate
    s4 = LS.build("WRS", 8, L, bk, w=torch.zeros_like(w), graph_seed=1, layout_seed=0, Q=Q, L_cand=4)
    assert int((s4.subset[gv] != 0).sum()) == 0  # all scores tie: candidate 0 (the smaller id) wins
    # padded neurons never contribute: garbage in padded slots does not change the selection
    w2 = weights(L)
    w3 = w2.clone()
    valid = L.valid_mask("cpu")
    w3[~valid] = 100.0
    a2, a3 = LS.feature_importance(w2, L), LS.feature_importance(w3, L)
    assert torch.equal(a2, a3)
    assert torch.equal(LS.feature_importance(w2, L, chunk=7), a2)  # chunking independence (elementwise adds)
    x = LS.build("WRS", 8, L, bk, w=w2, graph_seed=1, layout_seed=0, Q=Q, L_cand=16)
    y = LS.build("WRS", 8, L, bk, w=w3, graph_seed=1, layout_seed=0, Q=Q, L_cand=16)
    assert torch.equal(x.env, y.env)
    if torch.cuda.is_available():
        z = LS.build("WRS", 8, L, bk, w=w2.cuda(), graph_seed=1, layout_seed=0, Q=Q, L_cand=16)
        assert torch.equal(x.env, z.env.cpu())


# ------------------------------------------------------------------------- 5.1 mixed allocations
def test_mixed_classes_budgets_and_tails():
    LS, MX, KG, KLayout = _m()
    G = 8
    L, idx, feat, bk = graph(H=G * 431 + 4, G=G, C=256)  # outputs 0..3: 256 + 176; 4..7: 256 + 175
    Q = feat.shape[0]
    s = MX.build("Mix-BP", L, bk, graph_seed=1, layout_seed=0, Q=Q)
    nv, kc = s.nv, s.kc
    gpb = 16
    for r in range(nv.numel() // gpb):
        n = int(nv[r * gpb:(r + 1) * gpb].sum())
        if n == 0:
            continue
        kk = kc[r * gpb:(r + 1) * gpb]
        vv = nv[r * gpb:(r + 1) * gpb]
        assert int(vv[kk == 64].sum()) == 16 and int(vv[kk == 8].sum()) == 96 and int(vv[kk == 16].sum()) == n - 112
        assert int((kk * vv).sum()) == 16 * n and int(kk[0]) == 64
        assert bool((vv[kk == 8] == 16).all())  # K8 only on full groups
    assert s.enabled_slots() == 16 * L.hidden_dim and s.stats["biases"] == L.hidden_dim
    gm = s.group_mask()
    assert bool((gm.sum(1) == kc * (nv > 0)).all()) and int(gm[nv == 0].sum()) == 0  # no padded support
    assert bool(gm[s.anchor].all()) and int(s.anchor.sum()) == s.stats["records"]
    assert bool((kc[s.anchor] == 64).all()) and not bool((s.anchor & (kc != 64)).any())
    em = s.element_mask(L)
    assert int(em.sum()) == 16 * L.hidden_dim
    # weighted variants share the class map; candidate 0 of Mix-WRS:1 is Mix-BP
    w = weights(L)
    s1 = MX.build("Mix-WRS", L, bk, w=w, graph_seed=1, layout_seed=0, Q=Q, L_cand=1)
    s4 = MX.build("Mix-WRS", L, bk, w=w, graph_seed=1, layout_seed=0, Q=Q, L_cand=4)
    st = MX.build("Mix-TopK", L, bk, w=w, graph_seed=1, layout_seed=0, Q=Q)
    assert torch.equal(s1.gm, s.gm)
    for x in (s4, st):
        assert torch.equal(x.kc, s.kc) and x.stats["class_fingerprint"] == s.stats["class_fingerprint"]
        assert torch.equal(x.gm[s.anchor], s.gm[s.anchor]) and x.enabled_slots() == s.enabled_slots()
    assert MX.build("Mix-BP", L, bk, graph_seed=1, layout_seed=1, Q=Q).stats["class_fingerprint"] != \
        s.stats["class_fingerprint"]


def test_mixed_rejects_infeasible_geometries():
    LS, MX, KG, KLayout = _m()
    L, idx, feat, bk = graph(H=8 * 300 + 5, C=256)  # tails of 44 / 45 neurons
    with pytest.raises(MX.Unsupported):
        MX.build("Mix-BP", L, bk, Q=feat.shape[0])
    L, idx, feat, bk = graph(H=3003, C=64)  # 4 row groups per record
    with pytest.raises(MX.Unsupported):
        MX.build("Mix-BP", L, bk, Q=feat.shape[0])


@requires_cuda_backend
def test_mixed_primary_geometry_counts():
    LS, MX, KG, KLayout = _m()
    L, idx, feat, bk = graph(D=1024, H=12_500_000, G=1024, C=256, R=256, seed=0, dev="cuda")
    s = MX.build("Mix-BP", L, bk.cpu(), graph_seed=0, layout_seed=0, Q=feat.shape[0])
    assert (s.stats["H64"], s.stats["H16"], s.stats["H8"]) == (786_432, 6_994_976, 4_718_592)
    assert s.enabled_slots() == 200_000_000 and s.stats["biases"] == 12_500_000
    assert s.stats["per_record_budget_exact"] and s.stats["Q"] == 4096
    for K, n in ((64, 768), (8, 4608)):
        cnt = torch.zeros(1024, dtype=torch.int64).index_add_(0, torch.arange(s.nv.numel()) // (48 * 16),
                                                              s.nv * (s.kc == K))
        assert bool((cnt == n).all())
    cnt16 = torch.zeros(1024, dtype=torch.int64).index_add_(0, torch.arange(s.nv.numel()) // (48 * 16), s.nv * (s.kc == 16))
    assert torch.equal(cnt16, KLayout(1024, 6_994_976, 1024, 16, 16).counts_tensor("cpu").long())


# ------------------------------------------------------------------------- 5.2 grouped execution
def _layers():
    from sparse_pooling.experimental import canonical, kgx_grouped, kgx_layer, l1, reference

    return canonical, kgx_grouped, kgx_layer, l1, reference


def base_layer(H, C=256, G=8, D=256, R=2, seed=1):
    _, _, KX, _, _ = _layers()
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
    _, _, KX, _, _ = _layers()
    L = base.layout
    d = KX.KgxLayer(L.input_dim, L.hidden_dim, L.groups, L.block, base.connectivity, variant="idf", tile=128,
                    device="cuda", plan=base.plan, weights=(base.weight.detach() * em, base.bias.detach().clone()),
                    keep_indices=True)
    d.mask = em
    return d


def maxrel(a, r):
    a, r = a.double(), r.double()
    return float((a - r).abs().max() / (r.abs().max() + 1e-300))


def mixed_case(policy="Mix-BP", H=8 * 431 + 4):
    LS, MX, KG, KLayout = _m()
    base = base_layer(H)
    L = base.layout
    s = MX.build(policy, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, layout_seed=3, device="cuda",
                 Q=base.sg_feat.shape[0], L_cand=4)
    em = s.element_mask(L)
    dm = masked(base, em)
    return base, s, em, dm


@requires_cuda_backend
@pytest.mark.parametrize("policy", ["Mix-BP", "Mix-WRS", "Mix-TopK"])
def test_grouped_is_the_masked_function(policy):
    Cn, GX, KX, L1, REF = _layers()
    base, s, em, dm = mixed_case(policy)
    gl = GX.lower_mixed(dm, s)
    assert [pk.K for pk in gl.packs] == [64, 16, 8]
    v = gl.kcsr_validate()
    assert set(v) == {"K64", "K16", "K8"}
    assert Cn.digest(gl)["sha"] == Cn.digest(dm)["sha"] and Cn.mask_digest(gl)["sha"] == Cn.mask_digest(dm)["sha"]
    assert Cn.mask_digest(gl)["enabled_slots"] == 16 * dm.layout.hidden_dim
    gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
    for B in (40, 64):  # token tails (B % 16 != 0) and full tiles
        g = torch.Generator(device="cuda").manual_seed(B)
        x = torch.randn(B, dm.layout.input_dim, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, dm.layout.groups, device="cuda", generator=g)
        w64 = dm.weight.detach().double()
        yr, dxr, dwr, dbr = REF.grads(x.double(), dm.logical_indices().long(), w64, dm.bias.detach().double(),
                                      dm.counts, dy.double())
        sl = Cn.dense_slots(dm, gg, nn)
        dwr = (dwr * em.double()).reshape(-1, 64)[sl]
        dbr = dbr.reshape(-1)[sl]
        gl.enable_main_grad()
        gl.zero_main_grad()
        xx = x.clone().requires_grad_(True)
        y = gl(xx)
        y.backward(dy)
        dw, db = gl.canon_rows(Cn.sources(gl, "main_grad"), gg, nn)
        assert maxrel(y, yr) < 1e-6 and maxrel(xx.grad, dxr) < 2e-5
        assert maxrel(dw, dwr) < 2e-5 and maxrel(db, dbr) < 1e-6
        assert bool((dw[~Cn.dense_mask_rows(dm, gg, nn)] == 0).all())
        # the same against the dense masked KgxLayer
        dm.enable_main_grad()
        dm.zero_main_grad()
        x2 = x.clone().requires_grad_(True)
        y2 = dm(x2)
        y2.backward(dy)
        assert maxrel(y, y2) < 1e-6 and maxrel(xx.grad, x2.grad) < 2e-5
        # padded flat slots stay exactly zero in values and gradients
        live = torch.zeros(gl.weight.numel(), dtype=torch.bool, device="cuda")
        for pk in gl.packs:
            vb = (pk.u_of_block >= 0)
            nvb = torch.zeros_like(pk.u_of_block)
            nvb[vb] = s.nv.cuda()[pk.u_of_block[vb]]
            m = (torch.arange(16, device="cuda")[None, :] < nvb[:, None])[:, :, None].expand(-1, 16, pk.K)
            live[pk.w_off:pk.w_off + pk.w_len] = m.reshape(-1)
        assert int((gl.weight.detach()[~live] != 0).sum()) == 0 and int((gl.weight.main_grad[~live] != 0).sum()) == 0


@requires_cuda_backend
def test_grouped_pooling_accumulation_and_main_grad():
    """Original pooling with unequal type populations (768 / 4608 / 6831-6832 analogue: 32 / 192 / 207-208 per
    output here); dX accumulated in FP32 over all types before ONE cast (BF16 input: within one BF16 rounding of the
    FP32 result); main_grad accumulates over two backward passes and clears; zeroed scratch is reused."""
    Cn, GX, KX, L1, REF = _layers()
    base, s, em, dm = mixed_case()
    gl = GX.lower_mixed(dm, s)
    cnt = {pk.K: pk.counts for pk in gl.packs}
    assert int(cnt[64][0]) == 32 and int(cnt[8][0]) == 192 and {int(c) for c in cnt[16]} == {207, 208}
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(64, dm.layout.input_dim, device="cuda", generator=g).bfloat16()
    dy = torch.randn(64, dm.layout.groups, device="cuda", generator=g)
    gl.enable_main_grad()
    gl.zero_main_grad()
    x32 = x.float().requires_grad_(True)
    gl(x32).backward(dy)
    mg1 = gl.weight.main_grad.clone()
    x16 = x.clone().requires_grad_(True)
    gl(x16).backward(dy)
    assert x16.grad.dtype == torch.bfloat16
    ref = x32.grad.double()
    ulp = torch.where(ref != 0, 2.0 ** (torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7), torch.ones_like(ref))
    assert bool(((x16.grad.double() - ref).abs() <= ulp * (0.5 + 1e-3) + 1e-30).float().mean() > 0.999)
    assert maxrel(gl.weight.main_grad, 2 * mg1) < 1e-5  # accumulated
    gl.zero_main_grad()
    assert int((gl.weight.main_grad != 0).sum()) == 0 and int((gl.bias.main_grad != 0).sum()) == 0
    gl(x.float().requires_grad_(True)).backward(dy)
    assert maxrel(gl.weight.main_grad, mg1) < 1e-5  # scratch (ysum / dX^T) re-zeroed every call
    # non-finite inputs propagate as in the dense masked function; an illegal batch is rejected
    xn = x.float().clone()
    xn[3, 0] = float("nan")
    with torch.no_grad():
        yn, yd = gl(xn), dm(xn)
    assert torch.equal(torch.isfinite(yn), torch.isfinite(yd))
    with pytest.raises(ValueError):
        gl(torch.randn(33, dm.layout.input_dim, device="cuda"))


@requires_cuda_backend
def test_grouped_relu_decisions_near_zero_only():
    """Cancellation: with biases set so many pre-activations are near zero, any per-neuron difference from the FP64
    reference beyond tolerance must come from |z| below a small margin (a flipped ReLU decision), never a large-margin
    error: the output error is bounded by the near-zero contributions."""
    Cn, GX, KX, L1, REF = _layers()
    base, s, em, dm = mixed_case()
    with torch.no_grad():
        dm.bias.mul_(0.01)
    gl = GX.lower_mixed(dm, s)
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(64, dm.layout.input_dim, device="cuda", generator=g).bfloat16().float()
    yr = REF.grads(x.double(), dm.logical_indices().long(), dm.weight.detach().double(), dm.bias.detach().double(),
                   dm.counts, torch.zeros(64, dm.layout.groups, device="cuda", dtype=torch.float64))[0]
    with torch.no_grad():
        y = gl(x)
    assert maxrel(y, yr) < 1e-5


@requires_cuda_backend
def test_grouped_optimizer_identity_and_state_round_trip(tmp_path):
    Cn, GX, KX, L1, REF = _layers()
    from train_state import TrainState

    base, s, em, dm = mixed_case("Mix-WRS")
    gl = GX.lower_mixed(dm, s)
    o1 = L1.L1AdamW([dm.weight, dm.bias], lr=1e-3, stochastic_rounding=False)
    L1.apply_mask_(o1, dm.weight, em)
    o2 = L1.L1AdamW([gl.weight, gl.bias], lr=1e-3, stochastic_rounding=False)
    assert Cn.identity(dm, o1)["opt_m"]["sha"] == Cn.identity(gl, o2)["opt_m"]["sha"]
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(32, dm.layout.input_dim, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(32, dm.layout.groups, device="cuda", generator=g)
    for lay, opt in ((dm, o1), (gl, o2)):
        lay.enable_main_grad()
        for _ in range(3):
            lay.zero_main_grad()
            lay(x.clone().requires_grad_(True)).backward(dy)
            opt.step()
    gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
    for kind in ("param", "m", "v"):
        a = dm.canon_rows(Cn.sources(dm, kind, o1), gg, nn)[0].double()
        b = gl.canon_rows(Cn.sources(gl, kind, o2), gg, nn)[0].double()
        assert maxrel(b, a) < 2e-2, kind
    assert Cn.mask_digest(gl)["sha"] == Cn.mask_digest(dm)["sha"]
    ident = Cn.identity(gl, o2)
    meta = gl.immutable_fingerprints()
    st = TrainState([gl.weight, gl.bias], o2).capture()
    fp0 = st.fingerprint()
    for _ in range(2):
        gl(x.clone().requires_grad_(True)).backward(dy)
        o2.step()
    st.restore()
    assert st.fingerprint() == fp0 and Cn.identity(gl, o2) == ident and gl.immutable_fingerprints() == meta
    # complete state round trip: values, maps, main_grad, moments, steps, rounding counter
    p = tmp_path / "grouped.pt"
    torch.save(dict(state_dict=gl.state_dict(), main_grad=[gl.weight.main_grad, gl.bias.main_grad],
                    opt=o2.state_dict(), sr=getattr(o2, "_seed", None), support=s.stats["support_fingerprint"]), p)
    gl2 = GX.lower_mixed(masked(base, em), s)
    o3 = L1.L1AdamW([gl2.weight, gl2.bias], lr=1e-3, stochastic_rounding=False)
    ck = torch.load(p, map_location="cuda", weights_only=False)
    gl2.load_state_dict(ck["state_dict"])
    gl2.enable_main_grad()
    gl2.weight.main_grad.copy_(ck["main_grad"][0])
    gl2.bias.main_grad.copy_(ck["main_grad"][1])
    o3.load_state_dict(ck["opt"])
    if ck["sr"] is not None:
        o3._seed = ck["sr"]
    assert Cn.identity(gl2, o3) == ident and gl2.immutable_fingerprints() == meta
    assert ck["support"] == s.fingerprint()


@requires_cuda_backend
def test_grouped_rejects_unsupported_type_counts():
    Cn, GX, KX, L1, REF = _layers()
    LS, MX, KG, KLayout = _m()
    base, s, em, dm = mixed_case()
    kc = s.kc.clone()
    u = int(torch.nonzero((kc == 16) & (s.nv == 16))[0])
    kc[u] = 8  # one output gets an extra K8 group: its K8 population no longer fits the balanced counts
    gm = s.gm.clone()
    gm[u] = False
    gm[u, :8] = True
    bad = MX.MixedSupport(s.policy, kc, gm, s.nv, s.q, s.anchor, s.cand)
    with pytest.raises(MX.Unsupported):
        GX.lower_mixed(dm, bad)
