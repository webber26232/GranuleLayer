"""Interaction-coverage candidates (results/INTERACTION_COVERAGE_K16_DX.md section 3): Balanced-TopK/16 quota repair
(experimental/local_supports.py), Cover-Mix32 K32 / K16 / K8 reallocation (experimental/mix32_supports.py), the
variable-degree coverage repair, and grouped K32 / K16 / K8 execution (experimental/kgx_grouped.py, declared class
list).

Gates: quota feasibility / values / keyed layout, exact degrees and column counts, donor legality and the strict
count argument, ties / zeros, chunk / record-order independence, swap count = initial deficit, E(opt) >= E(repair)
against brute force / MILP / flow; Mix32 populations (record and full-model budgets), tail handling, q / neuron
identity, complete record coverage, rejection of too-small geometries; preserved old-policy fingerprints (values
computed at the parent commit 0e4c009); grouped K32 execution = the dense masked function and the FP64 reference
(forward, dX, dW, db; one output normalization, one dX cast), canonical parameters / optimizer states.
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.dirname(__file__))

from test_cover_topk import graph, weights  # noqa: E402


def _ls():
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental import mix32_supports as M32

    return LS, M32


ANISO = torch.linspace(0.1, 3, 64)

# fingerprints of the OLD policies on the fixed small case below, computed with the parent commit 0e4c009
PARENT_FP = {
    "TopK/16": "5287897a149380571fea8cabb66bce7531ad0ba44a88ed732bd270eab988a103",
    "Cover-TopK/16": "bae99eccae1dfadbe04af0c92271cc96d746e5ac827f003fd708239649bad4d5",
    "BP-row/16": "b84cfa418ceac8af948691b4aee6258f04532eb8c8a17581a7a1beff6e61d469",
    "Cover-TopK/8": "40e7f6239ac65b89dda3d197f7030d5ee522e86a2770b50e7ba9a4c848e2a888",
    "WRS/16": "492a84abf26af6b228716d168b4ea0923519850b1d2554c21c9171248fe26337",
    "FP-balanced/16": "53092699276eb859e40da1e796e0257e65a088ce3de696c24dc096fa086e1536",
    "RP-row/16": "79f36e999bce6cb97341d0579e6989d87317854d7f8abd8133660573a4c464b5",
    "Mix-BP": "286cf4b2c2ef228a906dc47bf4867e41467a2299c98b9921919ac52e5768e1c5",
    "Mix-TopK": "901b9ec60912bb716e7ec96f5c2715a8842701e351f160a8756667b2b8e4aa97",
}


def test_old_policy_fingerprints_preserved():
    LS, _ = _ls()
    from sparse_pooling.experimental import mixed_supports as MX

    L, idx, feat, bk = graph()
    w = weights(L, scale=ANISO)
    for name, fp in PARENT_FP.items():
        if name.startswith("Mix"):
            s = MX.build(name, L, bk, w=w, graph_seed=1, layout_seed=2, Q=feat.shape[0])
        else:
            pol, k = name.split("/")
            s = LS.build(pol, int(k), L, bk, w=w, graph_seed=1, layout_seed=2, Q=feat.shape[0],
                         L_cand=4 if pol == "WRS" else 1)
        assert s.fingerprint() == fp, name
    assert LS.RNG_SCHEMA == "granule.local_supports/1"


# ------------------------------------------------------------------------- Balanced-TopK quotas
def test_quota_values_and_keyed_layout():
    LS, _ = _ls()
    m = torch.tensor([16, 11, 11, 16, 4])
    q0 = LS.balanced_quotas(m, 16, ("Balanced-TopK", 16, "graph0", "layout0"))
    q1 = LS.balanced_quotas(m, 16, ("Balanced-TopK", 16, "graph0", "layout1"))
    assert bool((q0[0] == 4).all()) and bool((q0[3] == 4).all()) and bool((q0[4] == 1).all())
    for q in (q0, q1):
        for r in (1, 2):
            assert sorted(torch.bincount(q[r]).tolist()) == [0, 0, 16, 48]  # 48 x 3, 16 x 2
            assert int(q[r].sum()) == 176
    assert not torch.equal(q0[1], q1[1]) and not torch.equal(q0[1], q0[2])  # seed / record dependent
    assert torch.equal(q0, LS.balanced_quotas(m, 16, ("Balanced-TopK", 16, "graph0", "layout0")))  # reproducible
    with pytest.raises(ValueError):
        LS.balanced_quotas(torch.tensor([3]), 16, ("Balanced-TopK", 16, "graph0", "layout0"))  # b = 0


def scalar_quota(a, X, q):
    """Plain-Python reference of one record's quota repair: lexicographic (j, u, d) minimum of a[u][d] - a[u][j] over
    deficit j, group u without j, surplus donor d in u; checks legality and the count argument at every step."""
    m, S = len(a), len(a[0])
    X = [set(x) for x in X]
    swaps, loss = 0, 0.0
    while True:
        c = [sum(j in X[u] for u in range(m)) for j in range(S)]
        defi = [j for j in range(S) if c[j] < q[j]]
        if not defi:
            assert c == list(q)
            return X, swaps, loss
        sur = [d for d in range(S) if c[d] > q[d]]
        assert sur and all(c[d] > c[j] for d in sur for j in defi)  # strict count argument
        best = None
        for j in defi:
            for u in range(m):
                if j in X[u]:
                    continue
                for d in sorted(X[u]):
                    if c[d] > q[d]:
                        cost = a[u][d] - a[u][j]
                        if best is None or cost < best[0]:
                            best = (cost, j, u, d)
        assert best is not None, "a deficit without a legal direct swap"
        cost, j, u, d = best
        deg = len(X[u])
        X[u].remove(d)
        X[u].add(j)
        assert len(X[u]) == deg and c[d] - 1 >= 1
        swaps += 1
        loss += cost


def _record_case(seed, m, S, k, zero=False, tied=False):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand((m, S), generator=g, dtype=torch.float64) * torch.linspace(0.05, 2.0, S, dtype=torch.float64)
    if zero:
        a.zero_()
    if tied:
        a = torch.round(a * 2) / 2
    return a


@pytest.mark.parametrize("case", range(12))
def test_quota_repair_equals_scalar_reference(case):
    LS, _ = _ls()
    S, k = 16, 4
    m = [4, 5, 7, 8][case % 4]
    a = _record_case(case, m, S, k, zero=case == 10, tied=case in (3, 7, 11))
    env0 = LS.topk_select(a, k)
    rec = torch.zeros(m, dtype=torch.int64)
    rank = torch.arange(m)
    cov = LS.cover_repair(a, env0, rec, rank, k, m, S=S)
    q = LS.balanced_quotas(torch.tensor([m]), k, ("t", case), S=S)
    r = LS.quota_repair(a, cov["env"], rec, rank, k, m, q, S=S, check_every=1 + case % 3)
    Xs, sw, ls = scalar_quota(a.tolist(), [set(e.tolist()) for e in cov["env"]], q[0].tolist())
    assert [set(e.tolist()) for e in r["env"]] == Xs
    assert int(r["swaps"][0]) == sw == int(r["deficit0"][0])
    assert abs(float(r["loss"][0]) - ls) <= 1e-12 * max(1.0, abs(ls))
    X = torch.zeros((m, S), dtype=torch.bool).scatter_(1, r["env"], True)
    assert bool((X.sum(1) == k).all()) and torch.equal(X.sum(0), q[0])


def test_quota_repair_chunk_and_record_order_independent():
    LS, _ = _ls()
    L, idx, feat, bk = graph()
    w = weights(L, scale=ANISO)
    geo = LS.geometry(L, bk, "cpu", Q=feat.shape[0])
    a_v = LS.feature_importance(w, L)[geo["uv"]]
    cov = LS.cover_repair(a_v, LS.topk_select(a_v, 16), geo["rec"], geo["rank_rec"], 16, geo["gpb"])
    q = LS.balanced_quotas(geo["groups_per_record"], 16, ("Balanced-TopK", 16, "graph1", "layout0"))
    r1 = LS.quota_repair(a_v, cov["env"], geo["rec"], geo["rank_rec"], 16, geo["gpb"], q)
    r2 = LS.quota_repair(a_v, cov["env"], geo["rec"], geo["rank_rec"], 16, geo["gpb"], q, chunk_records=3,
                         check_every=5)
    assert torch.equal(r1["env"], r2["env"]) and torch.equal(r1["swaps"], r2["swaps"])
    s = LS.build("Balanced-TopK", 16, L, bk, w=w, graph_seed=1, layout_seed=0, Q=feat.shape[0])
    assert torch.equal(s.env[geo["uv"]], r1["env"])
    st = s.stats
    assert st["column_counts_equal_quotas"] and st["repair"]["swaps_equal_initial_deficit"]
    assert s.enabled_slots() == 16 * L.hidden_dim and st["layout_seed"] == 0
    full = geo["groups_per_record"] == geo["gpb"]
    gm = s.group_mask()[geo["uv"]]
    mult = LS.input_multiplicity(gm, geo["nv"][geo["uv"]], geo["rec"], geo["records"])
    assert bool((mult["groups"][full] == 4).all())
    assert set(mult["groups"][~full].flatten().tolist()) <= {2, 3}
    assert bool((mult["groups"] >= 1).all())  # complete input coverage kept
    # every row keeps exactly 16 sorted distinct local inputs, nothing on padded groups; q / neuron identity
    env = s.env[geo["uv"]]
    assert bool((env[:, 1:] > env[:, :-1]).all()) and torch.equal(s.q, LS.build("Cover-TopK", 16, L, bk, w=w,
                                                                                 Q=feat.shape[0]).q)
    em = s.element_mask(L)
    assert int(em.sum()) == 16 * L.hidden_dim and int((em.bool() & ~L.valid_mask("cpu")[..., None]).sum()) == 0


def test_quota_repair_zero_and_tied_importances_deterministic():
    LS, _ = _ls()
    L, idx, feat, bk = graph()
    z = torch.zeros(L.weight_shape, dtype=torch.bfloat16)
    s1 = LS.build("Balanced-TopK", 16, L, bk, w=z, Q=feat.shape[0])
    s2 = LS.build("Balanced-TopK", 16, L, bk, w=z, Q=feat.shape[0])
    assert s1.fingerprint() == s2.fingerprint() and s1.stats["column_counts_equal_quotas"]
    assert s1.stats["repair"]["energy_loss_total"] == 0.0


def test_quota_infeasible_geometry_rejected():
    LS, _ = _ls()
    L, idx, feat, bk = graph(H=8 * 40, C=64)  # C64: m = 4 groups, but a 40-neuron tail record has 3 groups
    w = weights(L)
    with pytest.raises(ValueError):
        LS.build("Balanced-TopK", 16, L, bk, w=w, Q=feat.shape[0])


@pytest.mark.parametrize("case", range(6))
def test_quota_optimum_bounds_brute_force_and_milp(case):
    LS, _ = _ls()
    import cover_reference as CR

    S, k, m = 6, 2, 4
    a = _record_case(100 + case, m, S, k, tied=case == 5)
    rec, rank = torch.zeros(m, dtype=torch.int64), torch.arange(m)
    cov = LS.cover_repair(a, LS.topk_select(a, k), rec, rank, k, m, S=S)
    q = LS.balanced_quotas(torch.tensor([m]), k, ("t", case), S=S)
    r = LS.quota_repair(a, cov["env"], rec, rank, k, m, q, S=S)
    Xr = torch.zeros((m, S), dtype=torch.bool).scatter_(1, r["env"], True).numpy()
    A = a.numpy()
    e_bf, Xbf = CR.quota_exhaustive(A, k, q[0].numpy())
    mo = CR.quota_milp_opt(A, k, q[0].numpy())
    fo = CR.quota_flow_opt(A, k, q[0].numpy())
    er = CR.energy(A, Xr)
    assert CR.quota_feasible(Xr, k, q[0].numpy()) and mo["certified"] and fo["feasible"]
    assert abs(mo["energy"] - e_bf) <= 1e-12 and abs(fo["energy"] - e_bf) <= 1e-9
    assert e_bf >= er - 1e-12
    # the Cover-TopK optimum (column lower bound 1) is >= the quota optimum
    assert CR.milp_opt(A, k)["energy"] >= e_bf - 1e-12


# ------------------------------------------------------------------------- variable-degree cover repair, Mix32
def test_cover_repair_mask_reproduces_cover_repair():
    LS, _ = _ls()
    L, idx, feat, bk = graph()
    w = weights(L, scale=ANISO)
    geo = LS.geometry(L, bk, "cpu", Q=feat.shape[0])
    a_v = LS.feature_importance(w, L)[geo["uv"]]
    for k in (8, 16):
        et = LS.topk_select(a_v, k)
        r0 = LS.cover_repair(a_v, et, geo["rec"], geo["rank_rec"], k, geo["gpb"])
        X0 = torch.zeros((len(et), 64), dtype=torch.bool).scatter_(1, et, True)
        r1 = LS.cover_repair_mask(a_v, X0, geo["rec"], geo["rank_rec"], geo["gpb"], chunk_records=5)
        assert torch.equal(r1["X"], torch.zeros_like(X0).scatter_(1, r0["env"], True))
        assert torch.equal(r1["swaps"], r0["swaps"]) and torch.allclose(r1["loss"], r0["loss"], rtol=0, atol=0)


@pytest.mark.parametrize("p", [1, 2])
def test_mix32_classes_budgets_tails_identity(p):
    LS, M32 = _ls()
    L, idx, feat, bk = graph()
    w = weights(L, scale=ANISO)
    s = M32.build("Cover-Mix32", p, L, bk, w, graph_seed=1, layout_seed=0, Q=feat.shape[0])
    geo = LS.geometry(L, bk, "cpu", Q=feat.shape[0])
    uv, rec, nv = geo["uv"], geo["rec"], geo["nv"]
    kc = s.kc[uv]
    gm = s.gm[uv]
    assert torch.equal(gm.sum(1), kc) and set(kc.tolist()) == {32, 16, 8}
    per = {K: torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, rec, (kc == K).long()) for K in (32, 16, 8)}
    assert bool((per[32] == p).all()) and bool((per[8] == 2 * p).all())
    assert bool((nv[uv][kc != 16] == 16).all())  # K32 / K8 only on FULL groups; the partial tail group is K16
    slots = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, rec, kc * nv[uv])
    n_rec = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, rec, nv[uv])
    assert torch.equal(slots, 16 * n_rec) and s.enabled_slots() == 16 * L.hidden_dim
    cov = torch.zeros((geo["records"], 64), dtype=torch.int64).index_add_(0, rec, gm.long())
    assert bool((cov > 0).all())  # complete record coverage
    assert s.stats["repair"]["swaps_equal_initial_missing"] and s.stats["per_record_budget_exact"]
    cv = LS.build("Cover-TopK", 16, L, bk, w=w, Q=feat.shape[0])
    assert torch.equal(s.q, cv.q) and torch.equal(s.nv, cv.nv)
    assert int((s.element_mask(L).bool() & ~L.valid_mask("cpu")[..., None]).sum()) == 0
    # layout seed changes the class map; the namespace differs from Mix-BP's class_order
    s1 = M32.build("Cover-Mix32", p, L, bk, w, graph_seed=1, layout_seed=1, Q=feat.shape[0])
    assert not torch.equal(s1.kc, s.kc) and s1.fingerprint() != s.fingerprint()
    assert M32.build("Cover-Mix32", p, L, bk, w, graph_seed=1, layout_seed=0, Q=feat.shape[0]).fingerprint() == \
        s.fingerprint()


def test_mix32_rejects_small_geometry():
    _, M32 = _ls()
    from sparse_pooling.experimental.mixed_supports import Unsupported

    L, idx, feat, bk = graph(H=8 * 64, C=64)  # 4 full groups per record < 3 p = 6
    with pytest.raises(Unsupported):
        M32.build("Cover-Mix32", 2, L, bk, weights(L), Q=feat.shape[0])


def test_tuple_incidence_table():
    _, M32 = _ls()
    rows = {"U16": {16: 16}, "p1": {32: 1, 16: 13, 8: 2}, "p2": {32: 2, 16: 10, 8: 4}}
    want = {"U16": (1920, 8960, 29120), "p1": (2112, 12352, 59760), "p2": (2304, 15744, 90400)}
    for name, cls in rows.items():
        assert tuple(M32.tuple_incidence(cls, t) for t in (2, 3, 4)) == want[name]
    # triple union bounds at the maximum 763 row groups per output (47 full records + one 11-group tail record)
    for name, b in (("U16", 0.23946), ("p1", 0.33071), ("p2", 0.42196)):
        p = 0 if name == "U16" else int(name[1])
        tail = {32: p, 16: 11 - 3 * p, 8: 2 * p} if p else {16: 11}
        tot = 47 * M32.tuple_incidence(rows[name], 3) + M32.tuple_incidence(tail, 3)
        assert abs(100 * min(1.0, tot / math.comb(1024, 3)) - b) < 1e-5


@requires_cuda_backend
@pytest.mark.parametrize("p", [1, 2])
def test_mix32_primary_geometry_counts(p):
    _, M32 = _ls()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan

    L = KLayout(1024, 12_500_000, 1024, 64, 256)
    counts = L.counts_tensor("cuda")
    idx, feat, ubk, _ = KG.generate(1024, counts, 256, 64, L.blocks_per_group, 1, ConnectivityPlan(64, 256, seed=0),
                                    "cuda")
    bk = KG.unit_to_block_buckets(ubk, counts, 256, 1).cpu()
    del idx
    w = torch.zeros(L.weight_shape, dtype=torch.bfloat16)  # classes are weight independent
    s = M32.build("Cover-Mix32", p, L, bk, w, graph_seed=0, layout_seed=0, Q=feat.shape[0])
    want = {1: (786_432, 10_140_704, 1_572_864), 2: (1_572_864, 7_781_408, 3_145_728)}[p]
    assert (s.stats["H32"], s.stats["H16"], s.stats["H8"]) == want
    assert s.enabled_slots() == 200_000_000 and s.stats["biases"] == 12_500_000
    gpo = 48 * 16
    for K, H_ in zip((32, 16, 8), want):
        cnt = torch.zeros(1024, dtype=torch.int64).index_add_(0, torch.arange(s.nv.numel()) // gpo, s.nv * (s.kc == K))
        assert torch.equal(cnt, KLayout(1024, H_, 1024, K, 16).counts_tensor("cpu").long())


# ------------------------------------------------------------------------- grouped K32 / K16 / K8 execution
@requires_cuda_backend
@pytest.mark.parametrize("p", [1, 2])
def test_grouped_k32_is_the_masked_function(p):
    from test_weighted_mixed_supports import _layers, base_layer, masked, maxrel

    _, M32 = _ls()
    Cn, GX, KX, L1, REF = _layers()
    base = base_layer(8 * 431 + 4)
    L = base.layout
    s = M32.build("Cover-Mix32", p, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, layout_seed=0,
                  device="cuda", Q=base.sg_feat.shape[0])
    em = s.element_mask(L)
    dm = masked(base, em)
    gl = GX.lower_mixed(dm, s)
    assert [pk.K for pk in gl.packs] == [32, 16, 8] and set(gl.kcsr_validate()) == {"K32", "K16", "K8"}
    assert Cn.digest(gl)["sha"] == Cn.digest(dm)["sha"] and Cn.mask_digest(gl)["sha"] == Cn.mask_digest(dm)["sha"]
    gg, nn = Cn.neurons(dm.counts, 0, dm.counts.numel())
    for B in (8, 24, 40, 64, 512):
        g = torch.Generator(device="cuda").manual_seed(B)
        x = torch.randn(B, L.input_dim, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, L.groups, device="cuda", generator=g)
        yr, dxr, dwr, dbr = REF.grads(x.double(), dm.logical_indices().long(), dm.weight.detach().double(),
                                      dm.bias.detach().double(), dm.counts, dy.double())
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
        dm.enable_main_grad()
        dm.zero_main_grad()
        x2 = x.clone().requires_grad_(True)
        dm(x2).backward(dy)
        assert maxrel(xx.grad, x2.grad) < 2e-5
    info = gl.last_info
    assert info["dx_accumulation"] == "one FP32 dX^T for all types, one cast" and info["launches_bwd"] == 6
    # BF16 input: dX is cast once (within one BF16 rounding of the FP32 accumulation)
    x16 = torch.randn(64, L.input_dim, device="cuda").bfloat16()
    dy = torch.randn(64, L.groups, device="cuda")
    a32 = x16.float().requires_grad_(True)
    gl(a32).backward(dy)
    b16 = x16.clone().requires_grad_(True)
    gl(b16).backward(dy)
    assert torch.equal(b16.grad, a32.grad.to(torch.bfloat16))


@requires_cuda_backend
def test_grouped_k32_optimizer_identity():
    from test_weighted_mixed_supports import _layers, base_layer, masked, maxrel

    _, M32 = _ls()
    Cn, GX, KX, L1, REF = _layers()
    base = base_layer(8 * 431 + 4)
    L = base.layout
    s = M32.build("Cover-Mix32", 1, L, base.kg_buckets, w=base.weight.detach(), graph_seed=1, device="cuda",
                  Q=base.sg_feat.shape[0])
    em = s.element_mask(L)
    dm = masked(base, em)
    gl = GX.lower_mixed(dm, s)
    o1 = L1.L1AdamW([dm.weight, dm.bias], lr=1e-3, stochastic_rounding=False)
    L1.apply_mask_(o1, dm.weight, em)
    o2 = L1.L1AdamW([gl.weight, gl.bias], lr=1e-3, stochastic_rounding=False)
    assert Cn.identity(dm, o1)["opt_m"]["sha"] == Cn.identity(gl, o2)["opt_m"]["sha"]
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(32, L.input_dim, device="cuda", generator=g).bfloat16().float()
    dy = torch.randn(32, L.groups, device="cuda", generator=g)
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
    # per-type main_grad slices: each pack's slice holds only its own neurons' gradients
    for pk in gl.packs:
        sl = gl.weight.main_grad[pk.w_off:pk.w_off + pk.w_len].view(-1, 16, pk.K)
        assert bool(torch.isfinite(sl).all())
