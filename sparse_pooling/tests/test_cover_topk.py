"""Coverage-constrained TopK (experimental/local_supports.py ``Cover-TopK`` / ``cover_repair``) and the corrected
shared-query coverage uncertainty (benchmarks/joint_coverage.py); results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md
sections 2.1, 3, 4 and 7.

Gates: exact k per valid group and record coverage S (full records and the n175 / n176 tails), source / neuron / q
identity and canonical sorted coordinates, exact k H valid slots; rejection of infeasible geometries (m k < S);
all-zero / tied importances, unchanged fully covered TopK masks, the lexicographic (j, u, d) tie rule against a scalar
reference, chunk / record-order independence, swap count = initial missing inputs, donor legality and monotone
coverage, no duplicate edges; E(TopK) >= E(optimum) >= E(repair) against brute force and the MILP / min-cost-flow
references; mask vs numerical coverage; preserved TopK / BP-row supports; the joint-query interval (duplicated
outputs add no Monte Carlo evidence, paired masks keep shared queries); the indexed K8 / K16 path vs the dense masked
function and an optimizer-state round trip on a Cover-TopK support.
"""
import math
import os
import sys

import numpy as np
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


def graph(D=256, H=8 * 431 + 4, G=8, C=256, R=2, seed=1):
    LS, KG, KLayout = _m()
    L = KLayout(D, H, G, 64, C)
    counts = L.counts_tensor("cpu")
    idx, feat, ubk, _ = KG.generate(D, counts, C, 64, L.blocks_per_group, 1, ConnectivityPlan(64, R, seed=seed), "cpu")
    return L, idx, feat, KG.unit_to_block_buckets(ubk, counts, C, 1)


def weights(L, seed=3, scale=None):
    from sparse_pooling.experimental.kgen_layout import zero_padding_

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(L.weight_shape, generator=g)
    if scale is not None:  # strongly anisotropic coordinates: TopK leaves many inputs uncovered
        w = w * scale
    w = w.bfloat16()
    zero_padding_(L, w)
    return w


def scalar_repair(a, X, k):
    """Plain-Python reference of one record: a [m][S] floats, X list of sets; returns (X, swaps, loss) and checks
    donor legality / monotone coverage / no duplicates at every step."""
    m, S = len(a), len(a[0])
    X = [set(x) for x in X]
    swaps, loss = 0, 0.0
    while True:
        c = [sum(j in X[u] for u in range(m)) for j in range(S)]
        miss = [j for j in range(S) if c[j] == 0]
        if not miss:
            return X, swaps, loss
        best = None
        for j in miss:
            for u in range(m):
                for d in sorted(X[u]):
                    if c[d] >= 2:
                        cost = a[u][d] - a[u][j]
                        if best is None or cost < best[0]:  # strict: the first (j, u, d) in lexicographic order wins
                            best = (cost, j, u, d)
        assert best is not None, "an uncovered input without a legal donor"
        cost, j, u, d = best
        assert j not in X[u] and d in X[u] and c[d] >= 2
        covered_before = S - len(miss)
        X[u].remove(d)
        X[u].add(j)
        assert len(X[u]) == k
        assert sum(any(jj in X[uu] for uu in range(m)) for jj in range(S)) == covered_before + 1
        swaps += 1
        loss += cost


def run_repair(LS, a, env0, k, chunk, S=64, order=None):
    """cover_repair on records given as lists of groups (a [n, S] float64, env0 [n, k]); optional record order
    permutation of the group storage (records stay intact)."""
    n = a.shape[0]
    rec = torch.arange(n) // 16 if order is None else order
    rank = torch.zeros(n, dtype=torch.int64)
    for r in torch.unique(rec):
        ii = torch.nonzero(rec == r).squeeze(1)
        rank[ii] = torch.arange(len(ii))
    return LS.cover_repair(a, env0, rec, rank, k, 16, S=S, chunk_records=chunk)


@pytest.mark.parametrize("k,S", [(3, 12), (4, 16), (8, 64)])
def test_batched_repair_matches_the_scalar_reference(k, S):
    LS, _, _ = _m()
    g = torch.Generator().manual_seed(k)
    m_of = [16, 11, 5, 16, 7]
    m_of = [m for m in m_of if m * k >= S]
    a_list, env_list, rec = [], [], []
    for r, m in enumerate(m_of):
        a = torch.rand((m, S), generator=g, dtype=torch.float64) ** 4  # skewed: TopK concentrates
        a[:, : S // 3] *= 8.0
        if r == 1:
            a[:, 5] = a[0, 5]  # exact ties across groups
            a[2, :] = 0.25  # an all-tied group
        e0 = torch.sort(a, dim=1, descending=True, stable=True).indices[:, :k].sort(1).values
        a_list.append(a)
        env_list.append(e0)
        rec += [r] * m
    a = torch.cat(a_list)
    env0 = torch.cat(env_list)
    rec = torch.tensor(rec)
    rank = torch.cat([torch.arange(m) for m in m_of])
    res = LS.cover_repair(a, env0, rec, rank, k, 16, S=S, chunk_records=2)
    o = 0
    for r, m in enumerate(m_of):
        X, sw, loss = scalar_repair(a_list[r].tolist(), [set(x) for x in env_list[r].tolist()], k)
        got = [set(x) for x in res["env"][o:o + m].tolist()]
        assert got == X, r
        assert int(res["swaps"][r]) == sw == int(res["missing0"][r])
        assert math.isclose(float(res["loss"][r]), loss, rel_tol=0, abs_tol=1e-12)
        o += m
    # chunking cannot change a result
    for ch in (1, 3, 1 << 15):
        r2 = LS.cover_repair(a, env0, rec, rank, k, 16, S=S, chunk_records=ch)
        assert torch.equal(r2["env"], res["env"]) and torch.equal(r2["loss"], res["loss"])


def test_record_order_independence():
    LS, _, _ = _m()
    g = torch.Generator().manual_seed(7)
    nrec, k = 6, 8
    a = torch.rand((nrec * 16, 64), generator=g, dtype=torch.float64) ** 6
    env0 = LS.topk_select(a, k)
    base = run_repair(LS, a, env0, k, chunk=4)
    perm_rec = torch.randperm(nrec, generator=g)  # relabel records: each keeps its groups (canonical order inside)
    rec = perm_rec[torch.arange(nrec * 16) // 16]
    rank = torch.arange(nrec * 16) % 16
    r2 = LS.cover_repair(a, env0, rec, rank, k, 16, chunk_records=4)
    assert torch.equal(r2["env"], base["env"])


def test_infeasible_geometry_is_rejected():
    LS, _, KLayout = _m()
    L, idx, feat, bk = graph(C=64, H=8 * 128)  # C64 records: exactly 4 full row groups each
    w = weights(L)
    with pytest.raises(ValueError, match="infeasible"):
        LS.build("Cover-TopK", 8, L, bk, w=w, Q=feat.shape[0])  # 4 x 8 = 32 < 64
    a = torch.rand((3, 64), dtype=torch.float64)
    with pytest.raises(ValueError, match="infeasible"):
        LS.cover_repair(a, LS.topk_select(a, 16), torch.zeros(3, dtype=torch.int64), torch.arange(3), 16, 16)
    # m k == S exactly is feasible: a partition of the 64 inputs
    s = LS.build("Cover-TopK", 16, L, bk, w=w, Q=feat.shape[0])
    rc = LS.record_cover(s, L, bk, Q=feat.shape[0])
    full = rc["groups"] == 4
    assert bool((rc["mask"][full] == 64).all())


@pytest.mark.parametrize("k", [8, 16])
@pytest.mark.parametrize("H", [8 * 431 + 4, 8 * 175 + 4])  # n175 / n176 tails: 11 groups, the last partial
def test_structure_budget_identity_and_tails(k, H):
    LS, _, _ = _m()
    L, idx, feat, bk = graph(H=H)
    Q = feat.shape[0]
    scale = torch.ones(64)
    scale[:12] = 6.0  # TopK concentrates on 12 coordinates: many inputs uncovered before repair
    w = weights(L, scale=scale)
    tk = LS.build("TopK", k, L, bk, w=w, Q=Q)
    cv = LS.build("Cover-TopK", k, L, bk, w=w, Q=Q)
    gv = cv.nv > 0
    env = cv.env[gv]
    assert bool((env[:, 1:] > env[:, :-1]).all()) and int(env.min()) >= 0 and int(env.max()) < 64
    assert cv.enabled_slots() == k * L.hidden_dim == tk.enabled_slots()
    assert torch.equal(cv.q, tk.q) and torch.equal(cv.nv, tk.nv)
    rc = LS.record_cover(cv, L, bk, Q=Q)
    assert bool((rc["mask"] == 64).all())
    rt = LS.record_cover(tk, L, bk, Q=Q)
    rep = cv.stats["repair"]
    assert rep["swaps"] == int((64 - rt["mask"]).sum()) > 0 and rep["swaps_equal_initial_missing"]
    assert rep["records_tail"] == (int((rc["groups"] < 16).sum()))
    if H == 8 * 175 + 4:
        assert rep["tail_groups"] == [11]
    # element mask: exactly the valid slots, one shared support per group, nothing on padded rows
    em = cv.element_mask(L)
    assert int(em.sum()) == k * L.hidden_dim
    assert int(em.view(-1, 16, 64)[~gv].sum()) == 0
    valid = L.valid_mask("cpu").view(-1, 16)
    assert int(em.view(-1, 16, 64)[gv][~valid[gv]].sum()) == 0
    # energy: repair loses exactly the recorded loss relative to TopK
    a = LS.feature_importance(w, L)
    et, ec = LS.subset_score(a[gv], tk.env[gv]).sum(), LS.subset_score(a[gv], env).sum()
    assert float(et) >= float(ec)
    assert math.isclose(float(et - ec), rep["energy_loss_total"], rel_tol=1e-9)
    # TopK itself is unchanged by the new policy (fingerprint of the old construction)
    assert torch.equal(tk.env[gv], LS.topk_select(a[gv], k))


def test_zero_tied_and_already_covered_masks():
    LS, _, _ = _m()
    L, idx, feat, bk = graph()
    Q = feat.shape[0]
    # all-zero importances: TopK = 0..k-1 everywhere; the repair is deterministic and covers every record
    w0 = torch.zeros(L.weight_shape, dtype=torch.bfloat16)
    s0 = LS.build("Cover-TopK", 8, L, bk, w=w0, Q=Q)
    s0b = LS.build("Cover-TopK", 8, L, bk, w=w0, Q=Q)
    assert torch.equal(s0.env, s0b.env) and s0.fingerprint() == s0b.fingerprint()
    assert bool((LS.record_cover(s0, L, bk, Q=Q)["mask"] == 64).all())
    assert s0.stats["repair"]["energy_loss_total"] == 0.0
    # numerical coverage: enabled zero weights create no input dependence
    rc0 = LS.record_cover(s0, L, bk, w0, Q=Q)
    assert bool((rc0["numerical"] == 0).all()) and bool((rc0["mask"] == 64).all())
    # TopK already covering every record: the repair changes nothing
    w = torch.zeros(L.weight_shape)
    n16 = w.numel() // (16 * 64)
    wv = w.view(n16, 16, 64)
    rank = torch.arange(n16) % 16
    for t in range(8):  # group rank r favours the tile (r mod 8) of 8 coordinates
        sel = rank % 8 == t
        wv[sel, :, 8 * t:8 * t + 8] = 1.0 + torch.arange(8) / 16
    w = w.bfloat16()
    from sparse_pooling.experimental.kgen_layout import zero_padding_

    zero_padding_(L, w)
    tk = LS.build("TopK", 8, L, bk, w=w, Q=Q)
    cv = LS.build("Cover-TopK", 8, L, bk, w=w, Q=Q)
    full = LS.record_cover(tk, L, bk, Q=Q)["mask"] == 64
    assert bool(full.any())
    geo = LS.geometry(L, bk, Q=Q)
    covered_groups = full[geo["rec"]]
    uv = geo["uv"]
    assert torch.equal(cv.env[uv][covered_groups], tk.env[uv][covered_groups])


def test_objective_inequalities_against_exhaustive_and_flow_references():
    import cover_reference as CR

    LS, _, _ = _m()
    rng = np.random.default_rng(11)
    for trial in range(40):
        S = int(rng.choice([6, 7, 8]))
        k = int(rng.choice([2, 3]))
        lo = max(2, math.ceil(S / k))
        m = int(rng.integers(lo, lo + 2))
        a = rng.random((m, S)) ** 3
        if trial % 5 == 0:
            a[:, 0] = 0.0
        if trial % 7 == 0:
            a[:] = 0.5  # all tied
        at = torch.tensor(a)
        e0 = LS.topk_select(at, k)
        r = LS.cover_repair(at, e0, torch.zeros(m, dtype=torch.int64), torch.arange(m), k, 16, S=S)
        Xr = np.zeros((m, S), dtype=bool)
        Xr[np.repeat(np.arange(m), k), r["env"].numpy().reshape(-1)] = True
        assert CR.feasible(Xr, k)
        e_rep = CR.energy(a, Xr)
        X0 = np.zeros((m, S), dtype=bool)
        X0[np.repeat(np.arange(m), k), e0.numpy().reshape(-1)] = True
        e_top = CR.energy(a, X0)
        e_opt, Xo = CR.exhaustive(a, k)
        mo = CR.milp_opt(a, k)
        fo = CR.flow_opt(a, k)
        assert mo["certified"] and fo["feasible"]
        assert math.isclose(mo["energy"], e_opt, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(fo["energy"], e_opt, rel_tol=1e-9, abs_tol=1e-9)
        assert e_top >= e_opt - 1e-12 >= e_rep - 2e-12
        assert mo["lp_upper_bound"] >= e_opt - 1e-7


def test_bp_row_and_topk_fingerprints_are_unchanged_by_the_new_policy():
    """The construction of the old policies is untouched (same keyed streams, same selection)."""
    LS, _, _ = _m()
    L, idx, feat, bk = graph()
    Q = feat.shape[0]
    w = weights(L)
    for pol, k in (("BP-row", 8), ("BP-row", 16), ("TopK", 8), ("TopK", 16)):
        a = LS.build(pol, k, L, bk, w=w, graph_seed=1, layout_seed=0, Q=Q)
        b = LS.build(pol, k, L, bk, w=w, graph_seed=1, layout_seed=0, Q=Q)
        assert a.fingerprint() == b.fingerprint()
    assert "Cover-TopK" in LS.WEIGHTED and LS.RNG_SCHEMA == "granule.local_supports/1"


# ------------------------------------------------------------------ 2.1 shared-query uncertainty
def test_duplicated_outputs_add_no_monte_carlo_evidence():
    import joint_coverage as JC

    rng = np.random.default_rng(0)
    n, N = 200_000, 10 ** 9
    hit = (rng.random(n) < 0.01).astype(np.int64)
    one = JC.summary(hit, 1, N, boot=200)
    for O in (2, 38):
        dup = JC.summary(hit * O, O, N, boot=200)  # O outputs with IDENTICAL membership
        assert math.isclose(dup["mean"], one["mean"], rel_tol=1e-12)
        assert math.isclose(dup["se"], one["se"], rel_tol=1e-9)
        naive = JC.naive_pooled_binomial(hit * O, O)
        w_naive = naive["ci95"][1] - naive["ci95"][0]
        w_joint = dup["ci95"][1] - dup["ci95"][0]
        assert w_naive < w_joint / math.sqrt(O) * 1.2  # the old interval shrinks like 1/sqrt(O): invalid
    # independent outputs do shrink the interval
    ind = np.stack([(rng.random(n) < 0.01) for _ in range(38)]).sum(0)
    s_ind = JC.summary(ind, 38, N, boot=200)
    assert s_ind["se"] < one["se"] / 4
    # the cluster bootstrap matches the analytic SE
    assert math.isclose(s_ind["bootstrap"]["se"], s_ind["se"], rel_tol=0.15)


def test_paired_coverage_keeps_shared_queries_and_fpc():
    import joint_coverage as JC

    rng = np.random.default_rng(1)
    n, O = 100_000, 10
    ca = rng.binomial(O, 0.3, n)
    flip = rng.random(n) < 0.02
    cb = np.where(flip, np.clip(ca - 1, 0, O), ca)
    p = JC.paired(ca, cb, O, 10 ** 12)
    assert p["mean_difference"] > 0 and p["ci95"][0] > 0  # a small shared-query difference is resolved
    # treating the two masks as independent samples would be far wider
    sa, sb = JC.summary(ca, O, 10 ** 12, boot=0), JC.summary(cb, O, 10 ** 12, boot=0)
    assert p["se"] < 0.5 * math.hypot(sa["se"], sb["se"])
    # sampling the whole population leaves no sampling error (without replacement)
    full = JC.summary(ca, O, n, boot=0)
    assert full["se"] == 0.0 and full["fpc"] == 0.0
    with pytest.raises(ValueError):
        JC.paired(ca, cb[:-1], O, 10 ** 12)


# ------------------------------------------------------------------ 7 indexed execution
@requires_cuda_backend
@pytest.mark.parametrize("k", [16, 8])
@pytest.mark.parametrize("H", [8 * 431 + 4, 8 * 175 + 4])
def test_indexed_cover_topk_is_the_masked_function(k, H):
    import test_local_supports as TL

    Cn, KX, OT, L1, REF = TL._layers()
    LS, _, _ = _m()
    base = TL.base_layer(H, 256)
    L = base.layout
    s = LS.build("Cover-TopK", k, L, base.kg_buckets, w=base.weight.detach(), device="cuda", Q=base.sg_feat.shape[0])
    assert s.stats["repair"]["swaps"] >= 0
    em = s.element_mask(L)
    dm = TL.masked(base, em)
    lo = OT.lower_local(dm, s)
    lo.kcsr_validate()
    assert Cn.digest(lo)["sha"] == Cn.digest(dm)["sha"] and Cn.mask_digest(lo)["sha"] == Cn.mask_digest(dm)["sha"]
    assert Cn.mask_digest(lo)["enabled_slots"] == k * H
    g = torch.Generator(device="cuda").manual_seed(0)
    for B in (40, 64):
        x = torch.randn(B, L.input_dim, device="cuda", generator=g).bfloat16().float()
        dy = torch.randn(B, L.groups, device="cuda", generator=g)
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
        assert TL.maxrel(y, yr) < 1e-6 and TL.maxrel(xx.grad, dxr) < 2e-5
        assert TL.maxrel(dw, dwr) < 2e-5 and TL.maxrel(db, dbr) < 1e-6
        assert bool((dw[~Cn.dense_mask_rows(dm, gg, nn)] == 0).all())


@requires_cuda_backend
def test_indexed_cover_topk_optimizer_state_and_restore():
    import test_local_supports as TL
    from train_state import TrainState

    Cn, KX, OT, L1, REF = TL._layers()
    LS, _, _ = _m()
    base = TL.base_layer(8 * 175 + 4, 256)
    L = base.layout
    s = LS.build("Cover-TopK", 8, L, base.kg_buckets, w=base.weight.detach(), device="cuda", Q=base.sg_feat.shape[0])
    em = s.element_mask(L)
    dm = TL.masked(base, em)
    lo = OT.lower_local(dm, s)
    o1 = L1.L1AdamW([dm.weight, dm.bias], lr=1e-3, stochastic_rounding=False)
    L1.apply_mask_(o1, dm.weight, em)
    o2 = L1.L1AdamW([lo.weight, lo.bias], lr=1e-3, stochastic_rounding=False)
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
        assert TL.maxrel(b, a) < 2e-2, kind
    ident = Cn.identity(lo, o2)
    st = TrainState([lo.weight, lo.bias], o2).capture()
    fp0 = st.fingerprint()
    xx = x.clone().requires_grad_(True)
    lo(xx).backward(dy)
    o2.step()
    assert st.fingerprint() != fp0
    st.restore()
    assert st.fingerprint() == fp0 and Cn.identity(lo, o2) == ident
