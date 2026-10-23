"""Coverage analyzer of the K/C/S/R sweep (benchmarks/kcsr_coverage.py) against a set-based oracle on tiny graphs:
generated graphs, K = S, repeated records, empty supergroups, partial C tails (padded neurons excluded), physical
lowering (logical records counted once), and the exact-vs-sampled agreement of the tuple queries."""
import itertools
import math
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")))
pytestmark = requires_cuda_backend


def _mods():
    import kcsr_coverage as KCOV
    import kcsr_graph

    from sparse_pooling.experimental import kgen_graph as KG

    return KCOV, kcsr_graph, KG


def oracle(tab, t):
    recs = [tuple(sorted(r)) for r in tab["records"].tolist()]
    cov = set()
    for r in recs:
        cov.update(itertools.combinations(r, t))
    grp = set()
    for f in tab["feat"].tolist():
        grp.update(itertools.combinations(sorted(f), t))
    D = tab["meta"]["D"]
    return len(cov) / math.comb(D, t), len(grp) / math.comb(D, t), cov


def check(tab, orders=(2, 3, 4)):
    KCOV, _, _ = _mods()
    gi = KCOV.GraphIndex(tab, "cuda")
    ex = KCOV.exact_coverage(gi, tuple(t for t in orders if t <= 3))
    D = tab["meta"]["D"]
    for t in orders:
        f, g, cov = oracle(tab, t)
        if t <= 3:
            assert ex[t]["fraction"] == pytest.approx(f, abs=0) and ex[t]["group_fraction"] == pytest.approx(g, abs=0)
        total = math.comb(D, t)
        s = KCOV.sampled_coverage(gi, t, total, total, 0)  # the whole universe: the query must equal the oracle
        assert s["hits"] == len(cov) and s["fraction"] == pytest.approx(f, abs=0)
        assert s["group_fraction"] == pytest.approx(g, abs=0)


def gen_table(D, H, G, C, K, S, R, seed=0, quantum=1):
    _, _, KG = _mods()
    from sparse_pooling.supergroups import ConnectivityPlan

    counts = torch.full((G,), H // G, dtype=torch.int32, device="cuda")
    counts[: H % G] += 1
    NB = -(-int(counts.max()) // (C * quantum)) * quantum
    plan = ConnectivityPlan(S, R, seed=seed)
    idx, feat, ubk, _ = KG.generate(D, counts, C, K, NB, quantum, plan, "cuda")
    bk = KG.unit_to_block_buckets(ubk, counts, C, quantum)
    return KG.record_table(D, H, counts, C, K, idx, feat, bk)


@pytest.mark.parametrize("D,H,G,C,K,S,R,quantum", [(32, 300, 4, 8, 8, 16, 3, 1), (24, 101, 3, 8, 4, 8, 2, 1),
                                                    (32, 999, 4, 16, 8, 32, 1, 1), (32, 250, 2, 8, 8, 16, 2, 4),
                                                    (16, 70, 2, 8, 2, 8, 4, 2)])
def test_analyzer_matches_oracle_on_generated_graphs(D, H, G, C, K, S, R, quantum):
    check(gen_table(D, H, G, C, K, S, R, quantum=quantum))


def test_k_equals_s_records_are_the_sets():
    tab = gen_table(32, 400, 4, 8, 8, 8, 3)
    KCOV, _, _ = _mods()
    gi = KCOV.GraphIndex(tab, "cuda")
    assert int(gi.n_distinct.max()) == 1  # every record of a supergroup is its whole set: one distinct record
    ex = KCOV.exact_coverage(gi, (2, 3))
    for t in (2, 3):  # every occupied set covers all its tuples: record coverage == realized group coverage
        assert ex[t]["covered"] == ex[t]["group_covered"]
    check(tab)


def test_repeated_records_and_empty_supergroups():
    tab = gen_table(32, 64, 2, 8, 8, 16, 2)
    # keep only the records of supergroup 0, three copies each: other supergroups are empty
    keep = tab["rec_q"] == 0
    rep = {k: (torch.cat([tab[k][keep]] * 3) if k not in ("feat", "meta") else tab[k]) for k in tab}
    KCOV, _, _ = _mods()
    gi = KCOV.GraphIndex(rep, "cuda")
    assert int(gi.n_distinct.sum()) == int(keep.sum()) and int((gi.n_distinct == 0).sum()) == gi.Q - 1
    check(rep)


def test_partial_c_tails_and_padded_neurons_excluded():
    """Native layer with partial C blocks and padded blocks inside partially valid units: only valid logical
    records enter the table (padded neurons never count)."""
    _, kcsr_graph, _ = _mods()
    from sparse_pooling import SparsePoolingLayer
    from sparse_pooling.supergroups import ConnectivityPlan

    lay = SparsePoolingLayer(64, 501, 4, shared_neurons=8, pooling="mean", device="cuda",
                             connectivity=ConnectivityPlan(16, 2, seed=3))
    tab = kcsr_graph.native_table(lay)
    L = lay.layout
    counts = L.counts_tensor("cpu")
    assert int(tab["rec_n"].sum()) == 501 and len(tab["records"]) == int(((counts + 7) // 8).sum())
    assert bool((tab["rec_n"] <= 8).all()) and int((tab["rec_n"] < 8).sum()) == int((counts % 8 != 0).sum())
    check(tab)


def test_physical_lowering_counts_logical_records_once():
    _, kcsr_graph, KG = _mods()
    from sparse_pooling.experimental import lowering as LW
    from sparse_pooling.supergroups import ConnectivityPlan

    L, idx, w, b, bk, feat, spec, _ = LW.build_logical(128, 5000, 4, 256, ConnectivityPlan(32, 2, seed=1), "cuda", seed=1)
    low = LW.lower(L, idx, w, b, bk, feat, spec, device="cuda")
    tab = kcsr_graph.native_table(low, logical_C=256)
    ref = KG.record_table(128, 5000, L.counts_tensor("cuda"), 256, 8, idx, feat, bk)
    assert kcsr_graph.fingerprint(tab) == kcsr_graph.fingerprint(ref)
    assert len(tab["records"]) == int(L.valid_blocks().sum())  # not 4x (the physical copies)
    check(tab, (2, 3))


def test_rank_unrank_and_distinct_samples():
    KCOV, _, _ = _mods()
    for D, t in ((1024, 4), (1024, 3), (33, 2)):
        ct = KCOV.comb_table(D, t, "cuda")
        tup = KCOV.sample_tuples(D, t, 20000, device="cuda")
        assert torch.equal(tup, tup.sort(1).values) and bool((tup[:, 1:] > tup[:, :-1]).all())
        r = KCOV.rank(tup, ct)
        assert torch.unique(r).numel() == r.numel() and int(r.max()) < math.comb(D, t)
        assert torch.equal(KCOV.unrank(r, D, t, ct), tup)
    a = KCOV.sample_tuples(1024, 4, 1000)
    assert torch.equal(a, KCOV.sample_tuples(1024, 4, 1000))  # fixed sequence


def test_sampled_triples_agree_with_exact_on_a_larger_graph():
    KCOV, _, _ = _mods()
    tab = gen_table(1024, 200_000, 64, 8, 8, 32, 8)
    gi = KCOV.GraphIndex(tab, "cuda")
    ex = KCOV.exact_coverage(gi, (3,))[3]["fraction"]
    s = KCOV.sampled_coverage(gi, 3, 2_000_000, 2_000_000, 0)
    assert s["ci95"][0] <= ex <= s["ci95"][1]


def test_refined_prediction_formula_matches_supplied_model_without_holes():
    """With quantum 1 and no invalid records the refined (per-pass) prediction equals the supplied model's
    finite-record formula for the same record counts (the supplied script's predict())."""
    import importlib.util

    import kcsr_cases as KC

    spec = importlib.util.spec_from_file_location("rep", KC.REPRODUCE)
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    KCOV, _, KG = _mods()
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan

    for K, C, S, R, H in ((16, 32, 64, 4, 1_000_003), (8, 8, 32, 8, 500_000), (32, 128, 128, 2, 2_000_000)):
        L = KLayout(1024, H, 1024, K, C)
        counts = L.counts_tensor("cpu")
        nvu = KG.valid_units(counts, C, 1)
        _, ubk = KG.bucket_table(1024, nvu, ConnectivityPlan(S, R))
        v, slots = KCOV.local_pass_table(1024, counts, C, K, S, L.blocks_per_group, 1, ubk)
        assert int((slots - v).sum()) == 0
        for t in (2, 3, 4):
            got = KCOV.refined_prediction(1024, K, S, R, v, t)
            want = rep.predict(1024, K, C, S, R, H, t)["model"]
            assert got == pytest.approx(want, rel=1e-9, abs=1e-15)
