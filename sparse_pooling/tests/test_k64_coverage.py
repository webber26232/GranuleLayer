"""Coverage oracle of the K64 study (benchmarks/k64_coverage.py) on tiny generated / masked graphs: per-neuron
supports (never the union of the masks of different neurons sharing a record), occupied vs empty sets, exact
pairs / triples and an exhaustive quadruple query against brute-force enumeration, per-output features and
pairs, fixed-size sampling with zero-hit upper bounds."""
import itertools
import math
import os
import sys

import pytest
import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
pytestmark = requires_cuda_backend


def _cov():
    import k64_coverage as KV

    return KV


def tiny(D=128, S=64, R=3, n_neurons=240, G=4, density=0.15, seed=0, empty_sets=()):
    """Random S = K = 64 graph: R rounds of D / S sets; neurons assigned to sets (some sets left empty);
    random per-neuron supports."""
    KV = _cov()
    g = torch.Generator().manual_seed(seed)
    Q = R * D // S
    feat = torch.cat([torch.randperm(D, generator=g).view(D // S, S) for _ in range(R)]).sort(1).values.cuda()
    occupied = [q for q in range(Q) if q not in empty_sets]
    q = torch.tensor([occupied[i % len(occupied)] for i in range(n_neurons)]).cuda()
    out = (torch.arange(n_neurons) % G).cuda()
    bits = torch.zeros(n_neurons, dtype=torch.int64)
    for i in range(n_neurons):
        m = torch.rand(64, generator=g) < density
        for k in torch.nonzero(m).squeeze(1).tolist():
            bits[i] |= (1 << k) if k < 63 else -(1 << 63)
    return KV.Supports(feat, q, out, bits.cuda(), D, G), feat.cpu(), q.cpu(), out.cpu(), bits


def supports_sets(feat, q, bits):
    out = []
    for i in range(q.numel()):
        b = int(bits[i])
        ks = [k for k in range(64) if (b >> k) & 1]
        out.append(sorted(int(feat[int(q[i]), k]) for k in ks))
    return out


def oracle(sets, t):
    cov = set()
    for s in sets:
        cov.update(itertools.combinations(s, t))
    return cov


def test_exact_pairs_triples_and_exhaustive_quads_match_brute_force():
    KV = _cov()
    sp, feat, q, out, bits = tiny(empty_sets=(1,))
    sets = supports_sets(feat, q, bits)
    ex = KV.exact_support_coverage(sp, (2, 3))
    D = 128
    for t in (2, 3):
        o = oracle(sets, t)
        assert ex[t]["covered"] == len(o) and ex[t]["total"] == math.comb(D, t)
    # group coverage: all sets vs occupied sets (set 1 empty)
    grp_all = set()
    grp_occ = set()
    for qq in range(feat.shape[0]):
        c = set(itertools.combinations(feat[qq].tolist(), 2))
        grp_all |= c
        if qq != 1:
            grp_occ |= c
    assert ex[2]["group_covered"] == len(grp_all) and ex[2]["group_occupied_covered"] == len(grp_occ)
    # exhaustive quadruple query
    from kcsr_coverage import comb_table, unrank

    ct = comb_table(D, 4, "cpu")
    allq = unrank(torch.arange(int(ct[D, 4])), D, 4, ct).cuda()
    hit, grp, occ = KV.query_supports(sp, allq)
    o4 = oracle(sets, 4)
    assert int(hit.sum()) == len(o4)
    got = {tuple(r) for r in allq[hit].tolist()}
    assert got == o4


def test_union_of_different_neurons_never_counts():
    KV = _cov()
    D, S = 128, 64
    feat = torch.arange(D).view(2, 64).cuda()  # R = 1
    q = torch.tensor([0, 0]).cuda()
    out = torch.tensor([0, 0]).cuda()
    bits = torch.tensor([0b0011, 0b1100], dtype=torch.int64).cuda()  # neuron 0: {0, 1}, neuron 1: {2, 3}
    sp = KV.Supports(feat, q, out, bits, D, 1)
    ex = KV.exact_support_coverage(sp, (2, 3))
    assert ex[2]["covered"] == 2 and ex[3]["covered"] == 0  # (0,1), (2,3) only; never (0,2) etc.
    tup = torch.tensor([[0, 2, 64, 65], [0, 1, 2, 3]]).cuda().sort(1).values
    hit, _, _ = KV.query_supports(sp, tup)
    assert not bool(hit.any())
    rows = KV.output_coverage(sp, [0], torch.tensor([[0, 1, 2]]).cuda(), torch.tensor([[0, 1, 2, 3]]).cuda())
    assert rows[0]["pair_covered"] == 2 and rows[0]["triple_hits"] == 0 and rows[0]["quad_hits"] == 0


def test_per_output_features_and_pairs_match_brute_force():
    KV = _cov()
    sp, feat, q, out, bits = tiny(seed=3, density=0.2)
    sets = supports_sets(feat, q, bits)
    feats = KV.per_output_features(sp).cpu()
    tri = torch.tensor(sorted(oracle(sets, 3))[:50] + [(0, 1, 2)]).cuda()
    quad = torch.tensor(sorted(oracle(sets, 4))[:50] + [(0, 1, 2, 3)]).cuda()
    rows = KV.output_coverage(sp, list(range(4)), tri, quad)
    for g in range(4):
        sg = [s for s, o in zip(sets, out.tolist()) if o == g]
        assert int(feats[g]) == len(set().union(*map(set, sg)))
        assert rows[g]["pair_covered"] == len(oracle(sg, 2))
        o3, o4 = oracle(sg, 3), oracle(sg, 4)
        assert rows[g]["triple_hits"] == sum(tuple(r) in o3 for r in tri.tolist())
        assert rows[g]["quad_hits"] == sum(tuple(r) in o4 for r in quad.tolist())


def test_zero_hit_samples_report_upper_bounds_and_empty_supports():
    KV = _cov()
    sp, *_ = tiny(density=0.0)
    ex = KV.exact_support_coverage(sp, (2, 3))
    assert ex[2]["covered"] == 0 and ex[3]["covered"] == 0
    r = KV.sampled(lambda tp: KV.query_supports(sp, tp), 128, 4, 20_000)
    assert r["hits"] == 0 and r["fixed_size"] and 0 < r["upper_bound_if_zero"] < 1e-3


def test_dense_supports_equal_records_and_iid_reference():
    KV = _cov()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan

    L = KLayout(256, 3001, 4, 64, 128)
    counts = L.counts_tensor("cuda")
    idx, feat, ubk, _ = KG.generate(256, counts, 128, 64, L.blocks_per_group, 1, ConnectivityPlan(64, 2, seed=1), "cuda")
    bk = KG.unit_to_block_buckets(ubk, counts, 128, 1)
    sp = KV.supports_from_dense(L, feat, bk, None)
    assert sp.q.numel() == 3001 and bool((sp.bits == -1).all())
    ex = KV.exact_support_coverage(sp, (2,))
    # dense S = K: covered = group coverage of the occupied sets
    assert ex[2]["covered"] == ex[2]["group_occupied_covered"]
    m = torch.zeros(L.weight_shape, dtype=torch.uint8, device="cuda")
    m[..., :3] = 1
    sp2 = KV.supports_from_dense(L, feat, bk, m * L.valid_mask("cuda")[..., None].to(torch.uint8))
    assert bool((sp2.bits == 0b111).all())
    assert abs(KV.iid_reference(64, 64, 0.5, 10, 2) - (1 - 0.75 ** 10)) < 1e-12
