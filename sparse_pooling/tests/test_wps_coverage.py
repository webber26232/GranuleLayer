"""Coverage additions of the width-preserving study (benchmarks/wps_coverage.py) on tiny graphs against brute
force: byte-lookup popcount, layer-wide tuple MULTIPLICITY (number of valid neurons whose own support contains
the tuple; supports of different neurons never united), per-output multiplicity, multiplicity categories,
width statistics and the structural anchor check (backbone / anchor supports, tuple order > fan-in)."""
import itertools
import os
import sys

import torch

from conftest import requires_cuda_backend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
pytestmark = requires_cuda_backend


def _m():
    import k64_coverage as KV
    import wps_coverage as WV

    return KV, WV


def test_popcount_and_categories():
    _, WV = _m()
    x = torch.tensor([0, 1, 3, -1, 1 << 40, -(1 << 63), 0x0F0F0F0F0F0F0F0F], dtype=torch.int64, device="cuda")
    want = [0, 1, 2, 64, 1, 1, 32]
    assert WV.popcount64(x).tolist() == want
    c = WV.categories(torch.tensor([0, 0, 1, 2, 15, 16, 63, 64, 500], device="cuda"))
    assert c["categories"] == {"0": 2, "1": 1, "2-15": 2, "16-63": 2, "64+": 2}
    assert c["covered_quantiles"]["covered"] == 7 and c["covered_quantiles"]["max"] == 500


def test_layer_and_output_multiplicity_match_brute_force():
    import test_k64_coverage as T  # tiny graphs / oracles of the K64 study

    KV, WV = _m()
    sp, feat, q, out, bits = T.tiny(D=128, R=3, n_neurons=300, G=3, density=0.2, seed=4)
    sets = T.supports_sets(feat, q, bits)
    for t in (2, 3, 4):
        tup = torch.tensor(list(itertools.combinations(range(0, 128, 9), t)), device="cuda")
        cov, cnt = WV.query_multiplicity(sp, tup)
        for i, tp in enumerate(tup.tolist()):
            n = sum(1 for s in sets if set(tp) <= set(s))  # neurons whose OWN support contains the tuple
            assert int(cnt[i]) == n and bool(cov[i]) == (n > 0)
    tri = torch.tensor(list(itertools.combinations(range(0, 128, 11), 3)), device="cuda")
    quad = torch.tensor(list(itertools.combinations(range(0, 128, 17), 4)), device="cuda")
    rows = WV.output_rows(sp, [0, 2], tri, quad)
    for r in rows:
        g = r["output"]
        so = [s for s, o in zip(sets, out.tolist()) if o == g]
        for name, tup in (("triple", tri), ("quad", quad)):
            cnts = torch.tensor([sum(1 for s in so if set(tp) <= set(s)) for tp in tup.tolist()])
            assert r[f"{name}_hits"] == int((cnts > 0).sum())
            assert r[f"{name}_multiplicity"]["categories"] == WV.categories(cnts.cuda())["categories"]


def test_width_stats_and_anchor_check_on_a_policy_support():
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout, init_parameters_by_neuron

    _, WV = _m()
    L = KLayout(256, 3003, 8, 64, 128)
    w, _ = init_parameters_by_neuron(L, "cuda", torch.bfloat16, seed=3)
    ms = WM.build("H2A", 0.2, w, L)
    em = ms.element_mask(L)
    with torch.no_grad():
        w2 = w * em
        w2.view(-1, 64)[0, :] = 0  # one anchor-group neuron loses all its weights (numerically)
    st = WV.width_stats(L, em, w2)
    valid = L.valid_mask("cuda").reshape(-1)
    deg = em.reshape(-1, 64).sum(1)[valid]
    assert st["mask_degree_hist"] == torch.bincount(deg, minlength=65).tolist()
    assert st["bias_only_mask"] == 0 and st["bias_only_numeric"] == 1  # (neuron 0: every weight zeroed)
    a = WV.anchor_check(L, ms, w2)
    assert a["all_records_have_full_mask_anchor"]
    if bool(ms.anchor[0]):
        assert a["zero_anchor_weights"] >= 64 and not a["all_records_have_numerically_full_anchor"]
