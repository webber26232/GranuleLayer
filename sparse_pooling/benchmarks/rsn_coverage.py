"""Coverage of the exact supports of the random-support study (results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md
sections 3.3 and 7), on the timed graph / checkpoint: the analyses of wps_coverage.coverage_of (exact layer-wide
pairs / triples, fixed seeded quadruple and triple samples with Clopper-Pearson intervals and zero-hit bounds,
exact per-output features for all outputs, pairs / triples / quadruples with neuron multiplicity on the parent's
fixed output selection, enabled vs numerical supports) for any rsn_cases spec, plus:

* predictions: the supplied ideal-quota expectation and the expectation refined with the ACTUAL m_q (valid row
  groups per set), b_q (records per set), groups per record and fixed-partition tile use
  (experimental/local_supports.set_hit / layer_expectation; the independent-round model) -- kept separate from
  the measured values; the per-output union bound min(1, n_rec C(16, t) / C(1024, t)) with the actual
  maximum record count per output; the one-tile union bound Q 4 C(16, t) / C(1024, t) (valid only for one common
  partition per set);
* support sharing at the original-record / set / output scopes: distinct row-group supports per scope, valid
  neurons per distinct support (an identical support repeated in two row groups of one scope adds neurons, not
  tuples);
* support-record multiplicity: on the common sampled triples / quadruples, the number of distinct ROW GROUPS
  (independent support records) whose support contains the tuple, beside the neuron multiplicity.

Supports of different neurons are never united.  Analysis buffers / time stay outside every timing run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import common  # noqa: F401
import k64_coverage as KC64
import rsn_cases as RC
import wps_cases as WC
import wps_coverage as WV
from kcsr_coverage import clopper_pearson, sample_tuples


@torch.no_grad()
def sharing(L, em: torch.Tensor, bq: torch.Tensor) -> dict:
    """Distinct row-group supports per original record / set / output and valid neurons per distinct support."""
    from sparse_pooling.experimental.width_masks import layout_counts

    dev = em.device
    nv = layout_counts(L, dev)["nv"]
    n16 = nv.numel()
    gpb = L.block // 16
    gm = em.view(n16, 16, 64)[:, 0, :].to(torch.int64)  # neuron 0 of a valid group is valid; groups share masks
    key = (gm << torch.arange(64, device=dev)).sum(1)
    gv = nv > 0
    u = torch.nonzero(gv).squeeze(1)
    k = key[u]
    scopes = dict(record=u // gpb, set=bq[u // gpb], output=u // (L.blocks_per_group * gpb))
    out = {}
    for name, sid in scopes.items():
        pair = torch.stack([sid, k], 1)
        uniq, inv = torch.unique(pair, dim=0, return_inverse=True)
        per_scope = torch.bincount(uniq[:, 0] - int(sid.min()))
        per_scope = per_scope[per_scope > 0]
        neurons = torch.zeros(len(uniq), dtype=torch.int64, device=dev).index_add_(0, inv, nv[u])
        groups = torch.bincount(inv)
        out[name] = dict(scopes=int(per_scope.numel()), distinct_supports=int(len(uniq)),
                         distinct_per_scope=KC64.dist(per_scope), groups_per_distinct_support=KC64.dist(groups),
                         neurons_per_distinct_support=KC64.dist(neurons))
    out["layer"] = dict(distinct_supports=int(torch.unique(k).numel()), valid_groups=int(len(u)))
    return out


@torch.no_grad()
def group_multiplicity(L, feat, bk, em, quad_n):
    """Distinct row groups (independent support records) containing each common sampled triple / quadruple."""
    from sparse_pooling.experimental.width_masks import layout_counts

    nv = layout_counts(L, em.device)["nv"]
    gm = em.view(nv.numel(), 16, 64).clone()
    gm[:, 1:] = 0  # one representative neuron per row group (neuron 0 is valid in every valid group)
    sp = KC64.supports_from_dense(L, feat, bk, gm.view(L.weight_shape))
    out = {}
    for name, t, n in (("quad", 4, quad_n), ("triple_sampled_check", 3, KC64.N_TRIPLE_CHECK)):
        seq = sample_tuples(WC.D, t, n, KC64.SAMPLE_SEED, sp.q.device)
        cov, cnt = WV.query_multiplicity(sp, seq)
        out[name] = dict(order=t, sample_size=n, covered=int(cov.sum()), group_multiplicity=WV.categories(cnt))
    del sp
    torch.cuda.empty_cache()
    return out


def predictions(sup, L, bk, spec) -> dict:
    from sparse_pooling.experimental import local_supports as LS

    if sup is None or not hasattr(sup, "env"):
        return None
    ri = LS.refined_inputs(sup, L, bk)
    out = dict(model="independent rounds: 1 - prod_r (1 - p_g(t) mean_{q in r} h_q); p_g = C(63, t-1) / C(1023, t-1)",
               refined=dict(), ideal=dict())
    groups = int((sup.nv > 0).sum())
    records = int(ri["rec_groups"].numel())
    for t in (2, 3, 4):
        h = LS.set_hit(sup.policy, t, sup.k, ri["m_q"], ri["b_q"], ri["rec_groups"], ri["rec_set"], ri["used_tiles"])
        out["refined"][f"t{t}"] = LS.layer_expectation(h, t, WC.D // WC.S)
        if sup.policy != "FP-mag":
            out["ideal"][f"t{t}"] = LS.ideal_prediction(sup.policy, t, sup.k, groups, ri["Q"], records)
    out["inputs"] = dict(m_q=KC64.dist(ri["m_q"]), b_q=KC64.dist(ri["b_q"]), groups_per_record=KC64.dist(ri["rec_groups"]),
                         used_tiles_per_set=KC64.dist(ri["used_tiles"]) if sup.policy.startswith("FP") else None)
    if sup.policy.startswith("FP"):
        out["note"] = ("fixed partition: (used tiles / nkt) a_t per set is an APPROXIMATION -- the tiles are ranges of the "
                       "sorted set, so the same nearby-id tuples share a tile in every set (rank correlation)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", default="dense")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    import kcsr_graph

    dev = torch.device("cuda")
    t_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    C = RC.C
    L = KLayout(WC.D, WC.H, WC.G, 64, C)
    counts = L.counts_tensor(dev)
    plan = ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed)
    idx, feat, ubk, _ = KG.generate(WC.D, counts, C, 64, L.blocks_per_group, 1, plan, dev)
    bk = KG.unit_to_block_buckets(ubk, counts, C, 1)
    tab = KG.record_table(WC.D, WC.H, counts, C, 64, idx, feat, bk)
    del idx
    res = dict(schema="granule.rsn_coverage/1", C=C, S=WC.S, R=WC.R, H=WC.H, graph_seed=a.graph_seed,
               support=a.support, graph_fingerprint=kcsr_graph.fingerprint(tab))
    del tab
    tri = sample_tuples(WC.D, 3, KC64.N_OUT_TRIPLE, KC64.SAMPLE_SEED + 7, dev)
    quad = sample_tuples(WC.D, 4, KC64.N_OUT_QUAD, KC64.SAMPLE_SEED + 8, dev)
    sp0 = KC64.supports_from_dense(L, feat, bk, None)
    feats0 = KC64.per_output_features(sp0)
    outs, why = WV.fixed_outputs(feats0, counts, WC.G)
    del sp0
    torch.cuda.empty_cache()
    res["output_selection"] = dict(why, fixed=outs, rule="the parent's fixed selection (wps_coverage.py)")
    n_rec_out = int(math.ceil(int(counts.max()) / 16))
    res["bounds"] = dict(per_output_union={f"t{t}": LS.per_output_union_bound(n_rec_out, t, 16) for t in (2, 3, 4)},
                         per_output_records_max=n_rec_out,
                         one_tile_union={f"t{t}": min(1.0, (WC.R * WC.D // WC.S) * 4 * math.comb(16, t) / math.comb(WC.D, t))
                                         for t in (2, 3, 4)},
                         note="per-output union bound applies to every K16 row-group support (<= n_rec independent "
                              "16-input records per output); the one-tile bound only to ONE common partition per set")
    if a.support == "dense":
        res["kind"] = "dense potential (all 64 inputs of every valid neuron)"
        res["enabled"] = WV.coverage_of(L, feat, bk, None, outs, tri, quad, KC64.N_QUAD)
    else:
        rec, w, b, _ = SP.load_checkpoint(WC.ckpt(C, a.graph_seed))
        if rec["spec"]["seed"] != a.graph_seed or rec["C"] != C:
            raise SystemExit("checkpoint graph does not match")
        res["checkpoint"] = dict(path=WC.ckpt(C, a.graph_seed), weight_sha256=rec["weight_sha256"])
        holder = SimpleNamespace(weight=w, layout=L, kg_buckets=bk)
        t0 = time.perf_counter()
        em, sup, srec = RC.support(a.support, holder, a.graph_seed)
        res["support_seconds"] = time.perf_counter() - t0
        res["support_record"] = srec
        res["kind"] = a.support
        res["width"] = WV.width_stats(L, em, w)
        res["anchors"] = WV.anchor_check(L, sup, w) if hasattr(sup, "anchor") else None
        dens = res["width"]["enabled_slots"] / WC.W0
        qn = KC64.N_QUAD_SPARSE if dens <= 0.10 else KC64.N_QUAD
        res["quad_sample_rule"] = f"{qn:,} (50M when the enabled density <= 0.10, else 10M; predeclared)"
        res["predictions"] = predictions(sup, L, bk, a.support)
        bq = LS.block_sets(L, bk, dev)
        res["sharing"] = sharing(L, em, bq)
        res["support_record_multiplicity"] = group_multiplicity(L, feat, bk, em, qn)
        res["enabled"] = WV.coverage_of(L, feat, bk, em, outs, tri, quad, qn)
        nzm = (em.bool() & (w != 0)).to(torch.uint8)
        if torch.equal(nzm, em):
            res["nonzero"] = dict(identical_to_enabled=True, zero_enabled_weights=0)
        else:
            res["nonzero"] = WV.coverage_of(L, feat, bk, nzm, outs, tri, quad, qn)
            res["nonzero"]["zero_enabled_weights"] = int((em.bool() & (w == 0)).sum())
        del w, b, em, nzm
    res["analysis_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    res["analysis_seconds"] = time.perf_counter() - t_all
    res["provenance"] = provenance(Path(common.__file__).resolve().parent.parent)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    e = res["enabled"]
    print(f"{res['kind']}: pair {100 * e['pair']['fraction']:.4f}% triple {100 * e['triple']['fraction']:.4f}% quad "
          f"{100 * e['quad']['fraction']:.5f}% | per-output triple median "
          f"{100 * e['per_output_summary']['triple']['median']:.4f}% | {res['analysis_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
