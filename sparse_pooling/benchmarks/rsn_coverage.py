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


GLOBAL_SUPPORT_SCHEMA = "granule.global_supports/1"


@torch.no_grad()
def global_supports(L, em: torch.Tensor, bq: torch.Tensor, feat: torch.Tensor, chunk: int = 1 << 16) -> dict:
    """Exact GLOBAL input set of every valid row group: its enabled local coordinates mapped through the group's
    actual feat[q] and stored as a D-bit bitmap (D / 64 int64 words; distinct global ids, so adding the bit values is
    an exact OR, the sign bit included).  The 16 neurons of a row group must share one enabled support (true for every
    support of these studies; checked, a violation raises).  Returns u [n] (ascending), q [n], words [n, D / 64],
    neurons nv[u], support size."""
    from sparse_pooling.experimental.width_masks import layout_counts

    dev = em.device
    nv = layout_counts(L, dev)["nv"]
    n16 = nv.numel()
    gpb = L.block // 16
    D = L.input_dim
    nw = (D + 63) // 64
    valid = L.valid_mask(dev).reshape(n16, 16)
    emv = em.view(n16, 16, 64)
    u = torch.nonzero(nv > 0).squeeze(1)
    q = bq[u // gpb]
    fe = feat.to(dev).to(torch.int64)
    words = torch.zeros((len(u), nw), dtype=torch.int64, device=dev)
    size = torch.zeros(len(u), dtype=torch.int64, device=dev)
    for s0 in range(0, len(u), chunk):
        uu = u[s0:s0 + chunk]
        m = emv[uu].bool()  # [c, 16, 64]
        g0 = m[:, 0]
        if not bool(((m == g0[:, None]) | ~valid[uu][:, :, None]).all()):
            raise AssertionError("the valid neurons of a row group do not share one enabled support")
        ids = fe[q[s0:s0 + chunk]]  # [c, 64] global ids of the local coordinates (sorted)
        val = torch.where(g0, torch.ones_like(ids) << (ids & 63), torch.zeros_like(ids))
        words[s0:s0 + chunk].scatter_add_(1, ids >> 6, val)
        size[s0:s0 + chunk] = g0.sum(1)
    return dict(u=u, q=q, words=words, nv=nv[u], size=size, gpb=gpb)


def _scope_stats(sid: torch.Tensor, key: torch.Tensor, nv: torch.Tensor) -> dict:
    """Distinct (scope, key) supports: count, per scope, row groups and valid neurons per distinct support."""
    rows = torch.cat([sid[:, None], key], 1)
    uniq, inv = torch.unique(rows, dim=0, return_inverse=True)
    per_scope = torch.bincount(uniq[:, 0] - int(sid.min()))
    per_scope = per_scope[per_scope > 0]
    neurons = torch.zeros(len(uniq), dtype=torch.int64, device=sid.device).index_add_(0, inv, nv)
    groups = torch.bincount(inv)
    return dict(scopes=int(per_scope.numel()), distinct_supports=int(len(uniq)),
                distinct_per_scope=KC64.dist(per_scope), groups_per_distinct_support=KC64.dist(groups),
                neurons_per_distinct_support=KC64.dist(neurons), repeated_supports=int((groups > 1).sum()))


@torch.no_grad()
def sharing(L, em: torch.Tensor, bq: torch.Tensor, feat: torch.Tensor) -> dict:
    """Support sharing at the original-record / set / output / layer scopes with GLOBAL identities (exact global-input
    bitmaps, ``global_supports``): unique input supports per scope, supporting ROW GROUPS and supporting NEURONS per
    distinct support (an anchor's 16 neurons are one support with multiplicity 16, never 16 supports; different
    neurons are never united into a tuple).

    Separate, labelled alternatives (NOT global identities): ``labelled_local`` = distinct (scope, q, 64-bit local
    mask) -- it distinguishes labelled input sets but does not merge identical global supports that occur in two
    sets; ``legacy_local_key`` = distinct (scope, local mask), the parent's metric (rsn_coverage at 03e189f), wrong at
    the output / layer scopes where two sets with equal local masks select different global inputs."""
    gs = global_supports(L, em, bq, feat)
    u, q, words, nv = gs["u"], gs["q"], gs["words"], gs["nv"]
    gpb = gs["gpb"]
    dev = em.device
    n16 = L.groups * L.blocks_per_group * gpb
    gm = em.view(n16, 16, 64)[u, 0, :].to(torch.int64)
    lkey = (gm << torch.arange(64, device=dev)).sum(1)[:, None]
    scopes = dict(record=u // gpb, set=q, output=u // (L.blocks_per_group * gpb), layer=torch.zeros_like(u))
    out = dict(schema=GLOBAL_SUPPORT_SCHEMA,
               identity="exact sorted global-input set (D-bit bitmap of feat[q][enabled local coordinates])",
               valid_groups=int(len(u)), valid_neurons=int(nv.sum()),
               support_size=KC64.dist(gs["size"]))
    for name, sid in scopes.items():
        out[name] = _scope_stats(sid, words, nv)
        out[name]["labelled_local"] = _scope_stats(sid, torch.cat([q[:, None], lkey], 1), nv)["distinct_supports"]
        out[name]["legacy_local_key"] = _scope_stats(sid, lkey, nv)["distinct_supports"]
    return out


@torch.no_grad()
def distinct_support_representatives(L, em: torch.Tensor, bq: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """[G, NB, C, 64] uint8: the enabled support of ONE representative neuron (neuron 0, first row group in
    ascending u) per distinct GLOBAL support; everything else zero."""
    gs = global_supports(L, em, bq, feat)
    _, inv = torch.unique(gs["words"], dim=0, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), len(inv), dtype=torch.int64, device=em.device)
    first.scatter_reduce_(0, inv, torch.arange(len(inv), device=em.device), reduce="amin")
    keep = gs["u"][first]
    n16 = gs["u"].new_tensor(L.groups * L.blocks_per_group * gs["gpb"])
    rep = torch.zeros((int(n16), 16, 64), dtype=torch.uint8, device=em.device)
    rep[keep, 0] = em.view(int(n16), 16, 64)[keep, 0]
    return rep.view(L.weight_shape)


@torch.no_grad()
def group_multiplicity(L, feat, bk, em, quad_n, bq=None):
    """On the common sampled triples / quadruples: the number of distinct ROW GROUPS (support records) and of
    distinct GLOBAL supports containing each tuple (a support repeated in two groups counts once in the latter)."""
    from sparse_pooling.experimental.width_masks import layout_counts

    nv = layout_counts(L, em.device)["nv"]
    gm = em.view(nv.numel(), 16, 64).clone()
    gm[:, 1:] = 0  # one representative neuron per row group (neuron 0 is valid in every valid group)
    out = {}
    reps = dict(group_multiplicity=gm.view(L.weight_shape))
    if bq is not None:
        reps["distinct_support_multiplicity"] = distinct_support_representatives(L, em, bq, feat)
    for label, m in reps.items():
        sp = KC64.supports_from_dense(L, feat, bk, m)
        for name, t, n in (("quad", 4, quad_n), ("triple_sampled_check", 3, KC64.N_TRIPLE_CHECK)):
            seq = sample_tuples(WC.D, t, n, KC64.SAMPLE_SEED, sp.q.device)
            cov, cnt = WV.query_multiplicity(sp, seq)
            out.setdefault(name, dict(order=t, sample_size=n, covered=int(cov.sum())))[label] = WV.categories(cnt)
        del sp
    del gm, reps
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def record_feature_cover(L, em: torch.Tensor, w=None, chunk: int = 1 << 12) -> dict:
    """Per original record (logical block): local inputs enabled in at least one valid neuron (mask), and with w
    those that also carry a nonzero weight there (numerical; an enabled zero weight adds no input dependence).
    Any mask (local, width, mixed).  Returns distributions over the valid records."""
    from sparse_pooling.experimental.width_masks import layout_counts

    nb = L.groups * L.blocks_per_group
    valid = L.valid_mask(em.device).view(nb, L.block)
    emv = em.view(nb, L.block, 64)
    wv = None if w is None else w.view(nb, L.block, 64)
    m_cnt, n_cnt, keep = [], [], []
    for b0 in range(0, nb, chunk):
        v = valid[b0:b0 + chunk]
        e = emv[b0:b0 + chunk].bool() & v[:, :, None]
        m_cnt.append(e.any(1).sum(1))
        if wv is not None:
            n_cnt.append((e & (wv[b0:b0 + chunk] != 0)).any(1).sum(1))
        keep.append(v.any(1))
    keep = torch.cat(keep)
    mc = torch.cat(m_cnt)[keep]
    out = dict(records=int(keep.sum()), mask=KC64.dist(mc), mask_complete=int((mc == 64).sum()),
               mask_missing_total=int((64 - mc).sum()))
    if wv is not None:
        nc = torch.cat(n_cnt)[keep]
        out.update(numerical=KC64.dist(nc), numerical_complete=int((nc == 64).sum()),
                   enabled_zero_inputs=int((mc - nc).sum()))
    return out


def feature_union_vs_dense(feats, feats_dense) -> dict:
    """Per-output exact input-feature union against the dense envelope's (all outputs)."""
    f, d = torch.as_tensor(feats).long(), torch.as_tensor(feats_dense).long().cpu()
    miss = d - f
    return dict(outputs=int(f.numel()), outputs_equal_dense=int((miss == 0).sum()), missing_total=int(miss.sum()),
                missing_per_output=KC64.dist(miss.double()), never_exceeds_dense=bool((miss >= 0).all()))


@torch.no_grad()
def mixed_anchor_check(L, sup, w) -> dict:
    """Mixed supports: every valid original record holds exactly one anchor row group whose valid neurons enable all
    64 envelope inputs (mask) and hold 64 nonzero weights (numerical)."""
    n16 = sup.nv.numel()
    gpb = L.block // 16
    wv = w.reshape(n16, 16, 64)
    valid = L.valid_mask(w.device).reshape(n16, 16)
    a = sup.anchor.to(w.device)
    full_mask = sup.gm.to(w.device).all(1)
    nz_full = ((wv != 0).all(2) | ~valid).all(1)
    blk_valid = sup.nv.to(w.device).view(-1, gpb).sum(1) > 0
    per = a.view(-1, gpb)
    return dict(records=int(blk_valid.sum()), records_with_one_anchor=int((per.sum(1)[blk_valid] == 1).sum()),
                anchor_groups=int(a.sum()), anchors_full_mask=int((a & full_mask).sum()),
                anchors_all_nonzero=int((a & nz_full).sum()),
                anchor_neurons=int(sup.nv.to(w.device)[a].sum()))


def predictions(sup, L, bk, spec, Q) -> dict:
    """Uniform-policy predictions only: magnitude-dependent (FP-mag partially, WRS / TopK) and mixed supports
    violate the uniform model's assumptions (measured only)."""
    from sparse_pooling.experimental import local_supports as LS

    if sup is None or not hasattr(sup, "env") or sup.policy in LS.WEIGHTED:
        return None
    ri = LS.refined_inputs(sup, L, bk, Q=Q)
    out = dict(model="independent rounds: 1 - prod_r (1 - p_g(t) mean_{q in r} h_q); p_g = C(63, t-1) / C(1023, t-1)",
               refined=dict(), ideal=dict())
    groups = int((sup.nv > 0).sum())
    records = int(ri["rec_groups"].numel())
    for t in (2, 3, 4):
        h = LS.set_hit(sup.policy, t, sup.k, ri["m_q"], ri["b_q"], ri["rec_groups"], ri["rec_set"], ri["used_tiles"])
        out["refined"][f"t{t}"] = LS.layer_expectation(h, t, WC.D // WC.S)
        if sup.policy != "FP-mag":
            out["ideal"][f"t{t}"] = LS.ideal_prediction(sup.policy, t, sup.k, groups, ri["Q"], records)
    out["inputs"] = dict(Q_total=ri["Q"], sets_occupied=ri["sets_occupied"], sets_empty=ri["sets_empty"],
                         m_q=KC64.dist(ri["m_q"]), b_q=KC64.dist(ri["b_q"]), groups_per_record=KC64.dist(ri["rec_groups"]),
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
    res = dict(schema="granule.rsn_coverage/2", C=C, S=WC.S, R=WC.R, H=WC.H, graph_seed=a.graph_seed,
               support=a.support, graph_fingerprint=kcsr_graph.fingerprint(tab))
    del tab
    tri = sample_tuples(WC.D, 3, KC64.N_OUT_TRIPLE, KC64.SAMPLE_SEED + 7, dev)
    quad = sample_tuples(WC.D, 4, KC64.N_OUT_QUAD, KC64.SAMPLE_SEED + 8, dev)
    sp0 = KC64.supports_from_dense(L, feat, bk, None)
    feats0 = KC64.per_output_features(sp0).cpu()
    outs, why = WV.fixed_outputs(feats0, counts, WC.G)
    del sp0
    torch.cuda.empty_cache()
    res["output_selection"] = dict(why, fixed=outs, rule="the parent's fixed selection (wps_coverage.py)")
    n_rec_out = int(math.ceil(int(counts.max()) / 16))
    res["bounds"] = dict(per_output_union={f"t{t}": LS.per_output_union_bound(n_rec_out, t, 16) for t in (2, 3, 4)},
                         per_output_union_k8={f"t{t}": LS.per_output_union_bound(n_rec_out, t, 8) for t in (2, 3, 4)},
                         per_output_union_k16={f"t{t}": LS.per_output_union_bound(n_rec_out, t, 16) for t in (2, 3, 4)},
                         per_output_records_max=n_rec_out,
                         one_tile_union={f"t{t}": min(1.0, (WC.R * WC.D // WC.S) * 4 * math.comb(16, t) / math.comb(WC.D, t))
                                         for t in (2, 3, 4)},
                         note="per_output_union_k<k> applies to every uniform fan-in-k row-group support (<= n_rec "
                              "k-input support records per output; 'per_output_union' = the K16 value, kept for "
                              "continuity); it does not apply to multi-tile (U*) or mixed / anchored supports; the "
                              "one-tile bound only to ONE common partition per set")
    joint = dict(enabled={}, nonzero={})
    if a.support == "dense":
        res["kind"] = "dense potential (all 64 inputs of every valid neuron)"
        res["enabled"] = WV.coverage_of(L, feat, bk, None, outs, tri, quad, KC64.N_QUAD, joint_out=joint["enabled"])
    else:
        rec, w, b, _ = SP.load_checkpoint(WC.ckpt(C, a.graph_seed))
        if rec["spec"]["seed"] != a.graph_seed or rec["C"] != C:
            raise SystemExit("checkpoint graph does not match")
        res["checkpoint"] = dict(path=WC.ckpt(C, a.graph_seed), weight_sha256=rec["weight_sha256"])
        holder = SimpleNamespace(weight=w, layout=L, kg_buckets=bk, sg_feat=feat)
        t0 = time.perf_counter()
        em, sup, srec = RC.support(a.support, holder, a.graph_seed)
        res["support_seconds"] = time.perf_counter() - t0
        res["support_record"] = srec
        res["kind"] = a.support
        res["width"] = WV.width_stats(L, em, w)
        res["anchors"] = ((mixed_anchor_check(L, sup, w) if hasattr(sup, "kc") else WV.anchor_check(L, sup, w))
                          if hasattr(sup, "anchor") else None)
        dens = res["width"]["enabled_slots"] / WC.W0
        qn = KC64.N_QUAD_SPARSE if dens <= 0.10 else KC64.N_QUAD
        res["quad_sample_rule"] = f"{qn:,} (50M when the enabled density <= 0.10, else 10M; predeclared)"
        res["predictions"] = predictions(sup, L, bk, a.support, int(feat.shape[0]))
        bq = LS.block_sets(L, bk, dev)
        res["sharing"] = sharing(L, em, bq, feat)
        res["support_record_multiplicity"] = group_multiplicity(L, feat, bk, em, qn, bq)
        res["enabled"] = WV.coverage_of(L, feat, bk, em, outs, tri, quad, qn, joint_out=joint["enabled"])
        res["record_feature_cover"] = record_feature_cover(L, em, w)
        res["feature_union_vs_dense"] = dict(
            enabled=feature_union_vs_dense(joint["enabled"]["per_output_features"], feats0))
        nzm = (em.bool() & (w != 0)).to(torch.uint8)
        if torch.equal(nzm, em):
            res["nonzero"] = dict(identical_to_enabled=True, zero_enabled_weights=0)
        else:
            res["nonzero"] = WV.coverage_of(L, feat, bk, nzm, outs, tri, quad, qn, joint_out=joint["nonzero"])
            res["nonzero"]["zero_enabled_weights"] = int((em.bool() & (w == 0)).sum())
            res["feature_union_vs_dense"]["nonzero"] = feature_union_vs_dense(
                joint["nonzero"]["per_output_features"], feats0)
        del w, b, em, nzm
    res["analysis_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    res["analysis_seconds"] = time.perf_counter() - t_all
    res["provenance"] = provenance(Path(common.__file__).resolve().parent.parent)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    import numpy as np

    np.savez_compressed(str(Path(a.out).with_suffix("")) + ".joint.npz",
                        **{f"{scope}_{k}": v for scope, d in joint.items() for k, v in d.items()},
                        outputs=np.array(outs, dtype=np.int64), tri_seed=KC64.SAMPLE_SEED + 7,
                        quad_seed=KC64.SAMPLE_SEED + 8)
    e = res["enabled"]
    print(f"{res['kind']}: pair {100 * e['pair']['fraction']:.4f}% triple {100 * e['triple']['fraction']:.4f}% quad "
          f"{100 * e['quad']['fraction']:.5f}% | per-output triple median "
          f"{100 * e['per_output_summary']['triple']['median']:.4f}% | {res['analysis_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
