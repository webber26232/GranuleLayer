"""Coverage inventory of one support on the actual checkpoint graph (results/INTERACTION_COVERAGE_K16_DX.md section
4.1), complementing rsn_coverage.py (exact layer pairs / triples, per-output pairs / triples / sampled quadruples on
common queries with joint intervals, global supports and tuple multiplicity, record / output feature unions):

* structural / numerical width (bias-only rows), zero-weight enabled slots, enabled slots;
* row-group and valid-neuron-weighted input multiplicity per original record (mean, spread, low / high quantiles),
  separately for full and tail records;
* EXACT record-scope tuple coverage: the fraction of the record's C(64, 2) local pairs and C(64, 3) local triples
  held together by at least one of its row groups (unique coverage), full / tail;
* tuple-INCIDENCE sums sum_u C(k_u, t) (a shared 16-neuron support counts once) per record and per output, and the
  per-output union bound min(1, sum_u C(k_u, t) / C(1024, t)) from the ACTUAL group populations (a ceiling, not a
  prediction);
* retained pre-pruning energy.

    python icdx_inventory.py --support Balanced-TopK/16#0 --graph-seed 0 --out X.json
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
import rsn_cases as RC
import wps_cases as WC


def dist(v):
    v = torch.as_tensor(v).double().flatten()
    if v.numel() == 0:
        return {}
    q = torch.quantile(v[: 1 << 24], torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64))
    return dict(mean=float(v.mean()), sd=float(v.std()) if v.numel() > 1 else 0.0, min=float(v.min()), p1=float(q[0]),
                p5=float(q[1]), median=float(q[2]), p95=float(q[3]), p99=float(q[4]), max=float(v.max()),
                n=int(v.numel()))


@torch.no_grad()
def record_tuple_coverage(G: torch.Tensor, chunk: int = 256) -> dict:
    """G [records, groups, 64] bool group supports (absent groups all False).  Exact fraction of local pairs / triples
    of each record contained in at least one group support."""
    R = G.shape[0]
    pair = torch.zeros(R, dtype=torch.float64, device=G.device)
    tri = torch.zeros(R, dtype=torch.float64, device=G.device)
    iu = torch.triu_indices(64, 64, 1, device=G.device)
    a, b, c = torch.combinations(torch.arange(64, device=G.device), 3).unbind(1)
    for r0 in range(0, R, chunk):
        g = G[r0:r0 + chunk]
        P = torch.zeros((g.shape[0], 64, 64), dtype=torch.bool, device=G.device)
        Tm = torch.zeros((g.shape[0], a.numel()), dtype=torch.bool, device=G.device)
        for u in range(g.shape[1]):
            x = g[:, u]
            P |= x[:, :, None] & x[:, None, :]
            Tm |= x[:, a] & x[:, b] & x[:, c]
        pair[r0:r0 + chunk] = P[:, iu[0], iu[1]].double().mean(1)
        tri[r0:r0 + chunk] = Tm.double().mean(1)
    return dict(pair=pair, triple=tri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    g = a.graph_seed
    L = KLayout(WC.D, WC.H, WC.G, 64, RC.C)
    counts = L.counts_tensor(dev)
    idx, feat, ubk, _ = KG.generate(WC.D, counts, RC.C, 64, L.blocks_per_group, 1, ConnectivityPlan(WC.S, WC.R, seed=g), dev)
    bk = KG.unit_to_block_buckets(ubk, counts, RC.C, 1)
    del idx
    rec, w, b, _ = SP.load_checkpoint(WC.ckpt(RC.C, g))
    if rec["spec"]["seed"] != g:
        raise SystemExit("checkpoint graph does not match")
    holder = SimpleNamespace(weight=w, layout=L, kg_buckets=bk, sg_feat=feat)
    em, sup, srec = RC.support(a.support, holder, g)
    geo = LS.geometry(L, bk.cpu(), "cpu", Q=int(feat.shape[0]))
    n16 = geo["nv"].numel()
    uv, rec_v, nv_v, gpb = geo["uv"], geo["rec"], geo["nv"][geo["uv"]], geo["gpb"]
    valid = L.valid_mask(dev).reshape(n16, 16)
    if em is None:
        gm = torch.ones((len(uv), 64), dtype=torch.bool)
    else:
        e = em.view(n16, 16, 64).bool()
        g0 = e[uv, 0].cpu()
        if not bool(((e[uv] == e[uv, :1]) | ~valid[uv][:, :, None]).all()):
            raise AssertionError("valid neurons of a row group do not share one support")
        gm = g0
    k_u = gm.sum(1)
    res = dict(schema="granule.icdx_inventory/1", support=a.support, graph_seed=g,
               support_record={k: v for k, v in srec.items() if k in ("mask_sha256", "support_fingerprint", "rho_mask")},
               provenance=provenance(Path(common.__file__).resolve().parent.parent))
    # width / numerical width
    wv = w.reshape(n16, 16, 64)
    m = torch.ones_like(wv, dtype=torch.bool) if em is None else em.view(n16, 16, 64).bool()
    nz = (wv != 0) & m & valid[:, :, None]
    deg_s = (m & valid[:, :, None]).sum(2)[valid]
    deg_n = nz.sum(2)[valid]
    res["width"] = dict(enabled_slots=int((m & valid[:, :, None]).sum()), zero_weight_enabled_slots=int(
        ((wv == 0) & m & valid[:, :, None]).sum()), structural_bias_only=int((deg_s == 0).sum()),
        numerical_bias_only=int((deg_n == 0).sum()), fan_in_structural=dist(deg_s.cpu()),
        fan_in_numerical=dist(deg_n.cpu()), valid_neurons=int(valid.sum()))
    # energy
    a_v = LS.feature_importance(w, L)[uv]
    res["energy_layer_fraction"] = float((a_v * gm).sum() / a_v.sum())
    del a_v
    full = geo["groups_per_record"] == gpb
    mult = LS.input_multiplicity(gm, nv_v, rec_v, geo["records"])
    res["multiplicity"] = {}
    for name, sel in (("all", torch.ones_like(full)), ("full", full), ("tail", ~full)):
        Gm, Nm = mult["groups"][sel].double(), mult["neurons"][sel].double()
        res["multiplicity"][name] = dict(records=int(sel.sum()), groups=dist(Gm), neurons=dist(Nm),
                                         per_record_spread_groups=dist(Gm.max(1).values - Gm.min(1).values),
                                         per_record_spread_neurons=dist(Nm.max(1).values - Nm.min(1).values),
                                         per_record_cv_neurons=dist(Nm.std(1) / Nm.mean(1).clamp_min(1e-30)),
                                         records_complete=int((Gm > 0).all(1).sum()))
    # exact record-scope tuple coverage
    gidx = LS.record_groups(rec_v, geo["rank_rec"], gpb)
    Gr = torch.zeros((geo["records"], gpb, 64), dtype=torch.bool)
    ok = gidx >= 0
    Gr[ok] = gm[gidx[ok]]
    rt = record_tuple_coverage(Gr.to(dev))
    res["record_tuple_coverage"] = {name: dict(pair=dist(rt["pair"][sel.to(dev)].cpu()), triple=dist(
        rt["triple"][sel.to(dev)].cpu())) for name, sel in (("all", torch.ones_like(full)), ("full", full),
                                                            ("tail", ~full))}
    # tuple incidence sums and per-output union bounds (actual populations)
    out_of = uv // (L.blocks_per_group * gpb)
    inc = {}
    for t in (2, 3, 4):
        cu = torch.tensor([math.comb(int(k), t) for k in range(65)], dtype=torch.float64)[k_u]
        per_rec = torch.zeros(geo["records"], dtype=torch.float64).index_add_(0, rec_v, cu)
        per_out = torch.zeros(L.groups, dtype=torch.float64).index_add_(0, out_of, cu)
        ub = (per_out / math.comb(WC.D, t)).clamp(max=1.0)
        inc[f"t{t}"] = dict(record_full=dist(per_rec[full]), record_tail=dist(per_rec[~full]),
                            output=dist(per_out), union_bound_per_output=dist(ub),
                            union_bound_max_output=float(ub.max()))
    res["tuple_incidence"] = inc
    res["class_populations"] = {f"K{int(k)}": int(nv_v[k_u == k].sum()) for k in torch.unique(k_u)}
    res["groups_per_output_max"] = int(torch.bincount(out_of).max())
    res["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"{a.support} g{g}: record pairs {res['record_tuple_coverage']['full']['pair']['mean']:.4f} triples "
          f"{res['record_tuple_coverage']['full']['triple']['mean']:.4f} | union t3 max "
          f"{100 * inc['t3']['union_bound_max_output']:.5f}% | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
