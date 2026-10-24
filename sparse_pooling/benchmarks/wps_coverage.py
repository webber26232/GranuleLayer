"""Coverage, tuple multiplicity and effective width of the width-preserving supports
(results/WIDTH_PRESERVING_SPARSITY.md section 8), on the exact timed graph, checkpoint and support.

For one (checkpoint, policy, rho, score, seed) support -- rebuilt deterministically exactly as the timing
factory builds it (its element-mask sha256 is recorded and compared by the report) -- or a saved support
(``--mask-file``: e.g. the exact nonzero support after fine-tuning):

1. realized group / dense potential-record coverage (the graph), with occupied sets distinguished;
2. enabled-mask coverage: a tuple lies in the support of at least ONE valid neuron (per-neuron 64-bit
   local supports; supports of different neurons are never united);
3. exact-nonzero coverage (enabled AND w != 0 of the evaluated values) and thresholded coverage
   (|w| > threshold, separately labelled);
4. per output: distinct features for all G outputs; exact pairs, sampled triples / quadruples on a FIXED
   selection shared by every support of the graph (32 seeded outputs, the 4 lowest-feature outputs of the
   dense potential, one output of each neuron-count class = partial-tail class), plus the 4 lowest-feature
   outputs of this support (flagged);
5. tuple multiplicity: for every sampled tuple, the number of valid neurons of the selected output whose
   support contains it (categories 0 / 1 / 2-15 / 16-63 / 64+ and quantiles of the covered), and the
   layer-wide number of supporting neurons on the common quadruple / triple samples;
6. anchors (A / H2A): every valid record must hold a neuron with full 64-input support (structural,
   all outputs) and numerically nonzero (all 64 weights != 0); per-output coverage compared with the dense
   potential on the selected outputs (exact pairs, the same sampled triples / quadruples);
7. exact all-neuron mask-degree and nonzero fan-in histograms, bias-only fractions under both
   definitions, full-support neurons, input-dependent neurons per output (min / p10 / median / p90 / max).

Pairs / triples layer-wide are exact; quadruples sampled with a predeclared fixed size (10M; 50M when the
enabled density is <= 0.10), the same seeded distinct tuples for every support; Clopper-Pearson 95 %
intervals and zero-hit upper bounds.  Analysis buffers / time are outside every timing run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import k64_coverage as KC64
import wps_cases as WC
from kcsr_coverage import clopper_pearson, sample_tuples

N_QUAD, N_QUAD_SPARSE = KC64.N_QUAD, KC64.N_QUAD_SPARSE
N_OUT_TRIPLE, N_OUT_QUAD = KC64.N_OUT_TRIPLE, KC64.N_OUT_QUAD
MULT_EDGES = (0, 1, 2, 16, 64)  # categories 0, 1, 2-15, 16-63, 64+

_LUT = None


def popcount64(x: torch.Tensor) -> torch.Tensor:
    """Per-element popcount of an int64 tensor (byte lookup)."""
    global _LUT
    if _LUT is None or _LUT.device != x.device:
        _LUT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int64, device=x.device)
    b = x.contiguous().view(torch.uint8).view(*x.shape, 8).long()
    return _LUT[b].sum(-1)


def categories(cnt: torch.Tensor) -> dict:
    e = MULT_EDGES
    names = ["0", "1", "2-15", "16-63", "64+"]
    out = {}
    for i, nm in enumerate(names):
        lo = e[i]
        hi = e[i + 1] if i + 1 < len(e) else None
        sel = (cnt >= lo) if hi is None else ((cnt >= lo) & (cnt < hi))
        out[nm] = int(sel.sum())
    cov = cnt[cnt > 0].double()
    q = dict(n=int(cnt.numel()), covered=int((cnt > 0).sum()))
    if cov.numel():
        qs = torch.quantile(cov.cpu(), torch.tensor([0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
        q.update(min=float(cov.min()), p10=float(qs[0]), median=float(qs[1]), p90=float(qs[2]), p99=float(qs[3]),
                 max=float(cov.max()), mean=float(cov.mean()))
    return dict(categories=out, covered_quantiles=q)


@torch.no_grad()
def query_multiplicity(sp: "KC64.Supports", tup: torch.Tensor, chunk: int = 0):
    """(covered [n] bool, supporting valid neurons [n] int64) layer-wide for sorted tuples [n, t]."""
    dev = sp.q.device
    bits = sp.bitsets()
    t = tup.shape[1]
    n = tup.shape[0]
    chunk = chunk or max(1024, (1 << 25) // (sp.R * t))
    cnt = torch.zeros(n, dtype=torch.int64, device=dev)
    for s0 in range(0, n, chunk):
        tp = tup[s0:s0 + chunk]
        s = sp.sid[:, tp]
        same = (s == s[:, :, :1]).all(-1)
        r_idx, i_idx = torch.nonzero(same, as_tuple=True)
        if r_idx.numel() == 0:
            continue
        q = r_idx * sp.per_round + s[r_idx, i_idx, 0].long()
        lf = sp.lid[q[:, None], tp[i_idx]].long()
        step = max(1, (1 << 24) // (t * bits.shape[2]))
        for c0 in range(0, q.numel(), step):
            qc, lc = q[c0:c0 + step], lf[c0:c0 + step]
            acc = bits[qc, lc[:, 0]]
            for k in range(1, t):
                acc = acc & bits[qc, lc[:, k]]
            cnt.index_add_(0, s0 + i_idx[c0:c0 + step], popcount64(acc).sum(1))
    return cnt > 0, cnt


@torch.no_grad()
def output_rows(sp, outputs, tri, quad, chunk: int = 1 << 15):
    """Per selected output: features, exact pairs, sampled triples / quadruples with multiplicity."""
    from kcsr_coverage import comb_table

    dev = sp.q.device
    ct2 = int(comb_table(sp.D, 2, "cpu")[sp.D, 2])
    iu = torch.triu_indices(sp.D, sp.D, 1, device=dev)
    rows = []
    for g in outputs:
        M = KC64.output_membership(sp, g)
        rec = dict(output=int(g), active_neurons=int(M.shape[0]))
        if M.shape[0] == 0:
            rec.update(features=0, pair=0.0, pair_covered=0, triple=0.0, quad=0.0, triple_hits=0, quad_hits=0)
            rows.append(rec)
            continue
        rec["features"] = int((M.sum(0) > 0).sum())
        P = (M.t() @ M) > 0
        rec["pair_covered"] = int(P[iu[0], iu[1]].sum())
        rec["pair"] = rec["pair_covered"] / ct2
        del P
        W = (M.shape[0] + 63) // 64
        Mb = M.t().bool()
        pad = W * 64 - Mb.shape[1]
        if pad:
            Mb = torch.nn.functional.pad(Mb, (0, pad))
        sh = torch.arange(64, device=dev)
        fb = (Mb.view(sp.D, W, 64).to(torch.int64) << sh).sum(-1)
        del Mb, M
        for name, tup in (("triple", tri), ("quad", quad)):
            cnt = torch.empty(tup.shape[0], dtype=torch.int64, device=dev)
            for s0 in range(0, tup.shape[0], chunk):
                tp = tup[s0:s0 + chunk]
                acc = fb[tp[:, 0]]
                for j in range(1, tp.shape[1]):
                    acc = acc & fb[tp[:, j]]
                cnt[s0:s0 + chunk] = popcount64(acc).sum(1)
            k = int((cnt > 0).sum())
            n = tup.shape[0]
            lo, hi = clopper_pearson(k, n)
            rec[f"{name}_hits"], rec[f"{name}_n"], rec[name], rec[f"{name}_ci95"] = k, n, k / n, [lo, hi]
            rec[f"{name}_multiplicity"] = categories(cnt)
        rows.append(rec)
    return rows


@torch.no_grad()
def width_stats(L, em: torch.Tensor, w: torch.Tensor, chunk: int = 1 << 21) -> dict:
    """Exact all-neuron mask-degree / nonzero fan-in histograms, bias-only fractions (both definitions),
    full-support neurons, input-dependent neurons per output."""
    dev = em.device
    valid = L.valid_mask(dev).reshape(-1)
    ev = em.reshape(-1, 64)
    wv = w.reshape(-1, 64)
    deg = torch.empty(ev.shape[0], dtype=torch.int64, device=dev)
    nz = torch.empty_like(deg)
    for s0 in range(0, ev.shape[0], chunk):
        e = ev[s0:s0 + chunk].bool()
        deg[s0:s0 + chunk] = e.sum(1)
        nz[s0:s0 + chunk] = (e & (wv[s0:s0 + chunk] != 0)).sum(1)
    deg, nz = deg[valid], nz[valid]
    H = int(valid.sum())
    g = torch.repeat_interleave(torch.arange(L.groups, device=dev), L.counts_tensor(dev).to(torch.int64))
    idep = torch.zeros(L.groups, dtype=torch.int64, device=dev).index_add_(0, g, (nz > 0).to(torch.int64))
    idep_m = torch.zeros(L.groups, dtype=torch.int64, device=dev).index_add_(0, g, (deg > 0).to(torch.int64))
    return dict(H=H, mask_degree_hist=torch.bincount(deg, minlength=65).tolist(),
                nonzero_fan_in_hist=torch.bincount(nz, minlength=65).tolist(),
                bias_only_mask=int((deg == 0).sum()), bias_only_numeric=int((nz == 0).sum()),
                bias_only_mask_fraction=float((deg == 0).double().mean()),
                bias_only_numeric_fraction=float((nz == 0).double().mean()),
                full_support_neurons_mask=int((deg == 64).sum()), full_support_neurons_numeric=int((nz == 64).sum()),
                mean_mask_degree=float(deg.double().mean()), mean_nonzero_fan_in=float(nz.double().mean()),
                enabled_slots=int(deg.sum()), nonzero_slots=int(nz.sum()),
                input_dependent_neurons=int((nz > 0).sum()),
                input_dependent_per_output=KC64.dist(idep), input_dependent_per_output_mask=KC64.dist(idep_m))


@torch.no_grad()
def anchor_check(L, ms, w) -> dict:
    """Every valid record holds a full-support neuron (mask) whose 64 weights are all nonzero (numeric)."""
    if ms is None or not ms.policy.anchors:
        return None
    n16 = ms.nv.numel()
    gpb = L.block // 16
    wv = w.reshape(n16, 16, 64)
    valid = L.valid_mask(w.device).reshape(n16, 16)
    a = ms.anchor
    full = ms.tiles.all(1) if ms.bb is None else ms.tiles.all(1)  # (backbone coordinates are enabled as well)
    nz_full = ((wv != 0).all(2) | ~valid).all(1)  # every valid neuron of the group has 64 nonzero weights
    per = a.view(-1, gpb)
    blk_valid = ms.nv.view(-1, gpb).sum(1) > 0
    ok_mask = bool(((per & full.view(-1, gpb)).any(1) | ~blk_valid).all())
    ok_num = bool(((per & nz_full.view(-1, gpb)).any(1) | ~blk_valid).all())
    zero_anchor_weights = int(((wv[a] == 0) & valid[a][:, :, None]).sum())
    near = {f"{t:g}": int(((wv[a].float().abs() <= t) & valid[a][:, :, None]).sum()) for t in (1e-4, 1e-3, 1e-2)}
    return dict(records=int(blk_valid.sum()), records_with_full_support_anchor=int((per & full.view(-1, gpb)).any(1).sum()),
                all_records_have_full_mask_anchor=ok_mask, all_records_have_numerically_full_anchor=ok_num,
                anchor_neurons=int(ms.nv[a].sum()), zero_anchor_weights=zero_anchor_weights,
                near_zero_anchor_weights=near,
                note="structural all-output guarantee: a full-support anchor neuron per valid record contains every "
                     "tuple of its record, so every dense-potential tuple of every output stays covered (support "
                     "existence only: not multiplicity, weight magnitude or accuracy)")


def fixed_outputs(feats_dense: torch.Tensor, counts: torch.Tensor, G: int):
    sel, why = KC64.select_outputs(G, feats_dense, counts)
    return sel, why


def coverage_of(L, feat, bk, mask, outputs, tri, quad, quad_n, extra_low=True):
    t0 = time.perf_counter()
    sp = KC64.supports_from_dense(L, feat, bk, mask)
    res = dict(supports=dict(valid_neurons=int(sp.q.numel()), active_neurons=int((sp.bits != 0).sum()),
                             all_zero_neurons=int((sp.bits == 0).sum())))
    ex = KC64.exact_support_coverage(sp, (2, 3))
    res["pair"], res["triple"] = ex[2], ex[3]
    res["exact_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for name, t, n in (("quad", 4, quad_n), ("triple_sampled_check", 3, KC64.N_TRIPLE_CHECK)):
        seq = sample_tuples(KC64_D(), t, n, KC64.SAMPLE_SEED, sp.q.device)
        cov, cnt = query_multiplicity(sp, seq)
        k = int(cov.sum())
        lo, hi = clopper_pearson(k, n)
        res[name] = dict(method="sampled", order=t, sample_size=n, sample_seed=KC64.SAMPLE_SEED, fixed_size=True,
                         hits=k, fraction=k / n, ci95=[lo, hi], upper_bound_if_zero=(hi if k == 0 else None),
                         layer_multiplicity=categories(cnt))
    res["sampled_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats = KC64.per_output_features(sp)
    res["per_output_features"] = KC64.dist(feats)
    outs = list(outputs)
    low = torch.argsort(feats.cpu(), stable=True)[:4].tolist() if extra_low else []
    res["outputs_low_of_this_support"] = low
    rows = output_rows(sp, outs + [o for o in low if o not in outs], tri, quad)
    for r in rows:
        r["fixed_selection"] = r["output"] in outs
    res["outputs"] = rows
    fixed = [r for r in rows if r["fixed_selection"]]
    res["per_output_summary"] = {k: KC64.dist(torch.tensor([r[k] for r in fixed], dtype=torch.float64))
                                 for k in ("pair", "triple", "quad", "features")}
    res["per_output_seconds"] = time.perf_counter() - t0
    del sp
    torch.cuda.empty_cache()
    return res


def KC64_D():
    return WC.D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--policy", default="dense", help="width_masks policy or 'dense' (the dense potential)")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--score", default="magnitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-file", default="", help="saved support (fine-tune output: packed bits + record)")
    ap.add_argument("--thresholds", default="", help="comma list: also |w| > t coverage (separately labelled)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    import kcsr_graph

    dev = torch.device("cuda")
    t_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    C = a.C
    L = KLayout(WC.D, WC.H, WC.G, 64, C)
    counts = L.counts_tensor(dev)
    plan = ConnectivityPlan(WC.S, WC.R, seed=a.graph_seed)
    idx, feat, ubk, _ = KG.generate(WC.D, counts, C, 64, L.blocks_per_group, 1, plan, dev)
    bk = KG.unit_to_block_buckets(ubk, counts, C, 1)
    tab = KG.record_table(WC.D, WC.H, counts, C, 64, idx, feat, bk)
    res = dict(C=C, S=WC.S, R=WC.R, H=WC.H, graph_seed=a.graph_seed, policy=a.policy, rho_target=a.rho,
               score=a.score, seed=a.seed, graph_fingerprint=kcsr_graph.fingerprint(tab))
    del tab
    # dense potential (every valid neuron: all 64 inputs) -> fixed output selection + per-output reference
    tri = sample_tuples(WC.D, 3, N_OUT_TRIPLE, KC64.SAMPLE_SEED + 7, dev)
    quad = sample_tuples(WC.D, 4, N_OUT_QUAD, KC64.SAMPLE_SEED + 8, dev)
    sp0 = KC64.supports_from_dense(L, feat, bk, None)
    feats0 = KC64.per_output_features(sp0)
    outs, why = fixed_outputs(feats0, counts, WC.G)
    res["output_selection"] = dict(why, fixed=outs, rule="32 seeded outputs (seed 20261024) + the 4 lowest-feature "
                                   "outputs of the dense potential + one output per neuron-count class (n_g = 12208 / "
                                   "12207: the latter has a 15-neuron partial row group)")
    pot_rows = output_rows(sp0, outs, tri, quad)
    del sp0
    torch.cuda.empty_cache()
    res["dense_potential_outputs"] = pot_rows
    if a.policy == "dense" and not a.mask_file:
        res["kind"] = "dense potential (all 64 inputs of every valid neuron)"
        res["dense"] = coverage_of(L, feat, bk, None, outs, tri, quad, N_QUAD)
    else:
        rec, w, b, _ = SP.load_checkpoint(a.checkpoint)
        if rec["spec"]["seed"] != a.graph_seed or rec["C"] != C:
            raise SystemExit("checkpoint graph does not match --graph-seed / --C")
        res["checkpoint"] = dict(path=a.checkpoint, weight_sha256=rec["weight_sha256"], tag=rec.get("meta", {}).get("tag"))
        ms = None
        if a.mask_file:
            blob = torch.load(a.mask_file, map_location="cpu", weights_only=False)
            em = SP._unpack_bits(blob["mask_bits"].to(dev), w.numel()).view(w.shape)
            res["mask_file"] = dict(path=a.mask_file, record=blob.get("rec"))
            if "weight" in blob:  # fine-tuned values
                w = blob["weight"].to(dev)
            res["kind"] = blob.get("rec", {}).get("kind", "saved support")
        else:
            ms = WM.build(a.policy, a.rho, w, L, score=a.score, seed=a.seed)
            em = ms.element_mask(L)
            res["mask_stats"] = ms.stats
            res["kind"] = f"{a.policy} rho {a.rho:g} ({a.score})"
        res["mask_sha256"] = SP.mask_fingerprint(em)
        res["width"] = width_stats(L, em, w)
        res["anchors"] = anchor_check(L, ms, w)
        dens = res["width"]["enabled_slots"] / WC.W0
        qn = N_QUAD_SPARSE if dens <= 0.10 else N_QUAD
        res["quad_sample_rule"] = f"{qn:,} (50M when the enabled density <= 0.10, else 10M; predeclared)"
        res["enabled"] = coverage_of(L, feat, bk, em, outs, tri, quad, qn)
        nzm = (em.bool() & (w != 0)).to(torch.uint8)
        if torch.equal(nzm, em):
            res["nonzero"] = dict(identical_to_enabled=True, zero_enabled_weights=0)
        else:
            res["nonzero"] = coverage_of(L, feat, bk, nzm, outs, tri, quad, qn)
            res["nonzero"]["zero_enabled_weights"] = int((em.bool() & (w == 0)).sum())
        for t in filter(None, a.thresholds.split(",")):
            thr = float(t)
            tm = (em.bool() & (w.float().abs() > thr)).to(torch.uint8)
            res[f"threshold_{t}"] = coverage_of(L, feat, bk, tm, outs, tri, quad, qn)
            res[f"threshold_{t}"]["label"] = f"|w| > {t} (thresholded; NOT the exact support)"
            del tm
        if res.get("anchors"):  # per-output comparison with the dense potential on the fixed selection
            cmp = []
            for r0, r1 in zip(pot_rows, [r for r in res["enabled"]["outputs"] if r["fixed_selection"]]):
                cmp.append(dict(output=r0["output"], pair_equal=r0["pair_covered"] == r1["pair_covered"],
                                triple_equal=r0["triple_hits"] == r1["triple_hits"],
                                quad_equal=r0["quad_hits"] == r1["quad_hits"],
                                features_equal=r0["features"] == r1["features"]))
            res["anchor_vs_potential_per_output"] = dict(rows=cmp, all_equal=all(all(v for k, v in c.items()
                                                                                     if k != "output") for c in cmp))
        del w, b, em, nzm
    res["one_tile_bounds"] = {f"b{b}": WM.describe_bounds(WC.R * WC.D // WC.S, b, WC.D) for b in (16, 8)}
    res["analysis_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    res["analysis_seconds"] = time.perf_counter() - t_all
    res["provenance"] = provenance(Path(common.__file__).resolve().parent.parent)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    e = res.get("enabled") or res.get("dense")
    w_ = res.get("width", {})
    print(f"{res['kind']}: pair {100 * e['pair']['fraction']:.4f}% triple {100 * e['triple']['fraction']:.4f}% quad "
          f"{100 * e['quad']['fraction']:.5f}% | bias-only {w_.get('bias_only_mask_fraction', 0):.4f} "
          f"| per-output triple median {100 * e['per_output_summary']['triple']['median']:.4f}% "
          f"| {res['analysis_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
