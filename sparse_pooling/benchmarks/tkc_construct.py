"""Construction of TopK / Cover-TopK on the actual alpha-0 checkpoints (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md
section 4): per graph seed and k, the measured importance / TopK / repair time, host and device peaks, swap counts,
donor / target positions, retained energy vs TopK (full records and the n175 / n176 tails), mask and numerical
record coverage, support fingerprints (TopK compared with the previous study's records), and determinism (a second
build with a different record chunking must be identical).  With ``--sample`` it also solves the exact constrained
optimum of a fixed stratified sample of <= 256 actual records per k (cover_reference: HiGHS MILP, networkx flow
cross-check, LP bound) and reports E(TopK) >= E(opt) >= E(repair) with the relative repair gaps.

    python tkc_construct.py --graph-seed 0 --out X.json [--sample 256]
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC

PREV = Path(__file__).resolve().parent.parent / "results" / "weighted_random_supports_anchors"
PREV_TOPK = {(8, 0): "w8_screen/cov_TopK_k8_g0.json", (8, 1): "w8_confirm/cov_TopK_k8_g1.json",
             (8, 2): "w8_confirm/cov_TopK_k8_g2.json", (16, 0): "m25_screen/cov_TopK_k16_g0.json"}
SAMPLE_SEED = 20261029


def prev_fingerprint(k, g):
    f = PREV_TOPK.get((k, g))
    if f is None or not (PREV / f).exists():
        return None
    return json.loads((PREV / f).read_text())["support_record"]["support_fingerprint"]


def dist(v):
    v = torch.as_tensor(v).double()
    if v.numel() == 0:
        return {}
    q = torch.quantile(v, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return dict(mean=float(v.mean()), min=float(v.min()), p5=float(q[0]), median=float(q[1]), p95=float(q[2]),
                max=float(v.max()), n=int(v.numel()))


def stratified(rel, full, repaired, n, seed):
    """<= n records: strata = (full / tail) x (unrepaired, repaired loss quartiles), equal allocation, seeded."""
    rng = np.random.default_rng(seed)
    strata = {}
    for fl in (True, False):
        base = np.flatnonzero((full == fl) & ~repaired)
        if base.size:
            strata[f"{'full' if fl else 'tail'}_unrepaired"] = base
        rp = np.flatnonzero((full == fl) & repaired)
        if rp.size:
            qs = np.quantile(rel[rp], [0.25, 0.5, 0.75])
            b = np.digitize(rel[rp], qs)
            for i in range(4):
                s = rp[b == i]
                if s.size:
                    strata[f"{'full' if fl else 'tail'}_loss_q{i + 1}"] = s
    per = max(1, n // max(1, len(strata)))
    pick = {}
    for name, s in strata.items():
        pick[name] = sorted(rng.choice(s, size=min(per, s.size), replace=False).tolist())
    return pick, {name: int(s.size) for name, s in strata.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--ks", default="8,16")
    a = ap.parse_args()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    import cover_reference as CR

    dev = torch.device("cuda")
    g = a.graph_seed
    L = KLayout(WC.D, WC.H, WC.G, 64, RC.C)
    counts = L.counts_tensor(dev)
    plan = ConnectivityPlan(WC.S, WC.R, seed=g)
    idx, feat, ubk, _ = KG.generate(WC.D, counts, RC.C, 64, L.blocks_per_group, 1, plan, dev)
    bk = KG.unit_to_block_buckets(ubk, counts, RC.C, 1)
    del idx
    rec, w, b, _ = SP.load_checkpoint(WC.ckpt(RC.C, g))
    if rec["spec"]["seed"] != g or rec["C"] != RC.C:
        raise SystemExit("checkpoint graph does not match")
    del b
    Q = int(feat.shape[0])
    res = dict(schema="granule.tkc_construct/1", graph_seed=g, checkpoint=dict(path=WC.ckpt(RC.C, g),
               weight_sha256=rec["weight_sha256"]), Q=Q, H=WC.H, provenance=provenance(Path(common.__file__).resolve().parent.parent))
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    a_all = LS.feature_importance(w, L)
    torch.cuda.synchronize()
    res["importance_seconds"] = time.perf_counter() - t0
    geo = LS.geometry(L, bk.cpu(), "cpu", Q=Q)
    a_v = a_all[geo["uv"]]
    del a_all
    res["geometry"] = dict(valid_groups=len(geo["uv"]), records=geo["records"],
                           groups_per_record_hist={int(k): int(v) for k, v in
                                                   zip(*torch.unique(geo["groups_per_record"], return_counts=True))})
    for k in [int(x) for x in a.ks.split(",")]:
        r = {}
        for pol in ("TopK", "Cover-TopK"):
            torch.cuda.reset_peak_memory_stats()
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            t0 = time.perf_counter()
            s = LS.build(pol, k, L, bk, w=w, graph_seed=g, device="cpu", Q=Q)
            sec = time.perf_counter() - t0
            st = {kk: v for kk, v in s.stats.items() if kk != "selection"}
            rc = LS.record_cover(s, L, bk, w, Q=Q)
            full = rc["groups"] == geo["gpb"]
            r[pol] = dict(build_seconds_total=sec, device_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          host_maxrss_growth_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0,
                          stats=st, support_fingerprint=s.fingerprint(), enabled_slots=s.enabled_slots(),
                          record_cover_mask=dict(all=dist(rc["mask"]), full=dist(rc["mask"][full]),
                                                 tail=dist(rc["mask"][~full]),
                                                 records_complete=int((rc["mask"] == 64).sum())),
                          record_cover_numerical=dict(all=dist(rc["numerical"]),
                                                      records_complete=int((rc["numerical"] == 64).sum()),
                                                      enabled_zero_weight_inputs=int((rc["mask"] - rc["numerical"]).sum())),
                          missing_inputs_total=int((64 - rc["mask"]).sum()))
            if pol == "TopK":
                tk = s
                pf = prev_fingerprint(k, g)
                r[pol]["previous_study_fingerprint"] = pf
                r[pol]["equal_to_previous_study"] = (pf == s.fingerprint()) if pf else None
            else:
                cv = s
        # determinism: a different record chunking and a second build reproduce the support exactly
        env_t = LS.topk_select(a_v, k)
        rr = LS.cover_repair(a_v, env_t, geo["rec"], geo["rank_rec"], k, geo["gpb"], chunk_records=4093)
        r["Cover-TopK"]["chunking_independent"] = bool(torch.equal(rr["env"], cv.env.cpu()[geo["uv"]]))
        if torch.cuda.is_available():
            t0 = time.perf_counter()
            rg = LS.cover_repair(a_v.cuda(), env_t.cuda(), geo["rec"], geo["rank_rec"], k, geo["gpb"])
            torch.cuda.synchronize()
            r["Cover-TopK"]["repair_seconds_on_gpu"] = time.perf_counter() - t0
            r["Cover-TopK"]["gpu_repair_identical"] = bool(torch.equal(rg["env"].cpu(), rr["env"]))
            del rg
        r["topk_groups_changed"] = int((cv.env.cpu()[geo["uv"]] != tk.env.cpu()[geo["uv"]]).any(1).sum())
        if a.sample:
            e_t = LS.subset_score(a_v, env_t)
            nrec = geo["records"]
            rec_t = torch.zeros(nrec, dtype=torch.float64).index_add_(0, geo["rec"], e_t).numpy()
            loss = rr["loss"].numpy()
            rel = np.where(rec_t > 0, loss / np.maximum(rec_t, 1e-300), 0.0)
            full = (rr["groups_per_record"] == geo["gpb"]).numpy()
            pick, sizes = stratified(rel, full, rr["swaps"].numpy() > 0, a.sample, SAMPLE_SEED + k)
            gidx = LS.record_groups(geo["rec"], geo["rank_rec"], geo["gpb"])
            rows, tm = [], []
            for name, recs in pick.items():
                for ri in recs:
                    gi = gidx[ri][gidx[ri] >= 0]
                    A = a_v[gi].numpy()
                    Xt = np.zeros(A.shape, dtype=bool)
                    Xt[np.repeat(np.arange(len(gi)), k), env_t[gi].numpy().reshape(-1)] = True
                    Xr = np.zeros(A.shape, dtype=bool)
                    Xr[np.repeat(np.arange(len(gi)), k), rr["env"][gi].numpy().reshape(-1)] = True
                    t1 = time.perf_counter()
                    mo = CR.milp_opt(A, k)
                    tm.append(time.perf_counter() - t1)
                    fo = CR.flow_opt(A, k)
                    et, er = CR.energy(A, Xt), CR.energy(A, Xr)
                    eo = mo.get("energy")
                    row = dict(record=int(ri), stratum=name, groups=int(len(gi)), swaps=int(rr["swaps"][ri]),
                               E_topk=et, E_repair=er, E_opt=eo, lp_bound=mo.get("lp_upper_bound"),
                               milp_certified=mo.get("certified"), milp_status=mo.get("status"),
                               milp_integrality=mo.get("max_integrality_violation"), flow_energy=fo["energy"],
                               flow_feasible=fo["feasible"], flow_minus_milp=(fo["energy"] - eo) if eo is not None else None,
                               repair_feasible=CR.feasible(Xr, k),
                               order_ok=(eo is not None and et >= eo - 1e-9 * abs(eo) and eo >= er - 1e-9 * abs(eo)),
                               repair_gap_abs=(eo - er) if eo is not None else None,
                               repair_gap_rel=((eo - er) / eo) if (eo is not None and eo > 0) else None,
                               repair_equal_opt_mask=bool(mo.get("certified") and np.array_equal(mo["X"], Xr)))
                    rows.append(row)
            rel_gaps = [x["repair_gap_rel"] for x in rows if x["repair_gap_rel"] is not None]
            r["exact_sample"] = dict(seed=SAMPLE_SEED + k, strata_sizes=sizes, picked={n: len(v) for n, v in pick.items()},
                                     rows=rows, milp_seconds=dist(tm), all_certified=all(x["milp_certified"] for x in rows),
                                     all_order_ok=all(x["order_ok"] for x in rows),
                                     repair_gap_rel=dist(rel_gaps) if rel_gaps else {},
                                     repair_gap_rel_by_stratum={n: dist([x["repair_gap_rel"] for x in rows
                                                                         if x["stratum"] == n and x["repair_gap_rel"] is not None])
                                                                for n in pick},
                                     repair_optimal_records=sum(x["repair_equal_opt_mask"] or (x["repair_gap_abs"] is not None
                                                                and abs(x["repair_gap_abs"]) <= 1e-12 * max(1.0, abs(x["E_opt"])))
                                                                for x in rows),
                                     full_model_solver_prediction_seconds=float(np.mean(tm)) * geo["records"],
                                     note="the sample is a STRATIFIED sample (equal allocation over strata), not a uniform "
                                          "sample of records: its gap summaries describe the strata, not the model")
        res[f"k{k}"] = r
        print(f"g{g} k{k}: TopK missing {r['TopK']['missing_inputs_total']} | Cover swaps "
              f"{r['Cover-TopK']['stats']['repair']['swaps']} loss rel {r['Cover-TopK']['stats']['repair']['energy_loss_relative_to_topk']:.4%} "
              f"repair {r['Cover-TopK']['stats']['repair']['seconds_repair']:.1f}s | prev-equal {r['TopK'].get('equal_to_previous_study')}"
              + (f" | sample gap median {r['exact_sample']['repair_gap_rel'].get('median')}" if a.sample else ""), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
