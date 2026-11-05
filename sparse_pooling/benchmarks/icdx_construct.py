"""Construction of the interaction-coverage candidates on the actual alpha-0 checkpoints
(results/INTERACTION_COVERAGE_K16_DX.md section 3): Balanced-TopK/16 (quota-layout seeds 0 / 1), Cover-Mix32:p1 / p2
(class-layout seeds 0 / 1) and the controls Cover-TopK/16 / BP-row/16 (layout 0), per graph seed.

Recorded per support: CPU build wall time (geometry + importance + selection + repair; FP64 on the host), host peak
RSS (sampled from /proc/self/status every 2 ms; peak minus the pre-build baseline), device peak allocation, statistics,
fingerprints, budgets, record coverage (mask and numerical); the SAME repairs rerun on the GPU with the wall time taken
around a full device synchronization and the number of host-read stop checks (each a host / device synchronization);
chunking independence (a different record chunk size) and GPU = CPU bitwise.  The old policies' fingerprints are
compared with the previous round's construction records.  With ``--sample N`` (graph 0) the exact Balanced quota
optimum (HiGHS MILP, networkx flow cross-check, LP bound) of a stratified full / tail sample of <= N records.

    python icdx_construct.py --graph-seed 0 --out X.json [--sample 128]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC

PREV = Path(__file__).resolve().parent.parent / "results" / "topk16_coverage_constrained_topk"
SAMPLE_SEED = 20261105


class RssPeak:
    """Peak resident set size (bytes) of this process between start() and stop(), sampled every ``dt`` s."""

    def __init__(self, dt=0.002):
        self.dt, self.peak, self._run = dt, 0, False

    @staticmethod
    def rss():
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) * 1024
        return 0

    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.rss())
            time.sleep(self.dt)

    def __enter__(self):
        self.base = self.rss()
        self.peak = self.base
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self._t.join()
        self.peak = max(self.peak, self.rss())

    def growth(self):
        return self.peak - self.base


def dist(v):
    v = torch.as_tensor(v).double().flatten()
    if v.numel() == 0:
        return {}
    q = torch.quantile(v[: 1 << 24], torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64))
    return dict(mean=float(v.mean()), sd=float(v.std()) if v.numel() > 1 else 0.0, min=float(v.min()), p1=float(q[0]),
                p5=float(q[1]), median=float(q[2]), p95=float(q[3]), p99=float(q[4]), max=float(v.max()),
                n=int(v.numel()))


def prev_cover_fingerprint(g):
    f = PREV / "construct_quiet" / f"construct_g{g}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["k16"]["Cover-TopK"]["support_fingerprint"]


def multiplicity(gm_v, nv_v, geo):
    from sparse_pooling.experimental import local_supports as LS

    m = LS.input_multiplicity(gm_v, nv_v, geo["rec"], geo["records"])
    full = geo["groups_per_record"] == geo["gpb"]
    out = {}
    for name, sel in (("all", torch.ones_like(full)), ("full", full), ("tail", ~full)):
        if not bool(sel.any()):
            continue
        G_, N_ = m["groups"][sel].double(), m["neurons"][sel].double()
        out[name] = dict(records=int(sel.sum()), groups=dist(G_), neurons=dist(N_),
                         per_record_spread_groups=dist(G_.max(1).values - G_.min(1).values),
                         per_record_spread_neurons=dist(N_.max(1).values - N_.min(1).values),
                         per_record_cv_neurons=dist(N_.std(1) / N_.mean(1).clamp_min(1e-30)),
                         records_complete=int((G_ > 0).all(1).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import local_supports as LS
    from sparse_pooling.experimental import mix32_supports as M32
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
    res = dict(schema="granule.icdx_construct/1", graph_seed=g, checkpoint=dict(path=WC.ckpt(RC.C, g),
               weight_sha256=rec["weight_sha256"]), Q=Q, H=WC.H,
               provenance=provenance(Path(common.__file__).resolve().parent.parent),
               timing_note="CPU builds: wall time of local_supports.build / mix32_supports.build (geometry, FP64 "
                           "importance on the device then host, selection, repairs on the host); GPU repairs: wall time "
                           "between two torch.cuda.synchronize() calls, i.e. including every synchronization; "
                           "preprocessing outside every training step")
    geo = LS.geometry(L, bk.cpu(), "cpu", Q=Q)
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    a_all = LS.feature_importance(w, L)
    torch.cuda.synchronize()
    res["importance_seconds"] = time.perf_counter() - t0
    a_v = a_all[geo["uv"]]
    del a_all
    nv_v = geo["nv"][geo["uv"]]
    res["geometry"] = dict(valid_groups=len(geo["uv"]), records=geo["records"],
                           groups_per_record_hist={int(k): int(v) for k, v in
                                                   zip(*torch.unique(geo["groups_per_record"], return_counts=True))},
                           partial_groups=int(((geo["nv"] > 0) & (geo["nv"] < 16)).sum()))
    sups = {}
    specs = ["Cover-TopK/16", "BP-row#0", "Balanced-TopK/16#0", "Balanced-TopK/16#1", "Cover-Mix32:p1#0",
             "Cover-Mix32:p1#1", "Cover-Mix32:p2#0", "Cover-Mix32:p2#1"]
    holder = type("H", (), {})()
    holder.weight, holder.layout, holder.kg_buckets, holder.sg_feat = w, L, bk, feat
    for spec in specs:
        c = RC.parse(spec)
        torch.cuda.reset_peak_memory_stats()
        with RssPeak() as rp:
            t0 = time.perf_counter()
            if c["kind"] == "mix32":
                s = M32.build(c["policy"], c["p"], L, bk, w, graph_seed=g, layout_seed=c["layout_seed"], device="cpu",
                              Q=Q)
            else:
                s = LS.build(c["policy"], c["k"], L, bk, w=w, graph_seed=g, layout_seed=c["layout_seed"],
                             device="cpu", Q=Q)
            sec = time.perf_counter() - t0
        gm = s.group_mask().cpu()[geo["uv"]]
        rc = LS.record_cover(s, L, bk, w, Q=Q) if hasattr(s, "env") else None
        if rc is None:  # mixed: the same record coverage from the group masks (mask) and the weights (numerical)
            n16 = s.nv.numel()
            valid = L.valid_mask(w.device).reshape(n16, 16)
            nz = ((w.reshape(n16, 16, 64) != 0) & valid[:, :, None]).any(1).cpu()[geo["uv"]] & gm
            mk = torch.zeros((geo["records"], 64), dtype=torch.int64).index_add_(0, geo["rec"], gm.long()) > 0
            nm = torch.zeros((geo["records"], 64), dtype=torch.int64).index_add_(0, geo["rec"], nz.long()) > 0
            rc = dict(mask=mk.sum(1), numerical=nm.sum(1), groups=geo["groups_per_record"])
        st = {k: v for k, v in s.stats.items() if k not in ("selection",)}
        sups[spec] = s
        r = dict(spec=spec, build_seconds_cpu=sec, host_peak_rss_growth_bytes=rp.growth(),
                 host_peak_rss_bytes=rp.peak, device_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 support_fingerprint=s.fingerprint(), enabled_slots=s.enabled_slots(), budget_ok=s.enabled_slots() ==
                 16 * WC.H, stats=st, multiplicity=multiplicity(gm, nv_v, geo),
                 energy_layer_fraction=float((a_v * gm).sum() / a_v.sum()),
                 record_cover=dict(mask_complete=int((rc["mask"] == 64).sum()),
                                   numerical_complete=int((rc["numerical"] == 64).sum()),
                                   enabled_zero_weight_inputs=int((rc["mask"] - rc["numerical"]).sum())))
        if spec == "Cover-TopK/16":
            pf = prev_cover_fingerprint(g)
            r["previous_round_fingerprint"] = pf
            r["equal_to_previous_round"] = (pf == s.fingerprint()) if pf else None
        res[spec] = r
        print(f"g{g} {spec}: {sec:.1f}s CPU, energy {r['energy_layer_fraction']:.5f}, complete "
              f"{r['record_cover']['mask_complete']}, budget {r['budget_ok']}, rss +{rp.growth() / 2 ** 20:.0f} MiB",
              flush=True)
    # ---- GPU repairs (timed with synchronization), determinism checks
    cov_env = sups["Cover-TopK/16"].env.cpu()[geo["uv"]]
    for ls in (0, 1):
        spec = f"Balanced-TopK/16#{ls}"
        qt = LS.balanced_quotas(geo["groups_per_record"], 16, ("Balanced-TopK", 16, f"graph{g}", f"layout{ls}"))
        ref = sups[spec].env.cpu()[geo["uv"]]
        t0 = time.perf_counter()
        rr = LS.quota_repair(a_v, cov_env, geo["rec"], geo["rank_rec"], 16, geo["gpb"], qt, chunk_records=4093)
        res[spec]["chunking_independent"] = bool(torch.equal(rr["env"], ref))
        res[spec]["cpu_repair_seconds_chunk4093"] = time.perf_counter() - t0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rg = LS.quota_repair(a_v.cuda(), cov_env.cuda(), geo["rec"], geo["rank_rec"], 16, geo["gpb"], qt)
        torch.cuda.synchronize()
        res[spec]["gpu_repair"] = dict(seconds_incl_sync=time.perf_counter() - t0, host_syncs=rg["host_syncs"],
                                       rounds=rg["rounds"], device_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       identical_to_cpu=bool(torch.equal(rg["env"].cpu(), ref)))
        del rg
    for spec in ("Cover-Mix32:p1#0", "Cover-Mix32:p2#0", "Cover-Mix32:p1#1", "Cover-Mix32:p2#1"):
        s = sups[spec]
        kc_v = s.kc.cpu()[geo["uv"]]
        X0 = torch.zeros((len(geo["uv"]), 64), dtype=torch.bool)
        for k in M32.CLASSES:
            sel = kc_v == k
            X0[torch.nonzero(sel).squeeze(1)[:, None], LS.topk_select(a_v[sel], k)] = True
        ref = s.gm.cpu()[geo["uv"]]
        rr = LS.cover_repair_mask(a_v, X0, geo["rec"], geo["rank_rec"], geo["gpb"], chunk_records=4093)
        res[spec]["chunking_independent"] = bool(torch.equal(rr["X"], ref))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rg = LS.cover_repair_mask(a_v.cuda(), X0.cuda(), geo["rec"], geo["rank_rec"], geo["gpb"])
        torch.cuda.synchronize()
        res[spec]["gpu_repair"] = dict(seconds_incl_sync=time.perf_counter() - t0, rounds=rg["rounds"],
                                       host_syncs_note="one host read per swap round (dynamic active set) plus a "
                                                       "finiteness check per round: >= 2 synchronizations per round",
                                       device_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       identical_to_cpu=bool(torch.equal(rg["X"].cpu(), ref)))
        del rg
    # generalized cover repair = the original cover_repair on uniform degrees (Cover-TopK/16 reproduced)
    et = LS.topk_select(a_v, 16)
    X0 = torch.zeros((len(geo["uv"]), 64), dtype=torch.bool).scatter_(1, et, True)
    rm = LS.cover_repair_mask(a_v, X0, geo["rec"], geo["rank_rec"], geo["gpb"])
    res["cover_repair_mask_reproduces_cover_topk16"] = bool(torch.equal(
        rm["X"], torch.zeros_like(X0).scatter_(1, cov_env, True)))
    # ---- exact Balanced quota optimum on a stratified full / tail sample (graph 0)
    if a.sample:
        rng = np.random.default_rng(SAMPLE_SEED + g)
        full = (geo["groups_per_record"] == geo["gpb"]).numpy()
        pick = {}
        for name, sel in (("full", np.flatnonzero(full)), ("tail", np.flatnonzero(~full))):
            pick[name] = sorted(rng.choice(sel, size=min(a.sample // 2, sel.size), replace=False).tolist())
        gidx = LS.record_groups(geo["rec"], geo["rank_rec"], geo["gpb"])
        qt0 = LS.balanced_quotas(geo["groups_per_record"], 16, ("Balanced-TopK", 16, f"graph{g}", "layout0"))
        bal = sups["Balanced-TopK/16#0"].env.cpu()[geo["uv"]]
        bp = sups["BP-row#0"].env.cpu()[geo["uv"]]
        rows, tm = [], []
        for name, recs in pick.items():
            for ri in recs:
                gi = gidx[ri][gidx[ri] >= 0]
                A = a_v[gi].numpy()

                def mask(env):
                    X = np.zeros(A.shape, dtype=bool)
                    X[np.repeat(np.arange(len(gi)), 16), env[gi].numpy().reshape(-1)] = True
                    return X

                Xb, Xc, Xp = mask(bal), mask(cov_env), mask(bp)
                q = qt0[ri].numpy()
                t1 = time.perf_counter()
                mo = CR.quota_milp_opt(A, 16, q)
                tm.append(time.perf_counter() - t1)
                fo = CR.quota_flow_opt(A, 16, q)
                eb, ec, ep = CR.energy(A, Xb), CR.energy(A, Xc), CR.energy(A, Xp)
                eo = mo.get("energy")
                rows.append(dict(record=int(ri), stratum=name, groups=int(len(gi)), E_balanced=eb, E_cover_topk=ec,
                                 E_bp_row=ep, E_opt=eo, lp_bound=mo.get("lp_upper_bound"),
                                 milp_certified=mo.get("certified"), milp_integrality=mo.get("max_integrality_violation"),
                                 flow_energy=fo["energy"], flow_feasible=fo["feasible"],
                                 flow_minus_milp=(fo["energy"] - eo) if eo is not None else None,
                                 repair_feasible=CR.quota_feasible(Xb, 16, q),
                                 order_ok=eo is not None and eo >= eb - 1e-9 * abs(eo),
                                 repair_gap_rel=((eo - eb) / eo) if (eo is not None and eo > 0) else None,
                                 repair_equal_opt_mask=bool(mo.get("certified") and np.array_equal(mo["X"], Xb)),
                                 balanced_vs_cover=(eb / ec) if ec > 0 else None,
                                 balanced_vs_bp=(eb / ep) if ep > 0 else None,
                                 opt_vs_cover=(eo / ec) if (eo is not None and ec > 0) else None))
        gaps = [x["repair_gap_rel"] for x in rows if x["repair_gap_rel"] is not None]
        res["exact_sample"] = dict(seed=SAMPLE_SEED + g, picked={n: len(v) for n, v in pick.items()}, rows=rows,
                                   milp_seconds=dist(tm), all_certified=all(x["milp_certified"] for x in rows),
                                   all_order_ok=all(x["order_ok"] for x in rows),
                                   all_repair_feasible=all(x["repair_feasible"] for x in rows),
                                   repair_gap_rel=dist(gaps),
                                   repair_gap_rel_by_stratum={n: dist([x["repair_gap_rel"] for x in rows
                                                                       if x["stratum"] == n and x["repair_gap_rel"]
                                                                       is not None]) for n in pick},
                                   balanced_vs_cover=dist([x["balanced_vs_cover"] for x in rows]),
                                   balanced_vs_bp=dist([x["balanced_vs_bp"] for x in rows]),
                                   opt_vs_cover=dist([x["opt_vs_cover"] for x in rows if x["opt_vs_cover"] is not None]),
                                   repair_optimal_records=sum(x["repair_equal_opt_mask"] for x in rows),
                                   note="stratified equal-allocation sample (full / tail), not a uniform sample; the "
                                        "solver is a reference, not a full-model solver")
        print(f"g{g} exact sample: gap median {res['exact_sample']['repair_gap_rel'].get('median')} p95 "
              f"{res['exact_sample']['repair_gap_rel'].get('p95')} certified {res['exact_sample']['all_certified']}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
