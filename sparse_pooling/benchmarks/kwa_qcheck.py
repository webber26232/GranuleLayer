"""Erratum check of the input-set accounting (results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md section 2.2).

For the reduced interaction geometry (H 262,144, C256, graph seed 0, the parent's init seed 17) and the full-size
model (graph seeds 0 / 1 / 2, actual checkpoints): rebuild the parent's local supports with the AUTHORITATIVE Q
(sg_feat rows) and record total / occupied / empty sets from the actual buckets, the support fingerprints (they must
equal the parent's records: Q never shifts a q identity or RNG stream) and the parent's recorded Q / occupancy.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC

PARENT = Path(__file__).resolve().parent.parent / "results" / "random_supports_normalization_controls"


def stats_of(lay, spec, graph_seed):
    em, sup, srec = RC.support(spec, lay, graph_seed)
    st = srec["support_stats"]
    bk = lay.kg_buckets
    occ = int(torch.unique(bk[:, 0]).numel())
    return dict(spec=spec, Q_total=st["Q_total"], sets_occupied=st["sets_occupied"], sets_empty=st["sets_empty"],
                buckets_distinct_sets=occ, records=st["records"], support_fingerprint=st["support_fingerprint"],
                mask_sha256=srec["mask_sha256"], m_q_mean_all=st["m_q"]["mean"], m_q_mean_occupied=st["m_q"]["mean_occupied"],
                b_q_mean_all=st["b_q"]["mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental.kgx_layer import KgxLayer
    from sparse_pooling.supergroups import ConnectivityPlan
    from provenance import provenance

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    res = dict(schema="granule.kwa_qcheck/1", provenance=provenance(Path(common.__file__).resolve().parent.parent))
    # reduced interaction geometry
    lay = KgxLayer(WC.D, 262_144, WC.G, RC.C, ConnectivityPlan(WC.S, WC.R, seed=0), variant="idf", tile=128,
                   device=dev, seed=WC.INIT_SEED, keep_indices=False)
    red = [stats_of(lay, s, 0) for s in ("BP-row#0", "BP-row/8#0", "FP-mag")]
    parent = {}
    for f in sorted(glob.glob(str(PARENT / "interaction" / "int_*_T16_none_i17_s1000.json"))):
        j = json.loads(Path(f).read_text())
        st = (j.get("support_record") or {}).get("support_stats")
        if st:
            parent[Path(f).name] = dict(Q=st.get("Q"), sets_occupied=st.get("sets_occupied"),
                                        support_fingerprint=st.get("support_fingerprint"),
                                        occupied_sets_field=j.get("occupied_sets"))
    for r in red:
        for name, p in parent.items():
            if p["support_fingerprint"] == r["support_fingerprint"]:
                r["parent_record"] = dict(p, file=name)
    res["reduced_H262144"] = dict(H=262_144, Q_total=int(lay.sg_feat.shape[0]), supports=red, parent_records=parent)
    del lay
    torch.cuda.empty_cache()
    full = []
    for g in (0, 1, 2):
        rec, w, b, _ = SP.load_checkpoint(WC.ckpt(RC.C, g))
        lay = KgxLayer(WC.D, WC.H, WC.G, RC.C, ConnectivityPlan(WC.S, WC.R, seed=g), variant="idf", tile=128,
                       device=dev, weights=(w, b), keep_indices=False)
        del w, b
        for s in ("BP-row#0", "BP-row#1", "BP-row/8#0", "BP-row/8#1"):
            r = stats_of(lay, s, g)
            r["graph_seed"] = g
            cands = [PARENT / d / f"cov_{RC.spec_id(s)}_g{g}.json" for d in ("coverage", "coverage_confirm", "k8")]
            pf = next((c for c in cands if c.exists()), cands[0])
            if pf.exists():
                pj = json.loads(pf.read_text())
                ps = pj["support_record"]["support_stats"]
                r["parent_record"] = dict(file=str(pf.name), Q=ps.get("Q"), sets_occupied=ps.get("sets_occupied"),
                                          support_fingerprint_equal=ps["support_fingerprint"] == r["support_fingerprint"],
                                          mask_sha256_equal=pj["support_record"]["mask_sha256"] == r["mask_sha256"])
            full.append(r)
        del lay
        torch.cuda.empty_cache()
    res["full_size"] = full
    res["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for r in red:
        print(f"reduced {r['spec']}: Q {r['Q_total']} occupied {r['sets_occupied']} empty {r['sets_empty']} "
              f"(parent {r.get('parent_record', {}).get('Q')})", flush=True)
    for r in full:
        print(f"g{r['graph_seed']} {r['spec']}: Q {r['Q_total']} occupied {r['sets_occupied']} "
              f"parent-equal {r.get('parent_record', {}).get('support_fingerprint_equal')}", flush=True)


if __name__ == "__main__":
    main()
