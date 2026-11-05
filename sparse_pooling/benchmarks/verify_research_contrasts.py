"""Recompute the research final cohort's four primary contrasts (interaction-coverage-k16-dx f4c538a,
results/INTERACTION_COVERAGE_K16_DX.md section 7) with the CORRECTED paired_stats (integration review 5.3: declared
experiment / method identities instead of the first record's) and check that the numbers are unchanged.

    python verify_research_contrasts.py --final-dir <research results root>/final --out verify.json \
        [--committed-csv final_primary.csv]

The declaration comes from the research round's frozen selection (final_selection.json: methods, seeds, primary
contrasts, family size 4, alpha 0.05, margin 0.003) and its construction manifest (support fingerprints of graph 0,
results/interaction_coverage_k16_dx/csv/construct.csv), copied below -- never from the records being checked.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import paired_stats as PS

EXPERIMENT = PS.Experiment(graph=0, protocol="icdx39000", steps=256)
SEEDS = [(s, s) for s in range(12, 18)]
FAMILY, ALPHA, MARGIN = 4, 0.05, 0.003
# research construct.csv (graph 0) support fingerprints (12-hex prefixes as published there)
FINGERPRINTS = {"Cover-TopK/16": "e4465520b9cc", "Balanced-TopK/16#0": "6f87528dd874", "Cover-Mix32:p1#0": "59c0732d8498"}
METHODS = {  # name -> (record filter: support spec, norm, schema prefix, backend variant), declared Method
    "Cover-TopK/16": ("Cover-TopK/16", "energy", "granule.rsn_learn/", None),
    "Balanced-TopK/16#0": ("Balanced-TopK/16#0", "energy", "granule.rsn_learn/", None),
    "Cover-Mix32:p1#0": ("Cover-Mix32:p1#0", "energy", "granule.rsn_learn/", None),
    "U16@0.25": ("U16@0.25", "energy", "granule.rsn_learn/", None),
    "Mix-TopK#0": ("Mix-TopK#0", "energy", "granule.rsn_learn/", None),
    "dense": ("dense", "none", "granule.rsn_learn/", None),
    "Cover-TopK/16@indexed:legacy": ("Cover-TopK/16", "energy", "granule.icdx_backend_train/", "legacy"),
    "Cover-TopK/16@indexed:mt4_nw8_wg4": ("Cover-TopK/16", "energy", "granule.icdx_backend_train/", "mt4_nw8_wg4"),
}
PRIMARY = [("Balanced-TopK/16#0", "Cover-TopK/16"), ("Cover-Mix32:p1#0", "Cover-TopK/16"),
           ("Cover-Mix32:p1#0", "Mix-TopK#0"), ("Cover-TopK/16@indexed:mt4_nw8_wg4", "Cover-TopK/16@indexed:legacy")]


def method_of(name: str) -> PS.Method:
    spec, norm, _, be = METHODS[name]
    fp = FINGERPRINTS.get(spec)
    return PS.Method(support=spec + (f"@{fp}" if fp else ""), treatment=norm, backend=be or "dense-masked")


def key_of(j: dict, spec: str) -> PS.ObsKey:
    fp = FINGERPRINTS.get(spec)
    sup = spec
    if fp:
        got = (j.get("support_record") or {}).get("support_fingerprint") or ""
        sup = spec + "@" + got[:12]  # the record's own fingerprint: must equal the declared one (else WrongIdentity)
    be = (j.get("backend") or {}).get("variant") or "dense-masked"
    return PS.ObsKey(int(j["graph_seed"]), int(j["data_seed"]), int(j["sr_seed"]), sup, str(j["norm"]),
                     str(j.get("final_test_set")), int(j["steps"]), be)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--committed-csv", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [(json.loads(Path(f).read_text()), Path(f).name) for f in sorted(glob.glob(str(Path(a.final_dir) / "*.json")))
            if not f.endswith(("manifest.json", "plan.json"))]
    cohorts = {}
    for name, (spec, norm, schema, be) in METHODS.items():
        c = PS.Cohort(name, EXPERIMENT, method_of(name))
        for j, f in recs:
            if not str(j.get("schema", "")).startswith(schema) or j.get("support") != spec or j.get("norm") != norm:
                continue
            if be is not None and (j.get("backend") or {}).get("variant") != be:
                continue
            if j.get("graph_seed") != 0:
                continue  # graph-1 / 2 checks are a separate declared experiment
            c.add(key_of(j, spec), j["final_test"], f)
        cohorts[name] = c
    out = dict(schema="granule.verify_research_contrasts/1", experiment=EXPERIMENT.__dict__, seeds=SEEDS,
               inventories={n: PS.inventory(c, SEEDS) for n, c in cohorts.items()}, contrasts=[])
    for x, y in PRIMARY:
        r = PS.paired(cohorts[x], cohorts[y], SEEDS, experiment=EXPERIMENT, method_a=method_of(x), method_b=method_of(y),
                      family_size=FAMILY, alpha=ALPHA, margin=MARGIN)
        out["contrasts"].append(dict(contrast=f"{x} - {y}", status=r["status"], mean=r.get("mean"), upper=r.get("upper"),
                                     verdict=r.get("verdict"), n=r.get("n"), df=r.get("df"),
                                     pointwise95=r["descriptive"]["pointwise95"]))
    if a.committed_csv:
        rows = list(csv.reader(open(a.committed_csv)))[1:]
        cmp = []
        for row, c in zip(rows, out["contrasts"]):
            lo, hi = c["pointwise95"]
            mine = [c["contrast"], f"{c['mean']:+.6f}", f"{c['upper']:+.6f}", f"[{lo:+.6f}, {hi:+.6f}]", c["verdict"]]
            theirs = [row[0], row[2], row[3], row[4], row[5]]
            cmp.append(dict(contrast=row[0], recomputed=mine, committed=theirs, equal=mine == theirs))
        out["committed_comparison"] = cmp
        out["unchanged"] = all(x["equal"] for x in cmp)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(contrasts=out["contrasts"], unchanged=out.get("unchanged")), indent=1))


if __name__ == "__main__":
    main()
