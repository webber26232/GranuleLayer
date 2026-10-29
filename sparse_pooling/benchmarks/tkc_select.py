"""Mechanical application of the PREREGISTERED rules of the TopK/16 / Cover-TopK round
($GRANULE_RESULTS_ROOT/selection_rules.json, written before any development result) to the measured records; every
selection file records its inputs.  Never reads a final-test value.

    python tkc_select.py continuity | treatments | final_methods | graph_methods | interaction_states
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import common  # noqa: F401
import rsn_cases as RC
import tkc_cases as TC
from k64_campaign import results_root

PREV = Path(__file__).resolve().parent.parent / "results" / "weighted_random_supports_anchors"
PREV_DEV = {"TopK/8": "w8_dev", "TopK/16": "m25_dev"}
PREV_Q32 = {"TopK/8": "w8_screen", "TopK/16": "m25_screen"}
CONTINUITY_TOL = 1e-3


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"missing record {path}")
    j = json.loads(path.read_text())
    if "final_test" in j:
        raise SystemExit(f"{path}: a final-test record must never enter a selection")
    return j


def tag(spec, norm, steps, d, g=0):
    return f"learn_{RC.spec_id(spec)}_g{g}_{norm}_s{steps}_d{d}"


def rules():
    return json.loads((results_root() / "selection_rules.json").read_text())


def write(name, obj):
    f = results_root() / f"{name}_selection.json"
    f.write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1))


def continuity():
    """TopK 32-update reruns at this round's code vs the previous round's records (same checkpoint / data / SR)."""
    rows, ok = {}, True
    for spec in ("TopK/8", "TopK/16"):
        for n in ("none", "energy"):
            new = load(results_root() / "dev32" / f"{tag(spec, n, 32, 0)}.json")
            old = load(PREV / PREV_Q32[spec] / f"{tag(spec, n, 32, 0)}.json")
            d = new["heldout_final"] - old["heldout_final"]
            same_mask = new["support_record"]["mask_sha256"] == old["support_record"]["mask_sha256"]
            rows[f"{spec} {n}"] = dict(new=new["heldout_final"], previous=old["heldout_final"], delta=d,
                                       same_mask=same_mask, same_checkpoint=new["checkpoint"]["weight_sha256"] ==
                                       old["checkpoint"]["weight_sha256"])
            ok &= abs(d) <= CONTINUITY_TOL and same_mask
    write("continuity", dict(rule=f"selection_rules.json continuity (|delta| <= {CONTINUITY_TOL}, same mask)",
                             reuse_previous_256=ok, rows=rows))


def dev_mean(spec, norm, src):
    v = [load(src / f"{tag(spec, norm, 256, d)}.json")["heldout_final"] for d in TC.DEV_SEEDS]
    return statistics.mean(v), v


def treatments():
    cont = json.loads((results_root() / "continuity_selection.json").read_text())
    out, inputs = {}, {}
    for spec in ("Cover-TopK/8", "Cover-TopK/16", "TopK/8", "TopK/16"):
        if spec.startswith("TopK"):
            src = PREV / PREV_DEV[spec] if cont["reuse_previous_256"] else results_root() / "dev256_topk"
        else:
            src = results_root() / "dev256"
        mn, vn = dev_mean(spec, "none", src)
        me, ve = dev_mean(spec, "energy", src)
        out[spec] = "energy" if me < mn - 1e-4 else "none"
        inputs[spec] = dict(source=str(src), none=dict(mean=mn, seeds=vn), energy=dict(mean=me, seeds=ve))
    write("treatments", dict(rule="selection_rules.json treatment_choice (development data only)", treatments=out,
                             inputs=inputs))


def final_methods():
    t = json.loads((results_root() / "treatments_selection.json").read_text())["treatments"]
    r = rules()
    fixed = r["fixed_treatments"]
    methods = []
    for s in TC.FINAL_METHODS:
        methods.append([s, t.get(s.split("#")[0], fixed.get(s))])
    if any(n is None for _, n in methods):
        raise SystemExit(f"a final method without a treatment: {methods}")
    write("final", dict(rule="selection_rules.json final_confirmation", methods=methods, test_protocol=TC.TEST_PROTOCOL,
                        seeds=list(TC.FINAL_SEEDS), primary=[list(p) for p in TC.PRIMARY],
                        secondary=r["secondary_comparisons"], margin=TC.NONINFERIORITY_MARGIN,
                        family_alpha=TC.FAMILY_ALPHA))


def graph_methods():
    m = dict(json.loads((results_root() / "final_selection.json").read_text())["methods"])
    specs = ("TopK/16", "Cover-TopK/16", "TopK/8", "Cover-TopK/8", "BP-row#0", "BP-row#1", "BP-row/8#0")
    norm = lambda s: m.get(s, m.get(s.split("#")[0] + "#0"))  # noqa: E731  BP layout 1 = layout 0's treatment
    write("graph", dict(rule="selection_rules.json graph_checks", methods=[[s, norm(s)] for s in specs],
                        data_seed=TC.GRAPH_CHECK_SEED, graphs=[1, 2]))


def interaction_states():
    d = results_root() / "interaction"
    states = {}
    for line in (d / "status.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["status"] != "measured" or not r["tag"].endswith(f"_s{TC.INTERACTION_STEPS}_lr{TC.INTERACTION_LR:g}"):
            continue
        j = json.loads((d / r["json"]).read_text())
        for s in j.get("saved_states", []):
            if s["step"] == TC.INTERACTION_STEPS:
                states[r["tag"]] = s["path"]
    write("interaction_cont", dict(rule="selection_rules.json interaction continuation (all three seeds of the K8 "
                                        "policies, no outcome-based choice)", states=states))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["continuity", "treatments", "final_methods", "graph_methods",
                                     "interaction_states"])
    a = ap.parse_args()
    globals()[a.what]()


if __name__ == "__main__":
    main()
