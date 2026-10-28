"""Mechanical application of the PREDECLARED selection rules of the K8 / weighted / anchor study
($GRANULE_RESULTS_ROOT/selection_rules.json, written before any result) to the measured records; every selection
file records its inputs.  Never reads a final-test value.

    python kwa_select.py k8_graph | w8_advance | w8_final | m25_advance | m25_final | final_methods
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import common  # noqa: F401
import kwa_cases as KW
import rsn_cases as RC
from k64_campaign import results_root


def load(stage: str, tag: str):
    f = results_root() / stage / f"{tag}.json"
    if not f.exists():
        raise SystemExit(f"missing record {f}")
    j = json.loads(f.read_text())
    if "final_test" in j:
        raise SystemExit(f"{f}: a final-test record must never enter a selection")
    return j


def learn_tag(spec, norm, steps, d, g=0):
    return f"learn_{RC.spec_id(spec)}_g{g}_{norm}_s{steps}_d{d}"


def dev_loss(stage, spec, norm, steps=256, seeds=KW.DEV_SEEDS, g=0):
    v = [load(stage, learn_tag(spec, norm, steps, d, g))["heldout_final"] for d in seeds]
    return statistics.mean(v), v


def treatment(stage, spec, steps=256):
    mn, vn = dev_loss(stage, spec, "none", steps)
    me, ve = dev_loss(stage, spec, "energy", steps)
    pick = "energy" if me < mn - 1e-4 else "none"
    return pick, dict(none=dict(mean=mn, seeds=vn), energy=dict(mean=me, seeds=ve))


def step_ms(stage, spec, fmt, g=0):
    j = load(stage, f"{RC.spec_id(spec)}_g{g}_{fmt}")
    return statistics.median(j["variants"]["default"]["samples_ms"]["step"])


def coverage(stage, spec, g=0):
    e = load(stage, f"cov_{RC.spec_id(spec)}_g{g}")["enabled"]
    return dict(layer_triple=e["triple"]["fraction"], per_output_triple=e["per_output_summary"]["triple"]["median"])


def q32(stage, spec):
    v = {n: load(stage, learn_tag(spec, n, 32, 0))["heldout_final"] for n in ("none", "energy")}
    return min(v.values()), v


def write(name, obj):
    f = results_root() / f"{name}_selection.json"
    f.write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1))


def k8_graph():
    bp, bpr = treatment("k8_q256", "BP-row/8")
    u8, u8r = treatment("k8_q256", "U8@0.125")
    fp, fpr = treatment("k8_q256", "FP-mag/8")
    write("k8_graph", dict(rule="selection_rules.json treatment_choice", bp_norm=bp, u8_norm=u8, fp_norm=fp,
                           inputs={"BP-row/8": bpr, "U8@0.125": u8r, "FP-mag/8": fpr}))


def w8_advance():
    base_t = step_ms("w8_screen", "BP-row/8#0", "indexed")
    base_c = coverage("k8_cov", "BP-row/8#0")
    rows = {}
    for s in ("WRS/8:4#0", "WRS/8:16#0", "TopK/8"):
        t = step_ms("w8_screen", s, "indexed")
        c = coverage("w8_screen", s)
        qs, qv = q32("w8_screen", s)
        pref = (t <= 1.10 * base_t and c["layer_triple"] >= 0.8 * base_c["layer_triple"]
                and c["per_output_triple"] >= 0.8 * base_c["per_output_triple"])
        rows[s] = dict(step_ms=t, step_ratio=t / base_t, coverage=c, layer_ratio=c["layer_triple"] / base_c["layer_triple"],
                       per_output_ratio=c["per_output_triple"] / base_c["per_output_triple"], q32=qv, score=qs,
                       preferred=pref)
    pref = sorted((r["score"], s) for s, r in rows.items() if r["preferred"])
    other = sorted((r["score"], s) for s, r in rows.items() if not r["preferred"])
    adv, labels = [], {}
    if pref:
        adv.append(pref[0][1])
        labels[pref[0][1]] = "strongest quality candidate in the preferred set"
    if other and (not pref or other[0][0] < pref[0][0]):
        adv.append(other[0][1])
        labels[other[0][1]] = "quality-versus-coverage tradeoff (outside the preferences)"
    bp_q, bp_qv = q32("k8_q32", "BP-row/8")
    write("w8_advance", dict(rule="selection_rules.json weighted_k8_screen", advance=adv, labels=labels,
                             baseline={"BP-row/8#0": dict(step_ms=base_t, coverage=base_c, q32=bp_qv)},
                             candidates=rows))


def w8_final():
    adv = json.loads((results_root() / "w8_advance_selection.json").read_text())["advance"]
    rows = {}
    for s in adv:
        t, r = treatment("w8_dev", s)
        rows[s] = dict(norm=t, mean=r[t]["mean"], inputs=r)
    best = min(rows, key=lambda s: rows[s]["mean"])
    rule = best.split("#")[0]
    layout_free = best.startswith("TopK")
    kpart = rule.split("/")[1] if "/" in rule else ""
    w16 = rule.replace("/8", "/16") + ("" if layout_free else "#0")
    mix = "Mix-TopK#0" if layout_free else f"Mix-WRS:{kpart.split(':')[1]}#0"
    bp = json.loads((results_root() / "k8_graph_selection.json").read_text())
    write("w8_final", dict(rule="selection_rules.json weighted_k8_screen (W*)", w_star=best, rule_spec=rule,
                           norm=rows[best]["norm"], layout_free=layout_free, bp_norm=bp["bp_norm"],
                           u8_norm=bp["u8_norm"], w16=w16, mix_w=mix, candidates=rows,
                           covered_g0=[best] if not layout_free else [rule]))


def m25_advance():
    sel = json.loads((results_root() / "w8_final_selection.json").read_text())
    w16, mixw = sel["w16"], sel["mix_w"]
    sc = {s: q32("m25_screen", s) for s in (w16, "Mix-BP#0", mixw)}
    mixed = sorted([(sc[s][0], s) for s in ("Mix-BP#0", mixw)])
    first = mixed[0][1]
    rest = sorted([(sc[s][0], s) for s in (mixed[1][1], w16)])
    adv = [first, rest[0][1]]
    fmts = {s: ("grouped" if s.startswith("Mix") else "indexed") for s in adv}
    write("m25_advance", dict(rule="selection_rules.json budget_25", advance=adv, formats=fmts,
                              scores={s: dict(score=v[0], q32=v[1]) for s, v in sc.items()},
                              controls={s: q32("m25_screen", s)[1] for s in ("BP-row#0", "A8@0.25", "U16@0.25")}))


def m25_final():
    a = json.loads((results_root() / "m25_advance_selection.json").read_text())
    rows = {}
    for s in a["advance"]:
        t, r = treatment("m25_dev", s)
        rows[s] = dict(norm=t, mean=r[t]["mean"], inputs=r)
    mixed = [s for s in a["advance"] if s.startswith("Mix")]
    best = min(mixed, key=lambda s: rows[s]["mean"])
    write("m25_final", dict(rule="selection_rules.json budget_25 (final mixed candidate)", mixed=best,
                            rule_spec=best.split("#")[0], fmt="grouped", norm=rows[best]["norm"], candidates=rows))


PARENT_NORM = Path(__file__).resolve().parent.parent / "results" / "random_supports_normalization_controls" / "norm"


def parent_treatments():
    """The treatment rule applied to the parent's 256-update records (seeds 0 / 1 / 2) of the 200M controls (reused,
    labelled: rsn_learn at the parent's frozen commits, same checkpoint / data / protocol)."""
    out, inputs = {}, {}
    for spec, tag in (("BP-row", "BP-row"), ("A8@0.25", "A8_r0.25"), ("U16@0.25", "U16_r0.25")):
        m = {}
        for n in ("none", "energy"):
            v = [json.loads((PARENT_NORM / f"learn_{tag}_{n}_s256_d{d}.json").read_text())["heldout_final"]
                 for d in KW.DEV_SEEDS]
            m[n] = dict(mean=statistics.mean(v), seeds=v)
        out[spec] = "energy" if m["energy"]["mean"] < m["none"]["mean"] - 1e-4 else "none"
        inputs[spec] = m
    obj = dict(out, rule="selection_rules.json treatment_choice on the parent's norm-stage records",
               source=str(PARENT_NORM), inputs=inputs)
    (results_root() / "parent_treatments.json").write_text(json.dumps(obj, indent=1))
    print(json.dumps(obj, indent=1))


def final_methods():
    w = json.loads((results_root() / "w8_final_selection.json").read_text())
    m = json.loads((results_root() / "m25_final_selection.json").read_text())
    k8 = json.loads((results_root() / "k8_graph_selection.json").read_text())
    ref = json.loads((results_root() / "parent_treatments.json").read_text())
    methods = [[w["w_star"], w["norm"]], ["BP-row/8#0", k8["bp_norm"]], ["U8@0.125", k8["u8_norm"]],
               [m["mixed"], m["norm"]], ["BP-row#0", ref["BP-row"]], ["A8@0.25", ref["A8@0.25"]],
               ["U16@0.25", ref["U16@0.25"]], ["dense", "none"]]
    write("final", dict(rule="selection_rules.json final_confirmation", methods=methods,
                        comparisons=[[w["w_star"], "BP-row/8#0"], [w["w_star"], "U8@0.125"], [m["mixed"], "A8@0.25"],
                                     [m["mixed"], "U16@0.25"], [m["mixed"], "BP-row#0"]],
                        margin=KW.NONINFERIORITY_MARGIN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["k8_graph", "w8_advance", "w8_final", "m25_advance", "m25_final", "final_methods",
                                     "parent_treatments"])
    a = ap.parse_args()
    globals()[a.what]()


if __name__ == "__main__":
    main()
