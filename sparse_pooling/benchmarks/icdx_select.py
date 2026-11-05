"""Mechanical, versioned selections of the interaction-coverage / K16 dX round (results/INTERACTION_COVERAGE_K16_DX.md
section 6).  Every selection file records its rule, its input records (file names and sha256) and its time.

    python icdx_select.py rules               -> selection_rules.json (preregistration, before any learning job)
    python icdx_select.py continuity          -> continuity_selection.json (dev32 controls vs earlier records)
    python icdx_select.py treatments          -> treatments_selection.json (none / energy per new support, Mix32 p)
    python icdx_select.py interaction_states  -> interaction_states_selection.json (1000-update state paths)
    python icdx_select.py dx_screen           -> confirm_selection.json (<= 2 finalists + selected Mix32)
    python icdx_select.py backend             -> backend_selection.json (chosen backend or legacy)
    python icdx_select.py final               -> final_selection.json, continuation_selection.json (freeze)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path

import common  # noqa: F401
import icdx_cases as IC
import paired_stats as PS
import rsn_cases as RC
from k64_campaign import results_root
from kwa_report import Root, step_samples

REPO_RESULTS = Path(__file__).resolve().parent.parent / "results"
RULES = dict(
    continuity="for each control: the 32-update seed-0 run (its frozen treatment) must have the SAME mask sha256 and "
               "checkpoint weight sha256 as the earlier record and |held-out difference| <= 1e-3 at every common "
               "evaluation step; pass -> earlier 256-update development records (if any) are reused, fail -> rerun",
    treatment="per new support: 'energy' iff mean_3(none) - mean_3(energy) > 1e-4 on validation 9_000..9_003 after 256 "
              "updates (seeds 0 / 1 / 2, all present exactly once), else 'none'",
    mixed="among Cover-Mix32 p1 / p2 with their frozen treatments: the lower mean 256-update validation loss; if the "
          "means differ by <= 1e-4, p1",
    balanced="Balanced-TopK/16 (quota layout 0) is a required final candidate regardless of development",
    dx_screen="per tiling (graph 0, 8 samples per process): median complete step (ms).  Eligible: legal launch, "
              "correctness gates passed (dX test file), executed variant = requested, Cover-TopK/16 M 8192 faster than "
              "legacy, BP-row/16 M 8192 <= 1.03 x legacy, Cover-TopK/16 M 512 <= 1.03 x legacy.  Finalists: the <= 2 "
              "eligible tilings of lowest Cover-TopK/16 M 8192 complete step (dX time is not the criterion)",
    backend="graph-mean (graphs 0 / 1 / 2) complete step at M 8192: a finalist qualifies iff Cover-TopK/16 <= 0.95 x "
            "legacy, BP-row/16 and the selected Mix32 <= 1.03 x legacy, and Cover-TopK/16 M 512 <= 1.03 x legacy (and "
            "every correctness gate passed); the qualifying finalist of largest Cover-TopK/16 gain is chosen; none -> "
            "legacy (negative result reported)",
    continuation="Cover-TopK/16, Balanced-TopK/16#0, the selected Mix32 and BP-row/16 continue 1000 -> 4000 updates "
                 "(all three init / SR pairs)",
    freeze="treatments, mixed candidate, continuation list and backend are frozen in final_selection.json BEFORE the "
           "icdx39000 / icdx89000 tests are opened")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name: str, obj: dict, inputs=()):
    root = results_root()
    obj = dict(obj, created=datetime.datetime.now().isoformat(timespec="seconds"),
               inputs={str(p.relative_to(root) if str(p).startswith(str(root)) else p): sha(p) for p in inputs})
    (root / name).write_text(json.dumps(obj, indent=1))
    print(json.dumps({k: v for k, v in obj.items() if k != "inputs"}, indent=1)[:3000])


def learn_records(stage, steps):
    r = Root(results_root())
    return [(j, results_root() / stage / f) for j, f in r.records(stage, lambda j: j.get("steps") == steps)]


def cmd_rules():
    write("selection_rules.json", dict(schema="granule.icdx_rules/1", rules=RULES, manifest=IC.manifest(),
                                       seed_collisions=IC.seed_collisions()))


def cmd_continuity():
    recs = {j["support"]: (j, p) for j, p in learn_records("dev32", 32)}
    out, inputs = {}, []
    for s in IC.CONTROLS:
        j, p = recs[s]
        ref_p = REPO_RESULTS / IC.CONTINUITY_REFERENCE[s]
        ref = json.loads(ref_p.read_text())
        inputs += [p, ref_p]
        ev = {e["step"]: e["heldout"] for e in j["evals"]}
        er = {e["step"]: e["heldout"] for e in ref["evals"]}
        common_steps = sorted(set(ev) & set(er))
        diff = max(abs(ev[t] - er[t]) for t in common_steps)
        same_mask = (j["support_record"].get("mask_sha256") == ref["support_record"].get("mask_sha256"))
        same_ck = j["checkpoint"]["weight_sha256"] == ref["checkpoint"]["weight_sha256"]
        ok = same_mask and same_ck and diff <= 1e-3 and j["norm"] == ref["norm"]
        out[s] = dict(pass_=ok, max_heldout_difference=diff, steps=common_steps, same_mask=same_mask,
                      same_checkpoint=same_ck, reusable_dev256=s in IC.REUSABLE_DEV256)
    rerun = [s for s in IC.REUSABLE_DEV256 if not out[s]["pass_"]]
    write("continuity_selection.json", dict(rule=RULES["continuity"], controls=out, rerun=rerun,
                                            reuse=[s for s in IC.REUSABLE_DEV256 if s not in rerun]), inputs)


def dev_cohorts():
    recs = learn_records("dev256", 256)
    cohorts, inputs = {}, []
    for j, p in recs:
        c = cohorts.setdefault((j["support"], j["norm"]), PS.Cohort(f"{j['support']} {j['norm']}"))
        c.add(PS.key_of_learn_record(j, protocol_field="__validation__"), j["heldout_final"], p.name)
        inputs.append(p)
    return cohorts, inputs


def cmd_treatments():
    cohorts, inputs = dev_cohorts()
    declared = [(d, d) for d in IC.DEV_SEEDS]
    tr, rows = {}, {}
    for s in IC.NEW:
        m = {}
        for n in ("none", "energy"):
            c = cohorts.get((s, n))
            iv = PS.inventory(c, declared) if c else dict(complete=False)
            if not iv["complete"]:
                raise SystemExit(f"treatment selection: {s} {n} incomplete ({iv})")
            m[n] = statistics.mean(c.obs[k] for k in declared)
        tr[s] = "energy" if m["none"] - m["energy"] > 1e-4 else "none"
        rows[s] = dict(mean_none=m["none"], mean_energy=m["energy"], selected=tr[s])
    p1, p2 = IC.MIXED
    m1, m2 = rows[p1][f"mean_{tr[p1]}"], rows[p2][f"mean_{tr[p2]}"]
    mixed = p2 if m1 - m2 > 1e-4 else p1
    tr.update(IC.CONTROL_TREATMENT)
    write("treatments_selection.json", dict(rules=dict(treatment=RULES["treatment"], mixed=RULES["mixed"]),
                                            treatments=tr, rows=rows, mixed=mixed, mixed_means=dict(p1=m1, p2=m2)),
          inputs)


def cmd_interaction_states():
    root = results_root()
    states, inputs = {}, []
    for f in sorted((root / "interaction").glob("int_*.json")):
        j = json.loads(f.read_text())
        st = [s for s in j.get("saved_states", []) if s["step"] == IC.INTERACTION_STEPS]
        if st:
            states[f.stem] = st[0]["path"]
            inputs.append(f)
    want = {f"int_{RC.spec_id(s)}_T16_i{i}_sr{sr}_s{IC.INTERACTION_STEPS}_lr{IC.INTERACTION_LR:g}" for s in IC.CORE8
            for i, sr in IC.INTERACTION_INIT_SR}
    missing = sorted(want - set(states))
    write("interaction_states_selection.json", dict(states=states, missing=missing, complete=not missing), inputs)
    if missing:
        sys.exit(1)


def timing_table(stage):
    """{(spec, fmt, dxv, batch, graph): [records]} of the measured timing jobs of a stage, with the executed dX variant
    checked against the request."""
    r = Root(results_root())
    out, inputs, bad = {}, [], []
    for s in r.status(stage):
        if s["kind"] != "timing" or s["status"] != "measured":
            if s["kind"] == "timing":
                bad.append((s["tag"], s["status"]))
            continue
        jb = s["job"]
        j = r.get(stage, s["tag"])
        req = jb.get("dx_variant", "legacy")
        got = (j.get("graph_stats") or {}).get("dx_variant", {}) if jb["support"] != "dense" else {}
        if jb["support"] != "dense" and got.get("request", "legacy") != req:
            raise SystemExit(f"{s['tag']}: executed dX request {got} != {req}")
        out.setdefault((jb["support"], jb["format"], req, jb.get("batch", 8192), jb["graph_seed"]), []).append(j)
        inputs.append(results_root() / stage / f"{s['tag']}.json")
    return out, inputs, bad


def med_step(js):
    return statistics.mean(statistics.median(step_samples(j)["step"]) for j in js)


def cmd_dx_screen():
    tab, inputs, bad = timing_table("dx_screen")
    corr = results_root() / "dx_correctness.txt"
    corr_ok = corr.exists() and " passed" in corr.read_text() and " failed" not in corr.read_text()
    leg = {k: med_step(tab[("Cover-TopK/16", "indexed", "legacy", 8192, 0)]) for k in [0]}[0]
    leg_bp = med_step(tab[("BP-row#0", "indexed", "legacy", 8192, 0)])
    leg512 = med_step(tab[("Cover-TopK/16", "indexed", "legacy", 512, 0)])
    rows = {}
    for (spec, fmt, v, B, g) in sorted(tab):
        if spec != "Cover-TopK/16" or B != 8192 or v == "legacy":
            continue
        c = med_step(tab[(spec, fmt, v, B, g)])
        bp = med_step(tab[("BP-row#0", "indexed", v, 8192, 0)]) if ("BP-row#0", "indexed", v, 8192, 0) in tab else None
        c512 = med_step(tab[("Cover-TopK/16", "indexed", v, 512, 0)]) if ("Cover-TopK/16", "indexed", v, 512, 0) in tab \
            else None
        el = bool(corr_ok and c < leg and bp is not None and bp <= 1.03 * leg_bp and c512 is not None and
                  c512 <= 1.03 * leg512)
        rows[v] = dict(cover_m8192=c, cover_ratio=c / leg, bp_m8192=bp, bp_ratio=None if bp is None else bp / leg_bp,
                       cover_m512=c512, cover512_ratio=None if c512 is None else c512 / leg512, eligible=el)
    fin = sorted([v for v in rows if rows[v]["eligible"]], key=lambda v: rows[v]["cover_m8192"])[:2]
    tr = json.loads((results_root() / "treatments_selection.json").read_text())
    write("confirm_selection.json", dict(rule=RULES["dx_screen"], legacy=dict(cover_m8192=leg, bp_m8192=leg_bp,
                                                                              cover_m512=leg512),
                                         tilings=rows, finalists=fin, mixed=tr["mixed"], correctness_gates=corr_ok,
                                         failed_jobs=bad), inputs + ([corr] if corr.exists() else []))


def cmd_backend():
    tab, inputs, bad = timing_table("confirm")
    sel = json.loads((results_root() / "confirm_selection.json").read_text())
    mixed = sel["mixed"]

    def gmean(spec, fmt, v, B=8192, graphs=IC.GRAPH_SEEDS):
        vals = [med_step(tab[(spec, fmt, v, B, g)]) for g in graphs if (spec, fmt, v, B, g) in tab]
        return statistics.mean(vals) if len(vals) == len(graphs) else None

    rows = {}
    for v in sel["finalists"]:
        r = {}
        for spec, fmt in (("Cover-TopK/16", "indexed"), ("Balanced-TopK/16#0", "indexed"), ("BP-row#0", "indexed"),
                          (mixed, "grouped")):
            a_, b_ = gmean(spec, fmt, v), gmean(spec, fmt, "legacy")
            r[spec] = dict(variant=a_, legacy=b_, ratio=None if a_ is None or b_ is None else a_ / b_)
        a5, b5 = gmean("Cover-TopK/16", "indexed", v, 512, (0,)), gmean("Cover-TopK/16", "indexed", "legacy", 512, (0,))
        r["Cover-TopK/16@M512"] = dict(variant=a5, legacy=b5, ratio=None if a5 is None or b5 is None else a5 / b5)
        q = (sel["correctness_gates"] and all(x["ratio"] is not None for x in r.values()) and
             r["Cover-TopK/16"]["ratio"] <= 0.95 and r["BP-row#0"]["ratio"] <= 1.03 and r[mixed]["ratio"] <= 1.03 and
             r["Cover-TopK/16@M512"]["ratio"] <= 1.03)
        rows[v] = dict(r, qualifies=bool(q))
    ok = sorted([v for v in rows if rows[v]["qualifies"]], key=lambda v: rows[v]["Cover-TopK/16"]["ratio"])
    chosen = ok[0] if ok else "legacy"
    write("backend_selection.json", dict(rule=RULES["backend"], finalists=sel["finalists"], rows=rows, chosen=chosen,
                                         mixed=mixed, failed_jobs=bad), inputs)


def cmd_final():
    root = results_root()
    tr = json.loads((root / "treatments_selection.json").read_text())
    be = json.loads((root / "backend_selection.json").read_text())
    st = json.loads((root / "interaction_states_selection.json").read_text())
    mixed = tr["mixed"]
    methods = [[s, tr["treatments"][s]] for s in ("Cover-TopK/16", "Balanced-TopK/16#0", mixed, "U16@0.25",
                                                   "Mix-TopK#0", "dense")]
    cont = ["Cover-TopK/16", "Balanced-TopK/16#0", mixed, "BP-row#0"]
    primary = [["Balanced-TopK/16#0", "Cover-TopK/16"], [mixed, "Cover-TopK/16"], [mixed, "Mix-TopK#0"],
               [f"Cover-TopK/16@indexed:{be['chosen']}", "Cover-TopK/16@indexed:legacy"]]
    sel = dict(rule=RULES["freeze"], methods=methods, treatments=tr["treatments"], mixed=mixed,
               backend=dict(chosen=be["chosen"], qualifies=be["chosen"] != "legacy"), continued=cont,
               seeds=list(IC.FINAL_SEEDS), margin=IC.MARGIN, family_alpha=IC.FAMILY_ALPHA, family_size=IC.FAMILY_SIZE,
               primary=primary, contrast4_estimable=be["chosen"] != "legacy",
               exploratory=[[s, "U16@0.25"] for s in ("Cover-TopK/16", "Balanced-TopK/16#0", mixed)] +
               [["Cover-TopK/16@indexed:legacy", "Cover-TopK/16"]])
    write("final_selection.json", sel, [root / "treatments_selection.json", root / "backend_selection.json"])
    write("continuation_selection.json", dict(states=st["states"], continued=cont),
          [root / "interaction_states_selection.json", root / "final_selection.json"])


if __name__ == "__main__":
    dict(rules=cmd_rules, continuity=cmd_continuity, treatments=cmd_treatments,
         interaction_states=cmd_interaction_states, dx_screen=cmd_dx_screen, backend=cmd_backend,
         final=cmd_final)[sys.argv[1]]()
