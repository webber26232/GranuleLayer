"""Tables of the width-preserving sparsity study (results/WIDTH_PRESERVING_SPARSITY.md) from the raw
records under a results root (stages: screen, inference, c2048, ablation, reject, confirm, winners,
coverage, finetune, coverage_ft, interaction, regen, confirm_s128).

    python wps_report.py --root RAW_DIR --out RAW_DIR/tables.md [--csv RAW_DIR/csv]

Every timing number is a median over the restored samples of one fresh process (screen: 8 = 2 rounds x 2 reps
x 2 steps; confirmation: 24 per seed); confirmation ratios are paired per graph seed, mean and t95 interval
over the 3 seeds (never over within-run samples).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def load(root: Path, stage: str):
    d = root / stage
    if not (d / "status.jsonl").exists():
        return [], {}
    rows = [json.loads(x) for x in (d / "status.jsonl").read_text().splitlines() if x.strip()]
    recs = {}
    for r in rows:
        if r.get("json") and (d / r["json"]).exists():
            recs[r["tag"]] = json.loads((d / r["json"]).read_text())
        elif r.get("json") and (d / (r["json"] + ".json")).exists():  # bench_supergroups appends .json
            recs[r["tag"]] = json.loads((d / (r["json"] + ".json")).read_text())
    return rows, recs


def bench_json(root, stage, tag):
    for p in (root / stage / f"{tag}.json.json", root / stage / f"{tag}.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def med(rec, ph="step", var="default"):
    v = rec["variants"][var]["phases"].get(ph)
    return None if v is None else v["median_ms"]


def fmt(x, nd=1, pct=False):
    if x is None:
        return "–"
    if pct:
        return f"{100 * x:.{nd}f}"
    if isinstance(x, (int,)) or (isinstance(x, float) and abs(x) >= 1000):
        return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"
    return f"{x:.{nd}f}"


def timing_rows(root, stage):
    rows, _ = load(root, stage)
    out = []
    for r in rows:
        j = r.get("job", {})
        rec = bench_json(root, stage, r["tag"]) if r["status"] == "measured" else None
        out.append(dict(tag=r["tag"], status=r["status"], policy=j.get("policy"), rho=j.get("rho"), score=j.get("score"),
                        format=(j.get("format") or "") + ("_ablated" if j.get("env") else ""), C=j.get("C"),
                        seed=j.get("graph_seed"), inference=j.get("inference"), ablated=bool(j.get("env")), rec=rec))
    return out


def key(r):
    return (r["C"], r["seed"], r["policy"], r["rho"], r["score"])


PH = ("fwd_layout", "fwd_kernel", "fwd_bias_only", "fwd_finalize", "fwd_total", "bwd_coef", "bwd_layout", "bwd_dx_init",
      "dx_kernel", "dw_kernel", "bwd_bias_only", "bwd_post", "bwd_total", "optimizer", "step")


def ident(rec):
    cs = (rec.get("kcsr") or {}).get("canonical_start") or {}
    return (cs.get("params", {}).get("sha"), cs.get("mask", {}).get("sha"), cs.get("opt_m", {}).get("sha"),
            cs.get("main_grad", {}).get("sha"))


def screen_tables(root, stage="screen", title="Screen"):
    rs = [r for r in timing_rows(root, stage) if r["rec"] is not None]
    by = defaultdict(dict)
    dense = []
    for r in rs:
        if r["policy"] == "dense":
            dense.append(r)
            continue
        by[key(r)][r["format"]] = r
    what = "forward pass only (no_grad)" if stage == "inference" else "complete training step"
    L = [f"### {title}: {what} per mask (ms, median of the restored samples)", "",
         "| mask | rho_mask | bias-only (mask) | wide | dense masked | wide vs dense masked | measured tile kernel | "
         "lowered K16/K8 | identical canonical start (params / mask) | peak GiB (wide) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for k in sorted(by, key=lambda k: (k[3], k[2], k[4])):
        f = by[k]
        w = f.get("wide")
        dm = f.get("dense_masked")
        if w is None and dm is None:
            continue
        ms = (w or dm)["rec"]["graph_stats"].get("mask_stats", {})
        sw = med(w["rec"]) if w else None
        sd = med(dm["rec"]) if dm else None
        st = med(f["tile"]["rec"]) if "tile" in f else None
        sl = med(f["lowered"]["rec"]) if "lowered" in f else None
        ids = {ident(x["rec"])[:2] for x in f.values()}
        same = "yes" if len(ids) == 1 else f"NO ({len(ids)})"
        nm = f"{k[2]} {k[3]:g}" + (" (random)" if k[4] == "random" else "") + ("" if k[0] == 256 else f" C{k[0]}")
        L.append(f"| {nm} | {ms.get('rho_mask', 0):.6f} | {100 * ms.get('structural_bias_only_neurons', 0) / 12.5e6:.1f} % | "
                 f"{fmt(sw)} | {fmt(sd)} | {('×%.2f' % (sd / sw)) if sw and sd else '–'} | {fmt(st)} | {fmt(sl)} | {same} | "
                 f"{(w or dm)['rec']['variants']['default']['peak_allocated_bytes'] / 2 ** 30:.2f} |")
    if dense:
        L += ["", "Unpruned dense control (same checkpoint, idf kernels): " +
              ", ".join(f"{d['tag']}: {fmt(med(d['rec']))} ms" for d in dense)]
    return "\n".join(L), by, dense


def ablation_table(root):
    rs = [r for r in timing_rows(root, "ablation") if r["rec"] is not None]
    if not rs:
        return ""
    by = defaultdict(dict)
    for r in rs:
        by[(r["policy"], r["rho"])][r["format"]] = r["rec"]
    L = ["### Backbone attribution: hyb16 vs the same layer with the backbone compiled out (ATTRIBUTION ONLY: the "
         "ablated build gives deliberately wrong results; every row group still runs)", "",
         "| mask | phase | hyb16 ms | ablated ms | backbone share ms | share of the phase |", "|---|---|---:|---:|---:|---:|"]
    for k in sorted(by, key=lambda k: (k[1], k[0])):
        a, b = by[k].get("wide"), by[k].get("wide_ablated")
        if not (a and b):
            continue
        for ph in ("fwd_kernel", "dx_kernel", "dw_kernel", "step"):
            x, y = med(a, ph), med(b, ph)
            L.append(f"| {k[0]} {k[1]:g} | {ph} | {fmt(x)} | {fmt(y)} | {fmt(x - y)} | {100 * (x - y) / x:.1f} % |")
    return "\n".join(L)


def phase_table(by, fmt_name="wide"):
    L = [f"### Per-phase medians ({fmt_name} format; ms)", "",
         "| mask | " + " | ".join(PH) + " | host enqueue | first-step wall s |",
         "|---|" + "---:|" * (len(PH) + 2)]
    for k in sorted(by, key=lambda k: (k[3], k[2], k[4])):
        r = by[k].get(fmt_name)
        if not r:
            continue
        rec = r["rec"]
        ph = [fmt(med(rec, p)) for p in PH]
        he = med(rec, "host_enqueue")
        cold = rec.get("cold_steps", {}).get("default", {}).get("first_step_wall_s")
        nm = f"{k[2]} {k[3]:g}" + (" (rnd)" if k[4] == "random" else "")
        L.append(f"| {nm} | " + " | ".join(ph) + f" | {fmt(he)} | {fmt(cold, 2)} |")
    return "\n".join(L)


def memory_table(by):
    L = ["### Memory and executed work (wide format; bytes in MB)", "",
         "| mask | peak alloc GiB | peak reserved GiB | compact values | sparse metadata | main_grad | Adam moments | "
         "work lists | fwd MMA products / token / W0 | backbone scalar FMA / step | executed fwd density |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(by, key=lambda k: (k[3], k[2], k[4])):
        r = by[k].get("wide")
        if not r:
            continue
        rec = r["rec"]
        v = rec["variants"]["default"]
        pb = rec["persistent_bytes"]
        ew = rec.get("executed_work") or {}
        mp = (ew.get("mma_products_per_token") or {}).get("fwd")
        nm = f"{k[2]} {k[3]:g}" + (" (rnd)" if k[4] == "random" else "")
        L.append(f"| {nm} | {v['peak_allocated_bytes'] / 2 ** 30:.2f} | {v['peak_reserved_bytes'] / 2 ** 30:.2f} | "
                 f"{pb.get('compact_values', 0) / 1e6:.0f} | {pb.get('sparse_metadata', 0) / 1e6:.1f} | "
                 f"{pb.get('main_grad', 0) / 1e6:.0f} | {pb.get('adam_moments', 0) / 1e6:.0f} | "
                 f"{pb.get('work_lists_cached_last_runtime', 0) / 1e6:.1f} | {fmt(mp / 8e8 if mp else None, 4)} | "
                 f"{fmt(((ew.get('scalar_fma') or {}).get('fwd', 0) + (ew.get('scalar_fma') or {}).get('dx', 0) + (ew.get('scalar_fma') or {}).get('dw', 0)) / 1e9, 1)}G | "
                 f"{fmt(ew.get('fwd_executed_density'), 4)} |")
    return "\n".join(L)


def setup_table(by, dense):
    """One-time setup of each compact model from the dense checkpoint and the break-even against the unpruned
    dense step: (select + convert + work lists) / (dense step - compact step); none if the saving <= 0."""
    ds = statistics.median([med(d["rec"]) for d in dense]) if dense else None
    L = ["### One-time setup and break-even (wide format; frozen-support training)", "",
         f"Unpruned dense step (median of the 3 anchors): {fmt(ds)} ms.", "",
         "| mask | checkpoint load + verify s | select s | compress s | work lists s | first-step (JIT) wall s | "
         "setup U s | per-step saving vs dense ms | break-even steps |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(by, key=lambda k: (k[3], k[2], k[4])):
        r = by[k].get("wide")
        if not r:
            continue
        rec = r["rec"]
        gs = rec["graph_stats"]
        wl = (rec["variants"]["default"].get("last_info") or {}).get("kgx_work_lists", {}).get("build_seconds", 0.0)
        U = gs.get("select_seconds", 0) + gs.get("conversion_seconds", 0) + wl
        sav = ds - med(rec) if ds else None
        be = (U * 1e3 / sav) if sav and sav > 0 else None
        nm = f"{k[2]} {k[3]:g}" + (" (rnd)" if k[4] == "random" else "")
        L.append(f"| {nm} | {gs['checkpoint']['load_seconds']:.1f} | {gs.get('select_seconds', 0):.3f} | "
                 f"{gs.get('conversion_seconds', 0):.3f} | {wl:.3f} | "
                 f"{rec['cold_steps']['default']['first_step_wall_s']:.2f} | {U:.3f} | {fmt(sav)} | "
                 f"{fmt(be, 2) if be is not None else 'none (no saving)'} |")
    return "\n".join(L)


def coverage_rows(root, stage="coverage"):
    rows, recs = load(root, stage)
    return recs


def coverage_table(recs, title="Coverage of the enabled supports (graph seed 0; exact pairs / triples, sampled quadruples)"):
    L = [f"### {title}", "",
         "| support | enabled density | bias-only mask / numeric | full-support neurons | layer pairs | triples | quadruples [95 % CI] | "
         "per-output features median | per-output triples median | per-output quads median | layer quad multiplicity median (covered) | "
         "per-output triple multiplicity: 0 / 1 / 2-15 / 16-63 / 64+ (median output) | anchors: every record full & nonzero | per-output = potential |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for tag in sorted(recs, key=lambda t: (recs[t].get("C", 0), (recs[t].get("width") or {}).get("enabled_slots", 8e8),
                                         t)):
        d = recs[tag]
        e = d.get("enabled") or d.get("dense")
        if e is None:
            continue
        w = d.get("width", {})
        q = e["quad"]
        po = e["per_output_summary"]
        lm = q.get("layer_multiplicity", {}).get("covered_quantiles", {})
        fixed = [r for r in e["outputs"] if r.get("fixed_selection", True)]
        cats = defaultdict(list)
        for r in fixed:
            for c, v in r.get("triple_multiplicity", {}).get("categories", {}).items():
                cats[c].append(v / r["triple_n"])
        cm = " / ".join(f"{100 * statistics.median(cats[c]):.3f}" for c in ("0", "1", "2-15", "16-63", "64+")) if cats else "–"
        an = d.get("anchors")
        av = d.get("anchor_vs_potential_per_output")
        dens = w.get("enabled_slots", 8e8) / 8e8
        L.append(f"| C{d.get('C')} {d.get('kind')} | {dens:.4f} | {100 * w.get('bias_only_mask_fraction', 0):.1f} / "
                 f"{100 * w.get('bias_only_numeric_fraction', 0):.1f} % | {fmt(w.get('full_support_neurons_mask'), 0)} | "
                 f"{100 * e['pair']['fraction']:.3f} % | {100 * e['triple']['fraction']:.3f} % | "
                 f"{100 * q['fraction']:.4f} [{100 * q['ci95'][0]:.4f}, {100 * q['ci95'][1]:.4f}] % | "
                 f"{po['features']['median']:.0f} | {100 * po['triple']['median']:.4f} % | {100 * po['quad']['median']:.5f} % | "
                 f"{lm.get('median', 0):.0f} | {cm} | "
                 f"{'–' if not an else ('yes' if an['all_records_have_full_mask_anchor'] and an['all_records_have_numerically_full_anchor'] else 'NO')} | "
                 f"{'–' if not av else ('yes' if av['all_equal'] else 'NO')} |")
    return "\n".join(L)


def mask_identity_check(root):
    """Coverage / fine-tune supports vs the timed masks: element-mask sha256 (graph seed 0)."""
    timed = {}
    for st in ("screen", "c2048"):
        for r in timing_rows(root, st):
            if r["rec"] is not None and r["policy"] != "dense":
                timed.setdefault((r["C"], r["policy"], r["rho"], r["score"]), set()).add(
                    r["rec"]["graph_stats"].get("mask_sha256"))
    out = []
    for st in ("coverage", "finetune"):
        _, recs = load(root, st)
        for tag, d in recs.items():
            if d.get("policy") in (None, "dense") or d.get("mask_file"):
                continue
            k = (d.get("C", 256), d["policy"], d["rho_target"], d.get("score", "magnitude"))
            t = timed.get(k)
            out.append(dict(stage=st, tag=tag, ok=bool(t) and t == {d.get("mask_sha256")}, timed=sorted(t or []),
                            analysed=d.get("mask_sha256")))
    return out


def finetune_table(root, stage="finetune"):
    _, recs = load(root, stage)
    if not recs:
        return "", {}
    dense = recs.get("ft_dense_C256", {})
    L = ["### Fixed-support fine-tuning (synthetic diagnostic; 32 updates, fresh optimizer state, no L1)", "",
         f"Unpruned dense control continued for the same updates: held-out {dense.get('heldout_checkpoint', 0):.5f} -> "
         f"{dense.get('heldout_finetuned', 0):.5f}.", "",
         "| support | held-out: checkpoint -> pruned -> fine-tuned | vs dense continued | sample: disconnected / connected-zero / never active / always active / switching | "
         "input variance median (connected) | activation fraction median | grad norm median | violations (protected / backbone) |",
         "|---|---|---:|---|---:|---:|---:|---|"]
    out = {}
    for tag in sorted(recs, key=lambda t: (recs[t].get("rho_target", 1), t)):
        d = recs[tag]
        a = d["diagnostics_after"]
        c = a["classes"]
        sa = d.get("support_after", {})
        iv = a.get("input_variance_connected") or a["input_variance"]
        af = a.get("activation_fraction_connected") or a["activation_fraction"]
        gn = a.get("grad_norm_connected") or a.get("grad_norm") or {}
        rel = d["heldout_finetuned"] / dense["heldout_finetuned"] if dense else None
        nm = f"{d['policy']} {d['rho_target']:g}" + (" (random)" if d.get("score") == "random" else "")
        out[(d["policy"], d["rho_target"], d.get("score"))] = d
        L.append(f"| {nm} | {d['heldout_checkpoint']:.5f} -> {d['heldout_pruned']:.5f} -> {d['heldout_finetuned']:.5f} | "
                 f"{('×%.4f' % rel) if rel else '–'} | {c['structurally_disconnected']} / {c['connected_all_zero']} / "
                 f"{c['never_active']} / {c['always_active']} / {c['switching']} | {iv['median']:.4g} | {af['median']:.3f} | "
                 f"{gn.get('median', 0):.3g} | {sa.get('protected_tile_violations', '–')} / {sa.get('backbone_violations', '–')} |")
    return "\n".join(L), out


def interaction_table(root, stage="interaction"):
    _, recs = load(root, stage)
    if not recs:
        return ""
    any_ = next(iter(recs.values()))
    tc = any_.get("target_constants", {})
    L = ["### Interaction diagnostic (reduced labelled configuration: H = 262,144, one C256 record per output; fresh init; "
         "1000 updates)", "",
         "Additive floors (the normalized MSE no support additive across the blocks can beat): " +
         ", ".join(f"order {k}: {v['additive_floor']:.4f}" for k, v in tc.items() if int(k) > 1), "",
         "| support | normalized held-out MSE: order 1 / 2 / 3 / 4 | explained interaction share: order 2 / 3 / 4 | "
         "neurons per output seeing their whole term: order 1 / 2 / 3 / 4 |", "|---|---|---|---|"]
    for tag in sorted(recs, key=lambda t: (recs[t].get("rho_target", 1), t)):
        d = recs[tag]
        f = d["final"]
        ns = d.get("neurons_seeing_their_whole_product") or {}
        nm = f"{d['policy']} {d['rho_target']:g}" + (" (random)" if d.get("score") == "random" else "")
        L.append(f"| {nm} | " + " / ".join(f"{f[f'order{o}']:.4f}" for o in (1, 2, 3, 4)) + " | " +
                 " / ".join(f"{f[f'explained_interaction_order{o}']:.3f}" for o in (2, 3, 4)) + " | " +
                 (" / ".join(f"{ns.get(f'order{o}', 0):.1f}" for o in (1, 2, 3, 4)) if ns else "256 (all)") + " |")
    return "\n".join(L)


def paired(root, stage, num_fmt, den_fmt):
    """{mask key without seed: [ratio per seed]} of den step / num step (paired per seed)."""
    rs = [r for r in timing_rows(root, stage) if r["rec"] is not None]
    by = defaultdict(dict)
    for r in rs:
        by[(r["C"], r["policy"], r["rho"], r["score"], r["seed"])][r["format"]] = r
    out = defaultdict(dict)
    for k, f in by.items():
        if num_fmt in f and den_fmt in f:
            out[k[:4]][k[4]] = med(f[den_fmt]["rec"]) / med(f[num_fmt]["rec"])
    return out


def ci(v):
    n = len(v)
    m = statistics.mean(v)
    if n < 2:
        return m, None
    h = T95[n - 1] * statistics.stdev(v) / math.sqrt(n)
    return m, h


def confirm_table(root, stage="confirm"):
    rs = [r for r in timing_rows(root, stage) if r["rec"] is not None]
    if not rs:
        return ""
    steps = defaultdict(lambda: defaultdict(dict))
    lists = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rs:
        lists[(r["policy"], r["rho"], r["score"])][r["format"]][r["seed"]].append(med(r["rec"]))
    for k, f in lists.items():  # the 3 dense anchors of a seed: their median
        for fm, ss in f.items():
            for sd, v in ss.items():
                steps[k][fm][sd] = statistics.median(v)
    L = ["### Confirmation (graph seeds 0 / 1 / 2, seed-specific checkpoints; 4 x 2 x 3 = 24 samples per seed)", "",
         "| mask | format | step per seed (ms) | mean | vs dense masked: ratio mean ± t95 | vs unpruned dense: ratio mean ± t95 |",
         "|---|---|---|---:|---:|---:|"]
    dense = steps.get(("dense", 1.0, "magnitude"), {}).get("dense", {})
    for k in sorted(steps, key=lambda k: (k[1], k[0])):
        for fm, ss in sorted(steps[k].items()):
            seeds = sorted(ss)
            dm = steps[k].get("dense_masked", {})
            r1 = [dm[s] / ss[s] for s in seeds if s in dm] if fm != "dense_masked" else []
            r2 = [dense[s] / ss[s] for s in seeds if s in dense]
            m1, h1 = ci(r1) if r1 else (None, None)
            m2, h2 = ci(r2) if r2 else (None, None)
            L.append(f"| {k[0]} {k[1]:g}{' (rnd)' if k[2] == 'random' else ''} | {fm} | " +
                     " / ".join(fmt(ss[s]) for s in seeds) + f" | {fmt(statistics.mean(ss.values()))} | " +
                     (f"×{m1:.3f} ± {h1:.3f}" if h1 is not None else (f"×{m1:.3f}" if m1 else "–")) + " | " +
                     (f"×{m2:.3f} ± {h2:.3f}" if h2 is not None else (f"×{m2:.3f}" if m2 else "–")) + " |")
    pairs = [(("P16", 0.25, "wide"), ("U16", 0.25, "wide")), (("P16", 0.25, "lowered"), ("P16", 0.25, "wide")),
             (("P16", 0.25, "tile"), ("P16", 0.25, "wide")), (("U16", 0.25, "tile"), ("U16", 0.25, "wide")),
             (("A8", 0.25, "wide"), ("U8", 0.25, "wide")), (("A8", 0.25, "wide"), ("U16", 0.25, "wide")),
             (("U8", 0.25, "wide"), ("U16", 0.25, "wide")), (("H2A", 0.10, "wide"), ("U16", 0.10, "wide")),
             (("H2A", 0.10, "wide"), ("H2", 0.10, "wide")), (("P8", 0.125, "lowered"), ("P8", 0.125, "wide")),
             (("P8", 0.125, "wide"), ("U8", 0.125, "wide")), (("A16", 0.375, "wide"), ("U16", 0.375, "wide")),
             (("A16", 0.375, "tile"), ("A16", 0.375, "wide"))]
    L += ["", "Paired step-time ratios (a / b per graph seed; mean ± t95 over the 3 seeds; > 1: a is slower):", "",
          "| a | b | per seed | mean ± t95 |", "|---|---|---|---:|"]
    for (pa, ra, fa), (pb, rb, fb) in pairs:
        A = steps.get((pa, ra, "magnitude"), {}).get(fa, {})
        Bv = steps.get((pb, rb, "magnitude"), {}).get(fb, {})
        sd = sorted(set(A) & set(Bv))
        if not sd:
            continue
        rr = [A[x] / Bv[x] for x in sd]
        m, h = ci(rr)
        L.append(f"| {pa} {ra:g} {fa} | {pb} {rb:g} {fb} | " + " / ".join(f"{v:.3f}" for v in rr) +
                 f" | {m:.3f}" + (f" ± {h:.3f}" if h is not None else "") + " |")
    return "\n".join(L)


def merged(root):
    """{(policy, rho, score): row} merging timing (screen), coverage, fine-tune and interaction (C256 seed 0)."""
    rows = defaultdict(dict)
    for r in timing_rows(root, "screen"):
        if r["rec"] is None or r["seed"] != 0:
            continue
        k = (r["policy"], r["rho"], r["score"])
        rows[k][f"step_{r['format']}"] = med(r["rec"])
        if r["format"] in ("wide", "dense"):
            gs = r["rec"]["graph_stats"]
            rows[k]["rho_mask"] = gs.get("rho_mask", 1.0)
    _, cov = load(root, "coverage")
    for tag, d in cov.items():
        if d.get("C") != 256:
            continue
        k = ("dense", 1.0, "magnitude") if d.get("policy") == "dense" else (d["policy"], d["rho_target"], d["score"])
        e = d.get("enabled") or d.get("dense")
        w = d.get("width") or dict(bias_only_mask_fraction=0.0, bias_only_numeric_fraction=0.0)
        fixed = [x for x in e["outputs"] if x.get("fixed_selection", True)]
        rows[k].update(triple=e["triple"]["fraction"], quad=e["quad"]["fraction"],
                       quad_mult=e["quad"].get("layer_multiplicity", {}).get("covered_quantiles", {}).get("median"),
                       out_triple=e["per_output_summary"]["triple"]["median"],
                       out_triple_mult=statistics.median(
                           [x["triple_multiplicity"]["covered_quantiles"].get("median", 0) for x in fixed]),
                       bias_only=w["bias_only_mask_fraction"], bias_only_num=w["bias_only_numeric_fraction"],
                       idep_med=(w.get("input_dependent_per_output") or {}).get("median", 12207))
    _, ft = load(root, "finetune")
    for tag, d in ft.items():
        k = ("dense", 1.0, "magnitude") if d["policy"] == "dense" else (d["policy"], d["rho_target"], d["score"])
        cl = d["diagnostics_after"]["classes"]
        rows[k].update(heldout=d["heldout_finetuned"], pruned=d["heldout_pruned"],
                       switching=cl["switching"] / d["diagnostics_after"]["sample"],
                       useful=d["diagnostics_after"]["variance_threshold_counts"]["0.01"] / d["diagnostics_after"]["sample"])
    _, it = load(root, "interaction")
    for tag, d in it.items():
        k = ("dense", 1.0, "magnitude") if d["policy"] == "dense" else (d["policy"], d["rho_target"], d["score"])
        rows[k].update({f"int{o}": d["final"][f"explained_interaction_order{o}"] for o in (2, 3, 4)})
    return rows


def policy_table(root):
    rows = merged(root)
    L = ["### Equal-budget policy comparison (C256 seed 0; same checkpoint, budget, data, optimizer; supports differ)", "",
         "| rho | policy | step ms (new kernels) | vs unpruned dense | bias-only rows (mask / numeric) | input-dependent neurons "
         "per output (median) | layer triples | layer quadruples | covered-quad multiplicity (median neurons) | "
         "per-output triples (median) | per-output covered-triple multiplicity (median) | held-out after fine-tune | "
         "switching neurons (sample) | interaction explained, order 2 / 3 / 4 |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    dense = rows.get(("dense", 1.0, "magnitude"), {})
    ds = dense.get("step_dense")
    for k in sorted(rows, key=lambda k: (k[1], k[0], k[2])):
        r = rows[k]
        st = r.get("step_wide") or r.get("step_dense")
        L.append(f"| {k[1]:g} | {k[0]}{' (random)' if k[2] == 'random' else ''} | {fmt(st)} | "
                 f"{('×%.2f' % (ds / st)) if st and ds else '–'} | "
                 f"{fmt(r.get('bias_only'), 1, pct=True)} / {fmt(r.get('bias_only_num'), 1, pct=True)} % | "
                 f"{fmt(r.get('idep_med'), 0)} | {fmt(r.get('triple'), 3, pct=True)} % | {fmt(r.get('quad'), 4, pct=True)} % | "
                 f"{fmt(r.get('quad_mult'), 0)} | {fmt(r.get('out_triple'), 4, pct=True)} % | {fmt(r.get('out_triple_mult'), 0)} | "
                 f"{fmt(r.get('heldout'), 5)} | {fmt(r.get('switching'), 3)} | " +
                 (" / ".join(fmt(r.get(f'int{o}'), 3) for o in (2, 3, 4)) if 'int2' in r else "–") + " |")
    return "\n".join(L), rows


PARETO_TOL = dict(step_rel=0.01, useful=0.01, triple=0.005, heldout=3e-4)


def pareto(rows, tol=PARETO_TOL):
    """Small Pareto set of the PRIMARY masks (random-score controls excluded) over: complete-step time (new
    kernels), USEFUL input-dependent width (fraction of the fixed neuron sample with input-driven variance > 0.01,
    ~5 % of the dense median), layer triple coverage, held-out loss after fine-tuning.  b dominates a when it is
    no worse than a beyond the tolerance in every objective and better beyond it in at least one (tolerances:
    1 % step time, 1 point width, 0.5 point triples, ``heldout`` = the measured run-to-run noise bound)."""
    pts = []
    for k, r in rows.items():
        st = r.get("step_wide") or r.get("step_dense")
        if k[2] == "random" or st is None or r.get("heldout") is None or r.get("triple") is None or r.get("useful") is None:
            continue
        pts.append((k, dict(step=st, useful=r["useful"], triple=r["triple"], heldout=r["heldout"])))

    def better(b, a):  # b better than a beyond tolerance, per objective
        return dict(step=b["step"] < a["step"] * (1 - tol["step_rel"]), useful=b["useful"] > a["useful"] + tol["useful"],
                    triple=b["triple"] > a["triple"] + tol["triple"], heldout=b["heldout"] < a["heldout"] - tol["heldout"])

    def worse(b, a):
        return dict(step=b["step"] > a["step"] * (1 + tol["step_rel"]), useful=b["useful"] < a["useful"] - tol["useful"],
                    triple=b["triple"] < a["triple"] - tol["triple"], heldout=b["heldout"] > a["heldout"] + tol["heldout"])

    keep = []
    for k, a in pts:
        dom = any(not any(worse(b, a).values()) and any(better(b, a).values()) for k2, b in pts if k2 != k)
        if not dom:
            keep.append(k)
    return sorted(keep, key=lambda k: (k[1], k[0]))


def frontier(rows, keep, tol=PARETO_TOL):
    """Non-dominated primary masks over (complete step, held-out loss) among those satisfying ``keep``."""
    pts = [(k, r) for k, r in rows.items() if k[2] != "random" and keep(r) and (r.get("step_wide") or r.get("step_dense"))
           and r.get("heldout") is not None]
    out = []
    for k, a in pts:
        sa = a.get("step_wide") or a.get("step_dense")
        dom = False
        for k2, b in pts:
            if k2 == k:
                continue
            sb = b.get("step_wide") or b.get("step_dense")
            no_worse = sb <= sa * (1 + tol["step_rel"]) and b["heldout"] <= a["heldout"] + tol["heldout"]
            better = sb < sa * (1 - tol["step_rel"]) or b["heldout"] < a["heldout"] - tol["heldout"]
            if no_worse and better:
                dom = True
                break
        if not dom:
            out.append(k)
    return sorted(out, key=lambda k: (k[1], k[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    parts = ["# Width-preserving sparsity: tables (generated by benchmarks/wps_report.py)", ""]
    t, by, dense = screen_tables(root)
    parts += [t, "", phase_table(by), "", phase_table(by, "dense_masked"), "", memory_table(by), "",
              setup_table(by, dense), ""]
    parts += [ablation_table(root), ""]
    for st, title in (("c2048", "C2048 low-density anchor test"), ("inference", "Inference (forward only)"),
                      ("winners", "Winner checks at C64 / C2048")):
        if (root / st).exists():
            t2, by2, d2 = screen_tables(root, st, title)
            parts += [t2, ""]
    cov = coverage_rows(root)
    if cov:
        parts += [coverage_table(cov), ""]
    covf = coverage_rows(root, "coverage_ft")
    if covf:
        parts += [coverage_table(covf, "Coverage of the exact nonzero supports after fine-tuning"), ""]
    ft, _ = finetune_table(root)
    parts += [ft, "", interaction_table(root), "", confirm_table(root), ""]
    chk = mask_identity_check(root)
    bad = [c for c in chk if not c["ok"]]
    parts += [f"Mask identity: {len(chk) - len(bad)} / {len(chk)} coverage / fine-tune supports have the element-mask "
              f"sha256 of the timed mask" + ("" if not bad else f"; MISMATCH: {[c['tag'] for c in bad]}"), ""]
    pt, rows = policy_table(root)
    nm = lambda ks: ", ".join(f"{k[0]} {k[1]:g}" for k in ks)  # noqa: E731
    dense = rows.get(("dense", 1.0, "magnitude"), {})
    full = lambda r: r.get("useful", 0) >= 0.95 and r.get("triple", 0) >= dense.get("triple", 1) - 0.001  # noqa: E731
    parts += [pt, "",
              f"Four-objective non-dominated set (complete step, useful width, layer triples, held-out; tolerances "
              f"{PARETO_TOL}): {nm(pareto(rows))}", "",
              "Small Pareto sets (complete step vs held-out loss after fine-tuning, same tolerances):", "",
              f"* unconstrained: {nm(frontier(rows, lambda r: True))}",
              f"* useful width >= 95 % of the neuron sample AND layer triples within 0.1 point of the dense potential: "
              f"{nm(frontier(rows, full))}",
              f"* rho <= 0.10 with every row connected (structural): "
              f"{nm(frontier(rows, lambda r: r.get('bias_only', 1) == 0 and r.get('rho_mask', 1) <= 0.1001))}", ""]
    Path(a.out).write_text("\n".join(parts) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
