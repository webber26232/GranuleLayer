"""Tables of the K8 / weighted / anchor study (results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md) from the raw records under
$GRANULE_RESULTS_ROOT (or --root): writes tables.md and csv/*.csv.  Sections whose records are missing are skipped
with a note (never filled with placeholders).

    python kwa_report.py [--root DIR] [--parent DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import common  # noqa: F401
import rsn_cases as RC

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def ci(v):
    """(mean, half-width of the 95 % t interval) of a list (half-width None for n < 2)."""
    m = statistics.mean(v)
    if len(v) < 2:
        return m, None
    return m, T95[len(v) - 1] * statistics.stdev(v) / math.sqrt(len(v))


def fmt_ci(v, nd=4, scale=1.0):
    m, h = ci([x * scale for x in v])
    return f"{m:.{nd}f}" + ("" if h is None else f" ± {h:.{nd}f}")


class Root:
    def __init__(self, root: Path):
        self.root = root

    def get(self, stage, tag):
        f = (self.root / stage / f"{tag}.json") if stage else (self.root / f"{tag}.json")
        return json.loads(f.read_text()) if f.exists() else None

    def status(self, stage):
        f = self.root / stage / "status.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def step_samples(j):
    return j["variants"]["default"]["samples_ms"]


def med(j, ph="step"):
    s = step_samples(j).get(ph) or []
    return statistics.median(s) if s else float("nan")


def cov_row(j):
    e = j["enabled"]
    outs = [o for o in e["outputs"] if o.get("fixed_selection")]
    th, tn = sum(o["triple_hits"] for o in outs), sum(o["triple_n"] for o in outs)
    from kcsr_coverage import clopper_pearson

    lo, hi = clopper_pearson(th, tn)
    return dict(pair=e["pair"]["fraction"], triple=e["triple"]["fraction"], quad=e["quad"]["fraction"],
                quad_ci=e["quad"]["ci95"], po_triple_median=e["per_output_summary"]["triple"]["median"],
                po_triple_pooled=th / tn, po_triple_ci=[lo, hi], po_quad_median=e["per_output_summary"]["quad"]["median"],
                po_features=e["per_output_summary"]["features"]["median"], outputs=len(outs))


class Report:
    def __init__(self, root, parent):
        self.r, self.parent = Root(root), Root(parent)
        self.md, self.csvs = [], {}

    def h(self, t):
        self.md += ["", f"## {t}", ""]

    def table(self, name, header, rows):
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
        for r in rows:
            self.md.append("| " + " | ".join(str(x) for x in r) + " |")
        self.md.append("")
        self.csvs[name] = (header, rows)

    # ---------------------------------------------------------------- 2. errata
    def errata(self):
        self.h("Erratum: global-support sharing (graph 0, parent K16 supports)")
        rows = []
        pdir = {"FP-mag": "coverage", "BP-row#0": "coverage", "U16@0.25": "coverage", "A8@0.25": "coverage"}
        for s in ("FP-mag", "BP-row#0", "U16@0.25", "A8@0.25"):
            new = self.r.get("errata_cov", f"cov_{RC.spec_id(s)}_g0")
            old = self.parent.get(pdir[s], f"cov_{RC.spec_id(s)}_g0")
            if new is None:
                continue
            sh = new["sharing"]
            same_mask = old is not None and old["support_record"]["mask_sha256"] == new["support_record"]["mask_sha256"]
            same_cov = old is not None and all(old["enabled"][t]["covered"] == new["enabled"][t]["covered"]
                                              for t in ("pair", "triple"))
            for sc in ("record", "set", "output", "layer"):
                o = (old or {}).get("sharing", {}).get(sc, {}).get("distinct_supports")
                rows.append([s, sc, o, sh[sc]["legacy_local_key"], sh[sc]["labelled_local"], sh[sc]["distinct_supports"],
                             f"{sh[sc]['groups_per_distinct_support']['mean']:.3f}",
                             f"{sh[sc]['neurons_per_distinct_support']['mean']:.2f}",
                             "yes" if same_mask else "NO", "yes" if same_cov else "NO"])
        self.table("errata_sharing", ["support", "scope", "parent (local key)", "legacy key (recomputed)",
                                      "labelled (q, local)", "global distinct", "groups / support", "neurons / support",
                                      "mask sha equal", "exact pair / triple equal"], rows)
        q = self.r.get("qcheck", "qcheck")
        if q:
            self.h("Erratum: total Q and occupied sets")
            rows = []
            for x in q["reduced_H262144"]["supports"]:
                p = x.get("parent_record") or {}
                rows.append(["H 262,144 (g0)", x["spec"], p.get("Q"), x["Q_total"], x["sets_occupied"], x["sets_empty"],
                             x["buckets_distinct_sets"], p.get("support_fingerprint") == x["support_fingerprint"]
                             if p else "n/a"])
            for x in q["full_size"]:
                p = x.get("parent_record") or {}
                rows.append([f"H 12.5M (g{x['graph_seed']})", x["spec"], p.get("Q"), x["Q_total"], x["sets_occupied"],
                             x["sets_empty"], x["buckets_distinct_sets"], p.get("support_fingerprint_equal", "n/a")])
            self.table("errata_q", ["model", "support", "parent Q", "total Q", "occupied", "empty", "sets in buckets",
                                    "support fingerprint unchanged"], rows)

    # ---------------------------------------------------------------- timing sessions
    def timing(self, stage, title, ref="U8@0.125", ref_fmt="wide"):
        st = [s for s in self.r.status(stage) if s["kind"] == "timing"]
        if not st:
            return
        self.h(title)
        by = {}
        for s in st:
            if s["status"] != "measured":
                by.setdefault((s["job"]["support"], s["job"]["format"]), {})[s["job"]["graph_seed"]] = s["status"]
                continue
            j = self.r.get(stage, s["tag"])
            key = (s["job"]["support"].split("#")[0], s["job"]["format"])
            by.setdefault(key, {}).setdefault(s["job"]["graph_seed"], []).append(dict(
                step=med(j), fwd=med(j, "fwd_total"), dx=med(j, "dx_kernel"), dw=med(j, "dw_kernel"),
                opt=med(j, "optimizer"), peak=j["variants"]["default"]["peak_allocated_bytes"] / 2 ** 30,
                n=len(step_samples(j)["step"]), spread=max(step_samples(j)["step"]) - min(step_samples(j)["step"])))
        graphs = sorted({g for v in by.values() for g in (v if isinstance(v, dict) else {})})
        refm = {g: statistics.mean(x["step"] for x in by.get((ref, ref_fmt), {}).get(g, [])) for g in graphs
                if isinstance(by.get((ref, ref_fmt), {}).get(g), list)}
        rows = []
        for (spec, fmt), v in sorted(by.items()):
            gm = {g: statistics.mean(x["step"] for x in xs) for g, xs in v.items() if isinstance(xs, list)}
            if not gm:
                rows.append([spec, fmt, "; ".join(f"g{g}: {x}" for g, x in v.items())] + [""] * 8)
                continue
            lay = [max(x["step"] for x in xs) - min(x["step"] for x in xs) for xs in v.values() if isinstance(xs, list)]
            ratios = [gm[g] / refm[g] for g in gm if g in refm]
            allx = [x for xs in v.values() if isinstance(xs, list) for x in xs]
            rows.append([spec, fmt, " / ".join(f"{gm[g]:.1f}" for g in sorted(gm)), fmt_ci(list(gm.values()), 1),
                         f"{max(lay):.1f}", f"{statistics.mean(x['spread'] for x in allx):.1f}",
                         fmt_ci(ratios, 3) if ratios else "",
                         f"{statistics.mean(x['fwd'] for x in allx):.1f}", f"{statistics.mean(x['dx'] for x in allx):.1f}",
                         f"{statistics.mean(x['dw'] for x in allx):.1f}", f"{max(x['peak'] for x in allx):.2f}"])
        self.table(stage + "_timing", ["support", "format", "step ms per graph (layout-averaged)",
                                       "graph mean ± 95 % CI", "max layout spread", "mean within-run range",
                                       f"ratio to {ref} (same session)", "fwd", "dX", "dW / db", "peak GiB"], rows)

    def coverage(self, stage, title, specs_graphs):
        rows = []
        for s, g, st in specs_graphs:
            j = self.r.get(st or stage, f"cov_{RC.spec_id(s)}_g{g}")
            if j is None:
                continue
            c = cov_row(j)
            sh = j.get("sharing") or {}
            pr = (j.get("predictions") or {}).get("refined", {}).get("t3")
            rows.append([s, g, f"{100 * c['pair']:.3f}", f"{100 * c['triple']:.3f}",
                         "" if pr is None else f"{100 * pr:.3f}",
                         f"{100 * c['quad']:.4f} [{100 * c['quad_ci'][0]:.4f}, {100 * c['quad_ci'][1]:.4f}]",
                         f"{100 * c['po_triple_median']:.4f}",
                         f"{100 * c['po_triple_pooled']:.4f} [{100 * c['po_triple_ci'][0]:.4f}, {100 * c['po_triple_ci'][1]:.4f}]",
                         f"{c['po_features']:.0f}",
                         (sh.get("output") or {}).get("distinct_supports", ""), (sh.get("layer") or {}).get("distinct_supports", ""),
                         f"{100 * j.get('width', {}).get('bias_only_mask_fraction', 0):.2f} %"])
        if rows:
            self.h(title)
            self.table(stage + "_coverage", ["support", "graph", "layer pairs %", "layer triples % (exact)",
                                             "refined prediction %", "layer quads % [95 % CI]",
                                             "per-output triples % (median of 38)",
                                             "per-output triples % pooled [95 % CI]", "per-output features",
                                             "distinct supports (output scope, sum)", "distinct supports (layer)",
                                             "structural bias-only"], rows)

    # ---------------------------------------------------------------- learning
    def learn(self, stage, title, specs, steps, seeds=(0, 1, 2), norms=("none", "energy"), g=0, parent_stage=None):
        rows = []
        for s in specs:
            for n in norms:
                vals, clip, rv, pr = [], [], [], []
                for d in seeds:
                    tag = f"learn_{RC.spec_id(s)}_g{g}_{n}_s{steps}_d{d}"
                    j = self.r.get(stage, tag)
                    if j is None and parent_stage:
                        ptag = f"learn_{RC.spec_id(s).replace('_ls0', '')}_{n}_s{steps}_d{d}"
                        j = self.parent.get(parent_stage, ptag)
                    if j is None:
                        continue
                    vals.append(j["heldout_final"])
                    nr = j.get("normalization") or {}
                    if "clipped_fraction" in nr:
                        clip.append(nr["clipped_fraction"])
                    sp = j["signal_start"]["probe"]
                    rv.append(sp["relative_variance_vs_dense_row_connected"]["median"]
                              if sp.get("relative_variance_vs_dense_row_connected") else float("nan"))
                    pr.append(j["signal_after"]["probe"]["diversity"]["participation_ratio"])
                if not vals:
                    continue
                rows.append([s, n, len(vals), " / ".join(f"{v:.4f}" for v in vals), fmt_ci(vals),
                             f"{statistics.mean(clip):.3f}" if clip else "", f"{statistics.mean(rv):.3f}",
                             f"{statistics.mean(pr):.1f}"])
        if rows:
            self.h(title)
            self.table(f"{stage}_learn_s{steps}", ["support", "treatment", "seeds", "held-out per seed",
                                                   "mean ± 95 % CI", "clipped (connected)", "rel. variance (start)",
                                                   "PR (after)"], rows)

    def final(self):
        sel = self.r.get("", "final_selection") if (self.r.root / "final_selection.json").exists() else None
        if sel is None:
            return
        self.h("Locked final test (batches 19,000-19,007; data / SR seeds 3 / 4 / 5; graph 0 / layout 0)")
        vals = {}
        rows = []
        for spec, n in sel["methods"]:
            v, dv = [], []
            for d in (3, 4, 5):
                j = self.r.get("final", f"learn_{RC.spec_id(spec)}_g0_{n}_s256_d{d}_final")
                if j is None:
                    continue
                v.append(j["final_test"])
                dv.append(j["heldout_final"])
            if not v:
                continue
            vals[spec] = v
            rows.append([spec, n, " / ".join(f"{x:.4f}" for x in v), fmt_ci(v), fmt_ci(dv)])
        self.table("final_test", ["method", "treatment", "final test per seed (3 / 4 / 5)", "final test mean ± 95 % CI",
                                  "dev validation (same runs) mean ± CI"], rows)
        rows = []
        for a, b in sel["comparisons"]:
            if a not in vals or b not in vals or len(vals[a]) != len(vals[b]):
                continue
            d = [x - y for x, y in zip(vals[a], vals[b])]
            m, h = ci(d)
            up = m + h
            verdict = ("improvement" if up < 0 else "noninferior (margin 0.003)" if up < sel["margin"] else
                       "worse (whole interval above 0; not noninferior)" if m - h > 0 else "inconclusive")
            rows.append([f"{a} - {b}", " / ".join(f"{x:+.4f}" for x in d), f"{m:+.4f}", f"[{m - h:+.4f}, {up:+.4f}]",
                         verdict])
        self.table("final_paired", ["comparison", "per seed", "mean", "paired 95 % CI", "verdict"], rows)

    def interaction(self, stage, title):
        st = self.r.status(stage)
        if not st:
            return
        rows = []
        for x in st:
            j = self.r.get(stage, x["tag"]) if x["status"] == "measured" else None
            if j is None:
                rows.append([x["tag"], x["status"]] + [""] * 7)
                continue
            h, e = j["final"], j["term_support_eligibility"]
            ft = j.get("final_test")
            q = (j.get("oracle_quantization") or {}).get("bf16_input_targets", {})
            rows.append([j["support"], j.get("oracle") or "ordinary", j["init_seed"], j["lr_layer"], j["steps"],
                         " / ".join(f"{h[f'explained_interaction_order{o}']:.3f}" for o in "234"),
                         "" if ft is None else " / ".join(f"{ft[f'explained_interaction_order{o}']:.3f}" for o in "234"),
                         " / ".join(str(e[f"order{o}"]["terms_with_no_neuron"]) for o in (2, 3, 4)),
                         "" if not q else f"{q['order2']:.2e}"])
        self.h(title)
        self.table(stage, ["support", "init", "init seed", "layer lr", "updates", "explained 2 / 3 / 4 (dev held-out)",
                           "explained 2 / 3 / 4 (reserved final batches)", "terms without eligible neuron (2 / 3 / 4)",
                           "oracle nMSE vs BF16-input targets (order 2)"], rows)

    def energy(self):
        rows = []
        for st, spec in (("k8_q32", "BP-row/8"), ("w8_screen", "WRS/8:4#0"), ("w8_screen", "WRS/8:16#0"), ("w8_screen", "TopK/8"),
                         ("m25_screen", "BP-row#0"), ("m25_screen", "TopK/16"), ("m25_screen", "Mix-BP#0"),
                         ("m25_screen", "Mix-TopK#0")):
            j = self.r.get(st, f"learn_{RC.spec_id(spec)}_g0_none_s32_d0")
            if j is None:
                continue
            ss = j["support_record"].get("support_stats", {})
            e = (ss.get("energy") or {}).get("layer_fraction", ss.get("energy_layer_fraction"))
            rows.append([spec, ss.get("enabled_slots"), "" if e is None else f"{e:.4f}", f"{j['support_select_seconds']:.1f}",
                         ss.get("chosen_candidate_hist", ""), (ss.get("duplicate_candidates") or ""),
                         ss.get("support_fingerprint", "")[:16]])
        if rows:
            self.h("Retained pre-pruning weight energy and selection cost (graph 0)")
            self.table("energy", ["support", "enabled slots", "layer energy fraction", "selection s (in fine-tuning)",
                                  "chosen candidate histogram", "duplicate candidates", "support fingerprint"], rows)

    def grouped_phases(self, stage, tag, ref_tags):
        j = self.r.get(stage, tag)
        if j is None:
            return
        rows = []
        for t in [tag] + ref_tags:
            x = self.r.get(stage, t)
            if x is None:
                continue
            sm = step_samples(x)
            rows.append([t] + [f"{statistics.median(sm[p]):.1f}" if sm.get(p) else "" for p in
                               ("fwd_layout", "fwd_zero", "fwd_kernel", "fwd_finalize", "bwd_coef", "bwd_layout",
                                "bwd_dx_init", "dx_kernel", "dw_kernel", "bwd_post", "optimizer", "step", "host_enqueue",
                                "fwd_t64", "fwd_t16", "fwd_t8", "dx_t64", "dx_t16", "dx_t8", "dw_t64", "dw_t16", "dw_t8")])
        self.h("Phase breakdown (median ms, 24 samples, graph 0)")
        self.table("phases", ["run", "fwd layout", "fwd zero", "fwd kernels", "fwd finalize", "bwd coef", "bwd layout",
                              "dX init", "dX", "dW / db", "bwd post", "optimizer", "step", "host enqueue",
                              "fwd K64", "fwd K16", "fwd K8", "dX K64", "dX K16", "dX K8", "dW K64", "dW K16", "dW K8"], rows)

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables.md").write_text("\n".join(self.md) + "\n")
        (out / "csv").mkdir(exist_ok=True)
        for name, (hd, rows) in self.csvs.items():
            with open(out / "csv" / f"{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(hd)
                w.writerows(rows)


def main():
    from k64_campaign import results_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--parent", default=str(Path(__file__).resolve().parent.parent / "results" /
                                            "random_supports_normalization_controls"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root) if a.root else results_root()
    R = Report(root, Path(a.parent))
    R.errata()
    R.coverage("k8_cov", "K8 coverage on the timed supports",
               [(f"BP-row/8#{ls}", g, None) for g in (0, 1, 2) for ls in (0, 1)] +
               [(s, g, None) for g in (0, 1, 2) for s in ("U8@0.125", "FP-mag/8")])
    R.timing("k8_timing", "K8 confirmation timing (24 samples per process)")
    R.learn("k8_q32", "K8 quality, 32 updates (development, seed 0)", ("BP-row/8", "U8@0.125", "FP-mag/8"), 32, (0,))
    R.learn("k8_q256", "K8 quality, 256 updates (development seeds 0/1/2)", ("BP-row/8", "U8@0.125", "FP-mag/8"), 256)
    R.learn("k8_graph", "K8 graph / layout checks (selected treatment, data / SR seed 0, 256 updates)",
            [f"BP-row/8#{ls}" for ls in (0, 1)] + ["U8@0.125"], 256, (0,), ("energy",), g=1)
    R.learn("k8_graph", "", [f"BP-row/8#{ls}" for ls in (0, 1)] + ["U8@0.125"], 256, (0,), ("energy",), g=2)
    R.coverage("w8_screen", "Weighted K8 screen: coverage (graph 0)", [(s, 0, None) for s in
               ("WRS/8:4#0", "WRS/8:16#0", "TopK/8")] + [("BP-row/8#0", 0, "k8_cov"), ("FP-mag/8", 0, "k8_cov")])
    R.timing("w8_screen", "Weighted K8 screen: timing (8 samples, same session)", ref="BP-row/8", ref_fmt="indexed")
    R.learn("w8_screen", "Weighted K8 screen: 32 updates", ("WRS/8:4#0", "WRS/8:16#0", "TopK/8"), 32, (0,))
    R.learn("w8_dev", "Weighted K8 development: 256 updates", ("TopK/8",), 256)
    R.coverage("m25_screen", "200M-slot 2 x 2: coverage (graph 0)", [(s, 0, None) for s in
               ("TopK/16", "Mix-BP#0", "Mix-TopK#0")] + [("BP-row#0", 0, "errata_cov"), ("A8@0.25", 0, "errata_cov"),
                                                        ("U16@0.25", 0, "errata_cov")])
    R.timing("m25_screen", "200M-slot 2 x 2: timing (8 samples, same session)", ref="BP-row", ref_fmt="indexed")
    R.learn("m25_screen", "200M-slot 2 x 2: 32 updates", ("TopK/16", "Mix-BP#0", "Mix-TopK#0", "BP-row#0", "A8@0.25",
                                                          "U16@0.25", "dense"), 32, (0,))
    R.learn("m25_dev", "200M-slot development: 256 updates", ("TopK/16", "Mix-TopK#0"), 256)
    R.learn("norm", "200M-slot controls: 256 updates (parent records, reused)", ("BP-row", "A8@0.25", "U16@0.25", "dense"),
            256, parent_stage="norm")
    R.timing("m25_dev", "200M-slot confirmation timing, graph 0 (24 samples)", ref="BP-row", ref_fmt="indexed")
    R.timing("w8_confirm", "TopK/8 graph / layout timing confirmation (24 samples)", ref="U8@0.125")
    R.timing("m25_confirm", "Mix-TopK graph / layout timing confirmation (24 samples)", ref="BP-row", ref_fmt="indexed")
    R.coverage("w8_confirm", "TopK/8 coverage, graphs 1 / 2", [("TopK/8", g, None) for g in (1, 2)])
    R.coverage("m25_confirm", "Mix-TopK coverage, graph / layout checks", [(f"Mix-TopK#{ls}", g, None)
                                                                         for g in (0, 1, 2) for ls in (0, 1)])
    for g in (1, 2):
        R.learn("w8_confirm", f"TopK/8 graph {g} (data / SR seed 0)", ("TopK/8",), 256, (0,), ("energy",), g=g)
    for g in (0, 1, 2):
        R.learn("m25_confirm", f"Mix-TopK graph {g} layouts (data / SR seed 0)", ("Mix-TopK#0", "Mix-TopK#1"), 256, (0,),
                ("energy",), g=g)
    R.final()
    R.energy()
    R.grouped_phases("m25_dev", "Mix-TopK_ls0_g0_grouped", ["TopK_k16_g0_indexed", "BP-row_ls0_g0_indexed",
                                                            "A8_r0.25_g0_wide", "U16_r0.25_g0_wide"])
    R.interaction("interaction", "Interaction screen (reduced H 262,144 / C256, T16)")
    R.interaction("interaction_extra", "Interaction: extra seeds / continuations (beyond the predeclared rule)")
    R.interaction("interaction_cont", "Interaction: 4,000-update continuations after the --resume fix")
    R.write(Path(a.out) if a.out else root)
    print(f"wrote {len(R.csvs)} tables")


if __name__ == "__main__":
    main()
