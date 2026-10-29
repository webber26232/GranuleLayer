"""Tables of the TopK/16 confirmation / coverage-constrained TopK round (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md)
from the raw records under $GRANULE_RESULTS_ROOT (or --root): writes tables.md and csv/*.csv.  Sections whose
records are missing are skipped (never filled with placeholders).

    python tkc_report.py [--root DIR] [--prev DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

import common  # noqa: F401
import joint_coverage as JC
import rsn_cases as RC
import tkc_cases as TC
from kwa_report import Root, ci, fmt_ci, med, step_samples

MiB = 2 ** 20


def tq(p, df):
    from scipy.stats import t

    return float(t.ppf(p, df))


class Report:
    def __init__(self, root: Path, prev: Path):
        self.r, self.prev = Root(root), Root(prev)
        self.md, self.csvs = [], {}

    def h(self, t, level=2):
        self.md += ["", "#" * level + f" {t}", ""]

    def note(self, s):
        self.md += [s, ""]

    def table(self, name, header, rows):
        self.md.append("| " + " | ".join(header) + " |")
        self.md.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
        for r in rows:
            self.md.append("| " + " | ".join(str(x) for x in r) + " |")
        self.md.append("")
        self.csvs[name] = (header, rows)

    # ------------------------------------------------------------ construction
    def construct(self, stage="construct_quiet"):
        rows, srows = [], []
        for g in TC.GRAPH_SEEDS:
            j = self.r.get(stage, f"construct_g{g}") or self.r.get("construct", f"construct_g{g}")
            if j is None:
                continue
            for k in (8, 16):
                r = j[f"k{k}"]
                c, t = r["Cover-TopK"], r["TopK"]
                rep = c["stats"]["repair"]
                rows.append([g, k, t["missing_inputs_total"], rep["records_repaired"], rep["swaps"],
                             "yes" if rep["swaps_equal_initial_missing"] else "NO",
                             f"{rep['missing0_full'].get('mean', 0):.2f} / {rep['missing0_tail'].get('mean', 0):.2f}",
                             rep["rounds"], f"{t['stats']['energy']['layer_fraction']:.5f}",
                             f"{c['stats']['energy']['layer_fraction']:.5f}", f"{100 * rep['energy_loss_relative_to_topk']:.3f}",
                             f"{100 * rep['record_relative_loss']['median']:.3f} / {100 * rep['record_relative_loss']['p99']:.3f}",
                             f"{j['importance_seconds']:.2f}", f"{rep['seconds_topk']:.2f}", f"{rep['seconds_repair']:.2f}",
                             f"{c.get('repair_seconds_on_gpu', float('nan')):.2f}",
                             "yes" if c["chunking_independent"] and c.get("gpu_repair_identical", True) else "NO",
                             f"{rep['working_buffer_bytes'] / MiB:.0f}", f"{c['host_maxrss_growth_kib'] / 1024:.0f}",
                             {True: "equal", False: "DIFFERENT", None: "n/a"}[t.get("equal_to_previous_study")],
                             c["record_cover_mask"]["records_complete"], c["record_cover_numerical"]["records_complete"],
                             t["record_cover_mask"]["records_complete"]])
                e = r.get("exact_sample")
                if e:
                    for name, d in e["repair_gap_rel_by_stratum"].items():
                        rr = [x for x in e["rows"] if x["stratum"] == name]
                        srows.append([k, name, e["strata_sizes"][name], len(rr),
                                      sum(x["milp_certified"] for x in rr), sum(x["order_ok"] for x in rr),
                                      sum(x["repair_equal_opt_mask"] for x in rr),
                                      f"{100 * d.get('median', 0):.4f}", f"{100 * d.get('p95', 0):.4f}",
                                      f"{100 * d.get('max', 0):.4f}",
                                      f"{max(abs(x['flow_minus_milp']) for x in rr):.1e}",
                                      f"{max(x['lp_bound'] - x['E_opt'] for x in rr):.1e}"])
                    g_all = e["repair_gap_rel"]
                    srows.append([k, "ALL (stratified, not uniform)", sum(e["strata_sizes"].values()), len(e["rows"]),
                                  sum(x["milp_certified"] for x in e["rows"]), sum(x["order_ok"] for x in e["rows"]),
                                  e["repair_optimal_records"], f"{100 * g_all['median']:.4f}", f"{100 * g_all['p95']:.4f}",
                                  f"{100 * g_all['max']:.4f}", f"{e['milp_seconds']['mean']:.3f} s / record",
                                  f"predicted full model {e['full_model_solver_prediction_seconds'] / 60:.1f} min"])
        if rows:
            self.h("Cover-TopK construction on the actual checkpoints")
            self.table("construct", ["graph", "k", "TopK missing inputs", "records repaired", "swaps",
                                     "swaps = initial missing", "missing / record (full / tail)", "rounds",
                                     "TopK energy", "Cover energy", "loss vs TopK %", "record loss % (median / p99)",
                                     "importance s", "TopK s", "repair s (CPU)", "repair s (GPU)", "chunk / GPU identical",
                                     "repair buffer MiB", "host RSS growth MiB", "TopK fingerprint vs previous round",
                                     "Cover records complete (mask)", "(numerical)", "TopK records complete"], rows)
        if srows:
            self.h("Exact constrained optimum on the stratified record sample (graph 0)")
            self.table("exact_sample", ["k", "stratum", "stratum size", "sampled", "MILP certified integral",
                                        "E(TopK) >= E(opt) >= E(repair)", "repair = optimum mask",
                                        "gap % median", "gap % p95", "gap % max", "|flow - MILP| max", "LP - opt max"],
                       srows)

    def floors(self, stage="construct"):
        rows = []
        for i, _ in TC.INTERACTION_INIT_SR:
            j = self.r.get(stage, f"floor_check_i{i}")
            if j is None:
                continue
            for s, v in j["supports"].items():
                rows.append([i, s, f"{v['mean_local_inputs_per_record']:.2f}", f"{v['missing_target_inputs_per_output']:.3f}",
                             v["outputs_with_missing"], f"{v['analytic']:.5f}",
                             f"{v['monte_carlo']:.5f} ± {1.96 * v['monte_carlo_batch_se']:.5f}",
                             "" if v["relative_difference"] is None else f"{100 * v['relative_difference']:+.2f} %"])
            bf = j["bf16_input_target_control"]
        if rows:
            self.h("Reduced task: order-1 missing-input floor, analytic vs Monte Carlo")
            self.table("floor_check", ["init seed", "support", "local inputs / record", "missing target inputs / output",
                                       "order-1 outputs with missing", "analytic floor (nMSE)",
                                       "Monte Carlo (64 x 512 tokens, ± 1.96 SE)", "relative difference"], rows)
            self.note("BF16-input target control (nMSE of targets of BF16-rounded inputs vs true targets): " +
                      ", ".join(f"order {o} {bf[f'order{o}']:.2e}" for o in (1, 2, 3, 4)))

    # ------------------------------------------------------------ timing
    def timing(self, stage, title, sets, refs):
        st = [s for s in self.r.status(stage) if s["kind"] == "timing"]
        if not st:
            return {}
        by, fails = {}, []
        for s in st:
            spec, fmt, g = s["job"]["support"], s["job"]["format"], s["job"]["graph_seed"]
            if s["status"] != "measured":
                fails.append([spec, fmt, g, s["status"]])
                continue
            j = self.r.get(stage, s["tag"])
            sm = step_samples(j)
            v = j["variants"]["default"]
            pb = j.get("persistent_bytes", {})
            by.setdefault((spec.split("#")[0] if not spec.startswith("dense") else "dense", fmt), {}).setdefault(g, []).append(dict(
                spec=spec, step=med(j), n=len(sm["step"]), q=(min(sm["step"]), max(sm["step"])),
                fwd=med(j, "fwd_total"), dx=med(j, "dx_kernel"), dw=med(j, "dw_kernel"), opt=med(j, "optimizer"),
                enq=med(j, "host_enqueue"), wall=med(j, "step_wall"),
                peak=v["peak_allocated_bytes"], res=v["peak_reserved_bytes"], scratch=v.get("step_scratch_peak_bytes", 0),
                pb=pb, first=(j.get("cold_steps") or {}).get("default", {}).get("first_step_wall_s"),
                build=j.get("build_seconds"), sm=(j.get("gpu_rounds") or [{}])))
        gm = {key: {g: statistics.mean(x["step"] for x in xs) for g, xs in v.items()} for key, v in by.items()}
        rows = []
        for key in sorted(by):
            spec, fmt = key
            ref = next((r for s_, r in refs.items() if spec in sets.get(s_, ())), None)
            g_ = gm[key]
            ratios = [g_[g] / gm[ref][g] for g in g_ if ref and ref in gm and g in gm[ref]]
            xs = [x for v in by[key].values() for x in v]
            lay = [max(x["step"] for x in v) - min(x["step"] for x in v) for v in by[key].values()]
            rows.append([spec, fmt, " / ".join(f"{g_[g]:.1f}" for g in sorted(g_)), fmt_ci(list(g_.values()), 1),
                         f"{max(lay):.1f}" if len(xs) > len(g_) else "", sum(x["n"] for x in xs),
                         "" if not ratios else fmt_ci(ratios, 3), f"{statistics.mean(x['fwd'] for x in xs):.1f}",
                         f"{statistics.mean(x['dx'] for x in xs):.1f}", f"{statistics.mean(x['dw'] for x in xs):.1f}",
                         f"{statistics.mean(x['opt'] for x in xs):.1f}", f"{statistics.mean(x['enq'] for x in xs):.1f}",
                         f"{max(x['peak'] for x in xs) / 2 ** 30:.2f}"])
        self.h(title)
        self.table(stage + "_timing", ["support", "format", "median step ms per graph (layout-averaged)",
                                       "graph mean ± 95 % CI (df 2)", "max layout spread", "samples",
                                       "ratio to same-session ref (graph mean ± CI)", "fwd", "dX", "dW / db", "optimizer",
                                       "host enqueue", "peak alloc GiB"], rows)
        if fails:
            self.table(stage + "_failed", ["support", "format", "graph", "status"], fails)
        mrows = []
        for key in sorted(by):
            x = by[key][min(by[key])][0]
            pb = x["pb"]
            core = ("weights", "bias", "main_grad", "adam_moments", "compact_values", "active_tiles", "indices")
            meta = {k: v for k, v in pb.items() if k not in core and isinstance(v, (int, float))}
            mrows.append([key[0], key[1], f"{pb.get('weights', 0) / MiB:.1f}", f"{pb.get('bias', 0) / MiB:.1f}",
                          f"{pb.get('main_grad', 0) / MiB:.1f}", f"{pb.get('adam_moments', 0) / MiB:.1f}",
                          f"{sum(meta.values()) / MiB:.1f}",
                          "; ".join(f"{k} {v / MiB:.2f}" for k, v in meta.items()) +
                          (f"; format {pb['sparse_format']}" if "sparse_format" in pb else ""),
                          f"{x['scratch'] / MiB:.1f}", f"{x['peak'] / 2 ** 30:.2f}", f"{x['res'] / 2 ** 30:.2f}",
                          "" if x["first"] is None else f"{x['first']:.2f}",
                          "" if x["build"] is None else f"{x['build']:.1f}"])
        self.table(stage + "_memory", ["support", "format", "values MiB", "bias MiB", "main_grad MiB", "moments MiB",
                                       "index / map / metadata / work lists MiB (total charged)", "components MiB", "step scratch peak MiB",
                                       "peak alloc GiB", "peak reserved GiB", "first step wall s", "build s (untimed)"],
                   mrows)
        return gm

    # ------------------------------------------------------------ coverage
    def coverage(self, stage, specs, graphs=TC.GRAPH_SEEDS, title="Coverage of the exact timed supports"):
        rows = []
        for s in specs:
            for g in graphs:
                j = self.r.get(stage, f"cov_{RC.spec_id(s)}_g{g}")
                if j is None:
                    continue
                e = j["enabled"]
                js = e.get("joint_selected", {})
                outs = [o for o in e["outputs"] if o.get("fixed_selection")]
                naive = JC.naive_pooled_binomial(np.array([o["triple_hits"] for o in outs]), 1) if outs else None
                th, tn = sum(o["triple_hits"] for o in outs), sum(o["triple_n"] for o in outs)
                from kcsr_coverage import clopper_pearson

                lo, hi = clopper_pearson(th, tn)
                rc = j.get("record_feature_cover") or {}
                fu = (j.get("feature_union_vs_dense") or {}).get("enabled") or {}
                tri, quad = js.get("triple", {}), js.get("quad", {})
                rows.append([s, g, rc.get("mask_complete", "dense" if s == "dense" else ""),
                             rc.get("numerical_complete", ""), fu.get("outputs_equal_dense", ""),
                             f"{e['per_output_summary']['features']['median']:.0f}",
                             f"{100 * e['pair']['fraction']:.3f}", f"{100 * e['triple']['fraction']:.3f}",
                             f"{100 * e['quad']['fraction']:.4f} [{100 * e['quad']['ci95'][0]:.4f}, {100 * e['quad']['ci95'][1]:.4f}]",
                             f"{100 * e['per_output_summary']['triple']['median']:.4f}",
                             "" if not tri else f"{100 * tri['mean']:.4f} [{100 * tri['ci95'][0]:.4f}, {100 * tri['ci95'][1]:.4f}]",
                             f"[{100 * lo:.4f}, {100 * hi:.4f}]",
                             "" if not quad else f"{100 * quad['mean']:.5f} [{100 * quad['ci95'][0]:.5f}, {100 * quad['ci95'][1]:.5f}]",
                             (j.get("sharing") or {}).get("layer", {}).get("distinct_supports", ""),
                             f"{100 * j.get('width', {}).get('bias_only_mask_fraction', 0):.2f}"])
        if rows:
            self.h(title)
            self.table(stage + "_coverage", ["support", "graph", "records with all 64 inputs (mask)", "(numerical)",
                                             "outputs = dense feature union (of 1024)", "per-output features (median)",
                                             "layer pairs % (exact)", "layer triples % (exact)", "layer quads % [95 % CI]",
                                             "per-output triples % (median of 38)",
                                             "per-output triples % mean of 38, joint-query 95 % CI",
                                             "old pooled-binomial CI (INVALID)", "per-output quads % joint 95 % CI",
                                             "distinct supports (layer)", "structural bias-only %"], rows)

    def paired_coverage(self, stage, pairs, graphs=TC.GRAPH_SEEDS):
        rows = []
        for a, b in pairs:
            for g in graphs:
                fa = self.r.root / stage / f"cov_{RC.spec_id(a)}_g{g}.joint.npz"
                fb = self.r.root / stage / f"cov_{RC.spec_id(b)}_g{g}.joint.npz"
                if not (fa.exists() and fb.exists()):
                    continue
                A, B = np.load(fa), np.load(fb)
                if not np.array_equal(A["outputs"], B["outputs"]):
                    continue
                O = len(A["outputs"])
                out = [a, b, g]
                for name, t in (("triple", 3), ("quad", 4)):
                    p = JC.paired(A[f"enabled_{name}"], B[f"enabled_{name}"], O, math.comb(TC.D, t))
                    out.append(f"{100 * p['mean_difference']:+.5f} [{100 * p['ci95'][0]:+.5f}, {100 * p['ci95'][1]:+.5f}]")
                ja = self.r.get(stage, f"cov_{RC.spec_id(a)}_g{g}")
                jb = self.r.get(stage, f"cov_{RC.spec_id(b)}_g{g}")
                out.append(f"{100 * (ja['enabled']['triple']['fraction'] - jb['enabled']['triple']['fraction']):+.4f}")
                out.append(f"{100 * (ja['enabled']['pair']['fraction'] - jb['enabled']['pair']['fraction']):+.4f}")
                ma = ja["support_record_multiplicity"]["quad"]
                mb = jb["support_record_multiplicity"]["quad"]
                out.append(f"{ma['covered'] - mb['covered']:+d}")
                rows.append(out)
        if rows:
            self.h("Paired coverage differences on the SAME sampled queries")
            self.table("paired_coverage", ["A", "B", "graph", "per-output triples A - B (pp, joint 95 % CI)",
                                           "per-output quads A - B (pp, joint 95 % CI)", "layer triples A - B (pp, exact)",
                                           "layer pairs A - B (pp, exact)", "layer sampled quads covered A - B"], rows)

    def coverage_errata(self):
        """The previous round's pooled-binomial per-output triple intervals vs the corrected joint-query ones (same
        support fingerprints, rerun query pass)."""
        prev = [("k8_cov", f"BP-row/8#{ls}", g) for g in TC.GRAPH_SEEDS for ls in (0, 1)]
        prev += [("k8_cov", "U8@0.125", g) for g in TC.GRAPH_SEEDS] + [("k8_cov", "FP-mag/8", g) for g in TC.GRAPH_SEEDS]
        prev += [("w8_screen", s, 0) for s in ("WRS/8:4#0", "WRS/8:16#0", "TopK/8")]
        prev += [("w8_confirm", "TopK/8", g) for g in (1, 2)]
        prev += [("m25_screen", s, 0) for s in ("TopK/16", "Mix-BP#0", "Mix-TopK#0")]
        prev += [("errata_cov", s, 0) for s in ("BP-row#0", "A8@0.25", "U16@0.25")]
        rows = []
        for st, s, g in prev:
            old = self.prev.get(st, f"cov_{RC.spec_id(s)}_g{g}")
            new = self.r.get("coverage", f"cov_{RC.spec_id(s)}_g{g}") or self.r.get("errata_cov", f"cov_{RC.spec_id(s)}_g{g}")
            if old is None or new is None:
                continue
            o = [x for x in old["enabled"]["outputs"] if x.get("fixed_selection")]
            th, tn = sum(x["triple_hits"] for x in o), sum(x["triple_n"] for x in o)
            from kcsr_coverage import clopper_pearson

            lo, hi = clopper_pearson(th, tn)
            js = new["enabled"]["joint_selected"]["triple"]
            same = old["support_record"]["mask_sha256"] == new["support_record"]["mask_sha256"]
            same_hits = [x["triple_hits"] for x in o] == [x["triple_hits"] for x in new["enabled"]["outputs"]
                                                         if x.get("fixed_selection")]
            rows.append([s, g, "yes" if same else "NO", "yes" if same_hits else "NO", f"{100 * th / tn:.4f}",
                         f"[{100 * lo:.4f}, {100 * hi:.4f}]", f"{100 * js['mean']:.4f}",
                         f"[{100 * js['ci95'][0]:.4f}, {100 * js['ci95'][1]:.4f}]",
                         f"{(js['ci95'][1] - js['ci95'][0]) / (hi - lo):.2f}",
                         f"{js['queries_covered_by_any']:,} of {js['queries']:,}",
                         f"{js['output_hits'] / max(1, js['queries_covered_by_any']):.2f}"])
        if rows:
            self.h("Erratum: per-output triple intervals with shared queries")
            self.table("errata_joint", ["support", "graph", "mask sha equal", "per-output hits equal",
                                        "pooled % (old = new mean)", "old pooled-binomial CI (invalid)",
                                        "corrected mean %", "corrected joint-query CI (FPC)", "width ratio new / old",
                                        "queries covered by >= 1 output", "outputs per covered query"], rows)

    # ------------------------------------------------------------ learning
    def learn(self, stage, title, specs, steps, seeds, norms=("none", "energy"), g=0, src=None):
        rows = []
        for s in specs:
            for n in norms:
                vals, clip, ret, rv, pr = [], [], [], [], []
                for d in seeds:
                    tag = f"learn_{RC.spec_id(s)}_g{g}_{n}_s{steps}_d{d}"
                    j = self.r.get(stage, tag) or (src.get(src_stage(s, steps), tag) if src else None)
                    if j is None:
                        continue
                    vals.append(j["heldout_final"])
                    nr = j.get("normalization") or {}
                    if "clipped_fraction" in nr:
                        clip.append(nr["clipped_fraction"])
                    ss = j["support_record"].get("support_stats", {})
                    e = (ss.get("energy") or {}).get("layer_fraction")
                    if e is not None:
                        ret.append(e)
                    sp = j["signal_start"]["probe"]
                    rv.append((sp.get("relative_variance_vs_dense_row_connected") or {}).get("median", float("nan")))
                    pr.append(j["signal_after"]["probe"]["diversity"]["participation_ratio"])
                if not vals:
                    continue
                rows.append([s, n, len(vals), " / ".join(f"{v:.4f}" for v in vals), fmt_ci(vals),
                             f"{statistics.mean(clip):.4f}" if clip else "", f"{statistics.mean(ret):.4f}" if ret else "",
                             f"{statistics.mean(rv):.3f}", f"{statistics.mean(pr):.1f}"])
        if rows:
            self.h(title)
            self.table(f"{stage}_learn_s{steps}_g{g}", ["support", "treatment", "seeds", "held-out per seed",
                                                         "mean ± 95 % CI", "clipped (connected)", "retained energy",
                                                         "rel. probe variance (start, median)", "PR (after)"], rows)

    def final(self):
        f = self.r.root / "final_selection.json"
        if not f.exists():
            return
        sel = json.loads(f.read_text())
        vals, dev = {}, {}
        rows = []
        for spec, n in sel["methods"]:
            v, dv, sh = [], [], []
            for d in sel["seeds"]:
                j = self.r.get("final", f"learn_{RC.spec_id(spec)}_g0_{n}_s256_d{d}_{TC.TEST_PROTOCOL}")
                if j is None:
                    continue
                assert j.get("final_test_set") == TC.TEST_PROTOCOL
                v.append(j["final_test"])
                dv.append(j["heldout_final"])
                sh.append((j.get("saved_state") or {}).get("sha256", "")[:12])
            if not v:
                continue
            vals[spec], dev[spec] = v, dv
            rows.append([spec, n, len(v), " / ".join(f"{x:.4f}" for x in v), fmt_ci(v), fmt_ci(dv)])
        if not rows:
            return
        self.h(f"New locked test {TC.TEST_PROTOCOL} (batches 29,000-29,007; data / SR seeds 6..11; graph 0)")
        self.table("final_test", ["method", "treatment", "seeds", "test loss per seed (6..11)",
                                  "test mean ± pointwise 95 % CI", "validation (same runs) mean ± CI"], rows)
        tcrit = tq(1 - TC.FAMILY_ALPHA / len(TC.PRIMARY), len(TC.FINAL_SEEDS) - 1)
        prow = []
        for a, b in sel["primary"]:
            if a not in vals or b not in vals or len(vals[a]) != len(vals[b]):
                prow.append([f"{a} - {b}", "missing"] + [""] * 5)
                continue
            d = [x - y for x, y in zip(vals[a], vals[b])]
            m, sd = statistics.mean(d), statistics.stdev(d)
            up = m + tcrit * sd / math.sqrt(len(d))
            _, h = ci(d)
            verdict = ("improvement" if up < 0 else "noninferior (margin 0.003)" if up < sel["margin"] else
                       "inconclusive")
            prow.append([f"{a} - {b}", " / ".join(f"{x:+.4f}" for x in d), f"{m:+.5f}", f"{up:+.5f}",
                         f"[{m - h:+.5f}, {m + h:+.5f}]", verdict])
        self.h("Primary preregistered decisions (Bonferroni over 3, one-sided)", 3)
        self.note(f"upper = mean + t(1 - 0.05/3, df 5) sd / sqrt(6), t = {tcrit:.4f}; noninferior iff upper < 0.003, "
                  "improvement iff upper < 0; otherwise inconclusive.")
        self.table("final_primary", ["comparison", "per seed", "mean", "adjusted one-sided upper bound",
                                     "pointwise two-sided 95 % CI (descriptive)", "decision"], prow)
        srow = []
        for a, b in sel["secondary"]:
            if a not in vals or b not in vals:
                continue
            d = [x - y for x, y in zip(vals[a], vals[b])]
            m, h = ci(d)
            srow.append([f"{a} - {b}", f"{m:+.5f}", f"[{m - h:+.5f}, {m + h:+.5f}]",
                         "below 0" if m + h < 0 else "above 0" if m - h > 0 else "includes 0"])
        if srow:
            self.h("Secondary / exploratory comparisons (pointwise 95 %, no family-wise claim)", 3)
            self.table("final_secondary", ["comparison", "mean", "pointwise 95 % CI", "interval"], srow)
        return vals

    def graph_checks(self):
        f = self.r.root / "graph_selection.json"
        if not f.exists():
            return
        sel = json.loads(f.read_text())
        rows = []
        for spec, n in sel["methods"]:
            r = [spec, n]
            for g in (0, 1, 2):
                st, d = ("final", TC.GRAPH_CHECK_SEED) if g == 0 else ("graph_checks", TC.GRAPH_CHECK_SEED)
                j = self.r.get(st, f"learn_{RC.spec_id(spec)}_g{g}_{n}_s256_d{d}_{TC.TEST_PROTOCOL}")
                r.append("" if j is None else f"{j['final_test']:.4f} ({j['heldout_final']:.4f})")
            rows.append(r)
        self.h("Graph checks (data / SR seed 6; test loss, validation in parentheses)")
        self.table("graph_checks", ["method", "treatment", "graph 0", "graph 1", "graph 2"], rows)

    # ------------------------------------------------------------ interaction
    def interaction(self):
        rows = []
        for stage, steps in (("interaction", TC.INTERACTION_STEPS), ("interaction_cont", TC.INTERACTION_LONG)):
            for spec in TC.INTERACTION_SPECS:
                recs = []
                for i, sr in TC.INTERACTION_INIT_SR:
                    j = self.r.get(stage, f"int_{RC.spec_id(spec)}_T16_i{i}_sr{sr}_s{steps}_lr{TC.INTERACTION_LR:g}")
                    if j is not None:
                        recs.append(j)
                if not recs:
                    continue
                ft = [j["final_test"] for j in recs]
                dv = [j["final"] for j in recs]
                fl = [j["missing_input_floor"] for j in recs]
                el = [j["term_support_eligibility"] for j in recs]
                rows.append([spec, steps, len(recs),
                             f"{statistics.mean(x['mean_local_inputs_per_record'] for x in fl):.2f}",
                             f"{statistics.mean(x['order1_missing_target_inputs_per_output'] for x in fl):.3f}",
                             fmt_ci([x["order1_mean_floor"] for x in fl], 4),
                             fmt_ci([x["order1"] for x in ft], 4), fmt_ci([x["order1"] for x in dv], 4),
                             fmt_ci([x["order1"] - y["order1_mean_floor"] for x, y in zip(ft, fl)], 4),
                             " / ".join(fmt_ci([x[f"explained_interaction_order{o}"] for x in ft], 3) for o in "234"),
                             " / ".join(f"{statistics.mean(e[f'order{o}']['terms_with_no_neuron'] for e in el):.0f}"
                                        for o in (2, 3, 4))])
        if rows:
            self.h(f"Reduced raw-input diagnostic ({TC.INTERACTION_TEST_PROTOCOL} test; 3 init / SR seed pairs)")
            self.table("interaction", ["support", "updates", "runs", "local inputs / record",
                                       "missing order-1 target inputs / output", "order-1 missing-input floor",
                                       "order-1 nMSE (test)", "order-1 nMSE (dev held-out)", "test nMSE - floor",
                                       "explained interaction 2 / 3 / 4 (test)",
                                       "terms without eligible neuron 2 / 3 / 4 (of 256 x 16)"], rows)

    def launches(self):
        j = self.r.get("launch2", "launch_counts_g0") or self.r.get("launch", "launch_counts_g0")
        if j is None:
            return
        rows = [[k, f"{v['kernels_total']:.2f}", f"{sum(v['memset_memcpy'].values()):.2f}", v["backend_calls"]["fwd"],
                 v["backend_calls"]["bwd"], "; ".join(f"{n.split('(')[0][-40:]} x{c:g}" for n, c in v["kernels_by_name"].items() if n.startswith(("void kg", "_l1", "void k")))]
                for k, v in j["runs"].items()]
        self.h("GPU kernel launches per complete training step (profiler, graph 0, outside timing)")
        self.table("launches", ["support:format", "CUDA kernels", "memset / memcpy", "adapter fwd backend calls",
                                "adapter bwd backend calls", "most frequent kernels"], rows)

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables.md").write_text("\n".join(self.md) + "\n")
        (out / "csv").mkdir(exist_ok=True)
        for name, (hd, rows) in self.csvs.items():
            with open(out / "csv" / f"{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(hd)
                w.writerows(rows)


def src_stage(spec, steps):
    return {("TopK/8", 256): "w8_dev", ("TopK/16", 256): "m25_dev", ("TopK/8", 32): "w8_screen",
            ("TopK/16", 32): "m25_screen"}.get((spec, steps), "")


def main():
    from k64_campaign import results_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--prev", default=str(Path(__file__).resolve().parent.parent / "results" /
                                          "weighted_random_supports_anchors"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root) if a.root else results_root()
    R = Report(root, Path(a.prev))
    R.construct()
    R.floors()
    R.timing("perf_screen", "Performance screen (graph 0, 8 samples)", {}, {})
    sets = {"k16": TC.K16_SET, "k8": TC.K8_SET}
    R.timing("perf_confirm", "Performance confirmation (graph 0 / 1 / 2, 24 samples per process)",
             {"U16@0.25": tuple(s.split("#")[0] for s in TC.K16_SET), "U8@0.125": tuple(s.split("#")[0] for s in TC.K8_SET)},
             {"U16@0.25": ("U16@0.25", "wide"), "U8@0.125": ("U8@0.125", "wide")})
    R.coverage("coverage", TC.K16_SET + TC.K8_SET + ("dense",))
    R.paired_coverage("coverage", [("Cover-TopK/8", "TopK/8"), ("Cover-TopK/16", "TopK/16"), ("Cover-TopK/8", "BP-row/8#0"),
                                   ("Cover-TopK/16", "BP-row#0"), ("TopK/16", "U16@0.25"), ("TopK/8", "U8@0.125")])
    R.coverage("errata_cov", ("FP-mag/8", "WRS/8:4#0", "WRS/8:16#0", "Mix-BP#0"),
               title="Coverage reruns of the previous round's other supports (corrected intervals)")
    R.coverage_errata()
    R.learn("dev32", "Development, 32 updates (continuity; data / SR seed 0)",
            ("Cover-TopK/8", "Cover-TopK/16", "TopK/8", "TopK/16"), 32, (0,))
    R.learn("_prev32", "Previous round's 32-update TopK records (continuity reference)", ("TopK/8", "TopK/16"), 32, (0,),
            src=R.prev)
    R.learn("dev256", "Development, 256 updates (seeds 0 / 1 / 2)", ("Cover-TopK/8", "Cover-TopK/16"), 256, TC.DEV_SEEDS)
    R.learn("_prev256", "TopK development, 256 updates (previous round's records, reused after the continuity check)",
            ("TopK/8", "TopK/16"), 256, TC.DEV_SEEDS, src=R.prev)
    R.final()
    R.graph_checks()
    R.interaction()
    R.launches()
    R.write(Path(a.out) if a.out else root)
    print(f"wrote {len(R.csvs)} tables")


if __name__ == "__main__":
    main()
