"""Tables of the interaction-coverage / K16 dX round (results/INTERACTION_COVERAGE_K16_DX.md) from the raw records under
$GRANULE_RESULTS_ROOT (or --root): writes tables.md and csv/*.csv.  Sections whose records are missing are skipped
(never filled with placeholders); comparisons are keyed by explicit seed keys (paired_stats) and fail closed.

    python icdx_report.py [--root DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

import common  # noqa: F401
import anova_bound as AB
import icdx_cases as IC
import joint_coverage as JC
import paired_stats as PS
import rsn_cases as RC
import tkc_report as TR
from kwa_report import ci, fmt_ci, step_samples

MiB = 2 ** 20


def med(j, ph="step"):
    s = step_samples(j).get(ph) or []
    return statistics.median(s) if s else float("nan")


def gci(vals, nd=1):
    """graph mean ± 95 % t half-width (df = n - 1)."""
    return fmt_ci(vals, nd)


class Report(TR.Report):
    def __init__(self, root: Path):
        super().__init__(root, root)

    # ------------------------------------------------------------ construction
    def construct(self):
        rows, srows = [], []
        for g in IC.GRAPH_SEEDS:
            j = self.r.get("construct", f"construct_g{g}")
            if j is None:
                continue
            for spec in ("Cover-TopK/16", "BP-row#0", "Balanced-TopK/16#0", "Balanced-TopK/16#1", "Cover-Mix32:p1#0",
                         "Cover-Mix32:p1#1", "Cover-Mix32:p2#0", "Cover-Mix32:p2#1"):
                r = j.get(spec)
                if r is None:
                    continue
                st = r["stats"]
                gpu = r.get("gpu_repair") or {}
                rep = st.get("repair") or {}
                pops = " / ".join(str(st.get(k, "")) for k in ("H32", "H16", "H8")) if "H32" in st else ""
                rows.append([spec, g, f"{r['build_seconds_cpu']:.1f}",
                             "" if not gpu else f"{gpu['seconds_incl_sync']:.2f}",
                             gpu.get("host_syncs", "per round" if gpu else ""), f"{r['host_peak_rss_growth_bytes'] / MiB:.0f}",
                             f"{r['device_peak_allocated_bytes'] / 2 ** 30:.2f}", f"{r['energy_layer_fraction']:.5f}",
                             r["record_cover"]["mask_complete"], r["record_cover"]["numerical_complete"],
                             "yes" if r["budget_ok"] else "NO", pops,
                             rep.get("swaps", ""), rep.get("rounds", ""),
                             {True: "yes", False: "NO", None: ""}[r.get("chunking_independent")],
                             {True: "yes", False: "NO", None: ""}[gpu.get("identical_to_cpu")] if gpu else "",
                             {True: "equal", False: "DIFFERENT", None: ""}[r.get("equal_to_previous_round")],
                             r["support_fingerprint"][:12]])
            e = j.get("exact_sample")
            if e:
                for name, d in e["repair_gap_rel_by_stratum"].items():
                    rr = [x for x in e["rows"] if x["stratum"] == name]
                    srows.append([g, name, len(rr), sum(bool(x["milp_certified"]) for x in rr),
                                  sum(bool(x["order_ok"]) for x in rr), sum(bool(x["repair_feasible"]) for x in rr),
                                  sum(bool(x["repair_equal_opt_mask"]) for x in rr), f"{100 * d.get('median', 0):.3f}",
                                  f"{100 * d.get('p95', 0):.3f}", f"{100 * d.get('max', 0):.3f}",
                                  f"{statistics.median(x['balanced_vs_cover'] for x in rr):.4f}",
                                  f"{statistics.median(x['opt_vs_cover'] for x in rr if x['opt_vs_cover']):.4f}",
                                  f"{statistics.median(x['balanced_vs_bp'] for x in rr):.3f}",
                                  f"{max(abs(x['flow_minus_milp']) for x in rr if x['flow_minus_milp'] is not None):.1e}"])
        if rows:
            self.h("Construction on the actual checkpoints (CPU build; GPU repair incl. synchronization)")
            self.table("construct", ["support", "graph", "CPU build s", "GPU repair s (incl. sync)", "host syncs (GPU)",
                                     "host peak RSS growth MiB", "device peak GiB", "retained energy",
                                     "records complete (mask)", "(numerical)", "200M budget", "H32 / H16 / H8",
                                     "repair swaps", "rounds", "chunking independent", "GPU = CPU",
                                     "Cover-TopK/16 vs previous round", "fingerprint"], rows)
        if srows:
            self.h("Balanced-TopK/16: exact quota optimum on the stratified sample (graph 0; reference solver)")
            self.table("exact_sample", ["graph", "stratum", "records", "MILP certified", "E(opt) >= E(repair)",
                                        "repair feasible", "repair = optimum mask", "gap % median", "gap % p95",
                                        "gap % max", "E(Balanced) / E(Cover-TopK) median", "E(opt) / E(Cover-TopK) median",
                                        "E(Balanced) / E(BP-row) median", "|flow - MILP| max"], srows)

    def quotas(self):
        j = self.r.get("construct", "construct_g0")
        if j is None:
            return
        rows = []
        for spec in ("Cover-TopK/16", "Balanced-TopK/16#0", "Balanced-TopK/16#1", "Cover-Mix32:p1#0", "Cover-Mix32:p2#0",
                     "BP-row#0"):
            r = j.get(spec)
            if r is None:
                continue
            m = r["multiplicity"]
            for part in ("full", "tail"):
                x = m[part]
                rows.append([spec, part, x["records"], f"{x['groups']['mean']:.3f}", f"{x['groups']['sd']:.3f}",
                             f"{x['groups']['min']:.0f} / {x['groups']['p5']:.0f} / {x['groups']['p95']:.0f} / {x['groups']['max']:.0f}",
                             f"{x['neurons']['mean']:.2f}", f"{x['neurons']['sd']:.2f}",
                             f"{x['per_record_spread_neurons']['median']:.0f}",
                             f"{x['per_record_cv_neurons']['median']:.3f}", x["records_complete"]])
        self.h("Input multiplicity per original record (graph 0): row groups and valid neurons keeping each input")
        self.table("multiplicity", ["support", "records", "count", "groups / input mean", "sd", "min / p5 / p95 / max",
                                    "neurons / input mean", "sd", "per-record spread (neurons, median)",
                                    "per-record CV (neurons, median)", "records complete"], rows)

    # ------------------------------------------------------------ coverage
    def coverage(self, stage="coverage"):
        rows, irows, brows = [], [], []
        for s in tuple(dict.fromkeys(_cov_specs())):
            for g in IC.GRAPH_SEEDS:
                j = self.r.get(stage, f"cov_{RC.spec_id(s)}_g{g}")
                inv = self.r.get(stage, f"inv_{RC.spec_id(s)}_g{g}")
                if j is not None:
                    e = j["enabled"]
                    js = e.get("joint_selected", {})
                    rc = j.get("record_feature_cover") or {}
                    fu = (j.get("feature_union_vs_dense") or {}).get("enabled") or {}
                    tri, quad, pr = js.get("triple", {}), js.get("quad", {}), js.get("pair", {})
                    w = j.get("width", {})
                    sm = j.get("support_record_multiplicity", {}).get("quad", {})
                    rows.append([s, g, rc.get("mask_complete", "dense" if s == "dense" else ""),
                                 rc.get("numerical_complete", ""), fu.get("outputs_equal_dense", ""),
                                 f"{100 * e['pair']['fraction']:.3f}", f"{100 * e['triple']['fraction']:.3f}",
                                 f"{100 * e['quad']['fraction']:.4f} [{100 * e['quad']['ci95'][0]:.4f}, {100 * e['quad']['ci95'][1]:.4f}]",
                                 f"{100 * e['per_output_summary']['pair']['median']:.3f}",
                                 "" if not tri else f"{100 * tri['mean']:.4f} [{100 * tri['ci95'][0]:.4f}, {100 * tri['ci95'][1]:.4f}]",
                                 "" if not quad else f"{100 * quad['mean']:.5f} [{100 * quad['ci95'][0]:.5f}, {100 * quad['ci95'][1]:.5f}]",
                                 (j.get("sharing") or {}).get("layer", {}).get("distinct_supports", ""),
                                 _mult(sm), f"{100 * w.get('bias_only_mask_fraction', 0):.2f}"])
                if inv is not None:
                    rt, wd = inv["record_tuple_coverage"], inv["width"]
                    irows.append([s, g, wd["structural_bias_only"], wd["numerical_bias_only"],
                                  wd["zero_weight_enabled_slots"], f"{inv['energy_layer_fraction']:.5f}",
                                  f"{100 * rt['full']['pair']['mean']:.2f} / {100 * rt['tail']['pair']['mean']:.2f}",
                                  f"{100 * rt['full']['triple']['mean']:.2f} / {100 * rt['tail']['triple']['mean']:.2f}",
                                  f"{inv['multiplicity']['full']['groups']['sd']:.3f} / {inv['multiplicity']['tail']['groups']['sd']:.3f}",
                                  f"{inv['multiplicity']['full']['neurons']['sd']:.2f}",
                                  inv["groups_per_output_max"],
                                  " / ".join(f"{inv['tuple_incidence'][f't{t}']['record_full']['median']:.0f}" for t in (2, 3, 4)),
                                  " / ".join(f"{100 * inv['tuple_incidence'][f't{t}']['union_bound_max_output']:.5f}"
                                             for t in (2, 3, 4)),
                                  "; ".join(f"{k} {v:,}" for k, v in inv["class_populations"].items())])
        if rows:
            self.h("Coverage of the candidates and controls (exact layer tuples; per-output tuples on common queries)")
            self.note("Per-output means are over the 38 fixed selected outputs and are CONDITIONAL on those outputs "
                      "(no output-sampling uncertainty); their intervals are shared-query (joint) intervals with the "
                      "finite-population correction (joint_coverage.py).  Quad multiplicity: of the 10M common sampled "
                      "quads, those contained in 1 / 2-15 / >= 16 row groups (layer scope).")
            self.table("coverage", ["support", "graph", "records with all 64 inputs (mask)", "(numerical)",
                                    "outputs = dense feature union", "layer pairs % (exact)", "layer triples % (exact)",
                                    "layer quads % [95 % CI]", "per-output pairs % (median of 38)",
                                    "per-output triples % (mean of 38, joint 95 % CI)",
                                    "per-output quads % (joint 95 % CI)", "distinct global supports",
                                    "sampled quads covered by 1 / 2-15 / 16+ row groups (mean)", "structural bias-only %"], rows)
        if irows:
            self.h("Inventory: width, record-scope unique tuples, multiplicity spread, incidence sums and union bounds")
            self.note("Record-scope tuples: fraction of a record's C(64, 2) / C(64, 3) local tuples held together by at "
                      "least one of its row groups (exact, unique).  Incidence: sum over the record's row groups of "
                      "C(k_u, t) (a shared 16-neuron support counts once).  Union bound: max over outputs of "
                      "min(1, sum_u C(k_u, t) / C(1024, t)) from the actual populations -- a ceiling, not a prediction.")
            self.table("inventory", ["support", "graph", "structural bias-only rows", "numerical bias-only rows",
                                     "zero-weight enabled slots", "retained energy",
                                     "record pairs % full / tail", "record triples % full / tail",
                                     "groups / input sd full / tail", "neurons / input sd (full)",
                                     "row groups / output max", "incidence per full record t2 / t3 / t4",
                                     "per-output union bound % t2 / t3 / t4", "fan-in populations (neurons)"], irows)
        brows = [[n, IC.incidence(c, 2), IC.incidence(c, 3), IC.incidence(c, 4)] + [
            f"{100 * v:.5f}" for v in IC.union_bounds_763()[n].values()] for n, c in IC.RECORD_CLASSES.items()]
        self.h("Combinatorial ceilings (one full record; 763 row groups per output)")
        self.table("ceilings", ["policy", "pair incidences", "triple incidences", "quad incidences",
                                "union bound % t2", "t3", "t4"], brows)
        self.paired_coverage(stage, [("Balanced-TopK/16#0", "Cover-TopK/16"), ("Cover-Mix32:p1#0", "Cover-TopK/16"),
                                     ("Cover-Mix32:p2#0", "Cover-TopK/16"), ("Cover-Mix32:p2#0", "Cover-Mix32:p1#0"),
                                     ("Balanced-TopK/16#0", "BP-row#0"), ("Cover-Mix32:p1#0", "Mix-TopK#0"),
                                     ("Cover-Mix32:p2#0", "Mix-TopK#0"), ("Cover-TopK/16", "BP-row#0")],
                             graphs=IC.GRAPH_SEEDS)

    # ------------------------------------------------------------ development
    def development(self):
        rows = []
        for s in IC.CONTROLS:
            j = self.r.get("dev32", f"learn_{RC.spec_id(s)}_g0_{IC.CONTROL_TREATMENT[s]}_s32_d0")
            if j:
                rows.append([s, IC.CONTROL_TREATMENT[s], " / ".join(f"{e['heldout']:.5f}" for e in j["evals"])])
        cs = self.r.get("", "continuity_selection")
        if rows:
            self.h("Continuity (32 updates, seed 0) of the controls")
            if cs:
                for r_ in rows:
                    c = cs["controls"][r_[0]]
                    r_ += ["pass" if c["pass_"] else "FAIL", f"{c['max_heldout_difference']:.2e}",
                           "yes" if c["same_mask"] else "NO", "yes" if c["same_checkpoint"] else "NO"]
            self.table("continuity", ["control", "treatment", "held-out at 8 / 16 / 24 / 32", "continuity",
                                      "max |difference| vs earlier", "same mask", "same checkpoint"], rows)
            if cs:
                self.note(f"Reused earlier 256-update records: {cs['reuse']}; rerun: {cs['rerun']}.")
        rows = []
        for s in IC.NEW + ("BP-row#0", "U16@0.25", "dense") + tuple((cs or {}).get("rerun", [])):
            for n in (("none", "energy") if s in IC.NEW else (IC.CONTROL_TREATMENT[s],)):
                recs = self.r.records("dev256", lambda j: j.get("support") == s and j.get("norm") == n) + \
                    self.r.records("dev256_rerun", lambda j: j.get("support") == s and j.get("norm") == n)
                if not recs:
                    continue
                c = PS.Cohort(f"{s} {n}")
                for j, f in recs:
                    c.add(PS.key_of_learn_record(j, protocol_field="_"), j["heldout_final"], f)
                v = [c.obs[k] for k in sorted(c.obs)]
                nr = [j.get("normalization") or {} for j, _ in recs]
                sc = [x.get("scale", {}).get("mean") for x in nr if x.get("scale")]
                rows.append([s, n, len(v), " / ".join(f"{x:.4f}" for x in v), fmt_ci(v),
                             "" if not sc else f"{statistics.mean(sc):.3f}",
                             f"{statistics.mean(x.get('clipped_fraction', 0) for x in nr):.5f}" if any(nr) else ""])
        for s in (cs or {}).get("reuse", []):
            v = []
            for d in IC.DEV_SEEDS:
                p = TR.Path(__file__).resolve().parent.parent / "results" / IC.REUSABLE_DEV256[s].format(d=d)
                if p.exists():
                    v.append(json.loads(p.read_text())["heldout_final"])
            if v:
                rows.append([s + " (earlier record, reused)", IC.CONTROL_TREATMENT[s], len(v),
                             " / ".join(f"{x:.4f}" for x in v), fmt_ci(v), "", ""])
        if rows:
            self.h("Development, 256 updates (validation 9,000-9,003; seeds 0 / 1 / 2; graph 0)")
            self.table("dev256", ["support", "treatment", "seeds", "held-out per seed", "mean ± 95 % CI",
                                  "normalization scale (mean)", "clipped fraction"], rows)
        t = self.r.get("", "treatments_selection")
        if t:
            self.note("Selected treatments: " + "; ".join(f"{k}: {v['selected']} (none {v['mean_none']:.5f}, energy "
                                                          f"{v['mean_energy']:.5f})" for k, v in t["rows"].items()) +
                      f".  Mixed candidate: {t['mixed']} (p1 {t['mixed_means']['p1']:.5f}, p2 {t['mixed_means']['p2']:.5f}).")

    # ------------------------------------------------------------ timing
    def timing_records(self, stage):
        out, fails = {}, []
        for s in self.r.status(stage):
            if s["kind"] != "timing":
                continue
            jb = s["job"]
            key = (jb["support"], jb["format"], jb.get("dx_variant", "legacy"), jb.get("batch", 8192))
            if s["status"] != "measured":
                fails.append(list(key) + [jb["graph_seed"], s["status"]])
                continue
            j = self.r.get(stage, s["tag"])
            out.setdefault(key, {}).setdefault(jb["graph_seed"], []).append((s["tag"], j))
        return out, fails

    def phases(self, js):
        return {ph: statistics.mean(med(j, ph) for j in js) for ph in ("step", "fwd_total", "dx_kernel", "dw_kernel",
                                                                          "optimizer", "host_enqueue", "step_wall")}

    def dx_screen(self):
        recs, fails = self.timing_records("dx_screen")
        if not recs:
            return
        rows = []
        for key in sorted(recs, key=lambda k: (k[3], k[0], k[2] != "legacy", k[2])):
            spec, fmt, v, B = key
            js = [j for _, j in recs[key].get(0, [])]
            p = self.phases(js)
            leg = recs.get((spec, fmt, "legacy", B), {}).get(0)
            pl = self.phases([j for _, j in leg]) if leg else None
            rows.append([spec, v, B, f"{p['step']:.1f}", "" if not pl else f"{p['step'] / pl['step']:.3f}",
                         f"{p['fwd_total']:.1f}", f"{p['dx_kernel']:.1f}",
                         "" if not pl else f"{p['dx_kernel'] / pl['dx_kernel']:.3f}", f"{p['dw_kernel']:.1f}",
                         f"{p['optimizer']:.1f}", f"{p['host_enqueue']:.2f}", sum(len(step_samples(j)["step"]) for j in js)])
        self.h("K16 dX tiling screen (graph 0, 8 samples per process)")
        self.table("dx_screen", ["support", "dX tiling", "M", "complete step ms (median)", "ratio to legacy", "fwd",
                                 "dX", "dX ratio", "dW / db", "optimizer", "host enqueue", "samples"], rows)
        if fails:
            self.table("dx_screen_failed", ["support", "format", "variant", "M", "graph", "status"], fails)
        c = self.r.get("", "confirm_selection")
        if c:
            self.note(f"Eligible / finalists (complete-step rule): {c['finalists']}; correctness gate passed: "
                      f"{c['correctness_gates']}.")

    def profile(self):
        j = self.r.get("profile", "profile_cover16_g0")
        if j is None:
            return
        rows = []
        for v, t in j["tilings"].items():
            at = t["attributes"]
            k = t["kernel_ms_per_step"]
            rows.append([v, f"{t['dynamic_smem_bytes'] / 1024:.0f}", at.get("registers_per_thread"),
                         at.get("local_bytes_per_thread"), at.get("active_ctas_per_sm"), t["theoretical_warps_per_sm"],
                         t["ctas"], f"{t['waves']:.2f}" if t["waves"] else "", f"{k.get('kg_dx', float('nan')):.1f}",
                         f"{k.get('kg_fwd', float('nan')):.1f}", f"{k.get('kg_dw', float('nan')):.1f}",
                         f"{t['analytic_per_dx_launch']['staged_x_bytes'] / 2 ** 30:.2f}",
                         f"{t['analytic_per_dx_launch']['global_fp32_atomics_upper'] / 1e9:.2f}"])
        self.h("Profile of the restored Cover-TopK/16 baseline and the screened tilings (outside timing sessions)")
        tl = j["tools"]
        self.note(f"Tools: {tl.get('ncu', {}).get('version')}, {tl.get('nsys', {}).get('version')}; GPU performance "
                  f"counters: {'DENIED (ERR_NVGPUCTRPERM, RmProfilingAdminOnly 1)' if (tl.get('ncu_counter_attempt') or {}).get('permission_denied') else 'available'}"
                  " -- DRAM / L2 traffic, tensor utilization, shared-memory / CAS behavior, warp stalls and instruction mix "
                  "could not be measured; kernel durations come from the CUPTI activity trace (torch.profiler).  Device: "
                  f"{j['device']['name']}, {j['device']['sm_count']} SMs, {j['device']['smem_per_block_optin']} B opt-in "
                  f"shared memory per block, {j['device'].get('smem_per_sm')} B per SM.")
        self.table("profile", ["tiling", "dyn. smem KiB", "registers / thread", "local B / thread",
                               "active CTAs / SM (occupancy API)", "theoretical warps / SM", "CTAs", "waves",
                               "dX kernel ms / step", "fwd ms", "dW ms", "staged X GiB (estimate)",
                               "FP32 global atomics 1e9 (upper estimate)"], rows)

    def confirm(self):
        recs, fails = self.timing_records("confirm")
        if not recs:
            return {}
        rows, gm = [], {}
        for key in sorted(recs, key=lambda k: (k[3], k[0], k[2] != "legacy", k[2])):
            spec, fmt, v, B = key
            per_g = {g: statistics.mean(med(j) for _, j in xs) for g, xs in recs[key].items()}
            gm[key] = per_g
            leg = recs.get((spec, fmt, "legacy", B))
            ratios = []
            if leg:
                lg = {g: statistics.mean(med(j) for _, j in xs) for g, xs in leg.items()}
                ratios = [per_g[g] / lg[g] for g in per_g if g in lg]
            js = [j for xs in recs[key].values() for _, j in xs]
            p = self.phases(js)
            rows.append([spec, fmt, v, B, " / ".join(f"{per_g[g]:.1f}" for g in sorted(per_g)),
                         gci(list(per_g.values())), "" if not ratios else fmt_ci(ratios, 3), f"{p['fwd_total']:.1f}",
                         f"{p['dx_kernel']:.1f}", f"{p['dw_kernel']:.1f}", f"{p['optimizer']:.1f}",
                         f"{p['host_enqueue']:.2f}", f"{p['step_wall'] - p['step']:.2f}",
                         sum(len(step_samples(j)["step"]) for j in js),
                         f"{max(j['variants']['default']['peak_allocated_bytes'] for j in js) / 2 ** 30:.2f}"])
        self.h("Backend confirmation (graphs 0 / 1 / 2, 24 samples per process; graph-mean intervals over graphs)")
        self.table("confirm", ["support", "format", "dX", "M", "median step per graph ms", "graph mean ± 95 % CI (df 2)",
                               "ratio to same-session legacy", "fwd", "dX", "dW / db", "optimizer", "host enqueue",
                               "host wall - GPU step", "samples", "peak alloc GiB"], rows)
        if fails:
            self.table("confirm_failed", ["support", "format", "variant", "M", "graph", "status"], fails)
        b = self.r.get("", "backend_selection")
        if b:
            rr = []
            for v, r in b["rows"].items():
                rr.append([v] + [("" if r[k]["ratio"] is None else f"{r[k]['ratio']:.4f}") for k in
                                 ("Cover-TopK/16", "Balanced-TopK/16#0", "BP-row#0", b["mixed"], "Cover-TopK/16@M512")] +
                          ["QUALIFIES" if r["qualifies"] else "does not qualify"])
            self.table("backend_rule", ["finalist", "Cover-TopK/16 M8192 ratio (<= 0.95)", "Balanced-TopK/16 (info)",
                                        "BP-row/16 (<= 1.03)", f"{b['mixed']} (<= 1.03)", "Cover-TopK/16 M512 (<= 1.03)",
                                        "decision"], rr)
            self.note(f"Chosen backend: **{b['chosen']}**.")
        return gm

    def guard_replicates(self):
        rows = []
        for stage in ("guard", "replicates"):
            recs, fails = self.timing_records(stage)
            for key in sorted(recs):
                for g, xs in recs[key].items():
                    for tag, j in xs:
                        p = self.phases([j])
                        rows.append([stage, key[0], key[2], key[3], tag, f"{p['step']:.2f}", f"{p['dx_kernel']:.2f}",
                                     f"{p['host_enqueue']:.2f}"])
        if rows:
            self.h("Shape guards (M 512, small batch) and session replicates (graph 0)")
            self.table("guard_replicates", ["stage", "support", "dX", "M", "job", "median step ms", "dX ms",
                                            "host enqueue ms"], rows)

    def memory(self):
        recs, _ = self.timing_records("confirm")
        rows = []
        for key in sorted(recs):
            if key[3] != 8192 or 0 not in recs[key]:
                continue
            j = recs[key][0][0][1]
            pb = j.get("persistent_bytes", {})
            v = j["variants"]["default"]
            core = ("weights", "bias", "main_grad", "adam_moments")
            meta = {k: x for k, x in pb.items() if k not in core and isinstance(x, (int, float))}
            rows.append([key[0], key[1], key[2], f"{pb.get('weights', 0) / MiB:.1f}", f"{pb.get('bias', 0) / MiB:.1f}",
                         f"{pb.get('main_grad', 0) / MiB:.1f}", f"{pb.get('adam_moments', 0) / MiB:.1f}",
                         f"{sum(meta.values()) / MiB:.1f}", f"{v.get('step_scratch_peak_bytes', 0) / MiB:.0f}",
                         f"{v['peak_allocated_bytes'] / 2 ** 30:.2f}", f"{v['peak_reserved_bytes'] / 2 ** 30:.2f}",
                         "" if j.get("build_seconds") is None else f"{j['build_seconds']:.1f}",
                         ((j.get("cold_steps") or {}).get("default") or {}).get("first_step_wall_s", "")])
        if rows:
            self.h("Memory and one-time costs (graph 0 confirmation records)")
            self.table("memory", ["support", "format", "dX", "values MiB", "bias", "main_grad", "moments",
                                  "index / map / work lists / metadata MiB", "step scratch MiB", "peak alloc GiB",
                                  "peak reserved GiB", "build s (untimed: load, construction, packing, compile)",
                                  "first step wall s"], rows)

    # ------------------------------------------------------------ final cohort
    def final(self):
        sel = self.r.get("", "final_selection")
        if sel is None:
            return
        declared = [(d, d) for d in sel["seeds"]]
        cohorts, rows, inv = {}, [], []
        for spec, n in sel["methods"]:
            c = PS.Cohort(spec)
            for j, f in self.r.records("final", lambda j: j.get("support") == spec and j.get("norm") == n and
                                       j["schema"].startswith("granule.rsn_learn/")):
                c.add(PS.key_of_learn_record(j), j["final_test"], f)
            cohorts[spec] = c
        for v in ("legacy", sel["backend"]["chosen"]):
            if v == "legacy" or sel["backend"]["chosen"] != "legacy":
                c = PS.Cohort(f"Cover-TopK/16@indexed:{v}")
                for j, f in self.r.records("final", lambda j, v=v: j["schema"].startswith("granule.icdx_backend_train/")
                                           and j.get("steps") == 256 and (j.get("backend") or {}).get("variant") == v):
                    c.add(PS.key_of_learn_record(j), j["final_test"], f)
                cohorts[c.name] = c
        for name, c in cohorts.items():
            iv = PS.inventory(c, declared)
            if not c.obs:
                continue
            v = [c.obs[k] for k in sorted(c.obs)]
            rows.append([name, len(v), " / ".join(f"{x:.5f}" for x in v), fmt_ci(v, 5),
                         "complete" if iv["complete"] else f"INCOMPLETE missing {iv['missing']}"])
        if not rows:
            return
        self.h(f"New locked test {IC.TEST_PROTOCOL} (batches 39,000-39,007; seeds 12..17; graph 0)")
        self.table("final_test", ["method (rows 1-6 dense-masked; Cover-TopK/16@indexed = compact backend rows 7 / 8)",
                                  "seeds", "test loss per seed (12..17)", "mean ± pointwise 95 % CI", "inventory"], rows)
        prow = []
        for a, b in sel["primary"]:
            if "@indexed" in a and not sel["contrast4_estimable"]:
                r = PS.not_estimable(a, b, IC.FAMILY_SIZE, IC.FAMILY_ALPHA, "no backend qualified")
            else:
                r = PS.paired(cohorts.get(a, PS.Cohort(a)), cohorts.get(b, PS.Cohort(b)), declared,
                              family_size=IC.FAMILY_SIZE, alpha=IC.FAMILY_ALPHA, margin=IC.MARGIN)
            if r["status"] == "complete":
                d = [r["deltas"][f"{s[0]}/{s[1]}"] for s in declared]
                lo, hi = r["descriptive"]["pointwise95"]
                prow.append([f"{a} - {b}", " / ".join(f"{x:+.5f}" for x in d), f"{r['mean']:+.6f}", f"{r['upper']:+.6f}",
                             f"[{lo:+.6f}, {hi:+.6f}]", r["verdict"]])
            else:
                prow.append([f"{a} - {b}", r["status"].upper(), "", "", "", r.get("reason") or "no verdict"])
        self.h("Primary preregistered contrasts (paired by seed key; Bonferroni over 4, one-sided)", 3)
        self.note(f"upper = mean + t(1 - 0.05/4, n - 1) sd / sqrt(n) (t = {PS.tq(1 - 0.05 / 4, 5):.8f} at df 5); "
                  "improvement iff upper < 0; noninferior iff upper < 0.003; otherwise inconclusive.  Pointwise two-sided "
                  "95 % intervals are descriptive.")
        self.table("final_primary", ["contrast (new - control)", "per seed", "mean", "adjusted one-sided upper",
                                     "pointwise 95 % (descriptive)", "decision"], prow)
        erow = []
        for a, b in sel["exploratory"]:
            r = PS.paired(cohorts.get(a, PS.Cohort(a)), cohorts.get(b, PS.Cohort(b)), declared, family_size=1,
                          alpha=0.05, margin=IC.MARGIN)
            d = r.get("descriptive")
            if d and d.get("pointwise95"):
                erow.append([f"{a} - {b}", r["status"], f"{d['mean']:+.5f}",
                             f"[{d['pointwise95'][0]:+.5f}, {d['pointwise95'][1]:+.5f}]"])
        if erow:
            self.h("Exploratory / secondary contrasts (pointwise 95 %, no family-wise claim)", 3)
            self.table("final_exploratory", ["contrast", "pairing", "mean", "pointwise 95 %"], erow)
        self.backend_rows(sel)
        return cohorts

    def backend_rows(self, sel):
        recs = self.r.records("final", lambda j: j["schema"].startswith("granule.icdx_backend_train/"))
        rows = []
        for j, f in sorted(recs, key=lambda x: (x[0]["steps"], x[0]["support"], x[0]["backend"]["variant"],
                                                x[0]["data_seed"])):
            b = j["backend"]
            rows.append([j["support"], j["steps"], j["data_seed"], b["variant"], b["format"], b.get("forward"),
                         b.get("dx"), (b.get("library") or {}).get("library_hash", "")[:12],
                         j["start_state"]["weight"][:10], j["start_state"]["sr_counter"],
                         f"{j['heldout_compact_start']:.6f}", f"{j['heldout_final']:.6f}",
                         "" if "final_test" not in j else f"{j['final_test']:.6f}",
                         "yes" if j["lowering"]["identical_to_dense_masked"] else "NO",
                         "yes" if j.get("mask_preserved") else "NO",
                         (j.get("saved_state") or {}).get("sha256", "")[:12]])
        if rows:
            self.h("Compact-backend training replay records (rows 7 / 8 and the grouped Mix32 K16-component check)", 3)
            self.note("Pairs of rows that differ only in the dX variant start from the same start-state fingerprint "
                      "(packed weight / bias / main_grad / moments / rounding counter / proj-head): checked per seed "
                      "below.  Dense-masked vs compact parameter layouts index stochastic rounding differently: row 7 vs "
                      "row 1 is a secondary lowering / trajectory check, not an identical rounding trajectory.")
            self.table("backend_train", ["support", "updates", "seed", "dX", "format", "forward", "dX kernel",
                                         "library", "start weight sha", "SR counter", "held-out start", "held-out final",
                                         "test", "lowered = dense masked", "mask preserved", "state sha"], rows)
            by = {}
            for j, _ in recs:
                by.setdefault((j["support"], j["steps"], j["data_seed"]), []).append(j)
            same = [all(x["start_state"] == v[0]["start_state"] for x in v) for v in by.values() if len(v) > 1]
            self.note(f"Identical start states within every legacy / optimized pair: {all(same) if same else 'n/a'} "
                      f"({sum(same)} of {len(same)} pairs).")

    def graph_checks(self):
        sel = self.r.get("", "final_selection")
        if sel is None:
            return
        rows = []
        for s in ("Cover-TopK/16", "Balanced-TopK/16#0", sel["mixed"]):
            n = sel["treatments"][s]
            r = [s, n]
            for g in (0, 1, 2):
                st = "final" if g == 0 else "graph_checks"
                j = self.r.get(st, f"learn_{RC.spec_id(s)}_g{g}_{n}_s256_d{IC.GRAPH_CHECK_SEED}_{IC.TEST_PROTOCOL}")
                r.append("" if j is None else f"{j['final_test']:.5f} ({j['heldout_final']:.5f})")
            rows.append(r)
        self.h("Graph checks (seed 12; test loss, validation in parentheses; reported separately)")
        self.table("graph_checks", ["method", "treatment", "graph 0", "graph 1", "graph 2"], rows)

    # ------------------------------------------------------------ reduced task
    def interaction(self):
        rows, erows = [], []
        cont = (self.r.get("", "final_selection") or {}).get("continued", [])
        for stage, steps, specs, sfx in (("interaction_eval", IC.INTERACTION_STEPS, IC.CORE8, "_eval"),
                                         ("interaction_cont", IC.INTERACTION_LONG, cont, "")):
            for spec in specs:
                recs = []
                for i, sr in IC.INTERACTION_INIT_SR:
                    t = f"int_{RC.spec_id(spec)}_T16_i{i}_sr{sr}_s{steps}_lr{IC.INTERACTION_LR:g}{sfx}"
                    j = self.r.get(stage, t)
                    if j is not None:
                        recs.append(j)
                if not recs:
                    continue
                ft = [j["final_test"] for j in recs]
                el = [j["term_support_eligibility"] for j in recs]
                rows.append([spec, steps, len(recs)] + [fmt_ci([x[f"order{o}"] for x in ft], 4) for o in (1, 2, 3, 4)] +
                            [" / ".join(fmt_ci([x[f"explained_interaction_order{o}"] for x in ft], 3) for o in "234")])
                if steps == IC.INTERACTION_STEPS:
                    for o in (2, 3, 4):
                        e = [x[f"order{o}"] for x in el]
                        erows.append([spec, o, f"{statistics.mean(x['terms_with_no_neuron'] for x in e):.0f}",
                                      f"{statistics.mean(x.get('terms_with_one_neuron', 0) for x in e):.0f}",
                                      f"{statistics.mean(x.get('terms_with_multiple_neurons', 0) for x in e):.0f}",
                                      f"{statistics.mean(x['mean_neurons_per_term'] for x in e):.2f}",
                                      f"{statistics.mean(x['matching']['matched_terms'] for x in e if 'matching' in x):.0f}"
                                      f" of {e[0]['terms']}",
                                      f"{statistics.mean(x['matching']['outputs_fully_matched'] for x in e if 'matching' in x):.0f}"])
        if rows:
            self.h(f"Reduced raw-input task ({IC.INTERACTION_TEST_PROTOCOL}; init / SR (30, 0), (31, 1), (32, 2); "
                   "descriptive)")
            self.note("Intervals describe the three init / SR replicates (common training batches), not redrawn data "
                      "distributions; not part of the confirmatory family.")
            self.table("interaction", ["support", "updates", "runs", "order-1 nMSE", "order-2", "order-3", "order-4",
                                       "explained interaction 2 / 3 / 4"], rows)
        if erows:
            self.table("eligibility", ["support", "order", "terms with 0 eligible neurons (of 4,096)", "with 1",
                                       "with > 1", "eligible neurons per term (mean)",
                                       "max term-neuron matching (matched terms)", "outputs fully matched (of 256)"],
                       erows)

    def anova(self):
        rows, grows = [], []
        cont = (self.r.get("", "final_selection") or {}).get("continued", [])
        for spec in IC.CORE8:
            js = [self.r.get("anova", f"anova_{RC.spec_id(spec)}_i{i}") for i, _ in IC.INTERACTION_INIT_SR]
            js = [j for j in js if j]
            if not js:
                continue
            val = [j["validation"] for j in js]
            rows.append([spec, len(js)] + [fmt_ci([j["bound"][f"order{o}"]["estimate"] for j in js], 4) for o in
                                           (1, 2, 3, 4)] +
                        [f"{max(v['reconstruction_max_abs'] for v in val):.1e}",
                         f"{max(v['order1_vs_missing_input_floor']['max_abs_difference'] for v in val):.4f}",
                         ", ".join(sorted({j['status'] for j in js})),
                         f"{max(j['seconds']['total'] for j in js):.0f}",
                         f"{max(j['device_peak_extra_bytes'] for j in js) / MiB:.0f}"])
            # training minus bound on the SAME sampled outputs and analysis batches (batch-cluster bootstrap)
            for stage, steps, sfx in (("interaction_eval", IC.INTERACTION_STEPS, "_eval"),
                                      ("interaction_cont", IC.INTERACTION_LONG, "")):
                if steps == IC.INTERACTION_LONG and spec not in cont:
                    continue
                gaps = {o: [] for o in (1, 2, 3, 4)}
                ok = True
                for (i, sr), j in zip(IC.INTERACTION_INIT_SR, js):
                    t = f"int_{RC.spec_id(spec)}_T16_i{i}_sr{sr}_s{steps}_lr{IC.INTERACTION_LR:g}{sfx}"
                    mf = self.r.root / stage / f"{t}.analysis.npz"
                    bf = self.r.root / "anova" / f"anova_{RC.spec_id(spec)}_i{i}.npz"
                    if not (mf.exists() and bf.exists()):
                        ok = False
                        break
                    M_, B_ = np.load(mf), np.load(bf)
                    outs = B_["outputs"]
                    se, t2m = M_["se"][:, outs], M_["t2"][:, outs]
                    num, den = B_["num"], B_["den"]
                    if not np.allclose(t2m, den, rtol=1e-9, atol=0):
                        raise SystemExit(f"{t}: model and bound target energies differ (not the same tokens / targets)")
                    for o in (1, 2, 3, 4):
                        cols = np.flatnonzero(B_["order"] == o)
                        gaps[o].append(_gap_boot(se[:, cols], num[:, cols], den[:, cols]))
                if ok:
                    grows.append([spec, steps] + [
                        f"{statistics.mean(x['model'] for x in gaps[o]):.4f} / {statistics.mean(x['bound'] for x in gaps[o]):.4f} / "
                        f"{statistics.mean(x['gap'] for x in gaps[o]):+.4f} [{min(x['ci95'][0] for x in gaps[o]):+.4f}, "
                        f"{max(x['ci95'][1] for x in gaps[o]):+.4f}]" for o in (1, 2, 3, 4)])
        if rows:
            self.h("Functional-ANOVA structural lower bound (32 outputs per order, analysis batches 97,000-97,063)")
            self.note("L_g = E[(sum of the disallowed functional-ANOVA components)^2] / E[t_g^2] is a mathematical lower "
                      "bound of the nMSE of ANY function of the available supports; the values are its Monte Carlo "
                      "estimates (mean ± 95 % over the three init seeds; each estimate's batch-cluster bootstrap interval "
                      "is in the record).  Scope: the declared 32-output sample per order -- not the all-output "
                      "locked-test metric.  BF16 quantization excluded.  A zero bound does not prove attainability.")
            self.table("anova_bound", ["support", "inits", "order 1", "order 2", "order 3", "order 4",
                                       "reconstruction max |err|", "order-1 |MC - analytic floor| max (per output)",
                                       "status", "max s", "max device MiB"], rows)
        if grows:
            self.h("Training minus bound on the same outputs and analysis tokens", 3)
            self.note("model nMSE / bound / model - bound [range of the per-init batch-cluster bootstrap 95 % "
                      "intervals]; a positive gap is NOT automatically pure optimization error (finite ReLU "
                      "representability, readout and BF16 inputs also enter).")
            self.table("anova_gap", ["support", "updates", "order 1", "order 2", "order 3", "order 4"], grows)
        o = self.r.get("interaction", "oracle_Mix-TopK_anchor_i30")
        if o:
            q = o.get("oracle_quantization", {})
            self.note("Mix-TopK T16 anchor oracle (attainable-solution control, init 30, analytic readout): nMSE vs "
                      "FP32-input targets " + ", ".join(f"order {k[-1]} {v:.2e}" for k, v in q.get("fp32_targets", {}).items()
                                                        if k.startswith("order")) +
                      "; vs BF16-input targets " + ", ".join(f"order {k[-1]} {v:.2e}" for k, v in
                                                            q.get("bf16_input_targets", {}).items() if k.startswith("order")))
        rc = self.r.get("interaction", "resume_check")
        if rc:
            self.note(f"Uninterrupted vs resumed (20 + 20 updates): pass {rc.get('pass')}; states bitwise equal "
                      f"{all(v['equal'] for v in rc.get('state_differences', {}).values())}; history max |difference| "
                      f"{rc.get('history_max_abs_difference')}; mismatched state refused {rc.get('mismatched_state_refused')}.")

    def launches(self):
        j = self.r.get("launch", "launch_counts_g0")
        if j is None:
            return
        rows = [[k, v["backend_calls_per_step"]["fwd"], v["backend_calls_per_step"]["bwd"],
                 f"{v['cuda_kernels_per_step']:g}", f"{sum(v['memset_memcpy_per_step'].values()):g}", v["label"],
                 "; ".join(f"{n} {c:g}" for n, c in v["expected_layer_kernels"].items())] for k, v in j["runs"].items()]
        self.h("Launches per complete training step (profiler, graph 0, outside timing)")
        self.table("launches", ["support:format:dX", "backend calls fwd", "backend calls bwd", "CUDA kernels",
                                "memset + memcpy", "trace", "layer kernels"], rows)


def _cov_specs():
    import icdx_jobs as J

    return J.COVERAGE_G0 + J.COVERAGE_G12


def _mult(sm):
    if not sm or "group_multiplicity" not in sm:
        return ""
    c = sm["group_multiplicity"]
    cat, q = c.get("categories", {}), c.get("covered_quantiles", {})
    return f"{cat.get('1', 0):,} / {cat.get('2-15', 0):,} / {cat.get('16-63', 0) + cat.get('64+', 0):,} (mean {q.get('mean', 0):.2f})"


def _gap_boot(se, num, den, n_boot=2000, seed=0):
    """model nMSE, bound and their difference (per output ratio of sums, mean over outputs) with a batch-cluster
    bootstrap resampling the analysis batches JOINTLY for the model and the bound."""
    import torch

    se, num, den = (torch.as_tensor(x, dtype=torch.float64) for x in (se, num, den))
    m = float((se.sum(0) / den.sum(0)).mean())
    b = float((num.sum(0) / den.sum(0)).mean())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, se.shape[0], (n_boot, se.shape[0]), generator=g)
    d = ((se[idx].sum(1) - num[idx].sum(1)) / den[idx].sum(1)).mean(1)
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return dict(model=m, bound=b, gap=m - b, ci95=[lo, hi])


def main():
    from k64_campaign import results_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root) if a.root else results_root()
    R = Report(root)
    for f in (R.construct, R.quotas, R.coverage, R.development, R.profile, R.dx_screen, R.confirm, R.guard_replicates,
              R.memory, R.final, R.graph_checks, R.interaction, R.anova, R.launches):
        f()
    R.write(Path(a.out) if a.out else root)
    print(f"wrote {len(R.csvs)} tables")


if __name__ == "__main__":
    main()
