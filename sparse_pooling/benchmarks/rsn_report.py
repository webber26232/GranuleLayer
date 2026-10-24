"""Tables of the random-support / normalization / optimization-control study
(results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md) from the raw records under a results root (stages: refresh,
coverage, screen, confirm, norm_first, norm, interaction, sanitizer, checkpoints).

    python rsn_report.py --root RAW_DIR --parent PARENT_RAW_DIR --out RAW_DIR/tables.md [--csv RAW_DIR/csv]

Timing numbers are medians over the restored samples of one fresh process (screen 8, confirmation 24).
Confirmation: layout replicates are averaged WITHIN a graph seed first; graph-seed intervals use the 3 graph
means (t95, 2 d.o.f.); layout variation is reported separately; within-run samples are never treated as
independent replicates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from wps_report import T95, bench_json, fmt, load, med

PH = ("fwd_layout", "fwd_kernel", "fwd_finalize", "fwd_total", "bwd_coef", "bwd_layout", "bwd_dx_init", "dx_kernel",
      "dw_kernel", "bwd_post", "bwd_total", "optimizer", "step")


def pct(x, nd=2):
    return "–" if x is None else f"{100 * x:.{nd}f}"


def ci(v):
    if len(v) < 2:
        return (v[0] if v else None), None
    m = statistics.mean(v)
    h = T95[len(v) - 1] * statistics.stdev(v) / math.sqrt(len(v))
    return m, h


# ------------------------------------------------------------------ refresh
def refresh_table(root: Path, parent: Path):
    out = ["### Refresh: parent (legacy rule) vs budget-exact rule", "",
           "| source | policy | T | rule | refreshes | B0 | final enabled | final shortfall | max shortfall | "
           "max legal tile | held-out @256 | tiles differing from start | rolled back / infeasible | remap verified |",
           "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    rows = []
    for src, d in (("parent", parent / "refresh"), ("this study", root / "refresh")):
        if not d.exists():
            continue
        for f in sorted(d.glob("refresh_*.json")):
            r = json.loads(f.read_text())
            if "policy" not in r:
                continue
            rule = r.get("rule", "legacy (parent)")
            B0 = r.get("B0") or r.get("budget_slots")
            refs = r.get("refreshes", [])
            cmax = 16 * (8 if r["policy"].endswith("8") else 16)
            short = [B0 - x["enabled_slots"] for x in refs] + [B0 - r["mask_stats0"]["actual_slots"]]
            rb = sum(x.get("prune_rolled_back", 0) or 0 for x in refs)
            inf = sum(bool(x.get("infeasible")) for x in refs)
            remap = all(x.get("state_remap", {}).get("surviving_preserved", False) and
                        x.get("state_remap", {}).get("grown_zero", False) and
                        x.get("state_remap", {}).get("nonzero_outside_mask", 1) == 0 for x in refs) if refs and \
                "state_remap" in refs[0] else ("n/a" if not refs else "not recorded")
            rows.append(dict(source=src, policy=r["policy"], rho=r["rho"], T=r["period"], rule=rule, refreshes=len(refs),
                             B0=B0, final=r["final_enabled_slots"], final_short=B0 - r["final_enabled_slots"],
                             max_short=max(short), cmax=cmax, heldout=r["heldout_final"],
                             changed=r["tiles_changed_vs_start"], rb=rb, inf=inf, remap=remap))
            out.append(f"| {src} | {r['policy']} {r['rho']:g} | {r['period'] or 'frozen'} | {rule.split(' ')[0]} | "
                       f"{len(refs)} | {B0:,} | {r['final_enabled_slots']:,} | {B0 - r['final_enabled_slots']:,} | "
                       f"{max(short):,} | {cmax} | {r['heldout_final']:.5f} | {r['tiles_changed_vs_start']:,} | "
                       f"{rb} / {inf} | {remap} |")
    return out, rows


# ------------------------------------------------------------------ coverage
def coverage_table(root: Path, stage="coverage", title="graph seed 0"):
    rows, recs = load(root, stage)
    out = [f"### Coverage of the exact supports ({title})", "",
           "Predicted values are the independent-round model (ideal quotas / refined with the actual m_q, b_q, groups "
           "per record); measured: exact layer-wide pairs / triples, 10M sampled quadruples (95 % CI); per output: median "
           "over the parent's 38 fixed outputs.", "",
           "| support | pairs pred. ideal / refined / measured % | triples pred. ideal / refined / measured % | "
           "quads pred. refined / measured % [CI] | per-output pairs / triples / quads median % | bias-only % | "
           "distinct supports per record (median) | row groups per covered quad (mean) |",
           "|---|---|---|---|---|---:|---:|---:|"]
    table = []
    for tag, r in recs.items():
        if "enabled" not in r:  # (timing records of a mixed stage)
            continue
        e = r["enabled"]
        p = r.get("predictions") or {}
        ide, ref = p.get("ideal", {}), p.get("refined", {})
        po = e["per_output_summary"]
        q = e["quad"]
        w = r.get("width", {})
        sh = r.get("sharing", {}).get("record", {})
        gm = r.get("support_record_multiplicity", {}).get("quad", {}).get("group_multiplicity", {})
        row = dict(stage=stage, graph_seed=r["graph_seed"], support=r["support"], pair=e["pair"]["fraction"], triple=e["triple"]["fraction"], quad=q["fraction"],
                   quad_ci=q["ci95"], po_pair=po["pair"]["median"], po_triple=po["triple"]["median"],
                   po_quad=po["quad"]["median"], bias_only=w.get("bias_only_mask_fraction", 0.0),
                   pred_ideal=ide, pred_refined=ref, distinct_per_record=sh.get("distinct_per_scope", {}).get("median"),
                   groups_per_quad=gm.get("covered_quantiles", {}).get("mean"), bounds=r.get("bounds"))
        table.append(row)
        gtag = "" if r["graph_seed"] == 0 else f" (g{r['graph_seed']})"
        out.append(f"| {r['support']}{gtag} | {pct(ide.get('t2'), 4)} / {pct(ref.get('t2'), 4)} / {pct(row['pair'], 4)} | "
                   f"{pct(ide.get('t3'))} / {pct(ref.get('t3'))} / {pct(row['triple'])} | "
                   f"{pct(ref.get('t4'), 3)} / {pct(row['quad'], 3)} [{pct(q['ci95'][0], 3)}, {pct(q['ci95'][1], 3)}] | "
                   f"{pct(row['po_pair'], 3)} / {pct(row['po_triple'], 4)} / {pct(row['po_quad'], 5)} | "
                   f"{pct(row['bias_only'], 1)} | {fmt(row['distinct_per_record'], 0)} | {fmt(row['groups_per_quad'], 2)} |")
    b = next((r["bounds"] for r in recs.values() if "bounds" in r), None)
    if b:
        out += ["", f"Per-output union bound for any K16 row-group support (<= {b['per_output_records_max']} records per "
                    f"output): pairs {pct(b['per_output_union']['t2'], 4)} %, triples {pct(b['per_output_union']['t3'], 4)} %, "
                    f"quads {pct(b['per_output_union']['t4'], 5)} %.  One-tile union bound (one common partition per set "
                    f"only): triples {pct(b['one_tile_union']['t3'], 3)} %, quads {pct(b['one_tile_union']['t4'], 4)} %."]
    return out, table


# ------------------------------------------------------------------ timing
def timing_records(root: Path, stage: str):
    rows, _ = load(root, stage)
    out = []
    for r in rows:
        j = r.get("job", {})
        rec = bench_json(root, stage, r["tag"]) if r["status"] == "measured" else None
        out.append(dict(tag=r["tag"], status=r["status"], support=j.get("support"), format=j.get("format"),
                        graph_seed=j.get("graph_seed"), rec=rec))
    return out


def base_spec(s):
    return s.split("#")[0] if s else s


def layout_seed(s):
    return int(s.split("#")[1]) if s and "#" in s else None


def phases(rec):
    ph = rec["variants"]["default"]["phases"]
    return {k: ph[k]["median_ms"] for k in PH if k in ph}


def mem(rec):
    v = rec["variants"]["default"]
    pb = rec.get("persistent_bytes", {})
    return dict(peak=v["peak_allocated_bytes"], reserved=v["peak_reserved_bytes"], scratch=v["step_scratch_peak_bytes"],
                index=pb.get("index_records_int16", 0),
                work=v.get("work_list_bytes", 0) or pb.get("work_lists_cached_last_runtime", 0), persistent=pb,
                executed=rec.get("executed_work"))


def screen_table(root: Path, stage="screen"):
    tr = timing_records(root, stage)
    dense = [med(t["rec"]) for t in tr if t["support"] == "dense" and t["rec"]]
    dmean = statistics.mean(dense) if dense else None
    by = defaultdict(list)
    for t in tr:
        if t["rec"] and t["support"] != "dense":
            by[(base_spec(t["support"]), t["format"])].append(t)
    out = [f"### Complete training step, screen (graph seed 0; median of 8 restored samples per process; "
           f"dense anchors {', '.join(f'{x:.1f}' for x in dense)} ms)", "",
           "| support | format | layout seeds | step ms (per seed) | step mean | x dense | fwd kernel | dX kernel | "
           "dW kernel | optimizer | peak GiB | index bytes MiB | work lists MiB | executed MMA products fwd / dX / dW (1e12) |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    table = []
    for (s, f), ts in sorted(by.items(), key=lambda kv: statistics.mean(med(t["rec"]) for t in kv[1])):
        st = [med(t["rec"]) for t in ts]
        ph = [phases(t["rec"]) for t in ts]
        m = mem(ts[0]["rec"])
        avg = lambda k: statistics.mean(p.get(k, 0.0) for p in ph)  # noqa: E731
        row = dict(support=s, format=f, seeds=[layout_seed(t["support"]) for t in ts], steps=st,
                   step=statistics.mean(st), fwd=avg("fwd_kernel"), dx=avg("dx_kernel"), dw=avg("dw_kernel"),
                   opt=avg("optimizer"), peak=m["peak"], index=m["index"], work=m["work"],
                   mma=(" / ".join(f"{m['executed']['mma_products'][k] / 1e12:.2f}" for k in ("fwd", "dx", "dw"))
                        if m.get("executed") else "–"))
        table.append(row)
        out.append(f"| {s} | {f} | {','.join('-' if x is None else str(x) for x in row['seeds'])} | "
                   f"{' / '.join(f'{x:.1f}' for x in st)} | {row['step']:.1f} | "
                   f"{row['step'] / dmean:.3f} | {row['fwd']:.1f} | {row['dx']:.1f} | {row['dw']:.1f} | {row['opt']:.1f} | "
                   f"{row['peak'] / 2**30:.2f} | {row['index'] / 2**20:.1f} | {row['work'] / 2**20:.1f} | {row['mma']} |")
    failed = [t for t in tr if t["status"] != "measured"]
    if failed:
        out += ["", "Not measured: " + ", ".join(f"{t['tag']} ({t['status']})" for t in failed)]
    return out, table, dmean


def confirm_table(root: Path, stage="confirm"):
    tr = timing_records(root, stage)
    if not tr:
        return [], []
    dense = defaultdict(list)
    vals = defaultdict(lambda: defaultdict(list))
    for t in tr:
        if not t["rec"]:
            continue
        if t["support"] == "dense":
            dense[t["graph_seed"]].append(med(t["rec"]))
        else:
            vals[(base_spec(t["support"]), t["format"])][t["graph_seed"]].append(med(t["rec"]))
    ref_key = ("U16@0.25", "wide")
    out = ["### Confirmation (graph seeds 0/1/2; 24 restored samples per process; layout seeds averaged within graph)",
           "", "| support | format | step ms per graph seed (layout values) | graph mean ± t95 | layout spread (max - min, "
           "mean over graphs) | x U16 0.25 wide (paired per graph) ± t95 | x dense (paired) ± t95 |",
           "|---|---|---|---:|---:|---:|---:|"]
    table = []
    for k, per in sorted(vals.items(), key=lambda kv: statistics.mean(statistics.mean(v) for v in kv[1].values())):
        gm = {g: statistics.mean(v) for g, v in per.items()}
        m, h = ci(list(gm.values()))
        spread = statistics.mean(max(v) - min(v) for v in per.values()) if all(len(v) > 1 for v in per.values()) else None
        ref = vals.get(ref_key, {})
        r_u = [gm[g] / statistics.mean(ref[g]) for g in gm if g in ref]
        r_d = [gm[g] / statistics.mean(dense[g]) for g in gm if dense.get(g)]
        mu, hu = ci(r_u)
        md, hd = ci(r_d)
        table.append(dict(support=k[0], format=k[1], graph_means=gm, mean=m, ci=h, spread=spread, x_u16=mu, x_u16_ci=hu,
                          x_dense=md, x_dense_ci=hd, values=per))
        cells = "; ".join(f"g{g}: {statistics.mean(v):.1f} ({'/'.join(f'{x:.1f}' for x in v)})" for g, v in sorted(per.items()))
        out.append(f"| {k[0]} | {k[1]} | {cells} | {m:.1f} ± {fmt(h)} | {fmt(spread)} | {fmt(mu, 3)} ± {fmt(hu, 3)} | "
                   f"{fmt(md, 3)} ± {fmt(hd, 3)} |")
    dm = {g: statistics.mean(v) for g, v in dense.items()}
    out += ["", "Dense anchors per graph seed: " + ", ".join(f"g{g} {v:.1f} ms" for g, v in sorted(dm.items()))]
    return out, table


def phase_detail(root: Path, stage, keys):
    tr = [t for t in timing_records(root, stage) if t["rec"]]
    out = ["| support | format | " + " | ".join(PH) + " | host enqueue ms | first step ms | build s | "
           "setup: select / convert s |", "|---|---|" + "---:|" * (len(PH) + 4)]
    seen = set()
    for t in tr:
        k = (t["support"], t["format"])
        if base_spec(t["support"]) not in keys or k in seen:
            continue
        seen.add(k)
        p = phases(t["rec"])
        rec = t["rec"]
        gs = rec.get("graph_stats", {})
        cold = rec.get("cold_steps") or {}
        host = rec["variants"]["default"]["phases"].get("host_enqueue", {}).get("median_ms") if \
            "host_enqueue" in rec["variants"]["default"]["phases"] else None
        first = None
        if isinstance(cold, dict) and isinstance(cold.get("default"), dict):
            fs = cold["default"].get("first_step_wall_s")
            first = None if fs is None else 1000 * fs
        out.append(f"| {t['support']} | {t['format']} | " + " | ".join(fmt(p.get(x)) for x in PH) +
                   f" | {fmt(host)} | {fmt(first)} | {fmt(rec.get('build_seconds'), 1)} | "
                   f"{fmt(gs.get('select_seconds'), 2)} / {fmt(gs.get('conversion_seconds'), 2)} |")
    return out


# ------------------------------------------------------------------ learning (normalization)
def learn_rows(root: Path, stages=("norm_first", "norm")):
    out = []
    for st in stages:
        _, recs = load(root, st)
        for tag, r in recs.items():
            n = r.get("normalization") or {}
            sp, s0, sa = r["signal_pruned"]["probe"], r["signal_start"]["probe"], r["signal_after"]["probe"]
            out.append(dict(stage=st, tag=tag, support=r["support"], norm=r["norm"] + ("+b" if r["scale_bias"] else ""),
                            steps=r["steps"], seed=r["data_seed"], heldout_ckpt=r["heldout_checkpoint"],
                            heldout_pruned=r["heldout_pruned"], heldout_norm=r.get("heldout_normalized"),
                            heldout=r["heldout_final"], scale_median=n.get("scale", {}).get("median"),
                            clipped=n.get("clipped_fraction"), rounding=n.get("bf16_rounding", {}).get("max_relative_error"),
                            rel_var_pruned=sp["relative_variance_vs_dense_row"]["median"],
                            rel_var_start=s0["relative_variance_vs_dense_row"]["median"],
                            rel_var_after=sa["relative_variance_vs_dense_row"]["median"],
                            rel_var_after_p10=sa["relative_variance_vs_dense_row"]["p10"],
                            abs_var_after=sa["input_variance"]["median"],
                            var01_pruned=sp["variance_threshold_counts"]["0.01"],
                            var01_after=sa["variance_threshold_counts"]["0.01"],
                            bias_only=sa["classes"]["structurally_disconnected"] + sa["classes"]["connected_all_zero"],
                            switching=sa["switching_fraction_all"], pr_pruned=sp["diversity"]["participation_ratio"],
                            pr_after=sa["diversity"]["participation_ratio"],
                            corr_after=sa["diversity"]["mean_abs_offdiag_corr_all"],
                            outvar_pruned=r["signal_pruned"]["output_variance"]["mean"],
                            outvar_after=r["signal_after"]["output_variance"]["mean"],
                            grad_after=sa.get("grad_norm", {}).get("median"),
                            e2e_rel_var=r["signal_after"]["e2e"]["relative_variance_vs_dense_row"]["median"],
                            mask_preserved=r.get("mask_preserved"), support_after=r.get("support_after"),
                            curve=[e["heldout"] for e in r["evals"]]))
    return out


def learn_table(rows, title, steps=None):
    out = [f"### {title}", "", "| support | treatment | updates | seed | held-out ckpt / pruned / normalized / final | "
           "scale median | clipped % | rel. Var(z-b) vs dense row (median): pruned / start / after | p10 after | "
           "Var>0.01 count pruned / after (of 65,536) | switching % | PR pruned / after | output var x1e4 pruned / "
           "after | grad-norm median after |", "|---|---|---:|---:|---|---:|---:|---|---:|---|---:|---|---|---:|"]
    for r in sorted(rows, key=lambda r: (r["support"], r["norm"], r["steps"], r["seed"])):
        if steps is not None and r["steps"] != steps:
            continue
        out.append(f"| {r['support']} | {r['norm']} | {r['steps']} | {r['seed']} | {r['heldout_ckpt']:.5f} / "
                   f"{r['heldout_pruned']:.5f} / {fmt(r['heldout_norm'], 5)} / {r['heldout']:.5f} | "
                   f"{fmt(r['scale_median'], 3)} | {pct(r['clipped'], 4)} | {r['rel_var_pruned']:.3f} / "
                   f"{r['rel_var_start']:.3f} / {r['rel_var_after']:.3f} | {r['rel_var_after_p10']:.3f} | "
                   f"{r['var01_pruned']:,} / {r['var01_after']:,} | {pct(r['switching'], 1)} | {r['pr_pruned']:.1f} / "
                   f"{r['pr_after']:.1f} | {1e4 * r['outvar_pruned']:.3f} / {1e4 * r['outvar_after']:.3f} | "
                   f"{'–' if r['grad_after'] is None else format(r['grad_after'], '.3e')} |")
    return out


def learn_seed_summary(rows, steps=256):
    by = defaultdict(list)
    for r in rows:
        if r["steps"] == steps:
            by[(r["support"], r["norm"])].append(r)
    out = [f"### {steps}-update fine-tuning over data / training seeds (held-out loss; paired energy - none per seed)", "",
           "| support | none: mean ± t95 (n) | energy: mean ± t95 (n) | energy - none, paired mean ± t95 |",
           "|---|---:|---:|---:|"]
    table = []
    for s in sorted({k[0] for k in by}):
        a = {r["seed"]: r["heldout"] for r in by.get((s, "none"), [])}
        b = {r["seed"]: r["heldout"] for r in by.get((s, "energy"), [])}
        ma, ha = ci(list(a.values())) if a else (None, None)
        mb, hb = ci(list(b.values())) if b else (None, None)
        d = [b[k] - a[k] for k in a if k in b]
        md, hd = ci(d) if d else (None, None)
        table.append(dict(support=s, none=ma, none_ci=ha, energy=mb, energy_ci=hb, diff=md, diff_ci=hd, n=len(d)))
        out.append(f"| {s} | {fmt(ma, 5)} ± {fmt(ha, 5)} ({len(a)}) | {fmt(mb, 5)} ± {fmt(hb, 5)} ({len(b)}) | "
                   f"{fmt(md, 5)} ± {fmt(hd, 5)} |")
    return out, table


# ------------------------------------------------------------------ interaction
def interaction_rows(root: Path, stages=("interaction",)):
    out = []
    for st in stages:
        _, recs = load(root, st)
        for tag, r in recs.items():
            h = r["history"]
            f = r["final"]

            def at(step):
                for e in h:
                    if e["step"] == step:
                        return e
                return None

            out.append(dict(stage=st, tag=tag, support=r["support"], terms=r["terms"], init=r["init_seed"],
                            norm=r["norm"], oracle=r.get("oracle"), perturb=r.get("perturb"), lr=r["lr_layer"],
                            steps=r["steps"], h0=h[0], h1000=at(1000), final=f, floor=r["additive_floor"],
                            elig=r["term_support_eligibility"], oracle_stats=r.get("oracle_stats"),
                            quant=r.get("oracle_quantization"), nonfinite=r.get("nonfinite_at"), history=h))
    return out


def ex(e, o):
    return None if e is None else e.get(f"explained_interaction_order{o}")


def interaction_table(rows, title, filt=lambda r: True):
    out = [f"### {title}", "", "explained interaction 1 - nMSE / floor_k (exact additive floor of the configuration; "
           "held-out estimates on 2,048 tokens), orders 2 / 3 / 4", "",
           "| support | T | start | init seed | lr | updates | at step 0 | at 1,000 | final | order-1 nMSE final | "
           "neurons per term (order 2 / 3 / 4) |", "|---|---:|---|---:|---:|---:|---|---|---|---:|---|"]
    for r in sorted(filter(filt, rows), key=lambda r: (r["terms"], r["support"], str(r["oracle"]), r["norm"],
                                                         r["perturb"] or 0, r["init"], r["lr"], r["steps"])):
        start = (f"oracle-{r['oracle']}" + (f"+{r['perturb']:g}" if r["perturb"] else "")) if r["oracle"] else r["norm"]
        tri = lambda e: "–" if e is None else " / ".join(f"{ex(e, o):.3f}" for o in (2, 3, 4))  # noqa: E731
        el = " / ".join(f"{r['elig'][f'order{o}']['mean_neurons_per_term']:.1f}" for o in (2, 3, 4))
        out.append(f"| {r['support']} | {r['terms']} | {start} | {r['init']} | {r['lr']:g} | {r['steps']} | "
                   f"{tri(r['h0'])} | {tri(r['h1000'])} | {tri(r['final'])} | {r['final']['order1']:.4f} | {el} |"
                   + (f" non-finite at {r['nonfinite']}" if r["nonfinite"] else ""))
    return out


# ------------------------------------------------------------------ sanitizer / checkpoints
def sanitizer_summary(root: Path):
    d = root / "sanitizer"
    out = ["### Compute Sanitizer", "", "| workload | tool | result |", "|---|---|---|"]
    s = d / "summary.txt"
    if s.exists():
        for ln in s.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 3 and "." in parts[0]:
                wl, tool = parts[0].split(".", 1)
                out.append(f"| {wl} | {tool} | {' '.join(parts[1:])} |")
            elif ln.strip():
                out.append(f"| {ln} | | |")
    else:
        for f in sorted(d.glob("*.log")):
            txt = f.read_text(errors="replace")
            ok = "ERROR SUMMARY: 0 errors" in txt or "RACECHECK SUMMARY: 0 hazards" in txt
            wl, tool = f.stem.split(".", 1)
            out.append(f"| {wl} | {tool} | {'0 errors' if ok else 'see log'} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    root, parent = Path(a.root), Path(a.parent)
    md = ["# Tables of the random-support / normalization / optimization-control study (generated by rsn_report.py)", ""]
    t, refresh = refresh_table(root, parent)
    md += t + [""]
    t, cov = coverage_table(root)
    md += t + [""]
    for st, ti in (("coverage_confirm", "graph seeds 1 / 2 of the confirmation"), ("k8", "CONDITIONAL K8 extension")):
        t, c2 = coverage_table(root, st, ti)
        md += t + [""]
        cov += c2
    t, scr, dmean = screen_table(root)
    md += t + [""]
    md += ["### Per-phase medians (ms), screen, first layout seed of each support", ""]
    md += phase_detail(root, "screen", {base_spec(r["support"]) for r in scr} | {"dense"}) + [""]
    t, conf = confirm_table(root)
    md += t + [""]
    lr = learn_rows(root)
    md += learn_table([r for r in lr if r["stage"] == "norm_first"], "Normalization: all treatments (32 updates)") + [""]
    md += learn_table([r for r in lr if r["stage"] == "norm" and r["steps"] == 32], "Normalization: bounded set, 32 updates")
    md += [""] + learn_table([r for r in lr if r["stage"] == "norm" and r["steps"] > 32 and r["seed"] == 0],
                             "Normalization: bounded set, 256 updates (seed 0 detail)") + [""]
    t, seeds = learn_seed_summary([r for r in lr if r["stage"] == "norm"])
    md += t + [""]
    ir = interaction_rows(root)
    for T in sorted({r["terms"] for r in ir}):
        md += interaction_table(ir, f"Interaction diagnostic, T = {T}", lambda r, T=T: r["terms"] == T) + [""]
    md += sanitizer_summary(root) + [""]
    Path(a.out).write_text("\n".join(md))
    if a.csv:
        d = Path(a.csv)
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in (("refresh", refresh), ("coverage", cov), ("screen", scr), ("confirm", conf),
                           ("learn", [{k: v for k, v in r.items() if k not in ("curve", "support_after")} for r in lr]),
                           ("learn_seeds", seeds),
                           ("interaction", [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()
                                             if k != "history"} for r in ir])):
            if not rows:
                continue
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(d / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
