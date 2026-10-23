"""Tables, CSVs and plots of the K64 large-C / index-free / L1 study from the raw records under
results/k64_large_c_indexfree_l1/ (copied from $GRANULE_RESULTS_ROOT).  Prose: K64_LARGE_C_INDEXFREE_L1.md.

    python k64_report.py [--raw DIR]

Writes <raw>/tables.md, <raw>/csv/*.csv, <raw>/plots/*.png.  Every number comes from a measured record
(status "measured"); supplied model expectations are labelled as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import k64_cases as KC

PH = ("fwd_kernel", "fwd_total", "dx_kernel", "dw_kernel", "bwd_total", "optimizer", "step")


def load_status(d: Path):
    st = d / "status.jsonl"
    if not st.exists():
        return []
    return [json.loads(ln) for ln in st.read_text().splitlines() if ln.strip()]


def med(v, ph):
    p = v["phases"].get(ph)
    return None if p is None else p["median_ms"]


def fmt(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def variant_rows(raw: Path, sub="screen"):
    """[{case, variant, phases..., memory, fingerprints}] of every measured dense timing record."""
    rows = []
    for s in load_status(raw / sub):
        if s["status"] != "measured":
            rows.append(dict(case=s["case"], run=s["tag"], variant="(run)", status=s["status"], seed=s["graph_seed"]))
            continue
        j = json.loads((raw / sub / s["json"]).read_text())
        for nm, v in j["variants"].items():
            r = dict(case=s["case"], run=s["tag"], seed=s["graph_seed"], variant=nm, status="measured",
                     C=j["config"]["C"], S=j["config"]["S"], R=j["config"]["R"], H=j["config"]["H"],
                     graph_fp=j["kcsr"]["graph_fingerprint"][:12], params_sha=j["kcsr"]["logical_params_sha"][:12],
                     peak_gib=v["peak_allocated_bytes"] / 2**30, restore_checks=v["restore_checks"],
                     source_dirty=j["provenance"]["source_dirty"], commit=j["provenance"]["commit"][:8],
                     lv=(v.get("execution") or {}).get("variant"), samples=v["phases"]["step"]["n"])
            for ph in PH:
                r[ph] = med(v, ph)
            st = v["phases"]["step"]
            r["step_min"], r["step_max"] = st["min_ms"], st["max_ms"]
            r["step_rounds"] = st["round_medians_ms"]
            li = v.get("last_info") or {}
            wl = li.get("kgx_work_lists") if (v.get("execution") or {}).get("variant") not in ("kgen", "lowered") \
                else li.get("kgen_route")
            r["work_items"] = (wl or {}).get("items")
            r["work_chunks"] = (wl or {}).get("chunks")
            r["items_per_cta"] = ((wl or {}).get("items_per_cta") or {}).get("mean")
            rows.append(r)
    return rows


def write_csv(path: Path, rows, keys=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items() if k in keys})


def coverage_rows(raw: Path, sub="coverage"):
    out = {}
    for s in load_status(raw / sub):
        if s["status"] != "measured":
            continue
        j = json.loads((raw / sub / s["json"]).read_text())
        out[(j["case"], j["graph_seed"])] = j
    return out


def dense_tables(raw: Path, lines: list):
    rows = variant_rows(raw)
    write_csv(raw / "csv" / "dense_screen_variants.csv", rows)
    cov = coverage_rows(raw)
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    lines += ["## Dense screen (graph seed 0; median of 8 restored steps; ms)", "",
              "Reference = existing kgen K64 kernels (C <= 256) or physical C256 lowering (C > 256) on the SAME tensors; "
              "direct@T / idf@T: kgx kernels with physical neuron tile T. Best = lowest step among measured variants.", "",
              "| case | variant | fwd | dX | dW | bwd | opt | step (min–max) | vs ref | peak GiB | items/CTA |",
              "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    best = []
    for c in KC.cases():
        rs = [r for r in by_case.get(c["id"], []) if r["status"] == "measured" and r["case"] == c["id"]]
        if not rs:
            lines.append(f"| {c['id']} | – | | | | | | not measured | | | |")
            continue
        ref = next((r for r in rs if r["variant"] == "ref"), None)
        b = min(rs, key=lambda r: r["step"])
        best.append(dict(b, ref_step=ref["step"] if ref else None))
        for r in rs:
            rel = f"×{ref['step'] / r['step']:.3f}" if ref else "–"
            mark = " **best**" if r is b else ""
            lines.append(f"| {c['id']} | {r['variant']}{mark} | {fmt(r['fwd_kernel'])} | {fmt(r['dx_kernel'])} | "
                         f"{fmt(r['dw_kernel'])} | {fmt(r['bwd_total'])} | {fmt(r['optimizer'])} | {fmt(r['step'])} "
                         f"({fmt(r['step_min'])}–{fmt(r['step_max'])}) | {rel} | {r['peak_gib']:.2f} | "
                         f"{fmt(r['items_per_cta'])} |")
    write_csv(raw / "csv" / "dense_screen_best.csv", best)
    # which C wins at each S / R
    lines += ["", "### Best step per (S, R) across C (ms; best variant of each case)", "",
              "| S | R | " + " | ".join(f"C{c}" for c in (64, 128, 256, 512, 1024, 2048)) + " | winner |",
              "|---|---|" + "---:|" * 6 + "---|"]
    for S in (64, 128):
        for R in (64, 128, 256):
            cells, vals = [], {}
            for C in (64, 128, 256, 512, 1024, 2048):
                b = next((x for x in best if x["C"] == C and x["S"] == S and x["R"] == R), None)
                cells.append(fmt(b["step"]) + f" ({b['variant']})" if b else "–")
                if b:
                    vals[C] = b["step"]
            win = min(vals, key=vals.get) if vals else None
            lines.append(f"| {S} | {R} | " + " | ".join(cells) + f" | C{win} |")
    # attribution for S = 64
    lines += ["", "### S = 64 attribution on identical tensors (step ms; ratios to the kgen / lowered reference)", "",
              "| case | ref | direct best | idf_rmw | idf best | ref→direct | direct→idf_rmw (index removal) | "
              "idf_rmw→idf (register dX) | tile spread idf (max/min) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in KC.cases():
        if c["S"] != 64:
            continue
        rs = {r["variant"]: r for r in by_case.get(c["id"], []) if r["status"] == "measured"}
        if "ref" not in rs:
            continue
        d = [r for k, r in rs.items() if k.startswith("direct")]
        i = [r for k, r in rs.items() if k.startswith("idf") and not k.startswith("idf_rmw")]
        rm = next((r for k, r in rs.items() if k.startswith("idf_rmw")), None)
        db, ib = min(d, key=lambda r: r["step"]), min(i, key=lambda r: r["step"])
        # same tile for the index-removal step
        t0 = rm["variant"].replace("idf_rmw", "")
        d_same = rs.get("direct" + t0, db)
        i_same = rs.get("idf" + t0, ib)
        spread = max(r["step"] for r in i) / min(r["step"] for r in i)
        lines.append(f"| {c['id']} | {fmt(rs['ref']['step'])} | {fmt(db['step'])} ({db['variant']}) | {fmt(rm['step'])} | "
                     f"{fmt(ib['step'])} ({ib['variant']}) | ×{rs['ref']['step'] / db['step']:.3f} | "
                     f"×{d_same['step'] / rm['step']:.3f} | ×{rm['step'] / i_same['step']:.3f} | {spread:.3f} |")
    # anchors
    lines += ["", "### Same-session anchors (screen order: E24 first / middle / last)", "",
              "| run | anchor | step ms | fwd | dX | dW |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["case"] in KC.ANCHORS and r["status"] == "measured":
            lines.append(f"| {r['run']} | {r['case']} {KC.ANCHORS[r['case']]['role']} | {fmt(r['step'])} | "
                         f"{fmt(r['fwd_kernel'])} | {fmt(r['dx_kernel'])} | {fmt(r['dw_kernel'])} |")
    return rows, best, cov


def coverage_tables(raw: Path, lines: list, cov: dict):
    rows = []
    lines += ["", "## Dense potential coverage (graph seed 0; measured on the timed graphs)", "",
              "Pairs / triples exact; quadruples fixed 10M sample (Clopper–Pearson 95 %). Model = supplied pre-experiment "
              "expectation; refined = the same model from the graph's actual local-pass metadata.", "",
              "| case | sets occ/Q | pair % (model) | triple % (model) | quad % [CI] (model) | group triple % | "
              "per-output features min/med/max | per-output pair % med (bound) | per-output triple % med (bound) |",
              "|---|---|---|---|---|---:|---|---|---|"]
    for c in KC.cases():
        j = cov.get((c["id"], 0))
        if j is None:
            lines.append(f"| {c['id']} | not measured |||||||")
            continue
        sp = j["supplied_prediction_pct"]
        po, pf, bd = j["per_output_summary"], j["per_output_features"], j["supplied_per_output_bounds"]
        q = j["quad"]
        rows.append(dict(case=c["id"], C=c["C"], S=c["S"], R=c["R"], pair=j["pair"]["fraction"], triple=j["triple"]["fraction"],
                         quad=q["fraction"], quad_lo=q["ci95"][0], quad_hi=q["ci95"][1],
                         pair_model=sp["pair"]["model"] / 100, triple_model=sp["triple"]["model"] / 100,
                         quad_model=sp["quad"]["model"] / 100, triple_group=j["triple"]["group_fraction"],
                         refined_triple=j["refined_prediction"]["triple"], occupied=j["sets"]["occupied"], Q=j["sets"]["Q"],
                         po_feat_min=pf["min"], po_feat_med=pf["median"], po_feat_max=pf["max"],
                         po_pair_med=po["pair"]["median"], po_pair_min=po["pair"]["min"],
                         po_triple_med=po["triple"]["median"], po_quad_med=po["quad"]["median"],
                         pair_bound=bd["pair_per_output_union_bound_pct"] / 100,
                         triple_bound=bd["triple_per_output_union_bound_pct"] / 100, fp=j["graph_fingerprint"][:12]))
        lines.append(f"| {c['id']} | {j['sets']['occupied']}/{j['sets']['Q']} | {100 * j['pair']['fraction']:.4f} "
                     f"({sp['pair']['model']:.4f}) | {100 * j['triple']['fraction']:.4f} ({sp['triple']['model']:.4f}) | "
                     f"{100 * q['fraction']:.4f} [{100 * q['ci95'][0]:.4f}, {100 * q['ci95'][1]:.4f}] ({sp['quad']['model']:.4f}) | "
                     f"{100 * j['triple']['group_fraction']:.4f} | {pf['min']:.0f}/{pf['median']:.0f}/{pf['max']:.0f} | "
                     f"{100 * po['pair']['median']:.3f} ({bd['pair_per_output_union_bound_pct']:.3f}) | "
                     f"{100 * po['triple']['median']:.4f} ({bd['triple_per_output_union_bound_pct']:.4f}) |")
    write_csv(raw / "csv" / "dense_coverage.csv", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(KC.RAW))
    a = ap.parse_args()
    raw = Path(a.raw)
    lines = ["# K64 study: generated tables", "",
             "Generated by benchmarks/k64_report.py from the raw records in this directory. Units ms unless noted.", ""]
    rows, best, cov = dense_tables(raw, lines)
    coverage_tables(raw, lines, cov)
    try:
        import k64_report_sparse as RS

        RS.tables(raw, lines)
    except ImportError:
        pass
    (raw / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {raw / 'tables.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
