"""Historical audit of the bucket descriptors of results/SCALE_DISPATCH_FUSED_BACKWARD.md (Track B).

For every run JSON of that phase it reconstructs the compact graph metadata from the recorded
plan (supergroups.bucket_table: the generator's own bucket construction, graph version checked)
and recomputes the canonical statistics (bucket_statistics, chunk_statistics at the recorded
chunk target).  It then
  * verifies the reconstruction against the descriptors LOGGED during the old timing run
    (units, nonempty buckets, units per bucket, chunks, units per chunk);
  * parses the "Units / bucket" values PRINTED in the report tables (sections 4.2 and 4.5) and
    states which denominator each printed value corresponds to;
  * evaluates the unchanged scale rule (cuda_ops.scale_rule) on the logged value, the corrected
    value and the possible-bucket mean, next to the plan that actually ran.
Old timing logs / JSON are only read.  Outputs: <out>.json, <out>.csv, <out>.md.

    python audit_bucket_stats.py --out ../results/determinism_metrics_dw_projection/historical_bucket_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import common  # noqa: F401
from sparse_pooling.kernels.cuda_ops import scale_rule
from sparse_pooling.layout import SparseLayout
from sparse_pooling.supergroups import (GRAPH_VERSION, ConnectivityPlan, SupergroupGraph, bucket_statistics,
                                        bucket_table, chunk_statistics, chunk_table, work_items)

OLD = Path(__file__).resolve().parents[1] / "results"
REPORT = OLD / "SCALE_DISPATCH_FUSED_BACKWARD.md"
RAW = OLD / "scale_dispatch_fused_backward"
DIRS = ["crossover_screen", "crossover_stage2", "crossover_bf16", "holdout", "final_100M"]


def parse_h(tok: str) -> int:
    tok = tok.replace(",", "").replace("(rerun)", "").replace("(held-out)", "").replace("(ragged)", "").strip()
    m = re.match(r"([\d.]+)\s*M$", tok)
    return int(round(float(m.group(1)) * 1e6)) if m else int(tok)


def printed_rows():
    """(section, line number, row text, key dict, printed value) of every report table row that
    prints a units-per-bucket value."""
    lines = REPORT.read_text().splitlines()
    rows, sec = [], ""
    for n, ln in enumerate(lines, 1):
        if ln.startswith("### "):
            sec = ln[4:].strip()
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip().strip("*") for x in ln.strip("|").split("|")]
        if sec.startswith("4.2") and len(c) == 9 and c[0] != "H":  # split crossover table
            r = re.match(r"(\d+)", c[2])
            key = dict(precision="split", H=parse_h(c[0]), G=int(c[1]), R=int(r.group(1)),
                       M=[int(v) for v in c[3].split("/")])
            rows.append((sec, n, ln, key, float(c[4])))
        elif sec.startswith("4.2") and len(c) == 6 and c[0] != "H":  # BF16 crossover table
            rows.append((sec, n, ln, dict(precision="bf16", H=parse_h(c[0]), G=1024, R=16, M=[8192]), float(c[1])))
        elif sec.startswith("4.5") and len(c) == 8 and c[0] != "Shape":
            sh = c[0]
            prec = "bf16" if sh.startswith("BF16") else "split"
            key = dict(precision=prec, H=parse_h(re.search(r"H = ([\d,.]+M?)", sh).group(1)),
                       G=int(re.search(r"G = (\d+)", sh).group(1)) if "G =" in sh else 1024,
                       R=int(re.search(r"R = (\d+)", sh).group(1)) if "R =" in sh else 16,
                       M=[int(re.search(r"M = (\d+)", sh).group(1)) if "M =" in sh else 8192])
            rows.append((sec, n, ln, key, float(c[1])))
    return rows


def ln_val(line: str, val: float) -> str:
    """The printed token of a value in a table row (to read its number of decimals)."""
    for tok in re.findall(r"[\d.]+", line):
        try:
            if float(tok) == val:
                return tok
        except ValueError:
            pass
    return str(val)


def runs():
    out = []
    for d in DIRS:
        for f in sorted((RAW / d).glob("*.json")):
            j = json.load(open(f))
            if "config" not in j:
                continue
            out.append((f, j))
    return out


_cache = {}


def reconstruct(cfg):
    key = (cfg["C"], cfg["D"], cfg["H"], cfg["G"], cfg["graph"])
    if key not in _cache:
        plan = ConnectivityPlan.from_json(cfg["graph"])
        if plan.version != GRAPH_VERSION:
            _cache[key] = None  # unresolvable: different generator version
        else:
            L = SparseLayout(cfg["D"], cfg["H"], cfg["G"], 8, cfg["C"], True, packed=True)
            feat, b = bucket_table(L.canonical, plan)
            _cache[key] = (L, plan, feat, b, bucket_statistics(L, b, plan.num_supergroups(cfg["D"]), plan.input_supergroup_dim,
                                                                plan))
    return _cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_file = []
    for f, j in runs():
        cfg = j["config"]
        rec = reconstruct(cfg)
        v0 = next(iter(j["variants"].values()))
        de = v0.get("descriptors") or {}
        row = dict(file=str(f.relative_to(OLD)), commit=j.get("commit", "")[:7], C=cfg["C"], D=cfg["D"], H=cfg["H"],
                   G=cfg["G"], S=cfg["S"], R=cfg["R"], M=cfg["M"], precision=cfg["precision"], graph=cfg["graph"],
                   logged_units=de.get("units"), logged_buckets=de.get("buckets"),
                   logged_units_per_bucket=de.get("units_per_bucket_mean"), logged_chunks=de.get("chunks"),
                   logged_units_per_chunk=de.get("units_per_chunk_mean"))
        if rec is None:
            row["status"] = "unresolvable (graph version)"
            per_file.append(row)
            continue
        L, plan, feat, b, st = rec
        tgt = de.get("chunk_target_units", 4096)
        g = SupergroupGraph(plan, feat, b)
        items, q, _ = work_items(L, g)
        cs = chunk_statistics(items, chunk_table(q, items[:, 2], g.Q, tgt), g.Q, tgt)
        row.update(total_work_units=st["total_work_units"], nonempty_buckets=st["nonempty_buckets"],
                   total_possible_buckets=st["total_possible_buckets"], bucket_occupancy=st["bucket_occupancy"],
                   mean_units_per_nonempty_bucket=st["mean_units_per_nonempty_bucket"],
                   mean_units_per_possible_bucket=st["mean_units_per_possible_bucket"],
                   chunks=cs["chunks"], units_per_chunk_mean=cs["units_per_chunk_mean"])
        match = (de.get("units") == st["total_work_units"] and de.get("buckets") == st["nonempty_buckets"]
                 and abs(de.get("units_per_bucket_mean", -1) - st["mean_units_per_nonempty_bucket"]) < 1e-9
                 and de.get("chunks") == cs["chunks"]
                 and abs(de.get("units_per_chunk_mean", -1) - cs["units_per_chunk_mean"]) < 1e-6) if de else None
        row["reconstruction_equals_logged"] = match
        ran = {n: ((v.get("info") or {}).get("grouped_dx") or {}).get("plan") for n, v in j["variants"].items()}
        row["plans_ran"] = ran
        for B in sorted(set([cfg["M"]])):
            p_new, why = scale_rule(cfg["C"], cfg["S"], cfg["precision"], "relu", (8, 6), B, st)
            fake = dict(st, mean_units_per_nonempty_bucket=st["mean_units_per_possible_bucket"])
            p_pos, _ = scale_rule(cfg["C"], cfg["S"], cfg["precision"], "relu", (8, 6), B, fake)
            row.update(rule_on_nonempty_mean=p_new, rule_on_possible_mean=p_pos, rule_reason=why)
        row["status"] = "ok" if match else ("no logged descriptors" if not de else "MISMATCH")
        per_file.append(row)
    # printed report values
    printed = []
    for sec, n, ln, key, val in printed_rows():
        cand = [r for r in per_file if r["precision"] == key["precision"] and r["H"] == key["H"] and r["G"] == key["G"]
                and r["R"] == key["R"] and r["M"] in key["M"] and "mean_units_per_nonempty_bucket" in r]
        if not cand:
            printed.append(dict(section=sec, line=n, row=ln, printed=val, status="no matching run JSON"))
            continue
        r = cand[0]
        ne, po = r["mean_units_per_nonempty_bucket"], r["mean_units_per_possible_bucket"]
        dec = len(ln_val(ln, val).split(".")[1]) if "." in ln_val(ln, val) else 0
        tol = 10.0 ** -dec + 1e-9  # printed with dec decimals, rounded or truncated

        def near(x):
            return abs(x - val) < tol

        which = ("nonempty mean" if near(ne) and not (near(po) and abs(po - val) < abs(ne - val)) else
                 "possible-bucket mean" if near(po) else "neither")
        if which == "nonempty mean" and round(ne, dec) != val:
            which = "nonempty mean (truncated, not rounded)"
        printed.append(dict(section=sec, line=n, H=key["H"], G=key["G"], R=key["R"], M=key["M"], precision=key["precision"],
                            printed=val, logged=r["logged_units_per_bucket"], nonempty_mean=ne, possible_mean=po,
                            occupancy=r["bucket_occupancy"], printed_matches=which,
                            mislabeled=not which.startswith("nonempty mean"), source=r["file"],
                            rule_on_nonempty_mean=r["rule_on_nonempty_mean"],
                            rule_on_possible_mean=r["rule_on_possible_mean"],
                            selection_changes_with_denominator=r["rule_on_nonempty_mean"] != r["rule_on_possible_mean"]))
    res = dict(method="supergroups.bucket_table (the generator's bucket construction) at the recorded plan; "
                      "canonical bucket_statistics / chunk_statistics; compared with the descriptors logged "
                      "by the old runs (bench_supergroups.py 'descriptors', SgExec.item_chunk_stats schema 1)",
               graph_version=GRAPH_VERSION, files=per_file, printed=printed)
    json.dump(res, open(str(out) + ".json", "w"), indent=1, default=str)
    with open(str(out) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(printed[0]) if printed else ["none"], extrasaction="ignore")
        w.writeheader()
        w.writerows([p for p in printed if "nonempty_mean" in p])
    md = ["| Report line | Row | Printed | Logged (old run) | Nonempty mean | Possible mean | Occupancy | Printed value is | Rule (nonempty) | Rule (possible) |",
          "|---:|---|---:|---:|---:|---:|---:|---|---|---|"]
    for p in printed:
        if "nonempty_mean" not in p:
            md.append(f"| {p['line']} | {p['row'][:40]} | {p['printed']} | | | | | {p['status']} | | |")
            continue
        md.append(f"| {p['line']} | {p['precision']} H={p['H']:,} G={p['G']} R={p['R']} M={'/'.join(map(str, p['M']))} | "
                  f"{p['printed']} | {p['logged']:.4g} | {p['nonempty_mean']:.4g} | {p['possible_mean']:.4g} | "
                  f"{p['occupancy']:.4g} | {p['printed_matches']} | {p['rule_on_nonempty_mean']} | {p['rule_on_possible_mean']} |")
    Path(str(out) + ".md").write_text("\n".join(md) + "\n")
    bad = [r for r in per_file if r["status"] == "MISMATCH"]
    print(f"{len(per_file)} run files, {len(bad)} reconstruction mismatches; {len(printed)} printed rows, "
          f"{sum(p.get('mislabeled', False) for p in printed)} not the nonempty mean")
    for p in printed:
        if p.get("mislabeled"):
            print("  line", p["line"], p["precision"], p["H"], "printed", p["printed"], "->", p["printed_matches"],
                  f"(nonempty {p['nonempty_mean']:.3f})")


if __name__ == "__main__":
    main()
