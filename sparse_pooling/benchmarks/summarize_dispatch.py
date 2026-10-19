"""Summaries of the Track D dispatch runs (bench_supergroups.py JSONs): per file and variant,
dX kernel / backward / step medians with round-median ranges, the dX plan that ran, and the
chunk descriptors; Plan B vs Plan A differences.

    python summarize_dispatch.py results/scale_dispatch_fused_backward/crossover_screen/*.json [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def row(path, name, v, cfg):
    ph = v["phases"]
    gd = (v.get("info") or {}).get("grouped_dx") or {}
    de = v.get("descriptors") or {}

    def m(k):
        return ph[k]["median_ms"] if k in ph else float("nan")

    def rng(k):
        r = ph[k]["round_medians_ms"] if k in ph else []
        return f"{min(r):.1f}-{max(r):.1f}" if r else ""

    return {"file": Path(path).stem, "variant": name, "H": cfg["H"], "M": cfg["M"], "G": cfg["G"], "S": cfg["S"],
            "R": cfg["R"], "precision": cfg["precision"], "plan": gd.get("plan"), "T": gd.get("tokens"),
            "WG": gd.get("reduction_group_warps"), "scatter": gd.get("scatter"), "kernel": gd.get("kernel"),
            "ctas": gd.get("ctas"), "chunks": de.get("chunks"), "units": de.get("units"),
            "units_per_chunk": round(de.get("units_per_chunk_mean", float("nan")), 1),
            # mean work units per NONEMPTY bucket (schema-1 JSONs: units_per_bucket_mean, the same quantity)
            "mean_units_per_nonempty_bucket": round(de.get("mean_units_per_nonempty_bucket",
                                                           de.get("units_per_bucket_mean", float("nan"))), 3),
            "mean_units_per_possible_bucket": de.get("mean_units_per_possible_bucket"),  # schema 2 only
            "bucket_occupancy": de.get("bucket_occupancy"),
            "stats_schema": de.get("schema", "granule.bucket_stats/1"),
            "chunks_per_q": de.get("chunks_per_q_mean"), "fwd": round(m("fwd_total"), 2),
            "dx_kernel": round(m("dx_kernel"), 2), "dx_round_range": rng("dx_kernel"), "dw_kernel": round(m("dw_kernel"), 2),
            "bwd_total": round(m("bwd_total"), 2), "step": round(m("step"), 2), "step_round_range": rng("step"),
            "peak_alloc_GiB": round(v["peak_allocated_bytes"] / 2**30, 3),
            "warm_peak_res_GiB": round(v.get("warmup_peak_reserved_bytes", 0) / 2**30, 3),
            "restore_checks": v.get("restore_checks")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for f in args.files:
        d = json.load(open(f))
        for name, v in d["variants"].items():
            r = row(f, name, v, d["config"])
            r["commit"], r["dirty"] = d.get("commit", "")[:7], d.get("dirty")
            rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["file"], {})[r["variant"]] = r
    for f, vs in by.items():
        print(f"== {f}")
        for n, r in vs.items():
            print(f"  {n:22s} plan {str(r['plan']):11s} T {r['T']} WG {r['WG']} {str(r['scatter']):28s} dX {r['dx_kernel']:8.2f} "
                  f"[{r['dx_round_range']}] step {r['step']:8.2f} [{r['step_round_range']}] upc {r['units_per_chunk']} "
                  f"ctas {r['ctas']}")
        a, b = vs.get("planA"), vs.get("planB")
        if a and b:
            print(f"  B - A: dX {b['dx_kernel'] - a['dx_kernel']:+.2f} ms, step {b['step'] - a['step']:+.2f} ms "
                  f"({100 * (b['step'] / a['step'] - 1):+.2f} %)")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
