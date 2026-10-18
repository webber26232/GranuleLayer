"""Tables from bench_supergroups.py JSONs: per (C, S, R) the generic control, every grouped
variant, the best measured value per phase, and the unrestricted reference.

    python summarize_sg.py results/input_supergroups/screen_a10M [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PH = ["fwd_kernel", "fwd_total", "dx_kernel", "dw_kernel", "bwd_total", "optimizer", "step"]


def med(v, ph):
    p = v["phases"].get(ph) or (v["phases"].get("fwd_kernel_generic") if ph == "fwd_kernel" else None)
    return None if p is None else p["median_ms"]


def rows(d: Path):
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        for name, v in r["variants"].items():
            yield dict(file=f.name, commit=r["commit"][:10], C=c["C"], S=c["S"], R=c["R"], Q=c["Q"], H=c["H"],
                       M=c["M"], G=c["G"], pooling=c["pooling"], precision=c["precision"],
                       graph=("unrestricted" if c["S"] == 0 else "constrained"), matched_c8=c.get("matched_c8"),
                       variant=name, kernels=v.get("kernels"), plan=v.get("plan"),
                       **{ph: med(v, ph) for ph in PH},
                       peak_alloc_GiB=v["peak_allocated_bytes"] / 2**30,
                       step_scratch_MiB=v["step_scratch_peak_bytes"] / 2**20,
                       graph_meta_bytes=r["persistent_bytes"].get("graph_metadata"),
                       work_list_bytes=r["persistent_bytes"].get("work_lists"),
                       infeasible=json.dumps(r.get("infeasible", {})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    allr = [x for d in args.dirs for x in rows(Path(d))]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
            w.writeheader()
            w.writerows(allr)
    key = lambda x: (x["H"], x["M"], -x["C"], x["S"])
    groups = {}
    for x in allr:
        groups.setdefault((x["H"], x["M"], x["C"], x["S"], x["R"]), []).append(x)
    print("| H | M | C | S | R | Q | generic step | best grouped variant | its step | Δ | best fwd (gen) | best dX (gen) | best dW (gen) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|")
    for k in sorted(groups, key=lambda k: (k[0], k[1], -k[2], k[3])):
        g = groups[k]
        gen = next((x for x in g if x["variant"] in ("generic", "default")), None)
        gr = [x for x in g if x["variant"] not in ("generic", "default")]
        best = min(gr, key=lambda x: x["step"]) if gr else None

        def bp(ph):
            if not gr or gen is None:
                return ""
            b = min(gr, key=lambda x: x[ph])
            return f"{b[ph]:.2f} {b['variant']} ({gen[ph]:.2f})"
        H, M, C, S, R = k
        Q = g[0]["Q"]
        if best is None:
            print(f"| {H:,} | {M} | {C} | {S or 'unrestr.'} | {R if S else '-'} | {Q or '-'} | {gen['step']:.2f} | — | | | | | |")
            continue
        print(f"| {H:,} | {M} | {C} | {S} | {R} | {Q} | {gen['step']:.2f} | {best['variant']} | {best['step']:.2f} | "
              f"{(best['step'] / gen['step'] - 1) * 100:+.1f}% | {bp('fwd_kernel')} | {bp('dx_kernel')} | {bp('dw_kernel')} |")


if __name__ == "__main__":
    main()
