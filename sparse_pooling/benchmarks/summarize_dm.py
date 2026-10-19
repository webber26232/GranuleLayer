"""Tables of the DETERMINISM_METRICS_DW_PROJECTION runs (bench_supergroups.py JSONs): per file and
variant the phase medians (forward, deterministic stages, dX, dW, backward, optimizer, step) with
the range of the per-round medians, the memory peaks, the resolved plan / projection / reduction,
and PAIRED per-round differences (same round, same restored state): det - fast (per projection)
and compact - padded (per reduction mode), as median and [min, max] over the rounds.

    python summarize_dm.py ../results/determinism_metrics_dw_projection/screen/*.json [--csv out.csv] [--md out.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PH = ["fwd_total", "det_fwd_stage1", "det_fwd_reduce", "dx_kernel", "det_dx_stage1", "det_dx_reduce", "dw_kernel",
      "bwd_total", "optimizer", "step"]
PAIRS = [("det_padded", "fast_padded"), ("det_compact", "fast_compact"), ("fast_compact", "fast_padded"),
         ("det_compact", "det_padded"), ("cap256M", "cap1G"), ("cap4G", "cap1G"), ("cap1G", "fast"), ("det", "fast"),
         ("compact", "padded")]


def med(v, k):
    p = v["phases"].get(k)
    return p["median_ms"] if p else None


def rounds(v, k):
    p = v["phases"].get(k)
    return p["round_medians_ms"] if p else []


def paired(a, b, k="step"):
    ra, rb = rounds(a, k), rounds(b, k)
    if not ra or len(ra) != len(rb):
        return None
    d = [x - y for x, y in zip(ra, rb)]
    pc = [100 * (x / y - 1) for x, y in zip(ra, rb)]
    return dict(median_ms=statistics.median(d), min_ms=min(d), max_ms=max(d), median_pct=statistics.median(pc),
                min_pct=min(pc), max_pct=max(pc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    args = ap.parse_args()
    rows, md = [], []
    for f in args.files:
        d = json.load(open(f))
        c = d["config"]
        name = Path(f).stem
        md.append(f"\n**{name}** (C {c['C']}, S {c['S']}, R {c['R']}, H {c['H']:,}, M {c['M']}, G {c['G']}, "
                  f"{c['precision']}; commit {d.get('commit', '')[:7]}{' dirty' if d.get('dirty') else ''})\n")
        md.append("| Variant | dX plan | dW proj. | Fwd | det fwd s1 / red | dX | det dX s1 / red | dW | Bwd | Opt | "
                  "Step [round range] | Peak alloc / res GiB (warm-up) |")
        md.append("|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|")
        for vn, v in d["variants"].items():
            info = v.get("info") or {}
            gd, gw = info.get("grouped_dx") or {}, info.get("grouped_dw") or {}
            red = (info.get("reduction") or {}).get("mode")
            r = dict(file=name, variant=vn, H=c["H"], M=c["M"], G=c["G"], S=c["S"], R=c["R"], precision=c["precision"],
                     reduction=red, dx_plan=gd.get("plan"), dx_tokens=gd.get("tokens"), dx_wg=gd.get("reduction_group_warps"),
                     dw_projection=gw.get("projection"), commit=d.get("commit", "")[:7], dirty=d.get("dirty"),
                     peak_alloc_GiB=v["peak_allocated_bytes"] / 2**30, peak_res_GiB=v["peak_reserved_bytes"] / 2**30,
                     warm_alloc_GiB=v.get("warmup_peak_allocated_bytes", 0) / 2**30,
                     warm_res_GiB=v.get("warmup_peak_reserved_bytes", 0) / 2**30,
                     det_fwd=(gd and (info.get("grouped_fwd") or {}).get("reduction")),
                     det_dx=gd.get("reduction"))
            for k in PH:
                r[k] = med(v, k)
            sr = rounds(v, "step")
            r["step_round_min"], r["step_round_max"] = (min(sr), max(sr)) if sr else (None, None)
            rows.append(r)
            fm = lambda x: "" if x is None else f"{x:.1f}"
            md.append(f"| {vn} | {gd.get('plan')} T{gd.get('tokens')}/WG{gd.get('reduction_group_warps')} | "
                      f"{gw.get('projection')} | {fm(r['fwd_total'])} | {fm(r['det_fwd_stage1'])} / {fm(r['det_fwd_reduce'])} | "
                      f"{fm(r['dx_kernel'])} | {fm(r['det_dx_stage1'])} / {fm(r['det_dx_reduce'])} | {fm(r['dw_kernel'])} | "
                      f"{fm(r['bwd_total'])} | {fm(r['optimizer'])} | {fm(r['step'])} [{fm(r['step_round_min'])}–"
                      f"{fm(r['step_round_max'])}] | {r['warm_alloc_GiB']:.2f} / {r['warm_res_GiB']:.2f} |")
        vs = d["variants"]
        for a, b in PAIRS:
            if a in vs and b in vs:
                for k in ("step", "dw_kernel", "fwd_total", "dx_kernel"):
                    p = paired(vs[a], vs[b], k)
                    if p:
                        md.append(f"* paired {a} − {b} {k}: {p['median_ms']:+.2f} ms [{p['min_ms']:+.2f}, {p['max_ms']:+.2f}]"
                                  f" ({p['median_pct']:+.2f} % [{p['min_pct']:+.2f}, {p['max_pct']:+.2f}])")
                        rows.append(dict(file=name, variant=f"{a}-{b}", paired=k, **p))
    text = "\n".join(md)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
