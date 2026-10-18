"""Tables of bench_supergroups.py results for the dX scatter experiment: per variant the phase
medians (forward, coefficient preparation, input layout, dX init, dX kernel, dX
post-processing, dX total, dW, backward, optimizer, step), round spread, kernels, memory and
GPU state; a CSV with every row and the change of each variant against the file's baseline.

    python summarize_c8dx.py results/c8_dx_scatter/final_100M/*.json --csv results/c8_dx_scatter/final_100M.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PH = ["fwd_total", "bwd_coef", "bwd_layout", "bwd_dx_init", "dx_kernel", "bwd_post", "dw_kernel", "bwd_total",
      "optimizer", "step"]


def med(v, ph):
    p = v["phases"].get(ph)
    return None if p is None else p["median_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = []
    for f in args.files:
        d = json.load(open(f))
        c = d["config"]
        base = d["variants"].get("baseline")
        for name, v in d["variants"].items():
            gd = (v.get("info") or {}).get("grouped_dx") or {}
            r = {"file": Path(f).stem, "protocol": d.get("protocol", "historical"), "commit": d["commit"][:10],
                 "dirty": d["dirty"], "C": c["C"], "S": c["S"], "R": c["R"], "precision": c["precision"], "M": c["M"],
                 "H": c["H"], "seed": d["args"]["graph_seed"], "variant": name, "dx": v["kernels"]["dx"],
                 "scatter": gd.get("scatter", ""), "order": gd.get("scatter_order", ""), "T": gd.get("tokens", ""),
                 "NW": gd.get("warps", ""), "WG": gd.get("reduction_group_warps", ""), "kernel": gd.get("kernel", "")}
            for ph in PH:
                r[ph] = med(v, ph)
            r["dx_total"] = sum(r[p] for p in ("bwd_dx_init", "dx_kernel", "bwd_post") if r[p] is not None)
            st = v["phases"]["step"]
            r["step_round_min"], r["step_round_max"] = min(st["round_medians_ms"]), max(st["round_medians_ms"])
            r["n_steps"] = st["n"]
            r["peak_alloc_GiB"] = v["peak_allocated_bytes"] / 2**30
            r["peak_reserved_GiB"] = v["peak_reserved_bytes"] / 2**30
            r["warmup_peak_alloc_GiB"] = v.get("warmup_peak_allocated_bytes", 0) / 2**30
            r["step_scratch_MiB"] = v["step_scratch_peak_bytes"] / 2**20
            r["work_list_MiB"] = v.get("work_list_bytes", 0) / 2**20
            r["snapshot_GiB"] = (d.get("snapshot") or {}).get("bytes", 0) / 2**30
            r["snapshot_device"] = (d.get("snapshot") or {}).get("device", "")
            r["restore_checks"] = v.get("restore_checks", 0)
            r["gpu_temp_before_after"] = f"{d['gpu_before'].get('temperature.gpu')}/{d['gpu_after'].get('temperature.gpu')}"
            r["sm_clock_before_after"] = f"{d['gpu_before'].get('clocks.sm')}/{d['gpu_after'].get('clocks.sm')}"
            r["power_before_after"] = f"{d['gpu_before'].get('power.draw')}/{d['gpu_after'].get('power.draw')}"
            if base is not None:
                for ph in ("dx_kernel", "step"):
                    b = med(base, ph)
                    r[f"{ph}_vs_baseline_ms"] = None if b is None or r[ph] is None else r[ph] - b
                bt = sum(med(base, p) or 0 for p in ("bwd_dx_init", "dx_kernel", "bwd_post"))
                r["dx_total_vs_baseline_ms"] = r["dx_total"] - bt
                r["step_speedup"] = med(base, "step") / r["step"]
            rows.append(r)
            print(f"{r['file']:28s} {name:30s} {r['scatter']:28s} T{r['T']} W{r['NW']} G{r['WG']}  fwd {r['fwd_total']:7.1f} "
                  f"dX {r['dx_kernel']:7.1f} (total {r['dx_total']:7.1f}) dW {r['dw_kernel']:7.1f} opt {r['optimizer']:6.1f} "
                  f"step {r['step']:7.1f} [{r['step_round_min']:.1f}-{r['step_round_max']:.1f}]"
                  + (f"  {r['step_vs_baseline_ms']:+7.1f} ms x{r['step_speedup']:.3f}" if base is not None else "")
                  + f"  peak {r['peak_alloc_GiB']:.2f} GiB", flush=True)
    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
