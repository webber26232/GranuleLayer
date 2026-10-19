"""Summary of det_training.py outputs: per run the repeatability verdict (tensors x steps compared,
first difference), and for --characterize runs (fast mode) the per-step deviation of key tensors
(relative L2 / max abs; n_diff / n) against the reference records.

    python summarize_det_training.py ../results/determinism_metrics_dw_projection/det_training/*.json [--md out.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ["y", "loss", "dX", "dW", "db", "W", "opt.m0", "proj.W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md")
    args = ap.parse_args()
    rep, char = [], []
    rep.append("| Run | Case | Seed | Precision | Optimizer | Reduction (resolved) | Role | Steps x tensors compared | Bitwise | First difference |")
    rep.append("|---|---|---:|---|---|---|---|---:|---|---|")
    for f in args.files:
        d = json.load(open(f))
        a = d["args"]
        trajs = d["warm"] + d["trajectory1"] + d["trajectory2"]
        n = sum(s["compared"] for s in trajs)
        first = next((s for s in trajs if s["differing"]), None)
        red = (d.get("info") or {}).get("reduction") or {}
        fd = "" if first is None else f"step {first['step']}: {first['first'][0]} (element {first['first'][1].get('first_index')})"
        rep.append(f"| {Path(f).stem} | {a['case']} | {a['seed']} | {a['precision']} | {a['opt_mode']} | {red.get('mode')} | "
                   f"{a['role']} | {n} | {'yes' if d['all_bitwise'] else 'no'} | {fd} |")
        if a.get("characterize"):
            for lab in ("trajectory1", "trajectory2"):
                for s in d[lab]:
                    det = s.get("detail") or {}
                    if not det:
                        continue
                    cells = []
                    for k in KEYS:
                        r = det.get(k)
                        if r is None:
                            cells.append("")
                        elif r.get("bitwise"):
                            cells.append("=")
                        else:
                            cells.append(f"{r.get('rel_l2', float('nan')):.1e} ({r['n_diff'] / r['n']:.0%})")
                    char.append(f"| {Path(f).stem} | {lab} | {s['step']} | " + " | ".join(cells) + " |")
    out = "\n".join(rep)
    if char:
        out += ("\n\nDeviation from the reference records (relative L2; share of differing elements; '=' bitwise):\n\n"
                "| Run | Trajectory | Step | " + " | ".join(KEYS) + " |\n|---|---|---:|" + "---|" * len(KEYS) + "\n"
                + "\n".join(char))
    print(out)
    if args.md:
        Path(args.md).write_text(out + "\n")


if __name__ == "__main__":
    main()
