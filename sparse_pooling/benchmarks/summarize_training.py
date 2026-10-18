"""Summary of numerics_training.py output: per configuration the final-step divergences
(max over equal numbers of run pairs) of every variant next to the null control
(baseline_null: an independent set of baseline runs), and how often the predeclared check C2 /
C3 failed for the null control versus each candidate.

    python summarize_training.py results/c8_dx_scatter/training_validation_split.json
"""
from __future__ import annotations

import collections
import json
import sys

KEYS = ("W", "dX", "m", "loss")


def main():
    d = json.load(open(sys.argv[1]))
    fails = collections.Counter()
    total = collections.Counter()
    ratios = collections.defaultdict(list)  # candidate / null, final step, per quantity
    rows = []
    for r in d["results"]:
        if not all(r["finite"].values()):
            print("NON-FINITE", r["graph"], r["cfg"]["model"], r["opt_mode"], r["finite"])
        null = r["variants"].get("baseline_null")
        for name, v in r["variants"].items():
            last = v["series"][-1]
            if "check" in v:
                total[name] += 1
                fails[name] += not v["check"]["pass"]
            if null is not None and name != "baseline_null":
                nl = null["series"][-1]
                for k in KEYS:
                    if k in last and nl.get(k, 0) > 0:
                        ratios[(name, k)].append(last[k] / nl[k])
            rows.append((r["graph"], r["cfg"]["model"], r["opt_mode"], name,
                         " ".join(f"{k} {last.get(k, float('nan')):.2e}" for k in KEYS)))
    for row in rows:
        print("{:18s} {:8s} {:5s} {:22s} {}".format(*row))
    print("\npredeclared check failures (configurations failing / checked):")
    for name in sorted(total):
        print(f"  {name:22s} {fails[name]:3d} / {total[name]}")
    print("\nfinal-step divergence relative to the null control (median, max over configurations):")
    for (name, k), v in sorted(ratios.items()):
        v = sorted(v)
        print(f"  {name:22s} {k:5s} median {v[len(v) // 2]:.2f}  max {v[-1]:.2f}  (n = {len(v)})")


if __name__ == "__main__":
    main()
