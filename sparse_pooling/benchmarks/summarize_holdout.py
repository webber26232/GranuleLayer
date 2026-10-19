"""Rule evaluation table (held-out / final runs): per file, the mean work units per NONEMPTY
bucket (u/nnz; schema-1 JSONs log it as units_per_bucket_mean) and the bucket occupancy (schema 2
only, else "-"), the plan the
automatic rule ran, Plan A / Plan B / auto step medians with round ranges, the better plan, and
the auto choice's regret vs the better plan (and gain vs Plan A / the previous BF16 default).

    python summarize_holdout.py results/scale_dispatch_fused_backward/holdout/*.json
"""
import json
import sys
from pathlib import Path


def med(v, k="step"):
    p = v["phases"].get(k)
    return (p["median_ms"], min(p["round_medians_ms"]), max(p["round_medians_ms"])) if p else (float("nan"),) * 3


def main():
    print(f"{'run':24s} {'prec':5s} {'M':>5s} {'G':>5s} {'R':>3s} {'H':>11s} {'u/nnz':>6s} {'occ':>5s} {'auto ran':12s} "
          f"{'A step':>17s} {'B step':>17s} {'auto step':>17s} {'B vs A':>7s} {'auto regret':>11s} {'auto vs A':>9s} {'vs prev':>8s}")
    for f in sys.argv[1:]:
        d = json.load(open(f))
        c, vs = d["config"], d["variants"]
        A, B, U = vs.get("planA"), vs.get("planB"), vs.get("auto")
        T = vs.get("prev_bf16_default")
        de = (A or U)["descriptors"]
        ub = de.get("mean_units_per_nonempty_bucket", de["units_per_bucket_mean"])  # schema 2 / schema 1 name
        occ = f"{de['bucket_occupancy']:5.3f}" if "bucket_occupancy" in de else "    -"
        gd = (U["info"]["grouped_dx"] or {})
        a, b, u = med(A), med(B), med(U)
        best = min(a[0], b[0])
        prev = f"{100 * (u[0] / med(T)[0] - 1):+7.2f}%" if T else ""
        print(f"{Path(f).stem:24s} {c['precision']:5s} {c['M']:5d} {c['G']:5d} {c['R']:3d} {c['H']:11,d} {ub:6.2f} {occ} "
              f"{gd.get('plan', '?'):12s} {a[0]:8.1f} [{a[1]:.0f}-{a[2]:.0f}] {b[0]:8.1f} [{b[1]:.0f}-{b[2]:.0f}] "
              f"{u[0]:8.1f} [{u[1]:.0f}-{u[2]:.0f}] {100 * (b[0] / a[0] - 1):+6.2f}% {100 * (u[0] / best - 1):+10.2f}% "
              f"{100 * (u[0] / a[0] - 1):+8.2f}% {prev}")


if __name__ == "__main__":
    main()
