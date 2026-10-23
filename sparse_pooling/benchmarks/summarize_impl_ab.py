"""Summarize run_impl_ab.sh: paired per-round differences B - A of complete-step and phase medians,
selected plans, build identities, memory peaks and GPU state.

    python summarize_impl_ab.py <output dir of run_impl_ab.sh> --json summary.json --md summary.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PHASES = ("step", "fwd_total", "dx_kernel", "dw_kernel", "bwd_total", "optimizer")


def load(d: Path):
    runs = defaultdict(dict)  # (config, variant) -> {(round, impl): record}
    for f in sorted(d.glob("*.json")):
        m = re.match(r"(.+)_r(\d+)_([AB])\.json$", f.name)
        if not m:
            continue
        name, r, impl = m.group(1), int(m.group(2)), m.group(3)
        out = json.loads(f.read_text())
        for var, v in out["variants"].items():
            runs[(name, var)][(r, impl)] = dict(phases={p: (v["phases"].get(p) or {}).get("median_ms") for p in PHASES},
                                                info=v.get("info"), peak_alloc=v["peak_allocated_bytes"],
                                                peak_res=v["peak_reserved_bytes"], kernels=v.get("kernels"),
                                                commit=out.get("commit"), dirty=out.get("dirty"),
                                                build=out.get("build") or {"library": out.get("library")},
                                                gpu_before=out.get("gpu_before"), gpu_after=out.get("gpu_after"))
    return runs


def plan_sig(info):
    if not info:
        return None
    gd, gw, gf = info.get("grouped_dx") or {}, info.get("grouped_dw") or {}, info.get("grouped_fwd") or {}
    return dict(reduction=(info.get("reduction") or {}).get("mode"),
                fwd=(gf.get("tokens"), gf.get("warps")) if gf else None,
                dx=(gd.get("plan"), gd.get("tokens"), gd.get("warps"), gd.get("reduction_group_warps"), gd.get("scatter"),
                    gd.get("kernel")) if gd else None,
                dw=(gw.get("cfg"), gw.get("projection"), gw.get("kernel")) if gw else None,
                epilogues=(info.get("dx_epilogue"), info.get("dw_epilogue")), dw_staging=info.get("dw_staging"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = load(Path(a.dir))
    rows, lines = [], ["| Config | Variant | A step ms | B step ms | paired B − A (median [min, max]) | % | rounds | "
                       "same plan | peak alloc A / B GiB |", "|---|---|---:|---:|---|---:|---:|---|---|"]
    for (name, var), recs in sorted(runs.items()):
        rounds = sorted({r for r, _ in recs if (r, "A") in recs and (r, "B") in recs})
        if not rounds:
            continue
        diff = {}
        for p in PHASES:
            d = [recs[(r, "B")]["phases"][p] - recs[(r, "A")]["phases"][p] for r in rounds
                 if recs[(r, "A")]["phases"][p] is not None and recs[(r, "B")]["phases"][p] is not None]
            if d:
                base = statistics.median(recs[(r, "A")]["phases"][p] for r in rounds)
                diff[p] = dict(median_ms=statistics.median(d), min_ms=min(d), max_ms=max(d),
                               pct=100 * statistics.median(d) / base, a_median_ms=base,
                               b_median_ms=statistics.median(recs[(r, "B")]["phases"][p] for r in rounds))
        sigs_a = {json.dumps(plan_sig(recs[(r, "A")]["info"])) for r in rounds}
        sigs_b = {json.dumps(plan_sig(recs[(r, "B")]["info"])) for r in rounds}
        same = sigs_a == sigs_b and len(sigs_a) == 1
        r0 = rounds[0]
        row = dict(config=name, variant=var, rounds=len(rounds), paired=diff, same_plan=same,
                   plan_a=json.loads(sorted(sigs_a)[0]), plan_b=json.loads(sorted(sigs_b)[0]),
                   peak_alloc_a=recs[(r0, "A")]["peak_alloc"], peak_alloc_b=recs[(r0, "B")]["peak_alloc"],
                   peak_res_a=recs[(r0, "A")]["peak_res"], peak_res_b=recs[(r0, "B")]["peak_res"],
                   commit_a=recs[(r0, "A")]["commit"], commit_b=recs[(r0, "B")]["commit"],
                   dirty_b=recs[(r0, "B")]["dirty"], build_a=recs[(r0, "A")]["build"], build_b=recs[(r0, "B")]["build"],
                   gpu=[recs[(r, i)]["gpu_before"] for r in rounds for i in "AB"])
        rows.append(row)
        s = diff.get("step")
        if s:
            lines.append(f"| {name} | {var} | {s['a_median_ms']:.1f} | {s['b_median_ms']:.1f} | "
                         f"{s['median_ms']:+.2f} [{s['min_ms']:+.2f}, {s['max_ms']:+.2f}] | {s['pct']:+.2f} | {len(rounds)} | "
                         f"{'yes' if same else 'NO'} | {row['peak_alloc_a'] / 2**30:.2f} / {row['peak_alloc_b'] / 2**30:.2f} |")
    Path(a.json).write_text(json.dumps(dict(schema="granule.impl_ab/1", rows=rows), indent=1, default=str))
    Path(a.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
