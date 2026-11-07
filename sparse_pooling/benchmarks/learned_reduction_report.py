"""Tables of the learned-reduction study from the raw per-job records: CSV + compact raw archives under
sparse_pooling/results/learned_sliding_window_reduction/ and markdown tables on stdout / tables.md.  Measured values only; a
missing stage or cell is printed as missing.

    python learned_reduction_report.py --out RAW_ROOT --dest sparse_pooling/results/learned_sliding_window_reduction
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

ORDER = ["A0", "A1", "A2", "A3", "A3mean", "A3gelu", "A2gelu", "A4", "A4mean", "A4gelu", "A5", "A5gelu", "MC", "TS_id", "TS_gelu"]
TASKS = ["compose", "interact", "local"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load(root: Path, stage: str):
    out = []
    d = root / stage
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if str(r.get("schema", "")).startswith("granule.lsw"):
            r["_file"] = f.name
            out.append(r)
    return out


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.{n}f}"


def md(rows, cols):
    s = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        s.append("| " + " | ".join(str(g(r)) for _, g in cols) + " |")
    return "\n".join(s)


def write_csv(path: Path, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])


def flat(r):
    d = {k: r.get(k) for k in ("H", "model", "task", "seed", "mode", "steps", "lr_w", "lr_small", "p_mult", "final_val", "final_train",
                               "gap_val_train", "best_val", "test", "ms_per_step", "gpu_seconds", "finite", "test_target_var")}
    d.update({f"shift_{k}": v for k, v in (r.get("shifts") or {}).items()})
    d["intermediate"] = r["weights"]["intermediate"]
    d["added_weights"] = r["weights"]["added_weights"]
    d["added_biases"] = r["weights"]["added_biases"]
    d["peak_gib"] = r.get("peak_alloc_bytes", 0) / 2 ** 30
    if r.get("hidden_start"):
        d.update(U_rms_start=r["hidden_start"]["U"]["rms"], U_token_std_start=r["hidden_start"].get("U_token_std_mean"),
                 relu_zero_start=r["hidden_start"]["relu_zero_fraction"], gamma=r["hidden_start"].get("gamma"))
    if r.get("hidden_end"):
        d.update(relu_zero_end=r["hidden_end"]["relu_zero_fraction"], U_rms_end=r["hidden_end"]["U"]["rms"])
    return d


KEYS = ["H", "model", "task", "seed", "mode", "steps", "lr_w", "lr_small", "p_mult", "final_val", "final_train", "gap_val_train",
        "best_val", "test", "shift_scale", "shift_corr", "test_target_var", "ms_per_step", "gpu_seconds", "finite", "intermediate",
        "added_weights", "added_biases", "peak_gib", "U_rms_start", "U_token_std_start", "U_rms_end", "relu_zero_start", "relu_zero_end", "gamma"]


def val_at(curve, x_key, x):
    """Validation loss interpolated at x along the curve; None past its end."""
    if x > curve[-1][x_key] * (1 + 1e-9):
        return None
    for a, b in zip(curve, curve[1:]):
        if a[x_key] <= x <= b[x_key]:
            f = 0 if b[x_key] == a[x_key] else (x - a[x_key]) / (b[x_key] - a[x_key])
            return a["val"] + f * (b["val"] - a["val"])
    return curve[-1]["val"]


def screen_table(rs, title):
    out = [f"\n### {title}\n"]
    for t in TASKS:
        rows = sorted([r for r in rs if r["task"] == t], key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else 99)
        if not rows:
            out.append(f"{t}: no records\n")
            continue
        var = rows[0].get("test_target_var")
        out.append(f"\n**{t}** (test target variance {fmt(var)}):\n")
        best = min((r.get("test", math.inf) for r in rows if r.get("test") is not None), default=None)
        out.append(md(rows, [("model", lambda r: r["model"]), ("J", lambda r: r["weights"]["final_intermediate"]),
                             ("added weights", lambda r: f"{r['weights']['added_weights']:,}"),
                             ("lr W / small / Pm", lambda r: f"{r['lr_w']:g} / {r['lr_small']:g} / {r['p_mult']:g}"),
                             ("val", lambda r: fmt(r["final_val"])), ("train", lambda r: fmt(r["final_train"])),
                             ("gap", lambda r: fmt(r["gap_val_train"])),
                             ("test", lambda r: ("**" + fmt(r.get("test")) + "**") if r.get("test") == best else fmt(r.get("test"))),
                             ("scale 1.5", lambda r: fmt((r.get("shifts") or {}).get("scale"))),
                             ("corr", lambda r: fmt((r.get("shifts") or {}).get("corr"))),
                             ("ms/step", lambda r: fmt(r["ms_per_step"], 1)),
                             ("finite", lambda r: "yes" if r["finite"] else "NO")]))
        out.append("")
    return "\n".join(out)


def equal_views(rs, title, steps=(256, 512, 1024, 2048, 4096)):
    out = [f"\n### {title}\n"]
    for t in sorted({r["task"] for r in rs}):
        sub = [r for r in rs if r["task"] == t]
        models = [m for m in ORDER if any(r["model"] == m for r in sub)]
        def mean(rr, f):
            v = [f(r) for r in rr]
            return st.mean(v) if v and all(x is not None for x in v) else None
        tmin = min(r["curve"][-1]["gpu_s"] for r in sub)
        tb = [tmin / 4, tmin / 2, tmin]
        rows = []
        for m in models:
            rr = [r for r in sub if r["model"] == m]
            d = dict(model=m)
            for s in steps:
                d[f"s{s}"] = mean(rr, lambda r: val_at(r["curve"], "step", s))
            for i, x in enumerate(tb):
                d[f"t{i}"] = mean(rr, lambda r: val_at(r["curve"], "gpu_s", x))
            d["end_s"] = st.mean(r["curve"][-1]["gpu_s"] for r in rr)
            rows.append(d)
        cols = [("model", lambda r: r["model"])] + [(f"val @ {s} steps", (lambda s: lambda r: fmt(r[f"s{s}"]))(s)) for s in steps]
        cols += [(f"val @ {x:.0f} s", (lambda i: lambda r: fmt(r[f"t{i}"]))(i)) for i, x in enumerate(tb)]
        cols += [("GPU s at end", lambda r: fmt(r["end_s"], 0))]
        out.append(f"\n**{t}**, mean over seeds, GPU-synchronized training seconds:\n")
        out.append(md(rows, cols))
        out.append("")
    return "\n".join(out)


def paired(rs, a, b, field="test"):
    """Per-seed a - b."""
    A = {r["seed"]: r.get(field) for r in rs if r["model"] == a and r.get(field) is not None}
    B = {r["seed"]: r.get(field) for r in rs if r["model"] == b and r.get(field) is not None}
    return {s: A[s] - B[s] for s in sorted(A) if s in B}


def contrast_rows(rs, pairs):
    out = []
    for t in sorted({r["task"] for r in rs}):
        sub = [r for r in rs if r["task"] == t]
        for a, b in pairs:
            d = paired(sub, a, b)
            if not d:
                continue
            v = list(d.values())
            m = st.mean(v)
            n = len(v)
            sd = st.stdev(v) if n > 1 else float("nan")
            hw = T95.get(n - 1, 2.0) * sd / math.sqrt(n) if n > 1 else float("nan")
            base = st.mean(r["test"] for r in sub if r["model"] == b and r.get("test") is not None)
            out.append(dict(task=t, a=a, b=b, n=n, per_seed=" / ".join(f"{x:+.5f}" for x in v), mean=m, sd=sd, ci_low=m - hw, ci_high=m + hw,
                            rel_percent=100 * m / base))
    return out


def primary_tables(rs):
    out = []
    keyf = lambda m: ORDER.index(m) if m in ORDER else 99
    for t in sorted({r["task"] for r in rs}):
        sub = [r for r in rs if r["task"] == t and r.get("test") is not None]
        rows = []
        for m in sorted({r["model"] for r in sub}, key=keyf):
            rr = sorted([r for r in sub if r["model"] == m], key=lambda r: r["seed"])
            rows.append(dict(model=m, seeds=" / ".join(f"{r['test']:.5f}" for r in rr), mean=st.mean(r["test"] for r in rr),
                             sd=st.stdev(r["test"] for r in rr) if len(rr) > 1 else float("nan"),
                             val=st.mean(r["final_val"] for r in rr), train=st.mean(r["final_train"] for r in rr),
                             gap=st.mean(r["gap_val_train"] for r in rr),
                             scale=st.mean(r["shifts"]["scale"] for r in rr), corr=st.mean(r["shifts"]["corr"] for r in rr),
                             ms=st.mean(r["ms_per_step"] for r in rr), lr=f"{rr[0]['lr_w']:g} / {rr[0]['lr_small']:g} / {rr[0]['p_mult']:g}",
                             weights=rr[0]["weights"]["added_weights"], J=rr[0]["weights"]["final_intermediate"]))
        out.append(f"\n**{t}**:\n")
        out.append(md(rows, [("model", lambda r: r["model"]), ("J", lambda r: f"{r['J']:,}"), ("added weights", lambda r: f"{r['weights']:,}"),
                             ("lr W / small / Pm", lambda r: r["lr"]), ("test per seed 0 / 1 / 2", lambda r: r["seeds"]),
                             ("test mean", lambda r: fmt(r["mean"], 5)), ("sd", lambda r: fmt(r["sd"], 5)), ("val", lambda r: fmt(r["val"], 5)),
                             ("train", lambda r: fmt(r["train"], 5)), ("gap", lambda r: fmt(r["gap"], 5)), ("scale 1.5", lambda r: fmt(r["scale"], 4)),
                             ("corr", lambda r: fmt(r["corr"], 4)), ("ms/step", lambda r: fmt(r["ms"], 1))]))
        out.append("")
    return "\n".join(out)


def perf_tables(rs):
    out = []
    phases = ["sparse_and_reduction", "compact_and_readout", "loss_and_dense_backward", "dx", "dw_db", "dp", "fused_optimizer", "dense_optimizer", "step"]
    def label(r):
        s = r["model"]
        if r["model"] in ("A0", "A1", "A2", "A5") and r.get("dx") != "mt2_nw4_wg2":
            s += f" {r['dx']}"
        cfg = r["config"]
        if cfg["kind"] == "learned":
            v = []
            if cfg["epilogue"] != "mma_split":
                v.append(cfg["epilogue"])
            if cfg["dp_mode"] != "fused" or r["wpt"] != 2:
                v.append(f"{cfg['dp_mode']} wpt{r['wpt']}")
            if (r["dx_mt"], r["dx_nw"], r["dx_wg"], r["dx_reload"]) != (2, 4, 2, 0):
                v.append(f"dX mt{r['dx_mt']} nw{r['dx_nw']} wg{r['dx_wg']} r{r['dx_reload']}")
            if v:
                s += " (" + ", ".join(v) + ")"
        if r.get("tf32"):
            s += " TF32"
        return s
    groups = {}
    for r in rs:
        groups.setdefault((r["H"], r["batch"], label(r)), []).append(r)
    for H in sorted({k[0] for k in groups}):
        for M in sorted({k[1] for k in groups if k[0] == H}):
            rows = []
            base = groups.get((H, M, "A0"))
            bs = st.median(r["median_ms"]["step"] for r in base) if base else None
            for (h, m, lab), rr in sorted(groups.items(), key=lambda kv: (ORDER.index(kv[1][0]["model"]) if kv[1][0]["model"] in ORDER else 99, kv[0][2])):
                if (h, m) != (H, M):
                    continue
                d = dict(label=lab, runs=len(rr))
                for p in phases:
                    v = [r["median_ms"].get(p) for r in rr if r["median_ms"].get(p) is not None]
                    d[p] = st.median(v) if v else None
                d["ratio"] = d["step"] / bs if bs else None
                d["spread"] = (max(r["median_ms"]["step"] for r in rr) - min(r["median_ms"]["step"] for r in rr)) / d["step"]
                d["iqr"] = st.median((r["dispersion_ms"]["step"]["q3"] - r["dispersion_ms"]["step"]["q1"]) / r["median_ms"]["step"] for r in rr)
                d["peak"] = max(r["memory"]["peak_full_step_bytes"] for r in rr) / 2 ** 30
                d["steady"] = max(r["memory"]["steady_with_optimizer_bytes"] for r in rr) / 2 ** 30
                rows.append(d)
            out.append(f"\n**H = {H:,}, M = {M:,}** (medians over rounds; ms):\n")
            cols = [("model", lambda r: r["label"]), ("runs", lambda r: r["runs"])]
            short = dict(sparse_and_reduction="sparse + reduce", compact_and_readout="finalize + readout", loss_and_dense_backward="dense bwd", dx="dX",
                         dw_db="dW/db (+dP fused)", dp="dP", fused_optimizer="FusedAdamW", dense_optimizer="AdamW", step="**step**")
            for p in phases:
                cols.append((short[p], (lambda p: lambda r: fmt(r[p], 1))(p)))
            cols += [("x A0", lambda r: fmt(r["ratio"], 2)), ("IQR/med", lambda r: fmt(r["iqr"], 3)), ("steady GiB", lambda r: fmt(r["steady"], 2)),
                     ("peak GiB", lambda r: fmt(r["peak"], 2))]
            out.append(md(rows, cols))
            out.append("")
    return "\n".join(out)


CONTROL_PAIRS = [("A1", "A0"), ("A2", "A0"), ("A3", "A0"), ("A3mean", "A0"), ("A4", "A0"), ("A4mean", "A0"), ("A5", "A0"), ("MC", "A0"),
                 ("A2", "A1"), ("A5", "A1"), ("A5", "A2"), ("A3", "A2"), ("A3mean", "A2"), ("A4", "A5"), ("A4mean", "A5"), ("A3mean", "A3"),
                 ("A4mean", "A4"), ("A4", "A3"), ("A4mean", "A3mean"), ("A4", "MC"), ("A4mean", "MC")]


def contrast_table(rows):
    return md(rows, [("task", lambda r: r["task"]), ("a - b", lambda r: f"{r['a']} - {r['b']}"), ("per seed", lambda r: r["per_seed"]),
                     ("mean", lambda r: f"{r['mean']:+.5f}"), ("sd", lambda r: fmt(r["sd"], 5)),
                     ("95 % CI", lambda r: f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"), ("% of b", lambda r: f"{r['rel_percent']:+.2f}")])


def supplementary_tables(allr, dest):
    """Stages declared after the main campaign's test losses were seen; see learned_reduction_campaign.py."""
    out = []
    if allr["control_final"]:
        both = allr["primary_final"] + allr["control_final"]
        out.append("\n## Supplementary (declared after the main campaign's test losses were seen)\n")
        out.append("\n### Full matrix, H = 10,485,760, three paired seeds, 4096 updates (primary_final + control_final)\n")
        out.append(primary_tables(both))
        out.append(equal_views(both, "Full matrix, equal steps and equal GPU time (validation loss, mean over seeds)"))
        rows = contrast_rows(both, CONTROL_PAIRS)
        write_csv(dest / "csv" / "matrix_contrasts.csv", rows, list(rows[0].keys()))
        out.append("\n### Full matrix, paired test-loss contrasts (a - b; negative: a better; 95 % t interval, n = 3; not adjusted for multiple comparisons)\n")
        out.append(contrast_table(rows))
    if allr["control_tune"]:
        out.append("\n### Control tuning, 512 updates, validation\n")
        out.append(md(sorted(allr["control_tune"], key=lambda r: (r["task"], ORDER.index(r["model"]), r["final_val"])), [
            ("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("lr W / small / Pm", lambda r: f"{r['lr_w']:g} / {r['lr_small']:g} / {r['p_mult']:g}"),
            ("val", lambda r: fmt(r["final_val"], 5))]))
    if allr["horizon"]:
        models = sorted({r["model"] for r in allr["horizon"]}, key=ORDER.index)
        out.append(f"\n### Longer horizon, H = 10,485,760, three paired seeds, 8192 updates: {' and '.join(models)}\n")
        out.append(primary_tables(allr["horizon"]))
        out.append(equal_views(allr["horizon"], "Longer horizon, equal steps and equal GPU time", steps=(1024, 2048, 4096, 6144, 8192)))
        rows = contrast_rows(allr["horizon"], [(m, "A0") for m in models if m != "A0"])
        write_csv(dest / "csv" / "horizon_contrasts.csv", rows, list(rows[0].keys()))
        out.append("\n**Paired test-loss contrasts at 8192 updates:**\n")
        out.append(contrast_table(rows))
    if allr["fresh"]:
        models = sorted({r["model"] for r in allr["fresh"]}, key=ORDER.index)
        out.append(f"\n### Fresh data, H = 10,485,760, three paired seeds, 4096 updates, no repeated training batch (pool 4096): {', '.join(models)}; LR not re-tuned\n")
        out.append(primary_tables(allr["fresh"]))
        out.append(equal_views(allr["fresh"], "Fresh data, equal steps and equal GPU time"))
        rows = contrast_rows(allr["fresh"], [(a, b) for i, a in enumerate(models) for b in models[:i]])
        write_csv(dest / "csv" / "fresh_contrasts.csv", rows, list(rows[0].keys()))
        out.append("\n**Paired test-loss contrasts on fresh data:**\n")
        out.append(contrast_table(rows))
    for name, title in (("secondary_leader", "Winner at the secondary width, H = 12,582,912, compose, seed 0, 4096 updates"),
                        ("m8192", "Training at M = 8,192 (16 distinct pool batches per step), H = 10,485,760, compose, seed 0, 128 updates, LR selected at M = 512")):
        if allr[name]:
            out.append(f"\n### {title}\n")
            out.append(md(sorted(allr[name], key=lambda r: ORDER.index(r["model"])), [
                ("model", lambda r: r["model"]), ("J", lambda r: f"{r['weights']['final_intermediate']:,}"), ("added weights", lambda r: f"{r['weights']['added_weights']:,}"),
                ("val start", lambda r: fmt(r["curve"][0]["val"], 5)), ("val end", lambda r: fmt(r["final_val"], 5)), ("train end", lambda r: fmt(r["final_train"], 5)),
                ("test", lambda r: fmt(r.get("test"), 5)), ("ms/step", lambda r: fmt(r["ms_per_step"], 1)), ("peak alloc GiB", lambda r: fmt(r["peak_alloc_bytes"] / 2 ** 30, 2)),
                ("finite", lambda r: r["finite"])]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    root, dest = Path(a.out), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tx = ["# Learned sliding-window reduction: generated tables\n\nGenerated by `learned_reduction_report.py` from the raw per-job records.\n"]
    stages = ("screen_tune1", "screen_tune2", "screen_final", "frozen_tune", "frozen_final", "primary_smoke", "primary_tune", "primary_final", "secondary",
              "control_tune", "control_final", "horizon", "fresh", "secondary_leader", "m8192")
    allr = {s: load(root, s) for s in stages}
    for s, rs in allr.items():
        if rs:
            write_csv(dest / "csv" / f"{s}.csv", [flat(r) for r in rs], KEYS)
    tx.append(screen_table(allr["screen_final"], "Screening, end-to-end, H = 2,621,440, seed 0, 4096 updates (validation-selected LR; test evaluated once)"))
    tx.append(equal_views(allr["screen_final"], "Screening, equal steps and equal GPU time (validation loss)"))
    tx.append(screen_table(allr["frozen_final"], "Screening, frozen W / b / proj: readout only, H = 2,621,440, seed 0, 2048 updates"))
    if allr["primary_smoke"]:
        tx.append("\n### Primary width optimization smoke, H = 10,485,760, compose, 256 updates (no test)\n")
        tx.append(md(sorted(allr["primary_smoke"], key=lambda r: ORDER.index(r["model"])), [
            ("model", lambda r: r["model"]), ("val start", lambda r: fmt(r["curve"][0]["val"])), ("val end", lambda r: fmt(r["final_val"])),
            ("train end", lambda r: fmt(r["final_train"])), ("ms/step", lambda r: fmt(r["ms_per_step"], 1)), ("finite", lambda r: r["finite"])]))
    if allr["primary_tune"]:
        tx.append("\n### Primary tuning, 512 updates, validation\n")
        tx.append(md(sorted(allr["primary_tune"], key=lambda r: (r["task"], ORDER.index(r["model"]), r["final_val"])), [
            ("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("lr W / small / Pm", lambda r: f"{r['lr_w']:g} / {r['lr_small']:g} / {r['p_mult']:g}"),
            ("val", lambda r: fmt(r["final_val"], 5))]))
    pf = allr["primary_final"]
    if pf:
        tx.append("\n### Confirmation, H = 10,485,760, three paired seeds (graph, init and training seeds move together), 4096 updates\n")
        tx.append(primary_tables(pf))
        tx.append(equal_views(pf, "Confirmation, equal steps and equal GPU time (validation loss, mean over seeds)"))
        models = {r["model"] for r in pf}
        strongest = [m for m in ("A3", "A3gelu", "A3mean", "A2") if m in models]
        pairs = [(m, "A0") for m in sorted(models - {"A0"}, key=lambda m: ORDER.index(m))] + [("A4", "A1")]
        pairs += [("A4", "MC")] + [("A4", s) for s in strongest] + [(s, "A1") for s in strongest] + [("MC", "A1")]
        if "A4gelu" in models:
            pairs.append(("A4gelu", "A4"))
        rows = contrast_rows(pf, pairs)
        write_csv(dest / "csv" / "primary_contrasts.csv", rows, list(rows[0].keys()) if rows else [])
        tx.append("\n### Paired test-loss contrasts, confirmation (a - b; negative: a better; 95 % t interval, n = 3)\n")
        tx.append(md(rows, [("task", lambda r: r["task"]), ("a - b", lambda r: f"{r['a']} - {r['b']}"), ("per seed", lambda r: r["per_seed"]),
                            ("mean", lambda r: f"{r['mean']:+.5f}"), ("sd", lambda r: fmt(r["sd"], 5)),
                            ("95 % CI", lambda r: f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"), ("% of b", lambda r: f"{r['rel_percent']:+.2f}")]))
    if allr["secondary"]:
        tx.append("\n### Secondary width, H = 12,582,912, compose, seed 0, 1024 updates\n")
        tx.append(md(allr["secondary"], [("model", lambda r: r["model"]), ("J", lambda r: f"{r['weights']['final_intermediate']:,}"),
                                         ("added weights", lambda r: f"{r['weights']['added_weights']:,}"), ("val", lambda r: fmt(r["final_val"], 5)),
                                         ("test", lambda r: fmt(r.get("test"), 5)), ("ms/step", lambda r: fmt(r["ms_per_step"], 1))]))
    tx.append(supplementary_tables(allr, dest))
    perf = load(root, "perf")
    if perf:
        tx.append("\n### Performance and memory (eager; one workload at a time)\n")
        tx.append(perf_tables(perf))
        rows = []
        for r in perf:
            d = dict(H=r["H"], model=r["model"], batch=r["batch"], dx=r["dx"], epilogue=r["config"]["epilogue"], dp_mode=r["config"]["dp_mode"], wpt=r["wpt"],
                     dx_mt=r["dx_mt"], dx_nw=r["dx_nw"], dx_wg=r["dx_wg"], dx_reload=r["dx_reload"], tf32=r["tf32"], rep=r.get("rep", 0))
            d.update(r["median_ms"])
            d.update({k: v for k, v in r["memory"].items() if k.endswith("bytes")})
            rows.append(d)
        keys = sorted({k for d in rows for k in d})
        write_csv(dest / "csv" / "perf.csv", rows, keys)
        with gzip.open(dest / "raw_perf.jsonl.gz", "wt") as f:
            for r in perf:
                r = dict(r)
                r.pop("rounds_detail", None)
                f.write(json.dumps(r, default=str) + "\n")
    with gzip.open(dest / "raw_quality.jsonl.gz", "wt") as f:
        for s, rs in allr.items():
            for r in rs:
                r = dict(r, stage=s)
                r.pop("kernel", None)
                f.write(json.dumps(r, default=str) + "\n")
    gpu = {s: sum(r.get("wall_seconds", 0) for r in rs) / 3600 for s, rs in allr.items()}
    gpu_perf = sum(sum(sum(x["step"] for x in rd["samples"]) for rd in r.get("rounds_detail", [])) for r in perf) / 3.6e6 if perf else 0
    tx.append("\n### Campaign accounting\n")
    tx.append(md([dict(s=s, n=len(rs), h=gpu[s]) for s, rs in allr.items()], [("stage", lambda r: r["s"]), ("jobs", lambda r: r["n"]), ("training + evaluation hours", lambda r: fmt(r["h"], 2))]))
    tx.append(f"\nTotal training + evaluation: {sum(gpu.values()):.2f} h; timed performance samples {gpu_perf:.2f} h (excludes warm-up, construction, and memory passes).\n")
    fails = [json.loads(ln) for s in stages for ln in ((root / s / "failures.jsonl").read_text().splitlines() if (root / s / "failures.jsonl").exists() else [])]
    tx.append(f"Failed jobs: {len(fails)}.\n")
    (dest / "tables.md").write_text("\n".join(tx))
    print("\n".join(tx))


if __name__ == "__main__":
    main()
