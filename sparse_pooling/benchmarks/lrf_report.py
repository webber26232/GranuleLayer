"""Tables of the A3mean follow-up from its raw per-job records (schema granule.lrf_*).  Measured values only; a missing cell prints as missing.
Validation and test are labelled separately, and final weights are kept apart from validation-selected checkpoints.  Only A3mean - A0 at 16,384
M_eff = 512 updates (Compose, Interact) are primary; every other contrast is secondary or exploratory.

    python lrf_report.py --out RUN --dest sparse_pooling/results/learned_reduction_followup
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

MODELS = ["A0", "A2", "A3mean"]
TASKS = ["compose", "interact", "local"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
SMALL_PERCENT = 2.0  # descriptive engineering rule, declared in the manifest; not a significance claim
CONTRASTS = [("A3mean", "A0"), ("A3mean", "A2"), ("A2", "A0")]


def load(root: Path, stage: str, sub=""):
    out = []
    d = root / sub / stage if sub else root / stage
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if str(r.get("schema", "")).startswith("granule.lrf"):
            r["_file"] = f.name
            out.append(r)
    return out


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a" if x is None else str(x)
    return f"{x:.{n}f}"


def md(rows, cols):
    head = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "|".join(" --- " for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(str(c[1](r)) for c in cols) + " |" for r in rows) + "\n"


def write_csv(path: Path, rows, keys=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = keys or sorted({k for d in rows for k in d})
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def t2_p(t):
    """Two-sided p of a t statistic with 2 degrees of freedom (closed form, n = 3 seeds)."""
    return 1.0 - abs(t) / math.sqrt(2.0 + t * t)


def holm(ps):
    """Holm-adjusted p values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(a: dict, b: dict):
    """a, b: seed -> value.  Paired a - b over common seeds; negative: a better."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n == 0:
        return None
    mean_b = sum(b[s] for s in seeds) / n
    mean = sum(d) / n
    r = dict(n=n, seeds=seeds, per_seed=[round(x, 6) for x in d], a_per_seed=[a[s] for s in seeds], b_per_seed=[b[s] for s in seeds], mean_a=sum(a[s] for s in seeds) / n, mean_b=mean_b,
             mean=mean, pct=100 * mean / mean_b)
    if n > 1:
        sd = st.stdev(d)
        hw = T95[n - 1] * sd / math.sqrt(n)
        r.update(sd=sd, ci_low=mean - hw, ci_high=mean + hw, ci_pct_low=100 * (mean - hw) / mean_b, ci_pct_high=100 * (mean + hw) / mean_b)
        if n == 3:
            t = mean / (sd / math.sqrt(n)) if sd > 0 else math.inf
            r.update(t=t, p=t2_p(t))
    return r


def contrast_row(task, a, b, res, label):
    if res is None:
        return dict(task=task, a=a, b=b, point=label, missing=True)
    return dict(task=task, a=a, b=b, point=label, **res)


def contrast_md(rows):
    return md(rows, [("task", lambda r: r["task"]), ("point", lambda r: r["point"]), ("a - b", lambda r: f"{r['a']} - {r['b']}"),
                     ("mean a", lambda r: fmt(r.get("mean_a"), 5)), ("mean b", lambda r: fmt(r.get("mean_b"), 5)), ("per seed", lambda r: " ".join(f"{x:+.5f}" for x in r.get("per_seed", []))),
                     ("% of b", lambda r: f"{r['pct']:+.2f}" if "pct" in r else "n/a"), ("95 % CI, % of b", lambda r: f"[{r['ci_pct_low']:+.2f}, {r['ci_pct_high']:+.2f}]" if "ci_pct_low" in r else "n/a"),
                     ("p", lambda r: fmt(r.get("p"), 3))])


def key(r):
    return (r["model"], r["task"], r["seed"])


def by_key(rs):
    return {key(r): r for r in rs}


def test_at(tests, model, task, update):
    d = {}
    for r in tests:
        if r["model"] == model and r["task"] == task and str(update) in r["milestones"]:
            d[r["seed"]] = r["milestones"][str(update)]["test"]
    return d


def curve_at(r, field, x, y="val"):
    """Linear interpolation of y at x along a curve; None outside the measured range (no extrapolation)."""
    c = [e for e in r["curve"] if e.get(y) is not None and e.get(field) is not None]
    if not c or x < c[0][field] or x > c[-1][field]:
        return None
    for e0, e1 in zip(c, c[1:]):
        if e0[field] <= x <= e1[field]:
            w = 0.0 if e1[field] == e0[field] else (x - e0[field]) / (e1[field] - e0[field])
            return e0[y] + w * (e1[y] - e0[y])
    return c[-1][y]


def val_at(rs, model, task, field, x, y="val"):
    d = {}
    for r in rs:
        if r["model"] == model and r["task"] == task:
            v = curve_at(r, field, x, y)
            if v is not None:
                d[r["seed"]] = v
    return d


def mean(d):
    return sum(d.values()) / len(d) if d else None


def milestone_tables(tests, rs, tasks, updates, title):
    out = [f"\n### {title}\n"]
    rows = []
    for t in tasks:
        for m in MODELS:
            row = dict(task=t, model=m)
            for u in updates:
                row[f"val{u}"] = mean(val_at(rs, m, t, "update", u))
                row[f"test{u}"] = mean(test_at(tests, m, t, u))
            rows.append(row)
    cols = [("task", lambda r: r["task"]), ("model", lambda r: r["model"])]
    for u in updates:
        cols.append((f"val u{u}", (lambda u: lambda r: fmt(r[f"val{u}"], 5))(u)))
        cols.append((f"**test** u{u}", (lambda u: lambda r: fmt(r[f"test{u}"], 5))(u)))
    out.append(md(rows, cols))
    return "\n".join(out), rows


def selection_table(tune, ext):
    rows = sorted(tune + ext, key=lambda r: (r["task"], MODELS.index(r["model"]), r["lr_multiplier"]))
    best = {}
    for r in rows:
        k = (r["model"], r["task"])
        if k not in best or r["score_val"] < best[k]["score_val"]:
            best[k] = r
    return md(rows, [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("x prior", lambda r: f"{r['lr_multiplier']:.4g}"), ("lr W", lambda r: f"{r['lr_w']:.4g}"), ("lr small", lambda r: f"{r['lr_small']:.4g}"),
                     ("P mult", lambda r: r["p_mult"]), ("score (val, last 4 evals)", lambda r: fmt(r["score_val"], 5)), ("final val", lambda r: fmt(r["final_val"], 5)), ("finite", lambda r: r["finite"]),
                     ("selected", lambda r: "yes" if best[(r["model"], r["task"])] is r else "")]), best


def late_rows(rs, tasks, u0, u1, u2):
    rows = []
    for t in tasks:
        for m in MODELS:
            a, b, c = val_at(rs, m, t, "update", u0), val_at(rs, m, t, "update", u1), val_at(rs, m, t, "update", u2)
            rows.append(dict(task=t, model=m, v0=mean(a), v1=mean(b), v2=mean(c), drop01=100 * (1 - mean(b) / mean(a)) if a and b else None, drop12=100 * (1 - mean(c) / mean(b)) if b and c else None))
    return rows


def diagnostics(rs, tasks):
    rows = []
    for t in tasks:
        for m in MODELS:
            rr = [r for r in rs if r["model"] == m and r["task"] == t]
            if not rr:
                continue
            last = [r["curve"][-1] for r in rr]
            f = lambda k: mean({i: e[k] for i, e in enumerate(last) if e.get(k) is not None})
            mx = max(max(e["val"] for e in r["curve"][1:]) for r in rr)
            rise = max(max(e1["val"] / e0["val"] - 1 for e0, e1 in zip(r["curve"][len(r["curve"]) // 2:], r["curve"][len(r["curve"]) // 2 + 1:])) for r in rr)
            rows.append(dict(task=t, model=m, running=f("running"), recent=f("recent_seen"), fixed=f("fixed_seen"), val=f("val"), gap_fixed=f("val") - f("fixed_seen"), gap_recent=f("val") - f("recent_seen"), max_val=mx, worst_late_rise=100 * rise, finite=all(r["finite"] for r in rr)))
    return rows


def p_rows(rs):
    rows = []
    for r in sorted(rs, key=lambda r: (r["task"], r["seed"])):
        if r["model"] != "A3mean":
            continue
        for ps in (r["p_stats"][0], r["p_stats"][-1]):
            rows.append(dict(task=r["task"], seed=r["seed"], **ps))
    return rows


def grad_rows(rs):
    rows = []
    for r in rs:
        if not r["grad_stats"]:
            continue
        g0, g1 = r["grad_stats"][0], r["grad_stats"][-1]
        rows.append(dict(task=r["task"], model=r["model"], seed=r["seed"], first=g0["update"], last=g1["update"], W0=g0["W_ratio"], W1=g1["W_ratio"], P0=g0.get("P_ratio"), P1=g1.get("P_ratio"), V0=g0.get("V_ratio"), V1=g1.get("V_ratio")))
    return rows


def equal_time(rs_a, rs_b, tasks, field, grid_frac=(0.25, 0.5, 0.75, 1.0), label_a="", label_b=""):
    """Validation at common times, only within the overlap of the two sets of runs."""
    rows = []
    for t in tasks:
        for m in MODELS:
            a = [r for r in rs_a if r["model"] == m and r["task"] == t]
            b = [r for r in rs_b if r["model"] == m and r["task"] == t]
            if not a or not b:
                continue
            limit = min(min(r["curve"][-1][field] for r in a), min(r["curve"][-1][field] for r in b))
            for x in grid_frac:
                sec = limit * x
                rows.append(dict(task=t, model=m, seconds=sec, a=mean(val_at(a, m, t, field, sec)), b=mean(val_at(b, m, t, field, sec))))
    return rows


def model_equal_time(rs, tasks, field):
    """Across models, one execution mode: validation at fractions of the shortest run's horizon; contrasts against A0."""
    rows = []
    for t in tasks:
        rr = [r for r in rs if r["task"] == t]
        if not rr:
            continue
        limit = min(r["curve"][-1][field] for r in rr)
        for frac in (0.125, 0.25, 0.5, 0.75, 1.0):
            sec = limit * frac
            row = dict(task=t, seconds=sec)
            for m in MODELS:
                row[m] = mean(val_at(rr, m, t, field, sec))
            pair = paired(val_at(rr, "A3mean", t, field, sec), val_at(rr, "A0", t, field, sec))
            row["A3mean_vs_A0_pct"] = pair["pct"] if pair else None
            row["ci"] = f"[{pair['ci_pct_low']:+.2f}, {pair['ci_pct_high']:+.2f}]" if pair and "ci_pct_low" in pair else "n/a"
            rows.append(row)
    return rows


def met_equal_time_test(tests, rs, tasks, a="A3mean", b="A0"):
    """Test at equal training time on retained checkpoints: b's endpoint against a's latest milestone at or below b's time (a is slower)."""
    rows = []
    for t in tasks:
        ra, rb = {r["seed"]: r for r in rs if r["model"] == a and r["task"] == t}, {r["seed"]: r for r in rs if r["model"] == b and r["task"] == t}
        ta, tb = {r["seed"]: r for r in tests if r["model"] == a and r["task"] == t}, {r["seed"]: r for r in tests if r["model"] == b and r["task"] == t}
        da, db, info = {}, {}, []
        for s in sorted(set(ra) & set(rb) & set(ta) & set(tb)):
            u_b = max(int(u) for u in tb[s]["milestones"])
            lim = tb[s]["milestones"][str(u_b)]["train_s"]
            cands = [int(u) for u, v in ta[s]["milestones"].items() if v["train_s"] <= lim]
            if not cands:
                continue
            u_a = max(cands)
            da[s], db[s] = ta[s]["milestones"][str(u_a)]["test"], tb[s]["milestones"][str(u_b)]["test"]
            info.append(f"s{s}: {a} u{u_a} ({ta[s]['milestones'][str(u_a)]['train_s']:.0f} s), {b} u{u_b} ({lim:.0f} s)")
        rows.append(contrast_row(t, a, b, paired(da, db), "; ".join(info)))
    return rows


def validation_selected(tests):
    rows = []
    for r in tests:
        ms = [v for v in r["milestones"].values() if v.get("val") is not None]
        best = min(ms, key=lambda v: v["val"])
        last = max(ms, key=lambda v: v["update"])
        rows.append(dict(task=r["task"], model=r["model"], seed=r["seed"], selected_update=best["update"], selected_val=best["val"], selected_test=best["test"], final_update=last["update"], final_test=last["test"]))
    return sorted(rows, key=lambda r: (r["task"], r["model"], r["seed"]))


def perf_section(rs):
    groups = {}
    for r in rs:
        groups.setdefault((r["kind"], r["model"], r["M_phys"], r["accum_steps"], r["dx_request"]), []).append(r)
    phases = ["sparse_and_reduction", "finalize_and_readout", "loss_and_dense_backward", "gradient_setup", "dx", "dw_db", "input_gradient_tail", "fused_optimizer", "dense_optimizer", "effective_update"]
    rows, mem = [], []
    for (kind, m, mp, ac, dxr), rr in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2] * kv[0][3], kv[0][3], MODELS.index(kv[0][1]), kv[0][4])):
        d = dict(kind=kind, model=m, M_phys=mp, accum_steps=ac, M_eff=mp * ac, dx_request=dxr, status=rr[0]["status"], jobs=len(rr))
        for p in phases:
            v = [r["median_ms"].get(p) for r in rr if r["median_ms"].get(p) is not None]
            d[p] = st.median(v) if v else None
        all_rounds = [x for r in rr for x in r["round_medians_ms"]]
        d["round_min"], d["round_max"] = min(all_rounds), max(all_rounds)
        d["iqr_over_median"] = st.median((r["dispersion_ms"]["effective_update"]["q3"] - r["dispersion_ms"]["effective_update"]["q1"]) / r["median_ms"]["effective_update"] for r in rr)
        d["rows_per_s"] = st.median(r["rows_per_second"] for r in rr)
        d["updates_per_s"] = st.median(r["updates_per_second"] for r in rr)
        rows.append(d)
        mm = rr[0]["memory"]
        e = dict(kind=kind, model=m, M_phys=mp, accum_steps=ac, dx_request=dxr, jobs=len(rr))
        for k, v in mm["categories_bytes"].items():
            e[k] = v / 2 ** 20
        e.update(ordinary_gradients=mm["ordinary_gradients_bytes"] / 2 ** 20, unattributed=mm["unattributed_steady_bytes"] / 2 ** 20, steady=max(r["memory"]["steady_allocated_bytes"] for r in rr) / 2 ** 30,
                 transient=max(r["memory"]["transient_above_steady_bytes"] for r in rr) / 2 ** 20, peak=max(r["memory"]["steady_training_peak_bytes"] for r in rr) / 2 ** 30,
                 first=max(r["memory"]["first_step_peak_bytes"] for r in rr) / 2 ** 30, construction=max(r["memory"]["construction_peak_bytes"] for r in rr) / 2 ** 30, reserved=max(r["memory"]["peak_reserved_bytes"] for r in rr) / 2 ** 30)
        mem.append(e)
    cols = [("kind", lambda r: r["kind"]), ("model", lambda r: r["model"]), ("M_phys x accum", lambda r: f"{r['M_phys']} x {r['accum_steps']}"), ("dX request", lambda r: r["dx_request"]),
            ("sparse + reduce", lambda r: fmt(r["sparse_and_reduction"], 1)), ("finalize + readout", lambda r: fmt(r["finalize_and_readout"], 1)), ("dense bwd", lambda r: fmt(r["loss_and_dense_backward"], 1)), ("grad setup", lambda r: fmt(r["gradient_setup"], 1)),
            ("dX", lambda r: fmt(r["dx"], 1)), ("dW/db/dP", lambda r: fmt(r["dw_db"], 1)), ("input grad", lambda r: fmt(r["input_gradient_tail"], 1)), ("FusedAdamW", lambda r: fmt(r["fused_optimizer"], 1)), ("AdamW", lambda r: fmt(r["dense_optimizer"], 1)),
            ("**effective update**", lambda r: fmt(r["effective_update"], 1)), ("IQR/med", lambda r: fmt(r["iqr_over_median"], 3)), ("rows/s", lambda r: f"{r['rows_per_s']:,.0f}"), ("updates/s", lambda r: fmt(r["updates_per_s"], 2))]
    mcols = [("kind", lambda r: r["kind"]), ("model", lambda r: r["model"]), ("M_phys x accum", lambda r: f"{r['M_phys']} x {r['accum_steps']}"), ("dX", lambda r: r["dx_request"]), ("params MiB", lambda r: fmt(r["model_parameters"], 1)), ("task params", lambda r: fmt(r["common_task_parameters"], 1)),
             ("main_grad", lambda r: fmt(r["main_grad"], 1)), ("ordinary grads", lambda r: fmt(r["ordinary_gradients"], 1)), ("FusedAdamW state", lambda r: fmt(r["fused_optimizer_state"], 1)), ("AdamW state", lambda r: fmt(r["dense_optimizer_state"], 1)),
             ("metadata", lambda r: fmt(r["immutable_metadata"], 1)), ("work lists", lambda r: fmt(r["runtime_work_lists"], 1)), ("input/target", lambda r: fmt(r["input_and_target"], 1)), ("unattributed", lambda r: fmt(r["unattributed"], 1)), ("steady GiB", lambda r: fmt(r["steady"], 3)),
             ("transient MiB", lambda r: fmt(r["transient"], 1)), ("**peak GiB**", lambda r: fmt(r["peak"], 3)), ("first step GiB", lambda r: fmt(r["first"], 3)), ("build GiB", lambda r: fmt(r["construction"], 3)), ("reserved GiB", lambda r: fmt(r["reserved"], 3))]
    return md(rows, cols), md(mem, mcols), rows, mem


def historical_perf_csv(raw_gz, dest):
    import learned_reduction_report as R

    with gzip.open(raw_gz, "rt") as h:
        rs = [json.loads(ln) for ln in h]
    write_csv(dest, [R.perf_row(r) for r in rs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--historical", default="")
    a = ap.parse_args()
    root, dest = Path(a.out), Path(a.dest)
    (dest / "csv").mkdir(parents=True, exist_ok=True)
    stages = ("A_tune", "A_ext", "B", "C_tune", "C_ext", "C", "D_tune", "D_direct", "D_accum", "anchor", "secondary", "B_ext")
    R = {s: load(root, s) for s in stages}
    Tst = {s: load(root, s, "test_eval") for s in stages}
    tx = ["# A3mean follow-up: generated tables\n\nGenerated by `lrf_report.py` from the raw per-job records.  val = fixed validation split; **test** = locked test split, read once after all choices were frozen.\n"]

    def add_sel(title, tune, ext):
        s, best = selection_table(R[tune], R.get(ext, []))
        tx.append(f"\n### {title}\n\n" + s)
        return best

    sel_a = add_sel("Stage A: fresh-data LR tuning, M_eff = 512, seed 0, 2,048 updates (validation only)", "A_tune", "A_ext") if R["A_tune"] else {}
    sel_c = add_sel("Stage C tuning: Local, M_eff = 512, seed 0, 2,048 updates (validation only)", "C_tune", "C_ext") if R["C_tune"] else {}
    sel_d = add_sel("Stage D tuning: M_eff = 8,192, direct, seed 0, 256 effective updates, x {1, 4, 16} of the Stage A pair (validation only)", "D_tune", "none") if R["D_tune"] else {}
    summary = dict(selected_A={f"{k[0]}/{k[1]}": dict(lr_w=v["lr_w"], lr_small=v["lr_small"], p_mult=v["p_mult"], multiplier=v["lr_multiplier"], score=v["score_val"]) for k, v in sel_a.items()},
                   selected_C={f"{k[0]}/{k[1]}": dict(lr_w=v["lr_w"], lr_small=v["lr_small"], p_mult=v["p_mult"], multiplier=v["lr_multiplier"], score=v["score_val"]) for k, v in sel_c.items()},
                   selected_D={f"{k[0]}/{k[1]}": dict(lr_w=v["lr_w"], lr_small=v["lr_small"], p_mult=v["p_mult"], multiplier=v["lr_multiplier"], score=v["score_val"]) for k, v in sel_d.items()})
    gt = ["compose", "interact"]
    B, TB = R["B"], Tst["B"]
    if B:
        s, rows = milestone_tables(TB, B, gt, (1024, 2048, 4096, 8192, 12288, 16384), "Stage B: long horizon, M_eff = 512, three paired seeds, 16,384 updates, final weights")
        tx.append(s)
        write_csv(dest / "csv" / "stage_B_milestones.csv", rows)
        prim = [contrast_row(t, "A3mean", "A0", paired(test_at(TB, "A3mean", t, 16384), test_at(TB, "A0", t, 16384)), "test u16384") for t in gt]
        if all("p" in r for r in prim):
            adj = holm([r["p"] for r in prim])
            for r, h in zip(prim, adj):
                r["holm_p"] = h
        tx.append("\n### PRIMARY: A3mean - A0, locked test, 16,384 updates (Holm over these two tests; n = 3)\n")
        tx.append(md(prim, [("task", lambda r: r["task"]), ("A3mean per seed", lambda r: " ".join(f"{x:.5f}" for x in r.get("a_per_seed", []))), ("A0 per seed", lambda r: " ".join(f"{x:.5f}" for x in r.get("b_per_seed", []))),
                            ("mean diff", lambda r: f"{r['mean']:+.5f}" if "mean" in r else "n/a"), ("% of A0", lambda r: f"{r['pct']:+.2f}" if "pct" in r else "n/a"), ("95 % CI, %", lambda r: f"[{r['ci_pct_low']:+.2f}, {r['ci_pct_high']:+.2f}]" if "ci_pct_low" in r else "n/a"),
                            ("t", lambda r: fmt(r.get("t"), 2)), ("p", lambda r: fmt(r.get("p"), 3)), ("Holm p", lambda r: fmt(r.get("holm_p"), 3))]))
        write_csv(dest / "csv" / "primary.csv", prim)
        summary["primary"] = prim
        sec = [contrast_row(t, x, y, paired(test_at(TB, x, t, u), test_at(TB, y, t, u)), f"test u{u}") for t in gt for u in (4096, 8192, 16384) for x, y in CONTRASTS]
        tx.append("\n### Stage B paired contrasts over time (test; secondary)\n")
        tx.append(contrast_md(sec))
        write_csv(dest / "csv" / "stage_B_contrasts.csv", sec)
        rows = late_rows(B, gt, 8192, 12288, 16384)
        tx.append("\n### Late validation slopes\n")
        tx.append(md(rows, [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("val 8192", lambda r: fmt(r["v0"], 5)), ("val 12288", lambda r: fmt(r["v1"], 5)), ("val 16384", lambda r: fmt(r["v2"], 5)), ("drop 8192-12288 %", lambda r: fmt(r["drop01"], 2)), ("drop 12288-16384 %", lambda r: fmt(r["drop12"], 2))]))
        dg = diagnostics(B, gt)
        tx.append("\n### Stage B training diagnostics, last evaluation (no optimizer update; only already consumed blocks)\n")
        dgc = [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("online (pre-update) loss", lambda r: fmt(r["running"], 5)), ("recent_seen", lambda r: fmt(r["recent"], 5)), ("fixed_seen", lambda r: fmt(r["fixed"], 5)), ("val", lambda r: fmt(r["val"], 5)), ("val - fixed_seen", lambda r: fmt(r["gap_fixed"], 5)), ("val - recent_seen", lambda r: fmt(r["gap_recent"], 5)), ("worst rise in late val %", lambda r: fmt(r["worst_late_rise"], 2)), ("finite", lambda r: r["finite"])]
        tx.append(md(dg, dgc))
        pr = p_rows(B)
        tx.append("\n### A3mean first-stage weights P, Stage B: start and end\n")
        tx.append(md(pr, [("task", lambda r: r["task"]), ("seed", lambda r: r["seed"]), ("update", lambda r: r["update"]), ("mean", lambda r: f"{r['mean']:.3e}"), ("std", lambda r: f"{r['std']:.3e}"), ("max |P|", lambda r: f"{r['max_abs']:.3e}"), ("positive", lambda r: fmt(r["positive"], 4)), ("negative", lambda r: fmt(r["negative"], 4)), ("near zero", lambda r: fmt(r["near_zero"], 4)), ("q01", lambda r: f"{r['q01']:.2e}"), ("q50", lambda r: f"{r['q50']:.2e}"), ("q99", lambda r: f"{r['q99']:.2e}"), ("relative movement", lambda r: fmt(r["rel_movement"], 3))]))
        write_csv(dest / "csv" / "stage_B_p_stats.csv", pr)
        gr = grad_rows(B)
        tx.append("\n### Gradient-to-weight norm ratios, first and last capture, Stage B\n")
        tx.append(md(sorted(gr, key=lambda r: (r["task"], r["model"], r["seed"])), [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("seed", lambda r: r["seed"]), ("updates", lambda r: f"{r['first']} / {r['last']}"), ("W", lambda r: f"{r['W0']:.2e} / {r['W1']:.2e}"), ("P", lambda r: "n/a" if r["P0"] is None else f"{r['P0']:.2e} / {r['P1']:.2e}"), ("V", lambda r: "n/a" if r["V0"] is None else f"{r['V0']:.2e} / {r['V1']:.2e}")]))
        vs = validation_selected(TB)
        tx.append("\n### Validation-selected checkpoint (among retained milestones) vs final weights, Stage B\n")
        tx.append(md(vs, [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("seed", lambda r: r["seed"]), ("selected update", lambda r: r["selected_update"]), ("val", lambda r: fmt(r["selected_val"], 5)), ("test, selected", lambda r: fmt(r["selected_test"], 5)), ("test, final", lambda r: fmt(r["final_test"], 5))]))
        write_csv(dest / "csv" / "stage_B_validation_selected.csv", vs)
        for field, title in (("train_s", "synchronized training time"), ("wall_s", "full wall time")):
            et = model_equal_time(B, gt, field)
            tx.append(f"\n### Stage B, validation at equal {title}, within the shortest run's horizon (linear interpolation, no extrapolation)\n")
            tx.append(md(et, [("task", lambda r: r["task"]), ("seconds", lambda r: fmt(r["seconds"], 0))] + [(m, (lambda m: lambda r: fmt(r[m], 5))(m)) for m in MODELS] + [("A3mean vs A0 %", lambda r: fmt(r["A3mean_vs_A0_pct"], 2)), ("95 % CI, %", lambda r: r["ci"])]))
            write_csv(dest / "csv" / f"stage_B_equal_{field}.csv", et)
        tt = met_equal_time_test(TB, B, gt)
        tx.append("\n### Stage B, test at about equal training time: A0's endpoint vs A3mean's latest retained milestone not later in training time\n")
        tx.append(contrast_md(tt))
        write_csv(dest / "csv" / "stage_B_test_equal_time.csv", tt)
        summary["equal_time_test"] = tt
    C_, TC = R["C"], Tst["C"]
    if C_:
        s, rows = milestone_tables(TC, C_, ["local"], (1024, 2048, 4096, 8192), "Stage C: Local, M_eff = 512, three paired seeds, 8,192 updates, final weights")
        tx.append(s)
        write_csv(dest / "csv" / "stage_C_milestones.csv", rows)
        sec = [contrast_row("local", x, y, paired(test_at(TC, x, "local", u), test_at(TC, y, "local", u)), f"test u{u}") for u in (4096, 8192) for x, y in CONTRASTS]
        tx.append("\n### Stage C paired contrasts (test; secondary)\n")
        tx.append(contrast_md(sec))
        write_csv(dest / "csv" / "stage_C_contrasts.csv", sec)
        tx.append(md(diagnostics(C_, ["local"]), dgc if B else []))
        summary["local"] = sec
        et = model_equal_time(C_, ["local"], "train_s")
        tx.append("\n### Stage C, validation at equal synchronized training time\n")
        tx.append(md(et, [("seconds", lambda r: fmt(r["seconds"], 0))] + [(m, (lambda m: lambda r: fmt(r[m], 5))(m)) for m in MODELS] + [("A3mean vs A0 %", lambda r: fmt(r["A3mean_vs_A0_pct"], 2)), ("95 % CI, %", lambda r: r["ci"])]))
    Dd, Da, TDd, TDa = R["D_direct"], R["D_accum"], Tst["D_direct"], Tst["D_accum"]
    if Dd:
        for lab, rs_, ts_ in (("direct, M_phys 8,192", Dd, TDd), ("accumulated, 512 x 16", Da, TDa)):
            if rs_:
                s, rows = milestone_tables(ts_, rs_, gt, (64, 128, 256, 512, 1024), f"Stage D {lab}: M_eff = 8,192, three paired seeds, 1,024 updates (8,388,608 samples)")
                tx.append(s)
                write_csv(dest / "csv" / f"stage_D_{'direct' if rs_ is Dd else 'accum'}_milestones.csv", rows)
        sec = [contrast_row(t, x, y, paired(test_at(TDd, x, t, 1024), test_at(TDd, y, t, 1024)), "direct, test u1024") for t in gt for x, y in CONTRASTS]
        tx.append("\n### Large-batch model contrasts, direct (test; secondary)\n")
        tx.append(contrast_md(sec))
        write_csv(dest / "csv" / "stage_D_contrasts.csv", sec)
        summary["large_contrasts"] = sec
        rows = []
        for t in gt:
            for m in MODELS:
                rows.append(contrast_row(t, f"{m} accumulated", f"{m} direct", paired(test_at(TDa, m, t, 1024), test_at(TDd, m, t, 1024)), "test u1024"))
        tx.append("\n### Direct vs accumulated large batch, same LR and samples (test)\n")
        tx.append(contrast_md(rows))
        write_csv(dest / "csv" / "stage_D_direct_vs_accum.csv", rows)
        summary["direct_vs_accum"] = rows
        times = []
        for mode, rr in (("direct", Dd), ("accumulated", Da)):
            for r in rr:
                times.append(dict(mode=mode, model=r["model"], task=r["task"], seed=r["seed"], train_s=r["train_s"], data_s=r["data_s"], eval_s=r["eval_s"], ckpt_s=r["ckpt_s"], wall_s=r["wall_s"], ms_per_update=r["ms_per_update"], peak=r["peak_train_window_bytes"] / 2 ** 30))
        for r in B:
            times.append(dict(mode="small (16,384 updates)", model=r["model"], task=r["task"], seed=r["seed"], train_s=r["train_s"], data_s=r["data_s"], eval_s=r["eval_s"], ckpt_s=r["ckpt_s"], wall_s=r["wall_s"], ms_per_update=r["ms_per_update"], peak=r["peak_train_window_bytes"] / 2 ** 30))
        write_csv(dest / "csv" / "time_accounting.csv", times)
        agg = {}
        for e in times:
            agg.setdefault((e["mode"], e["model"]), []).append(e)
        rows = [dict(mode=k[0], model=k[1], train=st.mean(x["train_s"] for x in v), data=st.mean(x["data_s"] for x in v), ev=st.mean(x["eval_s"] for x in v), ck=st.mean(x["ckpt_s"] for x in v), wall=st.mean(x["wall_s"] for x in v), ms=st.mean(x["ms_per_update"] for x in v), peak=max(x["peak"] for x in v)) for k, v in sorted(agg.items())]
        tx.append("\n### Time for 8,388,608 training samples (mean over runs; seconds)\n")
        tx.append(md(rows, [("mode", lambda r: r["mode"]), ("model", lambda r: r["model"]), ("train", lambda r: fmt(r["train"], 0)), ("data", lambda r: fmt(r["data"], 0)), ("eval", lambda r: fmt(r["ev"], 0)), ("checkpoints", lambda r: fmt(r["ck"], 0)), ("wall", lambda r: fmt(r["wall"], 0)), ("ms/update", lambda r: fmt(r["ms"], 1)), ("training-window peak GiB", lambda r: fmt(r["peak"], 3))]))
        summary["time"] = rows
        eq = []
        for t in gt:
            for m in MODELS:
                eq.append(contrast_row(t, f"{m} small", f"{m} large", paired(test_at(TB, m, t, 16384), test_at(TDd, m, t, 1024)), "equal samples 8,388,608, test"))
                eq.append(contrast_row(t, f"{m} small", f"{m} large", paired(test_at(TB, m, t, 8192), test_at(TDd, m, t, 512)), "equal samples 4,194,304, test"))
                eq.append(contrast_row(t, f"{m} small", f"{m} large", paired(test_at(TB, m, t, 4096), test_at(TDd, m, t, 256)), "equal samples 2,097,152, test"))
                eq.append(contrast_row(t, f"{m} small", f"{m} large", paired(test_at(TB, m, t, 1024), test_at(TDd, m, t, 1024)), "DIAGNOSTIC equal updates 1,024 (16x samples for large), test"))
        tx.append("\n### Small (M_eff 512) vs large (M_eff 8,192, direct): equal samples; equal updates as a diagnostic\n")
        tx.append(contrast_md(eq))
        write_csv(dest / "csv" / "small_vs_large.csv", eq)
        summary["small_vs_large"] = eq
        for field, name in (("train_s", "training time"), ("wall_s", "full wall time")):
            rows = equal_time(B, Dd, gt, field)
            tx.append(f"\n### Validation at equal {name} within the overlap: small (a) vs direct large (b)\n")
            tx.append(md(rows, [("task", lambda r: r["task"]), ("model", lambda r: r["model"]), ("seconds", lambda r: fmt(r["seconds"], 0)), ("small", lambda r: fmt(r["a"], 5)), ("large", lambda r: fmt(r["b"], 5))]))
            write_csv(dest / "csv" / f"small_vs_large_{field}.csv", rows)
            et = model_equal_time(Dd, gt, field)
            tx.append(f"\n### Direct large batch, models at equal {name} (validation)\n")
            tx.append(md(et, [("task", lambda r: r["task"]), ("seconds", lambda r: fmt(r["seconds"], 0))] + [(m, (lambda m: lambda r: fmt(r[m], 5))(m)) for m in MODELS] + [("A3mean vs A0 %", lambda r: fmt(r["A3mean_vs_A0_pct"], 2)), ("95 % CI, %", lambda r: r["ci"])]))
        tt = met_equal_time_test(TDd, Dd, gt)
        tx.append("\n### Direct large batch, test at about equal training time: A0's endpoint vs A3mean's latest milestone not later in training time\n")
        tx.append(contrast_md(tt))
        summary["large_equal_time_test"] = tt
    for st_, title, ups in (("anchor", "Inherited-LR anchors, seed 0, 8,192 updates (secondary)", (4096, 8192)), ("secondary", "Secondary width H = 12,582,912, 8,192 updates, primary-width LR (not re-tuned)", (2048, 4096, 8192)), ("B_ext", "Extension to 32,768 updates (secondary)", (20480, 24576, 32768))):
        if R[st_]:
            s, rows = milestone_tables(Tst[st_], R[st_], gt, ups, title)
            tx.append(s)
            write_csv(dest / "csv" / f"stage_{st_}.csv", rows)
            if Tst[st_] and st_ != "anchor":
                sec = [contrast_row(t, "A3mean", "A0", paired(test_at(Tst[st_], "A3mean", t, ups[-1]), test_at(Tst[st_], "A0", t, ups[-1])), f"test u{ups[-1]}") for t in gt]
                tx.append(contrast_md(sec))
                summary[st_] = sec
    perf = load(root, "perf")
    if perf:
        a_, b_, rows, mem = perf_section(perf)
        tx.append("\n### Performance, revised harness: medians over jobs (ms per effective update)\n\n" + a_)
        tx.append("\n### Memory, revised harness: deduplicated categories (MiB unless noted)\n\n" + b_)
        write_csv(dest / "csv" / "perf.csv", rows)
        write_csv(dest / "csv" / "memory.csv", mem)
        summary["perf"], summary["memory"] = rows, mem
        with gzip.open(dest / "raw_perf.jsonl.gz", "wt") as f:
            for r in perf:
                r = {k: v for k, v in r.items() if k not in ("rounds_detail",)}
                f.write(json.dumps(r, default=str) + "\n")
    with gzip.open(dest / "raw_quality.jsonl.gz", "wt") as f:
        for s, rs in R.items():
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "kernel"}, default=str) + "\n")
    with gzip.open(dest / "raw_test.jsonl.gz", "wt") as f:
        for s, rs in Tst.items():
            for r in rs:
                f.write(json.dumps(r, default=str) + "\n")
    if a.historical:
        historical_perf_csv(a.historical, dest / "csv" / "historical_perf_corrected.csv")
    acc = []
    base = {r["_file"]: r for r in R["B"]}
    for s, rs in R.items():
        # B_ext records resume from Stage B and carry its cumulative times: count only the extension's own share
        prior = [base.get(r["_file"]) for r in rs] if s == "B_ext" else []
        own = lambda f: sum(r[f] for r in rs) - sum(p[f] for p in prior if p)
        acc.append(dict(stage=s, jobs=len(rs), train_h=own("train_s") / 3600, wall_h=own("wall_s") / 3600, failures=len((root / s / "failures.jsonl").read_text().splitlines()) if (root / s / "failures.jsonl").exists() else 0, diverged=sum(not r["finite"] for r in rs)))
    tx.append("\n### Campaign accounting (B_ext: the extension's own time, excluding the resumed Stage B part)\n")
    tx.append(md(acc, [("stage", lambda r: r["stage"]), ("jobs", lambda r: r["jobs"]), ("training h", lambda r: fmt(r["train_h"], 2)), ("job wall h", lambda r: fmt(r["wall_h"], 2)), ("failed", lambda r: r["failures"]), ("non-finite", lambda r: r["diverged"])]))
    summary["accounting"] = acc
    ck = {}
    for s, rs in R.items():
        for r in rs:
            for u, m in r.get("milestones", {}).items():
                ck.setdefault(m["path"], dict(kind="model-only milestone", stage=s, job=r["_file"][:-5], update=int(u), path=m["path"], bytes=m["bytes"], sha256=m["sha256"]))
    full = root / "full_checkpoints.sha256"
    if full.exists():
        for ln in full.read_text().splitlines():
            h, p = ln.split(maxsplit=1)
            rel = p.removeprefix("./")
            ck.setdefault(rel, dict(kind="full resumable (weights, optimizer, RNG, stream position)", stage=rel.split("/")[0], job=rel.split("/")[1], update=int(rel.split("full_u")[1].split(".")[0]), path=f"<ckpt-root>/{rel}", bytes=None, sha256=h))
    write_csv(dest / "csv" / "checkpoints.csv", sorted(ck.values(), key=lambda d: (d["stage"], d["job"], d["update"], d["kind"])), ["stage", "job", "update", "kind", "bytes", "sha256", "path"])
    tx.append(f"\n{len(ck)} checkpoints listed with SHA-256 in `csv/checkpoints.csv`.\n")
    (dest / "tables.md").write_text("\n".join(tx))
    (dest / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(tx))


if __name__ == "__main__":
    main()
