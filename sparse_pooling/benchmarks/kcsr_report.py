"""Tables, CSV summaries and plots of the K/C/S/R sweep from the raw records (no GPU).

    python kcsr_report.py --raw ../results/random_kcsr_coverage_performance

Inputs under --raw: screen/campaign/ (status.jsonl + bench JSON), screen/native/ (unchanged main), coverage/,
confirm/ (optional), audit/, supplied/.  Outputs: csv/*.csv, tables.md, plots/*.png.  Every one of the 24
cases (and the X01 control) gets rows, including failure statuses.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import kcsr_cases as KC

PH = ["fwd_layout", "fwd_zero", "fwd_kernel", "fwd_kernel_generic", "fwd_finalize", "fwd_total", "bwd_coef",
      "bwd_layout", "bwd_dx_init", "dx_kernel", "dw_kernel", "bwd_kernels", "bwd_post", "bwd_total", "optimizer",
      "step", "host_enqueue", "step_wall"]

CLASS_STYLE = {  # execution class -> (color, marker): three validated slots + neutral, shapes as secondary encoding
    "experimental generalized": ("#2a78d6", "o"),
    "native optimized": ("#eb6834", "s"),
    "generic existing": ("#1baf7a", "^"),
    "physical lowering": ("#52514e", "D"),
}


def exec_class(backend: str, kernels: dict) -> str:
    if backend in ("kgen", "kgen_native"):
        return "experimental generalized"
    if backend == "lowered":
        return "physical lowering"
    return "native optimized" if kernels and "grouped" in kernels.values() else "generic existing"


def load_runs(raw: Path):
    rows = []
    for st_path, origin in ((raw / "screen/campaign/status.jsonl", "screen"), (raw / "confirm/status.jsonl", "confirm")):
        if not st_path.exists():
            continue
        for ln in st_path.read_text().splitlines():
            r = json.loads(ln)
            r["origin"] = origin
            r["dir"] = st_path.parent
            rows.append(r)
    nat = raw / "screen/native"
    if nat.exists():
        for j in sorted(nat.glob("*.json")):
            cid = j.name.split("_")[0]
            c = KC.case(cid)
            rows.append(dict(tag=j.stem, case=cid, backend="native", graph_seed=0, K=c["K"], C=c["C"], S=c["S"], R=c["R"],
                             H=c["H"], status="measured", json=j.name, origin="native-main", dir=nat))
    for r in rows:
        r["rec"] = json.loads((r["dir"] / r["json"]).read_text()) if r.get("json") else None
    return rows


def phase(rec, ph, key="median_ms"):
    v = rec["variants"]["default"]["phases"].get(ph)
    return None if v is None else v[key]


def run_row(r, cov):
    rec = r["rec"]
    out = dict(origin=r["origin"], tag=r["tag"], id=r["case"], K=r["K"], C=r["C"], S=r["S"], R=r["R"], H=r["H"],
               weights=r["H"] * r["K"], biases=r["H"], backend=r["backend"], graph_seed=r["graph_seed"], status=r["status"])
    if rec is None:
        out["reason"] = (r.get("log_tail") or [""])[-1][:300]
        return out
    v = rec["variants"]["default"]
    k = rec.get("kcsr", {})
    desc = k.get("layer_description") or {}
    out.update(exec_class=exec_class(r["backend"], v["kernels"]), kernels=json.dumps(v["kernels"]),
               logical_C=desc.get("logical_C", r["C"]), physical_C=desc.get("physical_C", rec["config"]["layout_block"]),
               physical_subblocks=desc.get("physical_subblocks", 1), index_duplication=desc.get("index_duplication", 1),
               commit=rec["commit"][:10], dirty=rec["dirty"], fingerprint=k.get("graph_fingerprint", "")[:16],
               params_sha=k.get("logical_params_sha", "")[:16], logical_records=k.get("graph_summary", {}).get("logical_records"))
    for ph in PH:
        out[f"{ph}_ms"] = phase(rec, ph)
    s = v["samples_ms"].get("step", [])
    out["step_min_ms"], out["step_max_ms"], out["step_n"] = (min(s), max(s), len(s)) if s else (None, None, 0)
    out["step_round_medians_ms"] = json.dumps([round(x["step"], 2) for x in v["round_values_ms"]])
    out["step_samples_ms"] = json.dumps([round(x, 2) for x in s])
    pb = rec["persistent_bytes"]
    out.update(peak_alloc_GiB=v["peak_allocated_bytes"] / 2**30, peak_reserved_GiB=v["peak_reserved_bytes"] / 2**30,
               warm_peak_alloc_GiB=v["warmup_peak_allocated_bytes"] / 2**30, scratch_MiB=v["step_scratch_peak_bytes"] / 2**20,
               weight_bytes=pb["weights"], bias_bytes=pb["bias"], index_bytes=pb["indices"], main_grad_bytes=pb["main_grad"],
               adam_bytes=pb["adam_moments"], graph_metadata_bytes=pb["graph_metadata"],
               work_list_bytes=pb.get("work_lists_cached_last_runtime"), build_s=rec["build_seconds"],
               graph_s=(rec.get("graph_stats") or {}).get("generation_seconds"),
               first_step_s=rec.get("cold_steps", {}).get("default", {}).get("first_step_wall_s"),
               restore_checks=v["restore_checks"])
    gr = v.get("gpu_rounds") or []
    if gr:
        out.update(sm_clock_MHz=statistics.mean(g.get("clocks.sm", 0) for g in gr),
                   temp_C=max(g.get("temperature.gpu", 0) for g in gr), power_W=statistics.mean(g.get("power.draw", 0) for g in gr))
    c = cov.get((r["case"], r["graph_seed"]))
    if c:
        out.update(pair_pct=100 * c["pair"]["fraction"], triple_pct=100 * c["triple"]["fraction"],
                   quad_pct=100 * c["quad"]["fraction"], cov_fingerprint_match=c["graph_fingerprint"][:16] == out["fingerprint"])
    return out


def load_cov(raw: Path):
    cov = {}
    for j in sorted((raw / "coverage").glob("*_s*.json")):
        c = json.loads(j.read_text())
        cov[(c["case"], c["graph_seed"])] = c
    return cov


def cov_row(c):
    q = c["quad"]
    out = dict(id=c["case"], K=c["K"], C=c["C"], S=c["S"], R=c["R"], H=c["H"], graph_seed=c["graph_seed"],
               generator=c["generator"]["generator"], quantum=c["generator"]["quantum"],
               logical_records=c["graph_summary"]["logical_records"], distinct_records=c["distinct_records"],
               K_equals_S=c["K_equals_S"],
               pair_measured_exact_pct=100 * c["pair"]["fraction"], triple_measured_exact_pct=100 * c["triple"]["fraction"],
               quad_measured_sampled_pct=100 * q["fraction"], quad_ci95_lo_pct=100 * q["ci95"][0],
               quad_ci95_hi_pct=100 * q["ci95"][1], quad_hits=q["hits"], quad_sample_size=q["sample_size"],
               quad_sample_seed=q["sample_seed"],
               pair_group_realized_pct=100 * c["pair"]["group_fraction"], triple_group_realized_pct=100 * c["triple"]["group_fraction"],
               quad_group_realized_sampled_pct=100 * q["group_fraction"],
               triple_sampled_check_pct=100 * c["triple_sampled_check"]["fraction"],
               triple_sampled_check_ci95=json.dumps([round(100 * x, 5) for x in c["triple_sampled_check"]["ci95"]]),
               pair_refined_pred_pct=100 * c["refined_prediction"]["pair"], triple_refined_pred_pct=100 * c["refined_prediction"]["triple"],
               quad_refined_pred_pct=100 * c["refined_prediction"]["quad"],
               passes_with_invalid_records=c["local_passes"]["passes_with_invalid_records"],
               invalid_record_slots=c["local_passes"]["invalid_record_slots"],
               bucket_occupancy=c["bucket_stats"]["bucket_occupancy"],
               mean_units_per_nonempty_bucket=c["bucket_stats"]["mean_units_per_nonempty_bucket"],
               mean_units_per_possible_bucket=c["bucket_stats"]["mean_units_per_possible_bucket"],
               analysis_s=c.get("analysis_seconds"), analysis_peak_GiB=c["analysis_peak_allocated_bytes"] / 2**30,
               fingerprint=c["graph_fingerprint"][:16])
    sp = c.get("supplied_prediction_pct")
    if sp:
        for t in ("pair", "triple", "quad"):
            out[f"{t}_supplied_model_pct"] = sp[t]["model"]
            out[f"{t}_supplied_group_expectation_pct"] = sp[t]["group_expectation"]
            out[f"{t}_supplied_iid_reference_pct"] = sp[t]["iid_reference"]
    return out


def write_csv(path: Path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{d}f}" if isinstance(x, (int, float)) else str(x)


def plots(rows, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    pts = [r for r in rows if r.get("status") == "measured" and r.get("triple_pct") is not None and r["origin"] == "screen"
           and r["id"] != "X01"]
    for metric, label, fname in (("triple_pct", "measured triple coverage (%, exact)", "coverage_triple_vs_step.png"),
                                 ("quad_pct", "measured quadruple coverage (%, sampled)", "coverage_quad_vs_step.png"),
                                 ("pair_pct", "measured pair coverage (%, exact)", "coverage_pair_vs_step.png")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
        for ax, ycol, xl in ((axes[0], "step_ms", "complete training step (ms, median of 8)"),
                             (axes[1], "peak_alloc_GiB", "peak allocated device memory (GiB)")):
            for cls, (col, mk) in CLASS_STYLE.items():
                sel = [r for r in pts if r["exec_class"] == cls and not (r["backend"] == "kgen_native")]
                if not sel:
                    continue
                ax.scatter([r[ycol] for r in sel], [r[metric] for r in sel], c=col, marker=mk, s=46, label=cls,
                           edgecolors="white", linewidths=1.2, zorder=3)
                for r in sel:
                    ax.annotate(f"{r['id']} K{r['K']}", (r[ycol], r[metric]), xytext=(4, 3), textcoords="offset points",
                                fontsize=7, color="#52514e")
            ctrl = [r for r in pts if r["backend"] == "kgen_native"]
            ax.scatter([r[ycol] for r in ctrl], [r[metric] for r in ctrl], facecolors="none", edgecolors="#2a78d6",
                       marker="o", s=46, label="experimental generalized, K8 control (same graph as native)", zorder=2)
            ax.set_xlabel(xl)
            ax.set_ylabel(label)
            if metric != "pair_pct":
                ax.set_yscale("log")
            ax.grid(True, color="#e6e5e0", linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=5)
        fig.suptitle("Fixed weight budget W = 800M valid connection weights (H = 800M / K; D = G = 1024, M = 8192, "
                     "Mean(ReLU), BF16/split, FusedAdamW) — RTX 3090, graph seed 0", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / fname)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(KC.RAW))
    a = ap.parse_args()
    raw = Path(a.raw)
    cov = load_cov(raw)
    runs = load_runs(raw)
    rows = [run_row(r, cov) for r in runs]
    write_csv(raw / "csv/timing_runs.csv", rows)
    write_csv(raw / "csv/coverage.csv", [cov_row(c) for c in cov.values()])
    plots(rows, raw / "plots")
    # markdown tables
    md = []
    md.append("| ID | K | C | S | R | H | backend | class | status | fwd | dX | dW | bwd | opt | step (min–max) | peak GiB | first step s | pair % | triple % | quad % |")
    md.append("|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    order = {c["id"]: i for i, c in enumerate(KC.cases())}
    order["X01"] = 99
    for r in sorted([r for r in rows if r["origin"] in ("screen", "native-main")],
                    key=lambda r: (order.get(r["id"], 100), r["backend"], r["origin"], r["tag"])):
        if r["status"] != "measured":
            md.append(f"| {r['id']} | {r['K']} | {r['C']} | {r['S']} | {r['R']} | {r['H']:,} | {r['backend']} | – | "
                      f"**{r['status']}** | – | – | – | – | – | – | – | – | – | – | – |")
            continue
        be = r["backend"] + (" (main)" if r["origin"] == "native-main" else "")
        md.append(f"| {r['id']} | {r['K']} | {r['C']} | {r['S']} | {r['R']} | {r['H']:,} | {be} | {r['exec_class']} | "
                  f"measured | {fmt(r['fwd_total_ms'])} | {fmt(r['dx_kernel_ms'])} | {fmt(r['dw_kernel_ms'])} | "
                  f"{fmt(r['bwd_total_ms'])} | {fmt(r['optimizer_ms'])} | {fmt(r['step_ms'])} ({fmt(r['step_min_ms'])}–"
                  f"{fmt(r['step_max_ms'])}) | {fmt(r['peak_alloc_GiB'], 2)} | {fmt(r['first_step_s'], 1)} | "
                  f"{fmt(r.get('pair_pct'), 2)} | {fmt(r.get('triple_pct'), 3)} | {fmt(r.get('quad_pct'), 4)} |")
    md += ["", "### Confirmation (graph seeds 0/1/2; per seed: median of 24 restored steps)", "",
           "| ID | K | C | S | R | backend | step per seed (ms) | mean | seed sd | 95 % CI (t, n=3) | dX mean | dW mean | "
           "triple % per seed | quad % per seed |", "|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|---:|---|---|"]
    conf = [r for r in rows if r["origin"] == "confirm" and r["id"] != "X01"]
    from scipy.stats import t as tdist

    groups = {}
    for r in conf:
        groups.setdefault((r["id"], r["backend"]), []).append(r)
    summary = []
    for (cid, be), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 100), kv[0][1])):
        rs = sorted(rs, key=lambda r: r["graph_seed"])
        ok = [r for r in rs if r["status"] == "measured"]
        st = [r["step_ms"] for r in ok]
        m = statistics.mean(st) if st else None
        sd = statistics.stdev(st) if len(st) > 1 else None
        hw = tdist.ppf(0.975, len(st) - 1) * sd / math.sqrt(len(st)) if sd is not None else None
        r0 = rs[0]
        summary.append(dict(id=cid, backend=be, K=r0["K"], C=r0["C"], S=r0["S"], R=r0["R"], H=r0["H"], seeds=len(rs),
                            measured=len(ok), steps_ms=json.dumps([round(x, 1) for x in st]), mean_ms=m, seed_sd_ms=sd,
                            ci95_halfwidth_ms=hw, statuses=json.dumps([r["status"] for r in rs])))
        md.append(f"| {cid} | {r0['K']} | {r0['C']} | {r0['S']} | {r0['R']} | {be} | "
                  f"{', '.join(fmt(x) for x in st)} | {fmt(m)} | {fmt(sd)} | ±{fmt(hw)} | "
                  f"{fmt(statistics.mean(r['dx_kernel_ms'] for r in ok) if ok else None)} | "
                  f"{fmt(statistics.mean(r['dw_kernel_ms'] for r in ok) if ok else None)} | "
                  f"{', '.join(fmt(r.get('triple_pct'), 3) for r in ok)} | {', '.join(fmt(r.get('quad_pct'), 4) for r in ok)} |")
    write_csv(raw / "csv/confirm_summary.csv", summary)
    pairs = []
    for cid in sorted({k[0] for k in groups}):
        a_, b_ = groups.get((cid, "native")), groups.get((cid, "kgen_native"))
        if a_ and b_:
            na = {r["graph_seed"]: r["step_ms"] for r in a_ if r["status"] == "measured"}
            nb = {r["graph_seed"]: r["step_ms"] for r in b_ if r["status"] == "measured"}
            d = [nb[s] / na[s] for s in sorted(na) if s in nb]
            pairs.append(f"| {cid} | {', '.join(f'{x:.3f}' for x in d)} | {statistics.mean(d):.3f} |")
    if pairs:
        md += ["", "Matched K8 control: experimental / native step ratio per graph seed (same graph and parameters)", "",
               "| ID | ratio per seed | mean |", "|---|---|---:|"] + pairs
    base = [r for r in rows if r["id"] == "X01" and r["backend"] == "native" and r["status"] == "measured"]
    md += ["", "X01 native drift baselines (step ms): " + "; ".join(f"{r['origin']} {r['tag'].split('_')[-1]} {r['step_ms']:.1f}"
                                                                for r in base)]
    (raw / "tables.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} timing rows, {len(cov)} coverage records -> {raw}/csv, tables.md, plots/")


if __name__ == "__main__":
    main()
