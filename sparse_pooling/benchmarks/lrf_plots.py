"""Learning-curve figures of the A3mean follow-up from its raw per-job records (validation curves; test markers at milestones once unlocked).

    python lrf_plots.py --out RUN --dest sparse_pooling/results/learned_reduction_followup/figures

Each panel draws the seed mean as a line and the per-seed range as a light band (paired seeds, n = 3).  Colors follow the model, never
its rank: A0 blue, A2 orange, A3mean aqua (first three slots of the validated reference palette).  Line style encodes the execution mode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lrf_report import load  # noqa: E402

COLOR = {"A0": "#2a78d6", "A2": "#eb6834", "A3mean": "#1baf7a"}
MODELS = list(COLOR)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
XLABEL = {"update": "optimizer updates", "samples": "training samples", "train_s": "synchronized training time (s)", "wall_s": "full wall time (s)"}


def style(ax, title, xfield, logy=True):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(XLABEL[xfield], color=MUTED, fontsize=8)
    ax.set_ylabel("validation MSE", color=MUTED, fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def seed_curves(rs, model, task, xfield):
    """Per-seed (x, val) lists, skipping the untrained point at update 0 (it would set the log scale)."""
    out = []
    for r in rs:
        if r["model"] == model and r["task"] == task:
            c = [(e[xfield], e["val"]) for e in r["curve"] if e.get("val") is not None and e["update"] > 0]
            out.append(c)
    return out


def band(ax, curves, color, ls="-", label=None, end_label=True):
    """Seed mean line and min-max band, on the x grid shared by the seeds (identical eval schedule), else per-seed lines only."""
    if not curves:
        return
    xs = [x for x, _ in curves[0]]
    if all([x for x, _ in c] == xs for c in curves):
        ys = list(zip(*[[y for _, y in c] for c in curves]))
        m = [sum(v) / len(v) for v in ys]
        ax.fill_between(xs, [min(v) for v in ys], [max(v) for v in ys], color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, m, color=color, linewidth=2, linestyle=ls, label=label)
        if end_label and label:
            ax.annotate(label, (xs[-1], m[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    else:
        for i, c in enumerate(curves):
            ax.plot([x for x, _ in c], [y for _, y in c], color=color, linewidth=1, linestyle=ls, label=label if i == 0 else None)


def fig_stage_b(B, dest):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, t in enumerate(("compose", "interact")):
        for j, xf in enumerate(("update", "train_s", "wall_s")):
            ax = axes[i, j]
            for m in MODELS:
                band(ax, seed_curves(B, m, t, xf), COLOR[m], label=m)
            style(ax, f"{t.capitalize()}: validation vs {XLABEL[xf].split(' (')[0]}", xf)
            if xf == "update":
                ax.axvspan(12288, 16384, color=GRID, alpha=0.4, linewidth=0)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Stage B: M_eff = 512, 16,384 updates (8,388,608 samples); line = mean of 3 seeds, band = seed range; shaded = late horizon", fontsize=9, color=MUTED)
    fig.savefig(dest / "stage_B_curves.png", dpi=130)
    plt.close(fig)


def fig_batch(B, Dd, Da, dest):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, t in enumerate(("compose", "interact")):
        for j, xf in enumerate(("samples", "train_s", "wall_s")):
            ax = axes[i, j]
            for m in MODELS:
                band(ax, seed_curves(B, m, t, xf), COLOR[m], "-", f"{m} 512", end_label=False)
                band(ax, seed_curves(Dd, m, t, xf), COLOR[m], "--", f"{m} 8192 direct", end_label=False)
                band(ax, seed_curves(Da, m, t, xf), COLOR[m], ":", f"{m} 512x16 accum", end_label=False)
            style(ax, f"{t.capitalize()}: validation vs {XLABEL[xf].split(' (')[0]}", xf)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, frameon=False, ncol=3, loc="outside lower center")
    fig.suptitle("Batch scaling: solid = M_eff 512 (Stage B), dashed = 8,192 direct, dotted = 512 x 16 accumulated (Stage D); same samples at the right edge of Stage D", fontsize=9, color=MUTED)
    fig.savefig(dest / "batch_scaling_curves.png", dpi=130)
    plt.close(fig)


def fig_local(C_, dest):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, xf in zip(axes, ("update", "train_s")):
        for m in MODELS:
            band(ax, seed_curves(C_, m, "local", xf), COLOR[m], label=m)
        style(ax, f"Local: validation vs {XLABEL[xf].split(' (')[0]}", xf, logy=False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(dest / "stage_C_local_curves.png", dpi=130)
    plt.close(fig)


def fig_p(B, dest):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (field, lab) in zip(axes, (("rel_movement", "||P - P0|| / ||P0||"), ("positive", "fraction of P > 0"))):
        for t, ls in (("compose", "-"), ("interact", "--")):
            for r in B:
                if r["model"] == "A3mean" and r["task"] == t and r["p_stats"]:
                    ax.plot([p["update"] for p in r["p_stats"]], [p[field] for p in r["p_stats"]], color=COLOR["A3mean"], linestyle=ls, linewidth=1.5,
                            label=f"{t}" if r["seed"] == 0 else None)
        ax.set_title(f"A3mean P: {lab}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("optimizer updates", color=MUTED, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, title="task (3 seeds each)", title_fontsize=7)
    fig.savefig(dest / "stage_B_p_movement.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    root, dest = Path(a.out), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    B, C_, Dd, Da = (load(root, s) for s in ("B", "C", "D_direct", "D_accum"))
    if B:
        fig_stage_b(B, dest)
        fig_p(B, dest)
    if C_:
        fig_local(C_, dest)
    if B and Dd:
        fig_batch(B, Dd, Da, dest)
    print(sorted(p.name for p in dest.glob("*.png")))


if __name__ == "__main__":
    main()
