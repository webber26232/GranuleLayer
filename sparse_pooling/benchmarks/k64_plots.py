"""Plots of the K64 study from the CSVs written by k64_report.py (results/k64_large_c_indexfree_l1/plots)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import k64_cases as KC  # noqa: E402

CS = (64, 128, 256, 512, 1024, 2048)


def rd(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def f(x):
    return float(x) if x not in ("", None) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(KC.RAW))
    a = ap.parse_args()
    raw = Path(a.raw)
    out = raw / "plots"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rd(raw / "csv" / "dense_screen_variants.csv") if r["status"] == "measured" and r["case"].startswith("K64")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, S in zip(axes, (64, 128)):
        for R, mk in zip((64, 128, 256), ("o", "s", "^")):
            for fam, ls in (("ref", ":"), ("direct", "--"), ("idf", "-")):
                ys = []
                for C in CS:
                    cand = [f(r["step"]) for r in rows if int(r["C"]) == C and int(r["S"]) == S and int(r["R"]) == R
                            and (r["variant"] == "ref" if fam == "ref" else r["variant"].startswith(fam)
                                 and not r["variant"].startswith("idf_rmw"))]
                    ys.append(min(cand) if cand else None)
                if any(y is not None for y in ys):
                    ax.plot(CS, ys, ls, marker=mk, label=f"R{R} {fam}")
        ax.set_xscale("log", base=2)
        ax.set_title(f"S = {S}: full training step (graph seed 0)")
        ax.set_xlabel("logical C")
        ax.set_ylim(1400, 2200)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("step ms (best tile of each family)")
    axes[0].legend(fontsize=6, ncol=3)
    axes[1].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "dense_step_vs_C.png", dpi=130)
    plt.close(fig)
    sp = [r for r in rd(raw / "csv" / "sparse_c256.csv") if r.get("status") == "measured"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (mask, fmt, inf), lab in ((("tile", "tile", "False"), "tile-sparse (tile mask), training"),
                                  (("tile", "dense_masked", "False"), "dense masked (tile mask), training"),
                                  (("magnitude", "csr", "False"), "CSR (magnitude mask), training"),
                                  (("magnitude", "dense_masked", "False"), "dense masked (magnitude), training"),
                                  (("tile", "tile", "True"), "tile-sparse, inference (fwd)"),
                                  (("magnitude", "csr", "True"), "CSR, inference (fwd)")):
        pts = sorted((f(r["density_target"]), f(r["step"])) for r in sp if r["mask"] == mask and r["fmt"] == fmt
                     and r["inference"] == inf and r["density_target"])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight density (kept fraction)")
    ax.set_ylabel("ms (median)")
    ax.set_title("C256 S64 R256 frozen-mask formats, M = 8192")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sparse_c256_step_vs_density.png", dpi=130)
    plt.close(fig)
    cov = rd(raw / "csv" / "dense_coverage.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for S, R, ls in ((64, 256, "-"), (128, 256, "--")):
        r_ = sorted((int(r["C"]), r) for r in cov if int(r["S"]) == S and int(r["R"]) == R)
        ax.plot([c for c, _ in r_], [100 * f(r["triple"]) for _, r in r_], ls, marker="o", label=f"layer-wide triples S{S} R{R}")
        ax.plot([c for c, _ in r_], [100 * f(r["po_pair_med"]) for _, r in r_], ls, marker="s", label=f"per-output pairs (median) S{S} R{R}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("logical C")
    ax.set_ylabel("% of tuples")
    ax.set_title("Dense potential coverage: layer-wide vs per output")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "coverage_layer_vs_output.png", dpi=130)
    plt.close(fig)
    mc = [r for r in rd(raw / "csv" / "mask_coverage.csv") if r.get("status") == "measured" and r["C"] == "256"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, mk in (("magnitude", "o"), ("tile", "s"), ("random", "^"), ("random_tile", "v"), ("learned", "D")):
        pts = sorted((f(r["density"]), r) for r in mc if r["kind"].startswith(kind + " ") or
                     (kind == "learned" and r["kind"].startswith("learned")))
        if not pts:
            continue
        axes[0].plot([d for d, _ in pts], [100 * f(r["triple"]) for _, r in pts], marker=mk, label=kind)
        axes[1].plot([d for d, _ in pts], [100 * f(r["po_pair_med"]) for _, r in pts], marker=mk, label=kind)
    for ax, t in zip(axes, ("layer-wide triple coverage %", "per-output pair coverage % (median)")):
        ax.set_xscale("log")
        ax.set_xlabel("density")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Retained-support coverage, C256 S64 R256 (per-neuron supports)")
    fig.tight_layout()
    fig.savefig(out / "mask_coverage_c256.png", dpi=130)
    plt.close(fig)
    print("plots written to", out)


if __name__ == "__main__":
    main()
