"""Render results/*.json into results/TABLES.md (measured numbers only).

    python sparse_pooling/benchmarks/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def gib(b):
    return f"{b / 2**30:.2f} GiB" if b is not None else "-"


def mib(b):
    return f"{b / 2**20:.1f} MiB" if b is not None else "-"


def e2e_table(name, title):
    d = load(name)
    if not d:
        return f"\n_{name}: not run_\n"
    rerun = load(name + "_rerun")  # cases re-measured after a fix replace the originals
    if rerun:
        newer = {r["name"]: r for r in rerun["records"]}
        d["records"] = [dict(newer[r["name"]], rerun=True) if r["name"] in newer else r for r in d["records"]]
    a = d["args"]
    out = [f"\n### {title}\n",
           f"H = {a['hidden']:,}, B = {a['batch']}, grad precision = {a['grad_precision']}, "
           f"optimizer step included in 'step' = {a['optimizer']}  ({d['gpu']['name']}, {d['timestamp']})\n",
           "| Pooling | C | Levels | Policy | Forward ms | Backward ms | Fwd+Bwd ms | Step ms | Peak VRAM | Ckpt | tokens/s | fwd TFLOPs | backends | Correct? |",
           "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in d["records"]:
        if "error" in r:
            out.append(f"| {r['name']} | {r.get('C', '')} | | | {r['error']} | | | | | | | | | |")
            continue
        out.append(
            f"| {r['name']} | {r['C']} | {r['levels']} | {r['effective_policy']} | {r['forward_ms']:.1f} "
            f"| {r['backward_ms']:.1f} | {r['forward_ms'] + r['backward_ms']:.1f} | {r['step_ms']:.1f} "
            f"| {gib(r['peak_bytes'])} | {mib(r['checkpoint_bytes'])} | {r['tokens_per_s']:.0f} "
            f"| {r['fwd_useful_TFLOPs']:.1f} | {r['forward_backend']}/{r['backward_backend']} | yes (tests) |"
            + (" (re-run)" if r.get("rerun") else ""))
    return "\n".join(out) + "\n"


def scale_table():
    rows = []
    for H in (1_000_000, 10_000_000, 100_000_000):
        for B in (256, 1024, 4096, 8192):
            d = load(f"e2e_scale_H{H}_B{B}")
            if not d:
                continue
            for r in d["records"]:
                if "error" in r:
                    continue
                rows.append((H, B, r))
    if not rows:
        return "\n_scaling: not run_\n"
    out = ["\n### Scaling with hidden width and batch (C = 16, split-precision gradients)\n",
           "| H | B | Pooling | Forward ms | Backward ms | Fwd+Bwd ms | Peak VRAM | fwd TFLOPs |",
           "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for H, B, r in rows:
        out.append(f"| {H:,} | {B} | {r['name']} | {r['forward_ms']:.1f} | {r['backward_ms']:.1f} "
                   f"| {r['forward_ms'] + r['backward_ms']:.1f} | {gib(r['peak_bytes'])} | {r['fwd_useful_TFLOPs']:.1f} |")
    return "\n".join(out) + "\n"


def sweep_table():
    d = load("sweep_H10M_B8192")
    if not d:
        return "\n_sweep: not run_\n"
    out = [f"\n### Configuration sweep, forward only (H = {d['args']['hidden']:,}, B = {d['args']['batch']})\n",
           "Traffic columns are analytic L2->SM bytes per forward (weights/bias, index records, gathered inputs).\n",
           "| C | Pool | Backend | Batch tile | Splits | Input layout | Scheduling | Forward ms | TFLOPs | Workspace | Weight GB | Index GB | Gather GB |",
           "|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in d["records"]:
        t = f"{r['fwd_ms']:.1f} | {r['tflops']:.1f}" if "fwd_ms" in r else f"{r.get('error', '')} | "
        out.append(f"| {r['C']} | {r['pool']} | {r['backend']} | {r['batch_tile']} | {r['splits']} | {r['input_layout']} "
                   f"| {r['scheduling']} | {t} | {mib(r['workspace_bytes'])} | {r['weight_bytes'] / 1e9:.1f} "
                   f"| {r['index_bytes'] / 1e9:.2f} | {r['gather_bytes'] / 1e9:.0f} |")
    return "\n".join(out) + "\n"


def tradeoff_table():
    d = load("tradeoff_H262144_B1024")
    if not d:
        return "\n_trade-off: not run_\n"
    a = d["args"]
    out = [f"\n### Memory-compute trade-off (H = {a['hidden']:,}, B = {a['batch']}, C = {a['C']})\n",
           "Errors are relative L2 errors of dX / dW against an FP64 reference on the same BF16 operands.\n",
           "| Operator | Strategy | Policy | Forward ms | Backward ms | Total ms | Peak extra memory | Checkpoint | dX err | dW err |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["records"]:
        if "error" in r:
            out.append(f"| {r['op']} | {r['strategy']} | | {r['error']} | | | | | | |")
            continue
        out.append(f"| {r['op']} | {r['strategy']} | {r.get('policy', 'autograd')} | {r['fwd_ms']:.2f} | {r['bwd_ms']:.2f} "
                   f"| {r['total_ms']:.2f} | {mib(r['peak_extra_bytes'])} | {mib(r.get('checkpoint_bytes'))} "
                   f"| {r['dx_err']:.1e} | {r['dw_err']:.1e} |")
    return "\n".join(out) + "\n"


def memory_table(name, title):
    d = load(name)
    if not d:
        return f"\n_{name}: not run_\n"
    out = [f"\n### {title}\n", "| Pooling | Policy | L1 ckpt | Measured peak | Steady | Checkpoint | Optimizer state | Model total |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in d["records"]:
        if "error" in r:
            out.append(f"| {r['pooling']} | | | {r['error']} | | | | |")
            continue
        pol = r["policy"] + (" (fallback: ids too large)" if r.get("policy_fallback") else "")
        out.append(f"| {r['pooling']} | {pol} | {r['level1_checkpoint'] or '-'} | {gib(r['measured_peak_bytes'])} "
                   f"| {gib(r['measured_steady_bytes'])} | {mib(r['checkpoint_bytes'])} | {gib(r['optimizer_state_bytes'])} "
                   f"| {gib(r['model']['total'])} |")
    m = d["records"][0]["model"]
    out.append("\nComponent model (first row):\n")
    out.append("| Component | Bytes |\n|---|---:|")
    for k, v in m.items():
        out.append(f"| {k} | {gib(v) if v > 2**30 else mib(v)} |")
    return "\n".join(out) + "\n"


def main():
    parts = ["# Measured results (generated by benchmarks/make_report.py)\n"]
    parts.append(e2e_table("e2e_final_100M_C8", "Final comparison, H = 100M, B = 8192, C = 8"))
    parts.append(e2e_table("e2e_final_100M_C64", "Final comparison, H = 100M, B = 8192, C = 64"))
    parts.append(e2e_table("e2e_bf16_100M_C8", "Single-BF16 dZ gradients, C = 8"))
    parts.append(e2e_table("e2e_bf16_100M_C64", "Single-BF16 dZ gradients, C = 64"))
    parts.append(e2e_table("e2e_lme_temps_100M", "LogMeanExp temperatures (no optimizer)"))
    parts.append(e2e_table("e2e_topk_100M", "Top-K sizes (no optimizer)"))
    parts.append(e2e_table("e2e_power_100M", "Power-mean exponents (no optimizer)"))
    parts.append(scale_table())
    parts.append(sweep_table())
    parts.append(tradeoff_table())
    parts.append(memory_table("memory_H100M_B8192", "Memory at full scale (BF16 Adam moments, no master weights)"))
    parts.append(memory_table("memory_H100M_B8192_fp32master", "Memory at full scale (FP32 Adam moments + FP32 master weights)"))
    (R / "TABLES.md").write_text("\n".join(parts))
    print("wrote", R / "TABLES.md")


if __name__ == "__main__":
    main()
