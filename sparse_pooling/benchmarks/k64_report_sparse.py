"""L1 / sparse / maintenance / mask-coverage / 2:4 tables of the K64 study (called by k64_report.py)."""
from __future__ import annotations

import json
from pathlib import Path

from k64_report import fmt, load_status, med, write_csv

PHASES = ("fwd_kernel", "fwd_bias_only", "fwd_total", "dx_kernel", "dw_kernel", "bwd_bias_only", "bwd_total", "optimizer",
          "step")


def learn_tables(raw: Path, lines: list):
    rows = []
    for sub in ("learn", "escalate"):
        for s in load_status(raw / sub):
            if s["status"] != "measured":
                rows.append(dict(tag=s["tag"], status=s["status"]))
                continue
            j = json.loads((raw / sub / s["json"]).read_text())
            c, cal, st, ft = j["config"], j["calibration"], j["checkpoint_stats"], j["finetune_stats"]
            ev = j["evals"]
            reg = [e for e in ev if e["phase"] == c["treatment"]]
            fin = [e for e in ev if e["phase"] == "finetune_fixed_mask"]
            warm = next(e for e in ev if e["phase"] == "warmup")
            steps = j["steps"]
            rr = [x for x in steps if x["phase"] == c["treatment"]]
            rows.append(dict(tag=s["tag"], status="measured", C=c["C"], treatment=c["treatment"], alpha=c["alpha"],
                             g_ref=cal["g_ref"], fallback=cal["fallback"], mu=cal["mu"], lambda_mean=cal["lambda_mean"],
                             heldout_init=ev[0]["heldout"], heldout_warmup=warm["heldout"],
                             heldout_reg=reg[-1]["heldout"] if reg else None, heldout_ft=fin[-1]["heldout"] if fin else None,
                             train_loss_last_reg=rr[-1]["loss"] if rr else None, density=st["density_nonzero"],
                             exact_zeros=st["exact_zeros"], nz_1e3=st["near_zero_fraction"]["0.001"],
                             nz_1e2=st["near_zero_fraction"]["0.01"], all_zero_neurons=st["all_zero_neurons"],
                             fan_in_mean=st["active_fan_in_mean"], l1_mean_abs=st["mean_abs"],
                             l1_term=cal["mu"] * st["l1"], gnorm_layer_last=rr[-1]["grad_norm_layer"] if rr else None,
                             step_ms_reg=sorted(x["step_ms"] for x in rr)[len(rr) // 2] if rr else None,
                             opt_ms_reg=sorted(x["opt_ms"] for x in rr)[len(rr) // 2] if rr else None,
                             ft_density=ft["density_nonzero"], mask_preserved=j["finetune_mask_preserved"],
                             checkpoint=j["checkpoint"]["path"], weight_sha256=j["checkpoint"]["weight_sha256"][:12],
                             mask_sha256=(j["checkpoint"].get("mask_sha256") or "")[:12],
                             fan_hist=st["active_fan_in_hist"]))
    write_csv(raw / "csv" / "l1_learning.csv", rows)
    lines += ["", "## L1 learning diagnostic (M_train = 512, S64 R256, full H / W0; synthetic composed task)", "",
              "Objective L = L_task + mu sum_valid |w|, lambda_mean = mu W0 (W0 = 800M fixed). prox: diagonal-metric "
              "proximal AdamW (data-gradient moments); coupled: mu sign(w) in main_grad. mu = alpha g_ref.", "",
              "| run | C | treatment | alpha | g_ref | mu | lambda_mean | held-out: warm-up → after L1 → fine-tuned | "
              "density (exact nonzero) | |w|<=1e-3 | all-zero neurons | mean active fan-in | L1 term | opt ms |",
              "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] != "measured":
            lines.append(f"| {r['tag']} | {r['status']} ||||||||||||")
            continue
        lines.append(f"| {r['tag']} | {r['C']} | {r['treatment']} | {r['alpha']:g} | {r['g_ref']:.3e} | {r['mu']:.3e} | "
                     f"{r['lambda_mean']:.3e} | {r['heldout_warmup']:.5f} → {r['heldout_reg']:.5f} → {r['heldout_ft']:.5f} | "
                     f"{r['density']:.4f} | {r['nz_1e3']:.4f} | {r['all_zero_neurons']:,} | {r['fan_in_mean']:.2f} | "
                     f"{r['l1_term']:.3e} | {fmt(r['opt_ms_reg'], 2)} |")
    return rows


def sparse_rows(raw: Path, sub: str):
    rows = []
    for s in load_status(raw / sub):
        job = s.get("job", {})
        if s["status"] != "measured" or job.get("kind") != "timing":
            if job.get("kind") == "timing":
                rows.append(dict(tag=s["tag"], status=s["status"]))
            continue
        j = json.loads((raw / sub / s["json"]).read_text())
        (nm, v), = j["variants"].items()
        gs = j["graph_stats"]
        r = dict(tag=s["tag"], status="measured", C=j["config"]["C"], fmt=job["factory"].get("format"),
                 mask=job["factory"].get("mask"), density_target=job["factory"].get("density"),
                 density=gs.get("mask_density"), mask_sha=(gs.get("mask_sha256") or "")[:12],
                 inference=bool(job.get("inference")), l1=job["opt"].get("l1"), variant=(v.get("execution") or {}).get("variant"),
                 peak_gib=v["peak_allocated_bytes"] / 2**30, restore_checks=v["restore_checks"],
                 compact_values=j["persistent_bytes"].get("compact_values"), sparse_metadata=j["persistent_bytes"].get("sparse_metadata"),
                 opt_state=j["persistent_bytes"]["adam_moments"], main_grad=j["persistent_bytes"]["main_grad"],
                 conversion_s=gs.get("conversion_seconds"), mask_s=gs.get("mask_seconds"),
                 params_sha=j["kcsr"]["logical_params_sha"][:12] if j["kcsr"].get("logical_params_sha") else None,
                 graph_fp=j["kcsr"]["graph_fingerprint"][:12], immutable_ok=j.get("immutable_metadata_unchanged"),
                 work=(v.get("last_info") or {}).get("kgx_work_lists", {}))
        for ph in PHASES:
            r[ph] = med(v, ph)
        rows.append(r)
    return rows


def sparse_tables(raw: Path, lines: list):
    out = {}
    for sub in sorted(p.name for p in raw.glob("sparse*") if p.is_dir()):
        rows = sparse_rows(raw, sub)
        out[sub] = rows
        write_csv(raw / "csv" / f"{sub}.csv", [{k: v for k, v in r.items() if k != "work"} for r in rows])
        lines += ["", f"## Frozen-mask sparse timing: {sub} (M = 8192, full H; median of 8 restored steps)", "",
                  "Same checkpoint values, mask and optimizer state on the dense masked reference and on each format; "
                  "training rows include fwd + bwd + L1AdamW (masked dense / compact values); inference rows: forward only.", "",
                  "| run | mask | density | format | fwd | dX | dW | opt | step | vs dense masked (same mask) | peak GiB | "
                  "values + metadata MB | bias-only rows |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        dm = {(r.get("mask"), r.get("density_target"), r.get("mask_sha")): r for r in rows
              if r.get("fmt") == "dense_masked" and not r.get("inference")}
        for r in rows:
            if r["status"] != "measured":
                lines.append(f"| {r['tag']} | {r['status']} |||||||||||")
                continue
            ref = dm.get((r["mask"], r["density_target"], r["mask_sha"]))
            rel = f"×{ref['step'] / r['step']:.2f}" if ref and not r["inference"] and r is not ref else "–"
            if r["inference"] and ref:
                rel = f"fwd ×{ref['fwd_total'] / r['fwd_total']:.2f}"
            mb = ((r["compact_values"] or 0) + (r["sparse_metadata"] or 0)) / 2**20 if r["compact_values"] else None
            lines.append(f"| {r['tag']} | {r['mask']} | {fmt(r['density'], 4)} | {r['fmt']}{' (inference)' if r['inference'] else ''} | "
                         f"{fmt(r['fwd_total'])} | {fmt(r['dx_kernel'])} | {fmt(r['dw_kernel'])} | {fmt(r['optimizer'])} | "
                         f"{fmt(r['step'])} | {rel} | {r['peak_gib']:.2f} | {fmt(mb, 0)} | "
                         f"{fmt(r['work'].get('bias_only_neurons'), 0) if r['work'] else '–'} |")
    return out


def refresh_tables(raw: Path, lines: list):
    rows = []
    for s in load_status(raw / "refresh"):
        if s["status"] != "measured":
            continue
        j = json.loads((raw / "refresh" / s["json"]).read_text())
        rows.append(dict(tag=s["tag"], C=j["C"], format=j["format"], mask=j["mask"], density=j["density"],
                         select=j["select"]["median_s"], compress=j["compress"]["median_s"],
                         work_lists=j["work_lists"]["median_s"], state_remap=j["state_remap"]["median_s"],
                         densify=j["densify"]["median_s"], U=j["U_seconds"],
                         metadata_bytes=sum(v for k, v in j["bytes"].items() if k not in ("dense_values", "mask_uint8",
                                                                                            "compact_values", "nnz",
                                                                                            "active_tiles"))))
    write_csv(raw / "csv" / "refresh.csv", rows)
    lines += ["", "## Support maintenance U (seconds; full size; outside the timed steps)", "",
              "| run | format | mask | density | select | compress | work lists | state remap | U | densify |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['format']} | {r['mask']} | {r['density']:.4f} | {r['select']:.2f} | "
                     f"{r['compress']:.2f} | {r['work_lists']:.2f} | {r['state_remap']:.2f} | {r['U']:.2f} | {r['densify']:.2f} |")
    return rows


def maskcov_tables(raw: Path, lines: list):
    rows = []
    for s in load_status(raw / "maskcov"):
        if s["status"] != "measured":
            rows.append(dict(tag=s["tag"], status=s["status"]))
            continue
        j = json.loads((raw / "maskcov" / s["json"]).read_text())
        m, q, po, pf = j["mask"], j["quad"], j["per_output_summary"], j["per_output_features"]
        rows.append(dict(tag=s["tag"], status="measured", C=j["C"], kind=j["kind"], density=m["density"],
                         mask_sha=m["mask_sha256"][:12], pair=j["pair"]["fraction"], triple=j["triple"]["fraction"],
                         quad=q["fraction"], quad_lo=q["ci95"][0], quad_hi=q["ci95"][1], quad_n=q["sample_size"],
                         quad_upper_if_zero=q["upper_bound_if_zero"], po_feat_min=pf["min"], po_feat_med=pf["median"],
                         po_pair_med=po["pair"]["median"], po_pair_min=po["pair"]["min"], po_triple_med=po["triple"]["median"],
                         po_quad_med=po["quad"]["median"], all_zero=j["supports"]["all_zero_neurons"],
                         fan_in_med=j["supports"]["active_fan_in"]["median"],
                         iid_triple=j["iid_bernoulli_reference"]["local_hit"]["3"] if "iid_bernoulli_reference" in j else None,
                         seconds=j["analysis_seconds"]))
    write_csv(raw / "csv" / "mask_coverage.csv", rows)
    lines += ["", "## Retained-support coverage (per-neuron supports; S64 R256; graph seed 0)", "",
              "Exact pairs / triples; quadruples: fixed 10M sample (50M at >= 90 % sparsity). Per-output: all G outputs for "
              "features; >= 32 selected outputs for pairs / triples / quadruples.", "",
              "| run | C | mask | density | pair % | triple % | quad % [CI] | per-output features min/med | "
              "per-output pair % med | per-output triple % med | all-zero neurons | fan-in med |",
              "|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] != "measured":
            lines.append(f"| {r['tag']} | {r['status']} ||||||||||")
            continue
        lines.append(f"| {r['tag']} | {r['C']} | {r['kind']} | {r['density']:.4f} | {100 * r['pair']:.4f} | "
                     f"{100 * r['triple']:.4f} | {100 * r['quad']:.4f} [{100 * r['quad_lo']:.4f}, {100 * r['quad_hi']:.4f}] | "
                     f"{r['po_feat_min']:.0f}/{r['po_feat_med']:.0f} | {100 * r['po_pair_med']:.3f} | "
                     f"{100 * r['po_triple_med']:.4f} | {r['all_zero']:,} | {r['fan_in_med']:.0f} |")
    return rows


def semi24_tables(raw: Path, lines: list):
    p = raw / "semi24" / "panel.json"
    if not p.exists():
        return
    j = json.loads(p.read_text())
    lines += ["", "## 2:4 feasibility (PANEL MICROBENCHMARK only; no training claim)", "",
              f"cuSPARSELt available: {j['cusparselt_available']} (version {j['cusparselt_version']}); API "
              f"{j['semi_structured_api']}; device {j['device']} sm_{j['capability'][0]}{j['capability'][1]}.", "",
              "Audit: " + ", ".join(f"{k} = {v}" for k, v in j.get("audit", {}).items()), "",
              "| C | status | fwd dense ms | fwd 2:4 ms | dX dense ms | dX 2:4 ms | dW dense ms | compress fwd / dX ms per panel | "
              "fwd / dX max rel err |", "|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for r in j["panels"]:
        if r["status"] != "measured":
            lines.append(f"| {r['C']} | {r['status']}: {r.get('reason', '')[:120]} ||||||||")
            continue
        lines.append(f"| {r['C']} | measured | {r['dense_fwd']['median_ms']:.3f} | {r['sparse_fwd']['median_ms']:.3f} | "
                     f"{r['dense_dx']['median_ms']:.3f} | {r['sparse_dx']['median_ms']:.3f} | {r['dense_dw']['median_ms']:.3f} | "
                     f"{r['compress_fwd_ms_per_panel']:.3f} / {r['compress_dx_ms_per_panel']:.3f} | "
                     f"{r['fwd_max_rel_err']:.1e} / {r['dx_max_rel_err']:.1e} |")


def tables(raw: Path, lines: list):
    learn_tables(raw, lines)
    sparse_tables(raw, lines)
    refresh_tables(raw, lines)
    maskcov_tables(raw, lines)
    semi24_tables(raw, lines)
