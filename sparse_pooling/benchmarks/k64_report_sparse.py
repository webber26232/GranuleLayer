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
              if r.get("fmt") == "dense_masked" and not r.get("inference") and r.get("mask") != "none"}
        l1_none = next((r for r in rows if r.get("mask") == "none" and r.get("l1") == "none"), None)
        for r in rows:
            if r["status"] != "measured":
                lines.append(f"| {r['tag']} | {r['status']} |||||||||||")
                continue
            ref = dm.get((r["mask"], r["density_target"], r["mask_sha"]))
            rel = f"×{ref['step'] / r['step']:.2f}" if ref and not r["inference"] and r is not ref else "–"
            if r["mask"] == "none" and l1_none is not None and r is not l1_none:
                rel = f"×{l1_none['step'] / r['step']:.3f} vs dense none (opt {r['optimizer'] - l1_none['optimizer']:+.2f} ms)"
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
    for s in load_status(raw / "refresh2"):
        if s["status"] != "measured":
            continue
        j = json.loads((raw / "refresh2" / s["json"]).read_text())
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


def finetune_tables(raw: Path, lines: list):
    rows = []
    for s in load_status(raw / "finetune"):
        if s["status"] != "measured":
            continue
        j = json.loads((raw / "finetune" / s["json"]).read_text())
        rows.append(dict(tag=s["tag"], C=j["C"], mask=j["mask"], density=j["density"], h0=j["heldout_checkpoint"],
                         h1=j["heldout_pruned"], h2=j["heldout_finetuned"], all_zero=j["all_zero_neurons"]))
    if not rows:
        return
    write_csv(raw / "csv" / "finetune.csv", rows)
    lines += ["", "## Fixed-mask fine-tuning of pruned alpha = 0 checkpoints (synthetic diagnostic, M_train = 512, 32 "
              "updates, fresh optimizer state)", "", "| run | mask | density | held-out: checkpoint → pruned → fine-tuned | "
              "all-zero neurons |", "|---|---|---:|---|---:|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['mask']} | {r['density']:.4f} | {r['h0']:.5f} → {r['h1']:.5f} → {r['h2']:.5f} | "
                     f"{r['all_zero']:,} |")


def tables(raw: Path, lines: list):
    learn_tables(raw, lines)
    finetune_tables(raw, lines)
    confirm_tables(raw, lines)
    sparse_tables(raw, lines)
    refresh_tables(raw, lines)
    amortization_tables(raw, lines)
    maskcov_tables(raw, lines)
    semi24_tables(raw, lines)


def _t95(n):
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)


def confirm_tables(raw: Path, lines: list):
    import statistics as stt

    from k64_report import variant_rows

    rows = [r for r in variant_rows(raw, "confirm") if r["status"] == "measured"]
    write_csv(raw / "csv" / "confirm_dense.csv", rows)
    by = {}
    for r in rows:
        by.setdefault((r["case"], r["variant"]), {})[r["seed"]] = r
    lines += ["", "## Confirmation: dense (graph seeds 0 / 1 / 2; per seed the median of 24 restored steps)", "",
              "Paired ratio = reference step / variant step on the SAME graph and parameters, per seed; mean ± t-95 % over "
              "the 3 graph seeds (within-run samples are not treated as independent graph samples).", "",
              "| case | variant | step per seed (ms) | mean | seed sd | paired ratio vs ref per seed | mean ratio ± 95 % |",
              "|---|---|---|---:|---:|---|---|"]
    cases = sorted({k[0] for k in by}, key=lambda c: (c in KCX.ANCHORS, c))
    for c in cases:
        ref = by.get((c, "ref"), {})
        for (cc, v), seeds in sorted(by.items()):
            if cc != c:
                continue
            if c in KCX.ANCHORS:  # anchors: one entry per run
                steps = [r["step"] for r in rows if r["case"] == c]
                lines.append(f"| {c} (anchor) | {v} | {', '.join(f'{x:,.1f}' for x in steps)} | {stt.mean(steps):,.1f} | "
                             f"{stt.pstdev(steps):.1f} | – | – |")
                break
            ss = [seeds[s]["step"] for s in sorted(seeds)]
            rat = [ref[s]["step"] / seeds[s]["step"] for s in sorted(seeds) if s in ref]
            m = stt.mean(ss)
            sd = stt.stdev(ss) if len(ss) > 1 else 0.0
            rs = ", ".join(f"×{x:.3f}" for x in rat)
            ci = f"×{stt.mean(rat):.3f} ± {_t95(len(rat)) * stt.stdev(rat) / len(rat) ** 0.5:.3f}" if len(rat) > 1 else "–"
            lines.append(f"| {c} | {v} | {', '.join(f'{x:,.1f}' for x in ss)} | {m:,.1f} | {sd:.1f} | {rs} | {ci} |")
    srows = sparse_rows(raw, "confirm_sparse")
    write_csv(raw / "csv" / "confirm_sparse.csv", [{k: v for k, v in r.items() if k != "work"} for r in srows])
    if not srows:
        return
    lines += ["", "## Confirmation: frozen-mask sparse vs identical-mask dense masked (graph seeds 0 / 1 / 2)", "",
              "| combination | seed | dense masked step | sparse step | paired speedup | mask sha |", "|---|---:|---:|---:|---:|---|"]
    combo = {}
    for r in srows:
        if r["status"] != "measured":
            continue
        tag = r["tag"]
        seed = int(tag.rsplit("_g", 1)[1])
        key = (r["C"], r["mask"], r["density_target"])
        combo.setdefault(key, {}).setdefault(seed, {})[r["fmt"]] = r
    for key, seeds in sorted(combo.items()):
        rats = []
        for s, d in sorted(seeds.items()):
            dm = d.get("dense_masked")
            sp = next((v for k, v in d.items() if k != "dense_masked"), None)
            if dm and sp:
                rats.append(dm["step"] / sp["step"])
                lines.append(f"| C{key[0]} {key[1]} {key[2]:g} {sp['fmt']} | {s} | {dm['step']:,.1f} | {sp['step']:,.1f} | "
                             f"×{rats[-1]:.3f} | {sp['mask_sha']} (= dense: {sp['mask_sha'] == dm['mask_sha']}) |")
        if len(rats) > 1:
            lines.append(f"| **C{key[0]} {key[1]} {key[2]:g}** | mean | | | ×{stt.mean(rats):.3f} ± "
                         f"{_t95(len(rats)) * stt.stdev(rats) / len(rats) ** 0.5:.3f} | |")


import k64_cases as KCX  # noqa: E402


def amortization_tables(raw: Path, lines: list):
    """base + U / T from MEASURED components (C256): ordinary frozen-mask step, dense step, refresh U."""
    sp = {r["tag"]: r for r in sparse_rows(raw, "sparse_c256") if r["status"] == "measured"}
    dense = sp.get("C256_dense_l1_none")
    ref = {}
    for s in load_status(raw / "refresh2"):
        if s["status"] == "measured":
            j = json.loads((raw / "refresh2" / s["json"]).read_text())
            ref[(j["format"], round(j["density_target"], 4))] = j["U_seconds"] * 1e3
    rows = []
    lines += ["", "## Mask-refresh amortization (C256; measured components, combined arithmetically)", "",
              "Periodic reselection (no regrowth: reselects among stored values): step = s + U / T. Dynamic refresh with "
              "regrowth candidates: one DENSE step (dense gradient) + U every T steps: ((T - 1) s + dense + U) / T. "
              f"Dense step (dense kernels, no mask) = {dense['step']:.1f} ms." if dense else "", "",
              "| format | density | ordinary step s | U (ms) | periodic T=1 / 16 / 128 | dynamic T=1 / 16 / 128 | "
              "dense masked (same mask) | break-even of U vs dense masked |", "|---|---:|---:|---:|---|---|---:|---|"]
    for (fmt, mask) in (("tile", "tile"), ("csr", "magnitude")):
        for d in (0.5, 0.25, 0.1, 0.05):
            r = sp.get(f"C256_{mask}_d{d:g}_{fmt}")
            dm = sp.get(f"C256_{mask}_d{d:g}_dense_masked")
            U = ref.get((fmt, d))
            if not (r and U is not None and dense and dm):
                continue
            s_ = r["step"]
            per = [s_ + U / T for T in (1, 16, 128)]
            dyn = [((T - 1) * s_ + dense["step"] + U) / T for T in (1, 16, 128)]
            save = dm["step"] - s_
            be = f"{U / save:.2f} steps" if save > 0 else "none (no per-step saving)"
            rows.append(dict(format=fmt, density=d, s=s_, U=U, periodic=per, dynamic=dyn, dense_masked=dm["step"],
                             break_even=be))
            lines.append(f"| {fmt} | {d:g} | {s_:,.1f} | {U:,.0f} | {' / '.join(f'{x:,.0f}' for x in per)} | "
                         f"{' / '.join(f'{x:,.0f}' for x in dyn)} | {dm['step']:,.1f} | {be} |")
    write_csv(raw / "csv" / "amortization.csv", rows)
