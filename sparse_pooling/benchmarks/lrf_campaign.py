"""Declared campaign of the A3mean follow-up: the rules in configs/learned_reduction_followup/manifest.json as manifests of training jobs.
Selection reads validation losses only.

    python lrf_campaign.py recover --raw .../raw_quality.jsonl.gz --write recovered_lrs.json
    python lrf_campaign.py <stage> --out RUN --lrs recovered_lrs.json --write manifest.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

H_PRIMARY, H_SECONDARY = 10_485_760, 12_582_912
MODELS = ("A0", "A2", "A3mean")
SEEDS = (0, 1, 2)
GLOBAL_TASKS = ("compose", "interact")
TUNE_MULT = (1 / 3, 1.0, 3.0)
LARGE_MULT = (1.0, 4.0, 16.0)
PRIOR_STAGE = dict(A0="primary_final", A2="control_final", A3mean="primary_final")


def recover(raw, out):
    """Prior validation-selected (lr_w, lr_small, p_mult), exact values from the raw records, with their source record."""
    with gzip.open(raw, "rt") as h:
        rs = [json.loads(ln) for ln in h]
    lrs = {}
    for r in rs:
        m, t = r["model"], r["task"]
        if m not in MODELS:
            continue
        if (t in GLOBAL_TASKS and r["stage"] == PRIOR_STAGE[m] and r["seed"] == 0) or (t == "local" and r["stage"] == "screen_final"):
            e = dict(lr_w=r["lr_w"], lr_small=r["lr_small"], p_mult=r["p_mult"], source_stage=r["stage"], source_file=r["_file"], source_H=r["H"], source_pool=r["pool"])
            if f"{m}/{t}" in lrs and {k: lrs[f"{m}/{t}"][k] for k in ("lr_w", "lr_small", "p_mult")} != {k: e[k] for k in ("lr_w", "lr_small", "p_mult")}:
                raise RuntimeError(f"conflicting prior LR for {m}/{t}")
            lrs[f"{m}/{t}"] = e
    want = {f"{m}/{t}" for m in MODELS for t in (*GLOBAL_TASKS, "local")}
    if set(lrs) != want:
        raise RuntimeError(f"missing {want - set(lrs)}")
    Path(out).write_text(json.dumps(lrs, indent=1, sort_keys=True))
    return lrs


def load(out: Path, stage: str):
    rs = []
    for f in sorted((out / stage).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("schema") == "granule.lrf_train/1":
            rs.append(r)
    return rs


def candidate(prior, mult):
    return dict(lr_w=prior["lr_w"] * mult, lr_small=prior["lr_small"] * mult, p_mult=prior["p_mult"], lr_multiplier=mult)


def score(r):
    return r["score_val"] if r["finite"] and math.isfinite(r["score_val"]) else float("inf")


def best_of(rs):
    """(model, task) -> the validation-selected record."""
    d = {}
    for r in rs:
        k = (r["model"], r["task"])
        if k not in d or score(r) < score(d[k]):
            d[k] = r
    return d


def tune_jobs(tasks, lrs):
    return [dict(model=m, task=t, **candidate(lrs[f"{m}/{t}"], x)) for t in tasks for m in MODELS for x in TUNE_MULT]


TUNE_DEFAULTS = dict(H=H_PRIMARY, seed=0, m_phys=512, accum=1, updates=2048, eval_every=128)


def extension_jobs(out, base_stage, lrs):
    """One further geometric candidate in the winning direction when the best trial is at an edge; at most four trials per cell."""
    jobs = []
    for (m, t), r in best_of(load(out, base_stage)).items():
        x = r["lr_multiplier"]
        if x == TUNE_MULT[0]:
            jobs.append(dict(model=m, task=t, **candidate(lrs[f"{m}/{t}"], 1 / 9)))
        elif x == TUNE_MULT[-1]:
            jobs.append(dict(model=m, task=t, **candidate(lrs[f"{m}/{t}"], 9.0)))
    return jobs


def selected(out, *stages):
    s = best_of([r for st in stages for r in load(out, st)])
    return {k: dict(lr_w=r["lr_w"], lr_small=r["lr_small"], p_mult=r["p_mult"], lr_multiplier=r["lr_multiplier"], score_val=score(r)) for k, r in s.items()}


def m_args(d):
    return dict(lr_w=d["lr_w"], lr_small=d["lr_small"], p_mult=d["p_mult"], lr_multiplier=d["lr_multiplier"])


def manifest(stage, out: Path, lrs):
    if stage == "A_tune":
        return dict(defaults=dict(stage=stage, **TUNE_DEFAULTS), jobs=tune_jobs(GLOBAL_TASKS, lrs))
    if stage == "A_ext":
        return dict(defaults=dict(stage=stage, **TUNE_DEFAULTS), jobs=extension_jobs(out, "A_tune", lrs))
    if stage == "C_tune":
        return dict(defaults=dict(stage=stage, **TUNE_DEFAULTS), jobs=tune_jobs(("local",), lrs))
    if stage == "C_ext":
        return dict(defaults=dict(stage=stage, **TUNE_DEFAULTS), jobs=extension_jobs(out, "C_tune", lrs))
    if stage == "B":
        sel = selected(out, "A_tune", "A_ext")
        jobs = [dict(model=m, task=t, seed=s, **m_args(sel[(m, t)])) for s in SEEDS for t in GLOBAL_TASKS for m in MODELS]
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, m_phys=512, accum=1, updates=16_384, eval_every=256, grad_every=1024, full_every=2048, keep_final_full=1,
                                  milestones=[1024, 2048, 4096, 8192, 12288, 16384]), jobs=jobs)
    if stage == "C":
        sel = selected(out, "C_tune", "C_ext")
        jobs = [dict(model=m, task="local", seed=s, **m_args(sel[(m, "local")])) for s in SEEDS for m in MODELS]
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, m_phys=512, accum=1, updates=8192, eval_every=256, grad_every=1024, full_every=2048,
                                  milestones=[1024, 2048, 4096, 6144, 8192]), jobs=jobs)
    if stage == "D_tune":
        sel = selected(out, "A_tune", "A_ext")
        jobs = [dict(model=m, task=t, lr_w=sel[(m, t)]["lr_w"] * x, lr_small=sel[(m, t)]["lr_small"] * x, p_mult=sel[(m, t)]["p_mult"], lr_multiplier=x)
                for t in GLOBAL_TASKS for m in MODELS for x in LARGE_MULT]
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, seed=0, m_phys=8192, accum=1, updates=256, eval_every=16), jobs=jobs)
    if stage in ("D_direct", "D_accum"):
        sel = selected(out, "D_tune")
        mode = dict(m_phys=8192, accum=1) if stage == "D_direct" else dict(m_phys=512, accum=16)
        jobs = [dict(model=m, task=t, seed=s, **m_args(sel[(m, t)])) for s in SEEDS for t in GLOBAL_TASKS for m in MODELS]
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, updates=1024, eval_every=64, grad_every=64, full_every=128, keep_final_full=1, milestones=[64, 128, 256, 512, 768, 1024], **mode), jobs=jobs)
    if stage == "anchor":
        sel = selected(out, "A_tune", "A_ext")
        jobs = [dict(model=m, task=t, seed=0, lr_w=lrs[f"{m}/{t}"]["lr_w"], lr_small=lrs[f"{m}/{t}"]["lr_small"], p_mult=lrs[f"{m}/{t}"]["p_mult"], lr_multiplier=1.0)
                for t in GLOBAL_TASKS for m in ("A0", "A3mean") if sel[(m, t)]["lr_multiplier"] != 1.0]
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, m_phys=512, accum=1, updates=8192, eval_every=256, full_every=2048, milestones=[4096, 8192]), jobs=jobs)
    if stage == "secondary":
        sel = selected(out, "A_tune", "A_ext")
        jobs = [dict(model=m, task=t, seed=s, **m_args(sel[(m, t)])) for s in SEEDS for t in GLOBAL_TASKS for m in ("A0", "A3mean")]
        return dict(defaults=dict(stage=stage, H=H_SECONDARY, m_phys=512, accum=1, updates=8192, eval_every=256, full_every=2048, milestones=[2048, 4096, 8192]), jobs=jobs)
    if stage == "B_ext":
        jobs = []
        for r in load(out, "B"):
            name = "_".join(f"{k}{v}" for k, v in dict(H=r["H"], model=r["model"], task=r["task"], seed=r["seed"], m_phys=512, accum=1, lr_w=r["lr_w"], lr_small=r["lr_small"], p_mult=r["p_mult"]).items())
            jobs.append(dict(model=r["model"], task=r["task"], seed=r["seed"], lr_w=r["lr_w"], lr_small=r["lr_small"], p_mult=r["p_mult"], lr_multiplier=r["lr_multiplier"],
                             init_from=f"{{CKPT}}/B/{name}/full_u{16384:07d}.pt"))
        return dict(defaults=dict(stage=stage, H=H_PRIMARY, m_phys=512, accum=1, updates=32_768, eval_every=256, grad_every=1024, full_every=2048, milestones=[20480, 24576, 32768]), jobs=jobs)
    raise ValueError(stage)


def extension_gate(out: Path, threshold=0.02):
    """The declared validation-only gate of the 32,768-update extension."""
    v = {}
    for r in load(out, "B"):
        c = {e["update"]: e["val"] for e in r["curve"]}
        v.setdefault((r["model"], r["task"]), []).append((c[12288], c[16384]))
    rows = {f"{m}/{t}": 1 - sum(b for _, b in x) / sum(a for a, _ in x) for (m, t), x in v.items()}
    return dict(relative_validation_drop_12288_to_16384=rows, threshold=threshold, still_improving=any(d >= threshold for d in rows.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--out", default="")
    ap.add_argument("--lrs", default="")
    ap.add_argument("--raw", default="")
    ap.add_argument("--write", required=True)
    ap.add_argument("--ckpt-root", default="")
    a = ap.parse_args()
    if a.stage == "recover":
        print(json.dumps(recover(a.raw, a.write), indent=1))
        return
    if a.stage == "extension_gate":
        g = extension_gate(Path(a.out))
        Path(a.write).write_text(json.dumps(g, indent=1))
        print(json.dumps(g, indent=1))
        return
    lrs = json.loads(Path(a.lrs).read_text())
    m = manifest(a.stage, Path(a.out), lrs)
    text = json.dumps(m, indent=1).replace("{CKPT}", a.ckpt_root)
    Path(a.write).parent.mkdir(parents=True, exist_ok=True)
    Path(a.write).write_text(text)
    print(a.stage, len(m["jobs"]), "jobs")


if __name__ == "__main__":
    main()
