"""Job plans and runner of the K64 study's L1 / sparse stages (learning diagnostic, sparse timing, maintenance
costs, mask coverage).  A plan is written to $GRANULE_RESULTS_ROOT/<out>/plan.json BEFORE any job runs
(pre-declaration); every job runs in a fresh process with explicit status (measured | failed | oom | timeout).

    python k64_jobs.py learn   --out learn   --ckpt-dir DIR
    python k64_jobs.py sparse  --out sparse_c256 --ckpt-dir DIR --C 256
    python k64_jobs.py anchors --out sparse_anchors --ckpt-dir DIR --densities 0.1,0.05
    python k64_jobs.py refresh --out refresh --ckpt-dir DIR --C 256
    python k64_jobs.py maskcov --out maskcov --ckpt-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import k64_cases as KC
from k64_campaign import HERE, classify, done_tags, record, results_root, run

ANCHOR_C = (64, 256, 2048)
ALPHAS = (0.0, 0.1, 1.0, 10.0)
# predeclared escalation (only if prox alpha = 10 leaves > 10 % of the valid weights nonzero on an anchor):
# two additional strengths, prox only, all three anchors
ESCALATION_ALPHAS = (30.0, 100.0)
DENSITIES = (0.5, 0.25, 0.1, 0.05)  # target sparsity 50 / 75 / 90 / 95 %
SCREEN_ROUNDS = (2, 2, 2)


def ckpt(ckdir, C, treatment, alpha, step=128, R=256):
    return str(Path(ckdir) / f"C{C}_R{R}_{treatment}_a{alpha:g}_step{step}.pt")


def learn_plan(ckdir, escalate=False):
    jobs = []
    alphas = ESCALATION_ALPHAS if escalate else ALPHAS
    for C in ANCHOR_C:
        for al in alphas:
            jobs.append(dict(kind="learn", tag=f"learn_C{C}_prox_a{al:g}", C=C, treatment="prox", alpha=al))
    if not escalate:
        for al in ALPHAS[1:]:
            jobs.append(dict(kind="learn", tag=f"learn_C256_coupled_a{al:g}", C=256, treatment="coupled", alpha=al))
    return dict(stage="learn", escalate=escalate, ckpt_dir=str(ckdir), jobs=jobs,
                note="M_train = 512 labelled diagnostic; checkpoints after the regularized phase (step 128)")


def _timing(tag, C, factory, opt=None, inference=False, rounds=SCREEN_ROUNDS):
    return dict(kind="timing", tag=tag, C=C, factory=factory, opt=opt or dict(l1="none"), inference=inference,
                rounds=list(rounds))


def sparse_plan(ckdir, C, densities=DENSITIES, learned_alphas=(0.1, 1.0, 10.0), with_random=True, formats_el=("dense_masked", "csr"),
                tile_formats=("dense_masked", "tile", "csr"), inference=True, rounds=SCREEN_ROUNDS):
    """Frozen-mask sparse timing of one anchor C (M = 8192, full H): every mask on the dense masked reference and
    on the sparse formats that represent it, plus inference of the compact formats and the dense+L1 controls."""
    base = ckpt(ckdir, C, "prox", 0.0)
    jobs = []
    # (b) dense + L1 / proximal training: the dense kernels with the L1 optimizer branches (values irrelevant)
    for l1 in ("none", "coupled", "prox"):
        jobs.append(_timing(f"C{C}_dense_l1_{l1}", C, dict(checkpoint=base, format="dense_masked", mask="none"),
                            dict(l1=l1, mu=1e-6), rounds=rounds))
    for d in densities:
        for kind, fmts in (("magnitude", formats_el), ("tile", tile_formats)) + (
                (("random", formats_el), ("random_tile", ("tile",))) if with_random else ()):
            for fmt in fmts:
                f = dict(checkpoint=base, format=fmt, mask=kind, density=d, mask_seed=0)
                jobs.append(_timing(f"C{C}_{kind}_d{d:g}_{fmt}", C, f, rounds=rounds))
                if inference and fmt in ("tile", "csr"):
                    jobs.append(_timing(f"C{C}_{kind}_d{d:g}_{fmt}_inference", C, f, inference=True, rounds=rounds))
    for al in learned_alphas:  # learned (proximal) supports
        f = dict(checkpoint=ckpt(ckdir, C, "prox", al), format=None, mask="checkpoint")
        for fmt in formats_el:
            jobs.append(_timing(f"C{C}_learned_a{al:g}_{fmt}", C, dict(f, format=fmt), rounds=rounds))
        if inference:
            jobs.append(_timing(f"C{C}_learned_a{al:g}_csr_inference", C, dict(f, format="csr"), inference=True,
                                rounds=rounds))
    return dict(stage="sparse", C=C, densities=list(densities), jobs=jobs)


def refresh_plan(ckdir, C, densities=DENSITIES):
    base = ckpt(ckdir, C, "prox", 0.0)
    jobs = []
    for d in densities:
        jobs.append(dict(kind="refresh", tag=f"refresh_C{C}_tile_d{d:g}", C=C, checkpoint=base, format="tile",
                         mask="tile", density=d))
        jobs.append(dict(kind="refresh", tag=f"refresh_C{C}_csr_d{d:g}", C=C, checkpoint=base, format="csr",
                         mask="magnitude", density=d))
    return dict(stage="refresh", C=C, jobs=jobs)


def maskcov_plan(ckdir, densities=DENSITIES, learned_alphas=(0.1, 1.0, 10.0), extra_alphas=()):
    """Retained-support coverage (exact pairs / triples, fixed-size sampled quadruples; 50M at >= 90 % sparsity),
    layer-wide and per output, of the masks that were timed (rebuilt deterministically)."""
    jobs = []
    for C in ANCHOR_C:
        base = ckpt(ckdir, C, "prox", 0.0)
        for al in tuple(learned_alphas) + tuple(extra_alphas):
            jobs.append(dict(kind="maskcov", tag=f"cov_C{C}_learned_a{al:g}", C=C, checkpoint=ckpt(ckdir, C, "prox", al),
                             source="mask", label=f"learned prox alpha {al:g} (exact nonzero support)",
                             sparse=al >= 10))
        for d in densities:
            kinds = ("magnitude", "tile") + (("random", "random_tile") if C == 256 else ())
            for kind in kinds:
                jobs.append(dict(kind="maskcov", tag=f"cov_C{C}_{kind}_d{d:g}", C=C, checkpoint=base, source=kind,
                                 density=d, mask_seed=0, label=f"{kind} density {d:g}", sparse=d <= 0.1))
        if C == 256:
            jobs.append(dict(kind="maskcov", tag="cov_C256_dense_a0_threshold0.01", C=C, checkpoint=base,
                             source="threshold", threshold=0.01, label="alpha 0 checkpoint, |w| > 0.01 (thresholded)"))
    return dict(stage="maskcov", jobs=jobs)


def cmd_of(job, ckdir):
    if job["kind"] == "learn":
        a = [sys.executable, str(HERE / "k64_learn.py"), "--C", str(job["C"]), "--treatment", job["treatment"],
             "--alpha", str(job["alpha"]), "--ckpt-dir", str(ckdir), "--out", str(job["_out"])]
        if job.get("reg_steps"):
            a += ["--reg-steps", str(job["reg_steps"])]
        return a
    if job["kind"] == "timing":
        case = KC.case(KC.case_id(job["C"], 64, 256))
        r, p, i = job["rounds"]
        f = dict(job["factory"])
        if f.get("mask") == "none":  # dense layer from the checkpoint, no mask
            f = dict(f, mask="checkpoint_dense")
        a = [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(case["H"]), "--C", str(case["C"]),
             "--S", "64", "--R", "256", "--graph-seed", "0", "--init-seed", str(KC.INIT_SEED), "--input-seed",
             str(KC.INPUT_SEED), "--rounds", str(r), "--reps", str(p), "--iters", str(i), "--kcsr",
             "--layer-factory", "k64_factory:sparse", "--factory-args", json.dumps(f),
             "--opt-factory", "k64_opt:l1", "--opt-args", json.dumps(job["opt"]), "--out", job["_rel"]]
        return a + (["--inference"] if job.get("inference") else [])
    if job["kind"] == "refresh":
        return [sys.executable, str(HERE / "k64_refresh.py"), "--checkpoint", job["checkpoint"], "--C", str(job["C"]),
                "--format", job["format"], "--mask", job["mask"], "--density", str(job["density"]), "--out",
                str(job["_out"])]
    if job["kind"] == "maskcov":
        a = [sys.executable, str(HERE / "k64_coverage.py"), "--case", KC.case_id(job["C"], 64, 256), "--checkpoint",
             job["checkpoint"], "--mask-source", job["source"], "--kind", job["label"], "--out", str(job["_out"])]
        if job.get("threshold") is not None:
            a += ["--threshold", str(job["threshold"])]
        if job.get("density") is not None:
            a += ["--density", str(job["density"]), "--mask-seed", str(job.get("mask_seed", 0))]
        return a + (["--sparse"] if job.get("sparse") else [])
    raise SystemExit(job["kind"])


def run_plan(plan, out, ckdir, timeout, resume):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "plan.json"
    if not pf.exists() or not resume:
        pf.write_text(json.dumps(plan, indent=1))
    status = d / "status.jsonl"
    skip = done_tags(status) if resume else set()
    for job in plan["jobs"]:
        tag = job["tag"]
        if tag in skip:
            continue
        job = dict(job, _out=d / f"{tag}.json", _rel=f"{out}/{tag}")
        rc, wall = run(cmd_of(job, ckdir), d / f"{tag}.log", timeout)
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, kind=job["kind"], status=st, exit=rc, wall_s=wall,
                            json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:300],
                            job={k: v for k, v in job.items() if not k.startswith("_")},
                            log_tail=log.splitlines()[-30:] if st != "measured" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["learn", "escalate", "sparse", "anchors", "refresh", "maskcov", "plan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--densities", default="")
    ap.add_argument("--plan-file", default="", help="stage plan: run a saved plan JSON")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dens = tuple(float(x) for x in a.densities.split(",")) if a.densities else DENSITIES
    if a.stage == "plan":
        plan = json.loads(Path(a.plan_file).read_text())
    elif a.stage == "learn":
        plan = learn_plan(a.ckpt_dir)
    elif a.stage == "escalate":
        plan = learn_plan(a.ckpt_dir, escalate=True)
    elif a.stage == "sparse":
        plan = sparse_plan(a.ckpt_dir, a.C, dens)
    elif a.stage == "anchors":
        plan = dict(stage="anchors", jobs=[])
        for C in ANCHOR_C:
            if C == 256:
                continue
            p = sparse_plan(a.ckpt_dir, C, dens, learned_alphas=(), with_random=False, formats_el=("dense_masked", "csr"),
                            tile_formats=("dense_masked", "tile"), inference=False)
            plan["jobs"] += [j for j in p["jobs"] if "_l1_" not in j["tag"] or j["tag"].endswith("_none")]
    elif a.stage == "refresh":
        plan = refresh_plan(a.ckpt_dir, a.C, dens)
    elif a.stage == "maskcov":
        plan = maskcov_plan(a.ckpt_dir, dens)
    else:
        raise SystemExit(a.stage)
    run_plan(plan, a.out, a.ckpt_dir, a.timeout, a.resume)


if __name__ == "__main__":
    main()
