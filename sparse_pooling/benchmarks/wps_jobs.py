"""Job plans and runner of the width-preserving sparsity study (results/WIDTH_PRESERVING_SPARSITY.md).

Every stage writes its plan to $GRANULE_RESULTS_ROOT/<out>/plan.json BEFORE any job runs (pre-declaration;
the case manifest goes to manifest.json); each job runs in a fresh process with explicit status
(measured | failed | oom | timeout | rejected-as-infeasible).

    python wps_jobs.py screen     --out screen       # C256 seed 0: every feasible mask x formats, dense anchors
    python wps_jobs.py inference  --out inference    # forward-only timings of the compact formats
    python wps_jobs.py c2048      --out c2048        # C2048 H2A / H2 / U16 at rho 0.05 + dense control
    python wps_jobs.py ablation   --out ablation     # hyb16 with the backbone compiled out (attribution)
    python wps_jobs.py reject     --out reject       # the explicit infeasibility checks
    python wps_jobs.py confirm    --out confirm --selection confirm_selection.json
    python wps_jobs.py winners    --out winners --selection winners_selection.json
    python wps_jobs.py plan       --out X --plan-file plan.json   # any saved plan

Timing protocol (bench_supergroups.py, unchanged harness + --canonical-start / --identity): fresh process,
two warm-up steps per variant, then the PRE-warm-up canonical state restored (values, zero moments, step 0,
rounding counter 0, zero main_grad) and captured; rounds x reps x iters restored repetitions (screen
2 x 2 x 2 = 8 samples, confirmation 4 x 2 x 3 = 24); L1AdamW (lr 1e-4, BF16 moments, stochastic rounding,
weight decay 0, no L1); inputs seed 23; the layer values come from the actual alpha = 0 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import common  # noqa: F401  (implementation under test on sys.path: the manifest uses width_masks)
import wps_cases as WC
from k64_campaign import HERE, classify, done_tags, record, results_root

SCREEN_ROUNDS = (2, 2, 2)
CONFIRM_ROUNDS = (4, 2, 3)
ORDER_SEED = 20261024
ABLATION_ENV = {"SPARSE_POOL_NVCC_EXTRA": "-DSP_ABLATE_KW_BB=1"}


def timing_job(tag, C, fmt, policy=None, rho=None, score="magnitude", seed=0, rounds=SCREEN_ROUNDS,
               inference=False, env=None, tile=128):
    f = dict(checkpoint=WC.ckpt(C, seed), format=fmt, tile=tile)
    if policy is not None:
        f.update(policy=policy, density=rho, score=score, seed=WC.RANDOM_SEED if score == "random" else 0)
    return dict(kind="timing", tag=tag, C=C, graph_seed=seed, factory=f, inference=inference, rounds=list(rounds),
                env=env or {}, policy=policy or "dense", rho=rho if policy else 1.0, score=score, format=fmt)


def mask_jobs(policy, rho, C=256, score="magnitude", seed=0, rounds=SCREEN_ROUNDS, fmts=None):
    out = []
    for fmt in fmts or WC.formats(policy, rho):
        cid = WC.case_id(policy, rho, C, score, seed)
        out.append(timing_job(f"{cid}_{fmt}", C, fmt, policy, rho, score, seed, rounds))
    return out


def with_anchors(jobs, C=256, seed=0, rounds=SCREEN_ROUNDS, order_seed=ORDER_SEED, tag="dense"):
    """Seeded shuffle; the unpruned dense control (same checkpoint) first, in the middle and last."""
    rng = random.Random(order_seed)
    body = list(jobs)
    rng.shuffle(body)
    mid = len(body) // 2
    a = [timing_job(f"{tag}_C{C}_g{seed}_anchor{i}", C, "dense", seed=seed, rounds=rounds) for i in range(3)]
    return [a[0]] + body[:mid] + [a[1]] + body[mid:] + [a[2]]


def screen_plan():
    jobs = []
    for p, d in WC.PRIMARY:
        jobs += mask_jobs(p, d)
    for p, d in WC.RANDOM_CONTROLS:
        jobs += mask_jobs(p, d, score="random")
    return dict(stage="screen", C=256, graph_seed=0, rounds=list(SCREEN_ROUNDS), jobs=with_anchors(jobs),
                note="identical-mask controls: wide vs dense_masked (vs tile, lowered) of the same mask; policy "
                     "comparisons: same checkpoint / budget / data / optimizer, different supports")


def inference_plan():
    jobs = [timing_job("dense_C256_inference", 256, "dense", inference=True)]
    for p, d in WC.PRIMARY:
        for fmt in [f for f in WC.formats(p, d) if f != "dense_masked"]:
            jobs.append(timing_job(f"{WC.case_id(p, d)}_{fmt}_inference", 256, fmt, p, d, inference=True))
    return dict(stage="inference", jobs=jobs, note="forward only, no_grad (compressed-checkpoint inference)")


def c2048_plan():
    jobs = []
    for p, d in WC.C2048_TEST:
        jobs += mask_jobs(p, d, C=2048, fmts=[f for f in WC.formats(p, d) if f != "lowered"])
    return dict(stage="c2048", jobs=with_anchors(jobs, C=2048, order_seed=ORDER_SEED + 2))


def ablation_plan():
    jobs = []
    for p, d in (("H2", 0.05), ("H2", 0.10), ("H2A", 0.10)):
        cid = WC.case_id(p, d)
        jobs.append(timing_job(f"{cid}_wide_ablated_backbone", 256, "wide", p, d, env=ABLATION_ENV))
        jobs.append(timing_job(f"{cid}_wide", 256, "wide", p, d))
    return dict(stage="ablation", jobs=jobs, env=ABLATION_ENV,
                note="ATTRIBUTION ONLY: the ablated build removes every backbone load / product / gradient "
                     "(deliberately wrong results; build kind 'ablation'); all row groups still run")


def reject_plan():
    return dict(stage="reject", jobs=[dict(kind="reject", tag=f"{WC.case_id(p, d, C)}_reject", policy=p, rho=d, C=C)
                                      for p, d, C in WC.INFEASIBLE_CHECKS])


def confirm_plan(sel):
    """sel: dict(comparisons=[dict(policy, rho, score, formats)], seeds, C, criteria)."""
    jobs = []
    for s in sel.get("seeds", list(WC.GRAPH_SEEDS)):
        body = []
        for c in sel["comparisons"]:
            body += mask_jobs(c["policy"], c["rho"], C=sel.get("C", 256), score=c.get("score", "magnitude"), seed=s,
                              rounds=CONFIRM_ROUNDS, fmts=c.get("formats"))
        jobs += with_anchors(body, C=sel.get("C", 256), seed=s, rounds=CONFIRM_ROUNDS, order_seed=ORDER_SEED + 10 + s)
    return dict(stage="confirm", selection=sel, rounds=list(CONFIRM_ROUNDS), jobs=jobs)


def winners_plan(sel):
    """Selected winners at other C (graph seed 0): each policy's feasibility is computed BEFORE running; an
    infeasible (policy, rho, C) becomes an explicit rejection job instead of a timing job."""
    from sparse_pooling.experimental import width_masks as WM
    from sparse_pooling.experimental.kgen_layout import KLayout

    jobs = []
    for C in sel.get("C", [64, 2048]):
        L = KLayout(WC.D, WC.H, WC.G, WC.K, C)
        body, rej = [], []
        for c in sel["comparisons"]:
            if not WM.feasibility(WM.POLICIES[c["policy"]], c["rho"], L)["feasible"]:
                rej.append(dict(kind="reject", tag=f"{WC.case_id(c['policy'], c['rho'], C)}_reject", policy=c["policy"],
                                rho=c["rho"], C=C))
                continue
            body += mask_jobs(c["policy"], c["rho"], C=C, fmts=c.get("formats", WC.formats(c["policy"], c["rho"])))
        jobs += rej + with_anchors(body, C=C, order_seed=ORDER_SEED + 20 + C)
    return dict(stage="winners", selection=sel, jobs=jobs)


SUPPORT_DIR = "/home/evga/Projects/pamachine/wps_ckpt"  # fine-tuned supports (outside git)
THRESHOLD_POLICIES = {("A8", 0.1875), ("A8", 0.25), ("H2A", 0.10), ("P16", 0.25), ("U16", 0.25), ("H2", 0.10)}
SAVE_SUPPORT = {("A8", 0.1875), ("A8", 0.25), ("A16", 0.375), ("H2A", 0.10), ("H2", 0.10), ("P16", 0.25),
                ("U16", 0.25), ("U16", 0.1875), ("U16", 0.10), ("P8", 0.125)}
INTERACTION = [("U16", 0.25), ("P16", 0.25), ("U8", 0.25), ("P8", 0.125), ("A8", 0.1875), ("A8", 0.25), ("A16", 0.375),
               ("H2", 0.10), ("H2A", 0.10), ("U16", 0.10), ("U16", 0.50), ("P16", 0.50)]


def coverage_plan():
    jobs = [dict(kind="coverage", tag="cov_dense_C256", script="wps_coverage.py", args=["--C", "256"])]
    for (p, d), score in [((p, d), "magnitude") for p, d in WC.PRIMARY] + [((p, d), "random") for p, d in
                                                                           WC.RANDOM_CONTROLS]:
        args = ["--C", "256", "--checkpoint", WC.ckpt(256), "--policy", p, "--rho", str(d), "--score", score,
                "--seed", str(WC.RANDOM_SEED if score == "random" else 0)]
        if (p, d) in THRESHOLD_POLICIES and score == "magnitude":
            args += ["--thresholds", "0.001,0.01"]
        jobs.append(dict(kind="coverage", tag=f"cov_{WC.case_id(p, d, 256, score)}", script="wps_coverage.py", args=args))
    jobs.append(dict(kind="coverage", tag="cov_dense_C2048", script="wps_coverage.py", args=["--C", "2048"]))
    for p, d in WC.C2048_TEST:
        jobs.append(dict(kind="coverage", tag=f"cov_{WC.case_id(p, d, 2048)}", script="wps_coverage.py",
                         args=["--C", "2048", "--checkpoint", WC.ckpt(2048), "--policy", p, "--rho", str(d)]))
    return dict(stage="coverage", jobs=jobs)


def learn_plan():
    jobs = [dict(kind="learn", tag="ft_dense_C256", script="wps_learn.py", args=["--policy", "dense"])]
    for (p, d), score in [((p, d), "magnitude") for p, d in WC.PRIMARY] + [((p, d), "random") for p, d in
                                                                           WC.RANDOM_CONTROLS]:
        args = ["--policy", p, "--rho", str(d), "--score", score, "--seed", str(WC.RANDOM_SEED if score == "random" else 0)]
        if (p, d) in SAVE_SUPPORT and score == "magnitude":
            args += ["--save-support", SUPPORT_DIR]
        jobs.append(dict(kind="learn", tag=f"ft_{WC.case_id(p, d, 256, score)}", script="wps_learn.py", args=args))
    return dict(stage="finetune", jobs=jobs, support_dir=SUPPORT_DIR)


def coverage_ft_plan():
    jobs = []
    for p, d in sorted(SAVE_SUPPORT):
        f = f"{SUPPORT_DIR}/{WC.case_id(p, d)}_finetuned.pt"
        jobs.append(dict(kind="coverage", tag=f"covft_{WC.case_id(p, d)}", script="wps_coverage.py",
                         args=["--C", "256", "--checkpoint", WC.ckpt(256), "--mask-file", f, "--policy", p, "--rho",
                               str(d), "--thresholds", "0.001,0.01"]))
    return dict(stage="coverage_finetuned", jobs=jobs)


def interaction_plan(steps=1000):
    jobs = [dict(kind="interaction", tag="int_dense_C256", script="wps_interaction.py",
                 args=["--policy", "dense", "--steps", str(steps)])]
    for p, d in INTERACTION:
        jobs.append(dict(kind="interaction", tag=f"int_{WC.case_id(p, d)}", script="wps_interaction.py",
                         args=["--policy", p, "--rho", str(d), "--steps", str(steps)]))
    jobs.append(dict(kind="interaction", tag="int_P16_r0.25_C256_rnd", script="wps_interaction.py",
                     args=["--policy", "P16", "--rho", "0.25", "--score", "random", "--seed", str(WC.RANDOM_SEED),
                           "--steps", str(steps)]))
    return dict(stage="interaction", jobs=jobs)


REPEATS = [("dense", None), ("U16", 0.25), ("A8", 0.25), ("P16", 0.25), ("U16", 0.10), ("H2A", 0.10), ("H2", 0.10)]


def finetune_repeat_plan():
    """Run-to-run noise of the fine-tuning diagnostic: the same jobs again (same inputs / seeds; only FP32
    atomic summation order differs between runs)."""
    jobs = []
    for p, d in REPEATS:
        args = ["--policy", p] + ([] if p == "dense" else ["--rho", str(d)])
        tag = "ft_dense_C256_rep" if p == "dense" else f"ft_{WC.case_id(p, d)}_rep"
        jobs.append(dict(kind="learn", tag=tag, script="wps_learn.py", args=args))
    return dict(stage="finetune_repeat", jobs=jobs)


def interaction_capacity_plan(steps=1000, terms=16):
    """Capacity stress of the interaction diagnostic: 16 independent terms per record (same policies)."""
    jobs = [dict(kind="interaction", tag=f"intT{terms}_dense_C256", script="wps_interaction.py",
                 args=["--policy", "dense", "--steps", str(steps), "--terms", str(terms)])]
    for p, d in INTERACTION:
        jobs.append(dict(kind="interaction", tag=f"intT{terms}_{WC.case_id(p, d)}", script="wps_interaction.py",
                         args=["--policy", p, "--rho", str(d), "--steps", str(steps), "--terms", str(terms)]))
    return dict(stage="interaction_capacity", jobs=jobs)


def regen_plan():
    """Regeneration check of the reused C256 graph-seed-0 alpha = 0 checkpoint (k64_learn.py, same code path):
    hashes compared by the report; byte identity is NOT expected (FP32 atomics)."""
    return dict(stage="regen", jobs=[dict(kind="regen", tag="regen_C256_a0_g0", script="k64_learn.py",
                                          args=["--C", "256", "--treatment", "prox", "--alpha", "0", "--ckpt-dir",
                                                "/home/evga/Projects/pamachine/wps_ckpt/regen"])])


def refresh_cost_plan():
    jobs = [dict(kind="refresh", tag=f"cost_{WC.case_id(p, d)}", script="wps_refresh_cost.py",
                 args=["--policy", p, "--rho", str(d)]) for p, d in WC.PRIMARY]
    return dict(stage="refresh_cost", jobs=jobs)


def refresh_plan(sel):
    """sel: dict(policies=[[policy, rho], ...], periods=[0, 16, 128], steps=256, frac=0.1, criteria=...)."""
    jobs = []
    for p, d in sel["policies"]:
        for T in sel.get("periods", [0, 16, 128]):
            jobs.append(dict(kind="refresh", tag=f"refresh_{WC.case_id(p, d)}_T{T}", script="wps_refresh.py",
                             args=["--policy", p, "--rho", str(d), "--period", str(T), "--steps",
                                   str(sel.get("steps", 256)), "--frac", str(sel.get("frac", 0.1))]))
    return dict(stage="refresh", selection=sel, jobs=jobs)


def cmd_of(job):
    if job["kind"] == "timing":
        r, p, i = job["rounds"]
        a = [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(WC.H), "--C", str(job["C"]),
             "--S", str(WC.S), "--R", str(WC.R), "--graph-seed", str(job["graph_seed"]), "--init-seed",
             str(WC.INIT_SEED), "--input-seed", str(WC.INPUT_SEED), "--rounds", str(r), "--reps", str(p), "--iters",
             str(i), "--kcsr", "--canonical-start", "--identity", "--layer-factory", "k64_factory:wide",
             "--factory-args", json.dumps(job["factory"]), "--opt-factory", "k64_opt:l1", "--opt-args",
             json.dumps(dict(l1="none")), "--out", job["_rel"]]
        return a + (["--inference"] if job.get("inference") else [])
    if job["kind"] == "reject":
        return [sys.executable, str(HERE / "wps_reject.py"), "--policy", job["policy"], "--rho", str(job["rho"]),
                "--C", str(job["C"]), "--out", str(job["_out"])]
    if job["kind"] in ("coverage", "learn", "interaction", "regen", "refresh"):
        return [sys.executable, str(HERE / job["script"])] + list(job["args"]) + ["--out", str(job["_out"])]
    raise SystemExit(job["kind"])


def run_env(cmd, log_path: Path, timeout: int, env: dict):
    t0 = time.time()
    e = dict(os.environ, **env)
    with open(log_path, "w") as fh:
        try:
            rc = subprocess.run(["timeout", str(timeout)] + cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                env=e).returncode
        except Exception as ex:  # noqa: BLE001
            fh.write(f"driver exception {ex!r}\n")
            rc = -1
    return rc, time.time() - t0


def run_plan(plan, out, timeout, resume):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "plan.json"
    if not pf.exists() or not resume:
        pf.write_text(json.dumps(plan, indent=1))
    mf = d / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps(WC.manifest(), indent=1))
    status = d / "status.jsonl"
    skip = done_tags(status) if resume else set()
    for job in plan["jobs"]:
        tag = job["tag"]
        if tag in skip:
            continue
        job = dict(job, _out=d / f"{tag}.json", _rel=f"{out}/{tag}")
        rc, wall = run_env(cmd_of(job), d / f"{tag}.log", timeout, job.get("env", {}))
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        if job["kind"] == "reject" and rc == 0 and "REJECTED" in log:
            st = "rejected-as-infeasible (expected)"
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, kind=job["kind"], status=st, exit=rc, wall_s=wall,
                            json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:300],
                            job={k: v for k, v in job.items() if not k.startswith("_")},
                            log_tail=log.splitlines()[-30:] if st != "measured" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["screen", "inference", "c2048", "ablation", "reject", "confirm", "winners", "plan",
                                      "coverage", "finetune", "coverage_ft", "interaction", "regen", "refresh_cost",
                                      "refresh", "interaction_capacity", "finetune_repeat"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--plan-file", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.stage == "plan":
        plan = json.loads(Path(a.plan_file).read_text())
    elif a.stage in ("confirm", "winners", "refresh"):
        sel = json.loads(Path(a.selection).read_text())
        plan = dict(confirm=confirm_plan, winners=winners_plan, refresh=refresh_plan)[a.stage](sel)
    else:
        plan = dict(screen=screen_plan, inference=inference_plan, c2048=c2048_plan, ablation=ablation_plan,
                    reject=reject_plan, coverage=coverage_plan, finetune=learn_plan, coverage_ft=coverage_ft_plan,
                    interaction=interaction_plan, regen=regen_plan, refresh_cost=refresh_cost_plan,
                    interaction_capacity=interaction_capacity_plan, finetune_repeat=finetune_repeat_plan)[a.stage]()
    run_plan(plan, a.out, a.timeout, a.resume)


if __name__ == "__main__":
    main()
