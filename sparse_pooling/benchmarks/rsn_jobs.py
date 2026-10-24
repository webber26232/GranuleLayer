"""Job plans and runner of the random-support / normalization / optimization-control study
(results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md).  Same runner discipline as wps_jobs.py: every stage writes its
plan to $GRANULE_RESULTS_ROOT/<out>/plan.json BEFORE any job runs, the case manifest to manifest.json, each job runs
in a fresh process with an explicit status (measured | failed | oom | timeout).

    python rsn_jobs.py refresh       --out refresh        # budget-exact refresh re-tests (+ one legacy reference)
    python rsn_jobs.py coverage      --out coverage       # exact supports of the screen
    python rsn_jobs.py screen        --out screen         # graph seed 0, layout seeds 0/1/2, controls
    python rsn_jobs.py confirm       --out confirm --selection confirm_selection.json
    python rsn_jobs.py norm_first    --out norm_first     # BP-row / H2 0.05 x none / fanin / energy (+ W-and-bias)
    python rsn_jobs.py norm          --out norm --selection norm_selection.json
    python rsn_jobs.py interaction   --out interaction --selection interaction_selection.json
    python rsn_jobs.py plan          --out X --plan-file plan.json

Timing protocol (bench_supergroups.py --canonical-start --identity, unchanged harness): fresh process, warm-up,
restore of the pre-warm-up canonical state, rounds x reps x iters restored repetitions (screen 2 x 2 x 2 = 8,
confirmation 4 x 2 x 3 = 24); L1AdamW lr 1e-4 BF16 moments, stochastic rounding, no L1; inputs seed 23; values of
the actual alpha = 0 checkpoint; the unpruned dense control first / middle / last of every session.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import common  # noqa: F401
import rsn_cases as RC
import wps_cases as WC
import wps_jobs as WJ
from k64_campaign import HERE, classify, done_tags, record, results_root

SCREEN_ROUNDS, CONFIRM_ROUNDS = WJ.SCREEN_ROUNDS, WJ.CONFIRM_ROUNDS
ORDER_SEED = 20261025


def factory_of(spec: str, fmt: str, graph_seed: int) -> dict:
    c = RC.parse(spec)
    f = dict(checkpoint=WC.ckpt(RC.C, graph_seed), format=fmt, tile=128)
    if c["kind"] == "width":
        f.update(policy=c["policy"], density=c["rho"], score="magnitude", seed=0)
    elif c["kind"] == "local":
        f.update(policy=c["policy"], k=c["k"], layout_seed=c["layout_seed"])
    return f


def timing_job(spec, fmt, graph_seed=0, rounds=SCREEN_ROUNDS, tag=None):
    tag = tag or f"{RC.spec_id(spec)}_g{graph_seed}_{fmt}"
    return dict(kind="timing", tag=tag, C=RC.C, graph_seed=graph_seed, factory=factory_of(spec, fmt, graph_seed),
                rounds=list(rounds), env={}, support=spec, format=fmt)


def with_anchors(jobs, graph_seed=0, rounds=SCREEN_ROUNDS, order_seed=ORDER_SEED):
    rng = random.Random(order_seed)
    body = list(jobs)
    rng.shuffle(body)
    mid = len(body) // 2
    a = [timing_job("dense", "dense", graph_seed, rounds, tag=f"dense_g{graph_seed}_anchor{i}") for i in range(3)]
    return [a[0]] + body[:mid] + [a[1]] + body[mid:] + [a[2]]


SCREEN = ([(f"{p}#{ls}", "indexed") for ls in RC.SCREEN_LAYOUT_SEEDS for p in RC.RANDOMIZED] +
          [("FP-mag", "indexed"), ("FP-mag", "wide"), ("FP-mag", "tile"),
           ("FP-balanced", "indexed"), ("FP-balanced", "wide"), ("FP-balanced", "tile"),
           ("U16@0.25", "wide"), ("U16@0.25", "tile"), ("A8@0.25", "wide"), ("A8@0.1875", "wide")])


def screen_plan():
    return dict(stage="screen", graph_seed=0, rounds=list(SCREEN_ROUNDS),
                jobs=with_anchors([timing_job(s, f) for s, f in SCREEN]),
                note="matched-function format controls: FP-mag / FP-balanced on indexed K16 vs index-free 16 x 16 "
                     "(new 'wide' and measured 'tile' kernels); same-implementation support comparison: FP-balanced vs "
                     "RP-* on indexed K16 (same dimensions / work counts / checkpoint, different supports); policy "
                     "comparisons at the same 200M budget: U16 0.25, A8 0.25 (+ optional A8 0.1875), dense control")


def confirm_plan(sel):
    """sel: dict(supports=[[spec, fmt], ...] (randomized specs WITHOUT layout seed: expanded over layout seeds),
    controls=[[spec, fmt], ...], graph_seeds, layout_seeds, criteria)."""
    jobs = []
    for g in sel.get("graph_seeds", list(RC.CONFIRM_GRAPH_SEEDS)):
        body = []
        for spec, fmt in sel["supports"]:
            for ls in sel.get("layout_seeds", list(RC.CONFIRM_LAYOUT_SEEDS)):
                body.append(timing_job(f"{spec}#{ls}", fmt, g, CONFIRM_ROUNDS))
        for spec, fmt in sel["controls"]:
            body.append(timing_job(spec, fmt, g, CONFIRM_ROUNDS))
        jobs += with_anchors(body, g, CONFIRM_ROUNDS, ORDER_SEED + 10 + g)
    return dict(stage="confirm", selection=sel, rounds=list(CONFIRM_ROUNDS), jobs=jobs)


COVERAGE = (["dense", "FP-mag", "FP-balanced"] + [f"{p}#{ls}" for p in RC.RANDOMIZED for ls in RC.SCREEN_LAYOUT_SEEDS]
            + ["U16@0.25", "A8@0.25", "A8@0.1875"])


def coverage_plan(specs=COVERAGE, graph_seed=0):
    return dict(stage="coverage", jobs=[dict(kind="script", tag=f"cov_{RC.spec_id(s)}_g{graph_seed}",
                                             script="rsn_coverage.py",
                                             args=["--support", s, "--graph-seed", str(graph_seed)]) for s in specs])


def refresh_plan():
    rows = [("H2A", 0.10, 0, "exact"), ("H2A", 0.10, 16, "exact"), ("H2A", 0.10, 128, "exact"),
            ("A8", 0.1875, 16, "exact"), ("H2A", 0.10, 16, "legacy")]
    jobs = [dict(kind="script", tag=f"refresh_{WC.case_id(p, d)}_T{T}_{rule}", script="wps_refresh.py",
                 args=["--policy", p, "--rho", str(d), "--period", str(T), "--steps", "256", "--frac", "0.1",
                       "--rule", rule]) for p, d, T, rule in rows]
    return dict(stage="refresh", jobs=jobs,
                note="corrected budget-exact rule on the parent's H2A 0.10 T16 / T128 256-update diagnostic (+ frozen "
                     "control) and one A8 schedule (0.1875, T16); one legacy-rule rerun reproduces the parent's drift")


def learn_job(spec, norm="none", steps=32, data_seed=0, sr_seed=0, scale_bias=False, tag=None):
    tag = tag or (f"learn_{RC.spec_id(spec)}_{norm}{'_wb' if scale_bias else ''}_s{steps}_d{data_seed}")
    args = ["--support", spec, "--norm", norm, "--steps", str(steps), "--data-seed", str(data_seed), "--sr-seed",
            str(sr_seed), "--eval-every", "8" if steps <= 32 else "32"]
    if scale_bias:
        args.append("--scale-bias")
    return dict(kind="script", tag=tag, script="rsn_learn.py", args=args)


def norm_first_plan():
    jobs = [learn_job(s, n) for s in RC.NORM_FIRST for n in ("none", "fanin", "energy")]
    jobs.append(learn_job("H2@0.05", "energy", scale_bias=True))
    jobs.append(learn_job("dense", "none"))
    return dict(stage="norm_first", jobs=jobs, note="all three treatments on BP-row 0.25 and H2 0.05 (32 updates, the "
                "parent protocol), the H2 0.05 W-and-bias energy control and the dense continuation")


def norm_plan(sel):
    """sel: dict(supports=[...], treatments=["none", "energy"], steps=[32, 256], seeds_256=[0, 1, 2])."""
    jobs = []
    for s in sel["supports"]:
        trs = ["none"] if s == "dense" else sel.get("treatments", ["none", "energy"])
        for n in trs:
            for steps in sel.get("steps", [32, 256]):
                seeds = sel.get("seeds_256", [0, 1, 2]) if steps > 32 else [0]
                for d in seeds:
                    jobs.append(learn_job(s, n, steps, d, d))
    return dict(stage="norm", selection=sel, jobs=jobs)


def interaction_job(spec, terms=1, norm="none", oracle="", perturb=0.0, init_seed=17, sr_seed=0, steps=1000,
                    lr=3e-3, save_at="", tag=None, resume=""):
    tag = tag or (f"int_{RC.spec_id(spec)}_T{terms}_{oracle or norm}" + (f"_p{perturb:g}" if perturb else "") +
                  f"_i{init_seed}_s{steps}" + (f"_lr{lr:g}" if lr != 3e-3 else ""))
    args = ["--support", spec, "--terms", str(terms), "--norm", norm, "--init-seed", str(init_seed), "--sr-seed",
            str(sr_seed), "--steps", str(steps), "--lr", str(lr), "--eval-every", "50" if steps <= 1000 else "100"]
    if oracle:
        args += ["--oracle", oracle]
    if perturb:
        args += ["--perturb", str(perturb)]
    if save_at:
        args += ["--save-at", save_at, "--state-dir", "{state_dir}"]
    if resume:
        args += ["--resume", resume]
    return dict(kind="script", tag=tag, script="rsn_interaction.py", args=args)


def interaction_plan(sel):
    """sel: dict(jobs=[dict(spec, terms, norm, oracle, perturb, init_seed, sr_seed, steps, lr, save_at)], ...)."""
    return dict(stage="interaction", selection=sel, jobs=[interaction_job(**j) for j in sel["jobs"]])


def cmd_of(job, d: Path):
    if job["kind"] == "timing":
        r, p, i = job["rounds"]
        return [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(WC.H), "--C", str(job["C"]),
                "--S", str(WC.S), "--R", str(WC.R), "--graph-seed", str(job["graph_seed"]), "--init-seed",
                str(WC.INIT_SEED), "--input-seed", str(WC.INPUT_SEED), "--rounds", str(r), "--reps", str(p),
                "--iters", str(i), "--kcsr", "--canonical-start", "--identity", "--layer-factory", "k64_factory:wide",
                "--factory-args", json.dumps(job["factory"]), "--opt-factory", "k64_opt:l1", "--opt-args",
                json.dumps(dict(l1="none")), "--out", job["_rel"]]
    args = [a.replace("{state_dir}", str(d / "state")) for a in job["args"]]
    extra = ["--floors", str(d / "floors.json")] if job["script"] == "rsn_interaction.py" else []
    return [sys.executable, str(HERE / job["script"])] + args + extra + ["--out", str(job["_out"])]


def run_plan(plan, out, timeout, resume):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "plan.json"
    if not pf.exists() or not resume:
        pf.write_text(json.dumps(plan, indent=1))
    mf = d / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps(RC.manifest(), indent=1))
    status = d / "status.jsonl"
    skip = done_tags(status) if resume else set()
    for job in plan["jobs"]:
        tag = job["tag"]
        if tag in skip:
            continue
        job = dict(job, _out=d / f"{tag}.json", _rel=f"{out}/{tag}")
        rc, wall = WJ.run_env(cmd_of(job, d), d / f"{tag}.log", timeout, job.get("env", {}))
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, kind=job["kind"], status=st, exit=rc, wall_s=wall,
                            json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:300],
                            job={k: v for k, v in job.items() if not k.startswith("_")},
                            log_tail=log.splitlines()[-30:] if st != "measured" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["screen", "confirm", "coverage", "refresh", "norm_first", "norm", "interaction",
                                      "plan"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--plan-file", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.stage == "plan":
        plan = json.loads(Path(a.plan_file).read_text())
    elif a.stage in ("confirm", "norm", "interaction"):
        sel = json.loads(Path(a.selection).read_text())
        plan = dict(confirm=confirm_plan, norm=norm_plan, interaction=interaction_plan)[a.stage](sel)
    else:
        plan = dict(screen=screen_plan, coverage=coverage_plan, refresh=refresh_plan,
                    norm_first=norm_first_plan)[a.stage]()
    run_plan(plan, a.out, a.timeout, a.resume)


if __name__ == "__main__":
    main()
