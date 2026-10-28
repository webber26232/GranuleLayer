"""Plans and runner of the K8 / weighted-support / anchor study (results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md).

Same discipline as rsn_jobs.py: every stage's plan (job list) is BUILT from declared builders plus, where a stage
depends on earlier results, an explicit selection file, written to $GRANULE_RESULTS_ROOT/<out>/plan.json before any
job runs; the case manifest goes to manifest.json; every job runs in a fresh process with an explicit status
(measured | failed | oom | timeout) appended to status.jsonl.

    python kwa_jobs.py <stage> --out DIR [--selection sel.json] [--resume]

Timing jobs: bench_supergroups.py --canonical-start --identity (rsn_jobs.cmd_of), screens 2 x 2 x 2 = 8 restored
samples, confirmations 4 x 2 x 3 = 24, the dense control first / middle / last of every session (one session per
graph seed).  Learning jobs: rsn_learn.py (development validation batches 9_000..9_003; the locked final test only
in the "final" stage).  Interaction jobs: rsn_interaction.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import common  # noqa: F401
import kwa_cases as KW
import rsn_cases as RC
import rsn_jobs as RJ
import wps_cases as WC
import wps_jobs as WJ
from k64_campaign import classify, done_tags, record, results_root

SCREEN_ROUNDS, CONFIRM_ROUNDS = RJ.SCREEN_ROUNDS, RJ.CONFIRM_ROUNDS
ORDER_SEED = 20261027


def factory_of(spec: str, fmt: str, graph_seed: int) -> dict:
    c = RC.parse(spec)
    f = dict(checkpoint=WC.ckpt(RC.C, graph_seed), format=fmt, tile=128)
    if c["kind"] == "width":
        f.update(policy=c["policy"], density=c["rho"], score="magnitude", seed=0)
    elif c["kind"] == "local":
        f.update(policy=c["policy"], k=c["k"], layout_seed=c["layout_seed"], L=c["L"])
    elif c["kind"] == "mixed":
        f.update(policy=c["policy"], layout_seed=c["layout_seed"], L=c["L"])
    return f


def timing_job(spec, fmt, graph_seed=0, rounds=SCREEN_ROUNDS, tag=None):
    tag = tag or f"{RC.spec_id(spec)}_g{graph_seed}_{fmt}"
    return dict(kind="timing", tag=tag, C=RC.C, graph_seed=graph_seed, factory=factory_of(spec, fmt, graph_seed),
                rounds=list(rounds), env={}, support=spec, format=fmt)


def session(pairs, graph_seed, rounds, order_seed):
    """One timing session (graph seed): shuffled body, dense control first / middle / last."""
    rng = random.Random(order_seed)
    body = [timing_job(s, f, graph_seed, rounds) for s, f in pairs]
    rng.shuffle(body)
    mid = len(body) // 2
    a = [timing_job("dense", "dense", graph_seed, rounds, tag=f"dense_g{graph_seed}_anchor{i}") for i in range(3)]
    return [a[0]] + body[:mid] + [a[1]] + body[mid:] + [a[2]]


def coverage_job(spec, graph_seed=0):
    return dict(kind="script", tag=f"cov_{RC.spec_id(spec)}_g{graph_seed}", script="rsn_coverage.py",
                args=["--support", spec, "--graph-seed", str(graph_seed)])


def learn_job(spec, norm="none", steps=32, data_seed=0, sr_seed=None, graph_seed=0, final=False, save=False):
    sr_seed = data_seed if sr_seed is None else sr_seed
    tag = f"learn_{RC.spec_id(spec)}_g{graph_seed}_{norm}_s{steps}_d{data_seed}" + ("_final" if final else "")
    args = ["--support", spec, "--norm", norm, "--steps", str(steps), "--data-seed", str(data_seed), "--sr-seed",
            str(sr_seed), "--graph-seed", str(graph_seed), "--eval-every", "8" if steps <= 32 else "32"]
    if final:
        args.append("--final-test")
    if save:
        args += ["--save-state", "{state_dir}"]
    return dict(kind="script", tag=tag, script="rsn_learn.py", args=args)


def interaction_job(spec, terms=16, init_seed=17, sr_seed=0, steps=1000, lr=3e-3, oracle="", perturb=0.0,
                    save_at="", resume="", final=False, tag=None):
    j = RJ.interaction_job(spec, terms=terms, oracle=oracle, perturb=perturb, init_seed=init_seed, sr_seed=sr_seed,
                           steps=steps, lr=lr, save_at=save_at, resume=resume, tag=tag)
    if final:
        j["args"].append("--final-test")
        j["tag"] += "_final"
    return j


# ------------------------------------------------------------------ stage builders
K8_SPECS = ("BP-row/8", "U8@0.125", "FP-mag/8")


def stage_errata_cov(sel=None):
    """Corrected global-support sharing on the parent's K16 supports (graph 0): the exact coverage must be unchanged
    where the support hashes are."""
    return dict(stage="errata_cov", jobs=[coverage_job(s) for s in ("FP-mag", "BP-row#0", "U16@0.25", "A8@0.25")])


def stage_qcheck(sel=None):
    return dict(stage="qcheck", jobs=[dict(kind="script", tag="qcheck", script="kwa_qcheck.py", args=[])])


def stage_k8_cov(sel=None):
    jobs = []
    for g in KW.GRAPH_SEEDS:
        jobs += [coverage_job(f"BP-row/8#{ls}", g) for ls in KW.LAYOUT_SEEDS]
        jobs += [coverage_job("U8@0.125", g), coverage_job("FP-mag/8", g)]
    return dict(stage="k8_cov", jobs=jobs)


def stage_k8_timing(sel=None):
    jobs = []
    for g in KW.GRAPH_SEEDS:
        pairs = [(f"BP-row/8#{ls}", "indexed") for ls in KW.LAYOUT_SEEDS]
        pairs += [("U8@0.125", "wide"), ("FP-mag/8", "indexed"), ("FP-mag/8", "wide"),
                  ("BP-row#0", "indexed"), ("U16@0.25", "wide"), ("A8@0.25", "wide")]
        jobs += session(pairs, g, CONFIRM_ROUNDS, ORDER_SEED + g)
    return dict(stage="k8_timing", rounds=list(CONFIRM_ROUNDS), jobs=jobs,
                note="BP-row/8 at graph 0/1/2 x layout 0/1 with same-session U8 0.125 wide and FP-mag/8 indexed / wide "
                     "(same-function control) per graph; BP-row/16 indexed, U16 0.25 wide, A8 0.25 wide = higher-budget "
                     "frontier references, not matched controls; dense first / middle / last")


def stage_k8_q32(sel=None):
    return dict(stage="k8_q32", jobs=[learn_job(s, n, 32, 0) for s in K8_SPECS for n in ("none", "energy")])


def stage_k8_q256(sel=None):
    return dict(stage="k8_q256", jobs=[learn_job(s, n, 256, d) for s in K8_SPECS for n in ("none", "energy")
                                       for d in KW.DEV_SEEDS])


def stage_k8_graph(sel):
    """sel: dict(bp_norm=..., u8_norm=...)."""
    jobs = [learn_job(f"BP-row/8#{ls}", sel["bp_norm"], 256, 0, graph_seed=g)
            for g in KW.GRAPH_SEEDS for ls in KW.LAYOUT_SEEDS if (g, ls) != (0, 0)]
    jobs += [learn_job("U8@0.125", sel["u8_norm"], 256, 0, graph_seed=g) for g in (1, 2)]
    return dict(stage="k8_graph", selection=sel, jobs=jobs)


W8_SCREEN = ("WRS/8:4#0", "WRS/8:16#0", "TopK/8")


def stage_w8_screen(sel=None):
    jobs = [coverage_job(s) for s in W8_SCREEN]
    jobs += session([(s, "indexed") for s in W8_SCREEN + ("BP-row/8#0", "FP-mag/8")], 0, SCREEN_ROUNDS, ORDER_SEED + 30)
    jobs += [learn_job(s, n, 32, 0) for s in W8_SCREEN for n in ("none", "energy")]
    return dict(stage="w8_screen", jobs=jobs, note="graph 0 / layout 0; BP-row/8 and FP-mag/8 coverage / 32-update "
                "controls from k8_cov / k8_q32 (same code, same checkpoint); timing: same-session indexed K8")


def stage_w8_dev(sel):
    """sel: dict(advance=[spec, ...] (<= 2))."""
    assert len(sel["advance"]) <= 2
    return dict(stage="w8_dev", selection=sel, jobs=[learn_job(s, n, 256, d) for s in sel["advance"]
                                                     for n in ("none", "energy") for d in KW.DEV_SEEDS])


def stage_w8_confirm(sel):
    """sel: dict(rule=spec without layout seed (e.g. WRS/8:16) or TopK/8, norm, bp_norm, layout_free=bool)."""
    rule = sel["rule"]
    lss = [None] if sel.get("layout_free") else list(KW.LAYOUT_SEEDS)
    specs = [rule if ls is None else f"{rule}#{ls}" for ls in lss]
    jobs = []
    for g in KW.GRAPH_SEEDS:
        pairs = [(s, "indexed") for s in specs] + [(f"BP-row/8#{ls}", "indexed") for ls in KW.LAYOUT_SEEDS]
        pairs += [("U8@0.125", "wide")]
        jobs += session(pairs, g, CONFIRM_ROUNDS, ORDER_SEED + 40 + g)
        jobs += [coverage_job(s, g) for s in specs if not (g == 0 and s in sel.get("covered_g0", []))]
        jobs += [learn_job(s, sel["norm"], 256, 0, graph_seed=g) for s in specs if not (g == 0 and s == specs[0])]
    return dict(stage="w8_confirm", selection=sel, jobs=jobs)


def stage_m25_screen(sel):
    """sel: dict(w16=spec of the selected weighted rule at k16 (e.g. WRS/16:16#0 or TopK/16), mix_w=spec)."""
    new = [sel["w16"], "Mix-BP#0", sel["mix_w"]]
    jobs = [coverage_job(s) for s in new]
    pairs = [(sel["w16"], "indexed"), ("Mix-BP#0", "grouped"), (sel["mix_w"], "grouped"), ("BP-row#0", "indexed"),
             ("A8@0.25", "wide"), ("U16@0.25", "wide")]
    jobs += session(pairs, 0, SCREEN_ROUNDS, ORDER_SEED + 50)
    jobs += [learn_job(s, n, 32, 0) for s in new + ["BP-row#0", "A8@0.25", "U16@0.25"] for n in ("none", "energy")]
    jobs.append(learn_job("dense", "none", 32, 0))
    return dict(stage="m25_screen", selection=sel, jobs=jobs,
                note="200M-slot 2 x 2 (uniform K16 / mixed allocation x unweighted / weighted) with A8 / U16 controls; "
                     "graph 0 / layout 0; quality runs execute the dense masked function")


def stage_m25_dev(sel):
    """sel: dict(advance=[spec, ...] (<= 2), formats={spec: fmt}, controls_256=[[spec, norm], ...])."""
    assert len(sel["advance"]) <= 2
    jobs = [learn_job(s, n, 256, d) for s in sel["advance"] for n in ("none", "energy") for d in KW.DEV_SEEDS]
    pairs = [(s, sel["formats"][s]) for s in sel["advance"]]
    pairs += [("BP-row#0", "indexed"), ("A8@0.25", "wide"), ("U16@0.25", "wide")]
    jobs += session(pairs, 0, CONFIRM_ROUNDS, ORDER_SEED + 60)
    return dict(stage="m25_dev", selection=sel, jobs=jobs)


def stage_m25_confirm(sel):
    """sel: dict(rule=mixed spec without layout seed, fmt, norm)."""
    rule = sel["rule"]
    jobs = []
    for g in KW.GRAPH_SEEDS:
        pairs = [(f"{rule}#{ls}", sel["fmt"]) for ls in KW.LAYOUT_SEEDS]
        pairs += [("BP-row#0", "indexed"), ("A8@0.25", "wide"), ("U16@0.25", "wide")]
        if g:
            jobs += session(pairs, g, CONFIRM_ROUNDS, ORDER_SEED + 70 + g)
        for ls in KW.LAYOUT_SEEDS:
            s = f"{rule}#{ls}"
            if (g, ls) != (0, 0):
                jobs.append(coverage_job(s, g))
                jobs.append(learn_job(s, sel["norm"], 256, 0, graph_seed=g))
    return dict(stage="m25_confirm", selection=sel, jobs=jobs,
                note="graph 1 / 2 timing sessions (graph 0 / layout 0+1 timing = m25_dev's session plus layout 1 here)")


def stage_final(sel):
    """sel: dict(methods=[[spec, norm], ...]) -- the frozen new methods and their budget-matched controls."""
    jobs = [learn_job(s, n, 256, d, final=True, save=True) for s, n in sel["methods"] for d in KW.FINAL_SEEDS]
    return dict(stage="final", selection=sel, jobs=jobs, note="locked final test (batches 19_000..19_007) once after "
                "training; fresh data / SR seeds 3 / 4 / 5; graph 0 / layout 0; final states kept")


def stage_interaction(sel):
    """sel: dict(jobs=[interaction_job kwargs, ...])."""
    return dict(stage="interaction", selection=sel, jobs=[interaction_job(**j) for j in sel["jobs"]])


def stage_timing(sel):
    """Generic timing session(s): sel: dict(sessions=[dict(graph_seed, pairs=[[spec, fmt]], rounds)], note)."""
    jobs = []
    for i, ss in enumerate(sel["sessions"]):
        jobs += session([tuple(p) for p in ss["pairs"]], ss.get("graph_seed", 0), ss.get("rounds", CONFIRM_ROUNDS),
                        ORDER_SEED + 100 + i)
    return dict(stage="timing", selection=sel, jobs=jobs)


def stage_scripts(sel):
    """Generic: sel: dict(jobs=[dict(kind='script', tag, script, args)])."""
    return dict(stage="scripts", selection=sel, jobs=sel["jobs"])


STAGES = dict(errata_cov=stage_errata_cov, qcheck=stage_qcheck, k8_cov=stage_k8_cov, k8_timing=stage_k8_timing,
              k8_q32=stage_k8_q32, k8_q256=stage_k8_q256, k8_graph=stage_k8_graph, w8_screen=stage_w8_screen,
              w8_dev=stage_w8_dev, w8_confirm=stage_w8_confirm, m25_screen=stage_m25_screen, m25_dev=stage_m25_dev,
              m25_confirm=stage_m25_confirm, final=stage_final, interaction=stage_interaction, timing=stage_timing,
              scripts=stage_scripts)
NEEDS_SELECTION = {"k8_graph", "w8_dev", "w8_confirm", "m25_screen", "m25_dev", "m25_confirm", "final", "interaction",
                   "timing", "scripts"}


def run_plan(plan, out, timeout, resume):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "plan.json"
    if not pf.exists() or not resume:
        pf.write_text(json.dumps(plan, indent=1))
    mf = d / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps(KW.manifest(), indent=1))
    status = d / "status.jsonl"
    skip = done_tags(status) if resume else set()
    for job in plan["jobs"]:
        tag = job["tag"]
        if tag in skip:
            continue
        job = dict(job, _out=d / f"{tag}.json", _rel=f"{out}/{tag}")
        rc, wall = WJ.run_env(RJ.cmd_of(job, d), d / f"{tag}.log", timeout, job.get("env", {}))
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, kind=job["kind"], status=st, exit=rc, wall_s=wall,
                            json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:300],
                            job={k: v for k, v in job.items() if not k.startswith("_")},
                            log_tail=log.splitlines()[-30:] if st != "measured" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="write plan.json / manifest.json and exit")
    a = ap.parse_args()
    sel = json.loads(Path(a.selection).read_text()) if a.selection else None
    if a.stage in NEEDS_SELECTION and sel is None:
        raise SystemExit(f"stage {a.stage} needs --selection")
    plan = STAGES[a.stage](sel)
    if a.plan_only:
        d = results_root() / a.out
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(json.dumps(plan, indent=1))
        (d / "manifest.json").write_text(json.dumps(KW.manifest(), indent=1))
        print(f"{len(plan['jobs'])} jobs -> {d / 'plan.json'}")
        return
    run_plan(plan, a.out, a.timeout, a.resume)


if __name__ == "__main__":
    main()
