"""Plans and runner of the TopK/16 confirmation / coverage-constrained TopK round
(results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md).

Same discipline as kwa_jobs.py / rsn_jobs.py: every stage's plan (job list) is BUILT from declared builders plus, where
a stage depends on earlier results, an explicit selection file, and written to $GRANULE_RESULTS_ROOT/<out>/plan.json
before any job runs; the case manifest goes to manifest.json; every job runs in a fresh process with an explicit status
(measured | failed | oom | timeout) appended to status.jsonl.

    python tkc_jobs.py <stage> --out DIR [--selection sel.json] [--resume] [--plan-only]

Timing jobs: bench_supergroups.py --canonical-start --identity (rsn_jobs.cmd_of): warm-up / compile, restore of the
pre-warm-up values, reset of the complete optimizer / main_grad / rounding state and a canonical-identity check before
every measured ROUND; inside a round the reps x iters steps run sequentially WITHOUT weight resets (the same boundaries
for every method).  Screens 2 x 2 x 2 = 8 samples, confirmations 4 x 2 x 3 = 24 per fresh process; dense first /
middle / last of every session (one session per graph seed).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import common  # noqa: F401
import kwa_jobs as KJ
import rsn_cases as RC
import rsn_jobs as RJ
import tkc_cases as TC
import wps_jobs as WJ
from k64_campaign import classify, done_tags, record, results_root

SCREEN_ROUNDS, CONFIRM_ROUNDS = RJ.SCREEN_ROUNDS, RJ.CONFIRM_ROUNDS
ORDER_SEED = 20261029


def session(pairs, graph_seed, rounds, order_seed):
    rng = random.Random(order_seed)
    body = [KJ.timing_job(s, f, graph_seed, rounds) for s, f in pairs]
    rng.shuffle(body)
    mid = len(body) // 2
    a = [KJ.timing_job("dense", "dense", graph_seed, rounds, tag=f"dense_g{graph_seed}_anchor{i}") for i in range(3)]
    return [a[0]] + body[:mid] + [a[1]] + body[mid:] + [a[2]]


def learn_job(spec, norm="none", steps=32, data_seed=0, graph_seed=0, final=False, save=False):
    j = KJ.learn_job(spec, norm, steps, data_seed, graph_seed=graph_seed, final=final, save=save)
    if final:
        j["args"] += ["--test-protocol", TC.TEST_PROTOCOL]
        j["tag"] = j["tag"].replace("_final", f"_{TC.TEST_PROTOCOL}")
    return j


def interaction_job(spec, init_seed, sr_seed, steps=TC.INTERACTION_STEPS, save_at="", resume="", final=True):
    j = KJ.interaction_job(spec, terms=16, init_seed=init_seed, sr_seed=sr_seed, steps=steps, lr=TC.INTERACTION_LR,
                           save_at=save_at, resume=resume, final=False,
                           tag=f"int_{RC.spec_id(spec)}_T16_i{init_seed}_sr{sr_seed}_s{steps}_lr{TC.INTERACTION_LR:g}")
    j["args"] += ["--lr-readout", str(TC.INTERACTION_LR_READOUT)]
    if final:
        j["args"] += ["--final-test", "--test-protocol", TC.INTERACTION_TEST_PROTOCOL]
    return j


def script(tag, name, args):
    return dict(kind="script", tag=tag, script=name, args=list(args))


# ------------------------------------------------------------------ stage builders
def stage_construct(sel=None):
    jobs = [script(f"construct_g{g}", "tkc_construct.py", ["--graph-seed", str(g)] + (["--sample", "256"] if g == 0 else []))
            for g in TC.GRAPH_SEEDS]
    jobs += [script(f"floor_check_i{i}", "tkc_floor_check.py",
                    ["--supports", ",".join(TC.INTERACTION_SPECS), "--init-seed", str(i), "--batches", "64"])
             for i, _ in TC.INTERACTION_INIT_SR]
    return dict(stage="construct", jobs=jobs, note="one-time support construction cost / determinism / exact-optimum "
                "sample (graph 0) and the reduced-task missing-input floor Monte Carlo checks (init seeds 20 / 21 / 22)")


def stage_perf_screen(sel=None):
    pairs = [(s, "indexed") for s in ("TopK/8", "Cover-TopK/8", "TopK/16", "Cover-TopK/16")]
    return dict(stage="perf_screen", rounds=list(SCREEN_ROUNDS), jobs=session(pairs, 0, SCREEN_ROUNDS, ORDER_SEED),
                note="graph 0, 8 restored samples: Cover-TopK vs TopK on the same indexed kgen K8 / K16 path")


def stage_perf_confirm(sel=None):
    jobs = []
    for g in TC.GRAPH_SEEDS:
        pairs = [(s, TC.FORMAT[s]) for s in TC.K16_SET + TC.K8_SET]
        jobs += session(pairs, g, CONFIRM_ROUNDS, ORDER_SEED + 10 + g)
    return dict(stage="perf_confirm", rounds=list(CONFIRM_ROUNDS), jobs=jobs,
                note="one fresh-process session per graph seed: the 200M set (TopK/16, Cover-TopK/16, BP-row/16 layout 0 / "
                     "1 indexed, U16 / A8 wide, Mix-TopK grouped) and the 100M set (TopK/8, Cover-TopK/8, BP-row/8 layout "
                     "0 / 1 indexed, U8 wide); dense first / middle / last; 24 samples per job")


def stage_coverage(sel=None):
    jobs = []
    for g in TC.GRAPH_SEEDS:
        jobs += [KJ.coverage_job(s, g) for s in TC.K16_SET + TC.K8_SET + ("dense",)]
    return dict(stage="coverage", jobs=jobs, note="every exact timed support at graph 0 / 1 / 2: exact pairs / triples, "
                "common fixed tuple samples, joint-query per-output intervals, record / output feature unions")


def stage_dev32(sel=None):
    specs = ("Cover-TopK/8", "Cover-TopK/16", "TopK/8", "TopK/16")
    return dict(stage="dev32", jobs=[learn_job(s, n, 32, 0) for s in specs for n in ("none", "energy")],
                note="32-update continuity checks (graph 0, data / SR seed 0); TopK = continuity controls against the "
                     "previous round's records")


def stage_dev256(sel=None):
    specs = ("Cover-TopK/8", "Cover-TopK/16")
    return dict(stage="dev256", jobs=[learn_job(s, n, 256, d) for s in specs for n in ("none", "energy")
                                      for d in TC.DEV_SEEDS])


def stage_final(sel):
    """sel: dict(methods=[[spec, norm], ...]) -- frozen before any job of this stage."""
    jobs = [learn_job(s, n, 256, d, final=True, save=True) for s, n in sel["methods"] for d in TC.FINAL_SEEDS]
    return dict(stage="final", selection=sel, jobs=jobs,
                note=f"NEW locked test {TC.TEST_PROTOCOL} (batches 29_000..29_007) once after training; fresh data / SR "
                     "seeds 6..11 paired across methods; graph 0 (BP layout 0); final states kept")


def stage_graph_checks(sel):
    """sel: dict(methods=[[spec, norm], ...]) -- graph 1 / 2 quality checks at data / SR seed 6."""
    jobs = [learn_job(s, n, 256, TC.GRAPH_CHECK_SEED, graph_seed=g, final=True) for g in (1, 2)
            for s, n in sel["methods"]]
    return dict(stage="graph_checks", selection=sel, jobs=jobs,
                note="same-graph checkpoints / projection / head; reported separately from the six-seed intervals")


def stage_interaction(sel=None):
    jobs = []
    for spec in TC.INTERACTION_SPECS:
        for i, sr in TC.INTERACTION_INIT_SR:
            save = str(TC.INTERACTION_STEPS) if spec in TC.INTERACTION_CONTINUED else ""
            jobs.append(interaction_job(spec, i, sr, save_at=save))
    return dict(stage="interaction", jobs=jobs, note="reduced raw-input task, T16, lr 1e-2 / readout 1e-2, 1000 updates, "
                f"init 20 / 21 / 22 with SR 0 / 1 / 2; final interaction test {TC.INTERACTION_TEST_PROTOCOL} at the endpoint")


def stage_interaction_cont(sel):
    """sel: dict(states={tag of the 1000-update run: state path})."""
    jobs = []
    for spec in TC.INTERACTION_CONTINUED:
        for i, sr in TC.INTERACTION_INIT_SR:
            t1000 = interaction_job(spec, i, sr)["tag"]
            jobs.append(interaction_job(spec, i, sr, steps=TC.INTERACTION_LONG, save_at=str(TC.INTERACTION_LONG),
                                        resume=sel["states"][t1000]))
    return dict(stage="interaction_cont", selection=sel, jobs=jobs,
                note="continuations 1000 -> 4000 updates from the complete saved state (same protocol)")


def stage_scripts(sel):
    return dict(stage="scripts", selection=sel, jobs=sel["jobs"])


def stage_timing(sel):
    jobs = []
    for i, ss in enumerate(sel["sessions"]):
        jobs += session([tuple(p) for p in ss["pairs"]], ss.get("graph_seed", 0), ss.get("rounds", CONFIRM_ROUNDS),
                        ORDER_SEED + 100 + i)
    return dict(stage="timing", selection=sel, jobs=jobs)


STAGES = dict(construct=stage_construct, perf_screen=stage_perf_screen, perf_confirm=stage_perf_confirm,
              coverage=stage_coverage, dev32=stage_dev32, dev256=stage_dev256, final=stage_final,
              graph_checks=stage_graph_checks, interaction=stage_interaction, interaction_cont=stage_interaction_cont,
              scripts=stage_scripts, timing=stage_timing)
NEEDS_SELECTION = {"final", "graph_checks", "interaction_cont", "scripts", "timing"}


def run_plan(plan, out, timeout, resume):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    pf = d / "plan.json"
    if not pf.exists() or not resume:
        pf.write_text(json.dumps(plan, indent=1))
    mf = d / "manifest.json"
    if not mf.exists():
        mf.write_text(json.dumps(TC.manifest(), indent=1))
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
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    sel = json.loads(Path(a.selection).read_text()) if a.selection else None
    if a.stage in NEEDS_SELECTION and sel is None:
        raise SystemExit(f"stage {a.stage} needs --selection")
    plan = STAGES[a.stage](sel)
    if a.plan_only:
        d = results_root() / a.out
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(json.dumps(plan, indent=1))
        (d / "manifest.json").write_text(json.dumps(TC.manifest(), indent=1))
        print(f"{len(plan['jobs'])} jobs -> {d / 'plan.json'}")
        return
    run_plan(plan, a.out, a.timeout, a.resume)


if __name__ == "__main__":
    main()
