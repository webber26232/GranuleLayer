"""Plans and runner of the interaction-coverage / K16 dX round (results/INTERACTION_COVERAGE_K16_DX.md).

Same discipline as tkc_jobs.py: every stage's plan (job list) is BUILT from declared builders plus, where a stage
depends on earlier results, an explicit selection file (icdx_select.py), and written to
$GRANULE_RESULTS_ROOT/<out>/plan.json before any job runs; the case manifest (with the seed collision check) goes to
manifest.json; every job runs in a fresh process with an explicit status (measured | failed | oom | timeout) appended
to status.jsonl.

    python icdx_jobs.py <stage> --out DIR [--selection sel.json] [--resume] [--plan-only]

Timing jobs: bench_supergroups.py --canonical-start --identity in a fresh process (warm-up / compile, restore of the
pre-warm-up values and the complete optimizer / main_grad / rounding state, canonical-identity check before every
measured round; inside a round the reps x iters steps run sequentially without resets).  Screens 2 x 2 x 2 = 8
samples, confirmations 4 x 2 x 3 = 24; dense first / middle / last; the body order shuffled with a declared seed.  The
dX variant travels in the factory arguments (``dx_variant``); the timing factories apply NO row-energy normalization
(identical for every compared backend).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import common  # noqa: F401
import icdx_cases as IC
import kwa_jobs as KJ
import rsn_cases as RC
import wps_cases as WC
import wps_jobs as WJ
from k64_campaign import classify, done_tags, record, results_root

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ job builders
def timing_job(spec, fmt, graph_seed=0, rounds=IC.SCREEN_ROUNDS, dxv="legacy", batch=WC.M, tag=None):
    f = KJ.factory_of(spec, fmt, graph_seed)
    if dxv != "legacy":
        f["dx_variant"] = dxv
    tag = tag or f"{RC.spec_id(spec)}_g{graph_seed}_{fmt}_{dxv}_M{batch}"
    return dict(kind="timing", tag=tag, C=RC.C, graph_seed=graph_seed, factory=f, rounds=list(rounds), env={},
                support=spec, format=fmt, dx_variant=dxv, batch=batch)


def session(items, graph_seed, rounds, order_seed, batch=WC.M, anchors=True):
    """items: [(spec, fmt, dxv)]; shuffled body, dense control first / middle / last."""
    rng = random.Random(order_seed)
    body = [timing_job(s, f, graph_seed, rounds, v, batch) for s, f, v in items]
    rng.shuffle(body)
    if not anchors:
        return body
    mid = len(body) // 2
    a = [timing_job("dense", "dense", graph_seed, rounds, "legacy", batch, tag=f"dense_g{graph_seed}_M{batch}_anchor{i}")
         for i in range(3)]
    return [a[0]] + body[:mid] + [a[1]] + body[mid:] + [a[2]]


def script(tag, name, args):
    return dict(kind="script", tag=tag, script=name, args=list(args))


def learn_job(spec, norm, steps, data_seed, graph_seed=0, final=False, save=False):
    j = KJ.learn_job(spec, norm, steps, data_seed, graph_seed=graph_seed, final=final, save=save)
    if final:
        j["args"] += ["--test-protocol", IC.TEST_PROTOCOL]
        j["tag"] = j["tag"].replace("_final", f"_{IC.TEST_PROTOCOL}")
    return j


def backend_job(spec, norm, steps, data_seed, dxv, graph_seed=0, final=False, save=False):
    tag = f"bt_{RC.spec_id(spec)}_g{graph_seed}_{norm}_s{steps}_d{data_seed}_{dxv}" + (f"_{IC.TEST_PROTOCOL}" if final
                                                                                        else "")
    args = ["--support", spec, "--norm", norm, "--steps", str(steps), "--data-seed", str(data_seed), "--sr-seed",
            str(data_seed), "--graph-seed", str(graph_seed), "--dx-variant", dxv, "--eval-every",
            "8" if steps <= 32 else "32"]
    if final:
        args += ["--final-test", "--test-protocol", IC.TEST_PROTOCOL]
    if save:
        args += ["--save-state", "{state_dir}"]
    return script(tag, "icdx_backend_train.py", args)


def interaction_tag(spec, init_seed, sr_seed, steps):
    return f"int_{RC.spec_id(spec)}_T16_i{init_seed}_sr{sr_seed}_s{steps}_lr{IC.INTERACTION_LR:g}"


def interaction_job(spec, init_seed, sr_seed, steps=IC.INTERACTION_STEPS, save_at="", resume="", final=False,
                    analysis=False, eval_state="", tag=None):
    tag = tag or interaction_tag(spec, init_seed, sr_seed, steps)
    args = ["--support", spec, "--terms", "16", "--norm", "none", "--init-seed", str(init_seed), "--sr-seed",
            str(sr_seed), "--steps", str(steps), "--lr", str(IC.INTERACTION_LR), "--lr-readout",
            str(IC.INTERACTION_LR_READOUT), "--eval-every", "50" if steps <= 1000 else "100"]
    if save_at:
        args += ["--save-at", save_at, "--state-dir", "{state_dir}"]
    if resume:
        args += ["--resume", resume]
    if eval_state:
        args += ["--eval-state", eval_state]
    if final:
        args += ["--final-test", "--test-protocol", IC.INTERACTION_TEST_PROTOCOL]
    if analysis:
        args += ["--analysis-out", "{out_stem}.analysis.npz"]
    return script(tag, "rsn_interaction.py", args)


# ------------------------------------------------------------------ stage builders
COVERAGE_G0 = ("Cover-TopK/16", "Balanced-TopK/16#0", "Balanced-TopK/16#1", "Cover-Mix32:p1#0", "Cover-Mix32:p2#0",
               "Cover-Mix32:p1#1", "Cover-Mix32:p2#1", "BP-row#0", "U16@0.25", "Mix-TopK#0", "dense")
COVERAGE_G12 = ("Cover-TopK/16", "Balanced-TopK/16#0", "Cover-Mix32:p1#0", "Cover-Mix32:p2#0", "BP-row#0", "U16@0.25",
                "Mix-TopK#0", "dense")


def stage_construct(sel=None):
    jobs = [script(f"construct_g{g}", "icdx_construct.py", ["--graph-seed", str(g)] + (["--sample", "128"] if g == 0
                                                                                        else [])) for g in IC.GRAPH_SEEDS]
    return dict(stage="construct", jobs=jobs, note="CPU / GPU construction, determinism, fingerprints, multiplicity; "
                "exact Balanced quota optimum on <= 128 stratified records (graph 0)")


def stage_coverage(sel=None):
    jobs = []
    for g in IC.GRAPH_SEEDS:
        for s in (COVERAGE_G0 if g == 0 else COVERAGE_G12):
            jobs.append(KJ.coverage_job(s, g))
            if s != "dense":
                jobs.append(script(f"inv_{RC.spec_id(s)}_g{g}", "icdx_inventory.py", ["--support", s, "--graph-seed",
                                                                                     str(g)]))
    return dict(stage="coverage", jobs=jobs, note="rsn_coverage (exact layer tuples, common-query per-output tuples with "
                "joint intervals, sharing / multiplicity) + icdx_inventory (multiplicity, record-scope tuples, incidence "
                "sums / union bounds) for every candidate and control; graph 1 / 2 for both Mix32 p (selection not yet "
                "made), Balanced / Cover / controls")


def stage_dev32(sel=None):
    jobs = [learn_job(s, IC.CONTROL_TREATMENT[s], 32, 0) for s in IC.CONTROLS]
    return dict(stage="dev32", jobs=jobs, note="32-update continuity checks of the controls (graph 0, seed 0) against "
                "their earlier records (icdx_cases.CONTINUITY_REFERENCE)")


def stage_dev256(sel=None):
    jobs = [learn_job(s, n, 256, d) for s in IC.NEW for n in ("none", "energy") for d in IC.DEV_SEEDS]
    jobs += [learn_job(s, IC.CONTROL_TREATMENT[s], 256, d) for s in ("BP-row#0", "U16@0.25", "dense")
             for d in IC.DEV_SEEDS]
    return dict(stage="dev256", jobs=jobs, note="new supports x {none, energy} x seeds 0 / 1 / 2 (treatment selection); "
                "controls without reusable 256-update development records (BP-row / U16 / dense), descriptive only")


def stage_dev256_rerun(sel):
    """sel: continuity_selection.json -> rerun list of controls whose continuity failed."""
    jobs = [learn_job(s, IC.CONTROL_TREATMENT[s], 256, d) for s in sel["rerun"] for d in IC.DEV_SEEDS]
    return dict(stage="dev256_rerun", selection=sel, jobs=jobs)


def stage_interaction(sel=None):
    jobs = [interaction_job(s, i, sr, save_at=str(IC.INTERACTION_STEPS)) for s in IC.CORE8
            for i, sr in IC.INTERACTION_INIT_SR]
    jobs.append(script("oracle_Mix-TopK_anchor_i30", "rsn_interaction.py",
                       ["--support", "Mix-TopK#0", "--terms", "16", "--oracle", "anchor", "--init-seed", "30",
                        "--steps", "0", "--lr", str(IC.INTERACTION_LR)]))
    jobs.append(script("resume_check", "icdx_resume_check.py", []))
    return dict(stage="interaction", jobs=jobs, note="reduced task, 8 core methods x init / SR (30, 0), (31, 1), (32, 2), "
                "1000 updates, complete states saved, NO test evaluation (icdx89000 only after the selections freeze); "
                "Mix-TopK T16 anchor oracle (attainable-solution control); uninterrupted vs resumed short run")


def stage_anova(sel=None):
    jobs = [script(f"anova_{RC.spec_id(s)}_i{i}", "icdx_anova.py", ["--support", s, "--init-seed", str(i),
                                                                     "--per-order", str(IC.ANOVA_PER_ORDER),
                                                                     "--sample-seed", str(IC.ANOVA_SAMPLE_SEED)])
            for s in IC.CORE8 for i, _ in IC.INTERACTION_INIT_SR]
    return dict(stage="anova", jobs=jobs, note="functional-ANOVA structural bound, 32 outputs per order (declared seed), "
                "analysis batches 97_000..97_063")


def stage_profile(sel=None):
    return dict(stage="profile", jobs=[script("profile_cover16_g0", "icdx_profile.py", ["--support", "Cover-TopK/16"])],
                note="restored Cover-TopK/16 baseline outside any timing session: compiled attributes / occupancy of "
                     "every screened tiling, nsys kernel timeline, Nsight Compute counters if permitted")


def stage_dx_screen(sel=None):
    from sparse_pooling.experimental import kgen_dx_variants as DXV

    vs = ["legacy"] + [f"mt{a}_nw{b}_wg{c}" for a, b, c in DXV.SCREEN if (a, b, c) != (4, 8, 2)]
    items = [(s, "indexed", v) for s in IC.DX_SCREEN_SUPPORTS for v in vs]
    jobs = session(items, 0, IC.SCREEN_ROUNDS, IC.ORDER_SEED + 1)
    jobs += session([("Cover-TopK/16", "indexed", v) for v in vs], 0, IC.SCREEN_ROUNDS, IC.ORDER_SEED + 2, batch=512,
                    anchors=False)
    return dict(stage="dx_screen", jobs=jobs, note="graph 0, 8 samples per process: legacy + 7 tilings on Cover-TopK/16 "
                "and BP-row/16 at M 8192 (dense anchors), and Cover-TopK/16 at M 512; (4, 8, 2) = legacy (original entry)")


def stage_confirm(sel):
    """sel: dx_screen_selection.json (finalists) + treatments_selection.json (mixed) merged: dict(finalists=[...],
    mixed=spec)."""
    vs = ["legacy"] + list(sel["finalists"])
    jobs = []
    for g in IC.GRAPH_SEEDS:
        items = [(s, "indexed", v) for s in ("Cover-TopK/16", "Balanced-TopK/16#0", "BP-row#0") for v in vs]
        items += [(sel["mixed"], "grouped", v) for v in vs]
        items += [("U16@0.25", "wide", "legacy")]
        jobs += session(items, g, IC.CONFIRM_ROUNDS, IC.ORDER_SEED + 10 + g)
    jobs += session([(s, "indexed", v) for s in ("Cover-TopK/16",) for v in vs], 0, IC.CONFIRM_ROUNDS,
                    IC.ORDER_SEED + 20, batch=512, anchors=False)
    return dict(stage="confirm", selection=sel, jobs=jobs, note="legacy vs <= 2 finalists, graphs 0 / 1 / 2, M 8192, 24 "
                "samples per process; grouped Mix32 changes only its K16 component; U16 / dense same-session anchors; "
                "Cover-TopK/16 at M 512")


def stage_guard(sel):
    """sel: backend_selection.json (chosen) -> M 512 on BP-row / Mix32 and a small-batch shape guard."""
    v = sel["chosen"]
    if v == "legacy":
        return dict(stage="guard", selection=sel, jobs=[], note="no backend qualified: nothing to guard")
    jobs = session([(s, f, x) for s, f in (("BP-row#0", "indexed"), (sel["mixed"], "grouped")) for x in ("legacy", v)],
                   0, IC.CONFIRM_ROUNDS, IC.ORDER_SEED + 30, batch=512, anchors=False)
    jobs += session([("Cover-TopK/16", "indexed", x) for x in ("legacy", v)], 0, IC.SCREEN_ROUNDS, IC.ORDER_SEED + 31,
                    batch=IC.SMALL_BATCH_GUARD, anchors=False)
    return dict(stage="guard", selection=sel, jobs=jobs)


def stage_replicates(sel):
    """Three independent graph-0 process pairs (legacy / chosen) in alternating order (session sensitivity)."""
    v = sel["chosen"]
    if v == "legacy":
        return dict(stage="replicates", selection=sel, jobs=[])
    jobs = []
    for r in range(3):
        pair = [("legacy", f"rep{r}_legacy"), (v, f"rep{r}_{v}")]
        if r % 2:
            pair.reverse()
        for x, tag in pair:
            jobs.append(timing_job("Cover-TopK/16", "indexed", 0, IC.CONFIRM_ROUNDS, x, tag=f"Cover-TopK_k16_g0_{tag}"))
    return dict(stage="replicates", selection=sel, jobs=jobs)


def stage_final(sel):
    """sel: final_selection.json: methods [[spec, norm]], backend dict(chosen), mixed."""
    jobs = [learn_job(s, n, 256, d, final=True, save=True) for s, n in sel["methods"] for d in IC.FINAL_SEEDS]
    cn = sel["treatments"]["Cover-TopK/16"]
    jobs += [backend_job("Cover-TopK/16", cn, 256, d, "legacy", final=True, save=True) for d in IC.FINAL_SEEDS]
    if sel["backend"]["chosen"] != "legacy":
        jobs += [backend_job("Cover-TopK/16", cn, 256, d, sel["backend"]["chosen"], final=True, save=True)
                 for d in IC.FINAL_SEEDS]
        mx, mn = sel["mixed"], sel["treatments"][sel["mixed"]]
        jobs += [backend_job(mx, mn, 32, d, x) for d in IC.DEV_SEEDS for x in ("legacy", sel["backend"]["chosen"])]
    return dict(stage="final", selection=sel, jobs=jobs,
                note=f"NEW locked test {IC.TEST_PROTOCOL} once per completed run; seeds 12..17; dense-masked rows 1-6, "
                     "compact indexed rows 7 / 8; grouped Mix32 32-update K16-component check on development seeds")


def stage_graph_checks(sel):
    jobs = [learn_job(s, sel["treatments"][s], 256, IC.GRAPH_CHECK_SEED, graph_seed=g, final=True)
            for g in (1, 2) for s in ("Cover-TopK/16", "Balanced-TopK/16#0", sel["mixed"])]
    return dict(stage="graph_checks", selection=sel, jobs=jobs)


def stage_interaction_eval(sel):
    """sel: dict(states={1000-update tag: state path}) -> one evaluation of icdx89000 + analysis batches per endpoint."""
    jobs = []
    for s in IC.CORE8:
        for i, sr in IC.INTERACTION_INIT_SR:
            t = interaction_tag(s, i, sr, IC.INTERACTION_STEPS)
            jobs.append(interaction_job(s, i, sr, eval_state=sel["states"][t], final=True, analysis=True,
                                        tag=t + "_eval"))
    return dict(stage="interaction_eval", selection=sel, jobs=jobs)


def stage_interaction_cont(sel):
    """sel: dict(states=..., continued=[specs])."""
    jobs = []
    for s in sel["continued"]:
        for i, sr in IC.INTERACTION_INIT_SR:
            t = interaction_tag(s, i, sr, IC.INTERACTION_STEPS)
            jobs.append(interaction_job(s, i, sr, steps=IC.INTERACTION_LONG, save_at=str(IC.INTERACTION_LONG),
                                        resume=sel["states"][t], final=True, analysis=True))
    return dict(stage="interaction_cont", selection=sel, jobs=jobs)


def stage_scripts(sel):
    return dict(stage="scripts", selection=sel, jobs=sel["jobs"])


STAGES = dict(construct=stage_construct, coverage=stage_coverage, dev32=stage_dev32, dev256=stage_dev256,
              dev256_rerun=stage_dev256_rerun, interaction=stage_interaction, anova=stage_anova, profile=stage_profile,
              dx_screen=stage_dx_screen, confirm=stage_confirm, guard=stage_guard, replicates=stage_replicates,
              final=stage_final, graph_checks=stage_graph_checks, interaction_eval=stage_interaction_eval,
              interaction_cont=stage_interaction_cont, scripts=stage_scripts)
NEEDS_SELECTION = {"dev256_rerun", "confirm", "guard", "replicates", "final", "graph_checks", "interaction_eval",
                   "interaction_cont", "scripts"}


def cmd_of(job, d: Path):
    if job["kind"] == "timing":
        r, p, i = job["rounds"]
        return [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(WC.H), "--C", str(job["C"]),
                "--S", str(WC.S), "--R", str(WC.R), "--graph-seed", str(job["graph_seed"]), "--init-seed",
                str(WC.INIT_SEED), "--input-seed", str(WC.INPUT_SEED), "--batch", str(job.get("batch", WC.M)),
                "--rounds", str(r), "--reps", str(p), "--iters", str(i), "--kcsr", "--canonical-start", "--identity",
                "--layer-factory", "k64_factory:wide", "--factory-args", json.dumps(job["factory"]),
                "--opt-factory", "k64_opt:l1", "--opt-args", json.dumps(dict(l1="none")), "--out", job["_rel"]]
    out_stem = str(job["_out"].with_suffix(""))
    args = [a.replace("{state_dir}", str(d / "state")).replace("{out_stem}", out_stem) for a in job["args"]]
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
        mf.write_text(json.dumps(dict(IC.manifest(), seed_collisions=IC.seed_collisions()), indent=1))
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
        (d / "manifest.json").write_text(json.dumps(dict(IC.manifest(), seed_collisions=IC.seed_collisions()), indent=1))
        print(f"{len(plan['jobs'])} jobs -> {d / 'plan.json'}")
        return
    run_plan(plan, a.out, a.timeout, a.resume)


if __name__ == "__main__":
    main()
