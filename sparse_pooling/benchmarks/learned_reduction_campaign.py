"""Declared campaign of the learned-reduction study: manifests, validation-only selection, resumable stages.  The rules
below were committed before any screening result existed.

Stage 4, screening (H = 2,621,440, seed 0, tasks compose / interact / local, M 512):
  e2e: every family gets the same number of trials (8).
    tune1 (6 trials, 1024 updates): lr_w in {1e-4, 3e-4, 1e-3} x lr_small in {3e-4, 1e-3}.
    tune2 (2 trials, 1024 updates) around the best tune1 trial on final validation loss: families with a learned P try
          p_mult in {0.3, 0.1}; fixed families try lr_w x {0.3, 3}.  (A declared, equal-budget search dimension.)
    final (4096 updates, --test 1): the validation-selected trial of each (model, task).  Test losses appear only here.
  frozen: 4 trials, 1024 updates: fixed families lr_small in {1e-4, 3e-4, 1e-3, 3e-3}; learned families lr_small in {3e-4, 1e-3,
          3e-3} at P lr 1e-4 plus (lr_small 1e-3, P lr 1e-3).  final: 2048 updates, --test 1.
Stage 5, confirmation (H = 10,485,760):
  smoke: every core variant, the screening-selected LR of its (model, task = compose), 256 updates (optimization smoke).
  tune: the finalist set on compose and interact, 3 trials each, 512 updates: the three best screening trials of that
        (model, task).
  final: finalists x seeds {0, 1, 2} x tasks {compose, interact}, 4096 updates, --test 1.
  Finalists (declared rule): A0, A1, A4, MC, and the strongest non-overlapping reducer = the one of {A2, A3, A3gelu, A3mean} with the
  lowest mean screening validation loss across the three tasks; plus A4gelu if it beats A4 on at least two of three tasks.
Stage 6: secondary (H = 12,582,912): A0 and the leading learned reducer, compose, seed 0, 1024 updates at their primary LR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORE = ("A0", "A1", "A2", "A3", "A4", "A5", "MC")
EXTRA = ("A3gelu", "A4gelu", "A3mean", "A4mean", "A2gelu", "A5gelu", "TS_id", "TS_gelu")
LEARNED = {"A3", "A4", "MC", "A3gelu", "A4gelu", "A3mean", "A4mean", "TS_id", "TS_gelu"}
TASKS = ("compose", "interact", "local")
H_SCREEN, H_PRIMARY, H_SECONDARY = 2_621_440, 10_485_760, 12_582_912


def load(out: Path, stage: str):
    rs = []
    for f in sorted((out / stage).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("schema") == "granule.lsw_train/1":
            rs.append(r)
    return rs


def key(r):
    return (r["model"], r["task"], r["mode"])


def score(r):
    return r["final_val"] if r["finite"] else float("inf")


def trial(r):
    return dict(lr_w=r["lr_w"], lr_small=r["lr_small"], p_mult=r["p_mult"])


def ranked(rs):
    d = {}
    for r in rs:
        d.setdefault(key(r), []).append(r)
    return {k: sorted(v, key=score) for k, v in d.items()}


def manifest_tune1():
    return dict(defaults=dict(H=H_SCREEN, seed=0, steps=1024, eval_every=128, mode="e2e"),
                jobs=[dict(model=m, task=t, lr_w=w, lr_small=s, p_mult=1.0)
                      for t in TASKS for m in CORE + EXTRA for w in (1e-4, 3e-4, 1e-3) for s in (3e-4, 1e-3)])


def manifest_tune2(out):
    best = ranked(load(out, "screen_tune1"))
    jobs = []
    for (m, t, _), v in best.items():
        b = v[0]
        if m in LEARNED:
            jobs += [dict(model=m, task=t, lr_w=b["lr_w"], lr_small=b["lr_small"], p_mult=p) for p in (0.3, 0.1)]
        else:
            jobs += [dict(model=m, task=t, lr_w=round(b["lr_w"] * f, 8), lr_small=b["lr_small"], p_mult=1.0) for f in (0.3, 3.0)]
    return dict(defaults=dict(H=H_SCREEN, seed=0, steps=1024, eval_every=128, mode="e2e"), jobs=jobs)


def manifest_final(out):
    best = ranked(load(out, "screen_tune1") + load(out, "screen_tune2"))
    return dict(defaults=dict(H=H_SCREEN, seed=0, steps=4096, eval_every=128, mode="e2e", test=1),
                jobs=[dict(model=m, task=t, **trial(v[0])) for (m, t, _), v in best.items()])


def manifest_frozen_tune():
    jobs = []
    for t in TASKS:
        for m in CORE + EXTRA:
            if m in LEARNED:
                tr = [(s, 1e-4) for s in (3e-4, 1e-3, 3e-3)] + [(1e-3, 1e-3)]
            else:
                tr = [(s, 1e-4) for s in (1e-4, 3e-4, 1e-3, 3e-3)]
            jobs += [dict(model=m, task=t, lr_small=s, lr_w=w, p_mult=1.0) for s, w in tr]
    return dict(defaults=dict(H=H_SCREEN, seed=0, steps=1024, eval_every=128, mode="frozen"), jobs=jobs)


def manifest_frozen_final(out):
    best = ranked(load(out, "frozen_tune"))
    return dict(defaults=dict(H=H_SCREEN, seed=0, steps=2048, eval_every=128, mode="frozen", test=1),
                jobs=[dict(model=m, task=t, **trial(v[0])) for (m, t, _), v in best.items()])


def finalists(out):
    best = ranked(load(out, "screen_tune1") + load(out, "screen_tune2"))
    mean = {m: sum(score(best[(m, t, "e2e")][0]) for t in TASKS) / 3 for m in ("A2", "A3", "A3gelu", "A3mean")}
    strongest = min(mean, key=mean.get)
    f = ["A0", "A1", "A4", "MC", strongest]
    wins = sum(score(best[("A4gelu", t, "e2e")][0]) < score(best[("A4", t, "e2e")][0]) for t in TASKS)
    if wins >= 2:
        f.append("A4gelu")
    return dict(finalists=f, strongest_non_overlap=strongest, non_overlap_scores=mean, a4gelu_wins=int(wins))


def manifest_primary_smoke(out):
    best = ranked(load(out, "screen_tune1") + load(out, "screen_tune2"))
    ms = list(CORE) + ["A3gelu", "A4gelu"]
    return dict(defaults=dict(H=H_PRIMARY, seed=0, steps=256, eval_every=64, mode="e2e", task="compose"),
                jobs=[dict(model=m, **trial(best[(m, "compose", "e2e")][0])) for m in ms])


def manifest_primary_tune(out):
    best = ranked(load(out, "screen_tune1") + load(out, "screen_tune2"))
    fin = finalists(out)["finalists"]
    jobs = []
    for t in ("compose", "interact"):
        for m in fin:
            jobs += [dict(model=m, task=t, **trial(r)) for r in best[(m, t, "e2e")][:3]]
    return dict(defaults=dict(H=H_PRIMARY, seed=0, steps=512, eval_every=128, mode="e2e"), jobs=jobs)


def manifest_primary_final(out):
    best = ranked(load(out, "primary_tune"))
    jobs = []
    for (m, t, _), v in best.items():
        for s in (0, 1, 2):
            jobs.append(dict(model=m, task=t, seed=s, **trial(v[0])))
    return dict(defaults=dict(H=H_PRIMARY, steps=4096, eval_every=128, mode="e2e", test=1), jobs=jobs)


def manifest_secondary(out):
    best = ranked(load(out, "primary_tune"))
    means = {m: sum(score(v[0]) for (mm, t, _), v in best.items() if mm == m) for m, _, _ in best if m in LEARNED}
    lead = min(means, key=means.get)
    return dict(defaults=dict(H=H_SECONDARY, seed=0, steps=1024, eval_every=128, mode="e2e", task="compose", test=1),
                jobs=[dict(model=m, **trial(best[(m, "compose", "e2e")][0])) for m in ("A0", lead)], leading=lead)


MANIFESTS = dict(screen_tune1=lambda o: manifest_tune1(), screen_tune2=manifest_tune2, screen_final=manifest_final,
                 frozen_tune=lambda o: manifest_frozen_tune(), frozen_final=manifest_frozen_final,
                 primary_smoke=manifest_primary_smoke, primary_tune=manifest_primary_tune,
                 primary_final=manifest_primary_final, secondary=manifest_secondary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(MANIFESTS) + ["finalists"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    m = finalists(out) if a.stage == "finalists" else MANIFESTS[a.stage](out)
    Path(a.write).parent.mkdir(parents=True, exist_ok=True)
    Path(a.write).write_text(json.dumps(m, indent=1))
    print(a.stage, len(m.get("jobs", [])) if "jobs" in m else m)


if __name__ == "__main__":
    main()
