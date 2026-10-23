"""Campaign driver of the K64 large-C / index-free / L1 study: timing runs (bench_supergroups.py, one fresh
process each), coverage runs (k64_coverage.py) and learning runs (k64_learn.py), with explicit per-run status.

    python k64_campaign.py screen   --out screen
    python k64_campaign.py coverage --out coverage [--seeds 0]
    python k64_campaign.py confirm  --out confirm --selection confirm_selection.json
    python k64_campaign.py sparse   --out sparse  --plan sparse_plan.json

Results go to $GRANULE_RESULTS_ROOT (a directory OUTSIDE the measurement checkout, so the frozen source
tree stays clean); every record carries the provenance of the checkout under test.

Dense screen: every case runs its execution variants interleaved on ONE layer (same graph, parameters,
inputs; --layer-variants), so variant comparisons are exactly matched.  Variants per case:
  reference  kgen (C <= 256) or lowered (C > 256: physical C = 256 subblocks) on the existing kgen kernels;
  direct@T   record-indexed kgx kernels with physical neuron tile T in tiles(C);
  idf_rmw@T0 / idf@T (S = 64 only): index-free without / with register-accumulated dX.
tiles(C) = {min(C, 256)} + {64, 128 below C}.  Order: seeded shuffle; the same-session E24 anchor runs
first, in the middle and last; anchors E22 / E04 / E08 once each (labelled, never counted as cases).
Status of every run (status.jsonl): measured | failed | oom | timeout, with the exit code and log tail.
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

import k64_cases as KC

HERE = Path(__file__).resolve().parent
ORDER_SEED = 20261025


def results_root() -> Path:
    r = os.environ.get("GRANULE_RESULTS_ROOT")
    if not r:
        raise SystemExit("set GRANULE_RESULTS_ROOT to a directory outside the measurement checkout")
    return Path(r)


def tiles(C: int):
    return sorted({min(C, 256)} | {t for t in (64, 128) if t < C})


def dense_variants(case: dict):
    C, S = case["C"], case["S"]
    v = ["ref=" + ("kgen" if C <= 256 else "lowered")]
    v += [f"direct{t}=direct@{t}" for t in tiles(C)]
    if S == 64:
        t0 = min(C, 128)
        v += [f"idf_rmw{t0}=idf_rmw@{t0}"] + [f"idf{t}=idf@{t}" for t in tiles(C)]
    return v


def bench_cmd(case: dict, seed: int, rounds: int, reps: int, iters: int, out: str, variants=None, extra=()):
    a = [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(case["H"]), "--C", str(case["C"]),
         "--S", str(case["S"]), "--R", str(case["R"]), "--graph-seed", str(seed), "--init-seed", str(KC.INIT_SEED),
         "--input-seed", str(KC.INPUT_SEED), "--rounds", str(rounds), "--reps", str(reps), "--iters", str(iters),
         "--kcsr", "--out", out]
    cid = case["id"]
    if cid in KC.ANCHORS:
        be = case["backend"]
        if be == "native":
            a += ["--validate"]
        else:
            a += ["--layer-factory", be, "--factory-args", json.dumps({"K": case["K"]})]
    else:
        a += ["--layer-factory", "k64_factory:kgx", "--factory-args", json.dumps({"variant": "direct", "tile": 256}),
              "--layer-variants", ",".join(variants or dense_variants(case))]
    return a + list(extra)


def classify(rc: int, log: str) -> str:
    if rc == 0:
        return "measured"
    if rc == 124:
        return "timeout"
    if "out of memory" in log or "OutOfMemoryError" in log:
        return "oom"
    if "validation" in log.lower() and "fail" in log.lower():
        return "validation-failed"
    return "failed"


def run(cmd, log_path: Path, timeout: int):
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            rc = subprocess.run(["timeout", str(timeout)] + cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
        except Exception as e:  # noqa: BLE001
            fh.write(f"driver exception {e!r}\n")
            rc = -1
    return rc, time.time() - t0


def record(status_path: Path, row: dict):
    with open(status_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {row['tag']}: {row['status']} ({row['wall_s']:.0f} s) {row.get('summary', '')}",
          flush=True)


def done_tags(status: Path):
    if not status.exists():
        return set()
    return {json.loads(ln)["tag"] for ln in status.read_text().splitlines() if ln.strip()
            and json.loads(ln)["status"] == "measured"}


def timing(runs, out: str, rounds: int, reps: int, iters: int, timeout: int, resume: bool):
    d = results_root() / out
    d.mkdir(parents=True, exist_ok=True)
    status = d / "status.jsonl"
    (d / "order.json").write_text(json.dumps([[r[0], r[1]] for r in runs], indent=1))
    skip = done_tags(status) if resume else set()
    for n, (cid, seed, variants) in enumerate(runs, 1):
        case = KC.case(cid)
        tag = f"{cid}_s{seed}_run{n:02d}"
        if tag in skip:
            continue
        rc, wall = run(bench_cmd(case, seed, rounds, reps, iters, f"{out}/{tag}", variants), d / f"{tag}.log", timeout)
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, case=cid, graph_seed=seed, C=case["C"], S=case["S"], R=case["R"], H=case["H"],
                            K=case.get("K", 64), variants=variants or (dense_variants(case) if cid not in KC.ANCHORS
                                                                         else [case["backend"]]),
                            status=st, exit=rc, wall_s=wall, json=f"{tag}.json" if st == "measured" else None,
                            summary=tail[0][:300], log_tail=log.splitlines()[-30:] if st != "measured" else None))


def with_baselines(entries, seed=ORDER_SEED):
    rng = random.Random(seed)
    body = list(entries)
    rng.shuffle(body)
    mid = len(body) // 2
    base = ("E24", 0, None)
    return [base] + body[:mid] + [base] + body[mid:] + [base]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "coverage", "confirm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--selection", default="", help="confirm: JSON with the predeclared selection")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true", help="skip runs already measured (same tags)")
    a = ap.parse_args()
    KC.check_manifest()
    seeds = [int(s) for s in a.seeds.split(",")]
    ids = [c["id"] for c in KC.cases()]
    if a.only:
        ids = a.only.split(",")
    if a.mode == "screen":
        entries = [(cid, 0, None) for cid in ids] + [(x, 0, None) for x in ("E22", "E04", "E08")]
        timing(with_baselines(entries), a.out, 2, 2, 2, a.timeout, a.resume)
    elif a.mode == "confirm":
        sel = json.loads(Path(a.selection).read_text())
        entries = [(r["case"], s, r.get("variants")) for r in sel["architectures"] for s in sel["seeds"]]
        entries += [(x, s, None) for x in sel.get("anchors", []) for s in [0]]
        timing(with_baselines(entries, ORDER_SEED + 1), a.out, 4, 2, 3, a.timeout, a.resume)
    else:
        d = results_root() / a.out
        d.mkdir(parents=True, exist_ok=True)
        status = d / "status.jsonl"
        skip = done_tags(status) if a.resume else set()
        for cid in ids:
            for s in seeds:
                tag = f"{cid}_s{s}"
                if tag in skip:
                    continue
                cmd = [sys.executable, str(HERE / "k64_coverage.py"), "--case", cid, "--graph-seed", str(s),
                       "--out", str(d / f"{tag}.json")]
                rc, wall = run(cmd, d / f"{tag}.log", a.timeout)
                log = (d / f"{tag}.log").read_text(errors="replace")
                st = classify(rc, log)
                tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
                record(status, dict(tag=tag, case=cid, graph_seed=s, status=st, exit=rc, wall_s=wall,
                                    json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:400],
                                    log_tail=log.splitlines()[-25:] if st != "measured" else None))


if __name__ == "__main__":
    main()
