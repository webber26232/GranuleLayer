"""Campaign driver of the K/C/S/R sweep: timing runs (bench_supergroups.py, one fresh process each) and the
separate coverage runs (kcsr_coverage.py), with explicit per-run status.

    python kcsr_campaign.py screen  --out random_kcsr_coverage_performance/screen   [--only ID,...]
    python kcsr_campaign.py confirm --out random_kcsr_coverage_performance/confirm --cases ID,... [--seeds 0,1,2]
    python kcsr_campaign.py coverage --out random_kcsr_coverage_performance/coverage [--seeds 0]

Backends per case (``backends``): native (production layer; E01-E04, X01), lowered (logical C = 128 / 256 on
the production C = 64 kernels; E05, E06), kgen (experimental generalized fan-in; every K > 8 case, and the
common-backend rows of E05 / E06), kgen_native (the matched K = 8 control on exactly the native graph).
The order is a seeded shuffle; the X01 native baseline runs first, in the middle and last (session drift).
Status of every run (status.jsonl): measured | failed | oom | timeout, with the exit code and log tail.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import kcsr_cases as KC

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
ORDER_SEED = 20261023


def backends(case: dict):
    K, C = case["K"], case["C"]
    if K == 8 and C <= 64:
        return ["native", "kgen_native"]
    if K == 8:
        return ["lowered", "kgen"]
    return ["kgen"]


def bench_args(case: dict, backend: str, seed: int, rounds: int, reps: int, iters: int):
    a = [sys.executable, str(HERE / "bench_supergroups.py"), "--hidden", str(case["H"]), "--C", str(case["C"]),
         "--S", str(case["S"]), "--R", str(case["R"]), "--graph-seed", str(seed), "--init-seed", str(KC.INIT_SEED),
         "--input-seed", str(KC.INPUT_SEED), "--rounds", str(rounds), "--reps", str(reps), "--iters", str(iters),
         "--kcsr"]
    if backend == "native":
        a += ["--validate"]
    elif backend == "lowered":
        a += ["--validate", "--layer-factory", "kcsr_factory:lowered"]
    elif backend == "kgen":
        a += ["--layer-factory", "kcsr_factory:kgen", "--factory-args", json.dumps({"K": case["K"]})]
    elif backend == "kgen_native":
        a += ["--layer-factory", "kcsr_factory:kgen_native"]
    return a


def classify(rc: int, log: str) -> str:
    if rc == 0:
        return "measured"
    if rc == 124:
        return "timeout"
    if "out of memory" in log or "OutOfMemoryError" in log:
        return "oom"
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


def timing(runs, out: str, rounds: int, reps: int, iters: int, timeout: int):
    d = RESULTS / out
    d.mkdir(parents=True, exist_ok=True)
    status = d / "status.jsonl"
    (d / "order.json").write_text(json.dumps([r[:3] for r in runs], indent=1))
    for n, (cid, backend, seed) in enumerate(runs, 1):
        case = KC.case(cid)
        tag = f"{cid}_{backend}_s{seed}_run{n:02d}"
        rc, wall = run(bench_args(case, backend, seed, rounds, reps, iters) + ["--out", f"{out}/{tag}"], d / f"{tag}.log",
                       timeout)
        log = (d / f"{tag}.log").read_text(errors="replace")
        st = classify(rc, log)
        tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        record(status, dict(tag=tag, case=cid, backend=backend, graph_seed=seed, K=case["K"], C=case["C"], S=case["S"],
                            R=case["R"], H=case["H"], status=st, exit=rc, wall_s=wall, json=f"{tag}.json" if st == "measured"
                            else None, summary=tail[0][:400], log_tail=log.splitlines()[-25:] if st != "measured" else None))


def order(entries, seed=ORDER_SEED):
    rng = random.Random(seed)
    body = list(entries)
    rng.shuffle(body)
    mid = len(body) // 2
    base = ("X01", "native", 0)
    return [base] + body[:mid] + [base] + body[mid:] + [base]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "confirm", "coverage"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--cases", default="", help="confirm: the selected case ids")
    ap.add_argument("--backends", default="", help="restrict to these backends")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()
    KC.check_manifest()
    seeds = [int(s) for s in a.seeds.split(",")]
    ids = [c["id"] for c in KC.cases()] + ["X01"]
    if a.only:
        ids = a.only.split(",")
    if a.mode == "confirm":
        ids = a.cases.split(",")
    keep = set(a.backends.split(",")) if a.backends else None
    if a.mode in ("screen", "confirm"):
        entries = [(cid, be, s) for cid in ids for be in backends(KC.case(cid)) for s in seeds
                   if keep is None or be in keep]
        if a.mode == "screen":
            entries = [e for e in entries if not (e[0] == "X01" and e[1] == "native")]
            runs = entries if a.no_baseline else order(entries)
            timing(runs, a.out, 2, 2, 2, a.timeout)
        else:
            runs = order(entries, ORDER_SEED + 1) if not a.no_baseline else entries
            timing(runs, a.out, 4, 2, 3, a.timeout)
    else:
        d = RESULTS / a.out
        d.mkdir(parents=True, exist_ok=True)
        status = d / "status.jsonl"
        for cid in ids:
            case = KC.case(cid)
            graph_be = "native" if (case["K"] == 8 and case["C"] <= 64) else "kgen"
            for s in seeds:
                tag = f"{cid}_s{s}"
                cmd = [sys.executable, str(HERE / "kcsr_coverage.py"), "--case", cid, "--backend", graph_be,
                       "--graph-seed", str(s), "--out", str(d / f"{tag}.json")]
                rc, wall = run(cmd, d / f"{tag}.log", a.timeout)
                log = (d / f"{tag}.log").read_text(errors="replace")
                st = classify(rc, log)
                tail = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
                record(status, dict(tag=tag, case=cid, graph_seed=s, status=st, exit=rc, wall_s=wall,
                                    json=f"{tag}.json" if st == "measured" else None, summary=tail[0][:400],
                                    log_tail=log.splitlines()[-25:] if st != "measured" else None))


if __name__ == "__main__":
    main()
