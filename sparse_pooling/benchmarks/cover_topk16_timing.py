"""Same-session performance acceptance of the named Cover-TopK/16 layer: legacy dX vs the automatically selected K16 dX
tiling (docs/validation/cover_topk16_rtx3090_dx.md section 5).

One JOB (fresh process) = one (graph seed, dx_variant request, M): the full model is built from the user-supplied K64
envelope checkpoint (normalization "none": the timing factories of the research round masked the checkpoint values
without restoration, identically for every compared backend), compiled / warmed up, then ROUNDS rounds of SAMPLES
complete layer training steps.  A step = forward (x [M, 1024] BF16, requires_grad) + loss y.float().square().mean() +
backward (dX, dW / db into FP32 main_grad) + FusedAdamW (BF16 moments, stochastic rounding, fused main_grad zeroing).
Before every round the canonical start state is restored: packed weights / bias, zero main_grad, empty (fresh)
optimizer moments, rounding counter 0.  Phases come from CUDA events (the kgen layer's phase marks + optimizer
events); host enqueue = CPU time to issue the step, wall = issue + synchronize.  Clocks / power / temperatures are
read before and after each round; allocation peaks after warm-up.

The SESSION driver runs every job of the declared plan in a fresh subprocess, in a random order drawn from
``--order-seed`` (legacy and auto interleaved), and writes the per-job records plus a summary: per graph the median
complete step of legacy and auto and their ratio, the graph mean, the M 512 / M 64 guards, and the start-fingerprint
check (identical support / canonical start state for both backends of a graph).

    python cover_topk16_timing.py session --checkpoints g0.pt,g1.pt,g2.pt --out DIR [--order-seed 20261106]
    python cover_topk16_timing.py job --checkpoint g0.pt --graph 0 --dx auto --batch 8192 --out job.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import common  # noqa: F401  (sys.path)
import torch

ROUNDS, SAMPLES, WARMUP = 3, 8, 3
INPUT_SEED = 1234


def job(a):
    from provenance import hardware_state, provenance
    from sparse_pooling import CoverTopK16Layer
    from sparse_pooling.backends.cuda import runtime as R
    from sparse_pooling.experimental import canonical as Cn
    from sparse_pooling.optim import FusedAdamW

    t0 = time.perf_counter()
    lay = CoverTopK16Layer.from_checkpoint(a.checkpoint, normalization="none", device="cuda", dx_variant=a.dx)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    lay.enable_main_grad()
    w0, b0 = lay.weight.detach().clone(), lay.bias.detach().clone()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16", stochastic_rounding=True)
    gx = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    x = torch.randn(a.batch, lay.layout.input_dim, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)

    def reset():
        with torch.no_grad():
            lay.weight.copy_(w0)
            lay.bias.copy_(b0)
        lay.zero_main_grad()
        opt.state.clear()
        opt._seed = 0
        x.grad = None

    def step(ev=None):
        if ev is not None:
            ev[0].record()
        y = lay(x)
        if ev is not None:
            ev[1].record()
        y.float().square().mean().backward()
        if ev is not None:
            ev[2].record()
        opt.step()
        if ev is not None:
            ev[3].record()
        x.grad = None

    reset()
    start = dict(canonical=Cn.digest(lay)["sha"], mask=Cn.mask_digest(lay)["sha"],
                 support=lay.metadata["support_fingerprint"], main_grad_zero=bool(lay.weight.main_grad.abs().max() == 0),
                 moments_empty=len(opt.state) == 0, sr_counter=opt._seed)
    t1 = time.perf_counter()
    first = None
    for i in range(WARMUP):
        step()
        torch.cuda.synchronize()
        if first is None:
            first = time.perf_counter() - t1
    reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, rounds = [], []
    for r in range(ROUNDS):
        reset()
        hw0 = hardware_state()
        rs = []
        for s in range(SAMPLES):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            R.PHASE_EVENTS = []
            torch.cuda.synchronize()
            h0 = time.perf_counter()
            step(ev)
            h1 = time.perf_counter()
            torch.cuda.synchronize()
            h2 = time.perf_counter()
            marks = {}
            for name, e in R.PHASE_EVENTS:
                marks.setdefault(name, e)
            R.PHASE_EVENTS = None

            def el(a_, b_):
                return a_.elapsed_time(b_)

            rec = dict(step=el(ev[0], ev[3]), fwd=el(ev[0], ev[1]), loss_and_bwd=el(ev[1], ev[2]), optimizer=el(ev[2], ev[3]),
                       host_enqueue=(h1 - h0) * 1e3, wall=(h2 - h0) * 1e3)
            if "bwd.dx_init" in marks and "bwd.dx" in marks:
                rec["dx"] = el(marks["bwd.dx_init"], marks["bwd.dx"])
            if "bwd.dx" in marks and "bwd.dw" in marks:
                rec["dw_db"] = el(marks["bwd.dx"], marks["bwd.dw"])
            if "fwd.zero" in marks and "fwd.kernel" in marks:
                rec["fwd_kernel"] = el(marks["fwd.zero"], marks["fwd.kernel"])
            rs.append(rec)
        rounds.append(dict(samples=rs, hardware_before=hw0, hardware_after=hardware_state()))
        samples += rs
    med = {k: statistics.median(s[k] for s in samples if k in s) for k in samples[0]}
    out = dict(schema="granule.cover_topk16_timing/1", graph=a.graph, request=a.dx, batch=a.batch,
               checkpoint=Path(a.checkpoint).name, rounds=ROUNDS, samples_per_round=SAMPLES, warmup=WARMUP,
               build_seconds=build_s, first_steps_wall_s=first, median_ms=med, start_state=start,
               executed=lay.last_info.get("kgen_dx"), work_lists=lay.last_info.get("kgen_work_lists"),
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(), memory_unique_storage=lay.memory_report(),
               optimizer_state_bytes=opt.state_bytes(), rounds_detail=rounds,
               build=lay._runtime().backend.build_info(), profile=lay._runtime().selection.summary(),
               provenance=provenance(Path(common.__file__).resolve().parent.parent),
               step_definition="forward + loss y.float().square().mean() + backward (dX, dW / db -> FP32 main_grad) + "
                               "FusedAdamW(lr 1e-4, BF16 moments, SR); canonical restore + fresh moments per round")
    for k in ("library",):
        out["build"].pop(k, None)  # absolute cache path: machine specific
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"[g{a.graph} {a.dx} M{a.batch}] step {med['step']:.1f} ms (dx {med.get('dx', 0):.1f}) executed "
          f"{out['executed']['variant']['name']}", flush=True)


def plan(checkpoints):
    jobs = []
    for g, ck in enumerate(checkpoints):
        for dx in ("legacy", "auto"):
            jobs.append(dict(graph=g, checkpoint=ck, dx=dx, batch=8192))
    for M in (512, 64):
        for dx in ("legacy", "auto"):
            jobs.append(dict(graph=0, checkpoint=checkpoints[0], dx=dx, batch=M))
    return jobs


def session(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cks = a.checkpoints.split(",")
    jobs = plan(cks)
    random.Random(a.order_seed).shuffle(jobs)
    (out / "plan.json").write_text(json.dumps(dict(order_seed=a.order_seed, jobs=[dict(j, checkpoint=Path(
        j["checkpoint"]).name) for j in jobs]), indent=1))
    status = []
    for i, j in enumerate(jobs):
        f = out / f"job{i:02d}_g{j['graph']}_{j['dx']}_M{j['batch']}.json"
        cmd = [sys.executable, __file__, "job", "--checkpoint", j["checkpoint"], "--graph", str(j["graph"]), "--dx",
               j["dx"], "--batch", str(j["batch"]), "--out", str(f)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / (f.stem + ".log")).write_text(r.stdout + r.stderr)
        status.append(dict(job=f.name, rc=r.returncode, seconds=time.perf_counter() - t0))
        print(f"{i + 1}/{len(jobs)} {f.name} rc {r.returncode} {r.stdout.strip()[-120:]}", flush=True)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    summarize(out)


def summarize(out: Path):
    recs = [json.loads(f.read_text()) for f in sorted(out.glob("job*.json"))]
    by = {(r["graph"], r["batch"], r["request"]): r for r in recs}
    rows, ok = [], True
    for (g, M) in sorted({(r["graph"], r["batch"]) for r in recs}):
        L, A = by.get((g, M, "legacy")), by.get((g, M, "auto"))
        if not (L and A):
            ok = False
            continue
        same_start = all(L["start_state"][k] == A["start_state"][k] for k in ("canonical", "mask", "support"))
        rows.append(dict(graph=g, batch=M, legacy_ms=L["median_ms"]["step"], auto_ms=A["median_ms"]["step"],
                         ratio=A["median_ms"]["step"] / L["median_ms"]["step"], legacy_dx=L["median_ms"].get("dx"),
                         auto_dx=A["median_ms"].get("dx"), legacy_fwd=L["median_ms"]["fwd"], auto_fwd=A["median_ms"]["fwd"],
                         legacy_dw=L["median_ms"].get("dw_db"), auto_dw=A["median_ms"].get("dw_db"),
                         legacy_opt=L["median_ms"]["optimizer"], auto_opt=A["median_ms"]["optimizer"],
                         legacy_enqueue=L["median_ms"]["host_enqueue"], auto_enqueue=A["median_ms"]["host_enqueue"],
                         legacy_peak_gib=L["peak_allocated_bytes"] / 2 ** 30, auto_peak_gib=A["peak_allocated_bytes"] / 2 ** 30,
                         executed_legacy=L["executed"]["variant"]["name"], executed_auto=A["executed"]["variant"]["name"],
                         auto_status=A["executed"]["variant"]["status"], identical_start=same_start))
        ok &= same_start
    m8 = [r for r in rows if r["batch"] == 8192]
    s = dict(rows=rows, identical_starts=ok)
    if m8:
        s["graph_mean_ratio_M8192"] = statistics.mean(r["ratio"] for r in m8)
        s["graph_mean_legacy_ms_M8192"] = statistics.mean(r["legacy_ms"] for r in m8)
        s["graph_mean_auto_ms_M8192"] = statistics.mean(r["auto_ms"] for r in m8)
        s["acceptance_floor"] = "auto complete step <= 0.95 x legacy (>= 5 % improvement) at M 8192, every graph"
        s["accepted"] = all(r["ratio"] <= 0.95 and r["executed_auto"] == "mt4_nw8_wg4" for r in m8) and ok
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--checkpoint", required=True)
    j.add_argument("--graph", type=int, required=True)
    j.add_argument("--dx", required=True)
    j.add_argument("--batch", type=int, default=8192)
    j.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--checkpoints", required=True, help="comma-separated K64 envelope checkpoints of graphs 0, 1, 2")
    s.add_argument("--order-seed", type=int, default=20261106)
    s.add_argument("--out", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "job":
        job(a)
    elif a.cmd == "session":
        session(a)
    else:
        summarize(Path(a.out))


if __name__ == "__main__":
    main()
