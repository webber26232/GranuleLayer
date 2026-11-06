"""Timing and memory of the reducers (docs/experiments/learned_sliding_window_reduction.md, performance section).

One JOB (fresh process) = one (H, reducer, M, variant): packed sparse base, warm-up (compilation and allocation first),
ROUNDS rounds of SAMPLES complete training steps with canonical restore between rounds.  A step = forward + readout +
loss y.float().square().mean() + backward + FusedAdamW (W, b, P: BF16 moments, stochastic rounding) + torch AdamW
(V, d, c: FP32).  Phases come from CUDA events; memory peaks from a separate pass (forward / backward / full step).
Compilation and graph construction are excluded from steady-state timing and reported separately.  CUDA Graph mode is not
measured: per-call scratch tensors have fresh addresses, so these layers do not support capture (eager only, for every
model).

    python learned_reduction_perf.py job --root R --H 10485760 --model A4 --batch 8192 --out job.json [variant flags]
    python learned_reduction_perf.py session --plan plan.json --root R --out DIR [--order-seed N]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import learned_reduction_common as C
import torch

ROUNDS, SAMPLES, WARMUP = 3, 8, 3
INPUT_SEED = 1234
PHASES = dict(sparse_and_reduction=("fwd.zero", "fwd.kernel"), compact_and_readout=("fwd.kernel", "readout.end"),
              loss_and_dense_backward=("readout.end", "bwd.begin"), dx=("bwd.dx_init", "bwd.dx"), dw_db=("bwd.dx", "bwd.dw"),
              dp=("bwd.dw", "bwd.dp"), fused_optimizer=("opt.begin", "opt.fused"), dense_optimizer=("opt.fused", "opt.dense"),
              zero_dense=("opt.dense", "opt.zero"), forward=("step.begin", "readout.end"), backward=("readout.end", "bwd.end"),
              optimizer=("bwd.end", "opt.zero"), step=("step.begin", "opt.zero"))


def job(a):
    from sparse_pooling.backends.cuda import runtime as R
    from sparse_pooling.experimental import learned_reduction as LR
    from sparse_pooling.optim import FusedAdamW

    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    t0 = time.perf_counter()
    base = C.get_base(a.root, a.H, a.graph, dx_variant=a.dx)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    cfgs = LR.core_configs(a.H, L=a.L, s=a.s, epilogue=a.epilogue, dp_mode=a.dp_mode)
    cfg = cfgs[a.model]
    t0 = time.perf_counter()
    kw = dict(wpt=a.wpt, dx_group_warps=a.dx_wg, dx_tiling=(a.dx_mt, a.dx_nw, a.dx_reload)) if cfg.kind == "learned" else {}
    model = LR.ReductionModel(base, cfg, output_dim=C.G, seed=a.seed, **kw)
    model.sparse.dx_variant = a.dx
    gx = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    x = torch.randn(a.batch, C.D, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)
    if cfg.kind != "group_mean" or cfg.dense and cfg.dense_init != "identity":
        model.gamma.fill_(LR.calibrate_gamma(model, x.detach()))
    if model.reducer is not None:
        model.reducer._exec()
    torch.cuda.synchronize()
    model_s = time.perf_counter() - t0
    snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    fused = FusedAdamW(model.sparse_parameters() + model.reducer_parameters(), lr=1e-4, moments="bf16",
                       stochastic_rounding=True)
    dense_params = model.dense_parameters()
    dense = torch.optim.AdamW(dense_params, lr=1e-3, weight_decay=0.0, fused=True) if dense_params else None

    def reset():
        with torch.no_grad():
            for k, v in model.named_parameters():
                v.copy_(snap[k])
        model.zero_main_grad()
        fused.state.clear()
        fused._seed = 0
        if dense is not None:
            dense.state.clear()
            dense.zero_grad(set_to_none=True)
        x.grad = None

    def step():
        y = model(x)
        y.float().square().mean().backward()
        LR._mark("bwd.end")
        LR._mark("opt.begin")
        fused.step()
        LR._mark("opt.fused")
        if dense is not None:
            dense.step()
        LR._mark("opt.dense")
        if dense is not None:
            dense.zero_grad(set_to_none=True)
        LR._mark("opt.zero")
        x.grad = None

    reset()
    t1 = time.perf_counter()
    first = None
    for _ in range(WARMUP):
        step()
        torch.cuda.synchronize()
        first = first if first is not None else time.perf_counter() - t1
    samples, rounds = [], []
    for r in range(ROUNDS):
        reset()
        rs = []
        for s in range(SAMPLES):
            ev0 = torch.cuda.Event(enable_timing=True)
            R.PHASE_EVENTS = []
            torch.cuda.synchronize()
            h0 = time.perf_counter()
            R.PHASE_EVENTS.append(("step.begin", ev0))
            ev0.record()
            step()
            h1 = time.perf_counter()
            torch.cuda.synchronize()
            h2 = time.perf_counter()
            marks = {}
            for name, e in R.PHASE_EVENTS:
                marks.setdefault(name, e)
            R.PHASE_EVENTS = None
            rec = dict(host_enqueue=(h1 - h0) * 1e3, wall=(h2 - h0) * 1e3)
            for k, (m0, m1) in PHASES.items():
                if m0 in marks and m1 in marks:
                    rec[k] = marks[m0].elapsed_time(marks[m1])
            rs.append(rec)
        from provenance import hardware_state

        rounds.append(dict(samples=rs, hardware_after=hardware_state()))
        samples += rs
    keys = sorted({k for s in samples for k in s})
    med = {k: statistics.median(s[k] for s in samples if k in s) for k in keys}
    disp = {k: dict(min=min(s[k] for s in samples if k in s), max=max(s[k] for s in samples if k in s),
                    q1=statistics.quantiles([s[k] for s in samples if k in s], n=4)[0],
                    q3=statistics.quantiles([s[k] for s in samples if k in s], n=4)[2]) for k in keys}
    # memory pass
    reset()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    steady0 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = model(x)
    torch.cuda.synchronize()
    peak_fwd = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y.float().square().mean().backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    del y
    torch.cuda.reset_peak_memory_stats()
    fused.step()
    if dense is not None:
        dense.step()
    torch.cuda.synchronize()
    peak_opt = torch.cuda.max_memory_allocated()
    if dense is not None:
        dense.zero_grad(set_to_none=True)
    x.grad = None
    steady1 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak_step = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    rec = dict(schema="granule.lsw_perf/1", H=a.H, L=a.L, s=a.s, model=a.model, batch=a.batch, graph=a.graph, seed=a.seed,
               rep=a.rep, config=cfg.describe(), dx=a.dx, dx_wg=a.dx_wg, dx_mt=a.dx_mt, dx_nw=a.dx_nw, dx_reload=a.dx_reload, wpt=a.wpt, tf32=bool(a.tf32), rounds=ROUNDS, samples_per_round=SAMPLES,
               warmup=WARMUP, cold=dict(load_base_s=build_s, model_and_work_lists_s=model_s, first_step_wall_s=first),
               median_ms=med, dispersion_ms=disp, memory=dict(
                   steady_before_step_bytes=steady0, steady_with_optimizer_bytes=steady1, peak_forward_bytes=peak_fwd,
                   peak_backward_bytes=peak_bwd, peak_optimizer_bytes=peak_opt, peak_full_step_bytes=peak_step,
                   peak_reserved_bytes=peak_res, unique_storage=model.memory_report(),
                   fused_state_bytes=fused.state_bytes(), m_by_h_bf16_bytes=a.batch * a.H * 2),
               weights=model.weight_counts(), fingerprints=model.fingerprints(),
               explain=model.explain() if not a.brief else None, last_info={k: v for k, v in model.sparse.last_info.items()
                                                                            if k.startswith(("reduction", "kgen_dx", "kgen_dw", "kgen_fwd"))},
               rounds_detail=rounds, env=C.env_record(),
               step_definition="forward + readout + y.float().square().mean() + backward + FusedAdamW(lr 1e-4, BF16 moments, "
                               "SR) + torch AdamW fused(lr 1e-3); canonical restore + fresh moments per round; eager")
    if model.reducer is not None:
        rec["kernel"] = model.reducer.execution_record(include_attrs=True)
    C.atomic_json(a.out, rec)
    print(f"[H{a.H} {a.model} M{a.batch} {a.epilogue} {a.dp_mode} wpt{a.wpt} dx mt{a.dx_mt} nw{a.dx_nw} wg{a.dx_wg} r{a.dx_reload}] step {med['step']:.1f} ms | "
          + " ".join(f"{k} {med[k]:.1f}" for k in ("sparse_and_reduction", "compact_and_readout", "loss_and_dense_backward",
                                                   "dx", "dw_db", "dp", "fused_optimizer", "dense_optimizer") if k in med)
          + f" | peak {peak_step / 2 ** 30:.2f} GiB", flush=True)


def session(a):
    plan = json.loads(Path(a.plan).read_text())
    jobs = plan["jobs"]
    if a.order_seed is not None:
        random.Random(a.order_seed).shuffle(jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.atomic_json(out / "plan_ordered.json", dict(plan=plan.get("name"), order_seed=a.order_seed, jobs=jobs))
    for i, j in enumerate(jobs):
        name = "_".join(f"{k}{v}" for k, v in sorted(j.items()))
        f = out / f"{name}.json"
        if f.exists():
            print(f"{i + 1}/{len(jobs)} {f.name} done", flush=True)
            continue
        cmd = [sys.executable, __file__, "job", "--root", a.root, "--out", str(f)]
        for k, v in j.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        print(f"{i + 1}/{len(jobs)} rc {r.returncode} {time.perf_counter() - t0:.0f}s {r.stdout.strip()[-300:]}"
              + (f" ERR {r.stderr.strip()[-400:]}" if r.returncode else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--root", required=True)
    j.add_argument("--H", type=int, required=True)
    j.add_argument("--L", type=int, default=10_240)
    j.add_argument("--s", type=int, default=2_048)
    j.add_argument("--model", required=True)
    j.add_argument("--batch", type=int, default=8192)
    j.add_argument("--graph", type=int, default=0)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--epilogue", default="mma_split")
    j.add_argument("--dp-mode", default="fused")
    j.add_argument("--wpt", type=int, default=2)
    j.add_argument("--dx-wg", type=int, default=2)
    j.add_argument("--dx-mt", type=int, default=2)
    j.add_argument("--dx-nw", type=int, default=4)
    j.add_argument("--dx-reload", type=int, default=0)
    j.add_argument("--dx", default="mt2_nw4_wg2")
    j.add_argument("--tf32", type=int, default=0)
    j.add_argument("--brief", type=int, default=0)
    j.add_argument("--rep", type=int, default=0)
    j.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--plan", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--order-seed", type=int, default=None)
    a = ap.parse_args()
    job(a) if a.cmd == "job" else session(a)


if __name__ == "__main__":
    main()
