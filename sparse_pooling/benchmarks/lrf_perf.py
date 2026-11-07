"""Timing and memory of A0 / A2 / A3mean for the A3mean follow-up.  Definitions differ from the historical learned_reduction_perf.py:

* the canonical parameter snapshot is on the CPU, so no benchmark-owned parameter copy is in any GPU total;
* optimizer moments exist before steady-state timing; between rounds they are zeroed in place, never reallocated in a timed region;
* peak counters are reset after setup: construction and first step are reported separately from the steady training peak;
* A3mean is measured with its mean initialization;
* an effective update (M_phys * accum_steps rows, one optimizer update) is the timed unit, and the accumulated calls run uninterrupted.

kind "reducer": x -> model -> y.float().square().mean(), no task layers.  kind "task": x0 -> proj -> model -> head -> MSE on a Compose
target, the real training step.  One JOB (fresh process) = one (kind, model, M_phys, accum, dX request): at least ROUNDS rounds of
SAMPLES warmed samples on the same canonical blocks.  Data is on the GPU before any timed region.

    python lrf_perf.py job --root R --kind task --model A3mean --m-phys 512 --accum 16 --out job.json
    python lrf_perf.py session --plan plan.json --root R --out DIR --order-seed N
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
import learned_reduction_tasks as T
import lrf_stream as S
import torch

SCHEMA = "granule.lrf_perf/1"
ROUNDS, SAMPLES, WARMUP = 5, 8, 3
PHASES = dict(sparse_and_reduction=("fwd.zero", "fwd.kernel"), finalize_and_readout=("fwd.kernel", "readout.end"),
              loss_and_dense_backward=("readout.end", "bwd.begin"), gradient_setup=("bwd.begin", "bwd.dx_init"), dx=("bwd.dx_init", "bwd.dx"),
              dw_db_dp=("bwd.dx", "bwd.dp"), dw_db=("bwd.dx", "bwd.dw"), input_gradient_tail=("bwd.dw", "bwd.end"),
              forward=("step.begin", "readout.end"), backward=("readout.end", "bwd.end"))
OPT_PHASES = dict(fused_optimizer=("opt.begin", "opt.fused"), dense_optimizer=("opt.fused", "opt.dense"), zero_dense=("opt.dense", "opt.zero"))


def storage_categories(model, proj, head, fused, small, live) -> dict:
    """Steady GPU storage, deduplicated by storage across and within categories (views and aliases count once)."""
    from sparse_pooling.memory_accounting import unique_storage

    sp = model.sparse
    groups = dict(
        model_parameters=[p for _, p in model.named_parameters()],
        common_task_parameters=list(proj.parameters()) + list(head.parameters()) if proj is not None else [],
        main_grad=[t for t in (getattr(p, "main_grad", None) for _, p in model.named_parameters()) if t is not None],
        fused_optimizer_state=[v for st in fused.state.values() for v in st.values() if torch.is_tensor(v)],
        dense_optimizer_state=[v for st in small.state.values() for v in st.values() if torch.is_tensor(v)] if small is not None else [],
        immutable_metadata=[sp.indices, sp.sg_feat, sp.kg_buckets, sp.kg_spec, sp.counts, sp.counts_f, model.gamma],
        runtime_work_lists=[],
        input_and_target=[t for t in live if t is not None],
    )
    if sp._rt is not None:
        s = sp._rt.sg
        groups["runtime_work_lists"] += [s.items, s.chunks, s.tasks, s.dw_chunks]
    if model.reducer is not None:
        groups["immutable_metadata"] += [model.reducer.w0_blocks, model.reducer.member, model.reducer.reduction_spec]
        if model.reducer._ex is not None:
            groups["runtime_work_lists"] += model.reducer._ex.tensors()
    seen, out = set(), {}
    for name, ts in groups.items():
        fresh = []
        for t in ts:
            k = t.untyped_storage().data_ptr()
            if k not in seen and t.untyped_storage().nbytes() > 0:
                seen.add(k)
                fresh.append(t)
        out[name] = unique_storage(fresh)["bytes"]
    return out


def job(a):
    from sparse_pooling.backends.cuda import runtime as R
    from sparse_pooling.experimental import learned_reduction as LR
    from sparse_pooling.optim import FusedAdamW

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    base = C.get_base(a.root, a.H, a.seed, init_seed=C.INIT_SEED + a.seed, dx_variant=a.dx)
    cfg = LR.core_configs(a.H, s=a.s)[a.model]
    model = LR.ReductionModel(base, cfg, output_dim=C.G, seed=a.seed)
    del base
    model.sparse.dx_variant = a.dx
    if a.kind == "task":
        task = T.Compose(dev)
        torch.manual_seed(29 + a.seed)
        proj = torch.nn.Linear(task.D0, C.D, device=dev)
        head = torch.nn.Linear(C.G, T.OUT, device=dev)
    else:
        task, proj, head = None, None, None
    D0 = task.D0 if task else C.D
    cal = S.block("calibration", a.seed, 0, D0).to(dev)
    if cfg.dense:
        with torch.no_grad():
            model.gamma.fill_(LR.calibrate_gamma(model, (proj(cal) if proj else cal).to(torch.bfloat16)))
    del cal
    # the same canonical training blocks for every model and execution mode
    per_call = a.m_phys // S.BLOCK
    micro = []
    for k in range(a.accum):
        x0 = S.blocks("train", a.seed, k * per_call, per_call, D0)
        micro.append(x0.to(dev))
    target = [task(x) for x in micro] if task else [None] * a.accum
    if not task:
        micro = [x.to(torch.bfloat16).requires_grad_(True) for x in micro]
    groups = [dict(params=model.sparse_parameters(), lr=1e-4)]
    if model.reducer_parameters():
        groups.append(dict(params=model.reducer_parameters(), lr=1e-4))
    fused = FusedAdamW(groups, lr=1e-4, moments="bf16", stochastic_rounding=True)
    small_params = model.dense_parameters() + (list(proj.parameters()) + list(head.parameters()) if proj else [])
    small = torch.optim.AdamW(small_params, lr=1e-3, weight_decay=0.0, fused=True) if small_params else None
    named = list(model.named_parameters()) + ([(f"proj.{k}", v) for k, v in proj.named_parameters()] + [(f"head.{k}", v) for k, v in head.named_parameters()] if proj else [])
    snap = {k: v.detach().cpu().clone() for k, v in named}  # on the host: not part of any GPU number below
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    construction_peak = torch.cuda.max_memory_allocated()

    def call_loss(k):
        if task:
            y = model(proj(micro[k]).to(torch.bfloat16))
            return (head(y.float()) - target[k]).square().mean() / a.accum
        return model(micro[k]).float().square().mean() / a.accum

    def calls(log=None):
        for k in range(a.accum):
            if log is not None:
                R.PHASE_EVENTS = []
                log.append(R.PHASE_EVENTS)
                ev = torch.cuda.Event(enable_timing=True)
                R.PHASE_EVENTS.append(("step.begin", ev))
                ev.record()
            call_loss(k).backward()
            LR._mark("bwd.end")
            if not task:
                micro[k].grad = None

    def optimize():
        LR._mark("opt.begin")
        fused.step()
        LR._mark("opt.fused")
        if small is not None:
            small.step()
        LR._mark("opt.dense")
        if small is not None:
            small.zero_grad(set_to_none=True)
        LR._mark("opt.zero")

    def restore():
        with torch.no_grad():
            for k, v in named:
                v.copy_(snap[k].to(v.device, non_blocking=False))
            model.zero_main_grad()
            for st in fused.state.values():
                st["step"] = 0
                st["m"].zero_()
                st["v"].zero_()
            if small is not None:
                small.zero_grad(set_to_none=True)
                for st in small.state.values():
                    st["step"].zero_()
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()
        fused._seed = 0

    # first effective update: moments allocate here, reported apart from the steady peak
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    calls()
    optimize()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t1
    first_peak = torch.cuda.max_memory_allocated()
    for _ in range(WARMUP - 1):
        calls()
        optimize()
    torch.cuda.synchronize()
    samples, rounds = [], []
    for r in range(a.rounds):
        restore()
        for _ in range(WARMUP):  # warmed, with a canonical state at the start of each round's timed samples
            calls()
            optimize()
        restore()
        torch.cuda.synchronize()
        rs = []
        for _s in range(SAMPLES):
            torch.cuda.synchronize()
            h0 = time.perf_counter()
            log = []
            calls(log)
            R.PHASE_EVENTS = []
            ev_op = torch.cuda.Event(enable_timing=True)
            ev_op.record()
            log_op = R.PHASE_EVENTS
            optimize()
            R.PHASE_EVENTS = None
            h1 = time.perf_counter()
            torch.cuda.synchronize()
            h2 = time.perf_counter()
            rec = dict(host_enqueue=(h1 - h0) * 1e3, effective_update=(h2 - h0) * 1e3)
            for evs in log:
                m = {}
                for name, e in evs:
                    m.setdefault(name, e)
                for key, (p0, p1) in PHASES.items():
                    if p0 in m and p1 in m:
                        rec[key] = rec.get(key, 0.0) + m[p0].elapsed_time(m[p1])
                if "bwd.dp" in m:
                    rec["input_gradient_tail"] = rec.get("input_gradient_tail", 0.0) - m["bwd.dw"].elapsed_time(m["bwd.dp"])
                rec["calls_total"] = rec.get("calls_total", 0.0) + m["step.begin"].elapsed_time(m["bwd.end"])
            mo = {}
            for name, e in log_op:
                mo.setdefault(name, e)
            for key, (p0, p1) in OPT_PHASES.items():
                rec[key] = mo[p0].elapsed_time(mo[p1])
            rec["optimizer"] = mo["opt.begin"].elapsed_time(mo["opt.zero"])
            rs.append(rec)
        from provenance import hardware_state

        rounds.append(dict(samples=rs, hardware_after=hardware_state(), round_median=statistics.median(s["effective_update"] for s in rs)))
        samples += rs
    keys = sorted({k for s in samples for k in s})
    med = {k: statistics.median(s[k] for s in samples if k in s) for k in keys}
    disp = {}
    for k in keys:
        v = [s[k] for s in samples if k in s]
        q = statistics.quantiles(v, n=4)
        disp[k] = dict(min=min(v), max=max(v), q1=q[0], q3=q[2])

    # memory pass: moments allocated, one microbatch live (training holds one at a time; data otherwise on the host)
    restore()
    if not task:
        for k in range(a.accum):
            micro[k].grad = None
    host = [x.detach().cpu() for x in micro]
    host_t = [t.cpu() if t is not None else None for t in target]
    del micro, target
    micro = [None] * a.accum
    target = [None] * a.accum
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    steady_no_data = torch.cuda.memory_allocated()
    per_call_peak = []
    torch.cuda.reset_peak_memory_stats()
    peaks = dict(forward=0, backward=0)
    ordinary = 0
    live_last = None
    for k in range(a.accum):
        micro[k] = host[k].to(dev)
        if not task:
            micro[k].requires_grad_(True)
        target[k] = host_t[k].to(dev) if host_t[k] is not None else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        loss = call_loss(k)
        torch.cuda.synchronize()
        peaks["forward"] = max(peaks["forward"], torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        peaks["backward"] = max(peaks["backward"], torch.cuda.max_memory_allocated())
        del loss
        if not task:
            micro[k].grad = None
        live_last = (micro[k], target[k])
        if k < a.accum - 1:
            micro[k], target[k] = None, None
    ordinary = sum(p.grad.numel() * p.grad.element_size() for p in small_params if p.grad is not None)
    torch.cuda.reset_peak_memory_stats()
    optimize()
    torch.cuda.synchronize()
    peaks["optimizer"] = torch.cuda.max_memory_allocated()
    steady = torch.cuda.memory_allocated()
    cats = storage_categories(model, proj, head, fused, small, list(live_last))
    attributed = sum(cats.values())
    cats_mib = {k: v / 2 ** 20 for k, v in cats.items()}
    memory = dict(steady_allocated_bytes=steady, steady_without_data_bytes=steady_no_data, peak_forward_bytes=peaks["forward"], peak_backward_bytes=peaks["backward"],
                  peak_optimizer_bytes=peaks["optimizer"], steady_training_peak_bytes=max(peaks.values()),
                  transient_above_steady_bytes=max(peaks.values()) - steady, reserved_bytes=torch.cuda.memory_reserved(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  construction_peak_bytes=construction_peak, first_step_peak_bytes=first_peak, categories_bytes=cats, categories_mib=cats_mib,
                  ordinary_gradients_bytes=ordinary, benchmark_only_gpu_bytes=0, unattributed_steady_bytes=steady - attributed,
                  m_by_h_bf16_bytes=a.m_phys * a.H * 2)
    rows = a.m_phys * a.accum
    rec = dict(schema=SCHEMA, kind=a.kind, H=a.H, s=a.s, model=a.model, M_phys=a.m_phys, accum_steps=a.accum, M_eff=rows, seed=a.seed, rep=a.rep, dx_request=a.dx,
               dx_validated_baseline=a.dx == "mt4_nw8_wg4", status="experimental, unvalidated" if model.reducer is not None or a.dx != "mt4_nw8_wg4" else "production-validated tiling",
               config=cfg.describe(), rounds=a.rounds, samples_per_round=SAMPLES, warmup=WARMUP, median_ms=med, dispersion_ms=disp, memory=memory,
               rows_per_second=rows / (med["effective_update"] / 1e3), updates_per_second=1e3 / med["effective_update"],
               round_medians_ms=[r["round_median"] for r in rounds], cold=dict(build_s=build_s, first_update_s=first_s), weights=model.weight_counts(), gamma=float(model.gamma),
               fingerprints=model.fingerprints(), init=cfg.init if model.reducer is not None else None,
               dx_resolved={k: v for k, v in model.sparse.last_info.items() if k in ("kgen_dx", "reduction_backward")}, gamma_read="device-side multiplication",
               rounds_detail=rounds, env=C.env_record(),
               definition="effective update: accum_steps x (forward, backward of mean loss / accum_steps), one FusedAdamW and one AdamW step; eager; moments allocated; restore outside timing; host snapshot")
    if model.reducer is not None:
        rec["kernel"] = model.reducer.execution_record(include_attrs=True)
    C.atomic_json(a.out, rec)
    print(f"[{a.kind} {a.model} M{a.m_phys}x{a.accum} {a.dx}] update {med['effective_update']:.1f} ms dX {med['dx']:.1f} peak {max(peaks.values()) / 2 ** 30:.3f} GiB steady {steady / 2 ** 30:.3f} "
          f"unattributed {(steady - attributed) / 2 ** 20:.1f} MiB", flush=True)


def session(a):
    plan = json.loads(Path(a.plan).read_text())
    jobs = plan["jobs"]
    random.Random(a.order_seed).shuffle(jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.atomic_json(out / "plan_ordered.json", dict(order_seed=a.order_seed, jobs=jobs))
    for i, j in enumerate(jobs):
        name = "_".join(f"{k}{v}" for k, v in sorted(j.items()))
        f = out / f"{name}.json"
        if f.exists():
            continue
        cmd = [sys.executable, __file__, "job", "--root", a.root, "--out", str(f)]
        for k, v in j.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        print(f"{i + 1}/{len(jobs)} rc {r.returncode} {time.perf_counter() - t0:.0f}s {r.stdout.strip()[-260:]}" + (f" ERR {r.stderr.strip()[-400:]}" if r.returncode else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--root", required=True)
    j.add_argument("--kind", default="task", choices=["task", "reducer"])
    j.add_argument("--H", type=int, default=10_485_760)
    j.add_argument("--s", type=int, default=2_048)
    j.add_argument("--model", required=True)
    j.add_argument("--m-phys", type=int, default=512)
    j.add_argument("--accum", type=int, default=1)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--dx", default="mt2_nw4_wg2")
    j.add_argument("--rounds", type=int, default=ROUNDS)
    j.add_argument("--rep", type=int, default=0)
    j.add_argument("--out", required=True)
    s = sub.add_parser("session")
    s.add_argument("--plan", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--order-seed", type=int, default=20261107)
    a = ap.parse_args()
    job(a) if a.cmd == "job" else session(a)


if __name__ == "__main__":
    main()
