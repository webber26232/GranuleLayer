"""Per-phase / per-kernel timing of a full training step (forward, backward, fused AdamW).

CUDA events are recorded at the phase boundaries inside the native launcher
(``cuda_ops.PHASE_EVENTS``), so dX and dW are timed separately inside the exact
computation that a training step runs.  Each repetition is an independent
group of steps; medians and the spread of the per-repetition medians are reported.

    python sparse_pooling/benchmarks/bench_kernels.py --hidden 100000000 --C 64 --pooling mean
    python sparse_pooling/benchmarks/bench_kernels.py --hidden 100000000 --C 64 --dw-staging indexed

An interval between two events also contains any GPU idle time between the two
launches (CPU launch latency); at H >= 10M every phase is long enough for this
to be negligible, at H = 1M the small phases are upper bounds.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess

import torch

from common import gpu_info, save
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.optim import FusedAdamW

POOLINGS = {"mean": "mean", "max": "max", "lme": {"type": "logmeanexp", "temperature": 0.5},
            "sw": {"type": "softmax_weighted", "beta": 1.0}, "pow2": {"type": "power_mean", "p": 2}}

# (phase, start mark, end mark); marks from cuda_ops plus the step-level ones below
PHASES = [
    ("fwd_layout", "fwd.begin", "fwd.layout"),
    ("fwd_kernel", "fwd.layout", "fwd.kernel"),
    ("fwd_finalize", "fwd.kernel", "fwd.finalize"),
    ("fwd_total", "step.begin", "fwd.end"),
    ("bwd_coef", "bwd.begin", "bwd.coef"),
    ("bwd_layout", "bwd.coef", "bwd.layout"),
    ("bwd_dx_init", "bwd.layout", "bwd.dx_init"),
    ("dx_kernel", "bwd.dx_init", "bwd.dx"),
    ("dw_kernel", "bwd.dx", "bwd.dw"),
    ("bwd_post", "bwd.dw", "bwd.end"),  # dX FP32 -> BF16 conversion, autograd bookkeeping
    ("bwd_total", "fwd.end", "bwd.end"),
    ("optimizer", "bwd.end", "opt.end"),
    ("step", "step.begin", "opt.end"),
]


def gpu_state():
    q = "temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        return dict(zip(q.split(","), map(float, r)))
    except Exception:  # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--pooling", default="mean", choices=list(POOLINGS))
    ap.add_argument("--grad-precision", default="split", choices=["split", "bf16"])
    ap.add_argument("--dw-staging", default="auto", choices=["auto", "full", "indexed"])
    ap.add_argument("--dw-stages", type=int, default=2, choices=[2, 3], help="indexed k_dw cp.async ring depth")
    ap.add_argument("--dx-warps", type=int, default=0, choices=[0, 8, 16, 32], help="0 = library default")
    ap.add_argument("--dw-cfg", type=int, default=-1, help="cuda_ops.DW_CFGS key; -1 = default")
    ap.add_argument("--reps", type=int, default=5, help="independent repetitions")
    ap.add_argument("--iters", type=int, default=3, help="training steps per repetition")
    ap.add_argument("--tag", default="", help="suffix of the result name (e.g. an ablation)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    cuda_ops.DW_INDEXED_STAGES = args.dw_stages
    plan = PhysicalPlan(grad_precision=args.grad_precision, cuda_dw_cfg=args.dw_cfg, cuda_dw_staging=args.dw_staging,
                        cuda_dx_warps=args.dx_warps)
    lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=args.C, pooling=POOLINGS[args.pooling],
                             checkpoint_policy="auto", backend="auto", device="cuda", plan=plan)
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16")
    x = torch.randn(args.batch, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step(events):
        cuda_ops.PHASE_EVENTS = events
        mark = cuda_ops._mark
        try:
            mark("step.begin")
            y = lay(x)
            mark("fwd.end")
            y.float().square().mean().backward()
            mark("bwd.end")
            opt.step()
            mark("opt.end")
        finally:
            cuda_ops.PHASE_EVENTS = None
        x.grad = None

    # warm-up: JIT/nvcc build, Triton compilation, lazy Adam state, allocator growth
    for _ in range(2):
        step(None)
    torch.cuda.synchronize()

    state_before = gpu_state()
    per_rep = []
    samples = {p[0]: [] for p in PHASES}
    peak_alloc = peak_res = 0
    for _ in range(args.reps):
        rep = {p[0]: [] for p in PHASES}
        for _ in range(args.iters):
            torch.cuda.reset_peak_memory_stats()
            ev = []
            step(ev)
            torch.cuda.synchronize()
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
            peak_res = max(peak_res, torch.cuda.max_memory_reserved())
            m = {}
            for name, e in ev:
                m.setdefault(name, e)
            for ph, a, b in PHASES:
                if a in m and b in m:
                    rep[ph].append(m[a].elapsed_time(m[b]))
        for ph, v in rep.items():
            samples[ph] += v
        per_rep.append({ph: statistics.median(v) for ph, v in rep.items() if v})
    state_after = gpu_state()

    def summary(ph):
        v = samples[ph]
        if not v:
            return None
        meds = [r[ph] for r in per_rep if ph in r]
        return {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v),
                "rep_median_spread_ms": max(meds) - min(meds), "n": len(v)}

    info = lay.last_info
    rec = {"hidden": args.hidden, "C": args.C, "batch": args.batch, "pooling": args.pooling,
           "grad_precision": args.grad_precision, "dw_staging": args.dw_staging, "dw_stages": args.dw_stages,
           "nvcc_extra": " ".join(cuda_ops.NVCC_FLAGS[7:]), "tag": args.tag,
           "dw_cfg": info.get("dw_cfg"), "dx_warps": info.get("dx_warps"),
           "forward_backend": info.get("forward_backend"), "backward_backend": info.get("backward_backend"),
           "dw_staging_used": info.get("dw_staging", "none"),
           "bwd_dx_grid": info.get("bwd_dx_grid"), "bwd_dw_grid": info.get("bwd_dw_grid"),
           "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_res,
           "phases": {ph: summary(ph) for ph, _, _ in PHASES if samples[ph]},
           "gpu_before": state_before, "gpu_after": state_after}
    ph = rec["phases"]
    line = "  ".join(f"{k} {ph[k]['median_ms']:.2f}" for k in ("fwd_kernel", "fwd_total", "dx_kernel", "dw_kernel",
                                                              "bwd_total", "optimizer", "step") if k in ph)
    print(f"{args.tag + ' ' if args.tag else ''}H={args.hidden:,} C={args.C} {args.pooling} {args.grad_precision} dw={rec['dw_staging_used']}{args.dw_stages if rec['dw_staging_used'] == 'indexed' else ''}"
          f"/cfg{rec['dw_cfg']}: {line}  peak {peak_alloc / 2**30:.2f}/{peak_res / 2**30:.2f} GiB", flush=True)
    name = args.out or (f"traffic/kernels_{args.pooling}_H{args.hidden}_C{args.C}_{args.grad_precision}"
                        f"_{rec['dw_staging_used']}{args.dw_stages if rec['dw_staging_used'] == 'indexed' else ''}_cfg{rec['dw_cfg']}"
                        + (f"_{args.tag}" if args.tag else ""))
    save(name, {"args": vars(args), "gpu": gpu_info(), "record": rec})


if __name__ == "__main__":
    main()
