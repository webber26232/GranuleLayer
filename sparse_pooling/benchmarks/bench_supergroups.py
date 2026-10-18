"""Input-supergroup experiments: per-phase timing of complete training steps for several
execution variants of ONE layer (same graph, inputs, upstream gradients, parameters).

Variants are interleaved: each round runs every variant once, in alternating order
(round r even: listed order, odd: reversed).  Before every variant group the parameters
are restored from a snapshot, so each variant starts from identical weights/bias (AdamW
updates inside a group are identical computations; moments do not affect timing).
Every requested grouped/generic phase is checked against last_info: a variant whose
kernels did not run as requested aborts the run.

    python bench_supergroups.py --hidden 100000000 --C 64 --S 64 --R 16 \\
        --variant generic:cuda_input_supergroup_backend=generic \\
        --variant grouped:cuda_input_supergroup_backend=grouped,cuda_grouped_fwd_tokens=64

``--S 0`` builds the unrestricted (existing) architecture: only generic execution exists.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import time

import torch

from bench_kernels import POOLINGS, gpu_state
from common import gpu_info, save
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan, validate

PHASES = [
    ("fwd_layout", "fwd.begin", "fwd.layout"),          # input transpose (gather operand)
    ("fwd_zero", "fwd.layout", "fwd.zero"),             # grouped: ysum [G, B] zeroing
    ("fwd_kernel", "fwd.zero", "fwd.kernel"),           # projection + streaming pooling (+ merge atomics)
    ("fwd_kernel_generic", "fwd.layout", "fwd.kernel"),  # generic: no zeroing mark
    ("fwd_finalize", "fwd.kernel", "fwd.finalize"),     # merge / divide / transpose
    ("fwd_total", "step.begin", "fwd.end"),
    ("bwd_coef", "bwd.begin", "bwd.coef"),              # c1 (+ grouped: c1T transpose)
    ("bwd_layout", "bwd.coef", "bwd.layout"),
    ("bwd_dx_init", "bwd.layout", "bwd.dx_init"),       # dX buffer zeroing
    ("dx_kernel", "bwd.dx_init", "bwd.dx"),             # local dX + global merge (atomics)
    ("dw_kernel", "bwd.dx", "bwd.dw"),                  # dW / db (+ non-finite fallback launch)
    ("bwd_post", "bwd.dw", "bwd.end"),                  # dX transpose + FP32 -> BF16, autograd
    ("bwd_total", "fwd.end", "bwd.end"),
    ("optimizer", "bwd.end", "opt.end"),
    ("step", "step.begin", "opt.end"),
]


def parse_variant(spec: str):
    name, _, kv = spec.partition(":")
    out = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        f = PhysicalPlan.__dataclass_fields__[k]
        out[k] = int(v) if f.type in ("int", int) else v
    return name, out


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_dirty():
    try:
        return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                   text=True).stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def expected_kernels(plan: PhysicalPlan, has_graph: bool):
    if not has_graph:
        return {"fwd": "generic", "dx": "generic", "dw": "generic"}
    return {ph: plan.grouped_request(ph) for ph in ("fwd", "dx", "dw")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--groups", type=int, default=1024)
    ap.add_argument("--input-dim", type=int, default=1024)
    ap.add_argument("--S", type=int, default=64, help="input supergroup dim; 0 = unrestricted architecture")
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--gen", default="balanced_permutation")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--input-seed", type=int, default=0)
    ap.add_argument("--pooling", default="mean", choices=list(POOLINGS))
    ap.add_argument("--grad-precision", default="split", choices=["split", "bf16"])
    ap.add_argument("--variant", action="append", default=[], help="name:plan_field=value,...")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--validate", action="store_true", help="structural validation of the graph (stored)")
    ap.add_argument("--matched-c8", action="store_true",
                    help="build the graph at C = 64 and run it as C = 8 blocks with repeated records")
    ap.add_argument("--skip-infeasible", action="store_true",
                    help="drop (and record) variants whose explicit grouped request cannot run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    variants = [parse_variant(v) for v in args.variant] or [("default", {})]

    torch.manual_seed(args.input_seed)
    conn = None if args.S == 0 else ConnectivityPlan(args.S, args.R, args.gen, args.graph_seed)
    t0 = time.perf_counter()
    lay = SparsePoolingLayer(args.input_dim, args.hidden, args.groups, shared_neurons=args.C,
                             pooling=POOLINGS[args.pooling], checkpoint_policy="auto", backend="auto",
                             device="cuda", plan=PhysicalPlan(grad_precision=args.grad_precision),
                             seed=args.init_seed, connectivity=conn)
    if args.matched_c8:
        from bench_matched import convert_to_matched_c8

        lay = convert_to_matched_c8(lay)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    val = None
    if args.validate:
        L = lay.layout
        idx_c = L.unpack_indices(lay.indices) if L.packed else lay.indices
        from sparse_pooling.supergroups import SupergroupGraph
        g = None if conn is None else SupergroupGraph(conn, lay.sg_feat, lay.sg_buckets)
        val = validate(L, idx_c, g)
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16")
    gx = torch.Generator(device="cuda")
    gx.manual_seed(args.input_seed)
    x = torch.randn(args.batch, args.input_dim, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)
    w_snap, b_snap = lay.weight.detach().clone(), lay.bias.detach().clone()

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

    plans = {name: PhysicalPlan(grad_precision=args.grad_precision, **kv) for name, kv in variants}
    infeasible = {}
    if args.skip_infeasible and conn is not None:
        for name, _ in list(variants):
            lay.plan = plans[name]
            rt = lay._runtime()
            why = {ph: rt.backend.sg_reason(rt, ph, args.batch) for ph in ("fwd", "dx", "dw")
                   if plans[name].grouped_request(ph) == "grouped"}
            why = {k: v for k, v in why.items() if v}
            if why:
                infeasible[name] = why
                print(f"[{name}] infeasible: {why}", flush=True)
        variants = [v for v in variants if v[0] not in infeasible]
    results = {name: {"rounds": [], "samples": {p[0]: [] for p in PHASES}, "peak_alloc": 0, "peak_res": 0,
                      "scratch_peak": 0} for name, _ in variants}
    # warm-up every variant once (compilation, allocator, lazy Adam state, work lists)
    for name, _ in variants:
        lay.plan = plans[name]
        for _ in range(2):
            step(None)
    torch.cuda.synchronize()
    gpu_before = gpu_state()
    for r in range(args.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for name, _ in order:
            lay.plan = plans[name]
            with torch.no_grad():
                lay.weight.copy_(w_snap)
                lay.bias.copy_(b_snap)
            step(None)  # one untimed step after switching variants
            torch.cuda.synchronize()
            res = results[name]
            rep_meds = {}
            for _ in range(args.reps):
                rep = {p[0]: [] for p in PHASES}
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    ev = []
                    step(ev)
                    torch.cuda.synchronize()
                    pk = torch.cuda.max_memory_allocated()
                    res["peak_alloc"] = max(res["peak_alloc"], pk)
                    res["peak_res"] = max(res["peak_res"], torch.cuda.max_memory_reserved())
                    res["scratch_peak"] = max(res["scratch_peak"], pk - base)
                    m = {}
                    for nm, e in ev:
                        m.setdefault(nm, e)
                    for ph, a, b in PHASES:
                        if a in m and b in m:
                            rep[ph].append(m[a].elapsed_time(m[b]))
                for ph, v in rep.items():
                    if v:
                        res["samples"][ph] += v
                        rep_meds.setdefault(ph, []).append(statistics.median(v))
            res["rounds"].append({ph: statistics.median(v) for ph, v in rep_meds.items()})
            info = lay.last_info
            want = expected_kernels(plans[name], conn is not None)
            got = {"fwd": info.get("forward_kernel"), "dx": info.get("dx_kernel"), "dw": info.get("dw_kernel")}
            for ph in ("fwd", "dx", "dw"):
                if want[ph] in ("generic", "grouped") and got[ph] != want[ph]:
                    raise SystemExit(f"variant {name}: {ph} requested {want[ph]}, ran {got[ph]} "
                                     f"({info.get('grouped_fallback')})")
            res["kernels"] = got
            res["info"] = {k: info.get(k) for k in ("grouped_fwd", "grouped_dx", "grouped_dw", "grouped_fallback",
                                                    "dx_epilogue", "dw_epilogue", "dw_staging", "dw_cfg", "dx_warps",
                                                    "bwd_dx_grid", "bwd_dw_grid")}
    gpu_after = gpu_state()

    persistent = {
        "weights": lay.weight.numel() * 2, "bias": lay.bias.numel() * 2, "indices": lay.indices.numel() * 2,
        "main_grad": (lay.weight.numel() + lay.bias.numel()) * 4,
        "adam_moments": opt.state_bytes(),
        "graph_metadata": 0 if conn is None else lay.sg_feat.numel() * 2 + lay.sg_buckets.numel() * 4,
        "work_lists": 0 if conn is None or lay._rt is None or lay._rt.sg is None else lay._rt.sg.runtime_bytes(),
    }
    L = lay.layout
    out = {"commit": git_sha(), "dirty": git_dirty(), "nvcc_flags": cuda_ops.NVCC_FLAGS,
           "library": str(cuda_ops.build_library()), "args": vars(args),
           "config": {"C": args.C, "S": args.S, "R": args.R, "Q": (conn.num_supergroups(args.input_dim) if conn else 0),
                      "H": args.hidden, "M": args.batch, "G": args.groups, "D": args.input_dim,
                      "pooling": args.pooling, "precision": args.grad_precision,
                      "graph": "unrestricted" if conn is None else conn.to_json(),
                      "matched_c8": args.matched_c8, "layout_block": L.block},
           "build_seconds": build_s, "graph_stats": lay.graph_stats, "validation": val,
           "persistent_bytes": persistent, "infeasible": infeasible, "gpu_before": gpu_before, "gpu_after": gpu_after, "variants": {}}
    for name, kv in variants:
        res = results[name]

        def summary(ph):
            v = res["samples"][ph]
            if not v:
                return None
            meds = [rr[ph] for rr in res["rounds"] if ph in rr]
            return {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v),
                    "round_medians_ms": meds, "n": len(v)}

        out["variants"][name] = {"plan": kv, "kernels": res.get("kernels"), "info": res.get("info"),
                                 "phases": {ph: summary(ph) for ph, _, _ in PHASES if res["samples"][ph]},
                                 "peak_allocated_bytes": res["peak_alloc"], "peak_reserved_bytes": res["peak_res"],
                                 "step_scratch_peak_bytes": res["scratch_peak"]}
        ph = out["variants"][name]["phases"]
        fk = "fwd_kernel" if "fwd_kernel" in ph else "fwd_kernel_generic"
        line = "  ".join(f"{k} {ph[k]['median_ms']:.2f}" for k in (fk, "fwd_total", "dx_kernel", "dw_kernel",
                                                                   "bwd_total", "step") if k in ph)
        print(f"[{name}] C={args.C} S={args.S} R={args.R} H={args.hidden:,} M={args.batch} {args.grad_precision}: "
              f"{line}  peak {res['peak_alloc'] / 2**30:.2f} GiB scratch {res['scratch_peak'] / 2**20:.0f} MiB "
              f"kernels {res.get('kernels')}", flush=True)
    out["gpu"] = gpu_info()
    save(args.out, out)


if __name__ == "__main__":
    main()
