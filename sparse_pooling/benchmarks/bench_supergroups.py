"""Input-supergroup experiments: per-phase timing of complete training steps for several
execution variants of ONE layer (same graph, inputs, upstream gradients, parameters).

Variants are interleaved: each round runs every variant once, in alternating order
(round r even: listed order, odd: reversed).

Protocol "complete" (default; train_state.TrainState): after compiling / warming every variant
(lazy AdamW state, work lists), ONE reference training state is captured: parameters,
main_grad / .grad, AdamW moments, per-parameter steps, param-group hyperparameters, the
optimizer's stochastic-rounding counter (_seed) and the torch RNG states.  After switching
variants one untimed step rebuilds that variant's runtime; the reference state is then
restored, and restored again before every measured repetition (outside the CUDA-event
intervals, verified by an exact fingerprint).  Within a repetition the --iters steps advance
the state normally on the same fixed input.  The snapshot lives in pinned host memory
(--snapshot-device cpu): it never counts towards the device memory of a variant.
Protocol "historical" (labelled diagnostic): the previous harness (weights / bias restored
before each variant group, moments / steps / seed / RNG not restored).

Every requested grouped/generic phase and dX scatter variant is checked against last_info:
a variant whose kernels did not run as requested aborts the run.

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
from pathlib import Path

import torch

from bench_kernels import POOLINGS, gpu_state
from common import gpu_info, save  # (puts the implementation under test on sys.path)
import sparse_pooling
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import DX_PLANS, PhysicalPlan
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.optim import FusedAdamW
from sparse_pooling.supergroups import ConnectivityPlan, validate
from train_state import TrainState

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
    ("bwd_kernels", "bwd.dx_init", "bwd.dw"),           # dX + dW kernels, or the fused kernel + its fallbacks
    ("fused_kernel", "bwd.dx_init", "bwd.fused"),       # fused shared-projection kernel alone
    ("fused_fallbacks", "bwd.fused", "bwd.dx"),         # its guarded generic fallback launches (exit at once)
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


def _impl_dir():
    return str(Path(sparse_pooling.__file__).resolve().parent)


def git_sha():
    """Commit of the implementation under test (not of the current directory)."""
    try:
        return subprocess.run(["git", "-C", _impl_dir(), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_dirty():
    try:
        return bool(subprocess.run(["git", "-C", _impl_dir(), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True,
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
    ap.add_argument("--protocol", default="complete", choices=["complete", "historical"])
    ap.add_argument("--snapshot-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--composed", default="", help="D0,G2: time the composed model x0 -> Linear(D0, D) -> BF16 -> "
                    "layer -> Linear(G, G2) -> tanh -> MSE (torch AdamW for the two Linear layers) instead of the "
                    "isolated layer step (y^2 loss)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    variants = [parse_variant(v) for v in args.variant] or [("default", {})]
    library = str(cuda_ops.build_library())  # the library this process loads (recorded before any timing)
    build = cuda_ops.build_info() if hasattr(cuda_ops, "build_info") else {"library": library,
                                                                          "flags": cuda_ops.NVCC_FLAGS}

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
        del idx_c  # (kept alive by the runs reported in INPUT_SUPERGROUP_OPTIMIZATION.md: +0.19 GiB peak at C = 8)
    lay.enable_main_grad()
    opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16")
    gx = torch.Generator(device="cuda")
    gx.manual_seed(args.input_seed)
    x = torch.randn(args.batch, args.input_dim, device="cuda", generator=gx).to(torch.bfloat16).requires_grad_(True)
    small = []
    if args.composed:
        D0, G2 = map(int, args.composed.split(","))
        torch.manual_seed(args.init_seed + 17)
        proj = torch.nn.Linear(D0, args.input_dim, device="cuda")
        head = torch.nn.Linear(args.groups, G2, device="cuda")
        small = list(proj.parameters()) + list(head.parameters())
        opt2 = torch.optim.AdamW(small, lr=1e-4)
        x0 = torch.randn(args.batch, D0, device="cuda", generator=gx)
        tgt = 0.5 * torch.randn(args.batch, G2, device="cuda", generator=gx)
    if args.protocol == "historical":  # device copies of the initial weights / bias (previous harness)
        w_snap, b_snap = lay.weight.detach().clone(), lay.bias.detach().clone()

    def step(events, intervals=None):
        cuda_ops.PHASE_EVENTS = events
        cuda_ops.PHASE_INTERVALS = intervals
        mark = cuda_ops._mark
        try:
            mark("step.begin")
            if args.composed:  # projection -> layer -> head; "fwd" / "bwd" include the two Linear layers
                y = lay(proj(x0).bfloat16())
                mark("fwd.end")
                ((torch.tanh(head(y.float())) - tgt) ** 2).mean().backward()
                mark("bwd.end")
                opt2.step()
                opt2.zero_grad(set_to_none=True)
                opt.step()
                mark("opt.end")
            else:
                y = lay(x)
                mark("fwd.end")
                y.float().square().mean().backward()
                mark("bwd.end")
                opt.step()
                mark("opt.end")
        finally:
            cuda_ops.PHASE_EVENTS = None
            cuda_ops.PHASE_INTERVALS = None
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
    det_phases = [f"det_{k}_{s}" for k in ("fwd", "dx") for s in ("stage1", "reduce")]
    results = {name: {"rounds": [], "samples": {p: [] for p in [q[0] for q in PHASES] + det_phases},
                      "peak_alloc": 0, "peak_res": 0,
                      "scratch_peak": 0, "warm_peak_alloc": 0, "warm_peak_res": 0, "restore_checks": 0}
               for name, _ in variants}
    # warm-up every variant (compilation, allocator, lazy Adam state, work lists).  The peaks of
    # these steps are taken before any snapshot exists: production-style memory per variant.
    for name, _ in variants:
        lay.plan = plans[name]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            step(None)
        torch.cuda.synchronize()
        results[name]["warm_peak_alloc"] = torch.cuda.max_memory_allocated()
        results[name]["warm_peak_res"] = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    state = None
    ref_fp = None
    if args.protocol == "complete":
        state = TrainState(list(lay.parameters()), opt, device=args.snapshot_device).capture()
        ref_fp = state.fingerprint()
        state2 = TrainState(small, opt2, device=args.snapshot_device).capture() if small else None
        ref_fp2 = state2.fingerprint() if small else None

    def restore_reference(res):
        if state2 is not None:
            state2.restore()
            if state2.fingerprint() != ref_fp2:
                raise SystemExit("training state restoration failed for the composed model's Linear layers")
        state.restore()
        fp = state.fingerprint()
        if fp != ref_fp:
            bad = [k for k in ref_fp if fp.get(k) != ref_fp[k]]
            raise SystemExit(f"training state restoration failed for {bad}")
        res["restore_checks"] += 1

    gpu_before = gpu_state()
    for r in range(args.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for name, _ in order:
            lay.plan = plans[name]
            res = results[name]
            if args.protocol == "historical":
                with torch.no_grad():
                    lay.weight.copy_(w_snap)
                    lay.bias.copy_(b_snap)
            step(None)  # one untimed step after switching variants (rebuilds its runtime / work lists)
            torch.cuda.synchronize()
            rep_meds = {}
            for _ in range(args.reps):
                if state is not None:
                    restore_reference(res)  # outside the timed region; synchronizes
                rep = {p[0]: [] for p in PHASES}
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    ev, iv = [], []
                    step(ev, iv)
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
                    tot = {}
                    for nm, e0, e1 in iv:  # deterministic stages: summed over token windows
                        tot[nm] = tot.get(nm, 0.0) + e0.elapsed_time(e1)
                    for nm, v in tot.items():
                        rep.setdefault(nm, []).append(v)
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
            want_sc = getattr(plans[name], "cuda_grouped_dx_scatter", "auto")
            got_sc = (info.get("grouped_dx") or {}).get("scatter") if got["dx"] == "grouped" else None
            if want_sc != "auto" and got["dx"] == "grouped" and got_sc != want_sc:
                raise SystemExit(f"variant {name}: dX scatter {want_sc} requested, ran {got_sc}")
            if got["dx"] == "grouped" and info.get("backward_organization", "separate") == "separate":
                # the whole requested dX plan (tile, copies, scatter) actually ran
                gd = info["grouped_dx"]
                pl = plans[name]
                want_t = dict(DX_PLANS.get(pl.cuda_grouped_dx_plan, {}))
                for f in ("tokens", "warps", "reduction"):
                    if getattr(pl, f"cuda_grouped_dx_{f}"):
                        want_t[f] = getattr(pl, f"cuda_grouped_dx_{f}")
                gotk = {"tokens": gd["tokens"], "warps": gd["warps"], "reduction": gd["reduction_group_warps"],
                        "scatter": gd["scatter"]}
                bad = {f: (v, gotk[f]) for f, v in want_t.items() if gotk[f] != v}
                if pl.cuda_grouped_dx_plan in DX_PLANS and gd["plan"] != pl.cuda_grouped_dx_plan:
                    bad["plan"] = (pl.cuda_grouped_dx_plan, gd["plan"])
                if bad:
                    raise SystemExit(f"variant {name}: dX plan mismatch (requested, ran): {bad}")
                res["descriptors"] = lay._rt.sg.item_chunk_stats(gd["chunk_units"])
            res["kernels"] = got
            want_red = plans[name].reduction
            got_red = (info.get("reduction") or {}).get("mode")
            if want_red in ("fast", "deterministic") and got_red != want_red:
                raise SystemExit(f"variant {name}: reduction {want_red} requested, ran {got_red}")
            want_org = plans[name].cuda_grouped_backward
            got_org = info.get("backward_organization", "separate")
            if want_org in ("fused", "fused_recompute") and got_org != want_org:
                raise SystemExit(f"variant {name}: backward {want_org} requested, ran {got_org}")
            res["info"] = {k: info.get(k) for k in ("reduction", "backward_organization", "grouped_fused",
                                                    "grouped_fwd", "grouped_dx", "grouped_dw", "grouped_fallback",
                                                    "dx_epilogue", "dw_epilogue", "dw_staging", "dw_cfg", "dx_warps",
                                                    "bwd_dx_grid", "bwd_dw_grid")}
            if got["dx"] != "grouped":
                res["info"]["grouped_dx"] = None  # stale entry of an earlier grouped run
            if got["fwd"] != "grouped":
                res["info"]["grouped_fwd"] = None
            if got["dw"] != "grouped":
                res["info"]["grouped_dw"] = None
            res["work_list_bytes"] = sum((res["info"].get(k) or {}).get("work_list_bytes", 0)
                                         for k in ("grouped_fwd", "grouped_dx", "grouped_dw", "grouped_fused"))
    gpu_after = gpu_state()

    persistent = {
        "weights": lay.weight.numel() * 2, "bias": lay.bias.numel() * 2, "indices": lay.indices.numel() * 2,
        "main_grad": (lay.weight.numel() + lay.bias.numel()) * 4,
        "adam_moments": opt.state_bytes(),
        "graph_metadata": 0 if conn is None else lay.sg_feat.numel() * 2 + lay.sg_buckets.numel() * 4,
        "work_lists_cached_last_runtime": 0 if conn is None or lay._rt is None or lay._rt.sg is None
        else lay._rt.sg.runtime_bytes(),  # all work lists cached by the LAST runtime (see per-variant entries)
    }
    L = lay.layout
    out = {"commit": git_sha(), "dirty": git_dirty(), "nvcc_flags": build.get("flags"), "build": build,
           "implementation": str(Path(sparse_pooling.__file__).resolve().parent),
           "library": library, "args": vars(args),
           "config": {"C": args.C, "S": args.S, "R": args.R, "Q": (conn.num_supergroups(args.input_dim) if conn else 0),
                      "H": args.hidden, "M": args.batch, "G": args.groups, "D": args.input_dim,
                      "pooling": args.pooling, "precision": args.grad_precision,
                      "graph": "unrestricted" if conn is None else conn.to_json(),
                      "matched_c8": args.matched_c8, "layout_block": L.block},
           "build_seconds": build_s, "graph_stats": lay.graph_stats, "validation": val,
           "persistent_bytes": persistent, "infeasible": infeasible, "protocol": args.protocol,
           "snapshot": ({"device": "cuda", "bytes": 2 * (lay.weight.numel() + lay.bias.numel()), "contents": "weight, bias"}
                        if state is None else {"device": args.snapshot_device, "bytes": state.nbytes(),
                                               "contents": sorted(state._snap["tensors"])}), "gpu_before": gpu_before, "gpu_after": gpu_after, "variants": {}}
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
                                 "phases": {ph: summary(ph) for ph in res["samples"] if res["samples"][ph]},
                                 "peak_allocated_bytes": res["peak_alloc"], "peak_reserved_bytes": res["peak_res"],
                                 "warmup_peak_allocated_bytes": res["warm_peak_alloc"],
                                 "warmup_peak_reserved_bytes": res["warm_peak_res"],
                                 "step_scratch_peak_bytes": res["scratch_peak"],
                                 "work_list_bytes": res.get("work_list_bytes", 0),
                                 "descriptors": res.get("descriptors"),
                                 "restore_checks": res["restore_checks"]}
        ph = out["variants"][name]["phases"]
        fk = "fwd_kernel" if "fwd_kernel" in ph else "fwd_kernel_generic"
        line = "  ".join(f"{k} {ph[k]['median_ms']:.2f}" for k in (fk, "fwd_total", "dx_kernel", "dw_kernel", "bwd_kernels",
                                                                   "bwd_total", "step") if k in ph)
        print(f"[{name}] C={args.C} S={args.S} R={args.R} H={args.hidden:,} M={args.batch} {args.grad_precision}: "
              f"{line}  peak {res['peak_alloc'] / 2**30:.2f} GiB scratch {res['scratch_peak'] / 2**20:.0f} MiB "
              f"kernels {res.get('kernels')}", flush=True)
    out["gpu"] = gpu_info()
    save(args.out, out)


if __name__ == "__main__":
    main()
