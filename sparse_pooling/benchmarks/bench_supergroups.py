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
    ("fwd_bias_only", "fwd.kernel", "fwd.bias_only"),   # kgx sparse: constant of all-zero weight rows
    ("fwd_finalize", "fwd.kernel", "fwd.finalize"),     # merge / divide / transpose (incl. fwd_bias_only)
    ("fwd_total", "step.begin", "fwd.end"),
    ("bwd_coef", "bwd.begin", "bwd.coef"),              # c1 (+ grouped: c1T transpose)
    ("bwd_layout", "bwd.coef", "bwd.layout"),
    ("bwd_dx_init", "bwd.layout", "bwd.dx_init"),       # dX buffer zeroing
    ("dx_kernel", "bwd.dx_init", "bwd.dx"),             # local dX + global merge (atomics)
    ("dw_kernel", "bwd.dx", "bwd.dw"),                  # dW / db (+ non-finite fallback launch)
    ("bwd_kernels", "bwd.dx_init", "bwd.dw"),           # dX + dW kernels, or the fused kernel + its fallbacks
    ("bwd_bias_only", "bwd.dw", "bwd.bias_only"),       # kgx sparse: db of all-zero weight rows (in bwd_post)
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
    ap.add_argument("--layer-factory", default="", help="module:function(args, conn) -> layer: build the layer "
                    "with another constructor (kcsr sweep: physical lowering, experimental generalized fan-in); the "
                    "timing protocol is unchanged")
    ap.add_argument("--factory-args", default="{}", help="JSON passed to the layer factory as args.factory")
    ap.add_argument("--layer-variants", default="", help="comma-separated [name=]variant[@tile] of the layer's own "
                    "execution variants (layer.select_variant; K64 study: kgen, lowered, direct, idf_rmw, idf), "
                    "interleaved like --variant on ONE layer (same graph, parameters, inputs)")
    ap.add_argument("--opt-factory", default="", help="module:function(args, layer) -> optimizer for the layer's "
                    "parameters (default: FusedAdamW lr 1e-4, BF16 moments)")
    ap.add_argument("--opt-args", default="{}", help="JSON passed to the optimizer factory as args.opt")
    ap.add_argument("--inference", action="store_true", help="time the forward pass only (no_grad, fixed parameters: "
                    "a compressed-checkpoint inference measurement; it cannot establish training speed)")
    ap.add_argument("--kcsr", action="store_true", help="also record the logical graph fingerprint, the full "
                    "execution record, per-round GPU state, cold-step and host-enqueue times (kcsr sweep)")
    ap.add_argument("--canonical-start", action="store_true", help="width-preserving study: after the warm-up "
                    "steps, restore the PRE-warm-up parameter values and reset the optimizer state (zero moments, "
                    "step 0, rounding counter 0) and main_grad, so every compared format starts its timed steps from "
                    "the same canonical state (verified by --identity); the warm-up never leaves formats at "
                    "different starting checkpoints")
    ap.add_argument("--identity", action="store_true", help="record the canonical identity (experimental/canonical.py:"
                    " parameters, mask, main_grad, both moments, storage) of the restored starting state")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import json as _json

    args.factory = _json.loads(args.factory_args)
    args.opt = _json.loads(args.opt_args)
    layer_variants = {}
    for item in filter(None, args.layer_variants.split(",")):
        nm, _, spec = item.rpartition("=")
        v, _, t = spec.partition("@")
        layer_variants[nm or spec] = (v, int(t) if t else None)
    variants = [parse_variant(v) for v in args.variant] or [("default", {})]
    if layer_variants:
        if args.variant:
            raise SystemExit("--layer-variants and --variant are exclusive")
        variants = [(nm, {}) for nm in layer_variants]
    library = str(cuda_ops.build_library())  # the library this process loads (recorded before any timing)
    build = cuda_ops.build_info() if hasattr(cuda_ops, "build_info") else {"library": library,
                                                                          "flags": cuda_ops.NVCC_FLAGS}

    torch.manual_seed(args.input_seed)
    conn = None if args.S == 0 else ConnectivityPlan(args.S, args.R, args.gen, args.graph_seed)
    t0 = time.perf_counter()
    if args.layer_factory:
        import importlib

        mod, _, fn = args.layer_factory.partition(":")
        lay = getattr(importlib.import_module(mod), fn)(args, conn)
        conn = getattr(lay, "connectivity", conn)
    else:
        lay = SparsePoolingLayer(args.input_dim, args.hidden, args.groups, shared_neurons=args.C,
                                 pooling=POOLINGS[args.pooling], checkpoint_policy="auto", backend="auto",
                                 device="cuda", plan=PhysicalPlan(grad_precision=args.grad_precision),
                                 seed=args.init_seed, connectivity=conn)
    if args.matched_c8:
        from bench_matched import convert_to_matched_c8

        lay = convert_to_matched_c8(lay)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    kcsr = {}
    if args.kcsr:  # identity of the exact graph / parameters that are timed (outside every timed region)
        import kcsr_graph
        from train_state import tensor_fingerprint

        t1 = time.perf_counter()
        tab = (lay.kcsr_table() if hasattr(lay, "kcsr_table") else
               kcsr_graph.native_table(lay, logical_C=getattr(lay, "kcsr_logical_C", None)))
        kcsr.update(graph_fingerprint=kcsr_graph.fingerprint(tab), graph_summary=kcsr_graph.summary(tab),
                    param_fingerprint={"weight": tensor_fingerprint(lay.weight), "bias": tensor_fingerprint(lay.bias)},
                    logical_params_sha=kcsr_graph.params_sha(lay),
                    logical_params_sha_schema=kcsr_graph.params_sha_schema(lay),
                    layer_description=lay.kcsr_describe() if hasattr(lay, "kcsr_describe") else None)
        del tab
        kcsr["fingerprint_seconds"] = time.perf_counter() - t1
    val = None
    if args.validate:
        L = lay.layout
        idx_c = L.unpack_indices(lay.indices) if L.packed else lay.indices
        from sparse_pooling.supergroups import SupergroupGraph
        g = None if conn is None else SupergroupGraph(conn, lay.sg_feat, lay.sg_buckets)
        val = validate(L, idx_c, g)
        del idx_c  # (kept alive by the runs reported in INPUT_SUPERGROUP_OPTIMIZATION.md: +0.19 GiB peak at C = 8)
    lay.enable_main_grad()
    if args.opt_factory:
        import importlib

        mod, _, fn = args.opt_factory.partition(":")
        opt = getattr(importlib.import_module(mod), fn)(args, lay)
    else:
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments="bf16")
    immutable_fp0 = lay.immutable_fingerprints() if hasattr(lay, "immutable_fingerprints") else None
    start_vals = None
    if args.canonical_start:  # host copies of the pre-warm-up parameter values (benchmark overhead, not charged)
        start_vals = [p.detach().to("cpu", copy=True).pin_memory() for p in lay.parameters()]
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
            elif args.inference:
                with torch.no_grad():
                    lay(x)
                mark("fwd.end")
                mark("bwd.end")
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
    results = {name: {"rounds": [], "samples": {p: [] for p in [q[0] for q in PHASES] + det_phases + ["host_enqueue",
                                                                                                  "step_wall"]},
                      "peak_alloc": 0, "peak_res": 0,
                      "scratch_peak": 0, "warm_peak_alloc": 0, "warm_peak_res": 0, "restore_checks": 0}
               for name, _ in variants}
    # warm-up every variant (compilation, allocator, lazy Adam state, work lists).  The peaks of
    # these steps are taken before any snapshot exists: production-style memory per variant.
    cold = {}
    def use(name):
        lay.plan = plans[name]
        if name in layer_variants:
            lay.select_variant(*layer_variants[name])

    for name, _ in variants:
        use(name)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        walls = []
        for _ in range(2):
            tw = time.perf_counter()
            step(None)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - tw)
        cold[name] = {"first_step_wall_s": walls[0], "second_step_wall_s": walls[1]}
        torch.cuda.synchronize()
        results[name]["warm_peak_alloc"] = torch.cuda.max_memory_allocated()
        results[name]["warm_peak_res"] = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()
    if start_vals is not None:  # canonical start: undo the warm-up updates (values, main_grad, optimizer state)
        with torch.no_grad():
            for p, v in zip(lay.parameters(), start_vals):
                p.data.copy_(v)
                if getattr(p, "main_grad", None) is not None:
                    p.main_grad.zero_()
                p.grad = None
            for st in opt.state.values():
                for k in list(st):
                    if torch.is_tensor(st[k]):
                        st[k].zero_()
                    elif k == "step":
                        st[k] = 0
            if hasattr(opt, "_seed"):
                opt._seed = 0
        torch.manual_seed(args.input_seed)
        torch.cuda.synchronize()
        del start_vals
    if args.identity and hasattr(lay, "canon_rows"):
        from sparse_pooling.experimental import canonical

        t1 = time.perf_counter()
        kcsr["canonical_start"] = canonical.identity(lay, opt)
        kcsr["canonical_start"]["seconds"] = time.perf_counter() - t1
        kcsr["canonical_start"]["after_reset"] = bool(args.canonical_start)
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
            use(name)
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
                    tw = time.perf_counter()
                    step(ev, iv)
                    t_enq = time.perf_counter()
                    torch.cuda.synchronize()
                    t_end = time.perf_counter()
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
                    if args.kcsr:  # host wall clock: enqueue (Python + launches) and step-to-sync
                        rep.setdefault("host_enqueue", []).append((t_enq - tw) * 1e3)
                        rep.setdefault("step_wall", []).append((t_end - tw) * 1e3)
                for ph, v in rep.items():
                    if v:
                        res["samples"][ph] += v
                        rep_meds.setdefault(ph, []).append(statistics.median(v))
            res["rounds"].append({ph: statistics.median(v) for ph, v in rep_meds.items()})
            if args.kcsr:
                res.setdefault("gpu_rounds", []).append(gpu_state())
            info = lay.last_info
            want = expected_kernels(plans[name], conn is not None)
            got = {"fwd": info.get("forward_kernel"), "dx": info.get("dx_kernel"), "dw": info.get("dw_kernel")}
            if name in layer_variants or args.inference or hasattr(lay, "kvariant"):  # the layer's own variants
                want = {ph: None for ph in want}
                lv = layer_variants[name][0] if name in layer_variants else getattr(lay, "kvariant", None)
                bad = {ph: k for ph, k in got.items() if lv is not None and not str(k).endswith("_" + lv)
                       and not (args.inference and ph != "fwd")}
                if bad:
                    raise SystemExit(f"variant {name}: requested {lv}, ran {bad}")
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
            if args.kcsr:
                res["execution"] = info.get("execution")
                res["last_info_keys"] = {k: v for k, v in info.items() if k != "execution"}
                res["layer_variant"] = layer_variants.get(name)
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
        "weights": lay.weight.numel() * 2, "bias": lay.bias.numel() * 2,
        "indices": 0 if getattr(lay, "indices", None) is None else lay.indices.numel() * 2,
        "main_grad": sum(p.numel() * 4 for p in lay.parameters() if getattr(p, "main_grad", None) is not None)
        or (lay.weight.numel() + lay.bias.numel()) * 4,
        "adam_moments": opt.state_bytes(),
        "graph_metadata": 0 if conn is None else lay.sg_feat.numel() * 2 + lay.sg_buckets.numel() * 4,
        "work_lists_cached_last_runtime": 0 if conn is None or lay._rt is None or lay._rt.sg is None
        else lay._rt.sg.runtime_bytes(),  # all work lists cached by the LAST runtime (see per-variant entries)
    }
    if hasattr(lay, "extra_persistent_bytes"):
        persistent.update(lay.extra_persistent_bytes())
    executed = None
    if hasattr(lay, "canon_rows"):  # executed tensor-core / CUDA-core work of one step (width-preserving study)
        from sparse_pooling.experimental.kgx_wide import executed_work

        executed = executed_work(lay, args.batch, args.grad_precision == "split") if hasattr(lay, "sg_feat") and \
            hasattr(lay, "index_free_ok") else None
        if executed is None and hasattr(lay, "env"):
            from sparse_pooling.experimental.kgx_onetile import executed_work_lowered

            executed = executed_work_lowered(lay, args.batch, args.grad_precision == "split")
    immutable_ok = None
    if immutable_fp0 is not None:  # frozen metadata (masks, row pointers, tile maps) unchanged by the steps
        immutable_ok = lay.immutable_fingerprints() == immutable_fp0
        if not immutable_ok:
            raise SystemExit("immutable layer metadata changed during the timed steps")
    from provenance import hardware_state, provenance

    L = lay.layout
    out = {"commit": git_sha(), "dirty": git_dirty(), "provenance": provenance(_impl_dir()),
           "hardware_state": hardware_state(), "immutable_metadata": immutable_fp0,
           "immutable_metadata_unchanged": immutable_ok, "nvcc_flags": build.get("flags"), "build": build,
           "implementation": str(Path(sparse_pooling.__file__).resolve().parent),
           "library": library, "args": vars(args),
           "config": {"C": args.C, "S": args.S, "R": args.R, "Q": (conn.num_supergroups(args.input_dim) if conn else 0),
                      "H": args.hidden, "M": args.batch, "G": args.groups, "D": args.input_dim,
                      "pooling": args.pooling, "precision": args.grad_precision,
                      "graph": "unrestricted" if conn is None else conn.to_json(),
                      "matched_c8": args.matched_c8, "layout_block": L.block},
           "build_seconds": build_s, "graph_stats": lay.graph_stats, "validation": val, "kcsr": kcsr,
           "cold_steps": cold, "executed_work": executed, "canonical_start_protocol": bool(args.canonical_start),
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
                                 "restore_checks": res["restore_checks"],
                                 "samples_ms": {ph: v for ph, v in res["samples"].items() if v},
                                 "round_values_ms": res["rounds"], "gpu_rounds": res.get("gpu_rounds"),
                                 "execution": res.get("execution"), "last_info": res.get("last_info_keys")}
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
