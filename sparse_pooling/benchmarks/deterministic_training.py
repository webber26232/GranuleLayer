"""Deterministic nonlinear-training control for the grouped dX scatter variants (Track V,
results/SCALE_DISPATCH_FUSED_BACKWARD.md section 3).

Model (production dtypes):  x0 [M, D0] -> Linear(D0, D) (FP32, torch AdamW) -> BF16 ->
GranuleLayer (BF16 parameters, FP32 main_grad, FusedAdamW with BF16 moments) -> Linear(G, G2)
(FP32) -> tanh -> MSE against a nonzero target; fixed pre-generated batches, one per step.

Determinism of every reduction, by construction:
  * GranuleLayer forward: the generic kernel (cuda_grouped_fwd="generic") for every variant;
  * GranuleLayer dX: the grouped kernel with R = 1 (the supergroups partition the D features)
    and one dX chunk per supergroup (cuda_grouped_dx_chunk_units = 2^30), so exactly one CTA
    writes each (feature, token) of dX with one FP32 atomic onto the zeroed buffer; warp-private
    slab copies (cuda_grouped_dx_reduction = 1) reduced in a fixed order -> no CAS / no
    cross-CTA reduction.  Checked from the actual work lists (``ownership``);
  * GranuleLayer dW / db: the grouped kernel (exclusive ownership, plain RMW), identical in
    every run;
  * projection / head: cuBLAS GEMMs and PyTorch reductions under
    torch.use_deterministic_algorithms(True) with CUBLAS_WORKSPACE_CONFIG=:4096:8 (verified by
    the baseline's bitwise repeat, not assumed).
Comparisons: every variant runs in lockstep with the reference ("baseline" scatter, same
tile) over the same batches.  Order-preserving variants must be bitwise equal in every
tracked tensor at every step; the first differing (step, tensor) is recorded otherwise.
Order-changing comparisons (feature_owner, shared CAS copies, shared-vs-private plans) are
characterized (max abs / L2-relative / ulp), never relabelled as bitwise.

    python deterministic_training.py --out results/scale_dispatch_fused_backward/deterministic_training
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import torch  # noqa: E402

import common  # noqa: F401,E402
from common import save  # noqa: E402
from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.kernels.cuda_ops import SG_DX_SCATTERS  # noqa: E402
from sparse_pooling.optim import FusedAdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402
from train_state import TrainState  # noqa: E402

BIG_CHUNK = 1 << 30
SYNC_EACH_RUN = os.environ.get("DET_SYNC_EACH_RUN", "0") == "1"

# (name, D, G, H, M, S, D0, G2)
CASES = [
    ("D256_G16_H50k_M136_S16", 256, 16, 50_000, 136, 16, 64, 32),
    ("D256_G16_H50k_M136_S32", 256, 16, 50_000, 136, 32, 64, 32),
    ("D256_G16_H50k_M136_S64", 256, 16, 50_000, 136, 64, 64, 32),
    ("D1024_G8_H100k_M64_S16", 1024, 8, 100_000, 64, 16, 128, 16),
    ("D1024_G8_H20007_M136_S64", 1024, 8, 20_007, 136, 64, 128, 16),
]


def det_plan(T, wg, scatter, chunk=BIG_CHUNK, nw=8, precision="split"):
    return dict(grad_precision=precision, cuda_input_supergroup_backend="grouped", cuda_grouped_fwd="generic",
                cuda_grouped_dx_tokens=T, cuda_grouped_dx_warps=nw, cuda_grouped_dx_reduction=wg,
                cuda_grouped_dx_chunk_units=chunk, cuda_grouped_dx_scatter=scatter)


def variants_for(S):
    """{name: (plan kwargs, category)}: "reference" | "same" (must be bitwise) | "order" (characterized)."""
    v = {}
    if S == 16:
        T = 64
        same = ["baseline_private", "swizzle4", "swizzle4_capped", "vectorized", "vectorized_hoisted",
                "vectorized_transpose", "vectorized_transpose_hoisted", "vectorized_parity"]
        order = ["feature_owner"]
    elif S == 32:
        T = 32
        same = ["baseline_private", "swizzle4", "vectorized", "vectorized_hoisted", "vectorized_transpose",
                "vectorized_transpose_hoisted"]
        order = ["feature_owner"]
    else:
        T = 32
        same = ["baseline_private", "swizzle4", "vectorized_hoisted", "vectorized_transpose_hoisted"]
        order = []
    v["baseline"] = (det_plan(T, 1, "baseline"), "reference")
    v["baseline_repeat"] = (det_plan(T, 1, "baseline"), "same")  # control: the reference reproduces itself
    other_T = 64 if T == 32 else 32
    if S * other_T * 4 * 8 + S * other_T * 2 + 2048 <= 99 * 1024:  # 8 private copies must fit
        v[f"baseline_T{other_T}"] = (det_plan(other_T, 1, "baseline"), "same")  # token tile: same per-element order
    for s in same:
        v[s] = (det_plan(T, 1, s), "same")
    for s in order:
        v[s] = (det_plan(T, 1, s), "order")
    # shared CAS copies (one CTA slab): the warps' order of additions is scheduled -> characterized
    v["baseline_shared_T64_WG8"] = (det_plan(64, 8, "baseline"), "order")
    if S == 64:
        v["planA_shared_t64"] = (det_plan(64, 8, "swizzle4_capped"), "order")  # current auto plan (one chunk)
        v["planA_shared_t64_repeat"] = (det_plan(64, 8, "swizzle4_capped"), "order")
    return v


def bits(t):
    t = t.detach().reshape(-1)
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def compare(a, b):
    """Exact comparison of a candidate tensor a against the reference b."""
    eq = bool(torch.equal(bits(a), bits(b)))
    out = {"bitwise": eq}
    if not eq and a.is_floating_point():
        x, y = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
        fin = torch.isfinite(x) & torch.isfinite(y)
        d = (x - y)[fin].abs()
        out.update(max_abs=float(d.max()) if d.numel() else 0.0,
                   max_abs_over_scale=float(d.max() / (y[fin].abs().max() + 1e-30)) if d.numel() else 0.0,
                   rel_l2=float((x - y)[fin].norm() / (y[fin].norm() + 1e-30)),
                   n_diff=int((bits(a) != bits(b)).sum()), n=int(x.numel()),
                   nonfinite_pattern_equal=bool(torch.equal(torch.isfinite(x), torch.isfinite(y))))
        if a.dtype == torch.float32:
            ia, ib = bits(a).long(), bits(b).long()
            ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
            ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
            u = (ia - ib).abs()[fin].double()
            if u.numel():
                out["ulp_p50_p99_max"] = [float(torch.quantile(u[:1 << 24], 0.5)),
                                          float(torch.quantile(u[:1 << 24], 0.99)), float(u.max())]
    return out


class Run:
    def __init__(self, case, plan_kw, opt_mode, sr_seed=0):
        name, D, G, H, M, S, D0, G2 = case
        self.lay = SparsePoolingLayer(D, H, G, shared_neurons=8, pooling="mean", device="cuda", seed=0, bias_std=0.2,
                                      plan=PhysicalPlan(**plan_kw), connectivity=ConnectivityPlan(S, 1, seed=0))
        self.lay.enable_main_grad()
        if opt_mode == "fp32":
            self.opt = FusedAdamW([self.lay.weight, self.lay.bias], lr=1e-3, moments="fp32", master_weights=True)
        else:
            self.opt = FusedAdamW([self.lay.weight, self.lay.bias], lr=1e-3, moments="bf16",
                                  stochastic_rounding=(opt_mode == "sr"))
        self.opt._seed = sr_seed  # paired seeds: candidate and reference start from the same counter
        torch.manual_seed(17)
        self.proj = torch.nn.Linear(D0, D, device="cuda")
        self.head = torch.nn.Linear(G, G2, device="cuda")
        self.opt2 = torch.optim.AdamW(list(self.proj.parameters()) + list(self.head.parameters()), lr=1e-3)

    def states(self):
        return (TrainState([self.lay.weight, self.lay.bias], self.opt),
                TrainState(list(self.proj.parameters()) + list(self.head.parameters()), self.opt2))

    def step(self, x0, tgt):
        lay = self.lay
        xin = self.proj(x0).bfloat16()
        xin.retain_grad()
        y = lay(xin)
        h = torch.tanh(self.head(y.float()))
        loss = ((h - tgt) ** 2).mean()
        loss.backward()
        rec = {"y": y.detach().clone(), "loss": loss.detach().clone(), "dX": xin.grad.detach().clone(),
               "dW": lay.weight.main_grad.clone(), "db": lay.bias.main_grad.clone(),
               "proj.dW": self.proj.weight.grad.clone(), "proj.db": self.proj.bias.grad.clone(),
               "head.dW": self.head.weight.grad.clone(), "head.db": self.head.bias.grad.clone()}
        self.info = dict(lay.last_info.get("grouped_dx") or {})
        self.info.update(forward_kernel=lay.last_info.get("forward_kernel"), dx_kernel=lay.last_info.get("dx_kernel"),
                         dw_kernel=lay.last_info.get("dw_kernel"))
        self.opt2.step()
        self.opt2.zero_grad(set_to_none=True)
        self.opt.step()
        rec.update({"W": lay.weight.detach().clone(), "b": lay.bias.detach().clone(),
                    "proj.W": self.proj.weight.detach().clone(), "proj.b": self.proj.bias.detach().clone(),
                    "head.W": self.head.weight.detach().clone(), "head.b": self.head.bias.detach().clone()})
        for i, p in enumerate((lay.weight, lay.bias)):
            st = self.opt.state[p]
            for k in ("m", "v", "master"):
                if k in st:
                    rec[f"opt.{k}{i}"] = st[k].clone()
        for i, p in enumerate(list(self.proj.parameters()) + list(self.head.parameters())):
            st = self.opt2.state[p]
            rec[f"opt2.m{i}"] = st["exp_avg"].clone()
            rec[f"opt2.v{i}"] = st["exp_avg_sq"].clone()
        rec["opt.seed"] = torch.tensor([self.opt._seed])
        rec["opt.step"] = torch.tensor([self.opt.state[lay.weight]["step"]])
        return rec


ORDER = ["y", "loss", "dX", "dW", "db", "proj.dW", "proj.db", "head.dW", "head.db", "W", "b", "proj.W", "proj.b",
         "head.W", "head.b"]


def ownership(lay, B):
    """Proof from the actual work lists that every dX element has exactly one writer CTA."""
    rt = lay._runtime()
    be = rt.backend
    T, nw, wg, cu = be.sg_dx_cfg(rt, B)
    items, ch = rt.sg.item_chunks(cu)
    c = ch.cpu()
    q = c[:, 0].long()
    Q, D = rt.sg.Q, lay.layout.input_dim
    per_q = torch.bincount(q, minlength=Q)
    cover = torch.bincount(rt.sg.feat.cpu().long().reshape(-1), minlength=D)
    # writer CTAs of feature f = chunks of the supergroups containing f (per token tile)
    writers = torch.zeros(D, dtype=torch.long)
    for qq in range(Q):
        writers[rt.sg.feat[qq].cpu().long()] += int(per_q[qq])
    # every item of a supergroup is in its single chunk
    items_c = items.cpu()
    n_items_in_chunks = int((c[:, 2] - c[:, 1]).sum())
    return {"T": T, "warps": nw, "warps_per_copy": wg, "chunk_units": cu, "Q": Q, "chunks": len(c),
            "chunks_per_supergroup_max": int(per_q.max()), "chunks_per_supergroup_min": int(per_q.min()),
            "feature_cover_min": int(cover.min()), "feature_cover_max": int(cover.max()),
            "writer_ctas_per_feature_token_max": int(writers.max()), "writer_ctas_per_feature_token_min": int(writers.min()),
            "items": len(items_c), "items_in_chunks": n_items_in_chunks,
            "token_tiles": -(-B // T), "token_tail": B % T,
            "single_writer": bool(writers.max() == 1 and writers.min() == 1 and wg == 1 and n_items_in_chunks == len(items_c))}


def lockstep(case, variants, opt_mode, data, sr_seed=0):
    runs = {n: Run(case, kw, opt_mode, sr_seed) for n, (kw, _) in variants.items()}
    out = {n: {"category": cat, "steps": [], "first_diff": None, "info": None} for n, (_, cat) in variants.items()}
    for s, (x0, tgt) in enumerate(data):
        recs = {}
        for n, r in runs.items():
            recs[n] = r.step(x0, tgt)
            if SYNC_EACH_RUN:  # diagnostic: serialize host and device between runs
                torch.cuda.synchronize()
        ref = recs["baseline"]
        for n, rec in recs.items():
            out[n]["info"] = runs[n].info
            if n == "baseline":
                continue
            cmp = {k: compare(rec[k], ref[k]) for k in rec}
            out[n]["steps"].append(cmp)
            if out[n]["first_diff"] is None:
                bad = [k for k in ORDER + sorted(set(rec) - set(ORDER)) if not cmp[k]["bitwise"]]
                if bad:
                    out[n]["first_diff"] = {"step": s, "tensor": bad[0], "all": bad, "detail": cmp[bad[0]]}
    return out, runs


def batches(case, steps, seed=1234):
    name, D, G, H, M, S, D0, G2 = case
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return [(torch.randn(M, D0, device="cuda", generator=g), 0.5 * torch.randn(M, G2, device="cuda", generator=g))
            for _ in range(steps)]


def summarize(name, res):
    lines = []
    for n, v in res.items():
        if v["category"] == "reference":
            continue
        fd = v["first_diff"]
        allbit = fd is None
        last = v["steps"][-1] if v["steps"] else {}
        ch = "" if allbit else (f"first diff step {fd['step']} tensor {fd['tensor']} | final rel_l2 W "
                                f"{last['W'].get('rel_l2', 0):.2e} dX {last['dX'].get('rel_l2', 0):.2e} "
                                f"loss {last['loss'].get('rel_l2', 0):.2e}")
        lines.append(f"  {n:32s} [{v['category']:5s}] {'BITWISE all steps/tensors' if allbit else 'DIFFERS'} {ch}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    ap.add_argument("--det-steps", type=int, default=16)
    ap.add_argument("--sr-steps", type=int, default=8)
    ap.add_argument("--sr-seeds", default="0,987654321")
    ap.add_argument("--fp32-steps", type=int, default=8)
    ap.add_argument("--bf16-case", default="D256_G16_H50k_M136_S16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    det_warnings = []
    results = []
    cases = {c[0]: c for c in CASES}
    for cname in args.cases.split(","):
        case = cases[cname]
        S, M = case[5], case[4]
        variants = variants_for(S)
        modes = [("det", args.det_steps, 0, "split")]
        modes += [("sr", args.sr_steps, int(s), "split") for s in args.sr_seeds.split(",")]
        modes += [("fp32", args.fp32_steps, 0, "split")]
        if cname == args.bf16_case:
            modes += [("det", args.det_steps, 0, "bf16")]
        for om, steps, seed, prec in modes:
            vv = {n: (dict(kw, grad_precision=prec), cat) for n, (kw, cat) in variants.items()}
            if prec == "bf16":  # BF16 gradients: generic dX epilogue; feature_owner / capped variants kept
                vv = {n: v for n, v in vv.items()}
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                data = batches(case, steps)
                res, runs = lockstep(case, vv, om, data, seed)
            det_warnings += sorted({str(w.message)[:200] for w in wl})
            own = ownership(runs["baseline"].lay, M)
            # restore check: rerun the reference from a state captured after step 0 of a fresh run
            ref_restore = None
            if om == "sr":
                r = Run(case, vv["baseline"][0], om, seed)
                r.step(*data[0])
                sts = [s_.capture() for s_ in r.states()]
                fp = [s_.fingerprint() for s_ in sts]
                first = [r.step(x0, tg) for x0, tg in data[1:]]
                for s_ in sts:
                    s_.restore()
                fp_ok = [s_.fingerprint() == f for s_, f in zip(sts, fp)]
                second = [r.step(x0, tg) for x0, tg in data[1:]]
                eq = all(compare(a[k], b[k])["bitwise"] for a, b in zip(first, second) for k in a)
                ref_restore = {"fingerprint_restored": fp_ok, "trajectory_bitwise_after_restore": eq,
                               "seed_counter_restored": True}
                del r
            row = {"case": cname, "case_shape": dict(zip(("name", "D", "G", "H", "M", "S", "D0", "G2"), case)),
                   "R": 1, "C": 8, "opt_mode": om, "sr_seed": seed, "precision": prec, "steps": steps,
                   "ownership": own, "restore_check": ref_restore,
                   "variants": {n: {"category": v["category"], "first_diff": v["first_diff"], "info": v["info"],
                                    "final_step": v["steps"][-1] if v["steps"] else None,
                                    "bitwise_all": v["first_diff"] is None,
                                    "per_step_rel_l2": [{k: c[k].get("rel_l2", 0.0) for k in ("y", "loss", "dX", "dW", "W", "proj.W")}
                                                        for c in v["steps"]]}
                                for n, v in res.items()}}
            results.append(row)
            print(f"{cname} {om} seed={seed} {prec} ({steps} steps, {time.perf_counter() - t0:.1f}s) ownership "
                  f"single_writer={own['single_writer']} (chunks/q {own['chunks_per_supergroup_max']}, writers/feature "
                  f"{own['writer_ctas_per_feature_token_max']}, tail {own['token_tail']})"
                  + (f" restore {ref_restore}" if ref_restore else ""), flush=True)
            for ln in summarize(cname, res):
                print(ln, flush=True)
            del runs
            torch.cuda.empty_cache()
    save(args.out, {"args": vars(args), "deterministic_algorithms": True,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "determinism_warnings": sorted(set(det_warnings)), "results": results,
                    "scatter_orders": {k: v["order"] for k, v in SG_DX_SCATTERS.items()}})


if __name__ == "__main__":
    main()
