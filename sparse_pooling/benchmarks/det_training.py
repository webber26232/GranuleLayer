"""Repeatability of complete composed-model training with PhysicalPlan.reduction.

Model (production dtypes): x0 [M, D0] -> Linear(D0, D) (FP32, torch AdamW) -> BF16 -> GranuleLayer
(real R > 1 connectivity, normal work chunks; BF16 parameters, FP32 main_grad, FusedAdamW) ->
Linear(G, G2) (FP32) -> tanh -> MSE against a nonzero target; a different pre-generated batch per
step (gradients reach the projection through the layer's dX).

Per process: build the model from the seeds, run one warm-up step (creates the lazy optimizer
state), capture the complete training state (both optimizers, moments, master weights, step
counters, the stochastic-rounding counter, torch RNG; train_state.TrainState), then run the
trajectory of --steps steps TWICE: trajectory 1, restore, trajectory 2.  Every step records

    y, loss, dX (gradient into the projection output), layer dW / db (main_grad, before the
    optimizer clears it), projection / head gradients, then after both optimizer steps all
    parameters, layer moments (and master weights), torch AdamW moments, step and SR counters.

--role write: the warm-up step and trajectory 1 are written to --records (det_records.py: raw bytes,
published atomically with a manifest of the experiment identity and the writer's provenance; an
existing record set is never overwritten unless --replace-records); trajectory 2 is compared against
them.  --role compare (an independent process, possibly another implementation of the same
experiment): the warm-up step and both trajectories are compared against the records.

Verdict (``status`` in the JSON, the last log line, and the exit code):
  * strict (default): "pass" only if every expected record was compared and is bitwise equal, no
    record is missing / unexpected / malformed, the restored training state matches its capture,
    and the resolved execution plan equals the writer's (unless --allow-plan-change); otherwise
    "fail" and exit code 1.  The first differing (step, tensor, element) is reported.
  * --characterize: never stops on a value difference and records max abs / relative differences
    (fast mode, or fast vs deterministic records).  Its status is "characterization" (exit 0) and is
    never a validation pass; structural errors still fail (exit 1).
Unusable records (absent, incomplete, corrupt, other experiment) exit with code 2.

    python det_training.py --case P_S64_R16 --steps 4 --records /scratch/rec --role write --out ...
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import common  # noqa: F401,E402
from common import gpu_info, save  # noqa: E402
from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
import sparse_pooling  # noqa: E402
from sparse_pooling.optim import FusedAdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402
from det_records import RecordComparer, RecordError, RecordWriter, identity_mismatch  # noqa: E402
from train_state import TrainState  # noqa: E402

# name -> (D, G, H, M, S, R, D0, G2)
CASES = {
    # medium: Q = 64 supergroups, several normal-size chunks per supergroup (fwd 2048 / dX 4096 units)
    "M_S16_R4": (256, 64, 20_000_000, 1024, 16, 4, 128, 64),
    "M_S32_R8": (256, 64, 20_000_000, 1024, 32, 8, 128, 64),
    "M_S64_R16": (256, 64, 20_000_000, 1024, 64, 16, 128, 64),
    # small: production D / G / Q = 256, one chunk per supergroup, token tails
    "S_S16_R4": (1024, 1024, 1_000_000, 1000, 16, 4, 128, 64),
    "S_S32_R8": (1024, 1024, 1_000_000, 1000, 32, 8, 128, 64),
    "S_S64_R16": (1024, 1024, 1_000_000, 1000, 64, 16, 128, 64),
    # production
    "P_S16_R4": (1024, 1024, 100_000_000, 8192, 16, 4, 256, 256),
    "P_S32_R8": (1024, 1024, 100_000_000, 8192, 32, 8, 256, 256),
    "P_S64_R16": (1024, 1024, 100_000_000, 8192, 64, 16, 256, 256),
    # harness self-test (run_det_training.sh selftest): seconds per process
    "T_S16_R4": (256, 64, 400_000, 256, 16, 4, 64, 32),
}
class Model:
    def __init__(self, case, seed, precision, opt_mode, reduction, plan_kw):
        D, G, H, M, S, R, D0, G2 = case
        plan = PhysicalPlan(grad_precision=precision, reduction=reduction, **plan_kw)
        self.lay = SparsePoolingLayer(D, H, G, shared_neurons=8, pooling="mean", device="cuda", seed=seed,
                                      bias_std=0.2, plan=plan, connectivity=ConnectivityPlan(S, R, seed=seed))
        self.lay.enable_main_grad()
        if opt_mode == "fp32":
            self.opt = FusedAdamW([self.lay.weight, self.lay.bias], lr=1e-3, moments="fp32", master_weights=True)
        else:
            self.opt = FusedAdamW([self.lay.weight, self.lay.bias], lr=1e-3, moments="bf16",
                                  stochastic_rounding=(opt_mode == "sr"))
        self.opt._seed = 1000 + seed
        torch.manual_seed(17 + seed)
        self.proj = torch.nn.Linear(D0, D, device="cuda")
        self.head = torch.nn.Linear(G, G2, device="cuda")
        self.small = list(self.proj.parameters()) + list(self.head.parameters())
        self.opt2 = torch.optim.AdamW(self.small, lr=1e-3)

    def states(self):
        return [TrainState([self.lay.weight, self.lay.bias], self.opt), TrainState(self.small, self.opt2)]

    def step(self, x0, tgt, sink):
        lay = self.lay
        xin = self.proj(x0).bfloat16()
        xin.retain_grad()
        y = lay(xin)
        h = torch.tanh(self.head(y.float()))
        loss = ((h - tgt) ** 2).mean()
        loss.backward()
        sink("y", y)
        sink("loss", loss)
        sink("dX", xin.grad)
        sink("dW", lay.weight.main_grad)
        sink("db", lay.bias.main_grad)
        for n, p in zip(("proj.W", "proj.b", "head.W", "head.b"), self.small):
            sink(n + ".grad", p.grad)
        self.info = {k: lay.last_info.get(k) for k in ("reduction", "grouped_fwd", "grouped_dx", "grouped_dw",
                                                         "forward_kernel", "dx_kernel", "dw_kernel")}
        self.opt2.step()
        self.opt2.zero_grad(set_to_none=True)
        self.opt.step()
        sink("W", lay.weight)
        sink("b", lay.bias)
        for i, p in enumerate((lay.weight, lay.bias)):
            st = self.opt.state[p]
            for k in ("m", "v", "master"):
                if k in st:
                    sink(f"opt.{k}{i}", st[k])
        for n, p in zip(("proj.W", "proj.b", "head.W", "head.b"), self.small):
            sink(n, p)
            st = self.opt2.state[p]
            sink(n + ".exp_avg", st["exp_avg"])
            sink(n + ".exp_avg_sq", st["exp_avg_sq"])
            sink(n + ".step", st["step"])
        sink("opt.seed", torch.tensor([self.opt._seed], dtype=torch.int64))
        sink("opt.step", torch.tensor([self.opt.state[lay.weight]["step"]], dtype=torch.int64))


def batches(case, steps, seed):
    D, G, H, M, S, R, D0, G2 = case
    g = torch.Generator(device="cuda")
    g.manual_seed(4321 + seed)
    return [(torch.randn(M, D0, device="cuda", generator=g), 0.5 * torch.randn(M, G2, device="cuda", generator=g))
            for _ in range(steps)]




PLAN_KEYS = {"grouped_fwd": ("tokens", "warps", "chunk_units"),
             "grouped_dx": ("plan", "tokens", "warps", "reduction_group_warps", "chunk_units", "scatter", "kernel",
                            "epilogue"),
             "grouped_dw": ("cfg", "projection", "kernel")}


def resolved_plan(info: dict) -> dict:
    """The execution decisions that determine the arithmetic (from last_info; the same keys in every
    implementation since the deterministic-reduction phase)."""
    out = {k: info.get(k) for k in ("forward_kernel", "dx_kernel", "dw_kernel")}
    out["reduction"] = (info.get("reduction") or {}).get("mode")
    for sect, fields in PLAN_KEYS.items():
        d = info.get(sect)
        out[sect] = None if d is None else {f: d.get(f) for f in fields}
    return out


def _git(*a):
    root = Path(sparse_pooling.__file__).resolve().parent
    try:
        r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def provenance() -> dict:
    """The implementation and environment of this process (reported, not part of the identity)."""
    out = dict(package=str(Path(sparse_pooling.__file__).resolve().parent), commit=_git("rev-parse", "HEAD"),
               dirty=bool(_git("status", "--porcelain", "--untracked-files=no")), torch=torch.__version__,
               cuda_runtime=torch.version.cuda, gpu=gpu_info(),
               cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
    try:
        import triton
        out["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        out["triton"] = None
    try:  # the native build this process loads (identity / kind where the implementation reports one)
        from sparse_pooling.kernels import cuda_ops
        if hasattr(cuda_ops, "build_info"):
            out["build"] = cuda_ops.build_info()
        else:
            out["build"] = dict(library=str(cuda_ops.build_library()), nvcc_flags=cuda_ops.NVCC_FLAGS, kind="production")
    except Exception as e:  # noqa: BLE001
        out["build"] = dict(error=repr(e))
    return out


def identity(args, case) -> dict:
    """The experiment: the same identity must give the same records under an unchanged plan."""
    return dict(case=args.case, shape=dict(zip(("D", "G", "H", "M", "S", "R", "D0", "G2"), case)), seed=args.seed,
                steps=args.steps, precision=args.precision, opt_mode=args.opt_mode, reduction=args.reduction,
                plan=args.plan, warn_only=args.warn_only, model="det_training composed/1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--precision", default="split", choices=["split", "bf16"])
    ap.add_argument("--opt-mode", default="det", choices=["det", "sr", "fp32"],
                    help="det: BF16 moments, no SR; sr: stochastic rounding; fp32: FP32 moments + master")
    ap.add_argument("--reduction", default="auto", choices=["auto", "fast", "deterministic"])
    ap.add_argument("--plan", default="", help="extra PhysicalPlan fields k=v,...")
    ap.add_argument("--records", required=True)
    ap.add_argument("--role", default="write", choices=["write", "compare"])
    ap.add_argument("--replace-records", action="store_true",
                    help="write: replace an existing record set (only once the new one is complete)")
    ap.add_argument("--characterize", action="store_true",
                    help="never stop on a value difference: statistics (status 'characterization', never a pass)")
    ap.add_argument("--ignore-identity", default="",
                    help="characterize: identity fields allowed to differ from the records (e.g. reduction,warn_only)")
    ap.add_argument("--allow-plan-change", action="store_true",
                    help="compare: accept a resolved execution plan that differs from the writer's")
    ap.add_argument("--warn-only", action="store_true", help="torch deterministic algorithms with warn_only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=args.warn_only)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan_kw = {}
    for item in filter(None, args.plan.split(",")):
        k, _, v = item.partition("=")
        f = PhysicalPlan.__dataclass_fields__[k]
        plan_kw[k] = int(v) if f.type in ("int", int) else v
    case = CASES[args.case]
    ident = identity(args, case)
    prov = provenance()
    rec = Path(args.records)
    try:  # unusable records fail before any work
        writer = RecordWriter(rec, ident, prov, replace=args.replace_records) if args.role == "write" else None
        ignore = tuple(filter(None, args.ignore_identity.split(",")))
        ref = None if writer else RecordComparer(rec, ident, args.characterize, ignore)
    except RecordError as e:
        print(f"{args.case} seed {args.seed} role {args.role}: status=error ({e})", flush=True)
        sys.exit(2)
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    model = Model(case, args.seed, args.precision, args.opt_mode, args.reduction, plan_kw)
    data = batches(case, args.steps + 1, args.seed)
    build_s = time.perf_counter() - t0
    failures = []  # strict-mode failures (value differences, structure, restore, plan)
    stopped = False

    def run_traj(label, steps, sink_fn):
        nonlocal stopped
        res = []
        for s in steps:
            out = {}

            def sink(name, t):
                r = sink_fn(s, name, t)
                if r is not None:
                    out[name] = r

            ts = time.perf_counter()
            model.step(*data[s], sink)
            torch.cuda.synchronize()
            diffs = [n for n, r in out.items() if not r.get("bitwise", False)]
            errs = {n: out[n]["error"] for n in diffs if "error" in out[n]}
            res.append(dict(step=s, seconds=time.perf_counter() - ts, compared=len(out), differing=diffs,
                            errors=errs, first=(diffs[0], out[diffs[0]]) if diffs else None,
                            detail=out if args.characterize else {n: out[n] for n in diffs}))
            print(f"[{label}] step {s}: {len(out)} tensors compared, {len(diffs)} differ"
                  + (f" (first {diffs[0]} {out[diffs[0]]})" if diffs else ""), flush=True)
            if errs:
                failures.append(dict(kind="structure", trajectory=label, step=s, errors=errs))
            if diffs and not errs:
                failures.append(dict(kind="value", trajectory=label, step=s, first=(diffs[0], out[diffs[0]]),
                                     n_tensors=len(diffs)))
            if (diffs and not args.characterize) or errs:
                stopped = True
                return res
        return res

    def check_complete(label, cmp, steps):
        missing = cmp.unvisited(steps)
        if missing and not stopped:
            failures.append(dict(kind="incomplete", trajectory=label, missing=missing[:20], n_missing=len(missing)))
            print(f"[{label}] {len(missing)} reference records were never compared (first {missing[0]})", flush=True)

    traj_steps = range(1, args.steps + 1)
    try:
        if writer:  # warm-up step 0 and trajectory 1 are written, then published
            w0 = run_traj("warm", [0], writer)
        else:
            w0 = run_traj("warm", [0], ref)
            check_complete("warm", ref, [0])
        states = model.states()
        for st in states:
            st.capture()
        fp0 = [st.fingerprint() for st in states]
        plan_now = resolved_plan(model.info)
        if writer:
            t1 = run_traj("trajectory1_write", traj_steps, writer)
            writer.finalize(dict(resolved_plan=plan_now))
        else:
            t1 = run_traj("trajectory1_vs_records", traj_steps, ref) if not stopped else []
            check_complete("trajectory1_vs_records", ref, traj_steps)
    finally:
        if writer:
            writer.abort()  # no-op once published
    for st in states:
        st.restore()
    restored = [st.fingerprint() == f for st, f in zip(states, fp0)]
    if not all(restored):
        failures.append(dict(kind="restore", restored=restored))
    t2 = []
    if not stopped:
        cmp2 = RecordComparer(rec, ident, args.characterize, ignore)
        t2 = run_traj("trajectory2_vs_records", traj_steps, cmp2)
        check_complete("trajectory2_vs_records", cmp2, traj_steps)
    man = RecordComparer(rec, ident, args.characterize, ignore).manifest if (writer is None or writer.published) else {}
    ref_plan = man.get("resolved_plan")
    plan_equal = ref_plan == plan_now
    if not plan_equal and not args.allow_plan_change and not args.characterize:  # characterization: reported
        failures.append(dict(kind="plan", reference=ref_plan, current=plan_now))
    structural = [f for f in failures if f["kind"] != "value"]
    if args.characterize:
        status = "fail" if structural else "characterization"
    else:
        status = "pass" if not failures else "fail"
    lay = model.lay
    sg = lay._rt.sg
    out = dict(args=vars(args), status=status, mode="characterization" if args.characterize else "strict",
               failures=failures, identity=ident, provenance=prov,
               reference_provenance=man.get("provenance"),
               identity_mismatch=identity_mismatch(ident, man.get("identity") or {}), identity_ignored=ignore,
               resolved_plan=plan_now, reference_resolved_plan=ref_plan, resolved_plan_equal=plan_equal,
               build_seconds=build_s, info=model.info, bucket_stats=sg.bucket_stats,
               restore_fingerprint_equal=restored, warm=w0, trajectory1=t1, trajectory2=t2,
               all_bitwise=status == "pass",
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               snapshot_bytes=sum(st.nbytes() for st in states))
    save(args.out, out)
    first = next((f for f in failures), None)
    print(f"{args.case} seed {args.seed} {args.precision} {args.opt_mode} reduction {args.reduction} role {args.role}: "
          f"status={status} restore={restored} plan_equal={plan_equal}"
          + (f" first_failure={first}" if first else "") + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    sys.exit(0 if status in ("pass", "characterization") else 1)


if __name__ == "__main__":
    main()
