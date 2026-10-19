"""Repeatability of complete composed-model training with PhysicalPlan.reduction (Track A,
results/DETERMINISM_METRICS_DW_PROJECTION.md section 5).

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

--role write: trajectory 1 writes every record to --records (raw bytes + SHA-1); trajectory 2 is
compared against them.  --role compare (a second, independent process): both trajectories are
compared against the records of the writing process.  Comparison is exact (bitwise, streamed in
chunks: no second copy of 8 GB state on the host), and reports the first differing (step, tensor,
element).  With --characterize the run never stops on a difference and records max abs / relative
differences instead (fast mode, or fast vs deterministic records via --records of a det run).

    python det_training.py --case P_S64_R16 --steps 4 --records /scratch/rec --role write --out ...
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: F401,E402
from common import gpu_info, save  # noqa: E402
from sparse_pooling import PhysicalPlan, SparsePoolingLayer  # noqa: E402
from sparse_pooling.kernels import cuda_ops  # noqa: E402
from sparse_pooling.optim import FusedAdamW  # noqa: E402
from sparse_pooling.supergroups import ConnectivityPlan  # noqa: E402
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
}
CHUNK = 1 << 26  # bytes per streamed comparison chunk


def bits_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().contiguous().reshape(-1)
    v = t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    return v


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


class Writer:
    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        self.meta = {}

    def __call__(self, step, name, t):
        b = bits_np(t)
        f = self.d / f"s{step}_{name}.bin"
        h = hashlib.sha1()
        with open(f, "wb") as fh:
            n = b.numel()
            per = max(1, CHUNK // b.element_size())
            for i in range(0, n, per):
                a = b[i:i + per].cpu().numpy()
                fh.write(a.tobytes())
                h.update(a.tobytes())
        self.meta[f"s{step}_{name}"] = dict(dtype=str(t.dtype), shape=list(t.shape), sha1=h.hexdigest())
        return None


class Comparer:
    """Streamed exact comparison against a record file; with characterize, difference statistics."""

    def __init__(self, d: Path, characterize: bool):
        self.d = d
        self.meta = json.load(open(d / "meta.json"))["records"]
        self.characterize = characterize

    def __call__(self, step, name, t):
        key = f"s{step}_{name}"
        if key not in self.meta:
            return dict(missing=True)
        b = bits_np(t)
        n = b.numel()
        per = max(1, CHUNK // b.element_size())
        first = None
        ndiff = 0
        stats = dict(max_abs=0.0, sq=0.0, ref_sq=0.0, nonfinite_mismatch=0)
        npdt = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}[b.element_size()]
        with open(self.d / f"{key}.bin", "rb") as fh:
            for i in range(0, n, per):
                a = b[i:i + per]
                ref = torch.from_numpy(np.frombuffer(fh.read(a.numel() * a.element_size()), dtype=npdt).copy()).to(a.device)
                ne = a != ref
                k = int(ne.sum())
                if k:
                    ndiff += k
                    if first is None:
                        first = i + int(torch.nonzero(ne)[0])
                    if self.characterize and t.is_floating_point():
                        x = a.view(t.dtype).double()
                        y = ref.view(t.dtype).double()
                        fin = torch.isfinite(x) & torch.isfinite(y)
                        dd = (x - y)[fin]
                        stats["max_abs"] = max(stats["max_abs"], float(dd.abs().max()) if dd.numel() else 0.0)
                        stats["sq"] += float((dd * dd).sum())
                        stats["nonfinite_mismatch"] += int((torch.isfinite(x) != torch.isfinite(y)).sum())
                if self.characterize and t.is_floating_point():
                    y = ref.view(t.dtype).double()
                    y = y[torch.isfinite(y)]
                    stats["ref_sq"] += float((y * y).sum())
        out = dict(bitwise=ndiff == 0, n=n, n_diff=ndiff)
        if first is not None:
            out["first_index"] = first
            if self.characterize and t.is_floating_point():
                out.update(max_abs=stats["max_abs"], rel_l2=(stats["sq"] / max(stats["ref_sq"], 1e-300)) ** 0.5,
                           rms=(stats["sq"] / n) ** 0.5, nonfinite_mismatch=stats["nonfinite_mismatch"])
        return out


def batches(case, steps, seed):
    D, G, H, M, S, R, D0, G2 = case
    g = torch.Generator(device="cuda")
    g.manual_seed(4321 + seed)
    return [(torch.randn(M, D0, device="cuda", generator=g), 0.5 * torch.randn(M, G2, device="cuda", generator=g))
            for _ in range(steps)]


def env_info():
    lib = str(cuda_ops.build_library())
    try:
        import triton
        tv = triton.__version__
    except Exception:  # noqa: BLE001
        tv = None
    return dict(torch=torch.__version__, cuda=torch.version.cuda, triton=tv, library=lib,
                nvcc_flags=cuda_ops.NVCC_FLAGS, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, gpu=gpu_info())


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
    ap.add_argument("--characterize", action="store_true", help="never stop on a difference: statistics")
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
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    model = Model(case, args.seed, args.precision, args.opt_mode, args.reduction, plan_kw)
    data = batches(case, args.steps + 1, args.seed)
    build_s = time.perf_counter() - t0
    rec = Path(args.records)
    traj = []
    ok = True

    def run_traj(label, sink_fn):
        nonlocal ok
        res = []
        for s in range(args.steps + 1):
            if label != "warm" and s == 0:
                continue
            if label == "warm" and s > 0:
                break
            out = {}

            def sink(name, t):
                r = sink_fn(s, name, t)
                if r is not None:
                    out[name] = r

            ts = time.perf_counter()
            model.step(*data[s], sink)
            torch.cuda.synchronize()
            diffs = [n for n, r in out.items() if not r.get("bitwise", True)]
            res.append(dict(step=s, seconds=time.perf_counter() - ts, compared=len(out),
                            differing=diffs, first=(diffs[0], out[diffs[0]]) if diffs else None,
                            detail=out if args.characterize else {n: out[n] for n in diffs}))
            print(f"[{label}] step {s}: {len(out)} tensors compared, {len(diffs)} differ"
                  + (f" (first {diffs[0]} {out[diffs[0]]})" if diffs else ""), flush=True)
            if diffs and not args.characterize:
                ok = False
                return res
        return res

    writer = Writer(rec) if args.role == "write" else None
    # warm-up step 0 (writes / compares the post-step-0 state: identical construction)
    w0 = run_traj("warm", writer if writer else Comparer(rec, args.characterize))
    states = model.states()
    for st in states:
        st.capture()
    fp0 = [st.fingerprint() for st in states]
    if writer:
        t1 = run_traj("trajectory1_write", writer)
        json.dump({"records": writer.meta}, open(rec / "meta.json", "w"))
    else:
        t1 = run_traj("trajectory1_vs_records", Comparer(rec, args.characterize))
    for st in states:
        st.restore()
    restored = [st.fingerprint() == f for st, f in zip(states, fp0)]
    t2 = run_traj("trajectory2_vs_records", Comparer(rec, args.characterize)) if (ok or args.characterize) else []
    lay = model.lay
    sg = lay._rt.sg
    out = dict(args=vars(args), case=dict(zip(("D", "G", "H", "M", "S", "R", "D0", "G2"), case)), env=env_info(),
               build_seconds=build_s, info=model.info, bucket_stats=sg.bucket_stats,
               restore_fingerprint_equal=restored, warm=w0, trajectory1=t1, trajectory2=t2,
               all_bitwise=ok and all(not r["differing"] for r in w0 + t1 + t2),
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               snapshot_bytes=sum(st.nbytes() for st in states))
    if writer:
        out["record_sha1"] = writer.meta
    save(args.out, out)
    print(f"{args.case} seed {args.seed} {args.precision} {args.opt_mode} reduction {args.reduction} role {args.role}: "
          f"all_bitwise={out['all_bitwise']} restore={restored} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
