"""FusedAdamW in-place race (results/SCALE_DISPATCH_FUSED_BACKWARD.md section 3.4): the kernel
before the fix (optim.py at commit f6cac91, loaded from git) vs the current one.
  * distinct final states over identical runs (8 steps, deterministic rounding),
  * elements whose first moment equals b1 * m_prev exactly (gradient read as the fused zero),
  * TTGIR load / store layouts of G and MASTER, barrier count,
  * kernel time at the production size (800M BF16 weights + 100M bias elements, BF16 moments).

    python optim_race_check.py --out results/scale_dispatch_fused_backward/optimizer_race/optim_race_check
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch
import triton

import common  # noqa: F401
from bench_supergroups import git_dirty, git_sha
from common import save
from sparse_pooling import optim as fixed

OLD_COMMIT = "f6cac91"


def load_old():
    src = subprocess.run(["git", "show", f"{OLD_COMMIT}:sparse_pooling/optim.py"], capture_output=True, text=True,
                         check=True).stdout
    path = Path(tempfile.mkdtemp()) / "optim_old.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("optim_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trial(mod, n, master, runs, steps=8):
    torch.manual_seed(0)
    p0 = (torch.randn(n, device="cuda") * 0.1).bfloat16()
    gs = [torch.randn(n, device="cuda") * 1e-3 for _ in range(steps)]
    finals, stale = {}, 0
    for _ in range(runs):
        p = torch.nn.Parameter(p0.clone())
        p.main_grad = torch.zeros(n, device="cuda")
        opt = mod.FusedAdamW([p], lr=1e-3, moments="fp32" if master else "bf16", master_weights=master,
                             stochastic_rounding=False)
        for g in gs:
            st = opt.state.get(p, {})
            m0 = st["m"].clone() if st else None
            p.main_grad.copy_(g)
            opt.step()
            if m0 is not None:
                m = opt.state[p]["m"]
                stale += int(((m == (0.9 * m0.float()).to(m.dtype)) & (g != 0) & (m0 != 0)
                              & ((0.1 * g).abs() > 2.0**-6 * (0.9 * m0.float()).abs())).sum())
        key = (int(p.detach().view(torch.int16).long().sum()), int(opt.state[p]["m"].float().double().sum() * 1e12))
        finals[key] = finals.get(key, 0) + 1
    return {"runs": runs, "distinct_final_states": len(finals), "state_counts": sorted(finals.values(), reverse=True),
            "elements_with_gradient_read_as_zero": stale}


def layouts(mod, master):
    n = 812_345
    P = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    G = torch.zeros(n, device="cuda")
    M = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
    V = torch.zeros_like(M)
    MA = torch.zeros(n, device="cuda")
    k = mod._adamw_kernel[(triton.cdiv(n, 2048),)](P, G, M, V, MA if master else P, n, 1e-3, 0.9, 0.999, 1e-8, 0.0,
                                                    0.1, 0.001, 1, HAS_MASTER=master, STOCH=False, BLOCK=2048,
                                                    num_warps=8)
    ir = k.asm["ttgir"]
    lay = dict(re.findall(r"(#blocked\d*) = #ttg.blocked<\{sizePerThread = \[(\d+)\]", ir))
    ops = [re.sub(r"\s*loc\(#loc\d+\)", "", ln.strip()) for ln in ir.splitlines() if "tt.load" in ln or "tt.store" in ln]
    return {"sizePerThread": lay, "loads_stores": ops, "barriers": ir.count("gpu.barrier")}


def timing(mod, reps=10):
    out = {}
    for name, n in (("weight_800M", 800_000_000), ("bias_100M", 100_000_000)):
        p = torch.nn.Parameter(torch.zeros(n, device="cuda", dtype=torch.bfloat16))
        p.main_grad = torch.zeros(n, device="cuda")
        opt = mod.FusedAdamW([p], lr=1e-4, moments="bf16")
        opt.step()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            opt.step()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        out[name] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts)}
        del p, opt
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    old = load_old()
    res = {"commit": git_sha(), "dirty": git_dirty(), "old_commit": OLD_COMMIT, "triton": triton.__version__,
           "torch": torch.__version__, "trials": {}, "ttgir": {}, "timing": {}}
    for name, mod in (("before_fix", old), ("fixed", fixed)):
        for master in (False, True):
            res["ttgir"][f"{name}_master{int(master)}"] = layouts(mod, master)
            for n in (812_345, 819_200, 4_096_007):
                r = trial(mod, n, master, args.runs)
                res["trials"][f"{name}_master{int(master)}_n{n}"] = r
                print(f"{name:10s} master={master!s:5s} n={n:>9d}: distinct final states {r['distinct_final_states']:>2d} "
                      f"of {r['runs']} runs {r['state_counts']}; gradient read as zero: "
                      f"{r['elements_with_gradient_read_as_zero']}", flush=True)
    for name, mod in (("before_fix", old), ("fixed", fixed), ("before_fix_again", old), ("fixed_again", fixed)):
        res["timing"][name] = timing(mod)
        print(name, res["timing"][name], flush=True)
    save(args.out, res)


if __name__ == "__main__":
    main()
