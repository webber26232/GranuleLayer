"""Fusion screening measurements (Track F0, results/SCALE_DISPATCH_FUSED_BACKWARD.md section 7):
global dX flush patterns of candidate fused schedules and the FP32 dW / db partial
read-modify-write pass, in isolation (kernels/cuda/screen_fusion.cu).  Observations, not
additive cost predictions.

Flush patterns use the real supergroup metadata (feat, per-supergroup unit counts) of the
H = 100M graph; every result is verified exactly (contribution 1.0 per CTA and token:
dXT[f, t] = number of CTAs covering (f, t)).

    python screen_fusion.py --out results/scale_dispatch_fused_backward/screen_fusion
"""
from __future__ import annotations

import argparse
import ctypes
import statistics

import torch

import common  # noqa: F401
from bench_supergroups import git_dirty, git_sha
from common import gpu_info, save
from sparse_pooling import PhysicalPlan, SparsePoolingLayer
from sparse_pooling.kernels import cuda_ops
from sparse_pooling.kernels.cuda_ops import CudaBackend, _ptr
from sparse_pooling.supergroups import ConnectivityPlan


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts), "n": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    be = CudaBackend.get(strict=True)
    lib = be.lib
    lib.sp_screen_flush.restype = ctypes.c_int
    lib.sp_screen_flush.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_float,
                                    ctypes.c_void_p]
    lib.sp_screen_rmw.restype = ctypes.c_int
    lib.sp_screen_rmw.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    D, G, B, S = 1024, 1024, args.batch, args.S
    lay = SparsePoolingLayer(D, args.hidden, G, shared_neurons=8, pooling="mean", device="cuda",
                             plan=PhysicalPlan(cuda_input_supergroup_backend="grouped"),
                             connectivity=ConnectivityPlan(S, args.R, seed=0))
    rt = lay._runtime()
    sg = rt.sg
    feat = sg.feat.contiguous()
    b = sg.graph.buckets.cpu().long()
    n_q = torch.bincount(b[:, 0], weights=b[:, 3].double(), minlength=sg.Q).long()  # valid work units per q
    params = {"weight_bytes": lay.weight.numel() * 2, "bias_bytes": lay.bias.numel() * 2,
              "index_bytes": lay.indices.numel() * 2, "units": int(n_q.sum())}
    _, ch = sg.item_chunks(cuda_ops.SG_CHUNK_UNITS["dx"])
    del lay, rt
    torch.cuda.empty_cache()
    dXT = torch.zeros((D, B), device="cuda", dtype=torch.float32)

    def cta_list(units_per_cta, hot=False):
        nc = (n_q + units_per_cta - 1) // units_per_cta
        q = torch.repeat_interleave(torch.arange(sg.Q), nc)
        if hot:
            q = torch.zeros_like(q)
        return q.to(torch.int32).cuda()

    def expected(cta_q):
        cnt = torch.bincount(cta_q.long(), minlength=sg.Q).double()
        e = torch.zeros(D, dtype=torch.float64, device="cuda")
        e.index_add_(0, feat.long().reshape(-1), cnt.repeat_interleave(S))
        return e[:, None].expand(D, B).float()

    rows = []

    def flush(name, cta_q, Tw, cta_windows, threads=256, reps=args.reps, note=""):
        nwin = -(-B // Tw)
        ncta = cta_q.numel()
        atomics = ncta * S * B  # one per (feature, token) per CTA covering the token
        fn = lambda: lib.sp_screen_flush(_ptr(feat), _ptr(cta_q), ncta, S, B, Tw, cta_windows, threads,  # noqa: E731
                                         _ptr(dXT), 1.0, stream)
        dXT.zero_()
        assert fn() == 0
        torch.cuda.synchronize()
        ok = bool(torch.equal(dXT, expected(cta_q)))
        t = {"median_ms": float("nan")}
        if ok:
            def run():
                dXT.zero_()
                fn()
            z = timed(lambda: dXT.zero_(), reps)
            t = timed(run, reps)
            t["zero_ms"] = z["median_ms"]
            t["flush_ms_minus_zero"] = t["median_ms"] - z["median_ms"]
        r = {"name": name, "ctas": ncta, "Tw": Tw, "windows": nwin, "windows_per_cta": cta_windows,
             "grid_y": -(-nwin // cta_windows), "atomics": atomics, "atomics_G": atomics / 1e9,
             "update_payload_GB": atomics * 4 / 1e9, "verified": ok, "note": note, **t}
        rows.append(r)
        print(f"{name:44s} ctas {ncta:>7d} Tw {Tw:>5d} win/cta {cta_windows:>4d} atomics {atomics / 1e9:7.3f} G "
              f"verified {ok}  {t['median_ms']:9.1f} ms", flush=True)

    # current separate dX (S = 16: T = 64, one window per CTA, 4096-unit chunks)
    flush("current dX (chunks 4096, T 64)", ch[:, 0].contiguous(), 64, 1, note="grid (chunk, token tile)")
    # F1: dW-owned CTAs loop all token windows (dW grid order: consecutive CTAs share q)
    for upc in (16, 32, 64):
        cq = cta_list(upc)
        flush(f"F1 dW-owned {upc} units/CTA, Tw 64", cq, 64, -(-B // 64))
        if upc == 16:
            flush(f"F1 dW-owned {upc} units/CTA, Tw 16", cq, 16, -(-B // 16))
            flush(f"F1 hot (all CTAs q = 0) {upc} units/CTA, Tw 64", cta_list(upc, hot=True), 64, -(-B // 64),
                  note="hot working set: S features x B tokens")
    # F2: coarse chunks (4096 units) loop all windows of Tw tokens inside one CTA
    for Tw in (512, 1024):
        flush(f"F2 coarse chunks 4096, Tw {Tw}, serial windows", ch[:, 0].contiguous(), Tw, -(-B // Tw))
    # dW / db partial read-modify-write pass (F2: once per token window)
    n = params["units"] * (256 + 32)  # FP32 dW (32 neurons x 8) + db per unit
    buf = torch.zeros(-(-n // 4) * 4, device="cuda", dtype=torch.float32)
    assert lib.sp_screen_rmw(_ptr(buf), buf.numel() // 4, 1.0, stream) == 0
    torch.cuda.synchronize()
    ok = bool((buf == 1.0).all())
    t = timed(lambda: lib.sp_screen_rmw(_ptr(buf), buf.numel() // 4, 1.0, stream), 5)
    rmw = {"floats": n, "bytes_read_plus_written": 8 * n, "verified": ok, **t,
           "achieved_GBps": 8 * n / (t["median_ms"] * 1e6)}
    print(f"dW/db RMW pass: {n / 1e6:.0f} M floats, {8 * n / 1e9:.2f} GB moved, {t['median_ms']:.1f} ms "
          f"({rmw['achieved_GBps']:.0f} GB/s) verified {ok}", flush=True)
    save(args.out, {"commit": git_sha(), "dirty": git_dirty(), "library": str(cuda_ops.build_library()),
                    "args": vars(args), "graph": {"Q": sg.Q, "S": S, "R": args.R, "D": D, "G": G, "B": B,
                                                  "units_per_q_min": int(n_q.min()), "units_per_q_max": int(n_q.max()),
                                                  "dx_chunks": int(ch.shape[0]), **params},
                    "flush": rows, "rmw_pass": rmw, "gpu": gpu_info()})


if __name__ == "__main__":
    main()
