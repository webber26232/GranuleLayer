"""Memory-compute trade-off: Strategy A (store hidden activations, ordinary
autograd) vs B (compact pooling statistics + recompute) vs C (aggressive
recomputation of statistics/winners).

Strategy A is only feasible at small hidden widths; it is the correctness and
performance reference, not a production path.

    python sparse_pooling/benchmarks/bench_tradeoff.py --hidden 262144 --batch 2048
"""
from __future__ import annotations

import argparse
import gc

import torch

from common import save, time_cuda
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.reference import pytorch_reference as ref

CASES = [
    ("Max", "max", "save_winners", "minimal_stats"),
    ("Top-8", {"type": "topk_mean", "k": 8}, "save_winners", "recompute_stats"),
    ("Mean(ReLU)", "mean", "minimal_stats", "recompute_stats"),
    ("LogMeanExp", {"type": "logmeanexp", "temperature": 0.5}, "minimal_stats", "recompute_stats"),
    ("Softmax-weighted", {"type": "softmax_weighted", "beta": 1.0}, "minimal_stats", "recompute_stats"),
]


def measure(fwd, bwd):
    holder = {}

    def f():
        holder["y"] = fwd()

    def b():
        bwd(holder["y"])

    tf = time_cuda(f, warmup=1, iters=3)["median_ms"]
    ts = []
    for _ in range(4):
        f()
        torch.cuda.synchronize()
        ts.append(time_cuda(lambda: b(), warmup=0, iters=1)["median_ms"])
    tb = sorted(ts[1:])[1]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    f()
    b()
    torch.cuda.synchronize()
    return tf, tb, torch.cuda.max_memory_allocated() - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = []
    for name, pooling, pol_b, pol_c in CASES:
        lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=args.C, pooling=pooling, device="cuda",
                                 checkpoint_policy=pol_b)
        # FP32 tensor holding BF16-exact values: kernels see the same BF16 operands,
        # and the returned gradients are not rounded to BF16 (error is then measurable)
        x = torch.randn(args.batch, 1024, device="cuda").bfloat16().float().requires_grad_(True)
        dy = torch.randn(args.batch, 1024, device="cuda")
        # FP64 reference gradient for numerical error
        w64 = lay.weight.detach().double().requires_grad_(True)
        b64 = lay.bias.detach().double().requires_grad_(True)
        x64 = x.detach().double().requires_grad_(True)
        y64 = ref.reference_forward(x64, lay.indices, w64, b64, lay.layout, lay.cfg)
        (y64 * dy.double()).sum().backward()
        g64 = (x64.grad.detach(), w64.grad.detach())
        del y64, x64, w64, b64
        gc.collect()
        torch.cuda.empty_cache()

        def errs(gx, gw):
            ex = ((gx.double() - g64[0]).norm() / g64[0].norm()).item()
            ew = ((gw.double() - g64[1]).norm() / g64[1].norm()).item()
            return ex, ew

        # ---- Strategy A: materialised hidden layer + autograd (fp32 z from bf16 operands)
        wA = lay.weight.detach().float().requires_grad_(True)
        bA = lay.bias.detach().float().requires_grad_(True)

        def fA():
            return ref.reference_forward(x, lay.indices, wA, bA, lay.layout, lay.cfg)

        def bA_(y):
            (y * dy).sum().backward()

        try:
            tf, tb, peak = measure(fA, bA_)
            x.grad = None
            wA.grad = None
            fA_out = fA()
            bA_(fA_out)
            eA = errs(x.grad, wA.grad)
            recs.append(dict(op=name, strategy="A store hidden", fwd_ms=tf, bwd_ms=tb, total_ms=tf + tb,
                             peak_extra_bytes=peak, dx_err=eA[0], dw_err=eA[1]))
        except torch.cuda.OutOfMemoryError:
            recs.append(dict(op=name, strategy="A store hidden", error="OOM"))
        x.grad = None
        del wA, bA
        gc.collect()
        torch.cuda.empty_cache()
        # ---- Strategies B and C: fused kernels
        for strat, pol in (("B compact stats", pol_b), ("C recompute", pol_c)):
            lay.cfg.checkpoint_policy = pol
            w = lay.weight.detach().clone().float().requires_grad_(True)
            lay2_w, lay2_b = lay.weight, lay.bias
            # FP32 copies so gradients are not rounded to BF16 for the error measurement
            lay.weight = torch.nn.Parameter(lay2_w.detach().float())
            lay.bias = torch.nn.Parameter(lay2_b.detach().float())

            def fB():
                return lay(x)

            def bB(y):
                (y * dy).sum().backward()

            tf, tb, peak = measure(fB, bB)
            x.grad = None
            lay.weight.grad = None
            bB(fB())
            e = errs(x.grad, lay.weight.grad)
            recs.append(dict(op=name, strategy=strat, policy=lay.last_info.get("policy"), fwd_ms=tf, bwd_ms=tb,
                             total_ms=tf + tb, peak_extra_bytes=peak,
                             checkpoint_bytes=lay.last_info.get("checkpoint_bytes"), dx_err=e[0], dw_err=e[1]))
            lay.weight, lay.bias = lay2_w, lay2_b
            x.grad = None
        for r in recs[-3:]:
            if "error" in r:
                print(f"{r['op']:16s} {r['strategy']:16s} {r['error']}", flush=True)
            else:
                print(f"{r['op']:16s} {r['strategy']:16s} fwd {r['fwd_ms']:8.2f} bwd {r['bwd_ms']:8.2f} "
                      f"total {r['total_ms']:8.2f} ms  peak+ {r['peak_extra_bytes'] / 2**20:9.1f} MiB  "
                      f"dx_err {r['dx_err']:.1e} dw_err {r['dw_err']:.1e}", flush=True)
        del lay, x
        gc.collect()
        torch.cuda.empty_cache()
    save(args.out or f"tradeoff_H{args.hidden}_B{args.batch}", {"args": vars(args), "records": recs})


if __name__ == "__main__":
    main()
