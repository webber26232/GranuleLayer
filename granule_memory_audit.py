"""Profile PyTorch CUDA allocation peaks for GranuleLayer, phase by phase.

Run from the repository root, e.g.:
  python granule_memory_audit.py --hidden 100000000 --C 64 --pooling mean

Measures PyTorch-managed GPU allocation/reservation, NOT physical DRAM/L2 traffic
or shared-memory consumption. Performs one warm training step first so lazy Adam
state and JIT kernels do not distort steady-state phase peaks.
"""
import argparse
import json

import torch

from sparse_pooling import SparsePoolingLayer
from sparse_pooling.optim import FusedAdamW

GIB = 2 ** 30


def snapshot():
    torch.cuda.synchronize()
    return {
        "allocated_GiB": torch.cuda.memory_allocated() / GIB,
        "reserved_GiB": torch.cuda.memory_reserved() / GIB,
    }


def measure_phase(name, fn):
    torch.cuda.synchronize()
    baseline = snapshot()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / GIB
    reserved_peak = torch.cuda.max_memory_reserved() / GIB
    after = snapshot()
    return {
        "phase": name,
        "allocated_before_GiB": round(baseline["allocated_GiB"], 4),
        "allocated_peak_GiB": round(peak, 4),
        "peak_extra_GiB": round(peak - baseline["allocated_GiB"], 4),
        "allocated_after_GiB": round(after["allocated_GiB"], 4),
        "reserved_peak_GiB": round(reserved_peak, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--C", type=int, choices=[8, 16, 32, 64], default=64)
    ap.add_argument("--pooling", choices=["max", "mean", "logmeanexp", "top8"], default="mean")
    ap.add_argument("--moments", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--master", action="store_true")
    ap.add_argument("--backend", choices=["auto", "triton", "cuda"], default="auto")
    ap.add_argument("--grad-precision", choices=["split", "bf16"], default="split")
    args = ap.parse_args()

    from sparse_pooling.config import PhysicalPlan

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(7)
    spec = {"type": "topk_mean", "k": 8} if args.pooling == "top8" else args.pooling
    device = "cuda"
    model = SparsePoolingLayer(
        input_dim=1024,
        hidden_dim=args.hidden,
        output_dim=1024,
        fan_in=8,
        shared_neurons=args.C,
        pooling=spec,
        checkpoint_policy="auto",
        backend=args.backend,
        device=device,
        dtype=torch.bfloat16,
        plan=PhysicalPlan(grad_precision=args.grad_precision,
                          input_layout="feature_major"),
    )
    model.enable_main_grad()
    opt = FusedAdamW([model.weight, model.bias], lr=1e-4,
                     moments=args.moments, master_weights=args.master)
    x = torch.randn(args.batch, args.seq, 1024,
                    device=device, dtype=torch.bfloat16, requires_grad=True)

    # Initialize lazy optimizer states and trigger JIT compilation before measuring.
    warm_y = model(x)
    warm_y.float().square().mean().backward()
    opt.step()
    x.grad = None
    del warm_y
    torch.cuda.synchronize()
    steady_state = snapshot()

    holder = {}

    def forward():
        holder["y"] = model(x)

    def backward():
        y = holder.pop("y")
        y.float().square().mean().backward()
        x.grad = None
        del y

    stages = [
        measure_phase("forward", forward),
        measure_phase("backward", backward),
        measure_phase("optimizer", opt.step),
    ]

    info = model.last_info
    result = {
        "device": torch.cuda.get_device_name(),
        "hidden": args.hidden,
        "tokens": args.batch * args.seq,
        "C": args.C,
        "pooling": args.pooling,
        "moments": args.moments,
        "master": args.master,
        "post_warmup_steady": steady_state,
        "checkpoint_MiB": round(info.get("checkpoint_bytes", 0) / 2 ** 20, 3),
        "workspace_reported_MiB": round(info.get("workspace_bytes", 0) / 2 ** 20, 3),
        "forward_backend": info.get("forward_backend"),
        "backward_backend": info.get("backward_backend"),
        "stages": stages,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
