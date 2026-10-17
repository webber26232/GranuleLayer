"""Memory accounting for the full configuration: analytic model + measurement.

    python sparse_pooling/benchmarks/bench_memory.py --hidden 100000000 --batch 8192
"""
from __future__ import annotations

import argparse
import gc

import torch

from common import save
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PoolingConfig, human_bytes
from sparse_pooling.memory import format_budget, memory_budget
from sparse_pooling.optim import FusedAdamW

CASES = [
    ("max", "save_winners"),
    ({"type": "topk_mean", "k": 8}, "save_winners"),
    ({"type": "logmeanexp", "temperature": 0.5}, "minimal_stats"),
    ({"type": "softmax_weighted", "beta": 1.0}, "minimal_stats"),
    ({"levels": [{"type": "topk_mean", "k": 8, "neurons_per_shard": 16384},
                 {"type": "logmeanexp", "temperature": 0.5}]}, "save_winners"),
    ({"levels": [{"type": "topk_mean", "k": 8, "neurons_per_shard": 2048},
                 {"type": "logmeanexp", "temperature": 0.5}], "level1_checkpoint": "save"}, "save_winners"),
    ({"levels": [{"type": "topk_mean", "k": 8, "neurons_per_shard": 2048},
                 {"type": "logmeanexp", "temperature": 0.5}], "level1_checkpoint": "recompute"}, "save_winners"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--moments", default="bf16")
    ap.add_argument("--master", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = []
    for pooling, pol in CASES:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        lay = SparsePoolingLayer(1024, args.hidden, 1024, shared_neurons=args.C, pooling=pooling,
                                 checkpoint_policy=pol, device="cuda")
        lay.enable_main_grad()
        opt = FusedAdamW([lay.weight, lay.bias], lr=1e-4, moments=args.moments, master_weights=args.master)
        x = torch.randn(args.batch, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        try:
            for _ in range(2):
                y = lay(x)
                y.float().square().mean().backward()
                opt.step()
                x.grad = None
                del y
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n== {lay.cfg.describe()}: OOM ({str(e)[:120]})", flush=True)
            recs.append({"pooling": lay.cfg.describe(), "error": "OOM"})
            del lay, opt, x
            continue
        torch.cuda.synchronize()
        model = memory_budget(lay.layout, lay.cfg, args.batch, lay.last_info.get("policy", lay.effective_policy),
                              moments=args.moments,
                              master_weights=args.master)
        rec = {"pooling": lay.cfg.describe(), "policy": lay.last_info.get("policy", lay.effective_policy),
               "policy_fallback": lay.last_info.get("policy_fallback"),
               "level1_checkpoint": lay.cfg.level1_checkpoint if lay.cfg.is_semantic_hierarchy else None,
               "measured_peak_bytes": torch.cuda.max_memory_allocated(),
               "measured_steady_bytes": torch.cuda.memory_allocated(),
               "checkpoint_bytes": lay.last_info.get("checkpoint_bytes"),
               "optimizer_state_bytes": opt.state_bytes(), "model": model}
        recs.append(rec)
        print(f"\n== {rec['pooling']}  policy={rec['policy']}  l1ckpt={rec['level1_checkpoint']}")
        print(format_budget(model))
        print(f"measured peak {human_bytes(rec['measured_peak_bytes'])}, steady {human_bytes(rec['measured_steady_bytes'])}, "
              f"checkpoint {human_bytes(rec['checkpoint_bytes'] or 0)}", flush=True)
        del lay, opt, x
    save(args.out or f"memory_H{args.hidden}_B{args.batch}", {"args": vars(args), "records": recs})


if __name__ == "__main__":
    main()
