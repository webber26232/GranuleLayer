"""Constrained configuration sweep (Section 'Required benchmarks').

Axes: shared_neurons C, batch tile, physical neuron shards (splits), input
layout, scheduling (token- vs weight-stationary), for Max / LogMeanExp / Top-8.

Notes on what each axis means for each backend:
  * CUDA (m16n8k8) forward: batch tile is 16 or 32 tokens (the feature-major
    X slab must fit in 99 KiB of shared memory: D * BM * 2 bytes); scheduling is
    always token-stationary (pooling state in registers).
  * Triton forward: batch tile 32..256; "weight_stationary" = each program owns
    a neuron chunk (physical split) and loops over ALL token tiles, so its
    weights stay hot in L1/L2 while partial pooling states for every
    (token, unit, split) go to the workspace (budget-checked).
Data-traffic columns are analytic (bytes the algorithm must move between L2
and the SMs); measured DRAM bytes need Nsight Compute (see profile_ncu.sh).

    python sparse_pooling/benchmarks/sweep_configs.py --hidden 10000000 --batch 8192
"""
from __future__ import annotations

import argparse
import gc
import math

import torch

from common import save, time_cuda, useful_tflops
from sparse_pooling.autograd import Level1Runtime
from sparse_pooling.config import LevelSpec, PhysicalPlan, human_bytes, state_bytes
from sparse_pooling.kernels import ops
from sparse_pooling.kernels.ops import WorkspaceBudgetError
from sparse_pooling.layout import SparseLayout, init_parameters

POOLS = {"max": LevelSpec("max"), "lme": LevelSpec("logmeanexp", temperature=0.5),
         "top8": LevelSpec("topk_mean", k=8)}


def traffic_model(L: SparseLayout, B: int, bm: int, ps: int, tt_all: bool, level: LevelSpec, packed: bool):
    """Analytic L2->SM bytes for one forward (no cache reuse beyond the tile)."""
    H = L.hidden_dim
    n_blocks = L.groups * L.blocks_per_group
    n_tiles = math.ceil(B / bm)
    w_bytes = n_blocks * L.block * 8 * 2
    b_bytes = n_blocks * L.block * 2
    i_bytes = n_blocks * 8 * 2
    # every token tile streams all parameters once
    weight_traffic = (w_bytes + b_bytes) * n_tiles
    index_traffic = i_bytes * n_tiles
    gather = n_blocks * B * 16  # 8 BF16 per (block, token) -> from smem slab (CUDA) or L1/L2 (Triton)
    ws = B * L.groups * ps * state_bytes(level, level.type == "max") if (ps > 1 or tt_all) else 0
    return {"weight_bytes": weight_traffic, "index_bytes": index_traffic, "gather_bytes": gather,
            "workspace_bytes": ws, "workspace_traffic": 2 * ws}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    B, H = args.batch, args.hidden
    from sparse_pooling.kernels.cuda_ops import CudaBackend

    be = CudaBackend.get()
    recs = []
    Cs = [8, 16, 32]
    for C in Cs:
        for packed in (True,):
            L = SparseLayout(1024, H, 1024, 8, C, True, packed=packed)
            idx, w, b = init_parameters(L, "cuda", seed=0)
            x = torch.randn(B, 1024, device="cuda").bfloat16()
            for pname, spec in POOLS.items():
                configs = []
                # CUDA m16n8k8 kernels (Top-K k <= 16 included)
                if True:
                    for bm in (16, 32):
                        for ps in (1, 4, 8, 16):
                            for lay in ("feature_major", "token_major"):
                                if args.quick and (ps not in (1, 8) or lay == "token_major" and bm == 16):
                                    continue
                                configs.append(("cuda", bm, ps, lay, "token_stationary"))
                for bm in (32, 64, 128, 256):
                    for ps in (1, 4, 8, 16):
                        for sched in ("token_stationary", "weight_stationary"):
                            if args.quick and ps not in (1, 8):
                                continue
                            if sched == "weight_stationary" and ps == 1:
                                continue  # one program per unit: no parallelism
                            if bm == 256 and spec.type == "topk_mean":
                                continue  # register spills
                            configs.append(("triton", bm, ps, "feature_major", sched))
                if C == 8:  # token-major only matters as a contrast; one Triton point
                    configs.append(("triton", 64, 1, "token_major", "token_stationary"))
                for backend, bm, ps, lay, sched in configs:
                    gc.collect()
                    torch.cuda.empty_cache()
                    n_tiles = math.ceil(B / bm)
                    plan = PhysicalPlan(phys_splits=ps, block_m=bm, cuda_block_m=bm if backend == "cuda" else 32,
                                        input_layout=lay,
                                        token_tiles_per_program=(n_tiles if sched == "weight_stationary" else 1))
                    rt = Level1Runtime(L, idx, spec, plan, None, args.budget_mib * 2**20,
                                       be if backend == "cuda" else None)
                    rec = {"C": C, "pool": pname, "backend": backend, "batch_tile": bm, "splits": ps,
                           "input_layout": lay, "scheduling": sched, "packed": packed}
                    rec.update(traffic_model(L, B, bm, ps, sched == "weight_stationary", spec, packed))
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        t = time_cuda(lambda: rt.forward(x, w, b, False), warmup=1, iters=3)
                        rec.update(fwd_ms=t["median_ms"], fwd_iqr_ms=t["iqr_ms"],
                                   tflops=useful_tflops(B, H, t["median_ms"]),
                                   peak_extra_bytes=torch.cuda.max_memory_allocated() - base,
                                   used_backend=rt.last_info.get("forward_backend"))
                    except WorkspaceBudgetError as e:
                        rec.update(error=f"workspace budget: {e}")
                    except torch.cuda.OutOfMemoryError:
                        rec.update(error="OOM")
                    recs.append(rec)
                    msg = (f"C={C} {pname:5s} {backend:6s} BM={bm:3d} splits={ps:2d} {lay[:5]} {sched[:5]}: "
                           + (f"{rec['fwd_ms']:8.1f} ms {rec['tflops']:5.1f} TF ws={human_bytes(rec['workspace_bytes'])}"
                              if "fwd_ms" in rec else rec["error"]))
                    print(msg, flush=True)
            del idx, w, b
    save(args.out or f"sweep_H{H}_B{B}", {"args": vars(args), "records": recs})


if __name__ == "__main__":
    main()
