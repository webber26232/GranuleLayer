"""Generator validation summaries (supergroups.validate) for the experiment graphs at H = 100M:
balance, coverage, record repetition, metadata and generation cost; plus the same statistics
for the unrestricted graph and for the "independent_subsets" diagnostic.

    python gen_stats.py --out results/input_supergroups/generator_stats.json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import common  # noqa: F401
from sparse_pooling.layout import SparseLayout, init_parameters
from sparse_pooling.supergroups import ConnectivityPlan, SgExec, generate_indices, validate

POINTS = [(8, 2), (16, 4), (32, 8), (64, 16), (128, 32), (256, 64), (1024, 1), (16, 16), (32, 16), (128, 16),
          (16, 8), (16, 32), (64, 8), (64, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=100_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = []
    for C in (8, 64):
        L = SparseLayout(1024, args.hidden, 1024, 8, C, True)
        # unrestricted reference graph (existing generator)
        t0 = time.perf_counter()
        idx, _, _ = init_parameters(L, "cuda", seed=0)
        torch.cuda.synchronize()
        t_unres = time.perf_counter() - t0
        rep = validate(L, idx, None)
        res.append({"C": C, "graph": "unrestricted", "init_seconds_incl_weights": t_unres, **rep})
        print(C, "unrestricted", rep["distinct_records"], rep.get("record_pair_coverage"), flush=True)
        del idx
        for gen in ("balanced_permutation", "independent_subsets"):
            for S, R in (POINTS if gen == "balanced_permutation" else [(16, 4), (64, 16)]):
                torch.cuda.empty_cache()
                idx, graph, st = generate_indices(L, ConnectivityPlan(S, R, gen, 0), "cuda")
                rep = validate(L, idx, graph)
                ex = SgExec(L, graph.feat, graph.buckets)
                ex.item_chunks(2048)
                ex.item_chunks(4096)
                ex.unit_chunks(16)
                rep.update(st)
                rep["work_list_bytes"] = ex.runtime_bytes()
                rep["mean_bucket_units"] = ex.mean_bucket_units()
                res.append({"C": C, "graph": gen, **rep})
                print(C, gen, S, R, f"{st['generation_seconds']:.2f}s", rep["distinct_records"],
                      f"{rep['record_pair_coverage']:.4f} / sets {rep['set_pair_coverage']:.4f} "
                      f"(expected {rep['set_pair_coverage_expected']:.4f})", rep["per_supergroup"]["neuron_imbalance"],
                      flush=True)
                del idx, graph, ex
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
