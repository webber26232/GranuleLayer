"""Layer factories of the K/C/S/R sweep for ``bench_supergroups.py --layer-factory kcsr_factory:<name>``.

Every factory builds the layer from the benchmark's arguments (D = --input-dim, H = --hidden, G = --groups,
C = --C logical, S / R / graph seed in ``conn``, init seed = --init-seed) and ``args.factory`` (JSON):

* ``lowered``      K = 8 logical C = 128 / 256 (--C) with the generalized generator (quantum 1), run as a
                   native C = 64 layer (experimental/lowering.py);
* ``kgen``         the experimental generalized-fan-in layer, K = args.factory["K"], logical C = --C,
                   generalized generator (quantum 1);
* ``kgen_native``  the matched-backend K = 8 control: the production layer (same seeds as the native
                   run) converted to a KgenLayer with exactly its graph and parameters.

Each factory validates the logical graph structurally (outside every timed region) and stores the
report in ``layer.graph_stats["kcsr_validation"]``.  Bias: zeros (bias_std = 0) everywhere.
"""
from __future__ import annotations

import time

import torch

import kcsr_cases as KC
from sparse_pooling import SparsePoolingLayer
from sparse_pooling.config import PhysicalPlan


def _plan(args):
    return PhysicalPlan(grad_precision=args.grad_precision)


def lowered(args, conn):
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental.lowering import build_logical, lower

    t0 = time.perf_counter()
    L, idx, w, b, bk, feat, spec, st = build_logical(args.input_dim, args.hidden, args.groups, args.C, conn, "cuda",
                                                     seed=args.init_seed, bias_std=KC.BIAS_STD)
    val = KG.validate(L.input_dim, L.counts_tensor("cuda"), L.block, 8, idx, feat, bk, spec.R)
    t1 = time.perf_counter()
    lay = lower(L, idx, w, b, bk, feat, spec, plan=_plan(args), device="cuda")
    del idx, w, b
    torch.cuda.empty_cache()
    lay.graph_stats.update(st, kcsr_validation=val, logical_build_seconds=t1 - t0,
                           lowering_seconds=time.perf_counter() - t1)
    return lay


def kgen(args, conn):
    from sparse_pooling.experimental.kgen_layer import KgenLayer

    t0 = time.perf_counter()
    K = int(args.factory["K"])
    lay = KgenLayer(args.input_dim, args.hidden, args.groups, K, args.C, conn, device="cuda", seed=args.init_seed,
                    bias_std=KC.BIAS_STD, plan=_plan(args))
    t1 = time.perf_counter()
    lay.graph_stats.update(kcsr_validation=lay.kcsr_validate(), layer_build_seconds=t1 - t0)
    lay._runtime()  # work lists (host build; recorded)
    lay.graph_stats["work_lists"] = dict(lay._rt.sg.stats)
    return lay


def kgen_native(args, conn):
    from sparse_pooling.experimental.kgen_layer import from_native

    t0 = time.perf_counter()
    nat = SparsePoolingLayer(args.input_dim, args.hidden, args.groups, shared_neurons=args.C, pooling="mean",
                             checkpoint_policy="auto", backend="auto", device="cuda", plan=_plan(args),
                             seed=args.init_seed, connectivity=conn)
    t1 = time.perf_counter()
    lay = from_native(nat, plan=_plan(args))
    del nat
    torch.cuda.empty_cache()
    lay.graph_stats.update(kcsr_validation=lay.kcsr_validate(), native_build_seconds=t1 - t0,
                           conversion_seconds=time.perf_counter() - t1)
    lay._runtime()
    lay.graph_stats["work_lists"] = dict(lay._rt.sg.stats)
    return lay
