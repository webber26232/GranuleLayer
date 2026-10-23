"""Layer factories of the K64 large-C / index-free / L1 study for ``bench_supergroups.py --layer-factory
k64_factory:<name>`` (D = --input-dim, H = --hidden, G = --groups, logical C = --C, S / R / graph seed in
``conn``, init seed = --init-seed; ``args.factory`` JSON).

* ``kgx``     the experimental K = 64 layer (experimental/kgx_layer.KgxLayer): generator of kgen_graph
              (quantum 1), weights drawn per valid neuron (init_parameters_by_neuron: identical valid-neuron
              draws for every C), zero bias.  Its execution variants (kgen / lowered reference, direct,
              idf_rmw, idf; physical tile) are selected per measured variant with --layer-variants on the
              SAME tensors.
* ``sparse``  a frozen-mask sparse layer (dense masked / tile / CSR) built from a saved checkpoint and mask
              (experimental/kgx_sparse.py; args.factory: checkpoint, format, ...).

Each factory validates the logical graph structurally (outside every timed region) and stores the report
in ``layer.graph_stats["kcsr_validation"]``.
"""
from __future__ import annotations

import time

import torch

from sparse_pooling.config import PhysicalPlan


def _plan(args):
    return PhysicalPlan(grad_precision=args.grad_precision)


def kgx(args, conn):
    from sparse_pooling.experimental.kgx_layer import KgxLayer

    t0 = time.perf_counter()
    f = args.factory
    lay = KgxLayer(args.input_dim, args.hidden, args.groups, args.C, conn, variant=f.get("variant", "direct"),
                   tile=int(f.get("tile", 256)), device="cuda", seed=args.init_seed, plan=_plan(args),
                   init=f.get("init", "by_neuron"), keep_indices=f.get("keep_indices", True))
    t1 = time.perf_counter()
    lay.graph_stats.update(kcsr_validation=lay.kcsr_validate(), layer_build_seconds=t1 - t0)
    return lay


def sparse(args, conn):
    from sparse_pooling.experimental import kgx_sparse as SP

    t0 = time.perf_counter()
    lay = SP.build_from_checkpoint(args, conn, _plan(args))
    lay.graph_stats.update(kcsr_validation=lay.kcsr_validate(), layer_build_seconds=time.perf_counter() - t0)
    return lay
