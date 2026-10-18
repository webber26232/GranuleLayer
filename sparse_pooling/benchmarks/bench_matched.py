"""Matched-connectivity control: run a C = 64 supergroup graph through the C = 8 physical
implementation.  Every 64-neuron block becomes eight 8-neuron blocks carrying its index
record; weights, biases, semantic ids (slots) and output membership are unchanged, so
both layers compute the same function (up to FP32 reduction order)."""
from __future__ import annotations

import torch

from sparse_pooling import SparsePoolingLayer
from sparse_pooling.supergroups import SupergroupGraph, matched_c8_from_c64


@torch.no_grad()
def convert_to_matched_c8(lay64: SparsePoolingLayer) -> SparsePoolingLayer:
    L = lay64.layout
    assert L.block == 64 and L.shared_indices
    idx, w, b = L.to_canonical(lay64.indices, lay64.weight.detach(), lay64.bias.detach())
    graph = None
    if lay64.connectivity is not None:
        graph = SupergroupGraph(lay64.connectivity, lay64.sg_feat, lay64.sg_buckets)
    L8c, idx8, w8, b8, g8 = matched_c8_from_c64(L.canonical, idx, w, b, graph)
    del idx, w, b
    lay8 = SparsePoolingLayer(L.input_dim, L.hidden_dim, L.groups, shared_neurons=8, pooling=lay64.cfg,
                              checkpoint_policy=lay64.cfg.checkpoint_policy, dtype=lay64.compute_dtype,
                              plan=lay64.plan, device=lay64.indices.device, backend=lay64.backend_name,
                              weight_layout="packed" if L.packed else "canonical")
    L8 = lay8.layout
    pk = (lambda f, t: f(t)) if L8.packed else (lambda f, t: t)
    lay8.indices.copy_(pk(L8.pack_indices, idx8))
    lay8.weight.data.copy_(pk(L8.pack_weights, w8))
    lay8.bias.data.copy_(pk(L8.pack_bias, b8))
    if g8 is not None:
        lay8.connectivity = lay64.connectivity
        lay8.register_buffer("sg_feat", g8.feat.clone())
        lay8.register_buffer("sg_buckets", g8.buckets.clone())
        lay8.register_buffer("sg_plan", lay64.sg_plan.clone())
    lay8.graph_stats = dict(lay64.graph_stats or {}, matched_from_c64=True)
    lay8._rt = None
    return lay8
