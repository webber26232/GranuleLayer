"""EXPERIMENTAL physical lowering: a logical K = 8 graph with C = 128 / 256 as a native C = 64 layer.

Logical block j of output g (C_l neurons sharing one 8-record, independent weights / bias) becomes the
f = C_l / 64 physical blocks f j .. f j + f - 1 of the production C = 64 layout, each carrying a copy of
the record.  In canonical storage this is a reshape: logical weights [G, NB_l, C_l, 8] ARE physical
weights [G, f NB_l, 64, 8] in the same memory order, so neuron identity (slot = j C_l + c = (f j + h) 64
+ c'), output membership, bias, padding, main_grad and optimizer state map one to one.  Physical
buckets (production units: one C = 64 block) are the logical buckets scaled by f, the last one of each
output trimmed to its valid physical blocks; a physical unit therefore never mixes supergroups, and the
f copies of a logical record always share its supergroup.  The production kernels then run unchanged
(dispatch as for any C = 64 layer, recorded in last_info).

This is a representation for reusing the K = 8 kernels, not a native C = 128 / 256 implementation:
the index records are duplicated f times in memory and in the kernels' work, and coverage must be
measured on the logical records (benchmarks/kcsr_graph.native_table(logical_C=...)).
"""
from __future__ import annotations

import json

import torch

from ..layer import SparsePoolingLayer
from ..supergroups import ConnectivityPlan
from . import kgen_graph as KG
from .kgen_layout import KLayout, init_parameters

PHYSICAL_C = 64


@torch.no_grad()
def lower(L: KLayout, idx_l, w_l, b_l, blk_buckets, feat, spec: KG.GraphSpec, plan=None, pooling="mean", device=None):
    """Native packed C = 64 SparsePoolingLayer computing the logical model (K = 8, C = L.block)."""
    if L.fan_in != 8 or L.block % PHYSICAL_C:
        raise ValueError("lowering to C = 64 needs K = 8 and logical C a multiple of 64")
    f = L.block // PHYSICAL_C
    device = torch.device(device or idx_l.device)
    lay = SparsePoolingLayer(L.input_dim, L.hidden_dim, L.groups, fan_in=8, shared_neurons=PHYSICAL_C, pooling=pooling,
                             checkpoint_policy="auto", device=device, plan=plan, seed=0, backend="auto",
                             weight_layout="packed")
    P = lay.layout  # packed physical layout
    G, NBl = L.groups, L.blocks_per_group
    NBp = P.blocks_per_group
    if f * NBl > NBp:
        raise AssertionError(f"physical NB {NBp} < {f} x logical NB {NBl}")
    wp = w_l.reshape(G, f * NBl, PHYSICAL_C, 8)
    bp = b_l.reshape(G, f * NBl, PHYSICAL_C)
    ip = idx_l.reshape(G, NBl, 1, 8).expand(G, NBl, f, 8).reshape(G, f * NBl, 8)
    pad = NBp - f * NBl
    if pad:
        wp = torch.nn.functional.pad(wp, (0, 0, 0, 0, 0, pad))
        bp = torch.nn.functional.pad(bp, (0, 0, 0, pad))
        ip = torch.cat([ip, ip[:, :1].expand(G, pad, 8)], 1)
    Pc = P.canonical
    # padded physical slots must be zero (they are: logical padding is zero and the pad is zero)
    inv = ~Pc.valid_mask(device).view(G, NBp, PHYSICAL_C)
    if bool((wp[inv] != 0).any()) or bool((bp[inv] != 0).any()):
        raise AssertionError("non-zero padded neuron in the lowered parameters")
    lay.indices.copy_(P.pack_indices(ip.contiguous()))
    lay.weight.data.copy_(P.pack_weights(wp.contiguous()))
    lay.bias.data.copy_(P.pack_bias(bp.contiguous()))
    b = blk_buckets.to(torch.int64)
    nvp = (P.counts_tensor(b.device).to(torch.int64) + PHYSICAL_C - 1) // PHYSICAL_C
    first = b[:, 2] * f
    end = torch.minimum((b[:, 2] + b[:, 3]) * f, nvp[b[:, 1]])
    pb = torch.stack([b[:, 0], b[:, 1], first, end - first], 1)
    pb = pb[pb[:, 3] > 0].to(torch.int32).contiguous()
    conn = ConnectivityPlan(spec.S, spec.R, spec.generation, spec.seed)
    lay.connectivity = conn
    lay.register_buffer("sg_feat", feat.to(device).clone())
    lay.register_buffer("sg_buckets", pb.to(device))
    lay.register_buffer("sg_plan", torch.frombuffer(bytearray(conn.to_json().encode()), dtype=torch.uint8).clone())
    lay.register_buffer("kg_spec", torch.frombuffer(bytearray(spec.to_json().encode()), dtype=torch.uint8).clone())
    lay.kcsr_logical_C = L.block
    lay.kcsr_spec = spec
    lay.graph_stats = dict(lowered_from_logical_C=L.block, physical_C=PHYSICAL_C, physical_subblocks=f,
                           index_duplication=f, generator=spec.generator, quantum=spec.quantum)
    lay._rt = None
    return lay


@torch.no_grad()
def unlower(lay) -> tuple:
    """Logical (idx [G, NB_l, 8], weights [G, NB_l, C_l, 8], bias [G, NB_l, C_l], block buckets) of a lowered
    layer (round trip; checks that the physical copies of every logical record are identical)."""
    Cl = lay.kcsr_logical_C
    f = Cl // PHYSICAL_C
    P = lay.layout
    L = KLayout(P.input_dim, P.hidden_dim, P.groups, 8, Cl)
    G, NBl = L.groups, L.blocks_per_group
    idx, w, b = P.to_canonical(lay.indices, lay.weight.detach(), lay.bias.detach())
    idx = idx[:, :f * NBl].reshape(G, NBl, f, 8)
    nvl = L.valid_blocks(idx.device)
    pv = (torch.clamp(P.counts_tensor(idx.device).to(torch.int64)[:, None]
                      - torch.arange(f * NBl, device=idx.device)[None, :] * PHYSICAL_C, min=0) > 0).view(G, NBl, f)
    same = ((idx == idx[:, :, :1]).all(-1) | ~pv).all(-1)
    if not bool(same[torch.arange(NBl, device=idx.device)[None, :] < nvl[:, None]].all()):
        raise AssertionError("physical copies of a logical record differ")
    wl = w[:, :f * NBl].reshape(G, NBl, Cl, 8).contiguous()
    bl = b[:, :f * NBl].reshape(G, NBl, Cl).contiguous()
    pb = lay.sg_buckets.to(torch.int64)
    lb = torch.stack([pb[:, 0], pb[:, 1], pb[:, 2] // f, (pb[:, 3] + f - 1) // f], 1).to(torch.int32)
    return idx[:, :, 0].contiguous(), wl, bl, lb


def describe(lay) -> dict:
    f = lay.kcsr_logical_C // PHYSICAL_C
    return dict(backend="native production kernels via physical lowering", status="extension (lowering)",
                logical_K=8, logical_C=lay.kcsr_logical_C, physical_C=PHYSICAL_C, physical_subblocks=f,
                index_duplication=f, spec=json.loads(lay.kcsr_spec.to_json()),
                note="each logical record stored f times (one per physical C = 64 block); coverage on logical records")


@torch.no_grad()
def build_logical(D: int, H: int, G: int, C: int, spec_plan: ConnectivityPlan, device, seed: int, bias_std: float = 0.0):
    """(KLayout, idx, weights, bias, block buckets, feat, GraphSpec, stats) of a logical K = 8 graph with the
    generalized generator (quantum = 1 logical record) and the kgen initialization."""
    L = KLayout(D, H, G, 8, C)
    counts = L.counts_tensor(device)
    idx, feat, ubk, st = KG.generate(D, counts, C, 8, L.blocks_per_group, 1, spec_plan, device)
    bk = KG.unit_to_block_buckets(ubk, counts, C, 1)
    w, b = init_parameters(L, device, seed=seed, bias_std=bias_std)
    spec = KG.GraphSpec(8, C, spec_plan.input_supergroup_dim, spec_plan.partition_rounds, spec_plan.seed, 1,
                        spec_plan.generation)
    return L, idx, w, b, bk, feat, spec, st
