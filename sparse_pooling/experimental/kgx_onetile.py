"""Secondary execution control of the width-preserving study: a support with EXACTLY ONE b-input tile per
valid row group (P16 at rho = 0.25, P8 at rho = 0.125) is the same function as a K = b, C = 16 kgen layer
whose 16-neuron logical blocks carry the tile's b inputs as their record (results/WIDTH_PRESERVING_SPARSITY.md).

``lower_one_tile(dense, maskset)`` builds that ``KgenLayer`` (bundle "kgen", the existing K16 / K8 kernels)
with identical retained indices, weights, bias and neuron identity:

    output g, neuron n (block j = n // C, c = n % C of the K64 layout, row group u = (g, j, c // 16))
      -> kgen block j16 = n // 16, position n % 16;  record = feat[q(g, j)][t b : t b + b] (t = the group's
         tile; ascending global ids, as the set is sorted);  weights = W[u, :, t b : t b + b];  bias = b[u].

Buckets are the K64 ones scaled to 16-neuron blocks (same supergroups, same outputs); padded blocks copy
the output's first record (as the generator does) and hold zero weights.  It is NOT a resampled K16 / K8
graph: the coverage / identity checks (``canon_rows``: the b weights scattered back into the 64-input
envelope) compare equal to the tile representation.
"""
from __future__ import annotations

import torch

from . import kgen_graph as KG
from .kgen_layer import KgenLayer
from .width_masks import MaskSet


class LoweredOneTile(KgenLayer):
    """KgenLayer (K = b, C = 16) of a one-tile support (module docstring); keeps the envelope map ``env``
    [G NB16, b] (local input positions of every record in its set) for canonical identity."""

    def canon_params(self):
        return {"weight": self.weight, "bias": self.bias}

    def canon_rows(self, src, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * 16) + n
        W = src["weight"].reshape(-1, L.fan_in)[s]
        rows = torch.zeros((len(s), 64), dtype=W.dtype, device=W.device)
        rows.scatter_(1, self.env[s // 16], W)
        return rows, src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * 16) + n
        m = torch.zeros((len(s), 64), dtype=torch.bool, device=s.device)
        m.scatter_(1, self.env[s // 16], True)
        return m

    def kcsr_describe(self):
        d = super().kcsr_describe() if hasattr(super(), "kcsr_describe") else {}
        d.update(lowered_from="K64 one-tile support (experimental/kgx_onetile.py)", envelope_K=64)
        return d


@torch.no_grad()
def lower_one_tile(dense, ms: MaskSet, plan=None) -> LoweredOneTile:
    if ms.bb is not None:
        raise ValueError("a backbone support is not a one-tile support")
    L = dense.layout
    tb, C = ms.tb, L.block
    gv = ms.nv > 0
    tiles = ms.tiles.sum(1)
    if bool((tiles[gv] != 1).any()) or bool((tiles[~gv] != 0).any()):
        raise ValueError("lower_one_tile needs exactly one tile per valid row group (and none elsewhere)")
    dev = dense.weight.device
    G, NB = L.groups, L.blocks_per_group
    gpb = C // 16  # row groups per K64 block
    counts = L.counts_tensor(dev).to(torch.int64)
    nb16 = (int(counts.max()) + 15) // 16
    nvb16 = (counts + 15) // 16
    # K64 block -> supergroup
    b = dense.kg_buckets.to(dev).to(torch.int64)
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((G * NB,), -1, dtype=torch.int64, device=dev)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    g_ = torch.arange(G, device=dev)[:, None].expand(G, nb16)
    j16 = torch.arange(nb16, device=dev)[None, :].expand(G, nb16)
    live = j16 < nvb16[:, None]
    u = g_ * (NB * gpb) + j16  # K64 row group of every 16-block (n = 16 j16 .. 16 j16 + 15)
    u = torch.where(live, u, g_ * (NB * gpb))  # padded blocks: the output's first group (record copy)
    t = torch.argmax(ms.tiles[u].to(torch.int8), dim=2)  # the group's single tile
    q = bq[u // gpb]
    if bool((q[live] < 0).any()):
        raise AssertionError("a valid K64 block is in no bucket")
    env = (t[..., None] * tb + torch.arange(tb, device=dev)).reshape(-1, tb)  # [G nb16, tb] local positions
    feat = dense.sg_feat.to(torch.int64)
    idx = feat[q.reshape(-1)[:, None], env].reshape(G, nb16, tb).to(torch.int16)
    wv = dense.weight.detach().reshape(-1, 16, 64)
    w = torch.gather(wv[u.reshape(-1)], 2, env[:, None, :].expand(-1, 16, tb)).reshape(G, nb16, 16, tb)
    bias = dense.bias.detach().reshape(-1, 16)[u.reshape(-1)].reshape(G, nb16, 16)
    w = w * live[:, :, None, None]
    bias = bias * live[:, :, None]
    valid16 = (torch.arange(nb16 * 16, device=dev)[None, :] < counts[:, None]).view(G, nb16, 16)
    w = w * valid16[..., None]
    bias = bias * valid16
    # buckets in 16-neuron blocks
    first = b[:, 2] * gpb
    end = torch.minimum((b[:, 2] + b[:, 3]) * gpb, nvb16[b[:, 1]])
    bk = torch.stack([b[:, 0], b[:, 1], first, end - first], 1)
    bk = bk[bk[:, 3] > 0].to(torch.int32).contiguous()
    sp = dense.spec
    spec = KG.GraphSpec(tb, 16, sp.S, sp.R, sp.seed, 1, sp.generation, source="lowered-one-tile-of-K64")
    lay = LoweredOneTile(L.input_dim, L.hidden_dim, G, tb, 16, dense.connectivity, device=dev,
                         plan=plan or dense.plan, graph=(idx.contiguous(), dense.sg_feat, bk, spec),
                         weights=(w.to(torch.bfloat16).contiguous(), bias.to(torch.bfloat16).contiguous()))
    lay.register_buffer("env", env.contiguous())
    lay.graph_stats.update(lowered_from_policy=ms.stats.get("policy"), tile_width=tb,
                           records_16=int(live.sum()))
    return lay


@torch.no_grad()
def executed_work_lowered(layer, B: int, split: bool = True) -> dict:
    """Executed work of the kgen K = b, C = 16 control (estimate from its block structure: every live 16-neuron
    block runs b-input products for its valid rows rounded to 8, fwd / dX recompute / dA / dW recompute / dW)."""
    L = layer.layout
    cnt = L.counts_tensor(layer.counts.device).to(torch.int64)
    nb = (cnt + 15) // 16
    j = torch.arange(L.blocks_per_group, device=cnt.device)[None, :]
    nv = torch.clamp(cnt[:, None] - 16 * j, 0, 16)
    rows8 = ((nv + 7) // 8 * 8).sum()
    K = L.fan_in
    fwd = int(rows8) * K
    sp = 2 if split else 1
    return dict(batch=B, split=split, tile_width=K, mma_products=dict(fwd=fwd * B, dx=(fwd + fwd * sp) * B,
                                                                       dw=(16 * K * int(nb.sum()) * (1 + sp)) * B),
                scalar_fma=dict(fwd=0, dx=0, dw=0), fwd_executed_density=fwd / (64 * L.hidden_dim),
                note="estimate from the kgen block structure (rows rounded to 8 per 16-neuron block)")
