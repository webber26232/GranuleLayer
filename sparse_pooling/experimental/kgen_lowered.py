"""K64 envelope and its compact K = k, C = 16 kgen lowering for local supports (the execution of the named
Cover-TopK/16 layer, sparse_pooling.cover_topk16).

Envelope (``Envelope``): the original K64 input-supergroup model with logical C (256 for the validated model): the
kgen_graph generator (quantum 1) gives every logical record (g, j) an input set q and its 64 local inputs
``feat[q]`` (ascending global ids); weights [G, NB, C, 64] / bias [G, NB, C] are the pre-pruning values in KLayout
order.  It is a host-side description (no kernels): the same generator, buckets and identities as the research
branch's ``KgxLayer`` envelope (``kgx_layer.KgxLayer(..., graph=None)``: kgen_graph.generate + unit_to_block_buckets).

Lowering (``lower_local`` / ``lower_env``; research commit f4c538a, experimental/kgx_onetile.py, moved here without the
kgx / tile dependencies): for an arbitrary sorted local k-subset per 16-neuron row group u = (g, j, c // 16)

    output g, neuron n (block j = n // C, c = n % C of the envelope layout)
      -> kgen block j16 = n // 16, position n % 16;  record = feat[q(g, j)][env[u]] (k global ids, ascending);
         weights = W[u, :, env[u]];  bias = b[u]

Buckets are the envelope's scaled to 16-neuron blocks (same supergroups, same outputs); padded blocks copy the
output's first record (as the generator does) and hold zero weights.  Every valid neuron keeps its output, slot,
record, input set and its own bias; the original per-output valid-neuron count stays the pooling denominator.  The
uint8 ``env`` map (local positions) is kept for canonical identity only (never read by a kernel).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import torch

from . import kgen_graph as KG
from .kgen_layer import KgenLayer
from .kgen_layout import KLayout


def _fp_tensor(t: torch.Tensor):
    """(sha256 of the raw bytes, shape, dtype) of a tensor (identity of immutable buffers)."""
    c = t.detach().contiguous()
    h = hashlib.sha256()
    v = c.view(-1).view(torch.uint8) if c.numel() else c.view(-1)
    for i in range(0, v.numel(), 1 << 28):
        h.update(v[i:i + (1 << 28)].cpu().numpy().tobytes())
    return h.hexdigest(), list(c.shape), str(c.dtype)


@dataclass
class Envelope:
    """The original K64 envelope (module docstring).  ``indices`` [G, NB, 64] (int16 global ids of every logical
    record) is optional (references / tests); ``mask`` [G, NB, C, 64] uint8 the enabled support, once selected."""

    layout: KLayout
    weight: torch.Tensor
    bias: torch.Tensor
    sg_feat: torch.Tensor
    kg_buckets: torch.Tensor
    spec: KG.GraphSpec
    connectivity: object
    plan: object
    indices: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    graph_stats: dict = field(default_factory=dict)

    @classmethod
    @torch.no_grad()
    def generate(cls, input_dim: int, hidden_dim: int, output_dim: int, shared_neurons: int, connectivity, weight,
                 bias, plan=None, device=None, keep_indices: bool = False) -> "Envelope":
        """Generate the envelope graph (kgen_graph, quantum 1, K = 64) for the given pre-pruning values."""
        from ..config import PhysicalPlan

        L = KLayout(input_dim, hidden_dim, output_dim, 64, shared_neurons)
        device = torch.device(device or weight.device)
        if tuple(weight.shape) != L.weight_shape or tuple(bias.shape) != L.bias_shape:
            raise ValueError(f"envelope values: weight {tuple(weight.shape)} / bias {tuple(bias.shape)}, expected "
                             f"{L.weight_shape} / {L.bias_shape} (KLayout [G, NB, C, 64] order)")
        if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
            raise ValueError("envelope values must be BF16 (the parameter dtype of the validated recipe)")
        counts = L.counts_tensor(device)
        idx, feat, ubk, st = KG.generate(input_dim, counts, L.block, 64, L.blocks_per_group, 1, connectivity, device)
        bk = KG.unit_to_block_buckets(ubk, counts, L.block, 1)
        spec = KG.GraphSpec(64, L.block, connectivity.input_supergroup_dim, connectivity.partition_rounds,
                            connectivity.seed, 1, connectivity.generation)
        return cls(L, weight.to(device), bias.to(device), feat.contiguous(), bk.contiguous(), spec, connectivity,
                   plan or PhysicalPlan(grad_precision="split"), idx.contiguous() if keep_indices else None,
                   graph_stats=dict(st, generator=KG.GENERATOR, generator_version=KG.GENERATOR_VERSION, quantum=1))

    @property
    def counts(self) -> torch.Tensor:
        return self.layout.counts_tensor(self.weight.device)

    # canonical identity (experimental/canonical.py): dense K64 rows, the mask as support
    def canon_params(self):
        return {"weight": self.weight, "bias": self.bias}

    def canon_rows(self, src, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * L.block) + n
        return src["weight"].reshape(-1, 64)[s], src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * L.block) + n
        if self.mask is None:
            return torch.ones((len(s), 64), dtype=torch.bool, device=s.device)
        return self.mask.reshape(-1, 64)[s] != 0


class LoweredLocal(KgenLayer):
    """KgenLayer (K = k, C = 16) of a local support of a K64 envelope (module docstring); keeps the envelope map
    ``env`` [G NB16, k] (local input positions of every record in its set) for canonical identity."""

    def canon_params(self):
        return {"weight": self.weight, "bias": self.bias}

    def canon_rows(self, src, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * 16) + n
        W = src["weight"].reshape(-1, L.fan_in)[s]
        rows = torch.zeros((len(s), 64), dtype=W.dtype, device=W.device)
        rows.scatter_(1, self.env[s // 16].long(), W)
        return rows, src["bias"].reshape(-1)[s]

    def canon_mask_rows(self, g, n):
        L = self.layout
        s = g * (L.blocks_per_group * 16) + n
        m = torch.zeros((len(s), 64), dtype=torch.bool, device=s.device)
        m.scatter_(1, self.env[s // 16].long(), True)
        return m

    def immutable_fingerprints(self):
        return dict(indices=list(_fp_tensor(self.indices)), env=list(_fp_tensor(self.env)),
                    buckets=list(_fp_tensor(self.kg_buckets)), feat=list(_fp_tensor(self.sg_feat)))


LoweredOneTile = LoweredLocal  # (research-branch name, experimental/kgx_onetile.py)


@torch.no_grad()
def lower_local(dense, sup, plan=None, cls=LoweredLocal) -> LoweredLocal:
    """The indexed K = k, C = 16 kgen execution of a local support (experimental/local_supports.LocalSupport):
    every valid row group's 16 neurons keep their own (g, n), record (g, j), set q and the k sorted local inputs
    ``sup.env[u]`` of that record (one 16-neuron kgen record per row group)."""
    lay = lower_env(dense, sup.env.to(dense.weight.device), plan, source=f"local-support-{sup.policy}-of-K64", cls=cls)
    lay.graph_stats.update(lowered_from_policy=sup.policy, local_support=sup.fingerprint(), tile_width=sup.k)
    return lay


@torch.no_grad()
def lower_env(dense, env_u: torch.Tensor, plan=None, source: str = "lowered-local-of-K64", cls=LoweredLocal):
    """Shared lowering: ``env_u`` [n16, k] sorted distinct local positions (0..63) of every K64 row group u."""
    L = dense.layout
    C = L.block
    tb = env_u.shape[1]
    dev = dense.weight.device
    env_u = env_u.to(dev).to(torch.int64)
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
    q = bq[u // gpb]
    if bool((q[live] < 0).any()):
        raise AssertionError("a valid K64 block is in no bucket")
    env = env_u[u.reshape(-1)]  # [G nb16, tb] local positions
    if bool((env[:, 1:] <= env[:, :-1]).any()):
        raise ValueError("local positions of a record must be sorted and distinct")
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
    spec = KG.GraphSpec(tb, 16, sp.S, sp.R, sp.seed, 1, sp.generation, source=source)
    lay = cls(L.input_dim, L.hidden_dim, G, tb, 16, dense.connectivity, device=dev, plan=plan or dense.plan,
              graph=(idx.contiguous(), dense.sg_feat, bk, spec),
              weights=(w.to(torch.bfloat16).contiguous(), bias.to(torch.bfloat16).contiguous()))
    lay.register_buffer("env", env.to(torch.uint8).contiguous())  # (canonical identity only; kernels read idx)
    lay.graph_stats.update(records_16=int(live.sum()), nominal_envelope=dict(K=64, C=C),
                           executed=dict(K=tb, C=16, sharing="16 neurons per record (one K64 row group)"))
    lay.envelope_KC = (64, C)
    return lay


def spec_json(spec: KG.GraphSpec) -> dict:
    return json.loads(spec.to_json())
