"""Neuron layout: deterministic partition of H logical neurons into G groups.

Storage (C = neurons per index block, K = fan-in = 8):

    Mode B (block-shared indices): indices [G, NB, K]      int16
    Mode A (independent indices):  indices [G, NB, C, K]   int16
    weights                        [G, NB, C, K]           bf16/fp16  (K contiguous)
    bias                           [G, NB, C]              bf16/fp16

Group g owns n_g = H // G + (g < H % G) valid neurons.  Its slot s = blk * C + c
is valid iff s < n_g; padded slots have zero weights/bias and are masked out of
every forward and backward computation.  Valid slot s of group g is the global
neuron id   offset_g + s,   offset_g = g * (H // G) + min(g, H % G).
So the valid neurons enumerate exactly range(H) with no gaps or repeats.

Indices are stored sorted ascending inside each 8-record (a permutation of the
fan-in that changes nothing mathematically, since each k has its own weight) and
are distinct within a record.  16-byte aligned records allow one 128-bit load.

Fragment-packed storage (``packed=True``, Mode B only)
------------------------------------------------------
Same values, permuted so that the native m16n8k8 kernel needs one 16-byte load
per lane per 4 MMA B-fragments (measured 2x faster at C=8 than the K-contiguous
layout because L2 request count, not bytes, limits the canonical layout).  With
NT = C/8 n8-tiles per block, fragment f = blk*NT + nt, quad q = f//4, fi = f%4,
and mma lane = n*4 + k//2 (n = c%8):

    weights [G, NB*NT/4, 32 lanes, 4 fi, 2]   (one 16 B record per lane)
    bias    [G, NB*NT/4, 4 (n//2), 4 fi, 2]
    indices [G, NB/UB, 8 k, UB]   UB = max(1, 4/NT) blocks per work unit

All three are plain permutations (see pack_* / unpack_*); NB is padded to a
multiple of 4 so quads never straddle groups.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseLayout:
    input_dim: int
    hidden_dim: int
    groups: int
    fan_in: int
    block: int  # C: neurons per stored block
    shared_indices: bool  # True: Mode B, False: Mode A
    packed: bool = False  # fragment-packed storage (Mode B only)

    def __post_init__(self):
        if self.fan_in != 8:
            raise ValueError("kernels are specialised for fan-in 8")
        if self.input_dim > 32767:
            raise ValueError("int16 indices require input_dim <= 32767")
        if self.block not in (8, 16, 32, 64):
            raise ValueError("block (shared_neurons) must be 8, 16, 32 or 64")
        if self.hidden_dim < self.groups:
            raise ValueError("need at least one neuron per group")
        if self.packed and not self.shared_indices:
            raise ValueError("fragment-packed storage is only defined for Mode B")

    # ---- partition -------------------------------------------------------
    @property
    def base(self) -> int:
        return self.hidden_dim // self.groups

    @property
    def rem(self) -> int:
        return self.hidden_dim % self.groups

    @property
    def max_group(self) -> int:
        return self.base + (1 if self.rem else 0)

    @property
    def blocks_per_group(self) -> int:
        # padded to a multiple of 4: Triton pairs blocks, packed storage uses quads
        nb = math.ceil(self.max_group / self.block)
        return (nb + 3) // 4 * 4

    @property
    def n_tiles8(self) -> int:  # NT
        return self.block // 8

    @property
    def unit_blocks(self) -> int:  # UB
        return max(1, 4 // self.n_tiles8)

    @property
    def slots_per_group(self) -> int:
        return self.blocks_per_group * self.block

    def group_count(self, g: int) -> int:
        return self.base + (1 if g < self.rem else 0)

    def group_offset(self, g: int) -> int:
        return g * self.base + min(g, self.rem)

    def counts_tensor(self, device) -> torch.Tensor:
        g = torch.arange(self.groups, device=device)
        return (self.base + (g < self.rem).to(torch.int64)).to(torch.int32)

    def offsets_tensor(self, device) -> torch.Tensor:
        g = torch.arange(self.groups, device=device, dtype=torch.int64)
        return g * self.base + torch.clamp(g, max=self.rem)

    def global_ids(self, device) -> torch.Tensor:
        """[G, slots] int64 global neuron id, -1 for padded slots."""
        s = torch.arange(self.slots_per_group, device=device, dtype=torch.int64)
        ids = self.offsets_tensor(device)[:, None] + s[None, :]
        valid = s[None, :] < self.counts_tensor(device)[:, None].to(torch.int64)
        return torch.where(valid, ids, torch.full_like(ids, -1))

    def valid_mask(self, device) -> torch.Tensor:
        s = torch.arange(self.slots_per_group, device=device)
        return s[None, :] < self.counts_tensor(device)[:, None]

    # ---- shapes ----------------------------------------------------------
    @property
    def index_shape(self):
        G, NB, C, K = self.groups, self.blocks_per_group, self.block, self.fan_in
        if self.packed:
            return (G, NB // self.unit_blocks, K, self.unit_blocks)
        return (G, NB, K) if self.shared_indices else (G, NB, C, K)

    @property
    def weight_shape(self):
        if self.packed:
            return (self.groups, self.blocks_per_group * self.n_tiles8 // 4, 32, 4, 2)
        return (self.groups, self.blocks_per_group, self.block, self.fan_in)

    @property
    def bias_shape(self):
        if self.packed:
            return (self.groups, self.blocks_per_group * self.n_tiles8 // 4, 4, 4, 2)
        return (self.groups, self.blocks_per_group, self.block)

    @property
    def canonical(self) -> "SparseLayout":
        return dataclasses.replace(self, packed=False)

    # ---- fragment packing (pure permutations) -----------------------------
    def pack_weights(self, w):  # [G, NB, C, 8] -> packed
        G, NB, NT = self.groups, self.blocks_per_group, self.n_tiles8
        x = w.reshape(G, NB * NT // 4, 4, 8, 4, 2)  # (quad, fi, n, kh, e)
        return x.permute(0, 1, 3, 4, 2, 5).reshape(self.weight_shape).contiguous()

    def unpack_weights(self, wp):
        G, NB, NT = self.groups, self.blocks_per_group, self.n_tiles8
        x = wp.reshape(G, NB * NT // 4, 8, 4, 4, 2)  # (quad, n, kh, fi, e)
        return x.permute(0, 1, 4, 2, 3, 5).reshape(G, NB, self.block, 8).contiguous()

    def pack_bias(self, b):  # [G, NB, C] -> packed
        G, NB, NT = self.groups, self.blocks_per_group, self.n_tiles8
        x = b.reshape(G, NB * NT // 4, 4, 4, 2)  # (quad, fi, l4, e)
        return x.permute(0, 1, 3, 2, 4).reshape(self.bias_shape).contiguous()

    def unpack_bias(self, bp):
        G, NB, NT = self.groups, self.blocks_per_group, self.n_tiles8
        x = bp.reshape(G, NB * NT // 4, 4, 4, 2)  # (quad, l4, fi, e)
        return x.permute(0, 1, 3, 2, 4).reshape(G, NB, self.block).contiguous()

    def pack_indices(self, idx):  # [G, NB, 8] -> packed
        G, NB, UB = self.groups, self.blocks_per_group, self.unit_blocks
        return idx.reshape(G, NB // UB, UB, 8).permute(0, 1, 3, 2).contiguous()

    def unpack_indices(self, ip):
        G, NB, UB = self.groups, self.blocks_per_group, self.unit_blocks
        return ip.permute(0, 1, 3, 2).reshape(G, NB, 8).contiguous()

    def to_canonical(self, idx, w, b):
        """(indices, weights, bias) in canonical layout (no-op if not packed)."""
        if not self.packed:
            return idx, w, b
        return self.unpack_indices(idx), self.unpack_weights(w), self.unpack_bias(b)

    def param_bytes(self, dtype_bytes: int = 2) -> dict:
        n_w = math.prod(self.weight_shape)
        n_b = math.prod(self.bias_shape)
        n_i = math.prod(self.index_shape)
        return {"weights": n_w * dtype_bytes, "bias": n_b * dtype_bytes, "indices": n_i * 2}

    def padding_fraction(self) -> float:
        return 1.0 - self.hidden_dim / (self.groups * self.slots_per_group)


def _sample_distinct(n_rec: int, d: int, k: int, gen: torch.Generator, device) -> torch.Tensor:
    """n_rec records of k distinct sorted indices in [0, d)."""
    out = torch.randint(0, d, (n_rec, k), generator=gen, device=device, dtype=torch.int64)
    while True:
        out, _ = torch.sort(out, dim=1)
        dup = (out[:, 1:] == out[:, :-1]).any(dim=1)
        nd = int(dup.sum())
        if nd == 0:
            break
        out[dup] = torch.randint(0, d, (nd, k), generator=gen, device=device, dtype=torch.int64)
    return out


@torch.no_grad()
def init_parameters(
    layout: SparseLayout,
    device,
    dtype=torch.bfloat16,
    seed: int = 0,
    weight_std: float | None = None,
    bias_std: float = 0.0,
    chunk_records: int = 1 << 22,
):
    """Create (indices, weights, bias) with padded slots zeroed.

    Generation is chunked so the 100M-neuron configuration never needs large
    fp32/int64 temporaries.
    """
    if layout.packed:
        idx, w, b = init_parameters(layout.canonical, device, dtype, seed, weight_std, bias_std, chunk_records)
        idx = layout.pack_indices(idx)
        w = layout.pack_weights(w)
        b = layout.pack_bias(b)
        return idx, w, b
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    G, NB, C, K = layout.groups, layout.blocks_per_group, layout.block, layout.fan_in
    if weight_std is None:
        weight_std = 1.0 / math.sqrt(K)

    idx = torch.empty(layout.index_shape, dtype=torch.int16, device=device)
    flat_idx = idx.view(-1, K)
    for s in range(0, flat_idx.shape[0], chunk_records):
        e = min(s + chunk_records, flat_idx.shape[0])
        flat_idx[s:e] = _sample_distinct(e - s, layout.input_dim, K, gen, device).to(torch.int16)

    w = torch.empty(layout.weight_shape, dtype=dtype, device=device)
    flat_w = w.view(-1)
    step = chunk_records * K
    for s in range(0, flat_w.numel(), step):
        e = min(s + step, flat_w.numel())
        flat_w[s:e] = (torch.randn(e - s, generator=gen, device=device) * weight_std).to(dtype)

    b = torch.empty(layout.bias_shape, dtype=dtype, device=device)
    if bias_std > 0:
        b.copy_((torch.randn(b.shape, generator=gen, device=device) * bias_std).to(dtype))
    else:
        b.zero_()

    zero_padding_(layout, weights=w, bias=b, indices=idx)
    return idx, w, b


@torch.no_grad()
def zero_padding_(layout: SparseLayout, weights=None, bias=None, indices=None):
    """Zero padded slots of weights [G,NB,C,K], bias [G,NB,C] and Mode-A indices.

    Mode-B indices [G, NB, K] are shared by valid and padded neurons of the
    tail block and are left untouched.
    """
    ref = next(t for t in (weights, bias, indices) if t is not None)
    G, NB, C = layout.groups, layout.blocks_per_group, layout.block
    counts = layout.counts_tensor(ref.device).to(torch.int64)
    slots = torch.arange(NB * C, device=ref.device)
    invalid = (slots[None, :] >= counts[:, None]).view(G, NB, C)
    if layout.packed:
        if weights is not None:
            weights.masked_fill_(layout.pack_weights(invalid[..., None].expand(G, NB, C, 8)), 0)
        if bias is not None:
            bias.masked_fill_(layout.pack_bias(invalid), 0)
        return
    if weights is not None:
        weights.masked_fill_(invalid[..., None], 0)
    if bias is not None:
        bias.masked_fill_(invalid, 0)
    if indices is not None and not layout.shared_indices:
        indices.masked_fill_(invalid[..., None], 0)
