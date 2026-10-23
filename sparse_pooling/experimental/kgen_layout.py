"""EXPERIMENTAL logical layout for fan-in K in {8, 16, 32, 64} and logical C (multiple of 8, <= 2048).

C <= 256 runs on the kgen kernels (kgen_layer.KgenLayer); 256 < C <= 2048 is the logical layout of the
K = 64 large-C study (kgx_layer.KgxLayer, kgx_lowering: physical C = 256 subblocks).

Partition and neuron identity are those of ``layout.SparseLayout``: output g owns
n_g = H // G + (g < H % G) valid neurons; slot s = j * C + c of output g (logical block j) is valid
iff s < n_g and is the global neuron offset_g + s.  Storage is canonical (no fragment packing):

    indices [G, NB, K]     int16   the shared record of each logical block (sorted, distinct)
    weights [G, NB, C, K]  bf16    K contiguous per neuron
    bias    [G, NB, C]     bf16
    NB = ceil(max_g n_g / C)       (no padding to a multiple of 4: nothing here pairs or packs blocks)

Padded neurons (s >= n_g) have zero weights / bias and are masked in every kernel; fully padded
blocks carry a copy of the output's first record and are never computed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

KGEN_LAYOUT_VERSION = 1
KGEN_LAYOUT_C_MAX = 2048


@dataclass(frozen=True)
class KLayout:
    input_dim: int
    hidden_dim: int
    groups: int
    fan_in: int
    block: int  # logical C

    def __post_init__(self):
        if self.fan_in not in (8, 16, 32, 64):
            raise ValueError(f"experimental kgen layout: fan_in must be 8, 16, 32 or 64 (got {self.fan_in})")
        if self.block % 8 or not 8 <= self.block <= KGEN_LAYOUT_C_MAX:
            raise ValueError(f"experimental kgen layout: C must be a multiple of 8 in [8, {KGEN_LAYOUT_C_MAX}] "
                             f"(got {self.block})")
        if self.input_dim > 32767:
            raise ValueError("int16 indices require input_dim <= 32767")
        if self.fan_in > self.input_dim:
            raise ValueError("fan_in > input_dim")
        if self.hidden_dim < self.groups:
            raise ValueError("need at least one neuron per group")

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
        return math.ceil(self.max_group / self.block)

    @property
    def slots_per_group(self) -> int:
        return self.blocks_per_group * self.block

    def counts_tensor(self, device) -> torch.Tensor:
        g = torch.arange(self.groups, device=device)
        return (self.base + (g < self.rem).to(torch.int64)).to(torch.int32)

    def valid_blocks(self, device="cpu") -> torch.Tensor:
        """[G] logical blocks with at least one valid neuron (= logical records of the output)."""
        return (self.counts_tensor(device).to(torch.int64) + self.block - 1) // self.block

    @property
    def index_shape(self):
        return (self.groups, self.blocks_per_group, self.fan_in)

    @property
    def weight_shape(self):
        return (self.groups, self.blocks_per_group, self.block, self.fan_in)

    @property
    def bias_shape(self):
        return (self.groups, self.blocks_per_group, self.block)

    def valid_mask(self, device) -> torch.Tensor:
        """[G, NB, C] bool: valid neurons."""
        s = torch.arange(self.slots_per_group, device=device)
        return (s[None, :] < self.counts_tensor(device)[:, None]).view(self.bias_shape)

    def param_bytes(self, dtype_bytes: int = 2) -> dict:
        return {"weights": math.prod(self.weight_shape) * dtype_bytes, "bias": math.prod(self.bias_shape) * dtype_bytes,
                "indices": math.prod(self.index_shape) * 2}

    def padding_fraction(self) -> float:
        return 1.0 - self.hidden_dim / (self.groups * self.slots_per_group)


@torch.no_grad()
def init_parameters(layout: KLayout, device, dtype=torch.bfloat16, seed: int = 0, weight_std=None, bias_std: float = 0.0,
                    chunk: int = 1 << 25):
    """(weights, bias): weights iid N(0, weight_std^2) with weight_std = 1 / sqrt(K) (the production rule),
    bias iid N(0, bias_std^2) (bias_std = 0: zeros, the production default), drawn in flat canonical order
    from one generator seeded with ``seed``; padded neurons zeroed."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if weight_std is None:
        weight_std = 1.0 / math.sqrt(layout.fan_in)
    w = torch.empty(layout.weight_shape, dtype=dtype, device=device)
    flat = w.view(-1)
    for s in range(0, flat.numel(), chunk):
        e = min(s + chunk, flat.numel())
        flat[s:e] = (torch.randn(e - s, generator=gen, device=device) * weight_std).to(dtype)
    b = torch.zeros(layout.bias_shape, dtype=dtype, device=device)
    if bias_std > 0:
        b.copy_((torch.randn(b.shape, generator=gen, device=device) * bias_std).to(dtype))
    zero_padding_(layout, w, b)
    return w, b


@torch.no_grad()
def zero_padding_(layout: KLayout, weights=None, bias=None):
    ref = weights if weights is not None else bias
    invalid = ~layout.valid_mask(ref.device)
    if weights is not None:
        weights.masked_fill_(invalid[..., None], 0)
    if bias is not None:
        bias.masked_fill_(invalid, 0)


INIT_BY_NEURON = "granule.init_by_neuron/1"


@torch.no_grad()
def init_parameters_by_neuron(layout: KLayout, device, dtype=torch.bfloat16, seed: int = 0, weight_std=None,
                              chunk_neurons: int = 1 << 19):
    """(weights, bias) with weights iid N(0, weight_std^2) (default 1 / sqrt(K)) drawn per VALID neuron in
    global neuron order (output g, slot s < n_g), independent of C and of padding: neuron i always receives
    the K values i K .. i K + K - 1 of one seeded stream (``INIT_BY_NEURON``), so layouts with different C
    hold identical valid-neuron draws.  Bias zero, padded neurons zero."""
    if weight_std is None:
        weight_std = 1.0 / math.sqrt(layout.fan_in)
    G, K = layout.groups, layout.fan_in
    w = torch.zeros(layout.weight_shape, dtype=dtype, device=device)
    wv = w.view(G, layout.slots_per_group, K)
    counts = layout.counts_tensor("cpu").to(torch.int64)
    off = torch.cumsum(counts, 0) - counts
    gen = torch.Generator(device=device)
    # chunk the neuron stream; each chunk gets its own deterministic sub-seed (seed, chunk index)
    H = int(counts.sum())
    g = 0
    for c0 in range(0, H, chunk_neurons):
        c1 = min(H, c0 + chunk_neurons)
        gen.manual_seed((seed * 1_000_003 + c0 // chunk_neurons) & 0x7FFFFFFFFFFFFFFF)
        vals = (torch.randn((c1 - c0, K), generator=gen, device=device) * weight_std).to(dtype)
        while g < G and int(off[g]) < c1:  # scatter the outputs overlapping [c0, c1)
            a, b = max(c0, int(off[g])), min(c1, int(off[g] + counts[g]))
            if a < b:
                wv[g, a - int(off[g]):b - int(off[g])] = vals[a - c0:b - c0]
            if int(off[g] + counts[g]) <= c1:
                g += 1
            else:
                break
    b = torch.zeros(layout.bias_shape, dtype=dtype, device=device)
    return w, b
