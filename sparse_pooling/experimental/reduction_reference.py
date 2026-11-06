"""Materialising PyTorch oracle of the reducers (tests only; deliberately separate from the optimized code).

    z[m, i] = sum_k W[i, k] x[m, idx[i, k]] + b[i]        h = ReLU(z)         (global neuron order)
    u[m, j] = sum_r P[j, r] h[m, start(j) + r] + c[j]     a = psi(u)
    y[m, o] = sum_j a[m, j] V[j, o] + d[o]

One explicit Python-visible window at a time, ordinary autograd, no packing and no stride blocks, so it shares no
routing logic with experimental/reduction_layout.Routing.  A size guard rejects any shape large enough to be an
accident: this module allocates M x H and M x J x L tensors on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

MAX_ELEMENTS = 200_000_000


def _guard(*shape):
    n = 1
    for s in shape:
        n *= int(s)
    if n > MAX_ELEMENTS:
        raise ValueError(f"reduction_reference materialises {tuple(shape)} ({n:,} elements > {MAX_ELEMENTS:,}): "
                         "reference shapes only")


def hidden(x, idx, W, b, counts):
    """[M, H] ReLU activations in global neuron order (valid neurons only: padding is not in the stream)."""
    G, NB, C, K = W.shape
    M = x.shape[0]
    H = int(counts.sum())
    _guard(M, H)
    xg = x[:, idx.long()]
    z = torch.einsum("mgnk,gnck->mgnc", xg, W) + b[None]
    h = torch.relu(z).reshape(M, G, NB * C)
    return torch.cat([h[:, g, :int(counts[g])] for g in range(G)], 1)


def window_outputs(h, P_logical, starts):
    """[M, J]: each output's explicit window dot product."""
    M, H = h.shape
    J, L = P_logical.shape
    _guard(M, J, L)
    return torch.stack([(h[:, int(starts[j]):int(starts[j]) + L] * P_logical[j]).sum(1) for j in range(J)], 1)


def stride_block_outputs(h, P_logical, s, ch=1):
    """The same sum evaluated block by block: U[:, relevant windows] += H_q @ P_q (the reordering of the kernels),
    for L = m s, channels = 1 (or L = s with channels)."""
    M, H = h.shape
    J, L = P_logical.shape
    if L % s:
        raise ValueError("stride-block evaluation needs L % s == 0")
    m = L // s
    U = torch.zeros((M, J), dtype=h.dtype, device=h.device)
    for q in range(H // s + (1 if H % s else 0)):
        blk = h[:, q * s:(q + 1) * s]
        for w in range(max(0, q - m + 1), min(q, J // ch - 1) + 1):
            r0 = (q - w) * s
            for c in range(ch):
                j = w * ch + c
                U[:, j] = U[:, j] + blk @ P_logical[j, r0:r0 + blk.shape[1]]
    return U


@dataclass
class ReferenceModel:
    idx: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    psi: Callable = torch.nn.Identity()
    gamma: float = 1.0

    def forward(self, x, W, b, P, c, V=None, d=None, second: Optional[tuple] = None):
        h = hidden(x, self.idx, W, b, self.counts)
        U = window_outputs(h, P, self.starts) + c[None]
        A = self.psi(U) * self.gamma
        if second is not None:  # (P2 [J2, L2], c2, s2, psi2)
            P2, c2, s2, psi2 = second
            A = psi2(window_outputs(A, P2, torch.arange(P2.shape[0]) * s2) + c2[None])
        if V is None:
            return A
        return A @ V + d[None]


def group_mean(x, idx, W, b, counts):
    """A0: the original per-output Mean(ReLU)."""
    h = hidden(x, idx, W, b, counts)
    out, o = [], 0
    for n in counts.tolist():
        out.append(h[:, o:o + n].mean(1))
        o += n
    return torch.stack(out, 1)
