"""Float64 materialising reference of the logical K/C model (small H only; tests)."""
from __future__ import annotations

import torch


def forward(x, idx, w, b, counts, act: str = "relu"):
    """x [B, D]; idx [G, NB, K] (long); w [G, NB, C, K]; b [G, NB, C]; counts [G].  Returns y [B, G] (Mean(act)
    over each output's valid neurons; padded neurons excluded).  Differentiable w.r.t. x, w, b."""
    G, NB, C, K = w.shape
    xg = x[:, idx.long()]  # [B, G, NB, K]
    z = torch.einsum("bgnk,gnck->bgnc", xg, w) + b[None]
    a = torch.relu(z) if act == "relu" else z
    s = torch.arange(NB * C, device=x.device).view(NB, C)
    valid = s[None] < counts.to(x.device).view(G, 1, 1)
    a = torch.where(valid[None], a, torch.zeros_like(a))
    return a.sum((2, 3)) / counts.to(x.dtype)[None]


def grads(x, idx, w, b, counts, gy, act: str = "relu"):
    """(y, dX, dW, db) in float64 for upstream gradient gy [B, G]."""
    x = x.detach().double().requires_grad_(True)
    w = w.detach().double().requires_grad_(True)
    b = b.detach().double().requires_grad_(True)
    y = forward(x, idx, w, b, counts, act)
    dx, dw, db = torch.autograd.grad(y, [x, w, b], gy.double())
    return y.detach(), dx, dw, db
