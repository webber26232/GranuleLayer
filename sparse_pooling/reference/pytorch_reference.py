"""Straightforward PyTorch reference (materialises the hidden activations).

Only for small hidden widths.  Everything is computed in FP32 from the actual
(quantised) BF16/FP16 operands, and gradients come from ordinary autograd.
Tie rules match the kernels: Max / Top-K prefer the smallest neuron id.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..config import LevelSpec, PoolingConfig
from ..layout import SparseLayout


def activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "relu":
        return torch.relu(x)
    if name == "gelu":
        return F.gelu(x)  # exact (erf) GELU, as in the kernels
    raise ValueError(name)


def hidden_preactivations(x, idx, w, b, layout: SparseLayout) -> torch.Tensor:
    """z [B, G, slots] in FP32 (padded slots hold garbage-free finite values).

    Accepts fragment-packed parameters: the unpacking permutation is
    differentiable, so gradients come back in the packed layout."""
    idx, w, b = layout.to_canonical(idx, w, b)
    layout = layout.canonical
    ct = torch.float64 if x.dtype == torch.float64 else torch.float32  # FP64 for finite differences
    xf = x.to(ct)
    G, NB, C, K = layout.groups, layout.blocks_per_group, layout.block, layout.fan_in
    il = idx.long()
    if layout.shared_indices:
        xg = xf[:, il]  # [B, G, NB, K]
        z = torch.einsum("bgnk,gnck->bgnc", xg, w.to(ct))
    else:
        xg = xf[:, il]  # [B, G, NB, C, K]
        z = (xg * w.to(ct)[None]).sum(-1)
    z = z + b.to(ct)[None]
    return z.reshape(x.shape[0], G, NB * C)


def pool(z: torch.Tensor, mask: torch.Tensor, level: LevelSpec, extra: Optional[Dict] = None):
    """Pool the last dim of z [B, U, N] under validity mask [U, N] (or [B,U,N]).

    Returns pooled values [B, U] (after out_activation).  ``extra`` receives
    diagnostics (winner indices, top-k indices, log-normaliser).
    """
    if mask.dim() == 2:
        mask = mask[None].expand_as(z)
    count = mask.sum(-1).to(z.dtype)  # [B, U]
    a = activation(z, level.activation)
    neg_inf = torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
    t = level.type
    if t == "max":
        am = torch.where(mask, a, neg_inf)
        win = torch.argmax(am, dim=-1, keepdim=True)  # first max -> smallest id
        pooled = am.gather(-1, win).squeeze(-1)
        if extra is not None:
            extra["winner"] = win.squeeze(-1)
    elif t == "topk_mean":
        zm = torch.where(mask, z, neg_inf)
        k = level.k
        order = torch.sort(zm.detach(), dim=-1, descending=True, stable=True).indices[..., :k]
        sel_valid = torch.arange(k, device=z.device)[None, None, :] < count[..., None]
        vals = a.gather(-1, order)
        kk = torch.clamp(count, max=k)
        pooled = torch.where(sel_valid, vals, torch.zeros_like(vals)).sum(-1) / kk
        if extra is not None:
            extra["topk"] = order
    elif t == "mean":
        pooled = torch.where(mask, a, torch.zeros_like(a)).sum(-1) / count
    elif t == "logmeanexp":
        tau = level.temperature
        u = torch.where(mask, a / tau, neg_inf)
        lse = torch.logsumexp(u, dim=-1)
        pooled = tau * (lse - torch.log(count))
        if extra is not None:
            extra["lse"] = lse
    elif t == "power_mean":
        p = level.p
        am = torch.where(mask, a, torch.zeros_like(a))  # a = relu(z) >= 0
        m = am.amax(-1, keepdim=True)
        safe = torch.where(m > 0, m, torch.ones_like(m))
        s = ((am / safe) ** p).sum(-1) / count
        pooled = torch.where(m.squeeze(-1) > 0, safe.squeeze(-1) * s ** (1.0 / p), torch.zeros_like(s))
    elif t == "softmax_weighted":
        beta = level.beta
        u = torch.where(mask, beta * a, neg_inf)
        pw = torch.softmax(u, dim=-1)
        pooled = (pw * torch.where(mask, a, torch.zeros_like(a))).sum(-1)
        if extra is not None:
            extra["lse"] = torch.logsumexp(u, dim=-1)
    elif t == "weighted_sum":
        raise ValueError("weighted_sum needs weights; use level2_forward")
    else:
        raise ValueError(t)
    return activation(pooled, level.out_activation)


# ---------------------------------------------------------------------------
# semantic level-2 (shared by the reference and the production layer: it only
# ever touches the compact [B, G, S] feature tensor)
# ---------------------------------------------------------------------------
def apply_transform(f: torch.Tensor, level1: LevelSpec, tparams: Dict[str, torch.Tensor]) -> torch.Tensor:
    tr = level1.transform
    if tr == "none":
        return f
    if tr == "activation":
        return activation(f, level1.transform_activation)
    if tr == "affine":
        return activation(tparams["alpha"] * f + tparams["beta"], level1.transform_activation)
    if tr == "gate":
        return f * torch.sigmoid(tparams["alpha"] * f + tparams["beta"])
    raise ValueError(tr)


def level2_forward(f: torch.Tensor, cfg: PoolingConfig, tparams: Dict[str, torch.Tensor]) -> torch.Tensor:
    """f [B, G, S] level-1 features -> y [B, G]."""
    l1, l2 = cfg.levels
    h = apply_transform(f, l1, tparams)
    if l2.type == "weighted_sum":
        return activation((h * tparams["v"]).sum(-1), l2.out_activation)
    mask = torch.ones(h.shape[1:], dtype=torch.bool, device=h.device)
    return pool(h, mask, l2)


def shard_view(z: torch.Tensor, layout: SparseLayout, nps: int):
    """[B, G, slots] -> [B, G*S, nps] plus validity mask [G*S, nps]."""
    B, G, slots = z.shape
    S = math.ceil(layout.max_group / nps)
    pad = S * nps - slots
    if pad > 0:
        z = F.pad(z, (0, pad))
    elif pad < 0:
        z = z[..., : S * nps]
    s = torch.arange(S * nps, device=z.device)
    mask = s[None, :] < layout.counts_tensor(z.device)[:, None]
    return z.reshape(B, G * S, nps), mask.reshape(G * S, nps), S


def reference_forward(x, idx, w, b, layout: SparseLayout, cfg: PoolingConfig, tparams=None, extra=None):
    """Full reference forward. Returns y [B, G] (FP32)."""
    z = hidden_preactivations(x, idx, w, b, layout)
    l1 = cfg.levels[0]
    if not cfg.is_semantic_hierarchy:
        return pool(z, layout.valid_mask(z.device), l1, extra)
    zs, mask, S = shard_view(z, layout, l1.neurons_per_shard)
    f = pool(zs, mask, l1, extra).reshape(x.shape[0], layout.groups, S)
    if extra is not None:
        extra["features"] = f
    return level2_forward(f, cfg, tparams or {})


def reference_physical_hierarchy(z: torch.Tensor, mask: torch.Tensor, level: LevelSpec, splits: int):
    """Pool by explicit two-stage state merge (Sum/count, LSE, Max, Top-K),
    for testing that the physical hierarchy equals the flat operator."""
    B, U, N = z.shape
    assert N % splits == 0
    zs = z.reshape(B, U, splits, N // splits)
    ms = mask.reshape(U, splits, N // splits)
    a = activation(zs, level.activation)
    neg_inf = float("-inf")
    t = level.type
    if t == "max":
        part = torch.where(ms[None], a, torch.full_like(a, neg_inf)).amax(-1)
        return activation(part.amax(-1), level.out_activation)
    if t == "mean":
        part = torch.where(ms[None], a, torch.zeros_like(a)).sum(-1)
        cnt = ms.sum(-1).to(z.dtype)
        return activation(part.sum(-1) / cnt.sum(-1), level.out_activation)
    if t == "logmeanexp":
        u = torch.where(ms[None], a / level.temperature, torch.full_like(a, neg_inf))
        mm = u.amax(-1)
        ss = torch.exp(u - mm[..., None]).sum(-1)
        M = mm.amax(-1)
        Sx = (ss * torch.exp(mm - M[..., None])).sum(-1)
        cnt = ms.sum((-1, -2)).to(z.dtype)
        return activation(level.temperature * (M + torch.log(Sx) - torch.log(cnt)), level.out_activation)
    if t == "topk_mean":
        k = level.k
        zm = torch.where(ms[None], zs, torch.full_like(zs, neg_inf))
        tv, ti = torch.topk(zm, k, dim=-1)  # per-split top-k
        allv = tv.reshape(B, U, -1)
        top = torch.topk(allv, k, dim=-1).values
        return activation(activation(top, level.activation).mean(-1), level.out_activation)
    raise ValueError(t)
