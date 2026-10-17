"""Explicit memory model for one training step (no allocation needed)."""
from __future__ import annotations

import math
from typing import Dict, Optional

from .config import PoolingConfig, human_bytes, state_bytes
from .layout import SparseLayout


def memory_budget(layout: SparseLayout, cfg: PoolingConfig, batch: int, policy: str,
                  param_bytes: int = 2, moments: str = "bf16", master_weights: bool = False,
                  phys_splits: int = 1, include_optimizer: bool = True) -> Dict[str, int]:
    """Bytes per component.  Returned dict also holds 'total'."""
    p = layout.param_bytes(param_bytes)
    n_w = math.prod(layout.weight_shape)
    n_b = math.prod(layout.bias_shape)
    B, G, D = batch, layout.groups, layout.input_dim
    l1 = cfg.levels[0]
    S = 1
    if cfg.is_semantic_hierarchy:
        S = math.ceil(layout.max_group / l1.neurons_per_shard)
    U = G * S
    out = {
        "weights": p["weights"],
        "bias": p["bias"],
        "indices": p["indices"],
        "input x (bf16)": B * D * 2,
        "input x^T copy (bf16)": B * D * 2,
        "output y (fp32)": B * G * 4,
        "grad main_grad W (fp32)": n_w * 4,
        "grad main_grad b (fp32)": n_b * 4,
        "grad dX (fp32 + cast)": B * D * (4 + 2),
        "pooling workspace (splits)": B * U * phys_splits * state_bytes(l1, track_ids=True) if phys_splits > 1 else 0,
    }
    # activation checkpoint (what ctx keeps)
    ck = 0
    if policy != "recompute_stats":
        ck += B * U * 4  # pooled pre-activation
        if policy == "save_winners" and l1.type == "max":
            ck += B * U * 4
        if policy == "save_winners" and l1.type == "topk_mean":
            ck += B * U * l1.k * 8
        if l1.type == "softmax_weighted":
            ck += B * U * 4
    if cfg.is_semantic_hierarchy and cfg.level1_checkpoint == "recompute":
        ck = 0
    out["activation checkpoint"] = ck
    out["backward coefficients c1,c2,c3 (fp32)"] = 3 * B * U * 4
    if include_optimizer:
        mb = 2 if moments == "bf16" else 4
        out["optimizer moments m,v"] = 2 * (n_w + n_b) * mb
        out["optimizer master weights"] = (n_w + n_b) * 4 if master_weights else 0
    out["hypothetical [B,H] fp32 hidden (never allocated)"] = 0
    out["total"] = sum(v for v in out.values())
    out["(reference) [B,H] fp32 hidden would need"] = B * layout.hidden_dim * 4
    return out


def format_budget(budget: Dict[str, int]) -> str:
    w = max(len(k) for k in budget)
    return "\n".join(f"{k:<{w}}  {human_bytes(v):>12}" for k, v in budget.items())
