"""Optimizer factories of the K64 study for ``bench_supergroups.py --opt-factory k64_opt:<name>`` (args.opt JSON).

* ``l1``  experimental.l1.L1AdamW over the layer's weight / bias: lr = args.opt["lr"] (default 1e-4), BF16 moments,
          stochastic rounding, weight_decay 0; the weight group gets treatment args.opt["l1"] ("none" | "coupled" |
          "prox") with coefficient args.opt["mu"]; a dense masked layer (layer.mask set by k64_factory:sparse)
          gets its frozen mask (inactive weights, gradients and moments stay zero).  Compact sparse layers
          (tile / CSR) need no mask: every stored value is active.  Width-preserving layers (k64_factory:wide):
          the backbone weights are a third group (no L1); the structural zeros of extra tiles overlapping a
          backbone coordinate get the tile value mask (tvmask).  args.opt["stochastic_rounding"] (default
          true) can switch stochastic rounding off for cross-format comparisons.
"""
from __future__ import annotations


def l1(args, lay):
    from sparse_pooling.experimental.l1 import L1AdamW

    o = args.opt
    groups = [{"params": [lay.weight], "l1": o.get("l1", "none"), "mu": float(o.get("mu", 0.0))},
              {"params": [lay.bias], "l1": "none", "mu": 0.0}]
    if getattr(lay, "bb", None) is not None:
        groups.append({"params": [lay.bb], "l1": "none", "mu": 0.0})
    opt = L1AdamW(groups, lr=float(o.get("lr", 1e-4)), moments="bf16", weight_decay=float(o.get("weight_decay", 0.0)),
                  stochastic_rounding=bool(o.get("stochastic_rounding", True)))
    mask = getattr(lay, "mask", None)
    if mask is not None:
        opt.set_mask(lay.weight, mask.contiguous())
    tv = getattr(lay, "tvmask", None)
    if tv is not None:
        opt.set_mask(lay.weight, tv.contiguous())
    return opt
