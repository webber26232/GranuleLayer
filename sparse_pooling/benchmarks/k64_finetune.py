"""Fixed-mask fine-tuning of pruned checkpoints on the synthetic diagnostic task of k64_learn.py (M_train = 512;
an implementation / training diagnostic, NOT a downstream-accuracy claim).

Starts from the alpha = 0 checkpoint (layer values after 128 updates, proj / head state saved with it), applies a
mask (magnitude / tile / random / random_tile at a density, rebuilt exactly as for timing), and fine-tunes for
32 updates with the frozen mask (dense masked layer, L1AdamW mask; FRESH optimizer state for every parameter,
which differs from the uninterrupted runs of k64_learn.py).  Records held-out loss before pruning, right after
pruning and after fine-tuning, plus the dense (unpruned) control continued for the same 32 updates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import common  # noqa: F401
import k64_cases as KC
from k64_learn import FINETUNE, LR_LAYER, LR_SMALL, WARMUP, REG, Teacher, batch, build, forward_loss, heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--mask", required=True, choices=["none", "magnitude", "tile", "random", "random_tile"])
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from provenance import provenance

    dev = torch.device("cuda")
    lay, proj, head = build(a.C, 256, dev)
    rec, w, b, _ = SP.load_checkpoint(a.checkpoint)
    with torch.no_grad():
        lay.weight.copy_(w)
        lay.bias.copy_(b)
    del w, b
    small = torch.load(a.checkpoint.replace(".pt", "_small.pt"), map_location=dev)
    proj.load_state_dict(small["proj"])
    head.load_state_dict(small["head"])
    teacher = Teacher(dev)
    valid = SP.valid_slots(lay)
    h0 = heldout(lay, proj, head, teacher, dev)
    mask = None
    if a.mask == "magnitude":
        mask = SP.magnitude_mask(lay.weight.detach(), valid, a.density)
    elif a.mask == "random":
        mask = SP.random_mask(valid, a.density, 0)
    elif a.mask == "tile":
        mask = SP.tiles_to_elements(SP.tile_mask(lay.weight.detach(), valid, a.density), valid)
    elif a.mask == "random_tile":
        mask = SP.tiles_to_elements(SP.random_tile_mask(valid, a.density, 0), valid)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=LR_LAYER)
    if mask is not None:
        L1.apply_mask_(opt, lay.weight, mask)
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)
    h1 = heldout(lay, proj, head, teacher, dev)
    curve = []
    step0 = WARMUP + REG
    for i in range(FINETUNE):
        x0 = batch(10_000 + step0 + i, dev)
        loss = forward_loss(lay, proj, head, x0, teacher(x0))
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        opt.step()
        curve.append(float(loss))
    h2 = heldout(lay, proj, head, teacher, dev)
    st = L1.weight_stats(lay.weight.detach(), valid)
    res = dict(checkpoint=a.checkpoint, checkpoint_rec=rec, C=a.C, mask=a.mask, density_target=a.density,
               density=st["density_nonzero"], mask_sha256=SP.mask_fingerprint(mask) if mask is not None else None,
               heldout_checkpoint=h0, heldout_pruned=h1, heldout_finetuned=h2, train_curve=curve,
               all_zero_neurons=st["all_zero_neurons"], fresh_optimizer_state=True, steps=FINETUNE, lr_layer=LR_LAYER,
               lr_small=LR_SMALL, provenance=provenance(Path(common.__file__).resolve().parent.parent))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"C{a.C} {a.mask} {a.density:g}: held-out {h0:.6f} -> pruned {h1:.6f} -> fine-tuned {h2:.6f} "
          f"(density {st['density_nonzero']:.4f})", flush=True)


if __name__ == "__main__":
    main()
