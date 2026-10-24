"""Periodic support refresh (prune + regrow) of constrained width-preserving policies on the synthetic diagnostic
task (assignment section 7; a training diagnostic, NOT an accuracy claim).

Start: the support of (policy, rho) selected from the actual alpha = 0 checkpoint (as for timing / fine-tuning),
the same proj / head state, teacher, batches and fresh optimizer state as wps_learn.py; ``--steps`` updates.
Every ``--period`` T updates (T = 0: frozen support, the control):

* mandatory support is never touched: protected tiles (P), anchor groups (A / H2A) and backbone coordinates
  (H2 / H2A) stay enabled, so no row group loses its last input and no anchor shrinks (no neuron ablation:
  the SRigL-style constant-minimum-support idea WITHOUT its ablation step);
* prune: the ``--frac`` x (number of extra tiles) enabled extra tiles of smallest L1 weight magnitude
  (valid slots, backbone coordinates excluded);
* grow: absent candidate tiles of largest L1 magnitude of the DENSE gradient dL/dW at their (zero) values
  (the dense masked kernels compute dW for every slot; the optimizer mask discards it for inactive slots), in
  descending order while the freed valid-slot budget fits (budget preserved within one tile; the actual
  count is recorded at every refresh); grown tiles start at zero weight with zero moments (true regrowth:
  they were absent, not merely re-selected among stored values); pruned tiles' values and moments are zeroed.

Time cost (labelled ARITHMETIC, not measured training): in a compact format a refresh needs one dense scoring
step (the absent edges' gradients) plus the maintenance U (selection, compression, work lists, state remap;
wps_refresh_cost.py); per-step cost of period T: ((T - 1) s_compact + s_dense + U) / T (see the report).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import common  # noqa: F401
import wps_cases as WC
from k64_learn import LR_LAYER, LR_SMALL, REG, WARMUP, Teacher, batch, build, forward_loss, heldout


@torch.no_grad()
def tile_l1(t: torch.Tensor, ms, valid) -> torch.Tensor:
    """[n16, NKT] L1 of a dense [G, NB, C, 64] tensor per tile over valid slots, backbone coordinates excluded."""
    n16, nkt, tb = ms.nv.numel(), ms.policy.nkt, ms.tb
    out = torch.empty((n16, nkt), dtype=torch.float32, device=t.device)
    vv = valid.reshape(n16, 16, 1)
    for u0 in range(0, n16, 1 << 16):
        a = t.reshape(n16, 16, 64)[u0:u0 + (1 << 16)].float().abs() * vv[u0:u0 + (1 << 16)]
        if ms.bb is not None:
            ar = torch.arange(a.shape[0], device=t.device)
            for j in range(2):
                a[ar, :, ms.bb[u0:u0 + (1 << 16), j]] = 0
        out[u0:u0 + (1 << 16)] = a.view(-1, 16, nkt, tb).sum((1, 3))
    return out


@torch.no_grad()
def refresh(ms, w, g, valid, frac):
    """Prune / regrow extra tiles in place of ms.tiles (mandatory fixed); returns the refresh record."""
    nkt, tb = ms.policy.nkt, ms.tb
    overlap = ms.tile_overlap()
    cost = ms.nv[:, None] * (tb - overlap)
    extra = ms.tiles & ~ms.mandatory
    n_extra = int(extra.sum())
    k = int(round(frac * n_extra))
    if k == 0:
        return dict(pruned=0, grown=0, freed_slots=0, used_slots=0)
    sw = tile_l1(w, ms, valid)
    idx = torch.nonzero(extra.reshape(-1)).squeeze(1)
    o = torch.argsort(sw.reshape(-1)[idx], stable=True)[:k]  # weakest enabled extra tiles
    pr = idx[o]
    freed = int(cost.reshape(-1)[pr].sum())
    ms.tiles.view(-1)[pr] = False
    sg = tile_l1(g, ms, valid)
    cand = ((ms.nv > 0)[:, None] & ~ms.tiles).reshape(-1)
    cand[pr] = False  # (a just-pruned tile is not regrown in the same refresh)
    ci = torch.nonzero(cand).squeeze(1)
    oc = torch.argsort(sg.reshape(-1)[ci], descending=True, stable=True)
    ci = ci[oc]
    cs = torch.cumsum(cost.reshape(-1)[ci], 0)
    n = int(torch.searchsorted(cs, torch.tensor(freed, device=cs.device), right=True))
    gr = ci[:n]
    ms.tiles.view(-1)[gr] = True
    used = int(cost.reshape(-1)[gr].sum())
    return dict(pruned=int(pr.numel()), grown=int(gr.numel()), freed_slots=freed, used_slots=used,
                budget_drift=used - freed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--period", type=int, default=16, help="T updates between refreshes (0 = frozen control)")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import width_masks as WM
    from provenance import provenance

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    ck = WC.ckpt(a.C, a.graph_seed)
    lay, proj, head = build(a.C, 256, dev, a.graph_seed)
    rec, w, b, _ = SP.load_checkpoint(ck)
    with torch.no_grad():
        lay.weight.copy_(w)
        lay.bias.copy_(b)
    del w, b
    small = torch.load(ck.replace(".pt", "_small.pt"), map_location=dev)
    proj.load_state_dict(small["proj"])
    head.load_state_dict(small["head"])
    teacher = Teacher(dev)
    valid = SP.valid_slots(lay)
    ms = WM.build(a.policy, a.rho, lay.weight.detach(), lay.layout)
    tiles0 = ms.tiles.clone()
    mask = ms.element_mask(lay.layout)
    opt = L1.L1AdamW([lay.weight, lay.bias], lr=LR_LAYER)
    L1.apply_mask_(opt, lay.weight, mask)
    opt2 = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()), lr=LR_SMALL, weight_decay=0.0)
    res = dict(policy=a.policy, rho=a.rho, period=a.period, steps=a.steps, frac=a.frac, C=a.C, graph_seed=a.graph_seed,
               checkpoint=dict(path=ck, weight_sha256=rec["weight_sha256"]), mask_stats0=ms.stats,
               heldout_pruned=heldout(lay, proj, head, teacher, dev),
               provenance=provenance(Path(common.__file__).resolve().parent.parent))
    evals, refreshes = [], []
    step0 = WARMUP + REG
    for i in range(a.steps):
        x0 = batch(10_000 + step0 + i, dev)
        loss = forward_loss(lay, proj, head, x0, teacher(x0))
        loss.backward()
        opt2.step()
        opt2.zero_grad(set_to_none=True)
        do_refresh = a.period and (i + 1) % a.period == 0 and i + 1 < a.steps
        g = lay.weight.main_grad.clone() if do_refresh else None  # dense dW (all slots) before the fused zeroing
        opt.step()
        if do_refresh:
            tr = time.perf_counter()
            r = refresh(ms, lay.weight.detach(), g, valid, a.frac)
            del g
            mask = ms.element_mask(lay.layout)
            L1.apply_mask_(opt, lay.weight, mask)  # pruned: values / moments 0; grown: enabled at 0 (moments 0)
            torch.cuda.synchronize()
            r.update(step=i + 1, seconds=time.perf_counter() - tr, enabled_slots=ms.enabled_slots(),
                     tiles_changed_vs_start=int((ms.tiles != tiles0).sum()),
                     mandatory_intact=bool((ms.tiles | ~ms.mandatory).all()))
            refreshes.append(r)
        if (i + 1) % 32 == 0 or i + 1 == a.steps:
            evals.append(dict(step=i + 1, train=float(loss), heldout=heldout(lay, proj, head, teacher, dev)))
            print(f"[{a.policy} {a.rho:g} T{a.period}] {i + 1}: held-out {evals[-1]['heldout']:.6f}", flush=True)
    res.update(evals=evals, refreshes=refreshes, heldout_final=evals[-1]["heldout"],
               final_enabled_slots=ms.enabled_slots(), budget_slots=ms.stats["budget_slots"],
               tiles_changed_vs_start=int((ms.tiles != tiles0).sum()), wall_seconds=time.perf_counter() - t0,
               mask_preserved=bool(((lay.weight.detach() != 0) <= mask.bool()).all()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
