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
  (the dense masked kernels compute dW for every slot; the optimizer mask discards it for inactive slots).
  ``--rule exact`` (default since the random-support study): experimental/support_refresh.refresh -- regrow by
  first fit against remaining = B0 - (entire enabled count after the prune), B0 = round(rho W0) immutable, so
  earlier remainders are recovered and the shortfall stays below one legal tile; ``--rule legacy``: the parent
  study's rule (regrow within the slots freed by THIS refresh only; its remainders accumulated: H2A 0.10 T16
  ended 2,352 slots below B0).  Grown tiles start at zero weight with zero moments (true regrowth: they were
  absent, not merely re-selected among stored values); pruned tiles' values and moments are zeroed; the remap
  is verified at every refresh (surviving values / moments unchanged, grown slots zero, nothing outside the
  mask).

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
def masked_sum(t: torch.Tensor, m: torch.Tensor, chunk: int = 1 << 26) -> float:
    """FP64 sum of t over the slots where m != 0 (chunked: surviving-state checks of the remap)."""
    tv, mv = t.reshape(-1), m.reshape(-1)
    s = 0.0
    for i in range(0, tv.numel(), chunk):
        s += float((tv[i:i + chunk].double() * (mv[i:i + chunk] != 0)).sum())
    return s


@torch.no_grad()
def remap_state(opt, lay, ms, mask_old, rec, valid):
    """Install the refreshed mask (L1.apply_mask_: pruned values / main_grad / moments -> 0, grown slots enabled at
    their zero values with zero moments) and verify the remap: surviving enabled slots keep their values and both
    moments exactly; grown slots hold zero values and moments; nothing outside the new mask is nonzero."""
    from sparse_pooling.experimental import l1 as L1

    mask = ms.element_mask(lay.layout)
    keep = (mask_old != 0) & (mask != 0)
    st = opt.state.get(lay.weight, {})
    before = dict(w=masked_sum(lay.weight.detach(), keep),
                  **{k: masked_sum(st[k], keep) for k in ("m", "v") if k in st})
    L1.apply_mask_(opt, lay.weight, mask)
    after = dict(w=masked_sum(lay.weight.detach(), keep),
                 **{k: masked_sum(st[k], keep) for k in ("m", "v") if k in st})
    grown = (mask != 0) & (mask_old == 0)
    gz = dict(w=masked_sum(lay.weight.detach().abs(), grown),
              **{k: masked_sum(st[k].float().abs(), grown) for k in ("m", "v") if k in st})
    nz = lay.weight.detach() != 0
    rec.update(state_remap=dict(surviving_slots=int(keep.sum()), grown_slots_mask=int(grown.sum()),
                                pruned_slots_mask=int(((mask_old != 0) & (mask == 0)).sum()),
                                surviving_preserved=before == after, grown_zero=all(v == 0 for v in gz.values()),
                                nonzero_outside_mask=int((nz & (mask == 0)).sum())),
               mask_density=int((mask != 0).sum()) / (64 * int(valid.sum())),
               nonzero_density=int((nz & valid[..., None]).sum()) / (64 * int(valid.sum())))
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--period", type=int, default=16, help="T updates between refreshes (0 = frozen control)")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--rule", default="exact", choices=["exact", "legacy"], help="exact: budget-exact refresh "
                    "(experimental/support_refresh.refresh, B0 immutable); legacy: the parent's freed-only rule")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sparse_pooling.experimental import kgx_sparse as SP
    from sparse_pooling.experimental import l1 as L1
    from sparse_pooling.experimental import support_refresh as SR
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
    B0 = int(ms.stats["budget_slots"])  # the immutable target round(rho W0)
    res = dict(policy=a.policy, rho=a.rho, period=a.period, steps=a.steps, frac=a.frac, C=a.C, graph_seed=a.graph_seed,
               rule=a.rule, B0=B0, refresh_schema=SR.SCHEMA,
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
            sw, sg = tile_l1(lay.weight.detach(), ms, valid), tile_l1(g, ms, valid)
            del g
            if a.rule == "exact":
                r = SR.public(SR.refresh(ms, sw, sg, B0, a.frac))
            else:
                r = SR.refresh_freed_only(ms, sw, sg, a.frac)
            del sw, sg
            mask = remap_state(opt, lay, ms, mask, r, valid)  # pruned: values / moments 0; grown: at 0
            torch.cuda.synchronize()
            r.update(step=i + 1, seconds=time.perf_counter() - tr, enabled_slots=ms.enabled_slots(),
                     shortfall_vs_B0=B0 - ms.enabled_slots(),
                     tiles_changed_vs_start=int((ms.tiles != tiles0).sum()),
                     mandatory_intact=bool((ms.tiles | ~ms.mandatory).all()))
            refreshes.append(r)
        if (i + 1) % 32 == 0 or i + 1 == a.steps:
            evals.append(dict(step=i + 1, train=float(loss), heldout=heldout(lay, proj, head, teacher, dev)))
            print(f"[{a.policy} {a.rho:g} T{a.period}] {i + 1}: held-out {evals[-1]['heldout']:.6f}", flush=True)
    res.update(evals=evals, refreshes=refreshes, heldout_final=evals[-1]["heldout"],
               final_enabled_slots=ms.enabled_slots(), budget_slots=B0, final_shortfall=B0 - ms.enabled_slots(),
               max_shortfall=max([r.get("shortfall_vs_B0", 0) for r in refreshes] + [B0 - int(ms.stats["actual_slots"])]),
               tiles_changed_vs_start=int((ms.tiles != tiles0).sum()), wall_seconds=time.perf_counter() - t0,
               mask_preserved=bool(((lay.weight.detach() != 0) <= mask.bool()).all()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
