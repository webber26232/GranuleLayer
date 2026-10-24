"""Budget-exact periodic support refresh (prune + regrow) of the width-preserving tile policies
(experimental/width_masks.MaskSet; results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md section 2).

The budget is the IMMUTABLE integer target of the selection, B0 = round(rho W0) valid slots.  One refresh:

1. prune: the ``frac`` x (enabled extra tiles) enabled extra (non-mandatory) tiles of smallest score
   (``prune_score``; ties: ascending flat (group, tile) index); mandatory tiles, anchors and backbone
   coordinates are never candidates;
2. recount: ``enabled_after_prune`` = the ENTIRE enabled valid-slot count of the pruned support
   (``MaskSet.enabled_slots``: mandatory tiles, backbone coordinates and tiles each counted once, a tile's
   backbone overlap charged once), never a running sum of freed slots;
3. regrow against ``remaining = B0 - enabled_after_prune`` -- this recovers every earlier remainder (the
   previous refresh's shortfall is part of ``remaining``) -- absent candidate tiles of a valid row group in
   descending ``grow_score`` (ties: ascending flat index), first fit: a candidate is taken when its exact
   valid-slot cost nv_u (b - overlap) fits what is left, and the scan continues past candidates that do not
   fit (so the final shortfall is below the cost of every eligible candidate left out);
4. invariants: ``enabled <= B0`` always; ``shortfall = B0 - enabled < max_tile_cost`` (16 b: the largest
   legal tile).  When the eligible candidates cannot meet the bound (e.g. ``exclude_pruned`` removes every
   fitting candidate) the prune is REDUCED: pruned tiles are re-enabled, strongest first, while they fit
   (they never lost their values: the caller remaps state only after the refresh); if the bound still fails
   the refresh is recorded ``infeasible`` (the long-term target is never lowered silently).

``refresh_freed_only`` is the LEGACY rule of the parent study (benchmarks/wps_refresh.py at ab649a9): regrow
only within the slots freed by the current prune (prefix greedy).  Its unfilled remainders accumulate; it is
kept for the old-versus-fixed comparison and the regression test only.

Pure torch, no GPU requirement (the stress tests run on the CPU).
"""
from __future__ import annotations

import torch

SCHEMA = "granule.support_refresh/1"


def tile_costs(ms) -> torch.Tensor:
    """[n16, NKT] int64 exact valid-slot cost of every tile (backbone coordinates inside it excluded)."""
    return ms.nv[:, None] * (ms.tb - ms.tile_overlap())


def max_tile_cost(ms) -> int:
    return 16 * ms.tb


def _order(score: torch.Tensor, idx: torch.Tensor, descending: bool) -> torch.Tensor:
    o = torch.argsort(score.reshape(-1)[idx], descending=descending, stable=True)
    return idx[o]


def first_fit(cost_sorted: torch.Tensor, remaining: int) -> torch.Tensor:
    """Bool [n] of the candidates (already in priority order) taken by first fit within ``remaining``: scan in
    order, take every candidate whose cost fits what is left.  Vectorized as repeated prefix sums over the
    candidates that can still fit (terminates: every round either takes a candidate or ends)."""
    n = cost_sorted.numel()
    take = torch.zeros(n, dtype=torch.bool, device=cost_sorted.device)
    left = int(remaining)
    alive = cost_sorted <= left
    while left > 0 and bool(alive.any()):
        pos = torch.nonzero(alive).squeeze(1)
        cs = torch.cumsum(cost_sorted[pos], 0)
        k = int(torch.searchsorted(cs, torch.tensor(left, device=cs.device, dtype=cs.dtype), right=True))
        if k == 0:  # (cannot happen: alive => the first one fits)
            break
        take[pos[:k]] = True
        left -= int(cs[k - 1])
        alive[pos[:k]] = False
        if k < pos.numel():  # pos[k] did not fit: everything before it is decided; later ones may still fit
            alive[pos[:k + 1]] = False
        alive &= cost_sorted <= left
    return take


@torch.no_grad()
def refresh(ms, prune_score: torch.Tensor, grow_score: torch.Tensor, B0: int, frac: float,
            exclude_pruned: bool = True) -> dict:
    """One budget-exact refresh of ``ms.tiles`` in place (module docstring); returns the refresh record."""
    cost = tile_costs(ms)
    flat = ms.tiles.view(-1)
    cmax = max_tile_cost(ms)
    enabled_before = ms.enabled_slots()
    extra = ms.tiles & ~ms.mandatory
    n_extra = int(extra.sum())
    k = int(round(frac * n_extra))
    rec = dict(schema=SCHEMA, rule="budget-exact (B0 immutable, remaining = B0 - enabled_after_prune, first fit)",
               B0=int(B0), enabled_before=enabled_before, carry_in=int(B0) - enabled_before, extra_tiles=n_extra,
               prune_requested=k, max_tile_cost=cmax)
    if enabled_before > B0:
        raise AssertionError(f"enabled {enabled_before} > B0 {B0} before the refresh")
    pr = _order(prune_score, torch.nonzero(extra.reshape(-1)).squeeze(1), descending=False)[:k]
    pruned_slots = int(cost.reshape(-1)[pr].sum())
    flat[pr] = False
    enabled_pruned = ms.enabled_slots()
    if enabled_pruned != enabled_before - pruned_slots:
        raise AssertionError("pruned-slot accounting differs from the recount")
    remaining = int(B0) - enabled_pruned
    cand = ((ms.nv > 0)[:, None] & ~ms.tiles).reshape(-1).clone()
    if exclude_pruned:
        cand[pr] = False
    ci = _order(grow_score, torch.nonzero(cand).squeeze(1), descending=True)
    take = first_fit(cost.reshape(-1)[ci], remaining)
    gr = ci[take]
    flat[gr] = True
    enabled = ms.enabled_slots()
    rolled = 0
    if int(B0) - enabled >= cmax and pr.numel():  # reduce the prune: re-enable pruned tiles, strongest first
        back = pr.flip(0)
        tb = first_fit(cost.reshape(-1)[back], int(B0) - enabled)
        flat[back[tb]] = True
        rolled = int(tb.sum())
        enabled = ms.enabled_slots()
    shortfall = int(B0) - enabled
    rec.update(pruned=int(pr.numel()) - rolled, prune_rolled_back=rolled, pruned_slots=pruned_slots,
               enabled_after_prune=enabled_pruned, remaining=remaining, grown=int(gr.numel()),
               grown_slots=int(cost.reshape(-1)[gr].sum()), enabled_after=enabled, shortfall=shortfall,
               carry_recovered=max(0, enabled - enabled_before) if rolled == 0 else None,
               net_change=enabled - enabled_before, within_budget=enabled <= B0, bound_ok=shortfall < cmax,
               infeasible=not shortfall < cmax, candidates=int(ci.numel()), exclude_pruned=exclude_pruned)
    if enabled > B0:
        raise AssertionError(f"enabled {enabled} > B0 {B0} after the refresh")
    rec["_pruned_idx"], rec["_grown_idx"] = pr, gr  # (callers strip these before writing records)
    return rec


@torch.no_grad()
def refresh_freed_only(ms, prune_score: torch.Tensor, grow_score: torch.Tensor, frac: float) -> dict:
    """LEGACY rule (parent study): regrow only within the slots freed by THIS prune, prefix greedy, just-pruned
    tiles excluded.  Remainders are lost and accumulate across refreshes (module docstring)."""
    cost = tile_costs(ms)
    extra = ms.tiles & ~ms.mandatory
    k = int(round(frac * int(extra.sum())))
    if k == 0:
        return dict(pruned=0, grown=0, freed_slots=0, used_slots=0)
    pr = _order(prune_score, torch.nonzero(extra.reshape(-1)).squeeze(1), descending=False)[:k]
    freed = int(cost.reshape(-1)[pr].sum())
    ms.tiles.view(-1)[pr] = False
    cand = ((ms.nv > 0)[:, None] & ~ms.tiles).reshape(-1)
    cand[pr] = False
    ci = _order(grow_score, torch.nonzero(cand).squeeze(1), descending=True)
    cs = torch.cumsum(cost.reshape(-1)[ci], 0)
    n = int(torch.searchsorted(cs, torch.tensor(freed, device=cs.device), right=True))
    gr = ci[:n]
    ms.tiles.view(-1)[gr] = True
    used = int(cost.reshape(-1)[gr].sum())
    return dict(pruned=int(pr.numel()), grown=int(gr.numel()), freed_slots=freed, used_slots=used,
                budget_drift=used - freed)


def public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}
