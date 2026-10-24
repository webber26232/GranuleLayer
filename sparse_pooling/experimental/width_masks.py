"""Width-preserving support policies of the K = 64 S = K layout (results/WIDTH_PRESERVING_SPARSITY.md).

Geometry (the KLayout of a KgxLayer, logical C % 16 == 0):

* valid neuron = (output g, neuron n < n_g); slot = (g, block j, c) with n = j C + c;
* row group u = 16 consecutive neuron slots of one logical block (flat u = slot // 16); nv_u <= 16 valid
  neurons (partial tails; wholly padded groups have nv_u = 0 and take part in nothing);
* input tile (u, t): the 16 neurons of u x the b local inputs [t b, t b + b) of the block's 64-input record
  (b = 16: 4 tiles, b = 8: 8 tiles); its exact VALID-slot cost is nv_u b (minus backbone coordinates it
  overlaps);
* logical record = valid block (g, j);  anchor group = one row group of a record keeping all 64 inputs;
* backbone (H2 / H2A): two local coordinates A_u per row group, enabled for every valid neuron of u
  (independent weights); extra tiles omit / mask A_u (no edge is counted or stored twice).

Policies (``POLICIES``): U (global free tiles), P (>= 1 protected tile per valid row group), A (P + one
full-support anchor group per valid record), H2 (backbone, no mandatory tile), H2A (backbone + anchors).
Budgets are exact VALID weight slots: budget = round(rho W0), W0 = 64 H; every mandatory part is charged
first; extra tiles are taken in descending score order (ties: ascending flat tile index) while the running
cost fits, so the actual count is within one tile below the budget (``budget_shortfall`` recorded).
Infeasible requests (skeleton > budget) raise ``Infeasible``: density is never raised and H never lowered.

Scores ("magnitude"): tile score = sum over the tile's valid slots (excluding backbone coordinates) of |w|
in FP32 of the pre-pruning checkpoint (raw sums: a partial group's score covers its valid neurons only, no
per-slot normalization); anchor score = sum of |w| over the whole row group; backbone column score =
sum over the group's valid neurons of |w_ik|.  "random": the same constraints and budget with seeded
uniform scores (the identical-constraint random control).  Nothing here depends on the GPU or its tuning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

SCHEMA = "granule.width_masks/1"


class Infeasible(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    name: str
    tb: int  # input tile width b
    protect: bool  # >= 1 tile per valid row group (P / A)
    anchors: bool
    backbone: int  # protected backbone coordinates per row group (H2: 2)

    @property
    def nkt(self) -> int:
        return 64 // self.tb


POLICIES = {
    "U16": Policy("U16", 16, False, False, 0),
    "P16": Policy("P16", 16, True, False, 0),
    "A16": Policy("A16", 16, True, True, 0),
    "U8": Policy("U8", 8, False, False, 0),
    "P8": Policy("P8", 8, True, False, 0),
    "A8": Policy("A8", 8, True, True, 0),
    "H2": Policy("H2", 16, False, False, 2),
    "H2A": Policy("H2A", 16, False, True, 2),
}


def layout_counts(L, device) -> dict:
    """Exact geometry counts of a KLayout: valid neurons per row group, valid records, anchor neurons."""
    valid = L.valid_mask(device)  # [G, NB, C]
    nv = valid.reshape(-1, 16).sum(1).to(torch.int64)  # [n16]
    per_blk = nv.view(-1, L.block // 16)  # [G NB, C / 16]
    blk_valid = per_blk.sum(1) > 0
    full = per_blk == 16
    has_full = full.any(1)
    # anchor neurons: 16 for a record with a full group, else its (single) partial group's count
    part = torch.where(has_full, torch.full_like(per_blk[:, 0], 16), per_blk.sum(1))
    A = int(torch.where(blk_valid, part, torch.zeros_like(part)).sum())
    return dict(H=int(nv.sum()), W0=64 * int(nv.sum()), groups=int((nv > 0).sum()), groups_total=int(nv.numel()),
                partial_groups=int(((nv > 0) & (nv < 16)).sum()), records=int(blk_valid.sum()),
                records_without_full_group=int((blk_valid & ~has_full).sum()), anchor_neurons=A, nv=nv)


def skeleton(policy: Policy, L, device="cpu", counts: Optional[dict] = None) -> dict:
    """Mandatory valid-slot cost of a policy (protected tiles / backbone / anchors) and its density floor."""
    c = counts or layout_counts(L, device)
    H, A = c["H"], c["A"] if "A" in c else c["anchor_neurons"]
    base = policy.backbone * H if policy.backbone else (policy.tb * H if policy.protect else 0)
    per_non_anchor = policy.backbone if policy.backbone else (policy.tb if policy.protect else 0)
    sk = base + ((64 - per_non_anchor) * A if policy.anchors else 0)
    return dict(policy=policy.name, skeleton_slots=sk, floor_density=sk / c["W0"], W0=c["W0"], H=H,
                anchor_neurons=A if policy.anchors else 0, records=c["records"])


def feasibility(policy: Policy, rho: float, L, device="cpu", counts=None) -> dict:
    sk = skeleton(policy, L, device, counts)
    budget = int(round(rho * sk["W0"]))
    ok = sk["skeleton_slots"] <= budget
    out = dict(sk, rho_target=rho, budget_slots=budget, feasible=ok)
    if not ok:
        out["reason"] = (f"{policy.name} at rho = {rho:g}: mandatory support {sk['skeleton_slots']:,} valid slots "
                         f"(floor {100 * sk['floor_density']:.6f} %) > budget {budget:,}")
    # structural bias-only lower bound of complete b-input tiles (assignment section 3.1)
    if not policy.backbone:
        out["bias_only_lower_bound"] = max(0.0, 1.0 - 64 * rho / policy.tb)
    return out


@dataclass
class MaskSet:
    """A selected support (``build``): tiles [n16, NKT] bool (every selected tile incl. anchors' / protected
    ones), anchor [n16] bool, mandatory [n16, NKT] bool, bb [n16, 2] int64 local backbone coordinates
    (or None), plus the selection record ``stats``."""

    policy: Policy
    tiles: torch.Tensor
    anchor: torch.Tensor
    mandatory: torch.Tensor
    bb: Optional[torch.Tensor]
    nv: torch.Tensor
    stats: dict

    @property
    def tb(self):
        return self.policy.tb

    def tile_overlap(self) -> torch.Tensor:
        """[n16, NKT] number of backbone coordinates inside each tile (0 without backbone)."""
        nkt = self.policy.nkt
        if self.bb is None:
            return torch.zeros(self.tiles.shape, dtype=torch.int64, device=self.tiles.device)
        t = self.bb // self.tb  # [n16, 2]
        return (t[:, :, None] == torch.arange(nkt, device=t.device)[None, None, :]).sum(1)

    def group_degree(self) -> torch.Tensor:
        """[n16] enabled inputs of every valid neuron of the group (identical within a group)."""
        d = self.tiles.sum(1) * self.tb - (self.tile_overlap() * self.tiles).sum(1)
        return d + (2 if self.bb is not None else 0)

    def element_mask(self, L) -> torch.Tensor:
        """[G, NB, C, 64] uint8 of the enabled valid slots (tiles + backbone)."""
        n16 = self.tiles.shape[0]
        e = self.tiles[:, None, :, None].expand(n16, 16, self.policy.nkt, self.tb).reshape(n16, 16, 64)
        if self.bb is not None:
            e = e.clone()
            ar = torch.arange(n16, device=e.device)
            for j in range(2):
                e[ar, :, self.bb[:, j]] = True
        valid = L.valid_mask(e.device).reshape(n16, 16)
        return (e & valid[:, :, None]).to(torch.uint8).view(L.weight_shape)

    def enabled_slots(self) -> int:
        return int((self.group_degree() * self.nv).sum())


@torch.no_grad()
def _abs_chunks(w: torch.Tensor, n16: int, tb: int, chunk: int = 1 << 16):
    """[n16, NKT] tile and [n16, 64] column sums of |w| (FP32), chunked over row groups."""
    nkt = 64 // tb
    wv = w.reshape(n16, 16, 64)
    ts = torch.empty((n16, nkt), dtype=torch.float32, device=w.device)
    cs = torch.empty((n16, 64), dtype=torch.float32, device=w.device)
    for s0 in range(0, n16, chunk):
        a = wv[s0:s0 + chunk].float().abs()
        cs[s0:s0 + chunk] = a.sum(1)
        ts[s0:s0 + chunk] = a.view(-1, 16, nkt, tb).sum((1, 3))
    return ts, cs


def _fill(cost: torch.Tensor, score: torch.Tensor, cand: torch.Tensor, remaining: int) -> torch.Tensor:
    """Greedy prefix in descending score (stable: ties by ascending flat index) of candidates whose running cost
    fits ``remaining``; returns the selected flat indices."""
    idx = torch.nonzero(cand.reshape(-1)).squeeze(1)
    if idx.numel() == 0 or remaining <= 0:
        return idx[:0]
    sc = score.reshape(-1)[idx]
    o = torch.argsort(sc, descending=True, stable=True)
    idx = idx[o]
    cs = torch.cumsum(cost.reshape(-1)[idx], 0)
    k = int(torch.searchsorted(cs, torch.tensor(remaining, device=cs.device), right=True))
    return idx[:k]


@torch.no_grad()
def build(policy, rho: float, w: torch.Tensor, L, score: str = "magnitude", seed: int = 0) -> MaskSet:
    """Select the support of ``policy`` at retained density ``rho`` from pre-pruning weights w [G, NB, C, 64]
    (see the module docstring).  Raises Infeasible."""
    if isinstance(policy, str):
        policy = POLICIES[policy]
    dev = w.device
    cnt = layout_counts(L, dev)
    feas = feasibility(policy, rho, L, dev, cnt)
    if not feas["feasible"]:
        raise Infeasible(feas["reason"])
    nv = cnt["nv"]
    n16, nkt, tb = nv.numel(), policy.nkt, policy.tb
    gv = nv > 0
    ts, cs = _abs_chunks(w, n16, tb)
    if score == "random":
        g = torch.Generator(device=dev).manual_seed(seed)
        ts = torch.rand(ts.shape, generator=g, device=dev)
        cs = torch.rand(cs.shape, generator=g, device=dev)
        gs = torch.rand((n16,), generator=g, device=dev)
    elif score == "magnitude":
        gs = cs.sum(1)
    else:
        raise ValueError(f"score {score!r}")
    bb = None
    overlap = torch.zeros((n16, nkt), dtype=torch.int64, device=dev)
    if policy.backbone:
        top = torch.argsort(cs, dim=1, descending=True, stable=True)[:, :policy.backbone]
        bb = top.sort(1).values
        t = bb // tb
        overlap = (t[:, :, None] == torch.arange(nkt, device=dev)[None, None, :]).sum(1)
        if score == "magnitude":  # extra tiles are scored on the edges they add (backbone columns excluded)
            bbsum = torch.zeros((n16, nkt), dtype=torch.float32, device=dev)
            for j in range(policy.backbone):
                bbsum.scatter_add_(1, t[:, j:j + 1], cs.gather(1, bb[:, j:j + 1]))
            ts = ts - bbsum
    cost = nv[:, None] * (tb - overlap)  # exact valid-slot cost of each tile
    tiles = torch.zeros((n16, nkt), dtype=torch.bool, device=dev)
    mandatory = torch.zeros_like(tiles)
    anchor = torch.zeros(n16, dtype=torch.bool, device=dev)
    if policy.anchors:
        per = nv.view(-1, L.block // 16)
        full = per == 16
        has_full = full.any(1, keepdim=True)
        elig = torch.where(has_full, full, per > 0)
        key = torch.where(elig, gs.view_as(per), torch.full_like(gs.view_as(per), -math.inf))
        pick = torch.argmax(key, dim=1)  # first maximum: lowest group index on ties
        blk_valid = (per > 0).any(1)
        a = torch.zeros_like(per, dtype=torch.bool)
        a[torch.arange(per.shape[0], device=dev), pick] = True
        anchor = (a & blk_valid[:, None]).reshape(-1)
        mandatory |= anchor[:, None]
    if policy.protect:
        prot = torch.argmax(torch.where(gv[:, None], ts, torch.full_like(ts, -math.inf)), dim=1)
        pm = torch.zeros_like(tiles)
        pm[torch.arange(n16, device=dev), prot] = True
        mandatory |= pm & (gv & ~anchor)[:, None]
    tiles |= mandatory
    mandatory_slots = int((cost * mandatory).sum()) + (policy.backbone * cnt["H"] if policy.backbone else 0)
    budget = feas["budget_slots"]
    if mandatory_slots != feas["skeleton_slots"]:
        raise AssertionError(f"skeleton {mandatory_slots} != predicted {feas['skeleton_slots']}")
    cand = gv[:, None] & ~tiles
    sel = _fill(cost, ts, cand, budget - mandatory_slots)
    tiles.view(-1)[sel] = True
    ms = MaskSet(policy, tiles, anchor, mandatory, bb, nv, {})
    actual = ms.enabled_slots()
    deg = ms.group_degree()
    tdeg = tiles.sum(1)
    st = dict(schema=SCHEMA, policy=policy.name, tb=tb, protect=policy.protect, anchors=policy.anchors,
              backbone=policy.backbone, score=score, seed=seed if score == "random" else None, rho_target=rho,
              W0=cnt["W0"], H=cnt["H"], budget_slots=budget, actual_slots=actual, budget_shortfall=budget - actual,
              rho_mask=actual / cnt["W0"], mandatory_slots=mandatory_slots, extra_slots=actual - mandatory_slots,
              extra_tiles=int(sel.numel()), tiles_selected=int(tiles.sum()), groups_valid=int(gv.sum()),
              partial_groups=cnt["partial_groups"], records=cnt["records"],
              anchor_groups=int(anchor.sum()), anchor_neurons=int(nv[anchor].sum()),
              records_with_anchor=int(anchor.view(-1, L.block // 16).any(1).sum()) if policy.anchors else 0,
              skeleton=feas, score_rule=("sum |w| over valid slots of the tile, backbone columns excluded; anchor: "
                                         "sum |w| of the row group; backbone: top-2 column sums |w| of the group"
                                         if score == "magnitude" else "seeded uniform scores, same constraints"),
              tie_break="descending score, stable: ascending flat (group, tile) index; anchors / protected / "
                        "backbone: lowest index among equal maxima",
              tiles_per_group=dict(min=int(tdeg[gv].min()), median=float(tdeg[gv].double().median()),
                                   max=int(tdeg[gv].max()),
                                   hist=torch.bincount(tdeg[gv], minlength=nkt + 1).tolist()),
              structural_bias_only_neurons=int(nv[(deg == 0) & gv].sum()),
              mask_degree_hist=torch.zeros(65, dtype=torch.int64, device=dev).index_add_(
                  0, deg[gv], nv[gv]).tolist())
    if bb is not None:
        st["backbone_slots"] = policy.backbone * cnt["H"]
        st["backbone_tile_overlaps"] = int((overlap * tiles).sum())
    ms.stats = st
    return ms


def describe_bounds(Q: int, b: int, D: int = 1024) -> dict:
    """Union upper bounds of layer-wide tuple coverage when every neuron of the layer keeps exactly one of the
    fixed disjoint b-input tiles of its set (assignment section 3.2): min(1, Q (64 / b) C(b, t) / C(D, t))."""
    return {t: min(1.0, Q * (64 // b) * math.comb(b, t) / math.comb(D, t)) for t in (2, 3, 4)}
