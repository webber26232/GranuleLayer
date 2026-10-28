"""Mixed fan-in local supports at a 25 % budget inside the C256 K64 envelope (weighted-support study,
results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md section 5; a BUDGET-ALLOCATION experiment, not an isolated effect of
anchors).

Every valid original record (g, j) of n valid neurons (16-neuron row groups in canonical order, full groups first,
at most one partial tail group) is partitioned into fan-in classes:

* K64 anchor: its FIRST row group (rank 0, full) keeps all 64 envelope inputs;
* K8: six of its other FULL row groups, chosen by a keyed uniform order independent of the weights
  (``class_order``: per record a keyed permutation of ranks 1..gpb-1; the first six ranks that are full groups);
* K16: every remaining valid group, the partial tail group included.

Slots: 16 * 64 + 96 * 8 + (n - 112) * 16 = 16 n, exactly the uniform-K16 budget of the record; a record with
n < 112 (fewer than seven full groups) cannot hold the mandatory classes and is REJECTED (``Unsupported``) -- never
a silent change of density, padded support or dropped neuron.  For the primary model (H 12.5M, G 1024, C256) the
totals are H64 786,432, H16 6,994,976, H8 4,718,592, 200,000,000 slots and 12,500,000 biases.

Non-anchor subsets (all original q / neuron identities kept):

* ``Mix-BP``: an unweighted BP stream separately within each (q, k) stratum (k in 16 / 8): the class-k groups of set
  q in canonical order consume independent keyed permutations of 64 cut into 64 / k disjoint subsets
  (local_supports.bp_stream, key ("Mix-BP", k, graph, layout));
* ``Mix-WRS:L`` / ``Mix-TopK``: the same class map; candidate 0 is that Mix-BP subset and the other candidates /
  the TopK rule are local_supports.wrs_select / topk_select on the pre-pruning row-group importances (key
  ("Mix-WRS", k, graph, layout, L-stream, cand<c>)).

Both mixed variants share the anchor / class map (its key has no weighting label).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from . import local_supports as LS

SCHEMA = "granule.mixed_supports/1"
POLICIES = ("Mix-BP", "Mix-WRS", "Mix-TopK")
N_K8, MIN_NEURONS = 6, 112
CLASSES = (64, 16, 8)


class Unsupported(ValueError):
    """A geometry whose records cannot hold the mandatory anchor + six K8 groups."""


@dataclass
class MixedSupport:
    """``kc`` [n16] fan-in class (64 / 16 / 8; 0 padded), ``gm`` [n16, 64] bool local support of every row group,
    ``nv`` / ``q`` as LocalSupport, ``anchor`` [n16] bool, ``cand`` [n16] chosen candidate (-1: anchor / padded)."""

    policy: str
    kc: torch.Tensor
    gm: torch.Tensor
    nv: torch.Tensor
    q: torch.Tensor
    anchor: torch.Tensor
    cand: torch.Tensor
    stats: dict = field(default_factory=dict)
    k = None  # mixed

    def group_mask(self) -> torch.Tensor:
        return self.gm & (self.nv > 0)[:, None]

    def element_mask(self, L) -> torch.Tensor:
        n16 = self.gm.shape[0]
        valid = L.valid_mask(self.gm.device).reshape(n16, 16)
        return (self.group_mask()[:, None, :] & valid[:, :, None]).to(torch.uint8).view(L.weight_shape)

    def enabled_slots(self) -> int:
        return int((self.kc * self.nv).sum())

    def env_of(self, k: int):
        """(groups [n] ascending u, env [n, k] sorted local positions) of the class-k groups."""
        u = torch.nonzero((self.kc == k) & (self.nv > 0)).squeeze(1)
        env = torch.nonzero(self.gm[u])[:, 1].view(len(u), k) if len(u) else torch.zeros((0, k), dtype=torch.int64)
        return u, env

    def fingerprint(self) -> str:
        h = hashlib.sha256(f"{SCHEMA}|support|mixed".encode())
        gv = self.nv > 0
        h.update(self.kc[gv].to(torch.int8).cpu().numpy().tobytes())
        h.update(_mask_words(self.gm[gv]).cpu().numpy().tobytes())
        h.update(torch.nonzero(gv).squeeze(1).to(torch.int64).cpu().numpy().tobytes())
        return h.hexdigest()


def _mask_words(gm: torch.Tensor) -> torch.Tensor:
    """[n] int64 exact 64-bit encoding of [n, 64] bool masks."""
    return (gm.to(torch.int64) << torch.arange(64, device=gm.device)).sum(1)


def class_order(records: int, gpb: int, graph_seed: int, layout_seed: int) -> torch.Tensor:
    """[records, gpb - 1] keyed uniform orders of the non-anchor ranks 1..gpb-1 of every record."""
    return LS.permutations(records, ("Mix-class", f"graph{graph_seed}", f"layout{layout_seed}"), width=gpb - 1) + 1


@torch.no_grad()
def build(policy: str, L, buckets: torch.Tensor, w=None, graph_seed: int = 0, layout_seed: int = 0, device="cpu",
          *, Q: int, L_cand: int = 1) -> MixedSupport:
    if policy not in POLICIES:
        raise ValueError(f"mixed policy {policy!r} (one of {POLICIES})")
    if policy != "Mix-BP" and w is None:
        raise ValueError(f"{policy} needs the pre-pruning weights")
    geo = LS.geometry(L, buckets.cpu(), "cpu", Q=Q)
    nv, q, uv, gpb = geo["nv"], geo["q"], geo["uv"], geo["gpb"]
    n16 = nv.numel()
    if gpb - 1 < N_K8:
        raise Unsupported(f"C = {L.block}: {gpb} row groups per record cannot hold one anchor + {N_K8} K8 groups")
    # per-record neurons / full groups (valid groups are a prefix of the record's groups)
    rec_n = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, geo["rec"], nv[uv])
    rec_full = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, geo["rec"], (nv[uv] == 16).long())
    bad = rec_n < MIN_NEURONS
    if bool(bad.any()):
        raise Unsupported(f"{int(bad.sum())} of {geo['records']} records have fewer than {MIN_NEURONS} valid neurons "
                          f"(min {int(rec_n.min())}): the anchor + {N_K8} K8 groups do not fit (rejected, no fallback)")
    order = class_order(geo["records"], gpb, graph_seed, layout_seed)  # [records, gpb - 1] ranks
    elig = order < rec_full[:, None]  # full non-anchor groups (rank < number of full groups)
    take = elig & (torch.cumsum(elig.long(), 1) <= N_K8)
    k8_rank = torch.zeros((geo["records"], gpb), dtype=torch.bool)
    k8_rank.scatter_(1, torch.where(take, order, torch.zeros_like(order)), take)
    k8_rank[:, 0] = False
    kc = torch.zeros(n16, dtype=torch.int64)
    rr = geo["rank_rec"]
    is_anchor = rr == 0
    is_k8 = k8_rank[geo["rec"], rr]
    kc[uv] = torch.where(is_anchor, 64, torch.where(is_k8, 8, 16))
    gm = torch.zeros((n16, 64), dtype=torch.bool)
    cand = torch.full((n16,), -1, dtype=torch.int64)
    anchor = torch.zeros(n16, dtype=torch.bool)
    anchor[uv[is_anchor]] = True
    gm[uv[is_anchor]] = True
    a_all = LS.feature_importance(w, L) if policy != "Mix-BP" else None
    extra = {}
    for k in (16, 8):
        sel = kc[uv] == k
        u_k = uv[sel]
        q_k = q[u_k]
        m_qk = torch.bincount(q_k, minlength=Q)
        # rank inside the (q, k) stratum, canonical (ascending u) order
        o = torch.argsort(q_k * n16 + u_k)
        qs = q_k[o]
        rank = torch.empty_like(u_k)
        rank[o] = torch.arange(len(u_k)) - torch.searchsorted(qs, qs, right=False)
        e0, _, npm, part = LS.bp_stream(m_qk, rank, q_k, k, ("Mix-BP", k, f"graph{graph_seed}", f"layout{layout_seed}"))
        st = dict(groups=int(len(u_k)), neurons=int(nv[u_k].sum()), bp_permutations=npm,
                  partial_last_permutation_sets=part, sets_with_class=int((m_qk > 0).sum()))
        if policy == "Mix-BP":
            env = e0
            cand[u_k] = 0
        else:
            a_v = a_all[u_k]
            if policy == "Mix-WRS":
                r = LS.wrs_select(a_v, e0, k, L_cand, ("Mix-WRS", k, f"graph{graph_seed}", f"layout{layout_seed}",
                                                       "L-stream"))
                env = r["env"]
                cand[u_k] = r["cand"]
                st.update(chosen_candidate_hist=torch.bincount(r["cand"], minlength=L_cand).tolist(),
                          duplicate_candidates=int(r["duplicates"].sum()))
            else:
                env = LS.topk_select(a_v, k)
                cand[u_k] = 0
            st.update(energy=LS.energy_stats(a_v, env), energy_candidate0=LS.energy_stats(a_v, e0))
        gm[u_k[:, None], env] = True
        extra[f"K{k}"] = st
    if a_all is not None:
        a_v = a_all[uv]
        kept = (a_v * gm[uv]).sum(1)
        extra["energy_layer_fraction"] = float(kept.sum() / a_v.sum())
        del a_all, a_v
    sup = MixedSupport(policy, kc.to(device), gm.to(device), nv.to(device), q.to(device), anchor.to(device),
                       cand.to(device))
    H = {f"H{k}": int(nv[kc == k].sum()) for k in CLASSES}
    sup.stats = dict(schema=SCHEMA, policy=policy, L=L_cand if policy == "Mix-WRS" else None, graph_seed=graph_seed,
                     layout_seed=layout_seed, H=int(nv.sum()), W0=64 * int(nv.sum()), enabled_slots=sup.enabled_slots(),
                     rho_mask=sup.enabled_slots() / (64 * int(nv.sum())), biases=int(nv.sum()), **H,
                     anchor_groups=int(anchor.sum()), records=geo["records"], Q=geo["Q"],
                     sets_occupied=geo["sets_occupied"], sets_empty=geo["sets_empty"],
                     record_neurons=dict(min=int(rec_n.min()), max=int(rec_n.max())),
                     class_rule=f"rank-0 group K64; {N_K8} K8 among the full non-anchor groups by the keyed order "
                                "('Mix-class', graph, layout); every other valid group K16",
                     per_record_budget_exact=bool(((torch.zeros(geo['records'], dtype=torch.int64).index_add_(
                         0, geo["rec"], kc[uv] * nv[uv])) == 16 * rec_n).all()),
                     support_fingerprint=sup.fingerprint(), class_fingerprint=class_fingerprint(sup), **extra)
    return sup


def class_fingerprint(sup: MixedSupport) -> str:
    """sha256 of the fan-in class map alone (equal for Mix-BP and the weighted variants of one graph / layout)."""
    h = hashlib.sha256(f"{SCHEMA}|classes".encode())
    h.update(sup.kc.to(torch.int8).cpu().numpy().tobytes())
    return h.hexdigest()
