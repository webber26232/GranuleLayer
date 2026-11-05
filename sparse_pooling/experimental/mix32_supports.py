"""Cover-Mix32: K32 / K16 / K8 fan-in reallocation at exactly the uniform-K16 budget of every original record, with
complete record-level input coverage (results/INTERACTION_COVERAGE_K16_DX.md section 3.2; a BUDGET-ALLOCATION
experiment: an effect cannot be attributed to the K32 groups alone, independently of the compensating K8 groups).

For p in {1, 2} and every valid original record (16-neuron row groups in canonical order, full groups first, at most
one partial tail group) of n valid neurons:

* p of its FULL row groups at K32 and 2p further full groups at K8, chosen by ONE keyed uniform permutation of the
  record's full groups (the first p in permutation order are K32, the next 2p K8; key
  ("Cover-Mix32", "p<p>", graph, layout, "class", perm, chunk): a new, policy-specific namespace);
* every other valid group, the partial tail group included, at K16.

Slots: 32 (16 p) + 8 (32 p) + 16 (n - 48 p) = 16 n.  A record with fewer than 3 p full groups is REJECTED
(``Unsupported``): its budget is never changed.  Primary model (H 12.5M, C256): p1 H32 786,432 / H16 10,140,704 /
H8 1,572,864; p2 1,572,864 / 7,781,408 / 3,145,728; 200,000,000 slots and 12,500,000 biases either way.

Coordinates: each group first keeps the TopK of its assigned degree (local_supports.topk_select on the FP64 row-group
importances a[u, j] = sum_valid c W_pre[u, c, j]^2, ties to the lower coordinate); then the minimum-energy-loss
coverage repair generalized to variable row degrees (local_supports.cover_repair_mask) makes all 64 local inputs of
the record present while keeping every group's assigned degree.  Labels / targets / losses are never read.
Every neuron keeps its output g, slot n, record (g, j), input set q, weights (its pre-pruning values at the kept
coordinates) and bias.  Execution: experimental/kgx_grouped.py with the declared class list (32, 16, 8).
"""
from __future__ import annotations

import hashlib
import time

import torch

from . import local_supports as LS
from .mixed_supports import MixedSupport, Unsupported

SCHEMA = "granule.mix32_supports/1"
POLICIES = ("Cover-Mix32",)
CLASSES = (32, 16, 8)
P_VALUES = (1, 2)


def class_permutation(records: int, gpb: int, p: int, graph_seed: int, layout_seed: int) -> torch.Tensor:
    """[records, gpb] keyed uniform permutations of the ranks 0..gpb-1 of every record (Cover-Mix32 namespace)."""
    return LS.permutations(records, ("Cover-Mix32", f"p{p}", f"graph{graph_seed}", f"layout{layout_seed}", "class"),
                           width=gpb)


def assign_classes(rec: torch.Tensor, rank: torch.Tensor, nv: torch.Tensor, gpb: int, records: int, p: int,
                   graph_seed: int, layout_seed: int) -> torch.Tensor:
    """[n] fan-in class of every valid group (rec / rank / nv of the valid groups in ascending u)."""
    rec_full = torch.zeros(records, dtype=torch.int64).index_add_(0, rec, (nv == 16).long())
    if bool((rec_full < 3 * p).any()):
        bad = int(torch.nonzero(rec_full < 3 * p)[0])
        raise Unsupported(f"Cover-Mix32 p{p}: record {bad} has {int(rec_full[bad])} full row groups < 3 p = {3 * p}: "
                          "the K32 / K8 classes do not fit (rejected; the budget is never changed)")
    perm = class_permutation(records, gpb, p, graph_seed, layout_seed)  # ranks in keyed order
    elig = perm < rec_full[:, None]  # full groups (valid groups are a prefix; full ones first)
    pos = torch.cumsum(elig.long(), 1)  # 1-based position among the eligible ranks
    k32 = elig & (pos <= p)
    k8 = elig & (pos > p) & (pos <= 3 * p)
    val =torch.where(k32, torch.full_like(perm, 32), torch.where(k8, torch.full_like(perm, 8), torch.full_like(perm, 16)))
    cls_rank = torch.full((records, gpb), 16, dtype=torch.int64)
    cls_rank.scatter_(1, perm, val)
    return cls_rank[rec, rank]


@torch.no_grad()
def build(policy: str, p: int, L, buckets: torch.Tensor, w, graph_seed: int = 0, layout_seed: int = 0, device="cpu",
          *, Q: int) -> MixedSupport:
    if policy not in POLICIES:
        raise ValueError(f"mix32 policy {policy!r} (one of {POLICIES})")
    if p not in P_VALUES:
        raise ValueError(f"Cover-Mix32: p in {P_VALUES}")
    if w is None:
        raise ValueError("Cover-Mix32 needs the pre-pruning weights")
    t_all = time.perf_counter()
    geo = LS.geometry(L, buckets.cpu(), "cpu", Q=Q)
    nv, q, uv, gpb = geo["nv"], geo["q"], geo["uv"], geo["gpb"]
    n16 = nv.numel()
    rec, rank = geo["rec"], geo["rank_rec"]
    kc_v = assign_classes(rec, rank, nv[uv], gpb, geo["records"], p, graph_seed, layout_seed)
    t0 = time.perf_counter()
    a_v = LS.feature_importance(w, L)[uv]
    t1 = time.perf_counter()
    X0 = torch.zeros((len(uv), 64), dtype=torch.bool)
    for k in CLASSES:
        s = kc_v == k
        if bool(s.any()):
            X0[torch.nonzero(s).squeeze(1)[:, None], LS.topk_select(a_v[s], k)] = True
    t2 = time.perf_counter()
    r = LS.cover_repair_mask(a_v, X0, rec, rank, gpb)
    t3 = time.perf_counter()
    X = r["X"]
    if not torch.equal(X.sum(1), kc_v):
        raise AssertionError("Cover-Mix32: a group's degree differs from its assigned class")
    kc = torch.zeros(n16, dtype=torch.int64)
    kc[uv] = kc_v
    gm = torch.zeros((n16, 64), dtype=torch.bool)
    gm[uv] = X
    sup = MixedSupport(policy, kc.to(device), gm.to(device), nv.to(device), q.to(device),
                       torch.zeros(n16, dtype=torch.bool, device=device),
                       torch.zeros(n16, dtype=torch.int64, device=device))
    sup.classes = CLASSES
    sup.p = p
    rec_n = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, rec, nv[uv])
    rec_slots = torch.zeros(geo["records"], dtype=torch.int64).index_add_(0, rec, kc_v * nv[uv])
    full = r["groups_per_record"] == gpb
    kept0 = (a_v * X0).sum(1)
    kept = (a_v * X).sum(1)
    tot = a_v.sum(1)
    rec_kept0 = torch.zeros(geo["records"], dtype=torch.float64).index_add_(0, rec, kept0)
    rel = torch.where(rec_kept0 > 0, r["loss"] / rec_kept0.clamp_min(1e-300), torch.zeros_like(r["loss"]))
    mult = LS.input_multiplicity(X, nv[uv], rec, geo["records"])
    H = {f"H{k}": int(nv[kc == k].sum()) for k in CLASSES}
    per_class = {}
    for k in CLASSES:
        s = kc_v == k
        per_class[f"K{k}"] = dict(groups=int(s.sum()), neurons=int(nv[uv][s].sum()),
                                  groups_per_record=LS._dist(torch.zeros(geo["records"], dtype=torch.int64)
                                                             .index_add_(0, rec, s.long())),
                                  energy_fraction=float(kept[s].sum() / tot[s].sum()) if bool(s.any()) else None)
    sup.stats = dict(schema=SCHEMA, policy=f"{policy}:p{p}", p=p, classes=list(CLASSES), graph_seed=graph_seed,
                     layout_seed=layout_seed, H=int(nv.sum()), W0=64 * int(nv.sum()), enabled_slots=sup.enabled_slots(),
                     rho_mask=sup.enabled_slots() / (64 * int(nv.sum())), biases=int(nv.sum()), **H,
                     anchor_groups=0, records=geo["records"], Q=geo["Q"], sets_occupied=geo["sets_occupied"],
                     sets_empty=geo["sets_empty"], record_neurons=dict(min=int(rec_n.min()), max=int(rec_n.max())),
                     class_rule=f"one keyed permutation of the record's full groups ('Cover-Mix32', p{p}, graph, layout, "
                                f"class): first {p} K32, next {2 * p} K8; every other valid group K16",
                     per_record_budget_exact=bool((rec_slots == 16 * rec_n).all()), per_class=per_class,
                     energy_topk_assigned=float(kept0.sum() / tot.sum()), energy=dict(layer_fraction=float(
                         kept.sum() / tot.sum())),
                     repair=dict(records=int(len(r["loss"])), records_full=int(full.sum()),
                                 records_tail=int((~full).sum()), records_repaired=int((r["swaps"] > 0).sum()),
                                 swaps=int(r["swaps"].sum()), swaps_equal_initial_missing=bool(
                                     torch.equal(r["swaps"], r["missing0"])),
                                 missing0_full=LS._dist(r["missing0"][full]),
                                 missing0_tail=LS._dist(r["missing0"][~full]) if bool((~full).any()) else {},
                                 rounds=r["rounds"], energy_loss_relative_to_assigned_topk=float(
                                     r["loss"].sum() / kept0.sum()), record_relative_loss=LS._dist(rel)),
                     records_complete=int((mult["groups"] > 0).all(1).sum()),
                     multiplicity_groups=LS._dist(mult["groups"].flatten()),
                     multiplicity_neurons=LS._dist(mult["neurons"].flatten()),
                     seconds=dict(importance=t1 - t0, topk=t2 - t1, repair=t3 - t2, total=time.perf_counter() - t_all),
                     importance="a[u, j] = sum_valid c W_pre[u, c, j]^2 (FP64, sequential over c)")
    sup.stats.update(support_fingerprint=sup.fingerprint(), class_fingerprint=class_fingerprint(sup))
    return sup


def class_fingerprint(sup: MixedSupport) -> str:
    h = hashlib.sha256(f"{SCHEMA}|classes|p{sup.p}".encode())
    h.update(sup.kc.to(torch.int8).cpu().numpy().tobytes())
    return h.hexdigest()


def tuple_incidence(classes_per_record: dict, t: int) -> int:
    """sum over row groups of C(k_u, t): t-tuple incidences of one record (a shared 16-neuron support counts once)."""
    import math

    return sum(n * math.comb(k, t) for k, n in classes_per_record.items())
