"""Randomized and fixed local supports of fan-in k inside the C256 K64 envelope (random-support study,
results/RANDOM_SUPPORTS_NORMALIZATION_CONTROLS.md section 3).

The controlled graph is the K64 envelope graph of a ``KgxLayer`` (kgen_graph generator, quantum 1): every valid
neuron keeps its output g, slot n, logical record (g, j) and the record's input set q (64 local inputs
``feat[q]``, ascending global ids).  A local support only chooses WHICH k of the record's 64 local inputs each
16-neuron row group u = (g, j, c // 16) retains (``env[u]``: k distinct sorted local positions); the 16 neurons
of a row group share it with independent weights (their pre-pruning values at those coordinates) and their
own bias.  Wholly padded row groups (nv_u = 0) take no support and no budget; partial groups keep fan-in k for
their valid neurons.  Exactly k H valid slots are enabled (k = 16: rho = 0.25 of W0 = 64 H).

Policies (``POLICIES``; nkt = 64 / k fixed tiles [t k, t k + k)):

* ``FP-mag``      the parent's P16 / P8 at its floor: the highest |w| tile of the four (eight) fixed tiles per
                  row group (width_masks.build, the score-based control: it keeps a magnitude advantage);
* ``FP-balanced`` the same fixed partition; inside each original record the i-th valid row group (canonical
                  order) takes tile (phi_r + i) mod nkt, phi_r a seeded uniform phase per record;
* ``RP-record``   per original record one seeded permutation of its 64 local inputs, cut into nkt disjoint
                  k-subsets; allocated to the record's row groups by the FP-balanced rule (phase phi_r);
* ``RP-row``      every valid row group draws an independent uniform k-subset (first k of a permutation);
* ``BP-row``      within each input set q, a stream of independent permutations of 64, each cut into nkt
                  disjoint k-subsets; the valid row groups of q in canonical (ascending u: g, j, c // 16) order
                  consume the stream (group i of q: permutation i // nkt, subset i mod nkt).

Random policies are uniform (no weight dependence).  RNG: ``torch.Generator`` (CPU, Philox-independent
Mersenne Twister) per purpose, seeded with the first 63 bits of sha256(SCHEMA | policy | k | graph seed |
layout seed | purpose | chunk); permutations are ``argsort`` (stable) of int64 keys drawn uniformly from
[0, 2^62) in chunks of ``CHUNK_ROWS`` streams (a tie -- probability below 64^2 / 2^63 per stream -- is
resolved by the lower local position).  Phases: ``randint(0, nkt)`` from their own keyed generator.  The
selected positions of every subset are sorted before packing, so global indices and weights stay paired.
The graph seed is the envelope graph's; the layout seed only selects local subsets; checkpoint / training and
data seeds are separate (callers).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch

SCHEMA = "granule.local_supports/1"
POLICIES = ("FP-mag", "FP-balanced", "RP-record", "RP-row", "BP-row")
CHUNK_ROWS = 1 << 16
RNG = ("torch.Generator('cpu') (mt19937) seeded per purpose / chunk with sha256-derived 63-bit seeds; "
       "permutation = stable argsort of int64 keys uniform in [0, 2^62); ties -> lower local position")


def key_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in (SCHEMA,) + parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF


def permutations(n: int, base: tuple, width: int = 64) -> torch.Tensor:
    """[n, width] int64: n independent uniform permutations of range(width) from the keyed stream ``base``."""
    out = torch.empty((n, width), dtype=torch.int64)
    for c, s0 in enumerate(range(0, n, CHUNK_ROWS)):
        m = min(CHUNK_ROWS, n - s0)
        g = torch.Generator().manual_seed(key_seed(*base, "perm", c))
        keys = torch.randint(0, 1 << 62, (m, width), generator=g, dtype=torch.int64)
        out[s0:s0 + m] = torch.argsort(keys, dim=1, stable=True)
    return out


def phases(n: int, nkt: int, base: tuple) -> torch.Tensor:
    g = torch.Generator().manual_seed(key_seed(*base, "phase"))
    return torch.randint(0, nkt, (n,), generator=g, dtype=torch.int64)


@dataclass
class LocalSupport:
    """``env`` [n16, k] int64 sorted local positions of every row group (rows of padded groups: arange(k), never
    used), ``nv`` [n16] valid neurons per group, ``q`` [n16] input set of the group (-1 padded), ``subset`` [n16]
    subset id inside its permutation / tile id (-1 padded), ``stats`` (selection record)."""

    policy: str
    k: int
    env: torch.Tensor
    nv: torch.Tensor
    q: torch.Tensor
    subset: torch.Tensor
    stats: dict = field(default_factory=dict)

    def group_mask(self) -> torch.Tensor:
        """[n16, 64] bool local support of every row group (padded groups all False)."""
        m = torch.zeros((self.env.shape[0], 64), dtype=torch.bool, device=self.env.device)
        m.scatter_(1, self.env, True)
        m &= (self.nv > 0)[:, None]
        return m

    def element_mask(self, L) -> torch.Tensor:
        """[G, NB, C, 64] uint8 of the enabled valid slots."""
        n16 = self.env.shape[0]
        gm = self.group_mask()
        valid = L.valid_mask(self.env.device).reshape(n16, 16)
        return (gm[:, None, :] & valid[:, :, None]).to(torch.uint8).view(L.weight_shape)

    def enabled_slots(self) -> int:
        return int(self.k * self.nv.sum())

    def fingerprint(self) -> str:
        """sha256 of (k, valid-group env rows): the retained-support identity (separate from the envelope graph,
        the index layout of an execution format and the parameter hashes)."""
        h = hashlib.sha256(f"{SCHEMA}|support|k={self.k}".encode())
        gv = self.nv > 0
        h.update(self.env[gv].to(torch.int8).cpu().numpy().tobytes())
        h.update(torch.nonzero(gv).squeeze(1).to(torch.int64).cpu().numpy().tobytes())
        return h.hexdigest()


def block_sets(L, buckets: torch.Tensor, device) -> torch.Tensor:
    """[G NB] int64 input set q of every logical block (-1: padded block)."""
    b = buckets.to(device).to(torch.int64)
    NB = L.blocks_per_group
    rb = torch.repeat_interleave(torch.arange(len(b), device=device), b[:, 3])
    kb = torch.arange(len(rb), device=device) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((L.groups * NB,), -1, dtype=torch.int64, device=device)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    return bq


def geometry(L, buckets: torch.Tensor, device="cpu") -> dict:
    """Row-group geometry of the envelope: nv [n16], q [n16], valid groups (ascending u), their rank inside the
    record and inside the set, per-set group / record counts m_q / b_q."""
    from .width_masks import layout_counts

    nv = layout_counts(L, device)["nv"]
    gpb = L.block // 16
    bq = block_sets(L, buckets, device)
    n16 = nv.numel()
    u = torch.arange(n16, device=device)
    q = torch.where(nv > 0, bq[u // gpb], torch.full_like(u, -1))
    if bool(((nv > 0) & (q < 0)).any()):
        raise AssertionError("a valid row group lies in a block without an input set")
    uv = torch.nonzero(nv > 0).squeeze(1)
    rank_rec = uv % gpb  # valid groups of a block are its first ones (valid slots are a prefix)
    blk = uv // gpb
    if bool((nv[uv[rank_rec > 0] - 1] != 16).any()):
        raise AssertionError("valid row groups of a block are not a prefix of full groups")
    qv = q[uv]
    order = torch.argsort(qv * n16 + uv)  # (q, u) order; stable by construction (keys distinct)
    rank_set = torch.empty_like(uv)
    qs = qv[order]
    starts = torch.searchsorted(qs, qs, right=False)
    rank_set[order] = torch.arange(len(uv), device=device) - starts
    Q = int(buckets[:, 0].max()) + 1 if len(buckets) else 0
    Q = max(Q, int(q.max()) + 1)
    m_q = torch.bincount(qv, minlength=Q)
    rec_valid = (nv.view(-1, gpb) > 0).any(1)
    b_q = torch.bincount(bq[rec_valid], minlength=Q)
    rec_id = torch.cumsum(rec_valid.to(torch.int64), 0) - 1  # canonical record index of every valid block
    return dict(nv=nv, q=q, uv=uv, rank_rec=rank_rec, rank_set=rank_set, blk=blk, rec=rec_id[blk], m_q=m_q,
                b_q=b_q, records=int(rec_valid.sum()), gpb=gpb, Q=Q,
                groups_per_record=(nv.view(-1, gpb) > 0).sum(1)[rec_valid])


@torch.no_grad()
def build(policy: str, k: int, L, buckets: torch.Tensor, w=None, graph_seed: int = 0, layout_seed: int = 0,
          device="cpu") -> LocalSupport:
    """The local support of ``policy`` with fan-in k (module docstring); ``w`` (pre-pruning weights) only for
    FP-mag."""
    if policy not in POLICIES:
        raise ValueError(f"local support policy {policy!r} (one of {POLICIES})")
    if 64 % k or k not in (8, 16):
        raise ValueError("local supports: k in (8, 16)")
    nkt = 64 // k
    geo = geometry(L, buckets.cpu(), "cpu")
    nv, q, uv = geo["nv"], geo["q"], geo["uv"]
    n16 = nv.numel()
    base = (policy, k, f"graph{graph_seed}", f"layout{layout_seed}")
    env = torch.arange(k, dtype=torch.int64).repeat(n16, 1)
    subset = torch.full((n16,), -1, dtype=torch.int64)
    ar = torch.arange(k, dtype=torch.int64)
    if policy == "FP-mag":
        from . import width_masks as WM

        if w is None:
            raise ValueError("FP-mag needs the pre-pruning weights")
        ms = WM.build("P16" if k == 16 else "P8", k / 64, w, L)
        t = torch.argmax(ms.tiles.to(torch.int8), dim=1).cpu()
        if not bool((ms.tiles.sum(1).cpu()[nv > 0] == 1).all()):
            raise AssertionError("FP-mag: not exactly one tile per valid group")
        subset[uv] = t[uv]
        env[uv] = t[uv, None] * k + ar
        extra = dict(selection=ms.stats)
    elif policy == "FP-balanced":
        ph = phases(geo["records"], nkt, base)
        t = (ph[geo["rec"]] + geo["rank_rec"]) % nkt
        subset[uv] = t
        env[uv] = t[:, None] * k + ar
        extra = {}
    elif policy == "RP-record":
        perm = permutations(geo["records"], base)
        ph = phases(geo["records"], nkt, base)
        s = (ph[geo["rec"]] + geo["rank_rec"]) % nkt
        sub = perm.view(-1, nkt, k)[geo["rec"], s]
        subset[uv] = s
        env[uv] = sub.sort(1).values
        extra = {}
    elif policy == "RP-row":
        perm = permutations(len(uv), base)
        subset[uv] = 0
        env[uv] = perm[:, :k].sort(1).values
        extra = {}
    else:  # BP-row
        m_q = geo["m_q"]
        npq = (m_q + nkt - 1) // nkt
        off = torch.cumsum(npq, 0) - npq
        perm = permutations(int(npq.sum()), base)
        qv = q[uv]
        pid = off[qv] + geo["rank_set"] // nkt
        s = geo["rank_set"] % nkt
        subset[uv] = s
        env[uv] = perm.view(-1, nkt, k)[pid, s].sort(1).values
        extra = dict(permutations=int(npq.sum()), partial_last_permutation_sets=int((m_q % nkt != 0).sum()))
    sup = LocalSupport(policy, k, env.to(device), nv.to(device), q.to(device), subset.to(device))
    sup.stats = dict(schema=SCHEMA, policy=policy, k=k, nkt=nkt, graph_seed=graph_seed,
                     layout_seed=layout_seed if policy not in ("FP-mag",) else None, rng=RNG,
                     H=int(nv.sum()), W0=64 * int(nv.sum()), enabled_slots=sup.enabled_slots(),
                     rho_mask=k / 64, valid_groups=len(uv), partial_groups=int(((nv > 0) & (nv < 16)).sum()),
                     records=geo["records"], sets_occupied=int((geo["m_q"] > 0).sum()), Q=geo["Q"],
                     groups_per_record=dict(min=int(geo["groups_per_record"].min()),
                                            max=int(geo["groups_per_record"].max())),
                     m_q=dict(min=int(geo["m_q"].min()), max=int(geo["m_q"].max()),
                              mean=float(geo["m_q"].double().mean())),
                     b_q=dict(min=int(geo["b_q"].min()), max=int(geo["b_q"].max()),
                              mean=float(geo["b_q"].double().mean())),
                     support_fingerprint=sup.fingerprint(), **extra)
    sup.stats.update(balance(sup, geo))
    return sup


def balance(sup: LocalSupport, geo: dict) -> dict:
    """Allocation checks: per record, the use counts of each subset / tile id (balanced policies: differ by at
    most one; all nkt subsets present when the record has >= nkt groups) and every row's distinct in-envelope
    coordinates."""
    nkt = 64 // sup.k
    uv = geo["uv"]
    env = sup.env.cpu()[uv]
    ok_distinct = bool((env[:, 1:] > env[:, :-1]).all()) and int(env.min()) >= 0 and int(env.max()) < 64
    out = dict(rows_sorted_distinct_in_envelope=ok_distinct)
    if sup.policy in ("FP-balanced", "RP-record", "FP-mag", "BP-row"):
        key = geo["rec"] if sup.policy != "BP-row" else None
        if key is not None:
            cnt = torch.zeros((geo["records"], nkt), dtype=torch.int64)
            cnt.index_put_((key, sup.subset.cpu()[uv]), torch.ones(len(uv), dtype=torch.int64), accumulate=True)
            spread = cnt.max(1).values - cnt.min(1).values
            full = geo["groups_per_record"] >= nkt
            out.update(per_record_subset_spread_max=int(spread.max()),
                       records_with_all_subsets=int((cnt > 0).all(1).sum()),
                       records_with_at_least_nkt_groups=int(full.sum()),
                       all_subsets_in_every_record_with_nkt_groups=bool((cnt[full] > 0).all()))
    return out


# ------------------------------------------------------------------ coverage models (section 3.3)
def p_group(t: int, S: int = 64, D: int = 1024) -> float:
    """Probability that t distinct features share one set of a global round: C(S-1, t-1) / C(D-1, t-1)."""
    return math.comb(S - 1, t - 1) / math.comb(D - 1, t - 1)


def a_subset(t: int, k: int, S: int = 64) -> float:
    """t given local positions fall in ONE k-subset of a uniform permutation of S cut into S / k subsets."""
    return math.comb(k - 1, t - 1) / math.comb(S - 1, t - 1) if t <= k else 0.0


def c_subset(t: int, k: int, S: int = 64) -> float:
    """t given local positions lie in one uniform k-subset of S."""
    return math.comb(k, t) / math.comb(S, t) if t <= k else 0.0


def set_hit(policy: str, t: int, k: int, m_q: torch.Tensor, b_q=None, rec_groups=None, rec_set=None,
            used_tiles=None, S: int = 64) -> torch.Tensor:
    """[Q] float64 probability that a tuple co-located in set q is inside the support of one row group of q.

    RP-record: ``rec_groups`` / ``rec_set`` (actual groups per record and their set) refine the model to
    1 - prod_records(1 - min(m_r, nkt) a_t / nkt) (= 1 - (1 - a_t)^b_q when every record uses all subsets);
    RP-row: 1 - (1 - c_t)^m_q;  BP-row: 1 - (1 - a_t)^L (1 - u a_t / nkt), L = m_q // nkt, u = m_q % nkt;
    FP-balanced / FP-mag (one fixed partition per set): (tiles of the partition used by the set / nkt) a_t, exact for
    the realized tile use (``used_tiles`` [Q]; default all nkt)."""
    nkt = S // k
    a, c = a_subset(t, k, S), c_subset(t, k, S)
    m = m_q.to(torch.float64)
    if policy == "RP-record":
        if rec_groups is None:
            return 1 - (1 - a) ** b_q.to(torch.float64)
        f = torch.clamp(rec_groups.to(torch.float64), max=nkt) * a / nkt
        lg = torch.zeros(m_q.numel(), dtype=torch.float64).index_add_(0, rec_set.to(torch.int64), torch.log1p(-f))
        return -torch.expm1(lg)
    if policy == "RP-row":
        return -torch.expm1(m * math.log1p(-c)) if c < 1 else (m > 0).to(torch.float64)
    if policy == "BP-row":
        L = torch.div(m_q, nkt, rounding_mode="floor").to(torch.float64)
        u = (m_q % nkt).to(torch.float64)
        return -torch.expm1(L * math.log1p(-a) + torch.log1p(-u * a / nkt)) if a < 1 else (m > 0).to(torch.float64)
    if policy in ("FP-balanced", "FP-mag"):
        ut = torch.full_like(m, nkt) if used_tiles is None else used_tiles.to(torch.float64)
        return torch.where(m > 0, ut / nkt * a, torch.zeros_like(m))
    raise ValueError(policy)


def layer_expectation(h_q: torch.Tensor, t: int, sets_per_round: int, S: int = 64, D: int = 1024) -> float:
    """Independent-round expectation 1 - prod_r (1 - p_g mean_{q in round r} h_q) (log1p / expm1)."""
    hr = h_q.view(-1, sets_per_round).mean(1)
    return float(-torch.expm1(torch.log1p(-p_group(t, S, D) * hr).sum()))


def ideal_prediction(policy: str, t: int, k: int, groups: int, Q: int, records: int, D: int = 1024,
                     S: int = 64) -> float:
    """Ideal balanced quotas: m_q in {floor, ceil}(groups / Q) with the exact total (the ceil quotas spread
    round-robin over the rounds), b_q = records / Q (its mean; RP-record with all subsets in every record)."""
    spr = D // S
    base, extra = divmod(groups, Q)
    idx = torch.arange(Q)
    key = (idx % spr) * (Q // spr) + idx // spr  # round-robin over the rounds
    m_q = torch.full((Q,), base, dtype=torch.int64)
    m_q[torch.argsort(key)[:extra]] = base + 1
    b_q = torch.full((Q,), records / Q, dtype=torch.float64)
    return layer_expectation(set_hit(policy, t, k, m_q, b_q, S=S), t, spr, S, D)


def per_output_union_bound(records_per_output: int, t: int, k: int, D: int = 1024) -> float:
    """min(1, n_rec C(k, t) / C(D, t)): an output's tuple coverage when it has n_rec independent k-input records."""
    return min(1.0, records_per_output * math.comb(k, t) / math.comb(D, t))


def refined_inputs(sup: "LocalSupport", L, buckets) -> dict:
    """Actual m_q, b_q, groups per record and their sets, and the tiles of the fixed partition used per set."""
    geo = geometry(L, buckets.cpu(), "cpu")
    nkt = 64 // sup.k
    gpb = geo["gpb"]
    rec_valid = (geo["nv"].view(-1, gpb) > 0).any(1)
    rec_set = block_sets(L, buckets.cpu(), "cpu")[rec_valid]
    used = torch.zeros((geo["Q"], nkt), dtype=torch.bool)
    if sup.policy in ("FP-balanced", "FP-mag"):
        used[geo["q"][geo["uv"]], sup.subset.cpu()[geo["uv"]]] = True
    return dict(m_q=geo["m_q"], b_q=geo["b_q"], rec_groups=geo["groups_per_record"], rec_set=rec_set,
                used_tiles=used.sum(1), Q=geo["Q"])
