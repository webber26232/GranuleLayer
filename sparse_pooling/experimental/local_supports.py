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
                  consume the stream (group i of q: permutation i // nkt, subset i mod nkt);
* ``WRS``         weighted random-subset selection with L candidates (weighted-support study,
                  results/WEIGHTED_RANDOM_SUPPORTS_ANCHORS.md section 4): candidate 0 is the exact BP-row subset of
                  the same (k, graph seed, layout seed); candidates 1..L-1 are independently keyed uniform
                  k-subsets (first k of a keyed permutation per candidate id, streamed in CHUNK_ROWS chunks of the
                  valid groups); the candidate of largest score(A) = sum_{j in A} a[u, j] wins, ties to the
                  smallest candidate id.  Candidate c is keyed by c alone, so the first L' candidates of any L >= L'
                  are those of L' (nested families: the retained energy is nondecreasing in L).  Duplicate candidates
                  are kept (no resampling); a duplicate can never win over its smaller-id copy;
* ``TopK``        the k largest a[u, j] of the row group, ties to the lower local coordinate (stable descending
                  sort) -- the unconstrained magnitude endpoint (FP-mag is the best CONTIGUOUS tile).
* ``Cover-TopK``  TopK followed by a per-record coverage repair (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md section 4):
                  inside each original record every one of its 64 local inputs must be retained by at least one of
                  the record's valid row groups, each group keeping exactly k distinct inputs.  While an input j is
                  uncovered, the swap (u, d -> j) of minimum energy loss a[u, d] - a[u, j] over every uncovered j,
                  group u and retained donor d with record count c[d] >= 2 is applied (ties lexicographic in (j, u, d),
                  original local / canonical group order); see ``cover_repair``.  Requires m k >= 64 for every
                  record of m valid groups (else ValueError, no fallback).

Feature importance (WRS / TopK): a[u, j] = sum_{valid neurons c of u} W_pre[u, c, j]^2, FP64, accumulated by
explicit sequential adds over c = 0..15 (padded neurons masked to 0) -- elementwise operations only, so the value
is independent of chunking and evaluation order; score(A) adds the k values in ascending coordinate order.

Random policies are uniform (no weight dependence).  RNG: ``torch.Generator`` (CPU, Philox-independent
Mersenne Twister) per purpose, seeded with the first 63 bits of sha256(SCHEMA | policy | k | graph seed |
layout seed | purpose | chunk); permutations are ``argsort`` (stable) of int64 keys drawn uniformly from
[0, 2^62) in chunks of ``CHUNK_ROWS`` streams (a tie -- probability below 64^2 / 2^63 per stream -- is
resolved by the lower local position).  Phases: ``randint(0, nkt)`` from their own keyed generator.  The
selected positions of every subset are sorted before packing, so global indices and weights stay paired.
The graph seed is the envelope graph's; the layout seed only selects local subsets; checkpoint / training and
data seeds are separate (callers).

Input sets: the AUTHORITATIVE total Q (``sg_feat.shape[0]`` = R D / S, or the connectivity plan) is passed to
``geometry`` / ``build`` / ``refined_inputs``; m_q / b_q have length Q including every empty set (leading, interior
or trailing), and the records report total / occupied / empty counts separately.  (Up to the random-support study
Q was inferred from the largest assigned q, which dropped trailing empty sets: the reduced H 262,144 record said
Q 4084 for 4096 sets of which 1024 are occupied.)  BP-row's stream offsets are cumulative over sets in q order and
an empty set consumes no permutation, so neither q identities nor any existing RNG stream depend on Q.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch

SCHEMA = "granule.local_supports/2"  # record schema (/2: authoritative Q, WRS / TopK)
RNG_SCHEMA = "granule.local_supports/1"  # key / fingerprint namespace: UNCHANGED, so every /1 support is reproduced
POLICIES = ("FP-mag", "FP-balanced", "RP-record", "RP-row", "BP-row", "WRS", "TopK", "Cover-TopK")
WEIGHTED = ("WRS", "TopK", "Cover-TopK")
COVER_CHUNK_RECORDS = 1 << 15
CHUNK_ROWS = 1 << 16
RNG = ("torch.Generator('cpu') (mt19937) seeded per purpose / chunk with sha256-derived 63-bit seeds; "
       "permutation = stable argsort of int64 keys uniform in [0, 2^62); ties -> lower local position")


def key_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in (RNG_SCHEMA,) + parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF


def permutation_chunks(n: int, base: tuple, width: int = 64):
    """Yields (s0, [m, width] int64): the CHUNK_ROWS chunks of ``permutations(n, base, width)`` one at a time."""
    for c, s0 in enumerate(range(0, n, CHUNK_ROWS)):
        m = min(CHUNK_ROWS, n - s0)
        g = torch.Generator().manual_seed(key_seed(*base, "perm", c))
        keys = torch.randint(0, 1 << 62, (m, width), generator=g, dtype=torch.int64)
        yield s0, torch.argsort(keys, dim=1, stable=True)


def permutations(n: int, base: tuple, width: int = 64) -> torch.Tensor:
    """[n, width] int64: n independent uniform permutations of range(width) from the keyed stream ``base``."""
    out = torch.empty((n, width), dtype=torch.int64)
    for s0, p in permutation_chunks(n, base, width):
        out[s0:s0 + len(p)] = p
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
        h = hashlib.sha256(f"{RNG_SCHEMA}|support|k={self.k}".encode())
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


def geometry(L, buckets: torch.Tensor, device="cpu", *, Q: int) -> dict:
    """Row-group geometry of the envelope: nv [n16], q [n16], valid groups (ascending u), their rank inside the
    record and inside the set, per-set group / record counts m_q / b_q (length Q, the AUTHORITATIVE number of input
    sets incl. empty ones; module docstring)."""
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
    Q = int(Q)
    if len(buckets) and (int(buckets[:, 0].min()) < 0 or int(buckets[:, 0].max()) >= Q):
        raise ValueError(f"a bucket's input set lies outside the Q = {Q} sets")
    m_q = torch.bincount(qv, minlength=Q)
    rec_valid = (nv.view(-1, gpb) > 0).any(1)
    b_q = torch.bincount(bq[rec_valid], minlength=Q)
    assert m_q.numel() == Q and b_q.numel() == Q
    rec_id = torch.cumsum(rec_valid.to(torch.int64), 0) - 1  # canonical record index of every valid block
    return dict(nv=nv, q=q, uv=uv, rank_rec=rank_rec, rank_set=rank_set, blk=blk, rec=rec_id[blk], m_q=m_q,
                b_q=b_q, records=int(rec_valid.sum()), gpb=gpb, Q=Q, sets_occupied=int((m_q > 0).sum()),
                sets_empty=int((m_q == 0).sum()),
                groups_per_record=(nv.view(-1, gpb) > 0).sum(1)[rec_valid])


@torch.no_grad()
def feature_importance(w: torch.Tensor, L, device="cpu", chunk: int = 1 << 16) -> torch.Tensor:
    """[n16, 64] float64 a[u, j] = sum over the VALID neurons c of row group u of W_pre[u, c, j]^2 (module
    docstring: FP64, explicit sequential adds over c = 0..15, padded neurons masked; chunked over groups)."""
    n16 = L.groups * L.blocks_per_group * L.block // 16
    wv = w.reshape(n16, 16, 64)
    valid = L.valid_mask(w.device).reshape(n16, 16)
    out = torch.empty((n16, 64), dtype=torch.float64, device=device)
    for u0 in range(0, n16, chunk):
        x = wv[u0:u0 + chunk].double() * valid[u0:u0 + chunk, :, None]
        acc = x[:, 0] * x[:, 0]
        for c in range(1, 16):
            acc = acc + x[:, c] * x[:, c]
        out[u0:u0 + chunk] = acc.to(device)
    return out


def subset_score(a: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """[n] float64 sum of a[i, env[i, :]] added in ascending coordinate order (env sorted)."""
    v = torch.gather(a, 1, env)
    acc = v[:, 0].clone()
    for i in range(1, v.shape[1]):
        acc = acc + v[:, i]
    return acc


def _local_key(env: torch.Tensor) -> torch.Tensor:
    """[n] int64 exact 64-bit local mask of sorted distinct positions (bit 63 wraps to the sign: still exact)."""
    return (torch.ones_like(env) << env).sum(1)


def bp_stream(m_q: torch.Tensor, rank_set: torch.Tensor, qv: torch.Tensor, k: int, base: tuple):
    """BP-row allocation of groups (set qv, rank rank_set inside their set / stratum) from the keyed permutation
    stream ``base``: (env [n, k] sorted, subset id [n], permutations drawn, sets with a partial last pass)."""
    nkt = 64 // k
    npq = (m_q + nkt - 1) // nkt
    off = torch.cumsum(npq, 0) - npq
    perm = permutations(int(npq.sum()), base)
    pid = off[qv] + rank_set // nkt
    sub = rank_set % nkt
    env = perm.view(-1, nkt, k)[pid, sub].sort(1).values
    return env, sub, int(npq.sum()), int((m_q % nkt != 0).sum())


def wrs_select(a_v: torch.Tensor, env0: torch.Tensor, k: int, L_cand: int, base: tuple) -> dict:
    """Weighted random-subset selection (module docstring) for n groups with importances a_v [n, 64] and the
    candidate-0 subsets env0 [n, k]: streamed per candidate (keyed by its id) in CHUNK_ROWS chunks of the groups.
    Returns env [n, k], chosen candidate id [n], best score, and the duplicate count (candidates whose local mask
    equals a smaller-id candidate of the same group)."""
    n = a_v.shape[0]
    best_env = env0.clone()
    best = subset_score(a_v, env0)
    cand = torch.zeros(n, dtype=torch.int64)
    keys = [_local_key(env0)]
    dup = torch.zeros(n, dtype=torch.int64)
    for c in range(1, L_cand):
        kc = torch.empty(n, dtype=torch.int64)
        for s0, p in permutation_chunks(n, base + (f"cand{c}",)):
            e = p[:, :k].sort(1).values
            sc = subset_score(a_v[s0:s0 + len(p)], e)
            win = sc > best[s0:s0 + len(p)]  # strict: ties keep the smaller candidate id
            sl = slice(s0, s0 + len(p))
            best[sl] = torch.where(win, sc, best[sl])
            best_env[sl] = torch.where(win[:, None], e, best_env[sl])
            cand[sl] = torch.where(win, torch.full_like(cand[sl], c), cand[sl])
            kc[sl] = _local_key(e)
        dup += torch.stack(keys, 1).eq(kc[:, None]).any(1).to(torch.int64)
        keys.append(kc)
    return dict(env=best_env, cand=cand, score=best, duplicates=dup)


def topk_select(a_v: torch.Tensor, k: int) -> torch.Tensor:
    """[n, k] sorted: the k largest importances, ties to the lower local coordinate (stable descending sort)."""
    order = torch.sort(a_v, dim=1, descending=True, stable=True).indices
    return order[:, :k].sort(1).values


def record_groups(rec: torch.Tensor, rank: torch.Tensor, gpb: int) -> torch.Tensor:
    """[records, gpb] int64 index (into the valid groups) of each record's row groups by rank, -1 where absent."""
    nrec = int(rec.max()) + 1 if rec.numel() else 0
    gidx = torch.full((nrec, gpb), -1, dtype=torch.int64, device=rec.device)
    gidx[rec, rank] = torch.arange(rec.numel(), device=rec.device)
    return gidx


@torch.no_grad()
def cover_repair(a_v: torch.Tensor, env0: torch.Tensor, rec: torch.Tensor, rank: torch.Tensor, k: int, gpb: int,
                 S: int = 64, chunk_records: int = COVER_CHUNK_RECORDS) -> dict:
    """Fixed-fan-in coverage repair of the supports env0 [n, k] (sorted) of n valid groups, group i in record rec[i] at
    rank rank[i] (module docstring, ``Cover-TopK``).  Each record is independent; records are processed in chunks of
    ``chunk_records`` (dense [chunk, gpb, S] buffers) and all records of a chunk advance one swap per round, so neither
    the chunking nor the record order can change a result.  For a group u its cheapest removable donor
    d*(u) = argmin {a[u, d] : d retained by u, c[d] >= 2} (first minimum = lowest d) does not depend on the target j, so
    the best swap is min over (j, u) of a[u, d*(u)] - a[u, j], flattened in (j, u) order (first minimum = lexicographic
    tie-break).  Termination / feasibility: an uncovered j with m k >= S implies a covered coordinate of count >= 2 (the
    m k selections fill at most S - 1 coordinates), i.e. a legal donor, and j is absent from every group; each swap
    covers exactly one more input and uncovers none, so the swap count equals the initial number of missing inputs
    (both checked).  Returns env [n, k] sorted, per-record initial missing / swaps / energy loss (FP64), donor-position
    histogram, rounds, working-buffer bytes."""
    n = env0.shape[0]
    dev = a_v.device
    gidx = record_groups(rec.to(dev), rank.to(dev), gpb)
    nrec = gidx.shape[0]
    m = (gidx >= 0).sum(1)
    if nrec and int((m * k).min()) < S:
        bad = int(torch.nonzero(m * k < S)[0])
        raise ValueError(f"Cover-TopK infeasible: record {bad} has {int(m[bad])} valid groups and k = {k} "
                         f"(m k = {int(m[bad]) * k} < {S}); no epsilon edges / padding / larger k are used")
    env = env0.clone().to(dev)
    miss0 = torch.zeros(nrec, dtype=torch.int64, device=dev)
    swaps = torch.zeros(nrec, dtype=torch.int64, device=dev)
    loss = torch.zeros(nrec, dtype=torch.float64, device=dev)
    donor_pos = torch.zeros(S, dtype=torch.int64, device=dev)
    target_pos = torch.zeros(S, dtype=torch.int64, device=dev)
    rounds = 0
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    coord = torch.arange(S, device=dev)
    for r0 in range(0, nrec, chunk_records):
        g = gidx[r0:r0 + chunk_records]
        valid = g >= 0
        gg = g.clamp_min(0)
        A = torch.where(valid[:, :, None], a_v[gg].to(torch.float64), torch.zeros((), dtype=torch.float64, device=dev))
        X = torch.zeros(A.shape, dtype=torch.bool, device=dev)
        X.scatter_(2, env[gg], True)
        X &= valid[:, :, None]
        c = X.sum(1)
        miss0[r0:r0 + len(g)] = (c == 0).sum(1)
        act = torch.nonzero((c == 0).any(1)).squeeze(1)
        rr = 0
        while act.numel():
            Xa, Aa, ca = X[act], A[act], c[act]
            legal = Xa & (ca >= 2)[:, None, :]
            dmin, dstar = torch.where(legal, Aa, inf).min(2)  # [na, gpb]: cheapest removable donor, lowest d on ties
            ok = (ca == 0)[:, None, :] & torch.isfinite(dmin)[:, :, None] & valid[act][:, :, None]
            cost = torch.where(ok, dmin[:, :, None] - Aa, inf)
            best, idx = cost.permute(0, 2, 1).reshape(len(act), -1).min(1)  # (j, u) order: lexicographic tie-break
            if not bool(torch.isfinite(best).all()):
                raise AssertionError("Cover-TopK: an uncovered input without a legal donor swap (feasibility violated)")
            j, u = idx // gpb, idx % gpb
            d = dstar[torch.arange(len(act), device=dev), u]
            X[act, u, d] = False
            X[act, u, j] = True
            c[act, d] -= 1
            c[act, j] += 1
            sl = act + r0
            loss[sl] += best
            swaps[sl] += 1
            donor_pos += torch.bincount(d, minlength=S)
            target_pos += torch.bincount(j, minlength=S)
            act = act[(c[act] == 0).any(1)]
            rr += 1
        rounds = max(rounds, rr)
        if not bool(((X.sum(2) == k) | ~valid).all()) or not bool((X.sum(1) >= 1).all()):
            raise AssertionError("Cover-TopK: repaired record violates degree k / coverage >= 1")
        key = torch.where(X, coord, coord + S)
        env[gg[valid]] = key.sort(2).values[:, :, :k][valid]
    if not torch.equal(swaps, miss0):
        raise AssertionError("Cover-TopK: swap count differs from the initial number of missing inputs")
    work = min(chunk_records, max(nrec, 1)) * gpb * S * (8 + 8 + 1 + 1)  # A, cost, X, legal (per chunk, active <= all)
    return dict(env=env, missing0=miss0, swaps=swaps, loss=loss, donor_position_hist=donor_pos,
                target_position_hist=target_pos, rounds=rounds, groups_per_record=m, chunk_records=chunk_records,
                working_buffer_bytes=work)


def _dist(v: torch.Tensor) -> dict:
    v = v.double()
    if v.numel() == 0:
        return {}
    qs = torch.quantile(v[:min(v.numel(), 1 << 24)], torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return dict(mean=float(v.mean()), min=float(v.min()), p1=float(qs[0]), median=float(qs[1]), p99=float(qs[2]),
                max=float(v.max()), n=int(v.numel()))


def energy_stats(a_v: torch.Tensor, env: torch.Tensor) -> dict:
    """Retained pre-pruning weight energy of the selected subsets: per-group fraction and layer total."""
    tot = a_v.sum(1)
    kept = subset_score(a_v, env)
    frac = torch.where(tot > 0, kept / tot.clamp_min(1e-300), torch.zeros_like(tot))
    return dict(layer_fraction=float(kept.sum() / tot.sum()) if float(tot.sum()) > 0 else 0.0,
                per_group_fraction=_dist(frac), zero_energy_groups=int((tot == 0).sum()))


@torch.no_grad()
def build(policy: str, k: int, L, buckets: torch.Tensor, w=None, graph_seed: int = 0, layout_seed: int = 0,
          device="cpu", *, Q: int, L_cand: int = 1) -> LocalSupport:
    """The local support of ``policy`` with fan-in k (module docstring); ``w`` (pre-pruning weights) for FP-mag,
    WRS and TopK; ``Q`` the authoritative number of input sets; ``L_cand`` the WRS candidate count."""
    if policy not in POLICIES:
        raise ValueError(f"local support policy {policy!r} (one of {POLICIES})")
    if 64 % k or k not in (8, 16):
        raise ValueError("local supports: k in (8, 16)")
    if policy == "WRS" and L_cand < 1:
        raise ValueError("WRS: L >= 1 candidates")
    nkt = 64 // k
    geo = geometry(L, buckets.cpu(), "cpu", Q=Q)
    nv, q, uv = geo["nv"], geo["q"], geo["uv"]
    n16 = nv.numel()
    base = (policy, k, f"graph{graph_seed}", f"layout{layout_seed}")
    env = torch.arange(k, dtype=torch.int64).repeat(n16, 1)
    subset = torch.full((n16,), -1, dtype=torch.int64)
    ar = torch.arange(k, dtype=torch.int64)
    if policy in WEIGHTED or policy == "FP-mag":
        if w is None:
            raise ValueError(f"{policy} needs the pre-pruning weights")
    if policy == "FP-mag":
        from . import width_masks as WM

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
    elif policy == "BP-row":
        e, s, npm, part = bp_stream(geo["m_q"], geo["rank_set"], q[uv], k, base)
        subset[uv] = s
        env[uv] = e
        extra = dict(permutations=npm, partial_last_permutation_sets=part)
    else:  # WRS / TopK: weight-aware selection from the pre-pruning row-group energies
        a_v = feature_importance(w, L)[uv]
        bp_base = ("BP-row", k, f"graph{graph_seed}", f"layout{layout_seed}")
        e0, s0, npm, part = bp_stream(geo["m_q"], geo["rank_set"], q[uv], k, bp_base)
        if policy == "WRS":
            r = wrs_select(a_v, e0, k, L_cand, base + (f"L-stream",))
            env[uv], subset[uv] = r["env"], r["cand"]
            ch = torch.bincount(r["cand"], minlength=L_cand)
            extra = dict(L=L_cand, candidate0="BP-row (same k / graph / layout seed)", bp_permutations=npm,
                         chosen_candidate_hist=ch.tolist(), chosen_candidate0=int(ch[0]),
                         duplicate_candidates=dict(total=int(r["duplicates"].sum()),
                                                   groups_with_any=int((r["duplicates"] > 0).sum())),
                         candidate_keys=f"key_seed({policy}, {k}, graph{graph_seed}, layout{layout_seed}, "
                                        "L-stream, cand<c>, perm, <chunk>); candidate c independent of L",
                         energy_candidate0=energy_stats(a_v, e0))
        elif policy == "TopK":
            env[uv] = topk_select(a_v, k)
            subset[uv] = 0
            extra = dict(tie_break="stable descending sort: equal importances -> lower local coordinate",
                         energy_bp_row=energy_stats(a_v, e0))
        else:  # Cover-TopK
            import resource
            import time

            t0 = time.perf_counter()
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            et = topk_select(a_v, k)
            t1 = time.perf_counter()
            r = cover_repair(a_v, et, geo["rec"], geo["rank_rec"], k, geo["gpb"])
            t2 = time.perf_counter()
            env[uv] = r["env"]
            subset[uv] = 0
            full = r["groups_per_record"] == geo["gpb"]
            el = r["loss"]
            tot_topk = subset_score(a_v, et)
            rec_topk = torch.zeros(len(el), dtype=torch.float64).index_add_(0, geo["rec"], tot_topk)
            rel = torch.where(rec_topk > 0, el / rec_topk.clamp_min(1e-300), torch.zeros_like(el))
            extra = dict(tie_break="TopK: stable descending sort; repair: lexicographic (j, u, d)",
                         energy_topk=energy_stats(a_v, et), energy_bp_row=energy_stats(a_v, e0),
                         repair=dict(records=len(el), records_full=int(full.sum()), records_tail=int((~full).sum()),
                                     tail_groups=sorted(set(r["groups_per_record"][~full].tolist())),
                                     records_repaired=int((r["swaps"] > 0).sum()), swaps=int(r["swaps"].sum()),
                                     swaps_equal_initial_missing=bool(torch.equal(r["swaps"], r["missing0"])),
                                     missing0_per_record=_dist(r["missing0"]),
                                     missing0_full=_dist(r["missing0"][full]),
                                     missing0_tail=_dist(r["missing0"][~full]) if bool((~full).any()) else {},
                                     energy_loss_total=float(el.sum()), energy_loss_relative_to_topk=float(
                                         el.sum() / tot_topk.sum()) if float(tot_topk.sum()) > 0 else 0.0,
                                     record_relative_loss=_dist(rel), record_relative_loss_tail=_dist(rel[~full])
                                     if bool((~full).any()) else {}, donor_position_hist=r["donor_position_hist"].tolist(),
                                     target_position_hist=r["target_position_hist"].tolist(), rounds=r["rounds"],
                                     chunk_records=r["chunk_records"], working_buffer_bytes=r["working_buffer_bytes"],
                                     seconds_topk=t1 - t0, seconds_repair=t2 - t1,
                                     host_maxrss_growth_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0,
                                     device=str(a_v.device)))
        extra.update(energy=energy_stats(a_v, env[uv]), importance="a[u, j] = sum_valid c W_pre[u, c, j]^2 (FP64, "
                     "sequential over c)", score="sum_{j in A} a[u, j] (FP64, ascending coordinate order)")
        del a_v
    sup = LocalSupport(policy, k, env.to(device), nv.to(device), q.to(device), subset.to(device))
    sup.stats = dict(schema=SCHEMA, rng_schema=RNG_SCHEMA, policy=policy, k=k, nkt=nkt, graph_seed=graph_seed,
                     layout_seed=layout_seed if policy not in ("FP-mag", "TopK", "Cover-TopK") else None, rng=RNG,
                     H=int(nv.sum()), W0=64 * int(nv.sum()), enabled_slots=sup.enabled_slots(),
                     rho_mask=k / 64, valid_groups=len(uv), partial_groups=int(((nv > 0) & (nv < 16)).sum()),
                     records=geo["records"], Q=geo["Q"], Q_total=geo["Q"], sets_occupied=geo["sets_occupied"],
                     sets_empty=geo["sets_empty"],
                     groups_per_record=dict(min=int(geo["groups_per_record"].min()),
                                            max=int(geo["groups_per_record"].max())),
                     m_q=dict(min=int(geo["m_q"].min()), max=int(geo["m_q"].max()),
                              mean=float(geo["m_q"].double().mean()),
                              mean_occupied=float(geo["m_q"][geo["m_q"] > 0].double().mean())),
                     b_q=dict(min=int(geo["b_q"].min()), max=int(geo["b_q"].max()),
                              mean=float(geo["b_q"].double().mean())),
                     support_fingerprint=sup.fingerprint(), **extra)
    sup.stats.update(balance(sup, geo))
    return sup


@torch.no_grad()
def record_cover(sup: "LocalSupport", L, buckets, w=None, *, Q: int) -> dict:
    """Per original record: the number of its S local inputs retained by at least one valid row group (MASK
    coverage), and with w the number that also carry a nonzero weight in at least one valid neuron of a group
    retaining it (NUMERICAL coverage: an enabled zero weight creates no input dependence).  [records] int64 each."""
    geo = geometry(L, buckets.cpu(), "cpu", Q=Q)
    uv, rec = geo["uv"], geo["rec"]
    gm = sup.group_mask().cpu()[uv]
    S = gm.shape[1]
    mask = torch.zeros((geo["records"], S), dtype=torch.int64).index_add_(0, rec, gm.to(torch.int64)) > 0
    out = dict(mask=mask.sum(1), groups=geo["groups_per_record"])
    if w is not None:
        n16 = sup.nv.numel()
        valid = L.valid_mask(w.device).reshape(n16, 16)
        nz = ((w.reshape(n16, 16, S) != 0) & valid[:, :, None]).any(1).cpu()[uv] & gm
        num = torch.zeros((geo["records"], S), dtype=torch.int64).index_add_(0, rec, nz.to(torch.int64)) > 0
        out["numerical"] = num.sum(1)
    return out


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


def refined_inputs(sup: "LocalSupport", L, buckets, *, Q: int) -> dict:
    """Actual m_q, b_q (length Q, empty sets included), groups per record and their sets, and the tiles of the fixed
    partition used per set."""
    geo = geometry(L, buckets.cpu(), "cpu", Q=Q)
    nkt = 64 // sup.k
    gpb = geo["gpb"]
    rec_valid = (geo["nv"].view(-1, gpb) > 0).any(1)
    rec_set = block_sets(L, buckets.cpu(), "cpu")[rec_valid]
    used = torch.zeros((geo["Q"], nkt), dtype=torch.bool)
    if sup.policy in ("FP-balanced", "FP-mag"):
        used[geo["q"][geo["uv"]], sup.subset.cpu()[geo["uv"]]] = True
    return dict(m_q=geo["m_q"], b_q=geo["b_q"], rec_groups=geo["groups_per_record"], rec_set=rec_set,
                used_tiles=used.sum(1), Q=geo["Q"], sets_occupied=geo["sets_occupied"], sets_empty=geo["sets_empty"])
