"""EXPERIMENTAL generalized input-supergroup generator (fan-in K, assignment quantum) and work lists.

The process is the documented one of ``supergroups.py`` with two parameters made explicit:

1. Global partitions: ``supergroups.make_feature_sets`` (unchanged): R seeded permutations of the D
   inputs, each cut into D / S sets of S features; Q = R * D / S supergroups.
2. Balanced assignment: output g's ``quantum``-block assignment units are spread over the Q
   supergroups with quotas differing by at most one (round-interleaved order, extras rotating over the
   outputs, partial tail unit greedily on the least-loaded position): ``supergroups.unit_quotas`` with
   the unit size as a parameter.  The production generator uses the fragment-packed work unit
   (quantum = 4 blocks at C = 8, 2 at C = 16, 1 at C = 32 / 64); the generalized layers use
   quantum = 1 (one logical record), so the assignment balances individual logical records and no
   unit holds padded records.
3. Local records: for each supergroup, its blocks (in (g, block) order) consume a stream of
   independent permutations of its S features, each cut into S / K disjoint K-records (production:
   S / 8 records of 8).  Records are stored sorted (global ids; feat rows are sorted, so local and
   global order agree).  C neurons share each record with independent weights / bias.

``generate`` with K = 8 and the production layout's (NB, quantum) reproduces
``supergroups.generate_indices`` bit for bit (tests/test_kgen.py): the generalization is a
parameterization, not a different generator.  Changes for K > 8: K-records (S / K per pass) and
quantum = 1.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..supergroups import ConnectivityPlan, _gen, chunk_table, interleaved_order, make_feature_sets

GENERATOR = "kgen_balanced_permutation"
GENERATOR_VERSION = 1


def valid_units(counts: torch.Tensor, C: int, quantum: int) -> torch.Tensor:
    un = quantum * C
    return (counts.to(torch.int64) + un - 1) // un


def unit_quotas(nvu: torch.Tensor, Q: int, R: int):
    """supergroups.unit_quotas with the valid-unit counts [G] given directly (identical algorithm)."""
    nvu = nvu.to("cpu", torch.int64)
    G = nvu.numel()
    base = nvu // Q
    rem = nvu % Q
    ptr = (torch.cumsum(rem, 0) - rem) % Q
    pos = torch.arange(Q)
    key = (pos[None, :] - ptr[:, None]) % Q
    counts_pos = base[:, None] + (key < rem[:, None]).to(torch.int64)
    tail_load = torch.zeros(Q, dtype=torch.int64)
    big = torch.iinfo(torch.int64).max // 4
    for g in range(G):
        cand = torch.where(counts_pos[g] > 0, tail_load * (Q + 1) + key[g], torch.full_like(key[g], big))
        p = int(torch.argmin(cand))
        tail_load[p] += 1
        key[g, p] = Q
    order = interleaved_order(Q, R)
    counts = torch.zeros((G, Q), dtype=torch.int64)
    keys = torch.zeros((G, Q), dtype=torch.int64)
    counts[:, order] = counts_pos
    keys[:, order] = key
    return counts, keys


def bucket_table(D: int, nvu: torch.Tensor, plan: ConnectivityPlan):
    """(feat [Q, S] int16, buckets [n, 4] int32 = (q, g, first unit, units)) on the CPU."""
    Q = plan.num_supergroups(D)
    feat = make_feature_sets(D, plan)
    counts, keys = unit_quotas(nvu, Q, plan.partition_rounds)
    order_in_g = torch.argsort(keys, dim=1)
    cnt_sorted = torch.gather(counts, 1, order_in_g)
    start_sorted = torch.cumsum(cnt_sorted, 1) - cnt_sorted
    start = torch.empty_like(start_sorted)
    start.scatter_(1, order_in_g, start_sorted)
    gq = torch.nonzero(counts.t() > 0)
    q_b, g_b = gq[:, 0], gq[:, 1]
    buckets = torch.stack([q_b, g_b, start[g_b, q_b], counts[g_b, q_b]], 1).to(torch.int32)
    return feat, buckets


@torch.no_grad()
def generate(D: int, counts: torch.Tensor, C: int, K: int, NB: int, quantum: int, plan: ConnectivityPlan, device,
             chunk_elems: int = 1 << 26):
    """(indices [G, NB, K] int16 global ids, feat [Q, S] int16, unit buckets [n, 4] int32, stats).

    ``counts`` [G] valid neurons per output; blocks of C neurons; assignment units of ``quantum``
    blocks.  Every block of an assigned unit (also a padded block inside a partially valid unit, as in
    the production generator) consumes one record of its supergroup's local permutation stream; blocks
    beyond the valid units copy the output's first record (never computed)."""
    S, R = plan.input_supergroup_dim, plan.partition_rounds
    if S % K:
        raise ValueError(f"S = {S} must be a multiple of the fan-in K = {K}")
    t_start = time.perf_counter()
    dev = torch.device(device)
    G = counts.numel()
    UB = quantum
    Q = plan.num_supergroups(D)
    nvu = valid_units(counts, C, UB)
    if int(nvu.max()) * UB > NB:
        raise ValueError("NB too small for the valid units")
    feat, buckets = bucket_table(D, nvu, plan)
    q_b, g_b = buckets[:, 0].to(torch.int64), buckets[:, 1].to(torch.int64)
    idx = torch.empty((G, NB, K), dtype=torch.int16, device=dev)
    nblk_b = buckets[:, 3].to(torch.int64) * UB
    nblk_q = torch.zeros(Q, dtype=torch.int64).index_add_(0, q_b, nblk_b)
    rpp = S // K
    feat_d = feat.to(dev).to(torch.int64)
    bstart = g_b * NB + buckets[:, 2].to(torch.int64) * UB
    rec_bucket_off = torch.cumsum(nblk_b, 0) - nblk_b
    q_rec_off = torch.cumsum(nblk_q, 0) - nblk_q
    flat_idx = idx.view(-1, K)
    q0 = 0
    while q0 < Q:
        q1, elems = q0, 0
        while q1 < Q and (q1 == q0 or elems + math.ceil(int(nblk_q[q1]) / rpp) * S <= chunk_elems):
            elems += math.ceil(int(nblk_q[q1]) / rpp) * S
            q1 += 1
        for q in range(q0, q1):
            n = int(nblk_q[q])
            if n == 0:
                continue
            g_ = _gen(plan.seed, 2, q, device=dev)
            if plan.generation == "balanced_permutation":
                nperm = math.ceil(n / rpp)
                loc = torch.argsort(torch.rand((nperm, S), generator=g_, device=dev), dim=1)
                loc = loc.reshape(nperm * rpp, K)[:n]
            else:
                from ..layout import _sample_distinct

                loc = _sample_distinct(n, S, K, g_, dev)
            loc = loc.sort(dim=1).values
            glob = feat_d[q][loc]
            bs, be = int(torch.searchsorted(q_b, q)), int(torch.searchsorted(q_b, q, right=True))
            nb = nblk_b[bs:be].to(dev)
            within = torch.arange(n, device=dev) - torch.repeat_interleave((rec_bucket_off[bs:be] - q_rec_off[q]).to(dev), nb)
            dest = torch.repeat_interleave(bstart[bs:be].to(dev), nb) + within
            flat_idx[dest] = glob.to(torch.int16)
        q0 = q1
    blk = torch.arange(NB, device=dev)
    pad = blk[None, :] >= (nvu.to(dev) * UB)[:, None]
    idx[pad] = idx[:, :1, :].expand(G, NB, K)[pad]
    stats = {"generation_seconds": time.perf_counter() - t_start}
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        stats["generation_seconds"] = time.perf_counter() - t_start
    return idx, feat.to(dev), buckets.to(dev), stats


def unit_to_block_buckets(buckets: torch.Tensor, counts: torch.Tensor, C: int, quantum: int) -> torch.Tensor:
    """Unit buckets (q, g, first unit, units) -> logical-block buckets (q, g, first block, blocks), each
    trimmed to the blocks holding valid neurons (a partially valid unit's padded blocks are not records)."""
    b = buckets.to(torch.int64)
    nvb = ((counts.to(b.device).to(torch.int64) + C - 1) // C)
    first = b[:, 2] * quantum
    end = torch.minimum(first + b[:, 3] * quantum, nvb[b[:, 1]])
    out = torch.stack([b[:, 0], b[:, 1], first, end - first], 1)
    return out[out[:, 3] > 0].to(torch.int32).contiguous()


# ------------------------------------------------------------------------- work lists
class KgExec:
    """Work lists of a kgen layer for a compiled tiling cfg (sp_kg_config): items / chunks (fwd, dX),
    tasks / chunks (dW).  Built once per layer and device (host work outside the timed steps)."""

    def __init__(self, NB: int, C: int, K: int, counts: torch.Tensor, feat: torch.Tensor, blk_buckets: torch.Tensor,
                 cfg, chunk_weights: int = 1 << 20, max_item_blocks: int = 64):
        t0 = time.perf_counter()
        dev = feat.device
        self.feat = feat.contiguous()
        self.Q, self.S = feat.shape
        self.cfg = tuple(cfg)
        b = blk_buckets.to("cpu", torch.int64)
        q, g, j0, nb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        # split long buckets into items of <= max_item_blocks blocks (bounded per-warp work)
        pieces = (nb + max_item_blocks - 1) // max_item_blocks
        rep = torch.repeat_interleave(torch.arange(len(b)), pieces)
        k = torch.arange(len(rep)) - torch.repeat_interleave(torch.cumsum(pieces, 0) - pieces, pieces)
        iq, ig = q[rep], g[rep]
        ij = j0[rep] + k * max_item_blocks
        inb = torch.clamp(nb[rep] - k * max_item_blocks, max=max_item_blocks)
        self.items = torch.stack([ig, ig * NB + ij, inb], 1).to(torch.int32)
        target_blocks = max(1, chunk_weights // (C * K))
        self.chunks = chunk_table(iq, inb, self.Q, target_blocks)
        # dW tasks: (flat block, 16-neuron group) of every valid neuron group, supergroup order
        rb = torch.repeat_interleave(torch.arange(len(self.items)), inb)
        kb = torch.arange(len(rb)) - torch.repeat_interleave(torch.cumsum(inb, 0) - inb, inb)
        fb = self.items[rb, 1].to(torch.int64) + kb
        bq = iq[rb]
        jj = fb - ig[rb] * NB
        nval = torch.clamp(counts.to("cpu", torch.int64)[ig[rb]] - jj * C, max=C)
        ncg = (nval + 15) // 16
        rt = torch.repeat_interleave(torch.arange(len(fb)), ncg)
        kt = torch.arange(len(rt)) - torch.repeat_interleave(torch.cumsum(ncg, 0) - ncg, ncg)
        self.tasks = torch.stack([fb[rt], kt], 1).to(torch.int32)
        tq = bq[rt]
        per_cta = cfg[4] * cfg[5]  # dW tasks per warp x warps
        q_off = torch.searchsorted(tq.contiguous(), torch.arange(self.Q + 1))
        n = q_off[1:] - q_off[:-1]
        nch = (n + per_cta - 1) // per_cta
        qs = torch.repeat_interleave(torch.arange(self.Q), nch)
        kk = torch.arange(len(qs)) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
        lo = q_off[qs] + kk * per_cta
        hi = torch.minimum(lo + per_cta, q_off[qs + 1])
        self.dw_chunks = torch.stack([qs, lo, hi], 1).to(torch.int32)
        self.items, self.chunks = self.items.to(dev).contiguous(), self.chunks.to(dev).contiguous()
        self.tasks, self.dw_chunks = self.tasks.to(dev).contiguous(), self.dw_chunks.to(dev).contiguous()
        self.stats = dict(items=len(self.items), chunks=len(self.chunks), tasks=len(self.tasks),
                          dw_chunks=len(self.dw_chunks), target_blocks_per_chunk=target_blocks,
                          max_item_blocks=max_item_blocks, build_seconds=time.perf_counter() - t0)

    def runtime_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.items, self.chunks, self.tasks, self.dw_chunks))


# ------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(D: int, counts: torch.Tensor, C: int, K: int, idx: torch.Tensor, feat: torch.Tensor,
             blk_buckets: torch.Tensor, R: int) -> Dict:
    """Structural checks of a logical graph (raise on violation): every valid logical block is in exactly
    one bucket of its output; its record has K distinct in-range indices, all in its supergroup's set;
    every feature is in exactly R sets; the valid neurons sum to H; bucket statistics."""
    t0 = time.perf_counter()
    dev = idx.device
    G, NB, Kx = idx.shape
    assert Kx == K
    counts = counts.to(dev).to(torch.int64)
    nvb = (counts + C - 1) // C
    Q, S = feat.shape
    rep: Dict = {}
    b = blk_buckets.to(dev).to(torch.int64)
    if bool((b[:, 3] <= 0).any()):
        raise AssertionError("empty bucket record")
    key = b[:, 0] * G + b[:, 1]
    if torch.unique(key).numel() != len(b):
        raise AssertionError("duplicate (supergroup, output) bucket")
    if not bool((key[1:] >= key[:-1]).all()):
        raise AssertionError("buckets are not sorted by (supergroup, output)")
    bq = torch.full((G * NB,), -1, dtype=torch.int64, device=dev)
    nblk = b[:, 3]
    start = b[:, 1] * NB + b[:, 2]
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), nblk)
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(nblk, 0) - nblk, nblk)
    cover = torch.zeros(G * NB, dtype=torch.int64, device=dev).index_add_(0, start[rb] + kb, torch.ones_like(kb))
    if int(cover.max()) > 1:
        raise AssertionError("a logical block is in two buckets")
    bq[start[rb] + kb] = b[rb, 0]
    j = torch.arange(NB, device=dev)
    valid_blk = (j[None, :] < nvb[:, None]).reshape(-1)
    if not bool((cover[valid_blk] == 1).all()) or bool((cover[~valid_blk] != 0).any()):
        raise AssertionError("buckets do not cover exactly the valid logical blocks")
    rv = idx.view(-1, K)[valid_blk].to(torch.int64)
    srt = rv.sort(dim=1).values
    if not bool((srt == rv).all()):
        raise AssertionError("a record is not sorted")
    if bool((srt[:, 1:] == srt[:, :-1]).any()):
        raise AssertionError("a record has repeated indices")
    if int(rv.min()) < 0 or int(rv.max()) >= D:
        raise AssertionError("index out of range")
    featd = feat.to(dev).to(torch.int64)
    member = torch.zeros(D, dtype=torch.int64, device=dev).index_add_(0, featd.reshape(-1), torch.ones(Q * S, dtype=torch.int64, device=dev))
    if not bool((member == R).all()):
        raise AssertionError("a feature is not in exactly R sets")
    onehot = torch.zeros((Q, D), dtype=torch.bool, device=dev)
    onehot.scatter_(1, featd, True)
    qv = bq[valid_blk]
    ok = torch.ones(len(qv), dtype=torch.bool, device=dev)
    for s0 in range(0, len(qv), 1 << 22):
        ok[s0:s0 + (1 << 22)] = onehot[qv[s0:s0 + (1 << 22), None], rv[s0:s0 + (1 << 22)]].all(1)
    if not bool(ok.all()):
        raise AssertionError("a record uses a feature outside its supergroup")
    vn = torch.clamp(counts[:, None] - j[None, :] * C, min=0, max=C).reshape(-1)[valid_blk]
    if int(vn.sum()) != int(counts.sum()):
        raise AssertionError("valid neurons in records != H")
    rep.update(H=int(counts.sum()), logical_records=int(valid_blk.sum()), partial_records=int((vn < C).sum()),
               full_records=int((vn == C).sum()), Q=Q, S=S, R=R, K=K, C=C, buckets=len(b),
               duplicate_sets=Q - int(torch.unique(featd.sort(1).values, dim=0).shape[0]))
    per_q = torch.zeros(Q, dtype=torch.int64, device=dev).index_add_(0, b[:, 0], b[:, 3])
    rep["records_per_supergroup"] = dict(min=int(per_q.min()), max=int(per_q.max()), mean=float(per_q.double().mean()))
    rep["validation_seconds"] = time.perf_counter() - t0
    return rep


@torch.no_grad()
def record_table(D: int, H: int, counts: torch.Tensor, C: int, K: int, idx: torch.Tensor, feat: torch.Tensor,
                 blk_buckets: torch.Tensor) -> Dict:
    """The backend-independent logical record table (benchmarks/kcsr_graph.py): valid logical blocks in
    (g, j) order."""
    dev = idx.device
    G, NB, _ = idx.shape
    counts = counts.to(dev).to(torch.int64)
    b = blk_buckets.to(dev).to(torch.int64)
    bq = torch.full((G, NB), -1, dtype=torch.int32, device=dev)
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq.view(-1)[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0].to(torch.int32)
    j = torch.arange(NB, device=dev)
    vn = torch.clamp(counts[:, None] - j[None, :] * C, min=0, max=C)
    valid = vn > 0
    g_ = torch.arange(G, device=dev)[:, None].expand(G, NB)
    return dict(records=idx[valid].to("cpu").contiguous(), rec_q=bq[valid].cpu(), rec_g=g_[valid].to(torch.int32).cpu(),
                rec_j=j[None].expand(G, NB)[valid].to(torch.int32).cpu(), rec_n=vn[valid].to(torch.int32).cpu(),
                feat=feat.to("cpu").contiguous(), meta=dict(D=D, G=G, H=H, K=K, C=C))


@dataclass(frozen=True)
class GraphSpec:
    """Provenance of a generated logical graph (stored with KgenLayer / lowered layers)."""

    K: int
    C: int
    S: int
    R: int
    seed: int
    quantum: int
    generation: str = "balanced_permutation"
    generator: str = GENERATOR
    generator_version: int = GENERATOR_VERSION
    source: str = "generated"  # "generated" | "converted-from-native" (the native graph's records)

    def to_json(self) -> str:
        import dataclasses

        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "GraphSpec":
        return GraphSpec(**json.loads(s))

    def plan(self) -> ConnectivityPlan:
        return ConnectivityPlan(self.S, self.R, self.generation, self.seed)
