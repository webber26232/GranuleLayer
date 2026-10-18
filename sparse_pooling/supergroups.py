"""Input-supergroup connectivity: seeded balanced generation, compact metadata, validation.

Logical graph (``ConnectivityPlan``)
------------------------------------
D global inputs are partitioned R times (independent seeded permutations) into D / S
disjoint sets of S features; the Q = R * D / S sets ``feat[Q, S]`` are the *input
supergroups*.  Every hidden neuron draws its K = 8 distinct inputs from one set; in
Mode B the C neurons of a block share that record (independent weights / bias).

Input supergroups and pooling outputs are independent axes.  Output g keeps its
exact valid-neuron count n_g and its semantic neuron ids offset_g + slot (layout.py);
its work units (UB blocks of C neurons: the fragment-packed parameter quantum, 32
neurons at C = 8, 64 at C = 64) are spread over the supergroups:

    bucket (q, g) = a contiguous range of output g's units whose records come from feat[q]

so the physical parameter storage is unchanged: the generic kernels run the exact
same tensors (global int16 indices), and the grouped kernels read the compact bucket
list (q, g, first unit, unit count) sorted by q.  No permutation of neurons, no
[G, Q, ...] padded tensor, no per-neuron metadata.

Assignment.  Output g's valid units are distributed over the Q supergroups with
quotas differing by at most one (base = n // Q everywhere plus one extra on rem
consecutive positions of an interleaved supergroup order), where the window of extras
rotates globally (a running pointer over the outputs), so per-supergroup unit totals
differ by at most one plus the tail effect of partial units.  The interleaved order
(position p -> round p % R, set p // R) makes any window cover all rounds evenly.

Local records ("balanced_permutation").  For each supergroup q, its blocks (all
buckets, in (g, block) order) consume a stream of records cut from successive
independent permutations of its S local features: randperm(S).reshape(S // 8, 8).
Each complete local round uses every feature of the set exactly once ("coordinated
local rounds": small buckets of one supergroup draw different records).  Records are
stored sorted ascending (the existing convention); since feat[q] is sorted, sorted
local and sorted global orders agree, so the grouped kernels' local index sits at the
same fan-in position k as the global one (identical weight pairing and MMA order).

"independent_subsets" draws each record as 8 distinct uniform local features
(diagnostic comparison of the balanced construction).
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .layout import SparseLayout

GENERATIONS = ("balanced_permutation", "independent_subsets")
GRAPH_VERSION = 1


@dataclass(frozen=True)
class ConnectivityPlan:
    """Logical connectivity (changes the model).  ``input_supergroup_dim`` = S,
    ``partition_rounds`` = R; Q = R * D / S supergroups."""

    input_supergroup_dim: int
    partition_rounds: int
    generation: str = "balanced_permutation"
    seed: int = 0
    version: int = GRAPH_VERSION

    def __post_init__(self):
        if self.generation not in GENERATIONS:
            raise ValueError(f"unknown generation {self.generation!r}")
        if self.input_supergroup_dim < 8 or self.input_supergroup_dim % 8:
            raise ValueError("input_supergroup_dim must be a multiple of 8 (fan-in 8)")
        if self.partition_rounds < 1:
            raise ValueError("partition_rounds must be >= 1")

    def num_supergroups(self, input_dim: int) -> int:
        if input_dim % self.input_supergroup_dim:
            raise ValueError(f"input_supergroup_dim {self.input_supergroup_dim} must divide D = {input_dim}")
        return self.partition_rounds * input_dim // self.input_supergroup_dim

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ConnectivityPlan":
        return ConnectivityPlan(**json.loads(s))


@dataclass
class SupergroupGraph:
    """Compact metadata of a generated graph (persistent; saved with the layer).

    feat     [Q, S] int16   sorted global feature ids of each supergroup
    buckets  [NBk, 4] int32 (q, g, first unit within output g, unit count), sorted by (q, g)
    """

    plan: ConnectivityPlan
    feat: torch.Tensor
    buckets: torch.Tensor

    @property
    def Q(self) -> int:
        return self.feat.shape[0]

    @property
    def S(self) -> int:
        return self.feat.shape[1]

    def metadata_bytes(self) -> int:
        return self.feat.numel() * 2 + self.buckets.numel() * 4

    def to(self, device) -> "SupergroupGraph":
        return SupergroupGraph(self.plan, self.feat.to(device), self.buckets.to(device))


def _gen(seed: int, *salt: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    h = seed & 0xFFFFFFFF
    for s in salt:  # deterministic mixing (splitmix-like), independent streams per salt
        h = (h * 0x9E3779B1 + s + 0x7F4A7C15) & 0xFFFFFFFFFFFF
    g.manual_seed(h)
    return g


def interleaved_order(Q: int, R: int) -> torch.Tensor:
    """Position p -> supergroup: round p % R, set p // R (sets of a round are contiguous)."""
    per = Q // R
    p = torch.arange(Q)
    return (p % R) * per + p // R


def valid_units(layout: SparseLayout, device="cpu") -> torch.Tensor:
    """[G] units with at least one valid neuron per output."""
    un = layout.unit_blocks * layout.block
    return (layout.counts_tensor(device).to(torch.int64) + un - 1) // un


def unit_quotas(layout: SparseLayout, Q: int, R: int):
    """counts [G, Q] units of output g in supergroup q, and the bucket order keys."""
    nvu = valid_units(layout)
    G = layout.groups
    base = nvu // Q
    rem = nvu % Q
    ptr = (torch.cumsum(rem, 0) - rem) % Q  # running pointer: extras rotate over outputs
    pos = torch.arange(Q)
    key = (pos[None, :] - ptr[:, None]) % Q  # [G, Q position] order key within output g
    counts_pos = base[:, None] + (key < rem[:, None]).to(torch.int64)
    # the partial tail unit of an output lands in its last bucket: pick that bucket greedily
    # on the least-loaded nonempty position (otherwise tails pile up on Q / rem positions)
    tail_load = torch.zeros(Q, dtype=torch.int64)
    big = torch.iinfo(torch.int64).max // 4
    for g in range(G):
        cand = torch.where(counts_pos[g] > 0, tail_load * (Q + 1) + key[g], torch.full_like(key[g], big))
        p = int(torch.argmin(cand))
        tail_load[p] += 1
        key[g, p] = Q  # last in the output's bucket order
    order = interleaved_order(Q, R)
    counts = torch.zeros((G, Q), dtype=torch.int64)
    keys = torch.zeros((G, Q), dtype=torch.int64)
    counts[:, order] = counts_pos
    keys[:, order] = key
    return counts, keys


def make_feature_sets(D: int, plan: ConnectivityPlan, device="cpu") -> torch.Tensor:
    S, R = plan.input_supergroup_dim, plan.partition_rounds
    if S == D:
        if R != 1:
            raise ValueError("S = D is the unrestricted working-set control: use a single set (R = 1)")
        return torch.arange(D, dtype=torch.int16, device=device)[None]
    sets = []
    for r in range(R):
        perm = torch.randperm(D, generator=_gen(plan.seed, 1, r))
        sets.append(perm.reshape(D // S, S))
    feat = torch.cat(sets).sort(dim=1).values
    return feat.to(torch.int16).to(device)


@torch.no_grad()
def generate_indices(layout: SparseLayout, plan: ConnectivityPlan, device, chunk_elems: int = 1 << 26):
    """Return (canonical Mode-B indices [G, NB, 8] int16 (global ids), SupergroupGraph, stats)."""
    if not layout.shared_indices:
        raise ValueError("input supergroups are defined for Mode B (shared index records)")
    t_start = time.perf_counter()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
    D, G, NB, UB = layout.input_dim, layout.groups, layout.blocks_per_group, layout.unit_blocks
    S, R = plan.input_supergroup_dim, plan.partition_rounds
    Q = plan.num_supergroups(D)
    feat = make_feature_sets(D, plan)  # CPU int16 [Q, S]
    counts, keys = unit_quotas(layout, Q, R)  # [G, Q]
    # bucket unit ranges inside each output, in key order (tail unit lands in the last bucket)
    order_in_g = torch.argsort(keys, dim=1)
    cnt_sorted = torch.gather(counts, 1, order_in_g)
    start_sorted = torch.cumsum(cnt_sorted, 1) - cnt_sorted
    start = torch.empty_like(start_sorted)
    start.scatter_(1, order_in_g, start_sorted)
    gq = torch.nonzero(counts.t() > 0)  # (q, g) sorted by q then g
    q_b, g_b = gq[:, 0], gq[:, 1]
    buckets = torch.stack([q_b, g_b, start[g_b, q_b], counts[g_b, q_b]], 1).to(torch.int32)

    # ---- records: per-supergroup stream of local permutation rounds
    dev = torch.device(device)
    idx = torch.empty((G, NB, 8), dtype=torch.int16, device=dev)
    nblk_b = buckets[:, 3].to(torch.int64) * UB
    nblk_q = torch.zeros(Q, dtype=torch.int64).index_add_(0, q_b, nblk_b)
    rpp = S // 8  # records per local permutation
    feat_d = feat.to(dev).to(torch.int64)
    # destination block (flat g * NB + block) of every stream record, in (q, g, block) order
    bstart = (g_b * NB + buckets[:, 2].to(torch.int64) * UB)
    rec_bucket_off = torch.cumsum(nblk_b, 0) - nblk_b
    q_rec_off = torch.cumsum(nblk_q, 0) - nblk_q
    flat_idx = idx.view(-1, 8)
    # supergroups in batches whose permutation buffers fit chunk_elems
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
                loc = loc.reshape(nperm * rpp, 8)[:n]
            else:
                from .layout import _sample_distinct

                loc = _sample_distinct(n, S, 8, g_, dev)
            loc = loc.sort(dim=1).values
            glob = feat_d[q][loc]  # sorted (feat rows are sorted)
            # destinations of this q's records
            bs, be = int(torch.searchsorted(q_b, q)), int(torch.searchsorted(q_b, q, right=True))
            nb = nblk_b[bs:be].to(dev)
            within = torch.arange(n, device=dev) - torch.repeat_interleave(
                (rec_bucket_off[bs:be] - q_rec_off[q]).to(dev), nb)
            dest = torch.repeat_interleave(bstart[bs:be].to(dev), nb) + within
            flat_idx[dest] = glob.to(torch.int16)
        q0 = q1
    # fully padded blocks (beyond the valid units): copy the output's first record (never computed)
    nvu = valid_units(layout)
    blk = torch.arange(NB, device=dev)
    pad = blk[None, :] >= (nvu.to(dev) * UB)[:, None]
    idx[pad] = idx[:, :1, :].expand(G, NB, 8)[pad]
    graph = SupergroupGraph(plan, feat.to(dev), buckets.to(dev))
    stats = {"generation_seconds": time.perf_counter() - t_start}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        stats["generation_seconds"] = time.perf_counter() - t_start
        stats["generation_peak_extra_bytes"] = torch.cuda.max_memory_allocated() - mem0
    return idx, graph, stats


# ----------------------------------------------------------------------------------------
# physical (kernel) work lists, derived from the compact buckets (rebuilt on load)
# ----------------------------------------------------------------------------------------
def work_items(layout: SparseLayout, graph: SupergroupGraph, max_units: int = 0):
    """Items [n, 3] int32 = (g, first flat unit, unit count) sorted by q, plus per-q offsets.
    Flat unit = g * NB / UB + unit within output (packed storage).  ``max_units`` > 0 splits
    larger buckets (bounded per-warp work)."""
    b = graph.buckets.to(torch.int64)
    NU = layout.blocks_per_group // layout.unit_blocks
    q, g, u0, nu = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if max_units > 0:
        pieces = (nu + max_units - 1) // max_units
        rep = torch.repeat_interleave(torch.arange(len(b), device=b.device), pieces)
        k = torch.arange(len(rep), device=b.device) - torch.repeat_interleave(torch.cumsum(pieces, 0) - pieces, pieces)
        q, g = q[rep], g[rep]
        u0 = u0[rep] + k * max_units
        nu = torch.clamp(nu[rep] - k * max_units, max=max_units)
    items = torch.stack([g, g * NU + u0, nu], 1).to(torch.int32).contiguous()
    q_off = torch.searchsorted(q.contiguous(), torch.arange(graph.Q + 1, device=b.device))
    return items, q, q_off


def chunk_table(q_of: torch.Tensor, weight: torch.Tensor, Q: int, target: int) -> torch.Tensor:
    """Split each supergroup's ordered list (items or units, with per-element weight) into
    chunks of about ``target`` weight.  Returns [n_chunks, 3] int32 (q, lo, hi)."""
    q_of = q_of.to("cpu")
    w = weight.to("cpu").to(torch.int64)
    cw = torch.cumsum(w, 0)
    n = len(q_of)
    q_off = torch.searchsorted(q_of.contiguous(), torch.arange(Q + 1))
    rows = []
    for q in range(Q):
        lo, hi = int(q_off[q]), int(q_off[q + 1])
        if lo == hi:
            continue
        base = int(cw[lo - 1]) if lo > 0 else 0
        tot = int(cw[hi - 1]) - base
        nch = max(1, round(tot / max(target, 1)))
        # cut points at equal weight
        cuts = [lo]
        for c in range(1, nch):
            cut = int(torch.searchsorted(cw[lo:hi], base + tot * c // nch)) + lo
            cut = min(max(cut, cuts[-1] + 1), hi)
            if cut < hi:
                cuts.append(cut)
        cuts.append(hi)
        for a, z in zip(cuts[:-1], cuts[1:]):
            if z > a:
                rows.append((q, a, z))
    del n
    return torch.tensor(rows, dtype=torch.int32).reshape(-1, 3)


def unit_list(layout: SparseLayout, graph: SupergroupGraph):
    """Flat unit ids [n_units] int32 ordered by supergroup (dW ownership list) and their q."""
    items, q, _ = work_items(layout, graph)
    nu = items[:, 2].to(torch.int64)
    rep = torch.repeat_interleave(torch.arange(len(items), device=items.device), nu)
    k = torch.arange(len(rep), device=items.device) - torch.repeat_interleave(torch.cumsum(nu, 0) - nu, nu)
    units = (items[rep, 1].to(torch.int64) + k).to(torch.int32).contiguous()
    return units, q[rep]


# ----------------------------------------------------------------------------------------
# structural validation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def block_supergroup(layout: SparseLayout, graph: SupergroupGraph) -> torch.Tensor:
    """[G, NB] int32 supergroup of every block (-1 for blocks outside any bucket)."""
    G, NB, UB = layout.groups, layout.blocks_per_group, layout.unit_blocks
    dev = graph.buckets.device
    bq = torch.full((G * NB,), -1, dtype=torch.int32, device=dev)
    b = graph.buckets.to(torch.int64)
    nblk = b[:, 3] * UB
    start = b[:, 1] * NB + b[:, 2] * UB
    rep = torch.repeat_interleave(torch.arange(len(b), device=dev), nblk)
    k = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(nblk, 0) - nblk, nblk)
    bq[start[rep] + k] = b[rep, 0].to(torch.int32)
    return bq.view(G, NB)


@torch.no_grad()
def validate(layout: SparseLayout, idx_canonical: torch.Tensor, graph: Optional[SupergroupGraph],
             pair_coverage: bool = True) -> Dict:
    """Structural checks (raise on violation) and balance / coverage statistics.

    ``graph=None`` validates an unrestricted graph (statistics only)."""
    t0 = time.perf_counter()
    dev = idx_canonical.device
    D, G, NB, C, UB = layout.input_dim, layout.groups, layout.blocks_per_group, layout.block, layout.unit_blocks
    rep: Dict = {}
    counts = layout.counts_tensor(dev).to(torch.int64)
    if int(counts.sum()) != layout.hidden_dim:
        raise AssertionError("valid neuron count != H")
    rep["H"] = layout.hidden_dim
    # valid neurons per block
    slot0 = torch.arange(NB, device=dev, dtype=torch.int64) * C
    vb = torch.clamp(counts[:, None] - slot0[None, :], min=0, max=C)  # [G, NB]
    valid_blk = vb > 0
    rec = idx_canonical.view(G, NB, 8).to(torch.int64)
    rv = rec[valid_blk]  # [n_valid_blocks, 8]
    wv = vb[valid_blk]
    srt = rv.sort(dim=1).values
    if bool((srt[:, 1:] == srt[:, :-1]).any()):
        raise AssertionError("a record has repeated indices")
    if int(rv.min()) < 0 or int(rv.max()) >= D:
        raise AssertionError("index out of range")
    rep["valid_blocks"] = int(valid_blk.sum())
    rep["valid_neurons_check"] = int(wv.sum())
    if rep["valid_neurons_check"] != layout.hidden_dim:
        raise AssertionError("valid neurons in blocks != H")
    # per-feature edge degree weighted by valid neurons (C and tail validity)
    deg = torch.zeros(D, dtype=torch.int64, device=dev).index_add_(0, rv.reshape(-1), wv.repeat_interleave(8))
    dm = deg.double()
    rep["feature_degree"] = {"min": int(deg.min()), "max": int(deg.max()), "mean": float(dm.mean()),
                             "cv": float(dm.std() / dm.mean()), "expected": layout.hidden_dim * 8 / D}
    # record repetition / diversity
    key = (srt[:, 0] << 50) | (srt[:, 1] << 40) | (srt[:, 2] << 30) | (srt[:, 3] << 20)
    key2 = (srt[:, 4] << 30) | (srt[:, 5] << 20) | (srt[:, 6] << 10) | srt[:, 7]
    uk = torch.unique(torch.stack([key, key2], 1), dim=0)
    rep["distinct_records"] = int(uk.shape[0])
    rep["records_per_distinct"] = rep["valid_blocks"] / rep["distinct_records"]
    rep["neurons_per_distinct_record"] = layout.hidden_dim / rep["distinct_records"]
    if pair_coverage:  # fraction of unordered input pairs that share at least one record
        cov = torch.zeros(D * D, dtype=torch.bool, device=dev)
        for a in range(8):
            for b2 in range(a + 1, 8):
                cov[srt[:, a] * D + srt[:, b2]] = True
        rep["record_pair_coverage"] = int(cov.sum()) / (D * (D - 1) // 2)
        del cov
    if graph is not None:
        S, Q, R = graph.S, graph.Q, graph.plan.partition_rounds
        feat = graph.feat.to(dev).to(torch.int64)
        fs = feat.sort(dim=1).values
        if bool((fs[:, 1:] == fs[:, :-1]).any()):
            raise AssertionError("a feature set has duplicates")
        member = torch.zeros(D, dtype=torch.int64, device=dev).index_add_(
            0, feat.reshape(-1), torch.ones(feat.numel(), dtype=torch.int64, device=dev))
        if not bool((member == R).all()):
            raise AssertionError("a feature is not in exactly R sets")
        # duplicate sets across rounds
        rep["duplicate_sets"] = Q - int(torch.unique(fs, dim=0).shape[0])
        bq = block_supergroup(layout, graph)
        if bool((bq[valid_blk] < 0).any()):
            raise AssertionError("a valid block belongs to no supergroup")
        # every index lies in its assigned set
        onehot = torch.zeros((Q, D), dtype=torch.bool, device=dev)
        onehot.scatter_(1, feat, True)
        qv = bq[valid_blk].to(torch.int64)
        if not bool(onehot[qv[:, None], rv].all()):
            raise AssertionError("a record uses a feature outside its supergroup")
        # buckets: unit ranges of each output are disjoint and cover exactly its valid units
        b = graph.buckets.to(torch.int64)
        nvu = valid_units(layout, dev)
        tot = torch.zeros(G, dtype=torch.int64, device=dev).index_add_(0, b[:, 1], b[:, 3])
        if not bool((tot == nvu).all()):
            raise AssertionError("buckets do not cover each output's valid units exactly")
        cover = torch.zeros(G * (NB // UB), dtype=torch.int64, device=dev)
        units, _ = unit_list(layout, graph)
        cover.index_add_(0, units.to(torch.int64), torch.ones(len(units), dtype=torch.int64, device=dev))
        if int(cover.max()) > 1:
            raise AssertionError("a unit is assigned to two buckets")
        # balance per supergroup
        unit_q = torch.zeros(Q, dtype=torch.int64, device=dev).index_add_(0, b[:, 0], b[:, 3])
        neur_q = torch.zeros(Q, dtype=torch.int64, device=dev).index_add_(0, qv, wv)
        blk_q = torch.zeros(Q, dtype=torch.int64, device=dev).index_add_(
            0, qv, torch.ones_like(wv))
        rep["per_supergroup"] = {
            "units_min": int(unit_q.min()), "units_max": int(unit_q.max()),
            "valid_blocks_min": int(blk_q.min()), "valid_blocks_max": int(blk_q.max()),
            "valid_neurons_min": int(neur_q.min()), "valid_neurons_max": int(neur_q.max()),
            "mma_work_units_min": int(unit_q.min()) * UB * C, "mma_work_units_max": int(unit_q.max()) * UB * C,
            "neuron_imbalance": float(neur_q.max() / neur_q.double().mean()),
        }
        # per-output coverage of supergroups / rounds
        per = D // S
        out_q = torch.zeros((G, Q), dtype=torch.bool, device=dev)
        out_q[b[:, 1], b[:, 0]] = True
        out_r = torch.zeros((G, R), dtype=torch.int64, device=dev)
        out_r.index_put_((b[:, 1], b[:, 0] // per), b[:, 3], accumulate=True)
        nq = out_q.sum(1)
        rep["per_output"] = {
            "supergroups_min": int(nq.min()), "supergroups_max": int(nq.max()),
            "rounds_min": int((out_r > 0).sum(1).min()), "rounds_max": int((out_r > 0).sum(1).max()),
            "round_units_spread_max": int((out_r.max(1).values - out_r.min(1).values).max()),
        }
        nb_units = b[:, 3]
        rep["buckets"] = {"nonempty": len(b), "possible": G * Q, "units_min": int(nb_units.min()),
                          "units_max": int(nb_units.max()), "units_mean": float(nb_units.double().mean())}
        # set co-membership: pairs covered by at least one set (vs the independent-round formula)
        oh = onehot.to(torch.float32)
        co = oh.t() @ oh  # [D, D] number of sets containing both
        iu = torch.triu_indices(D, D, 1, device=dev)
        cop = co[iu[0], iu[1]]
        rep["set_pair_coverage"] = float((cop > 0).double().mean())
        rep["set_pair_coverage_expected"] = 1 - (1 - (S - 1) / (D - 1)) ** R if S < D else 1.0
        rep["set_pair_comembership_max"] = int(cop.max())
        del co, oh, onehot, cop
        rep["metadata_bytes"] = graph.metadata_bytes()
        rep["S"], rep["Q"], rep["R"] = S, Q, R
        rep["plan"] = dataclasses.asdict(graph.plan)
    rep["index_bytes"] = idx_canonical.numel() * 2
    rep["padding_fraction"] = layout.padding_fraction()
    rep["validation_seconds"] = time.perf_counter() - t0
    return rep


@torch.no_grad()
def matched_c8_from_c64(lay64: SparseLayout, idx64, w64, b64, graph64: Optional[SupergroupGraph]):
    """Represent a C = 64 graph as C = 8 blocks carrying the same records (matched control).

    Every 64-neuron block becomes eight 8-neuron blocks with its index record; weights,
    biases, semantic ids (slots) and output membership are unchanged.  Canonical inputs
    (unpacked); returns canonical (layout8, idx8, w8, b8, graph8)."""
    assert lay64.block == 64 and lay64.unit_blocks == 1
    lay8 = SparseLayout(lay64.input_dim, lay64.hidden_dim, lay64.groups, 8, 8, True, packed=lay64.packed)
    G, NB64, NB8 = lay64.groups, lay64.blocks_per_group, lay8.blocks_per_group
    slots64, slots8 = NB64 * 64, NB8 * 8
    pad = max(0, slots8 - slots64)  # padded (invalid) slots only
    w = torch.nn.functional.pad(w64.reshape(G, slots64, 8), (0, 0, 0, pad))[:, :slots8]
    w = w.reshape(G, NB8, 8, 8).contiguous()
    b = torch.nn.functional.pad(b64.reshape(G, slots64), (0, pad))[:, :slots8].reshape(G, NB8, 8).contiguous()
    idx = idx64.reshape(G, NB64, 1, 8).expand(G, NB64, 8, 8).reshape(G, NB64 * 8, 8)
    if pad:
        idx = torch.cat([idx, idx[:, :1].expand(G, pad // 8, 8)], 1)
    idx = idx[:, :NB8].contiguous()
    g8 = None
    if graph64 is not None:  # a C=64 unit (64 neurons) = two C=8 units (2 x 4 blocks x 8)
        bk = graph64.buckets.clone()
        bk[:, 2] *= 2
        bk[:, 3] *= 2
        # the last bucket of an output may now cover a fully padded C=8 unit: trim to valid units
        nvu8 = valid_units(lay8, bk.device)
        end = bk[:, 2] + bk[:, 3]
        over = torch.clamp(end.to(torch.int64) - nvu8[bk[:, 1].long()], min=0)
        bk[:, 3] -= over.to(torch.int32)
        bk = bk[bk[:, 3] > 0].contiguous()
        g8 = SupergroupGraph(graph64.plan, graph64.feat, bk)
    return lay8, idx, w, b, g8


class SgExec:
    """Per-layer execution metadata of a supergroup graph: work lists for the grouped
    kernels, built once per execution configuration and cached (steady-state training
    reuses them; X-dependent data is never cached)."""

    def __init__(self, layout: SparseLayout, feat: torch.Tensor, buckets: torch.Tensor):
        self.layout = layout
        self.feat = feat.contiguous()
        self.graph = SupergroupGraph(None, self.feat, buckets.contiguous())
        self.S, self.Q = self.feat.shape[1], self.feat.shape[0]
        b = self.graph.buckets
        # host copy once at construction: no device sync in the per-step dispatch
        self._mean_bucket_units = float(b[:, 3].double().mean()) if len(b) else 0.0
        self._cache: Dict = {}

    def item_chunks(self, target_units: int, max_units: int = 0):
        """(items [n, 3], chunks [m, 3]) for sg_fwd / sg_dx: chunks hold whole items of one
        supergroup, about target_units work units each."""
        key = ("items", target_units, max_units)
        if key not in self._cache:
            items, q_of, _ = work_items(self.layout, self.graph, max_units)
            ch = chunk_table(q_of, items[:, 2], self.Q, target_units).to(items.device)
            self._cache[key] = (items, ch)
        return self._cache[key]

    def unit_chunks(self, units_per_cta: int):
        """(units [n], chunks [m, 3]) for sg_dw: <= units_per_cta consecutive q-ordered units
        per CTA (exclusive ownership of each unit's dW/db)."""
        key = ("units", units_per_cta)
        if key not in self._cache:
            units, q_of = unit_list(self.layout, self.graph)
            qc = q_of.to("cpu")
            q_off = torch.searchsorted(qc.contiguous(), torch.arange(self.Q + 1))
            n = (q_off[1:] - q_off[:-1])
            nch = (n + units_per_cta - 1) // units_per_cta
            qs = torch.repeat_interleave(torch.arange(self.Q), nch)
            k = torch.arange(len(qs)) - torch.repeat_interleave(torch.cumsum(nch, 0) - nch, nch)
            lo = q_off[qs] + k * units_per_cta
            hi = torch.minimum(lo + units_per_cta, q_off[qs + 1])
            ch = torch.stack([qs, lo, hi], 1).to(torch.int32).to(units.device).contiguous()
            self._cache[key] = (units, ch)
        return self._cache[key]

    def mean_bucket_units(self) -> float:
        return self._mean_bucket_units

    def runtime_bytes(self) -> int:
        """Persistent bytes of the cached work lists (excluding feat / buckets)."""
        n = 0
        for v in self._cache.values():
            for t in v:
                n += t.numel() * t.element_size()
        return n
