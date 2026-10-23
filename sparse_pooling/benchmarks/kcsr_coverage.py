"""Bounded measurement of feature-combination coverage on actual generated graphs (K/C/S/R sweep).

Quantities (RANDOM_KCSR_COVERAGE_PERFORMANCE.md section 4), all layer-wide over unordered input tuples of D:

* measured record coverage: fraction of t-tuples contained in at least one VALID logical record
  (padded neurons never count; a logical record lowered to several physical blocks counts once);
* realized group coverage: fraction of t-tuples inside at least one realized supergroup set (an upper
  bound of the record coverage of the same graph);
* model predictions: the supplied pre-experiment values (manifest, unchanged) and a refined prediction
  from the graph's metadata (actual valid records per local permutation pass, see ``refined_prediction``).

Methods (no occurrence lists, no array over the quadruple universe):

* pairs, triples: EXACT.  Per supergroup q, the distinct records form a 0/1 matrix M [n_q, S] (local
  features); covered local pairs = (M^T M > 0), covered local triples = ((M o M)^T M > 0) (BF16 0/1
  operands, FP32 accumulation: a positive count stays positive).  The covered local tuples are written
  into global bitmaps of C(D, 2) / C(D, 3) bytes (178 MB for D = 1024 triples).  Cost ~ n_q S^3 per q.
* quadruples (and a triple cross-check): SAMPLED.  A fixed seeded set of distinct uniform tuples (the
  same set for every graph).  For a tuple, the candidate supergroups are the realized sets (one per
  round) that contain all its features; a candidate is covered if the AND of the tuple's per-feature
  record bitsets of q is non-zero.  Clopper-Pearson 95 % intervals; the sample is extended in fixed
  prefixes of the same sequence while there are fewer than ``min_hits`` hits.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

import common  # noqa: F401  (puts the implementation under test on sys.path)

SAMPLE_SEED = 20261023
N_BASE = 10_000_000
N_MAX = 50_000_000


# ------------------------------------------------------------------------------- tuple ranks
def comb_table(D: int, t: int, device) -> torch.Tensor:
    """[D + 1, t + 1] int64 binomials C(n, k)."""
    c = torch.zeros((D + 1, t + 1), dtype=torch.int64)
    for n in range(D + 1):
        for k in range(min(n, t) + 1):
            c[n, k] = math.comb(n, k)
    return c.to(device)


def rank(tup: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
    """Combinatorial-number-system rank of sorted tuples [n, t] (a_0 < ... < a_{t-1}): sum C(a_i, i + 1)."""
    t = tup.shape[1]
    r = torch.zeros(tup.shape[0], dtype=torch.int64, device=tup.device)
    for i in range(t):
        r += ct[tup[:, i].long(), i + 1]
    return r


def unrank(r: torch.Tensor, D: int, t: int, ct: torch.Tensor) -> torch.Tensor:
    """Inverse of ``rank``: sorted tuples [n, t] of ranks r in [0, C(D, t))."""
    out = torch.empty((r.numel(), t), dtype=torch.int64, device=r.device)
    rem = r.clone()
    for i in range(t - 1, -1, -1):  # largest element: max a with C(a, i + 1) <= rem
        col = ct[:, i + 1].contiguous()
        a = torch.searchsorted(col, rem, right=True) - 1
        out[:, i] = a
        rem -= col[a]
    return out


def sample_tuples(D: int, t: int, n: int, seed: int = SAMPLE_SEED, device="cuda") -> torch.Tensor:
    """n distinct uniform sorted t-tuples of [0, D), a fixed sequence (prefixes of it are uniform samples)."""
    ct = comb_table(D, t, "cpu")
    total = int(ct[D, t])
    if n >= total:
        return unrank(torch.arange(total), D, t, ct).to(device)
    g = torch.Generator().manual_seed(seed * 10 + t)
    draw = int(n * 1.02) + 1000
    r = torch.unique(torch.randint(0, total, (draw,), generator=g, dtype=torch.int64))
    while r.numel() < n:
        r = torch.unique(torch.cat([r, torch.randint(0, total, (draw,), generator=g, dtype=torch.int64)]))
    r = r[torch.randperm(r.numel(), generator=g)][:n]
    return unrank(r.to(device), D, t, ct.to(device))


def clopper_pearson(k: int, n: int, a: float = 0.05):
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# ------------------------------------------------------------------------------- per-supergroup data
class GraphIndex:
    """Distinct records per supergroup as local 0/1 matrices / bitsets, plus feature -> set maps."""

    def __init__(self, tab, device="cuda"):
        m = tab["meta"]
        self.D, self.K = m["D"], m["K"]
        feat = tab["feat"].to(device).long()
        self.feat = feat
        self.Q, self.S = feat.shape
        rec = tab["records"].to(device).long()
        q = tab["rec_q"].to(device).long()
        order = torch.argsort(q, stable=True)
        rec, q = rec[order], q[order]
        self.q_off = torch.searchsorted(q, torch.arange(self.Q + 1, device=device)).cpu()
        # local ids (feat rows are sorted ascending)
        lid = torch.full((self.Q, self.D), -1, dtype=torch.int16, device=device)
        lid.scatter_(1, feat, torch.arange(self.S, device=device, dtype=torch.int16).expand(self.Q, self.S).contiguous())
        self.lid = lid
        loc = lid[q[:, None], rec].long()
        if bool((loc < 0).any()):
            raise AssertionError("a record uses a feature outside its supergroup")
        # distinct records per q: unique (q, local record) rows, grouped by q (lexicographic order)
        self.n_records = rec.shape[0]
        uq = torch.unique(torch.cat([q[:, None], loc], 1), dim=0)
        self.U = uq[:, 1:].contiguous()  # [N_u, K] local ids of the distinct records
        self.Uq = uq[:, 0].contiguous()
        del uq, loc
        self.u_off = torch.searchsorted(self.Uq, torch.arange(self.Q + 1, device=device)).cpu()
        self.n_distinct = (self.u_off[1:] - self.u_off[:-1]).to(torch.int64)
        # rounds: set index of every feature per round (supergroups r * (D / S) + j)
        self.per_round = self.D // self.S
        self.R = self.Q // self.per_round
        sid = torch.empty((self.R, self.D), dtype=torch.int32, device=device)
        for r in range(self.R):
            rows = feat[r * self.per_round:(r + 1) * self.per_round]
            sid[r, rows.reshape(-1)] = torch.arange(self.per_round, device=device, dtype=torch.int32).repeat_interleave(self.S)
        self.sid = sid
        self._bits = None

    def local(self, qq: int) -> torch.Tensor:
        return self.U[int(self.u_off[qq]):int(self.u_off[qq + 1])]

    def membership(self, qq: int) -> torch.Tensor:
        u = self.local(qq)
        M = torch.zeros((u.shape[0], self.S), dtype=torch.bfloat16, device=u.device)
        if u.shape[0]:
            M.scatter_(1, u, 1.0)
        return M

    def bitsets(self) -> torch.Tensor:
        """[Q, S, W] int64: bit r of row (q, f) = distinct record r of q contains local feature f."""
        if self._bits is None:
            W = max(1, int((self.n_distinct.max() + 63) // 64))
            dev = self.feat.device
            bits = torch.zeros((self.Q * self.S, W), dtype=torch.int64, device=dev)
            r = torch.arange(self.U.shape[0], device=dev) - self.u_off.to(dev)[self.Uq]  # record index within q
            word, val = r // 64, torch.ones_like(r) << (r % 64)
            for k in range(self.U.shape[1]):  # distinct bits per (row, word): the sum is the OR
                bits.index_put_((self.Uq * self.S + self.U[:, k], word), val, accumulate=True)
            self._bits = bits.view(self.Q, self.S, W)
        return self._bits


# ------------------------------------------------------------------------------- exact pairs / triples
def local_tuples(S: int, t: int, device) -> torch.Tensor:
    return torch.tensor(list(itertools.combinations(range(S), t)), dtype=torch.int64, device=device).reshape(-1, t)


@torch.no_grad()
def exact_coverage(gi: GraphIndex, orders=(2, 3), chunk_elems: int = 1 << 27):
    """{t: dict(covered, total, fraction, group_covered, group_fraction)} for t in orders (exact)."""
    dev = gi.feat.device
    out = {}
    for t in orders:
        ct = comb_table(gi.D, t, dev)
        total = int(ct[gi.D, t])
        cov = torch.zeros(total, dtype=torch.bool, device=dev)
        grp = torch.zeros(total, dtype=torch.bool, device=dev)
        lt = local_tuples(gi.S, t, dev)
        for qq in range(gi.Q):
            gr = rank(gi.feat[qq][lt], ct)
            grp[gr] = True
            M = gi.membership(qq)
            n = M.shape[0]
            if n == 0:
                continue
            if t == 2:
                P = (M.t() @ M) > 0
                hit = P[lt[:, 0], lt[:, 1]]
            else:
                rows = max(1, chunk_elems // (gi.S * gi.S))
                T = torch.zeros((gi.S * gi.S, gi.S), dtype=torch.float32, device=dev)
                for s0 in range(0, n, rows):
                    Mc = M[s0:s0 + rows]
                    X = (Mc[:, :, None] * Mc[:, None, :]).reshape(Mc.shape[0], gi.S * gi.S)
                    T += (X.t() @ Mc).float()
                hit = T[lt[:, 0] * gi.S + lt[:, 1], lt[:, 2]] > 0
            cov[gr[hit]] = True
        c, g = int(cov.sum()), int(grp.sum())
        out[t] = dict(method="exact", covered=c, total=total, fraction=c / total, group_covered=g,
                      group_fraction=g / total)
        del cov, grp
    return out


# ------------------------------------------------------------------------------- sampled queries
@torch.no_grad()
def query(gi: GraphIndex, tup: torch.Tensor, chunk: int = 0):
    """(record-covered [n] bool, group-covered [n] bool) for sorted tuples [n, t]."""
    dev = gi.feat.device
    bits = gi.bitsets()
    t = tup.shape[1]
    n = tup.shape[0]
    chunk = chunk or max(1024, (1 << 26) // (gi.R * t))
    rec_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    grp_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    for s0 in range(0, n, chunk):
        tp = tup[s0:s0 + chunk]
        s = gi.sid[:, tp]  # [R, n, t]
        same = (s == s[:, :, :1]).all(-1)  # [R, n]
        r_idx, i_idx = torch.nonzero(same, as_tuple=True)
        if r_idx.numel() == 0:
            continue
        grp_hit[s0 + torch.unique(i_idx)] = True
        q = r_idx * gi.per_round + s[r_idx, i_idx, 0].long()
        lf = gi.lid[q[:, None], tp[i_idx]].long()  # [m, t] local ids
        step = max(1, (1 << 25) // (t * bits.shape[2]))
        for c0 in range(0, q.numel(), step):
            qc, lc = q[c0:c0 + step], lf[c0:c0 + step]
            acc = bits[qc, lc[:, 0]]
            for k in range(1, t):
                acc = acc & bits[qc, lc[:, k]]
            ok = (acc != 0).any(1)
            rec_hit[s0 + i_idx[c0:c0 + step][ok]] = True
    return rec_hit, grp_hit


@torch.no_grad()
def sampled_coverage(gi: GraphIndex, t: int, n_base: int = N_BASE, n_max: int = N_MAX, min_hits: int = 1000,
                     seed: int = SAMPLE_SEED):
    """Sampled record and group coverage with Clopper-Pearson intervals; prefixes of one fixed sequence."""
    seq = sample_tuples(gi.D, t, n_max, seed, gi.feat.device)
    n = n_base
    while True:
        rec, grp = query(gi, seq[:n])
        k, kg = int(rec.sum()), int(grp.sum())
        if k >= min_hits or n >= n_max:
            break
        n = min(n_max, n * 2)
    lo, hi = clopper_pearson(k, n)
    glo, ghi = clopper_pearson(kg, n)
    return dict(method="sampled", order=t, sample_size=n, sample_seed=seed, sample_sequence_max=n_max,
                distinct_tuples=True, hits=k, fraction=k / n, ci95=[lo, hi], group_hits=kg, group_fraction=kg / n,
                group_ci95=[glo, ghi], interval="Clopper-Pearson 95 %",
                upper_bound_if_zero=(hi if k == 0 else None))


# ------------------------------------------------------------------------------- model predictions
def local_pass_table(D: int, counts: torch.Tensor, C: int, K: int, S: int, NB: int, quantum: int, unit_buckets):
    """[Q, P] valid records per local pass of every supergroup's record stream (generator order: the blocks of
    its buckets in (g, block) order; a block is a valid record iff it holds a valid neuron), plus passes [Q]."""
    b = unit_buckets.to("cpu", torch.int64)
    Q = int(b[:, 0].max()) + 1 if len(b) else 0
    nblk = b[:, 3] * quantum
    rb = torch.repeat_interleave(torch.arange(len(b)), nblk)
    kb = torch.arange(len(rb)) - torch.repeat_interleave(torch.cumsum(nblk, 0) - nblk, nblk)
    j = b[rb, 2] * quantum + kb
    valid = (j * C < counts.to("cpu", torch.int64)[b[rb, 1]]).to(torch.int64)
    q = b[rb, 0]
    q_start = torch.zeros(Q + 1, dtype=torch.int64)
    q_start[1:] = torch.cumsum(torch.bincount(q, minlength=Q), 0)
    pos = torch.arange(len(q)) - q_start[q]
    bpp = S // K
    p = pos // bpp
    P = int(p.max()) + 1 if len(p) else 0
    v = torch.zeros((Q, P), dtype=torch.int64).index_put_((q, p), valid, accumulate=True)
    slots = torch.zeros((Q, P), dtype=torch.int64).index_put_((q, p), torch.ones_like(valid), accumulate=True)
    return v, slots


def refined_prediction(D: int, K: int, S: int, R: int, v: torch.Tensor, t: int) -> float:
    """1 - prod_r (1 - p_group hbar_r) with h_t(q) = 1 - prod_l (1 - v_l C(K, t) / C(S, t)) from the actual valid
    records v_l of every local pass (round r = q // (D / S)); evaluated with log1p / expm1."""
    if t > K:
        return 0.0
    pg = math.comb(S - 1, t - 1) / math.comb(D - 1, t - 1)
    a = math.comb(K, t) / math.comb(S, t)
    vv = v.double()
    term = 1.0 - vv * a  # >= 0 (v <= S / K)
    with torch.no_grad():
        logm = torch.where(term > 0, torch.log(torch.clamp(term, min=1e-300)), torch.full_like(term, -math.inf)).sum(1)
    h = -torch.expm1(logm)  # [Q]
    per = D // S
    hbar = h.view(R, per).mean(1)
    lm = torch.log1p(-(pg * hbar)).sum()
    return float(-torch.expm1(lm))


def pass_distribution(v: torch.Tensor, slots: torch.Tensor, bpp: int):
    full = slots == bpp
    holes = (slots - v)[slots > 0]
    return dict(passes_per_supergroup_min=int((slots > 0).sum(1).min()), passes_per_supergroup_max=int((slots > 0).sum(1).max()),
                records_per_pass=bpp, full_passes=int(full.sum()), partial_passes=int(((slots > 0) & ~full).sum()),
                passes_with_invalid_records=int((holes > 0).sum()), invalid_record_slots=int(holes.sum()),
                valid_per_pass_histogram={int(k): int(c) for k, c in zip(*torch.unique(v[slots > 0], return_counts=True))})


# ------------------------------------------------------------------------------- driver
def build_graph(case: dict, backend: str, graph_seed: int):
    """(record table, generator-order metadata) of the graph a timing run of ``backend`` uses (same code path and
    seeds; indices only, no parameters)."""
    import kcsr_graph  # noqa: F401
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.layout import SparseLayout
    from sparse_pooling.supergroups import ConnectivityPlan, generate_indices

    import kcsr_cases as KC

    K, C, S, R, H = case["K"], case["C"], case["S"], case["R"], case["H"]
    plan = ConnectivityPlan(S, R, seed=graph_seed)
    t0 = time.perf_counter()
    if backend in ("native", "kgen_native"):  # production generator (K = 8, C <= 64)
        L = SparseLayout(KC.D, H, KC.G, 8, C, True, packed=False)
        idx, graph, _ = generate_indices(L, plan, "cuda")
        counts = L.counts_tensor("cuda")
        blk = KG.unit_to_block_buckets(graph.buckets, counts, C, L.unit_blocks)
        tab = KG.record_table(KC.D, H, counts, C, 8, idx, graph.feat, blk)
        meta = dict(generator="production supergroups.generate_indices", quantum=L.unit_blocks, NB=L.blocks_per_group,
                    unit_buckets=graph.buckets.cpu(), counts=counts.cpu())
    else:  # generalized generator, quantum 1 (lowered logical C, kgen)
        L = KLayout(KC.D, H, KC.G, K, C)
        counts = L.counts_tensor("cuda")
        idx, feat, ubk, _ = KG.generate(KC.D, counts, C, K, L.blocks_per_group, 1, plan, "cuda")
        blk = KG.unit_to_block_buckets(ubk, counts, C, 1)
        tab = KG.record_table(KC.D, H, counts, C, K, idx, feat, blk)
        meta = dict(generator=KG.GENERATOR, generator_version=KG.GENERATOR_VERSION, quantum=1, NB=L.blocks_per_group,
                    unit_buckets=ubk.cpu(), counts=counts.cpu())
    meta["graph_seconds"] = time.perf_counter() - t0
    del idx
    return tab, meta


def bucket_stats_logical(case, meta):
    from types import SimpleNamespace

    from sparse_pooling.supergroups import ConnectivityPlan, bucket_statistics

    import kcsr_cases as KC

    lay = SimpleNamespace(groups=KC.G, block=case["C"], unit_blocks=meta["quantum"], input_dim=KC.D,
                          hidden_dim=case["H"], blocks_per_group=meta["NB"])
    Q = case["R"] * KC.D // case["S"]
    return bucket_statistics(lay, meta["unit_buckets"], Q, case["S"], ConnectivityPlan(case["S"], case["R"]))


def analyze(case: dict, backend: str, graph_seed: int, quad_base: int = N_BASE, quad_max: int = N_MAX):
    import kcsr_graph

    import kcsr_cases as KC

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    res = dict(case=case["id"], backend_graph=backend, graph_seed=graph_seed, K=case["K"], C=case["C"], S=case["S"],
               R=case["R"], H=case["H"], D=KC.D, G=KC.G)
    tab, meta = build_graph(case, backend, graph_seed)
    res["graph_fingerprint"] = kcsr_graph.fingerprint(tab)
    res["graph_summary"] = kcsr_graph.summary(tab)
    res["generator"] = {k: meta[k] for k in ("generator", "quantum", "NB") if k in meta}
    res["graph_seconds"] = meta["graph_seconds"]
    res["bucket_stats"] = bucket_stats_logical(case, meta)
    t0 = time.perf_counter()
    gi = GraphIndex(tab, dev)
    res["distinct_records"] = int(gi.n_distinct.sum())
    res["distinct_records_per_supergroup"] = dict(min=int(gi.n_distinct.min()), max=int(gi.n_distinct.max()))
    res["index_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex = exact_coverage(gi, (2, 3))
    res["pair"], res["triple"] = ex[2], ex[3]
    res["exact_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res["quad"] = sampled_coverage(gi, 4, quad_base, quad_max)
    res["triple_sampled_check"] = sampled_coverage(gi, 3, 1_000_000, 1_000_000, 0)
    res["sampled_seconds"] = time.perf_counter() - t0
    v, slots = local_pass_table(KC.D, meta["counts"], case["C"], case["K"], case["S"], meta["NB"], meta["quantum"],
                                meta["unit_buckets"])
    res["local_passes"] = pass_distribution(v, slots, case["S"] // case["K"])
    res["refined_prediction"] = {name: refined_prediction(KC.D, case["K"], case["S"], case["R"], v, t)
                                 for name, t in (("pair", 2), ("triple", 3), ("quad", 4))}
    res["refined_prediction_provenance"] = ("graph metadata: valid records per local permutation pass of every "
                                            "supergroup's record stream (generator order); independent-round p_group")
    if case["id"].startswith("E"):
        res["supplied_prediction_pct"] = {t: {k: case[f"{t}_{k}_pct"] for k in ("model", "group_expectation",
                                                                                 "iid_reference")}
                                          for t in ("pair", "triple", "quad")}
    res["K_equals_S"] = case["K"] == case["S"]
    res["analysis_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return res


def main():
    import kcsr_cases as KC

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--backend", default="auto", help="graph generator of the timing backend: native | kgen_native "
                    "| lowered | kgen | auto (native for K = 8, C <= 64; else the generalized generator)")
    ap.add_argument("--graph-seed", type=int, default=KC.GRAPH_SEED)
    ap.add_argument("--quad-base", type=int, default=N_BASE)
    ap.add_argument("--quad-max", type=int, default=N_MAX)
    ap.add_argument("--timing-json", default="", help="assert the graph fingerprint of this timing record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = KC.case(a.case)
    be = a.backend
    if be == "auto":
        be = "native" if case["K"] == 8 and case["C"] <= 64 else "kgen"
    t0 = time.perf_counter()
    res = analyze(case, be, a.graph_seed, a.quad_base, a.quad_max)
    res["analysis_seconds"] = time.perf_counter() - t0
    if a.timing_json:
        tj = json.loads(Path(a.timing_json).read_text())
        res["timing_fingerprint_match"] = tj["kcsr"]["graph_fingerprint"] == res["graph_fingerprint"]
        if not res["timing_fingerprint_match"]:
            raise SystemExit(f"graph fingerprint differs from the timing run {a.timing_json}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    q = res["quad"]
    print(f"{a.case} seed {a.graph_seed} {be}: pair {100 * res['pair']['fraction']:.4f}% triple "
          f"{100 * res['triple']['fraction']:.4f}% quad {100 * q['fraction']:.4f}% "
          f"[{100 * q['ci95'][0]:.4f}, {100 * q['ci95'][1]:.4f}] (n={q['sample_size']:,}, hits {q['hits']}) "
          f"fp {res['graph_fingerprint'][:12]} {res['analysis_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
