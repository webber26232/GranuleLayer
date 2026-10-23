"""Coverage of the K64 study on actual graphs and masks: potential supports, retained supports, every output.

Definitions (results/K64_LARGE_C_INDEXFREE_L1.md section 6), over unordered input tuples of D (t = 2, 3, 4):

1. realized group coverage: the tuple lies in some generated S-set (all sets, and occupied sets only); a
   geometric upper bound, not retained-weight coverage;
2. dense potential-record coverage: the tuple lies in the nominal K64 record of some VALID neuron (padding and
   physical duplicates never count);
3. frozen support-mask coverage: some valid neuron's ENABLED inputs contain the whole tuple;
4. numerically-nonzero coverage: some valid neuron's nonzero weights (in the evaluated checkpoint) contain
   it; optionally |w| > threshold (explicit, separate from exact zero).

Supports are always per neuron: the union of the masks of different neurons sharing a record is never used
as a record.  S = K = 64: every record is its set; a neuron's support is a 64-bit mask over the set's local
features.  S = 128 (dense only): the logical K64 records (kcsr_coverage.GraphIndex machinery).

Methods (no occurrence lists, no array over the quadruple universe):

* pairs / triples, layer-wide: EXACT, one set at a time: neuron (or record) membership M [n_q, S] (0/1 BF16,
  FP32 accumulation), covered local pairs M^T M > 0, triples (M o M)^T M > 0, written into global bitmaps
  of C(D, 2) / C(D, 3) bytes;
* quadruples (and a triple cross-check): SAMPLED with a predeclared FIXED size (10M; 50M at target sparsity
  >= 90 %), the same seeded distinct uniform tuples for every graph / mask; a tuple's candidate sets are
  the realized sets containing it (one per round); covered if the AND of its features' neuron bitsets is
  non-zero; Clopper-Pearson 95 % intervals and zero-hit upper bounds (no adaptive extension);
* per output g: exact distinct-feature coverage for ALL outputs (min / p10 / median / p90 / max); exact pair
  coverage on a fixed reproducible selection of >= 32 outputs (seeded sample, the lowest-coverage outputs
  and both neuron-count classes); sampled triples / quadruples on the same outputs and the same tuples.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import common  # noqa: F401  (implementation under test on sys.path)
from kcsr_coverage import GraphIndex, clopper_pearson, comb_table, exact_coverage, local_pass_table, \
    local_tuples, rank, refined_prediction, sample_tuples

SAMPLE_SEED = 20261023
OUTPUT_SEED = 20261024
N_QUAD = 10_000_000
N_QUAD_SPARSE = 50_000_000  # target sparsity >= 90 %
N_TRIPLE_CHECK = 1_000_000
N_OUT_TRIPLE = 2_000_000
N_OUT_QUAD = 10_000_000
N_OUTPUTS_RANDOM = 32


# ------------------------------------------------------------------------------ neuron supports (S = K)
class Supports:
    """Valid neurons of an S = K graph with their set q, output g and 64-bit local support (int64 bits)."""

    def __init__(self, feat: torch.Tensor, q: torch.Tensor, g: torch.Tensor, bits: torch.Tensor, D: int, G: int):
        o = torch.argsort(q, stable=True)
        self.q, self.g, self.bits = q[o].contiguous(), g[o].contiguous(), bits[o].contiguous()
        self.feat = feat.long()
        self.Q, self.S = feat.shape
        self.D, self.G = D, G
        self.q_off = torch.searchsorted(self.q, torch.arange(self.Q + 1, device=q.device)).cpu()
        self.n_q = (self.q_off[1:] - self.q_off[:-1]).to(torch.int64)
        per = D // self.S
        self.R = self.Q // per
        self.per_round = per
        sid = torch.empty((self.R, D), dtype=torch.int32, device=q.device)
        for r in range(self.R):
            rows = self.feat[r * per:(r + 1) * per]
            sid[r, rows.reshape(-1)] = torch.arange(per, device=q.device, dtype=torch.int32).repeat_interleave(self.S)
        self.sid = sid
        lid = torch.full((self.Q, D), -1, dtype=torch.int16, device=q.device)
        lid.scatter_(1, self.feat, torch.arange(self.S, device=q.device, dtype=torch.int16).expand(self.Q, self.S).contiguous())
        self.lid = lid
        self._bitsets = None

    def membership(self, qq: int) -> torch.Tensor:
        b = self.bits[int(self.q_off[qq]):int(self.q_off[qq + 1])]
        b = b[b != 0]
        sh = torch.arange(64, device=b.device)
        return ((b[:, None] >> sh) & 1).to(torch.bfloat16)

    def bitsets(self) -> torch.Tensor:
        """[Q, S, W] int64: bit r of row (q, f) = neuron r of q enables local feature f."""
        if self._bitsets is None:
            W = max(1, int((self.n_q.max() + 63) // 64))
            dev = self.q.device
            bits = torch.zeros((self.Q * self.S, W), dtype=torch.int64, device=dev)
            r = torch.arange(self.q.numel(), device=dev) - self.q_off.to(dev)[self.q]
            word, val = r // 64, torch.ones_like(r) << (r % 64)
            for f in range(self.S):
                on = ((self.bits >> f) & 1).bool()
                bits.index_put_((self.q[on] * self.S + f, word[on]), val[on], accumulate=True)
            self._bitsets = bits.view(self.Q, self.S, W)
        return self._bitsets


@torch.no_grad()
def supports_from_dense(L, feat, blk_buckets, mask=None, chunk: int = 1 << 22):
    """Supports of every valid neuron of a KLayout (S = K = 64).  ``mask`` [G, NB, C, 64] (uint8 / bool):
    enabled inputs (None: dense potential = all 64 inputs of the record)."""
    G, NB, C = L.groups, L.blocks_per_group, L.block
    dev = feat.device
    b = blk_buckets.to(dev).to(torch.int64)
    rb = torch.repeat_interleave(torch.arange(len(b), device=dev), b[:, 3])
    kb = torch.arange(len(rb), device=dev) - torch.repeat_interleave(torch.cumsum(b[:, 3], 0) - b[:, 3], b[:, 3])
    bq = torch.full((G * NB,), -1, dtype=torch.int64, device=dev)
    bq[b[rb, 1] * NB + b[rb, 2] + kb] = b[rb, 0]
    valid = L.valid_mask(dev).reshape(-1)
    slots = torch.nonzero(valid).squeeze(1)
    q = bq[slots // C]
    g = slots // (NB * C)
    if bool((q < 0).any()):
        raise AssertionError("a valid neuron's block is in no bucket")
    if mask is None:
        bits = torch.full_like(slots, -1)  # all 64 bits
    else:
        mv = mask.reshape(-1, 64)
        bits = torch.empty_like(slots)
        w = (torch.ones(64, dtype=torch.int64, device=dev) << torch.arange(64, device=dev))
        for s0 in range(0, slots.numel(), chunk):
            sl = slots[s0:s0 + chunk]
            bits[s0:s0 + chunk] = (mv[sl].to(torch.int64) * w).sum(1)  # distinct powers of two: sum = OR
    return Supports(feat, q, g, bits, L.input_dim, G)


# ------------------------------------------------------------------------------ layer-wide coverage
@torch.no_grad()
def exact_support_coverage(sp: Supports, orders=(2, 3), chunk_rows: int = 2048):
    """{t: covered / total / fraction, group (all sets) and group over occupied sets} exact."""
    dev = sp.q.device
    out = {}
    for t in orders:
        ct = comb_table(sp.D, t, dev)
        total = int(ct[sp.D, t])
        cov = torch.zeros(total, dtype=torch.bool, device=dev)
        grp = torch.zeros(total, dtype=torch.bool, device=dev)
        grp_occ = torch.zeros(total, dtype=torch.bool, device=dev)
        lt = local_tuples(sp.S, t, dev)
        for qq in range(sp.Q):
            gr = rank(sp.feat[qq][lt], ct)
            grp[gr] = True
            M = sp.membership(qq)
            if int(sp.n_q[qq]) > 0:
                grp_occ[gr] = True
            if M.shape[0] == 0:
                continue
            if t == 2:
                P = (M.t() @ M) > 0
                hit = P[lt[:, 0], lt[:, 1]]
            else:
                T = torch.zeros((sp.S * sp.S, sp.S), dtype=torch.float32, device=dev)
                for s0 in range(0, M.shape[0], chunk_rows):
                    Mc = M[s0:s0 + chunk_rows]
                    X = (Mc[:, :, None] * Mc[:, None, :]).reshape(Mc.shape[0], sp.S * sp.S)
                    T += (X.t() @ Mc).float()
                hit = T[lt[:, 0] * sp.S + lt[:, 1], lt[:, 2]] > 0
            cov[gr[hit]] = True
        out[t] = dict(method="exact", covered=int(cov.sum()), total=total, fraction=int(cov.sum()) / total,
                      group_covered=int(grp.sum()), group_fraction=int(grp.sum()) / total,
                      group_occupied_covered=int(grp_occ.sum()), group_occupied_fraction=int(grp_occ.sum()) / total)
        del cov, grp, grp_occ
    return out


@torch.no_grad()
def query_supports(sp: Supports, tup: torch.Tensor, chunk: int = 0):
    """(covered [n], group-covered [n], occupied-group-covered [n]) for sorted tuples [n, t]."""
    dev = sp.q.device
    bits = sp.bitsets()
    t = tup.shape[1]
    n = tup.shape[0]
    chunk = chunk or max(1024, (1 << 25) // (sp.R * t))
    hit = torch.zeros(n, dtype=torch.bool, device=dev)
    grp = torch.zeros(n, dtype=torch.bool, device=dev)
    occ = torch.zeros(n, dtype=torch.bool, device=dev)
    nq = sp.n_q.to(dev)
    for s0 in range(0, n, chunk):
        tp = tup[s0:s0 + chunk]
        s = sp.sid[:, tp]
        same = (s == s[:, :, :1]).all(-1)
        r_idx, i_idx = torch.nonzero(same, as_tuple=True)
        if r_idx.numel() == 0:
            continue
        grp[s0 + torch.unique(i_idx)] = True
        q = r_idx * sp.per_round + s[r_idx, i_idx, 0].long()
        o = nq[q] > 0
        occ[s0 + torch.unique(i_idx[o])] = True
        lf = sp.lid[q[:, None], tp[i_idx]].long()
        step = max(1, (1 << 25) // (t * bits.shape[2]))
        for c0 in range(0, q.numel(), step):
            qc, lc = q[c0:c0 + step], lf[c0:c0 + step]
            acc = bits[qc, lc[:, 0]]
            for k in range(1, t):
                acc = acc & bits[qc, lc[:, k]]
            ok = (acc != 0).any(1)
            hit[s0 + i_idx[c0:c0 + step][ok]] = True
    return hit, grp, occ


def sampled(fn, D: int, t: int, n: int, seed: int = SAMPLE_SEED, device="cuda"):
    seq = sample_tuples(D, t, n, seed, device)
    res = fn(seq)
    k = int(res[0].sum())
    lo, hi = clopper_pearson(k, n)
    out = dict(method="sampled", order=t, sample_size=n, sample_seed=seed, fixed_size=True, distinct_tuples=True,
               hits=k, fraction=k / n, ci95=[lo, hi], interval="Clopper-Pearson 95 %",
               upper_bound_if_zero=(hi if k == 0 else None))
    if len(res) > 1:
        kg = int(res[1].sum())
        out.update(group_hits=kg, group_fraction=kg / n, group_ci95=list(clopper_pearson(kg, n)))
    if len(res) > 2:
        ko = int(res[2].sum())
        out.update(group_occupied_hits=ko, group_occupied_fraction=ko / n)
    return out


# ------------------------------------------------------------------------------ per output
@torch.no_grad()
def per_output_features(sp: Supports, chunk: int = 1 << 21) -> torch.Tensor:
    """[G] exact number of distinct input features reached by some valid neuron's support of each output."""
    dev = sp.q.device
    hit = torch.zeros((sp.G, sp.D), dtype=torch.bool, device=dev)
    sh = torch.arange(64, device=dev)
    for s0 in range(0, sp.q.numel(), chunk):
        b = sp.bits[s0:s0 + chunk]
        on = ((b[:, None] >> sh) & 1).bool()  # [n, 64]
        f = sp.feat[sp.q[s0:s0 + chunk]]  # [n, 64]
        gg = sp.g[s0:s0 + chunk][:, None].expand_as(f)
        hit[gg[on], f[on]] = True
    return hit.sum(1)


def select_outputs(G: int, feats: torch.Tensor, counts: torch.Tensor, n_random: int = N_OUTPUTS_RANDOM,
                   seed: int = OUTPUT_SEED):
    """Fixed reproducible output selection: n_random seeded outputs, the 4 outputs with the fewest distinct
    features (graph dependent, listed), and one output of each neuron-count class (n_g = base + 1, base)."""
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randperm(G, generator=g)[:n_random].tolist()
    low = torch.argsort(feats.cpu(), stable=True)[:4].tolist()
    c = counts.cpu()
    cls = [int(torch.nonzero(c == c.max())[0]), int(torch.nonzero(c == c.min())[-1])]
    sel = []
    for x in rnd + low + cls:
        if x not in sel:
            sel.append(x)
    return sel, dict(random=rnd, lowest_feature_coverage=low, count_classes=cls)


@torch.no_grad()
def output_membership(sp: Supports, g: int) -> torch.Tensor:
    """[n_active_neurons_of_g, D] 0/1 BF16 membership of output g's neuron supports (global features)."""
    sel = sp.g == g
    b, q = sp.bits[sel], sp.q[sel]
    b, q = b[b != 0], q[b != 0]
    sh = torch.arange(64, device=b.device)
    on = ((b[:, None] >> sh) & 1).bool()
    M = torch.zeros((b.numel(), sp.D), dtype=torch.bfloat16, device=b.device)
    rows = torch.arange(b.numel(), device=b.device)[:, None].expand_as(on)
    M[rows[on], sp.feat[q][on]] = 1.0
    return M


@torch.no_grad()
def output_coverage(sp: Supports, outputs, tri: torch.Tensor, quad: torch.Tensor, chunk: int = 1 << 16):
    """Per selected output: exact pairs, sampled triples / quadruples (same tuples for every output)."""
    dev = sp.q.device
    ct2 = int(comb_table(sp.D, 2, "cpu")[sp.D, 2])
    iu = torch.triu_indices(sp.D, sp.D, 1, device=dev)
    rows = []
    for g in outputs:
        M = output_membership(sp, g)
        rec = dict(output=int(g), active_neurons=int(M.shape[0]))
        if M.shape[0] == 0:
            rec.update(features=0, pair=0.0, triple_hits=0, quad_hits=0)
            rows.append(rec)
            continue
        rec["features"] = int((M.sum(0) > 0).sum())
        P = (M.t() @ M) > 0
        rec["pair_covered"] = int(P[iu[0], iu[1]].sum())
        rec["pair"] = rec["pair_covered"] / ct2
        W = (M.shape[0] + 63) // 64
        # feature -> neuron bitsets of this output
        Mb = M.t().bool()  # [D, n]
        pad = W * 64 - Mb.shape[1]
        if pad:
            Mb = torch.nn.functional.pad(Mb, (0, pad))
        sh = torch.arange(64, device=dev)
        fb = (Mb.view(sp.D, W, 64).to(torch.int64) << sh).sum(-1)  # [D, W]
        for name, tup in (("triple", tri), ("quad", quad)):
            k = 0
            for s0 in range(0, tup.shape[0], chunk):
                tp = tup[s0:s0 + chunk]
                acc = fb[tp[:, 0]]
                for j in range(1, tp.shape[1]):
                    acc = acc & fb[tp[:, j]]
                k += int((acc != 0).any(1).sum())
            n = tup.shape[0]
            lo, hi = clopper_pearson(k, n)
            rec[f"{name}_hits"], rec[f"{name}_n"], rec[name], rec[f"{name}_ci95"] = k, n, k / n, [lo, hi]
        rows.append(rec)
    return rows


def dist(v: torch.Tensor) -> dict:
    v = v.double().cpu()
    q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return dict(min=float(v.min()), p10=float(q[0]), median=float(q[1]), p90=float(q[2]), max=float(v.max()),
                mean=float(v.mean()))


# ------------------------------------------------------------------------------ dense record outputs (any S)
def _popcount(b: torch.Tensor, chunk: int = 1 << 22) -> torch.Tensor:
    sh = torch.arange(64, device=b.device)
    return torch.cat([((b[s:s + chunk, None] >> sh) & 1).sum(1) for s in range(0, b.numel(), chunk)])


@torch.no_grad()
def record_output_coverage(rec, rg, D, G, outputs, tri, quad, chunk: int = 1 << 16):
    """Per-output coverage of dense potential records (any S): a record shared by C neurons covers what any of
    them covers, so its rows stand for all of its valid neurons."""
    dev = rec.device
    hit = torch.zeros((G, D), dtype=torch.bool, device=dev)
    hit[rg[:, None].expand_as(rec), rec] = True
    feats = hit.sum(1)
    ct2 = D * (D - 1) // 2
    iu = torch.triu_indices(D, D, 1, device=dev)
    rows = []
    for g in outputs:
        r = rec[rg == g]
        M = torch.zeros((r.shape[0], D), dtype=torch.bfloat16, device=dev)
        M.scatter_(1, r, 1.0)
        P = (M.t() @ M) > 0
        row = dict(output=int(g), records=int(r.shape[0]), features=int((M.sum(0) > 0).sum()))
        row["pair_covered"] = int(P[iu[0], iu[1]].sum())
        row["pair"] = row["pair_covered"] / ct2
        W = (r.shape[0] + 63) // 64
        Mb = M.t().bool()
        pad = W * 64 - Mb.shape[1]
        if pad:
            Mb = torch.nn.functional.pad(Mb, (0, pad))
        sh = torch.arange(64, device=dev)
        fb = (Mb.view(D, W, 64).to(torch.int64) << sh).sum(-1)
        for name, tup in (("triple", tri), ("quad", quad)):
            k = 0
            for s0 in range(0, tup.shape[0], chunk):
                tp = tup[s0:s0 + chunk]
                acc = fb[tp[:, 0]]
                for j in range(1, tp.shape[1]):
                    acc = acc & fb[tp[:, j]]
                k += int((acc != 0).any(1).sum())
            n = tup.shape[0]
            lo, hi = clopper_pearson(k, n)
            row[f"{name}_hits"], row[f"{name}_n"], row[name], row[f"{name}_ci95"] = k, n, k / n, [lo, hi]
        rows.append(row)
    return feats, rows


# ------------------------------------------------------------------------------ driver
def iid_reference(S: int, K: int, rho: float, N_q: float, t: int) -> float:
    """Independent-Bernoulli REFERENCE for S = K (not an L1 prediction): conditional local tuple hit of a set
    with N_q neurons whose edges are kept independently with probability rho: 1 - (1 - rho^t)^N_q."""
    return 1.0 - (1.0 - rho ** t) ** N_q


def analyze(case: dict, graph_seed: int, mask_spec: dict = None, quad_n: int = N_QUAD, out_quad_n: int = N_OUT_QUAD):
    """Coverage of one architecture (dense potential) or of a checkpoint / mask (mask_spec)."""
    import k64_cases as KC
    from sparse_pooling.experimental import kgen_graph as KG
    from sparse_pooling.experimental.kgen_layout import KLayout
    from sparse_pooling.supergroups import ConnectivityPlan

    import kcsr_graph

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    C, S, R = case["C"], case["S"], case["R"]
    H = case.get("H", KC.H)
    res = dict(case=case["id"], C=C, S=S, R=R, K=64, H=H, D=KC.D, G=KC.G, graph_seed=graph_seed)
    plan = ConnectivityPlan(S, R, seed=graph_seed)
    L = KLayout(KC.D, H, KC.G, 64, C)
    counts = L.counts_tensor(dev)
    t0 = time.perf_counter()
    idx, feat, ubk, _ = KG.generate(KC.D, counts, C, 64, L.blocks_per_group, 1, plan, dev)
    bk = KG.unit_to_block_buckets(ubk, counts, C, 1)
    tab = KG.record_table(KC.D, H, counts, C, 64, idx, feat, bk)
    res["graph_fingerprint"] = kcsr_graph.fingerprint(tab)
    res["graph_summary"] = kcsr_graph.summary(tab)
    res["graph_seconds"] = time.perf_counter() - t0
    Q = feat.shape[0]
    occ = torch.zeros(Q, dtype=torch.bool, device=dev)
    occ[bk[:, 0].long()] = True
    res["sets"] = dict(Q=Q, occupied=int(occ.sum()), empty=int((~occ).sum()))
    if mask_spec is None:
        res["kind"] = "dense potential records"
    t0 = time.perf_counter()
    tri = sample_tuples(KC.D, 3, N_OUT_TRIPLE, SAMPLE_SEED + 7, dev)
    quad = sample_tuples(KC.D, 4, out_quad_n, SAMPLE_SEED + 8, dev)
    if mask_spec is not None:  # retained supports: one row per valid neuron (S = K = 64)
        if S != 64:
            raise SystemExit("masked coverage is implemented for S = K = 64 only")
        mask, mrec = load_mask(mask_spec, L, idx, feat, bk)
        res["mask"] = mrec
        res["kind"] = mrec["kind"]
        sp = supports_from_dense(L, feat, bk, mask)
        del mask
        res["supports"] = dict(valid_neurons=int(sp.q.numel()), active_neurons=int((sp.bits != 0).sum()),
                               all_zero_neurons=int((sp.bits == 0).sum()), neurons_per_set=dist(sp.n_q.to(torch.float64)),
                               active_fan_in=dist(_popcount(sp.bits).to(torch.float64)))
        ex = exact_support_coverage(sp, (2, 3))
        res["pair"], res["triple"] = ex[2], ex[3]
        res["exact_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        res["quad"] = sampled(lambda tp: query_supports(sp, tp), KC.D, 4, quad_n)
        res["triple_sampled_check"] = sampled(lambda tp: query_supports(sp, tp), KC.D, 3, N_TRIPLE_CHECK)
        res["sampled_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        feats = per_output_features(sp)
        res["per_output_features"] = dist(feats)
        outs, why = select_outputs(KC.G, feats, counts)
        rows = output_coverage(sp, outs, tri, quad)
        res["per_output_seconds"] = time.perf_counter() - t0
        # independent-Bernoulli reference at the mask's density (assumptions stated; not an L1 prediction)
        rho = mrec["density"]
        nq = float(sp.n_q.double().mean())
        res["iid_bernoulli_reference"] = dict(
            assumptions="S = K; every edge of every valid neuron kept independently with probability rho; "
                        "N_q = mean valid neurons per set; tuple inside a set: 1 - (1 - rho^t)^N_q",
            rho=rho, N_q=nq, local_hit={t: iid_reference(64, 64, rho, nq, t) for t in (2, 3, 4)})
    else:  # dense potential: one row per DISTINCT valid logical record (any S; kcsr_coverage machinery)
        gi = GraphIndex(tab, dev)
        res["distinct_records"] = int(gi.n_distinct.sum())
        ex = exact_coverage(gi, (2, 3))
        res["pair"], res["triple"] = ex[2], ex[3]
        nq = gi.n_distinct.to(dev)
        for t, key in ((2, "pair"), (3, "triple")):  # group coverage of the OCCUPIED sets only
            ct = comb_table(KC.D, t, dev)
            occ_b = torch.zeros(int(ct[KC.D, t]), dtype=torch.bool, device=dev)
            lt = local_tuples(S, t, dev)
            for qq in torch.nonzero(nq > 0).squeeze(1).tolist():
                occ_b[rank(gi.feat[qq][lt], ct)] = True
            res[key]["group_occupied_covered"] = int(occ_b.sum())
            res[key]["group_occupied_fraction"] = int(occ_b.sum()) / occ_b.numel()
            del occ_b
        res["exact_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        from kcsr_coverage import query

        res["quad"] = sampled(lambda tp: query(gi, tp), KC.D, 4, quad_n)
        res["triple_sampled_check"] = sampled(lambda tp: query(gi, tp), KC.D, 3, N_TRIPLE_CHECK)
        res["sampled_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rec, rg = tab["records"].to(dev).long(), tab["rec_g"].to(dev).long()
        feats0 = torch.zeros((KC.G, KC.D), dtype=torch.bool, device=dev)
        feats0[rg[:, None].expand_as(rec), rec] = True
        outs, why = select_outputs(KC.G, feats0.sum(1), counts)
        feats, rows = record_output_coverage(rec, rg, KC.D, KC.G, outs, tri, quad)
        res["per_output_features"] = dist(feats)
        res["per_output_seconds"] = time.perf_counter() - t0
    v, slots = local_pass_table(KC.D, counts.cpu(), C, 64, S, L.blocks_per_group, 1, ubk)
    res["refined_prediction"] = {nm: refined_prediction(KC.D, 64, S, R, v, t) for nm, t in
                                 (("pair", 2), ("triple", 3), ("quad", 4))}
    res["outputs"] = rows
    res["output_selection"] = why
    res["per_output_summary"] = {k: dist(torch.tensor([r[k] for r in rows], dtype=torch.float64))
                                 for k in ("pair", "triple", "quad", "features")}
    if case["id"].startswith("K64_"):
        res["supplied_prediction_pct"] = {t: dict(model=case[f"{t}_model_pct"],
                                                  group_expectation=case[f"{t}_group_expectation_pct"])
                                          for t in ("pair", "triple", "quad")}
        bounds = KC.per_output_bounds()[C]
        res["supplied_per_output_bounds"] = bounds
    res["analysis_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    res["analysis_seconds"] = time.perf_counter() - t_all
    return res


def load_mask(spec: dict, L, idx, feat, bk):
    """(mask [G, NB, C, 64] uint8, record) of a checkpoint mask spec: dict(checkpoint, source = mask | nonzero |
    threshold, threshold, kind label).  The checkpoint's graph spec is checked against the generated one."""
    from sparse_pooling.experimental import kgx_sparse as SP

    rec, w, b, m = SP.load_checkpoint(spec["checkpoint"])
    valid = L.valid_mask(w.device)
    src = spec.get("source", "mask")
    if src == "mask":
        if m is None:
            raise SystemExit("checkpoint has no mask")
        mask = m
    elif src == "nonzero":
        mask = SP.learned_mask(w, valid)
    elif src == "threshold":
        thr = float(spec["threshold"])
        mask = ((w.float().abs() > thr) & valid[..., None]).to(torch.uint8)
    else:
        raise SystemExit(f"mask source {src!r}")
    W0 = int(valid.sum()) * 64
    nnz = int(mask.sum(dtype=torch.int64))
    out = dict(checkpoint=spec["checkpoint"], checkpoint_rec=rec, source=src, threshold=spec.get("threshold"),
               kind=spec.get("kind", src), density=nnz / W0, nnz=nnz, mask_sha256=SP.mask_fingerprint(mask))
    del w, b
    return mask, out


def main():
    import k64_cases as KC

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--graph-seed", type=int, default=KC.GRAPH_SEED)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--mask-source", default="mask", choices=["mask", "nonzero", "threshold"])
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--kind", default="")
    ap.add_argument("--sparse", action="store_true", help="predeclared 50M quadruple sample (target sparsity >= 90 %)")
    ap.add_argument("--timing-json", default="", help="assert the graph fingerprint of this timing record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case = KC.case(a.case)
    spec = None
    if a.checkpoint:
        spec = dict(checkpoint=a.checkpoint, source=a.mask_source, threshold=a.threshold, kind=a.kind or a.mask_source)
    res = analyze(case, a.graph_seed, spec, quad_n=N_QUAD_SPARSE if a.sparse else N_QUAD)
    from provenance import provenance

    res["provenance"] = provenance(Path(common.__file__).resolve().parent.parent)
    if a.timing_json:
        tj = json.loads(Path(a.timing_json).read_text())
        res["timing_fingerprint_match"] = tj["kcsr"]["graph_fingerprint"] == res["graph_fingerprint"]
        if not res["timing_fingerprint_match"]:
            raise SystemExit(f"graph fingerprint differs from the timing run {a.timing_json}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    q = res["quad"]
    po = res["per_output_summary"]
    print(f"{a.case} seed {a.graph_seed} {res['kind']}: pair {100 * res['pair']['fraction']:.4f}% triple "
          f"{100 * res['triple']['fraction']:.4f}% quad {100 * q['fraction']:.4f}% [{100 * q['ci95'][0]:.4f}, "
          f"{100 * q['ci95'][1]:.4f}] | per-output feat median {res['per_output_features']['median']:.0f} "
          f"pair median {100 * po['pair']['median']:.3f}% triple median {100 * po['triple']['median']:.4f}% "
          f"| fp {res['graph_fingerprint'][:12]} {res['analysis_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
