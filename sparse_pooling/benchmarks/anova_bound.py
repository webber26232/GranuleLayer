"""Functional-ANOVA structural lower bound of the reduced raw-input interaction task
(results/INTERACTION_COVERAGE_K16_DX.md section 4.3).

Target of output g (wps_interaction.targets): t_g(x) = sum over its valid records j and terms u of
(f_r(x_{A_ju}) - mu_r) / (sigma_r sqrt(n_g T)), x ~ N(0, I) independent, r = order(g), A_ju the term's r distinct global
inputs, f_1(x) = x_i, f_r(x) = ReLU(sum_{i in A} x_i - 1) for r >= 2 (mu_r / sigma_r the code's Monte Carlo constants).

Hoeffding / functional-ANOVA decomposition: f(x_A) = sum_{B subset A} h_B(x_B) with mutually orthogonal components
h_B = sum_{C subset B} (-1)^{|B| - |C|} E[f | x_C].  For r >= 2, with b = |C| of the r coordinates fixed, v = sum_{i in C}
x_i - 1 and s = sqrt(r - b) (the free sum is N(0, s^2)):

    E[ReLU(sum x - 1) | x_C] = psi(v, s) = v Phi(v / s) + s phi(v / s),   psi(v, 0) = ReLU(v);

for r = 1 E[x_i | x_C] = x_i if i in C else 0.  Contributions of every (record, term) with the same GLOBAL subset add
coherently (the residual is summed per token before squaring).

Allowed_g(A): some valid neuron of g has an enabled support containing A (global ids).  Any sum of functions of the
available supports (and constants) lies in the span of the allowed components, so

    L_g = E[(sum_{A nonempty, not allowed} H[g, A](x_A))^2] / E[t_g^2]

is the normalized error of the projection onto that (larger) function space: a MATHEMATICAL lower bound of the nMSE of
the finite Mean(ReLU) model with any readout.  Here it is ESTIMATED by Monte Carlo on fixed analysis tokens (the
residual and t^2 on the same tokens, per output, then averaged over outputs like the learning metric); uncertainty by
resampling analysis BATCHES (query clusters), never treating outputs as independent.  BF16 input quantization is not
part of the bound (separate control).  A positive bound is a structural obstruction; a zero bound does not prove
that finitely many ReLU neurons attain the relaxed optimum.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import torch

SQRT2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def psi(v: torch.Tensor, s: float) -> torch.Tensor:
    """E[ReLU(v + Z)] with Z ~ N(0, s^2) (FP64)."""
    if s == 0:
        return torch.relu(v)
    z = v / s
    return v * 0.5 * (1.0 + torch.erf(z / SQRT2)) + s * INV_SQRT2PI * torch.exp(-0.5 * z * z)


def cond_expectation(x: torch.Tensor, B: Sequence[int], r: int, offset: float = 1.0) -> torch.Tensor:
    """E[f_r | x_B] for the columns B of x [N, ...] (B: column indices of the term's FIXED inputs)."""
    if r == 1:
        return x[:, B[0]] if len(B) else torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    v = (x[:, list(B)].sum(1) if len(B) else torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)) - offset
    return psi(v, math.sqrt(r - len(B)))


def subsets(A: Sequence[int]):
    for k in range(len(A) + 1):
        yield from itertools.combinations(A, k)


def residual_program(terms: List[Tuple[Tuple[int, ...], int, float]], allowed) -> List[Tuple[Tuple[int, ...], int, float]]:
    """Coefficients of the disallowed-component residual.

    terms: [(A sorted global inputs, order r, weight w)] -- every (record, term) of an output with its normalization
    weight w = 1 / (sigma_r sqrt(n_g T)).  allowed(A) -> bool for a nonempty global subset.
    Returns [(C, r, coef)]: residual(x) = sum coef * E[f_r | x_C], aggregated over identical (C, r) (exact
    inclusion-exclusion: coef = w sum_{A disallowed, C subset A subset A_term} (-1)^{|A| - |C|})."""
    acc: Dict[Tuple[Tuple[int, ...], int], float] = {}
    cache: Dict[Tuple[int, ...], bool] = {}
    for A_term, r, w in terms:
        A_term = tuple(sorted(A_term))
        for A in subsets(A_term):
            if not A:
                continue
            if A not in cache:
                cache[A] = bool(allowed(A))
            if cache[A]:
                continue
            for C in subsets(A):
                key = (C, r)
                acc[key] = acc.get(key, 0.0) + w * (-1) ** (len(A) - len(C))
    return [(C, r, c) for (C, r), c in acc.items() if c != 0.0]


def full_program(terms) -> List[Tuple[Tuple[int, ...], int, float]]:
    """Every component (allowed or not) of the target: reconstructs sum w (f - 0) exactly (the -mu_r shift is a
    separate constant)."""
    return residual_program(terms, lambda A: False)


def evaluate(program, x: torch.Tensor) -> torch.Tensor:
    """[N] FP64 value of a residual program on tokens x [N, D] (global input columns)."""
    out = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for C, r, c in program:
        out += c * cond_expectation(x, list(C), r)
    return out


class OutputSupports:
    """Per output: the distinct enabled global supports of its valid neurons (as Python frozensets), for Allowed_g."""

    def __init__(self, supports: List[frozenset]):
        self.supports = supports

    def allowed(self, A) -> bool:
        s = set(A)
        return any(s <= S for S in self.supports)


def output_supports(em: torch.Tensor, valid: torch.Tensor, feat: torch.Tensor, bq: torch.Tensor, g: int) -> OutputSupports:
    """em [G, NB, C, 64] mask (None = dense), valid [G, NB, C], feat [Q, 64], bq [G, NB] input set per record."""
    NB = valid.shape[1]
    sup = set()
    for j in range(NB):
        q = int(bq[g, j])
        if q < 0:
            continue
        ids = feat[q].tolist()
        vj = valid[g, j]
        if em is None:
            if bool(vj.any()):
                sup.add(frozenset(ids))
            continue
        m = em[g, j][vj].bool()  # [n_valid, 64]
        if m.shape[0] == 0:
            continue
        um = torch.unique(m, dim=0)
        for row in um:
            sup.add(frozenset(ids[i] for i in torch.nonzero(row).squeeze(1).tolist()))
    return OutputSupports(sorted(sup, key=lambda s: sorted(s)))


def output_terms(ids: torch.Tensor, rvalid: torch.Tensor, order: torch.Tensor, sig, g: int, T: int):
    """[(A, r, w)] of output g: ids [G, NB, T, 4] global inputs, rvalid [G, NB], order [G], sig {r: (mu, sd, _)}."""
    r = int(order[g])
    n_g = int(rvalid[g].sum())
    w = 1.0 / (sig[r][1] * math.sqrt(max(n_g, 1) * T))
    out = []
    for j in range(ids.shape[1]):
        if not bool(rvalid[g, j]):
            continue
        for u in range(T):
            out.append((tuple(int(i) for i in ids[g, j, u, :r].tolist()), r, w))
    return out


def target_fp64(x: torch.Tensor, terms, mu: float) -> torch.Tensor:
    """Direct FP64 evaluation of the output's target from its terms (for reconstruction checks)."""
    t = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for A, r, w in terms:
        xs = x[:, list(A)]
        f = xs[:, 0] if r == 1 else torch.relu(xs.sum(1) - 1.0)
        t += w * (f - mu)
    return t


def sample_outputs(order: torch.Tensor, per_order: int, seed: int) -> Dict[int, List[int]]:
    """Fixed declared output sample: per order, ``per_order`` outputs drawn by a seeded permutation (chosen before any
    loss is examined)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for r in (1, 2, 3, 4):
        cand = torch.nonzero(order.cpu() == r).squeeze(1)
        perm = torch.randperm(len(cand), generator=g)
        out[r] = sorted(cand[perm[:per_order]].tolist())
    return out


def cluster_bootstrap(num: torch.Tensor, den: torch.Tensor, n_boot: int = 2000, seed: int = 0) -> dict:
    """num / den [batches, outputs] per-batch sums; the statistic = mean over outputs of sum_b num / sum_b den.
    Batches are the resampling unit (query clusters).  Returns estimate, bootstrap SE and percentile 95 % interval."""
    est = float((num.sum(0) / den.sum(0)).mean())
    g = torch.Generator().manual_seed(seed)
    nb = num.shape[0]
    idx = torch.randint(0, nb, (n_boot, nb), generator=g)
    bs = (num[idx].sum(1) / den[idx].sum(1)).mean(1)
    lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=bs.dtype)).tolist()
    return dict(estimate=est, boot_se=float(bs.std()), ci95=[lo, hi], batches=nb, n_boot=n_boot)


def evaluate_many(programs: Sequence[list], x: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """[N, len(programs)] FP64 values of several residual programs on the same tokens x [N, D] (vectorized: entries
    grouped by (|C|, r), evaluated ``chunk`` entries at a time, accumulated per program with index_add)."""
    N = x.shape[0]
    dev = x.device
    out = torch.zeros((N, len(programs)), dtype=torch.float64, device=dev)
    groups: Dict[Tuple[int, int], list] = {}
    for p, prog in enumerate(programs):
        for C, r, c in prog:
            groups.setdefault((len(C), r), []).append((p, C, c))
    for (b, r), ents in groups.items():
        for s0 in range(0, len(ents), chunk):
            e = ents[s0:s0 + chunk]
            pi = torch.tensor([t[0] for t in e], dtype=torch.int64, device=dev)
            cf = torch.tensor([t[2] for t in e], dtype=torch.float64, device=dev)
            if b == 0:
                v = torch.full((N, len(e)), -1.0, dtype=torch.float64, device=dev)
            else:
                Ci = torch.tensor([list(t[1]) for t in e], dtype=torch.int64, device=dev)  # [n, b]
                xs = x[:, Ci.reshape(-1)].view(N, len(e), b)
                v = xs.sum(2) if r > 1 else xs[:, :, 0]
                if r > 1:
                    v = v - 1.0
            if r == 1:
                val = v if b else torch.zeros_like(v)
            else:
                val = psi(v, math.sqrt(r - b))
            out.index_add_(1, pi, val * cf[None, :])
    return out
