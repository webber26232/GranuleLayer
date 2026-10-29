"""Exact references of the coverage-constrained support problem of one original record
(results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md section 4.2):

    maximize sum_{u, j} a[u, j] X[u, j]   s.t.  sum_j X[u, j] = k (every valid group u),
                                                sum_u X[u, j] >= 1 (every local input j),  X[u, j] in {0, 1}

-- a maximum-weight bipartite b-matching / transportation problem with a column LOWER bound.  Solvers (mature, not
written here):

* ``milp_opt``   scipy.optimize.milp (HiGHS) on the binary program with mip_rel_gap = 0 and the FP64 objective as is
                 (no quantization); the returned X is rounded, checked integral (|x - round(x)| <= 1e-9) and feasible,
                 and its energy recomputed in FP64; the LP relaxation (scipy linprog, HiGHS) is reported as an upper
                 bound beside it (the constraint matrix is totally unimodular, so the LP optimum is attained by an
                 integral vertex: bound == optimum up to solver tolerance);
* ``flow_opt``   networkx network simplex on the min-cost-flow form with INTEGER costs (a quantized to
                 round(a 2^Q_BITS / max a)): source -> group u (exactly k), u -> j (capacity 1, cost -a_q[u, j]),
                 j -> sink with the lower bound 1 removed by the standard transformation (node demand +1 at j, -1 at
                 the sink; residual capacity m - 1); the integral flow is the mask; its FP64 energy is reported
                 beside the quantized objective (the gap to milp_opt measures quantization);
* ``exhaustive`` brute force over all C(S, k)^m masks for tiny (m, S, k), the oracle of both.

A fractional result is never called a mask; an infeasible or uncertified case is reported as such.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

Q_BITS = 40


def energy(a: np.ndarray, X: np.ndarray) -> float:
    """FP64 energy summed in (u, j) order with Python floats (math.fsum: exactly rounded)."""
    return math.fsum(float(v) for v in a[X.astype(bool)])


def feasible(X: np.ndarray, k: int) -> bool:
    X = X.astype(np.int64)
    return bool(((X == 0) | (X == 1)).all() and (X.sum(1) == k).all() and (X.sum(0) >= 1).all())


def exhaustive(a: np.ndarray, k: int):
    """Brute force over all C(S, k)^m masks (vectorized; tiny m, S only).  Energies of the winner are recomputed with
    ``energy``; among equal FP64 sums the first mask in product order is returned."""
    m, S = a.shape
    if m * k < S:
        return None, None
    subs = list(itertools.combinations(range(S), k))
    masks = np.zeros((len(subs), S), dtype=bool)
    for i, c in enumerate(subs):
        masks[i, list(c)] = True
    bits = (masks.astype(np.int64) << np.arange(S)).sum(1)
    n = len(subs)
    tot = np.zeros((1,), dtype=np.float64)
    cov = np.zeros((1,), dtype=np.int64)
    for u in range(m):  # product order: group 0 slowest
        eu = masks @ a[u]
        tot = (tot[:, None] + eu[None, :]).reshape(-1)
        cov = (cov[:, None] | bits[None, :]).reshape(-1)
    ok = cov == (1 << S) - 1
    if not ok.any():
        return None, None
    idx = int(np.flatnonzero(ok)[np.argmax(tot[ok])])
    combo = np.unravel_index(idx, (n,) * m)
    X = masks[list(combo)]
    return energy(a, X), X


def milp_opt(a: np.ndarray, k: int) -> dict:
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix, vstack

    m, S = a.shape
    if m * k < S:
        return dict(status="infeasible", reason=f"m k = {m * k} < S = {S}")
    n = m * S
    r = np.repeat(np.arange(m), S)
    cidx = np.tile(np.arange(S), m)
    Arow = coo_matrix((np.ones(n), (r, np.arange(n))), shape=(m, n))
    Acol = coo_matrix((np.ones(n), (cidx, np.arange(n))), shape=(S, n))
    A = vstack([Arow, Acol]).tocsr()
    lo = np.concatenate([np.full(m, k), np.ones(S)])
    hi = np.concatenate([np.full(m, k), np.full(S, np.inf)])
    c = -a.reshape(-1).astype(np.float64)
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1),
               options=dict(mip_rel_gap=0.0, presolve=True, disp=False))
    out = dict(solver=f"scipy {scipy.__version__} milp (HiGHS)", options="mip_rel_gap 0, presolve on, default "
               "feasibility tolerances (1e-7)", status=int(res.status), message=str(res.message))
    # LP relaxation with the column lower bound as -sum_u X <= -1
    lp = linprog(c, A_ub=-Acol.tocsr(), b_ub=-np.ones(S), A_eq=Arow.tocsr(), b_eq=np.full(m, k), bounds=(0, 1),
                 method="highs")
    out["lp_upper_bound"] = float(-lp.fun) if lp.status == 0 else None
    if res.status != 0 or res.x is None:
        out["certified"] = False
        return out
    x = res.x.reshape(m, S)
    X = np.rint(x)
    out["max_integrality_violation"] = float(np.abs(x - X).max())
    out["integral"] = out["max_integrality_violation"] <= 1e-9
    out["feasible"] = feasible(X, k)
    out["X"] = X.astype(bool)
    out["energy"] = energy(a, X)
    out["solver_objective"] = float(-res.fun)
    out["certified"] = bool(out["integral"] and out["feasible"])
    return out


def flow_opt(a: np.ndarray, k: int, q_bits: int = Q_BITS) -> dict:
    import networkx as nx

    m, S = a.shape
    if m * k < S:
        return dict(status="infeasible")
    amax = float(a.max())
    scale = (2 ** q_bits) / amax if amax > 0 else 1.0
    aq = np.rint(a * scale).astype(np.int64)
    G = nx.DiGraph()
    G.add_node("s", demand=-m * k)
    G.add_node("t", demand=m * k - S)  # lower bound 1 on every j -> t: S units pre-routed
    for u in range(m):
        G.add_node(("u", u), demand=0)
        G.add_edge("s", ("u", u), capacity=k, weight=0)
    for j in range(S):
        G.add_node(("j", j), demand=1)  # receives its pre-routed unit
        G.add_edge(("j", j), "t", capacity=m - 1, weight=0)
    for u in range(m):
        for j in range(S):
            G.add_edge(("u", u), ("j", j), capacity=1, weight=-int(aq[u, j]))
    cost, flow = nx.network_simplex(G)
    X = np.zeros((m, S), dtype=bool)
    for u in range(m):
        for j, f in flow[("u", u)].items():
            if f:
                X[u, j[1]] = True
    return dict(solver=f"networkx {nx.__version__} network_simplex", q_bits=q_bits, scale=scale,
                quantized_objective=int(-cost), X=X, feasible=feasible(X, k), energy=energy(a, X))
