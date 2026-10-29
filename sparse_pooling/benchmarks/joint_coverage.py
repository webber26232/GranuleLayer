"""Selected-output tuple coverage with the SHARED-query uncertainty (results/TOPK16_COVERAGE_CONSTRAINED_TOPK.md
section 2.1).

Every selected output is checked against the SAME sampled tuples (kcsr_coverage.sample_tuples: n distinct uniform
sorted t-tuples, a simple random sample WITHOUT replacement from the N = C(D, t) tuples).  The O x n output-by-query
hits are therefore not O n independent Bernoulli trials: the independent unit is the query.  For query i let
c_i = number of selected outputs whose support contains it, y_i = c_i / O.  The selected-output mean coverage is

    p = mean_i y_i,   SE(p) = sqrt((1 - n / N) s_y^2 / n)          (s_y^2 the sample variance of y_i),

the finite-population-corrected SRSWOR standard error; the 95 % interval is p +- 1.96 SE (n = 2M / 10M queries),
clipped to [0, 1].  A query-cluster bootstrap resamples queries (each with all of its outputs); because y_i depends on
c_i only, it is a multinomial resampling of the histogram of c_i -- bounded memory, exact equivalence.  A paired
difference of two masks on the same queries uses d_i = (c_i^A - c_i^B) / O and the same formula (per-query arrays of
both masks are kept for that: ``*.joint.npz``).

The pooled value is CONDITIONAL ON THE CHOSEN OUTPUTS (kcsr_coverage / k64_coverage select_outputs: 32 seeded random
outputs, the 4 dense-lowest-feature outputs and 2 count-class outputs); it is not an unweighted random sample of all
outputs and carries no output-sampling uncertainty.  Individual-output Clopper-Pearson intervals stay valid.
"""
from __future__ import annotations

import math

import numpy as np

SCHEMA = "granule.joint_coverage/1"
Z95 = 1.959963984540054


def summary(c: np.ndarray, O: int, N: int, boot: int = 2000, seed: int = 0) -> dict:
    """c [n] per-query covering-output counts (0..O) over n SRSWOR queries from N tuples."""
    c = np.asarray(c, dtype=np.int64)
    n = int(c.size)
    if O <= 0 or n == 0:
        raise ValueError("joint coverage: O >= 1 outputs and n >= 1 queries")
    if c.min() < 0 or c.max() > O:
        raise ValueError("joint coverage: counts outside 0..O")
    hist = np.bincount(c, minlength=O + 1)
    y = np.arange(O + 1) / O
    p = float((hist * y).sum() / n)
    var = float((hist * (y - p) ** 2).sum() / (n - 1)) if n > 1 else 0.0
    fpc = 1.0 - n / N if N else 1.0
    se = math.sqrt(max(fpc, 0.0) * var / n)
    out = dict(schema=SCHEMA, outputs=O, queries=n, population=N, fpc=fpc, mean=p, se=se,
               ci95=[max(0.0, p - Z95 * se), min(1.0, p + Z95 * se)],
               counts_histogram=hist.tolist(), queries_covered_by_any=int(n - hist[0]),
               output_hits=int((hist * np.arange(O + 1)).sum()),
               sampling="simple random sample of distinct tuples without replacement; unit = query (all selected "
                        "outputs of a query kept together); finite-population correction 1 - n / N",
               conditional_on="the fixed selected outputs (not a random sample of all outputs)")
    if boot:
        rng = np.random.default_rng(seed)
        bs = rng.multinomial(n, hist / n, size=boot) @ y / n
        bs = bs * math.sqrt(max(fpc, 0.0)) + p * (1 - math.sqrt(max(fpc, 0.0)))  # FPC-scaled about the estimate
        out["bootstrap"] = dict(replicates=boot, seed=seed, percentile95=[float(np.quantile(bs, 0.025)),
                                                                           float(np.quantile(bs, 0.975))],
                                se=float(bs.std(ddof=1)), method="query-cluster bootstrap = multinomial resampling "
                                "of the c_i histogram, deviations scaled by sqrt(FPC)")
    return out


def paired(ca: np.ndarray, cb: np.ndarray, O: int, N: int) -> dict:
    """Paired difference A - B of the selected-output mean coverage on the SAME queries (same outputs)."""
    ca, cb = np.asarray(ca, dtype=np.int64), np.asarray(cb, dtype=np.int64)
    if ca.shape != cb.shape:
        raise ValueError("paired coverage: the two masks must be evaluated on the same queries")
    n = int(ca.size)
    d = (ca - cb) / O
    m = float(d.mean())
    var = float(d.var(ddof=1)) if n > 1 else 0.0
    fpc = 1.0 - n / N if N else 1.0
    se = math.sqrt(max(fpc, 0.0) * var / n)
    return dict(schema=SCHEMA, outputs=O, queries=n, population=N, mean_difference=m, se=se,
                ci95=[m - Z95 * se, m + Z95 * se], queries_differing=int((ca != cb).sum()))


def naive_pooled_binomial(c: np.ndarray, O: int) -> dict:
    """The INVALID pooled interval of kwa_report.cov_row at b18b7a6 (O n treated as independent trials), kept only to
    show the error next to the corrected one."""
    from kcsr_coverage import clopper_pearson

    c = np.asarray(c, dtype=np.int64)
    k, n = int(c.sum()), int(c.size) * O
    return dict(hits=k, trials=n, ci95=list(clopper_pearson(k, n)), invalid="shared queries are not independent")
