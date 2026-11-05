"""Functional-ANOVA structural bound of the reduced task (benchmarks/anova_bound.py; results/INTERACTION_COVERAGE_K16_DX.md
section 4.3): conditional expectations against Monte Carlo, exact reconstruction of the target from all components,
zero residual for dense / full supports, additive-only supports vs the code's additive floor, order-1 equivalence with
the missing-input floor, coherent duplicated / overlapping terms, vectorized = scalar evaluation."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import anova_bound as AB  # noqa: E402


@pytest.mark.parametrize("r,b", [(2, 0), (2, 1), (3, 1), (3, 2), (4, 0), (4, 2), (4, 3)])
def test_conditional_expectation_matches_monte_carlo(r, b):
    g = torch.Generator().manual_seed(10 * r + b)
    fixed = torch.randn(32, b, generator=g, dtype=torch.float64)
    x = torch.zeros(32, r, dtype=torch.float64)
    x[:, :b] = fixed
    ce = AB.cond_expectation(x, list(range(b)), r)
    n = 400_000
    z = torch.randn(n, r - b, generator=g, dtype=torch.float64)
    for i in range(32):
        mc = torch.relu(fixed[i].sum() + z.sum(1) - 1.0)
        se = float(mc.std()) / math.sqrt(n)
        assert abs(float(mc.mean()) - float(ce[i])) < 5 * se + 1e-8  # deep negative tails: MC sees no mass


def test_numerical_integration_one_free_coordinate():
    """b = r - 1: E[ReLU(v + Z)] by trapezoidal integration over Z ~ N(0, 1)."""
    z = torch.linspace(-12, 12, 200_001, dtype=torch.float64)
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    for v in (-3.0, -1.0, -0.2, 0.0, 0.7, 2.5):
        num = torch.trapezoid(torch.relu(v + z) * pdf, z)
        assert abs(float(num) - float(AB.psi(torch.tensor(v, dtype=torch.float64), 1.0))) < 1e-9


def _terms(seed, n, D=40):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        r = int(torch.randint(1, 5, (1,), generator=g))
        A = tuple(sorted(torch.randperm(D, generator=g)[:r].tolist()))
        out.append((A, r, float(torch.rand(1, generator=g)) + 0.5))
    return out


def test_full_reconstruction_of_the_target():
    terms = _terms(1, 12)
    x = torch.randn(4096, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    mu = 0.37  # any code constant: the shift is a constant component
    t = AB.target_fp64(x, terms, mu)
    full = AB.evaluate(AB.full_program(terms), x)
    const = sum(w * ((0.0 if r == 1 else float(AB.psi(torch.tensor(-1.0, dtype=torch.float64), math.sqrt(r)))) - mu)
                for _, r, w in terms)
    assert float((t - (full + const)).abs().max()) < 1e-12


def test_dense_and_full_support_have_zero_residual():
    terms = _terms(3, 10)
    assert AB.residual_program(terms, lambda A: True) == []
    single = [((3, 7, 11), 3, 1.0)]
    sup = AB.OutputSupports([frozenset({3, 7, 11, 20})])
    assert AB.residual_program(single, sup.allowed) == []


def test_additive_only_support_matches_the_additive_floor():
    import wps_interaction as WI

    sig = WI.sigmas(torch.device("cpu"))
    x = torch.randn(400_000, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    for r in (2, 3, 4):
        prog = AB.residual_program([(tuple(range(r)), r, 1.0)], lambda A: len(A) == 1)
        R = AB.evaluate(prog, x)
        est = float((R ** 2).mean()) / sig[r][1] ** 2
        assert abs(est - sig[r][2]) / sig[r][2] < 0.02, (r, est, sig[r][2])


def test_order1_residual_is_the_missing_input_sum():
    terms = [((5,), 1, 0.3), ((9,), 1, 0.3), ((5,), 1, 0.3), ((2,), 1, 0.3)]
    sup = AB.OutputSupports([frozenset({2, 9})])
    prog = AB.residual_program(terms, sup.allowed)
    x = torch.randn(1000, 12, dtype=torch.float64)
    assert float((AB.evaluate(prog, x) - 0.6 * x[:, 5]).abs().max()) < 1e-14  # duplicates add (2 x 0.3)


def test_duplicated_and_overlapping_terms_add_coherently():
    a = ((1, 2, 3), 3, 1.0)
    allowed = AB.OutputSupports([frozenset({1, 2})]).allowed
    x = torch.randn(2000, 6, dtype=torch.float64)
    r1 = AB.evaluate(AB.residual_program([a], allowed), x)
    r2 = AB.evaluate(AB.residual_program([a, a], allowed), x)
    assert float((r2 - 2 * r1).abs().max()) < 1e-12
    # overlapping terms share the component {1, 2}-free parts: the residual is the sum of the per-term residuals
    b = ((2, 3, 4), 3, 0.7)
    r_ab = AB.evaluate(AB.residual_program([a, b], allowed), x)
    r_b = AB.evaluate(AB.residual_program([b], allowed), x)
    assert float((r_ab - r1 - r_b).abs().max()) < 1e-12


def test_unsupported_interaction_example_is_positive_and_supported_is_zero():
    """ReLU(x1 + x2 - 1) with the two inputs in different neurons: the pair component is excluded (positive residual
    with variance = 1 - additive part); with one neuron holding both, zero."""
    term = [((0, 1), 2, 1.0)]
    x = torch.randn(200_000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    split = AB.OutputSupports([frozenset({0}), frozenset({1})]).allowed
    R = AB.evaluate(AB.residual_program(term, split), x)
    assert float((R ** 2).mean()) > 1e-3
    assert AB.residual_program(term, AB.OutputSupports([frozenset({0, 1})]).allowed) == []


def test_vectorized_evaluation_matches_scalar():
    progs = [AB.residual_program(_terms(s, 8), AB.OutputSupports([frozenset(range(0, 40, 2))]).allowed)
             for s in range(5)]
    x = torch.randn(777, 40, dtype=torch.float64)
    many = AB.evaluate_many(progs, x, chunk=7)
    for i, p in enumerate(progs):
        assert float((many[:, i] - AB.evaluate(p, x)).abs().max()) < 1e-12


def test_cluster_bootstrap_estimate():
    num = torch.rand(64, 10, dtype=torch.float64)
    den = num + 1
    r = AB.cluster_bootstrap(num, den, n_boot=200)
    assert abs(r["estimate"] - float((num.sum(0) / den.sum(0)).mean())) < 1e-15 and r["ci95"][0] <= r["estimate"]
