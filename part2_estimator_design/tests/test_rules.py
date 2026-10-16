"""Elimination rules: the absolute-CI rule, plug-in pairwise, and the sign-aware rule."""
from __future__ import annotations

import numpy as np

from rules import eliminate, rho_good_stop


def _random_case(rng, n=6):
    root = rng.normal(size=(n, n))
    covariance = 1e-3 * root @ root.T
    estimates = rng.normal(scale=0.2, size=n)
    return estimates, covariance


def test_marginal_is_the_absolute_interval_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, cov = _random_case(rng)
        active = list(range(g.size))
        r = 2.5 * np.sqrt(np.diag(cov))
        lower, upper = np.maximum(0, np.abs(g) - r), np.abs(g) + r
        expected = [i for i in active if upper[i] >= lower.max()]
        assert eliminate(active, g, cov, 2.5, "marginal").survivors == expected


def test_safe_and_pairwise_contain_the_absolute_rule():
    """With a PSD covariance a contrast's sd is at most the sum of the two sds."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        g, cov = _random_case(rng)
        active = list(range(g.size))
        marginal = set(eliminate(active, g, cov, 2.0, "marginal").survivors)
        for rule in ("pairwise", "safe"):
            assert set(eliminate(active, g, cov, 2.0, rule).survivors) <= marginal


def test_safe_equals_pairwise_when_every_sign_is_resolved():
    rng = np.random.default_rng(2)
    for _ in range(200):
        g, cov = _random_case(rng)
        g = np.sign(g) * (np.abs(g) + 1.0)  # far from zero: every sign resolved
        active = list(range(g.size))
        assert eliminate(active, g, cov, 2.0, "safe").survivors == eliminate(active, g, cov, 2.0, "pairwise").survivors


def test_plug_in_signs_eliminate_the_best_arm_and_the_safe_rule_does_not():
    """Best arm b with a small, unresolved gradient; arm a strongly anti-correlated.

    With the signs estimated as (s_a, s_b) = (+, -) the plug-in contrast
    ``s_a g_a - s_b g_b`` is ``g_a + g_b``, whose true value is positive and whose
    variance is tiny, so plug-in pairwise removes the best arm.  The safe rule
    needs a resolved sign for the eliminating arm and both signs for the other.
    """
    truth = np.array([0.020, -0.010])  # b = 0, a = 1
    sd = 0.05
    cov = sd ** 2 * np.array([[1.0, -0.999], [-0.999, 1.0]])
    z = 2.0
    rng = np.random.default_rng(3)
    draws = rng.multivariate_normal(truth, cov, size=4000)
    lost = {"pairwise": 0, "safe": 0}
    for g in draws:
        for rule in lost:
            lost[rule] += 0 not in eliminate([0, 1], g, cov, z, rule).survivors
    assert lost["pairwise"] / len(draws) > 0.2
    assert lost["safe"] / len(draws) < 0.05


def test_designed_contrasts_take_precedence():
    g = np.array([1.0, 0.97])
    cov = np.diag([1e-4, 1e-4])
    # The absolute intervals overlap, but the arms' contrast separates the pair
    # (0.03 > 2 * sqrt(2e-4) = 0.028) ...
    assert eliminate([0, 1], g, cov, 2.0, "safe").survivors == [0]
    # ... but a designed contrast with a larger variance overrides the arms.
    contrasts = {(0, 1, 1.0): (0.03, 0.01)}
    decision = eliminate([0, 1], g, cov, 2.0, "safe", contrasts)
    assert decision.survivors == [0, 1]
    contrasts = {(0, 1, 1.0): (0.03, 1e-4)}
    decision = eliminate([0, 1], g, cov, 2.0, "safe", contrasts)
    assert decision.survivors == [0] and decision.eliminated == [(1, 0, "contrast")]


def test_rho_good_stop():
    cov = np.diag([1e-4, 1e-4, 1e-4])
    tied = np.array([0.50, -0.50, 0.10])
    assert not rho_good_stop([0, 1, 2], tied, cov, 2.0, 0.0)
    assert rho_good_stop([0, 1, 2], tied, cov, 2.0, 0.1)  # |g_1| (1 - 0.1) < |g_0| is certified
    assert not rho_good_stop([0, 1, 2], tied, cov * 100, 2.0, 0.1)  # but not with 10x the noise
    assert rho_good_stop([0], tied, cov, 2.0, 0.1)
