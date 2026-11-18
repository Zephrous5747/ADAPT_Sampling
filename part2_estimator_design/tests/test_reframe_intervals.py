"""Sanity checks of the finite-sample interval functions of ``src/reframe.py`` (GS-1.0, SI S5) and of the round schedule.

The functions are pure: they are checked on synthetic +-1 outcomes with a known mean, so no molecule is needed.
"""
import math

import numpy as np
import pytest

from reframe import bernstein_interval, epsilon_round, history_fragment, magnitude_interval


def test_round_schedule():
    assert epsilon_round(1) == pytest.approx(4.6e-3)
    assert epsilon_round(10) == pytest.approx(1e-3)
    assert epsilon_round(11) == pytest.approx(5e-4)
    assert epsilon_round(30) == pytest.approx(1e-3 * 2.0 ** -20)
    values = [epsilon_round(r) for r in range(1, 31)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_magnitude_interval():
    assert magnitude_interval(-0.2, 0.3) == (0.0, 0.3)
    assert magnitude_interval(0.1, 0.4) == (0.1, 0.4)
    assert magnitude_interval(-0.4, -0.1) == (0.1, 0.4)


def _draw(rng, mean, n):
    plus = rng.binomial(n, (1.0 + mean) / 2.0)
    outcomes = np.concatenate([np.ones(plus), -np.ones(n - plus)])
    return float(outcomes.mean()), float(outcomes.var(ddof=1))


@pytest.mark.parametrize("mean", [0.0, 0.3, -0.8])
@pytest.mark.parametrize("n", [32, 256])
def test_block_interval_covers(mean, n):
    """A block interval is valid at failure probability eta: over 2000 draws it must essentially always contain the true mean."""
    rng = np.random.default_rng(1)
    eta = 0.05 / 20
    misses = 0
    for _ in range(2000):
        m, s2 = _draw(rng, mean, n)
        lo, hi = bernstein_interval([n], [m], [s2], [1.0], eta, 1, 0.0, 0.0)
        misses += not (lo <= mean <= hi)
    assert misses <= 2000 * eta * 4  # nominal rate eta; the construction is conservative, so the real rate is far below


def test_block_interval_shrinks_with_shots():
    widths = []
    for n in (32, 128, 512, 2048):
        lo, hi = bernstein_interval([n], [0.1], [0.99], [1.0], 1e-3, 1, 0.0, 0.0)
        widths.append(hi - lo)
    assert all(a > b for a, b in zip(widths, widths[1:]))


def test_history_fragment_is_clipped_to_the_range():
    lo, hi = history_fragment(10, 0.95, 0.1, 1.0, 1e-3)
    assert -1.0 <= lo <= hi <= 1.0
    assert history_fragment(1, 0.0, 0.0, 2.5, 1e-3) == (-2.5, 2.5)


@pytest.mark.parametrize("mean", [0.0, 0.5])
def test_history_fragment_covers(mean):
    rng = np.random.default_rng(2)
    misses = 0
    for _ in range(2000):
        n = int(rng.integers(2, 300))
        m, s2 = _draw(rng, mean, n)
        lo, hi = history_fragment(n, m, s2, 1.0, 0.025 / 20)
        misses += not (lo <= mean <= hi)
    assert misses <= 5


def test_residual_widens_the_interval():
    a = bernstein_interval([64], [0.2], [0.9], [1.0], 1e-3, 1, 0.0, 0.0)
    b = bernstein_interval([64], [0.2], [0.9], [1.0], 1e-3, 1, 0.0, 0.05)
    assert b[1] - b[0] == pytest.approx(a[1] - a[0] + 0.1)
    assert math.isfinite(b[0])


@pytest.mark.parametrize("means", [(0.0, 0.0), (0.99, -0.99), (0.0, 0.97), (-0.995, 0.995)])
def test_family_wise_coverage_of_a_weighted_block_near_worst_case_variances(means):
    """Adversarial regimes of SI S5: two fragments with unequal weights (1.0 and 0.2) and unequal shot counts (32 x 2^l and 64 x 2^l) over six doubling
    checkpoints of one block, with means that give the maximal variance (0) or almost none (+-0.99, where a sample variance of zero is likely at 32 shots).
    The per-checkpoint budget is eta = (delta / 2) / (M b (b + 1) l (l + 1)) with M = 20 candidates; the probability that any checkpoint interval misses the
    true weighted sum must stay below delta / 2 = 0.025 (the construction is conservative, so the observed rate should be far below it)."""
    rng = np.random.default_rng(7)
    delta, M, weights = 0.05, 20, (1.0, 0.2)
    truth = sum(w * m for w, m in zip(weights, means))
    trials, misses = 4000, 0
    for _ in range(trials):
        draws = [np.where(rng.random(1024) < (1.0 + m) / 2.0, 1.0, -1.0) * w for m, w in zip(means, weights)]
        missed = False
        for ell in range(1, 7):
            counts = [32 * 2 ** (ell - 1), 64 * 2 ** (ell - 1)]
            eta = (delta / 2.0) / (M * 1 * 2 * ell * (ell + 1))
            lo, hi = bernstein_interval(counts, [d[:n].mean() for d, n in zip(draws, counts)], [d[:n].var(ddof=1) for d, n in zip(draws, counts)],
                                        list(weights), eta, 2, 0.0, 0.0)
            if not (lo <= truth <= hi):
                missed = True
                break
        misses += missed
    assert misses / trials <= delta / 2.0
