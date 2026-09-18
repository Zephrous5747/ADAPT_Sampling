"""Oracle shot-proxy formulas shared by the M1, M2 and M3 method modules.

All Part I numbers are oracle planning estimates: the state is known exactly in
simulation, so exact fragment variances are available and no finite-shot Monte
Carlo is performed.

Two targets appear throughout:

``epsilon``
    the one-standard-error target on an estimated gradient;
``radius``
    the half-width of a confidence interval, ``radius = z * epsilon``, where
    ``z`` comes from a normal model with a family-wise failure probability.

Every caller converts a radius to an ``epsilon`` with :func:`epsilon_from_radius`
so that the confidence convention is applied in exactly one place.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

DEFAULT_DELTA = 0.05


def z_from_delta(delta: float = DEFAULT_DELTA, n_arms: int = 1) -> float:
    """Two-sided normal quantile with a Bonferroni family-wise correction."""
    from scipy.stats import norm

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return float(norm.ppf(1.0 - delta / (2.0 * max(1, n_arms))))


def z_for_selection_error(alpha: float = DEFAULT_DELTA) -> float:
    """Confidence factor calibrated to the realised selection-error rate.

    :func:`z_from_delta` applies a Bonferroni correction across the pool, which
    turns out not to control the quantity of interest. A run stops when the common
    radius reaches half the gap, at which point each estimate has standard error
    ``gap / (2 z)``; the leader's margin over the runner-up is one gap, so in units
    of the standard deviation of their difference it is ``z * sqrt(2)`` and the
    probability of selecting the wrong generator is about
    ``2 * Phi_bar(z * sqrt(2))``. Inverting that gives the ``z`` for which the
    stated failure probability is the realised one.

    Measured error rates follow this expression rather than the nominal delta, so
    the Bonferroni convention over-delivers confidence by orders of magnitude and
    correspondingly over-spends shots.
    """
    from scipy.stats import norm

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(norm.isf(alpha / 2.0) / math.sqrt(2.0))


def predicted_selection_error(z: float) -> float:
    """The realised selection-error rate implied by a confidence factor."""
    from scipy.stats import norm

    return float(2.0 * norm.sf(z * math.sqrt(2.0)))


def epsilon_from_radius(radius: float, z: float) -> float:
    """Convert a confidence radius to a one-standard-error target."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if z <= 0:
        raise ValueError("z must be positive")
    return radius / z


def fragment_sum_shots(sigmas: Iterable[float], epsilon: float, z: float = 1.0) -> int:
    """Optimal independent-fragment shot estimate for a sum of fragments.

    For ``F = sum_a F_a`` with one-shot standard deviations ``sigma_a``, the
    optimal allocation ``m_a ~ sigma_a`` gives a total of
    ``M = (z * sum_a sigma_a / epsilon)**2`` shots.
    """
    total_sigma = float(np.sum(np.asarray(list(sigmas), dtype=float)))
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return int(math.ceil((z * total_sigma / epsilon) ** 2))


def one_observable_shots(variance: float, epsilon: float, z: float = 1.0) -> int:
    """Shot estimate for a single directly measured observable."""
    if -1e-12 < variance < 0:
        variance = 0.0
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    return fragment_sum_shots([math.sqrt(variance)], epsilon, z=z)


def confidence_radius_from_gap(gap: float, divisor: float = 4.0) -> float:
    """Default planning radius: a fraction of the top absolute-gradient gap."""
    if gap <= 0:
        raise ValueError("gap must be positive")
    return gap / divisor


def allocate_context_shots(
    sigmas: np.ndarray,
    epsilon: float,
    *,
    iterations: int = 500,
    damping: float = 0.5,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Allocate shots over shared measurement contexts.

    ``sigmas[i, alpha]`` is the one-shot standard deviation of the fragment of
    gradient ``i`` inside context ``alpha``.  With ``n_alpha`` shots in context
    ``alpha`` the reconstructed gradient has variance

        Var(g_i) = sum_alpha sigmas[i, alpha]**2 / n_alpha,

    so the cheapest allocation that resolves every gradient solves

        minimise   sum_alpha n_alpha
        subject to Var(g_i) <= epsilon**2  for every i,  n_alpha >= 0.

    This is convex, and its stationarity condition is
    ``n_alpha = sqrt(sum_i lambda_i * sigmas[i, alpha]**2)`` for multipliers
    ``lambda_i >= 0``.  The multipliers are found by a damped multiplicative
    fixed-point iteration, which needs no external solver and is deterministic.

    The returned allocation is always feasible: whatever the iteration converges
    to, it is rescaled at the end so that the tightest constraint holds exactly.
    Convergence therefore affects only how close to optimal the answer is, never
    its validity.  For a single gradient the result matches
    :func:`fragment_sum_shots` exactly.
    """
    sigmas = np.atleast_2d(np.asarray(sigmas, dtype=float))
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    n_arms, n_contexts = sigmas.shape
    if n_arms == 0 or n_contexts == 0:
        return np.zeros(n_contexts)

    squared = sigmas ** 2
    binding = squared.sum(axis=1) > 0.0
    if not binding.any():
        return np.zeros(n_contexts)
    squared = squared[binding]

    multipliers = np.ones(squared.shape[0])
    shots = _shots_from_multipliers(squared, multipliers)
    for _ in range(iterations):
        variances = _variances(squared, shots)
        worst = float((variances / variances.max()).min())
        if 1.0 - worst < tolerance:
            break
        multipliers *= (variances / variances.max()) ** damping
        multipliers = np.maximum(multipliers, 1e-300)
        multipliers /= multipliers.sum()
        shots = _shots_from_multipliers(squared, multipliers)

    return _rescale_to_feasible(squared, shots, epsilon)


def _shots_from_multipliers(squared: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Stationarity condition ``n_alpha = sqrt(sum_i lambda_i sigma_i,alpha**2)``."""
    return np.sqrt(multipliers @ squared)


def _variances(squared: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """Per-arm reconstruction variance, treating unused contexts as unavailable."""
    used = shots > 0.0
    if not used.any():
        return np.full(squared.shape[0], np.inf)
    variances = (squared[:, used] / shots[used]).sum(axis=1)
    unreachable = squared[:, ~used].sum(axis=1) > 0.0
    variances[unreachable] = np.inf
    return variances


def _rescale_to_feasible(
    squared: np.ndarray, shots: np.ndarray, epsilon: float
) -> np.ndarray:
    """Scale an allocation so that the tightest variance constraint holds exactly."""
    total = shots.sum()
    if total <= 0.0:
        return shots
    variances = _variances(squared, shots)
    if not np.isfinite(variances).all():
        raise ValueError(
            "allocation leaves a gradient unmeasured; this indicates a context "
            "with no shots that still carries a non-zero fragment"
        )
    return shots * float((variances / epsilon ** 2).max())


def reconstruction_variances(sigmas: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """Per-gradient variance ``sum_alpha sigma**2 / n`` for a given allocation."""
    sigmas = np.atleast_2d(np.asarray(sigmas, dtype=float))
    shots = np.asarray(shots, dtype=float)
    used = shots > 0.0
    return (sigmas[:, used] ** 2 / shots[used]).sum(axis=1)
