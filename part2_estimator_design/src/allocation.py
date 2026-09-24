"""Context-shot allocation, including top-up from shots already committed.

Part I's :func:`shot_models.allocate_context_shots` solves

    minimise sum_a n_a   subject to   sum_a sigma_ia**2 / n_a <= epsilon**2  (all i)

from scratch.  A run that already holds ``c_a`` shots in context ``a`` should solve
the same programme with ``n_a >= c_a`` instead: those shots are paid for and
reusable, and ignoring them is what lets Part I's per-breakpoint "sum of maxima"
accounting exceed M1 (Part I report, Sec. "M3 is not guaranteed to beat M1").
This module solves the lower-bounded programme.

The lower bounds break the scale invariance Part I's solver relies on, so the
multipliers are found by multiplicative ascent on the Lagrange dual instead.  For
multipliers ``lambda`` the Lagrangian is minimised by

    n_a(lambda) = max(c_a, sqrt(sum_i lambda_i sigma_ia**2)),

and the dual gradient in ``lambda_i`` is ``Var_i(n(lambda)) - epsilon**2``, so
``lambda_i <- lambda_i (Var_i / epsilon**2)**damping`` raises the multiplier of
every violated constraint and lowers that of every slack one.  A final
one-dimensional search scales ``sqrt(sum_i lambda_i sigma_ia**2)`` to the smallest
feasible value, so the result is always feasible and the iteration affects only
how close to optimal it is.  With no lower bounds Part I's own solver is used, so
every Part I number is reproduced exactly.
"""
from __future__ import annotations

import numpy as np

from part1_bridge import allocate_context_shots


def reconstruction_variances(squared: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """``sum_a sigma**2 / n`` per arm; infinite where a needed context has no shots."""
    squared = np.atleast_2d(squared)
    funded = shots > 0
    variances = (squared[:, funded] / shots[funded]).sum(axis=1)
    starved = (squared[:, ~funded] > 0).any(axis=1)
    variances[starved] = np.inf
    return variances


def allocate(
    sigmas: np.ndarray, epsilon: float, lower: np.ndarray | None = None, **kwargs
) -> np.ndarray:
    """Cheapest allocation meeting ``epsilon`` for every row, above ``lower``."""
    sigmas = np.atleast_2d(np.asarray(sigmas, dtype=float))
    if lower is None or not np.any(lower > 0):
        return allocate_context_shots(sigmas, epsilon)
    return allocate_topup(sigmas, epsilon, lower, **kwargs)


def allocate_topup(
    sigmas: np.ndarray,
    epsilon: float,
    lower: np.ndarray,
    *,
    iterations: int = 3000,
    damping: float = 0.5,
    rtol: float = 1e-11,
) -> np.ndarray:
    """Solve the allocation programme with ``n_a >= lower_a``."""
    sigmas = np.atleast_2d(np.asarray(sigmas, dtype=float))
    lower = np.asarray(lower, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    squared = sigmas ** 2
    squared = squared[squared.sum(axis=1) > 0]
    target = epsilon ** 2
    if squared.size == 0 or (reconstruction_variances(squared, lower) <= target).all():
        return lower.copy()

    # Single-arm optimum lambda = (sum_a sigma_a)**2 / epsilon**4, shared between arms.
    multipliers = (np.sqrt(squared).sum(axis=1) ** 2) / (target ** 2) / squared.shape[0]
    previous = np.inf
    stall = 0
    for _ in range(iterations):
        weights = multipliers @ squared
        shots = np.maximum(lower, np.sqrt(weights))
        variances = reconstruction_variances(squared, shots)
        variances = np.where(np.isfinite(variances), variances, 1e300)
        multipliers = np.clip(multipliers * (variances / target) ** damping, 1e-300, 1e300)
        total = shots.sum()
        if abs(previous - total) <= rtol * total and np.max(variances) <= target * (1 + 1e-9):
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        previous = total
    return _smallest_feasible(squared, lower, np.sqrt(multipliers @ squared), target)


def _smallest_feasible(
    squared: np.ndarray, lower: np.ndarray, shape: np.ndarray, target: float
) -> np.ndarray:
    """``max(lower, t * shape)`` at the smallest feasible ``t``."""

    def worst(t: float) -> float:
        return float(reconstruction_variances(squared, np.maximum(lower, t * shape)).max())

    high = 1.0
    while worst(high) > target:
        high *= 2.0
        if high > 1e30:  # pragma: no cover - a context carrying variance has no shape
            raise ValueError("no feasible allocation: a needed context cannot be funded")
    low = 0.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if worst(mid) > target:
            low = mid
        else:
            high = mid
        if high - low <= 1e-15 * high:
            break
    return np.maximum(lower, high * shape)
