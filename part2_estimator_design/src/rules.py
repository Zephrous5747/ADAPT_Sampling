"""Elimination and stopping rules for best-arm identification on |g_i|.

All rules act on the active arms, given estimates ``g_hat``, their covariance
matrix over the active arms and the Bonferroni ``z`` (Part I's convention: ``z``
over the pool, shared by marginal and pairwise statements).

``marginal``
    The conservative absolute-gradient rule of the work order: with
    ``r_i = z sd(g_i)``, ``L_i = max(0, |g_i| - r_i)`` and ``U_i = |g_i| + r_i``,
    arm ``i`` goes when ``U_i < max_j L_j``.  Valid whatever the signs.
``pairwise``
    Part I's covariance-aware rule: ``i`` goes when some ``j`` has
    ``|g_j| - |g_i| > z sd(s_j g_j - s_i g_i)``, with the signs ``s`` taken from
    the estimates.  The plug-in signs make it invalid while a sign is unresolved.
``safe``
    ``pairwise`` restricted to what the data support.  A sign is *resolved* when
    the marginal interval excludes zero, ``|g_k| > r_k``.  Arm ``i`` goes when

    * the marginal rule removes it, or
    * some ``j`` with a resolved sign satisfies, with ``t = s_i`` if ``i`` is
      resolved and for *both* ``t = +1`` and ``t = -1`` otherwise,
      ``s_j g_j - t g_i > z sd(s_j g_j - t g_i)``.

    Given ``s_j``, ``|g_j| > |g_i|`` is equivalent to ``s_j g_j > g_i`` and
    ``s_j g_j > -g_i``, so the unresolved case needs no sign for ``i``; on the
    event that every interval holds, no rule step is ever wrong.

Directly designed contrast estimators (level II-E, :mod:`learning`) are passed as
``contrasts[(j, i, t)] = (estimate, variance)`` for ``s_j g_j - t g_i`` and take
precedence over the contrast implied by the arm estimates.

:func:`rho_good_stop` is the stopping rule for approximate selection: return the
leader ``l`` once ``(1 - rho) |g_i| < |g_l|`` is certified for every other
survivor.  Elimination itself stays exact, so the true best arm survives (on the
good event) and the returned arm has ``|g_l| >= (1 - rho) max_i |g_i|``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RULES = ("marginal", "pairwise", "safe")


@dataclass
class Decision:
    survivors: list[int]
    eliminated: list[tuple[int, int, str]] = field(default_factory=list)  # (arm, by, how)
    lower: np.ndarray | None = None  # L_i over the active arms (input order)
    upper: np.ndarray | None = None
    resolved: np.ndarray | None = None
    leader: int | None = None


def _contrast(j, i, sj, t, g, cov):
    """Estimate and variance of ``s_j g_j - t g_i`` (``j``, ``i`` are row positions)."""
    return (sj * g[j] - t * g[i], cov[j, j] + cov[i, i] - 2.0 * sj * t * cov[j, i])


def eliminate(
    active: list[int],
    estimates: np.ndarray,
    covariance: np.ndarray,
    z: float,
    rule: str,
    contrasts: dict | None = None,
) -> Decision:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    g = np.asarray(estimates, dtype=float)[active]
    cov = np.asarray(covariance, dtype=float)
    variance = np.maximum(np.diag(cov), 0.0)
    radius = z * np.sqrt(variance)
    magnitude = np.abs(g)
    lower = np.maximum(0.0, magnitude - radius)
    upper = magnitude + radius
    resolved = magnitude > radius
    signs = np.sign(g)
    leader_row = int(np.argmax(magnitude))
    decision = Decision([], lower=lower, upper=upper, resolved=resolved, leader=active[leader_row])

    best_row = int(np.argmax(lower))
    gone: dict[int, tuple[int, str]] = {}
    for row in np.flatnonzero(upper < lower[best_row]):
        gone[int(row)] = (active[best_row], "absolute")

    if rule == "pairwise":
        pairwise = variance[:, None] + variance[None, :] - 2.0 * np.outer(signs, signs) * cov
        margin = (magnitude[None, :] - magnitude[:, None]) - z * np.sqrt(np.maximum(pairwise, 0.0))
        for row in np.flatnonzero((margin > 0).any(axis=1)):
            gone.setdefault(int(row), (active[int(np.argmax(margin[row]))], "pairwise"))
    elif rule == "safe":
        contrasts = contrasts or {}
        for i_row in range(len(active)):
            if i_row in gone:
                continue
            ts = (signs[i_row],) if resolved[i_row] else (1.0, -1.0)
            for j_row in np.flatnonzero(resolved):
                if j_row == i_row or magnitude[j_row] <= magnitude[i_row]:
                    continue
                sj = signs[j_row]
                ok = True
                for t in ts:
                    key = (active[j_row], active[i_row], float(t))
                    if key in contrasts:
                        value, var = contrasts[key]
                    else:
                        value, var = _contrast(j_row, i_row, sj, t, g, cov)
                    if not value > z * np.sqrt(max(var, 0.0)):
                        ok = False
                        break
                if ok:
                    how = "contrast" if any((active[j_row], active[i_row], float(t)) in contrasts
                                            for t in ts) else "pairwise"
                    gone[i_row] = (active[int(j_row)], how)
                    break
    decision.survivors = [arm for row, arm in enumerate(active) if row not in gone]
    decision.eliminated = [(active[row], by, how) for row, (by, how) in sorted(gone.items())]
    return decision


def rho_good_stop(
    active: list[int],
    estimates: np.ndarray,
    covariance: np.ndarray,
    z: float,
    rho: float,
) -> bool:
    """True when the leader is certified ``rho``-good against every other survivor."""
    if rho <= 0.0 or len(active) <= 1:
        return len(active) <= 1
    g = np.asarray(estimates, dtype=float)[active]
    cov = np.asarray(covariance, dtype=float)
    magnitude = np.abs(g)
    radius = z * np.sqrt(np.maximum(np.diag(cov), 0.0))
    lead = int(np.argmax(magnitude))
    if not magnitude[lead] > radius[lead]:
        return False
    s = np.sign(g[lead])
    c = 1.0 - rho
    for i in range(len(active)):
        if i == lead:
            continue
        for t in (1.0, -1.0):
            value = s * g[lead] - c * t * g[i]
            var = cov[lead, lead] + c * c * cov[i, i] - 2.0 * c * s * t * cov[lead, i]
            if not value > z * np.sqrt(max(var, 0.0)):
                return False
    return True
