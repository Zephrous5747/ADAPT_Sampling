"""Best-arm identification schedule shared by M2 and M3.

Both BAI methods use the same oracle successive-elimination rule from the Part I
report.  A generator ``i`` is eliminated once

    |g_i| + r < max_{j in A} (|g_j| - r),

where ``r`` is the common confidence radius of the current round and ``A`` is
the active set.  Because Part I is an oracle diagnostic, the estimates used in
that rule are the exact gradients, so a round eliminates exactly those arms
whose absolute-gradient deficit exceeds ``2r``.

The two methods differ only in how they pay for a round, so this module owns the
schedule and the elimination rule while each method module owns its own shot
accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

DEFAULT_SHRINK = 0.5
MINIMUM_RADIUS = 1e-12
MAXIMUM_ROUNDS = 400


@dataclass(frozen=True)
class Threshold:
    """One breakpoint of the schedule-free (exact-limit) elimination trajectory."""

    index: int
    radius: float
    active: tuple[int, ...]
    eliminated: tuple[int, ...]


def deficits(abs_gradients: np.ndarray) -> np.ndarray:
    """Absolute-gradient deficit of every arm, ``Delta_i = |g_1| - |g_i|``.

    The leader's own deficit is zero; :func:`final_radii` replaces it with the gap,
    since the leader must still be separated from the runner-up.
    """
    abs_gradients = np.asarray(abs_gradients, dtype=float)
    return float(abs_gradients.max()) - abs_gradients


def final_radii(abs_gradients: np.ndarray) -> np.ndarray:
    """Radius at which each arm stops being measured, in the exact limit.

    As the common radius shrinks continuously, arm ``i`` leaves the active set the
    moment ``2r`` falls below its deficit, so it is resolved to exactly
    ``Delta_i / 2``.  The leader survives to ``gap / 2``.
    """
    abs_gradients = np.asarray(abs_gradients, dtype=float)
    delta = deficits(abs_gradients)
    positive = delta[delta > 0]
    gap = float(positive.min()) if positive.size else float(abs_gradients.max())
    radii = delta / 2.0
    radii[delta <= 0] = gap / 2.0
    return radii


def elimination_thresholds(abs_gradients: np.ndarray) -> list[Threshold]:
    """The exact-limit trajectory: one breakpoint per distinct deficit.

    Between two consecutive distinct deficits the active set does not change, and
    the shot requirement is largest at the tighter end of the interval, so the cost
    of the whole trajectory is determined by these finitely many radii.  This
    replaces the geometric schedule and its arbitrary shrink factor.
    """
    abs_gradients = np.asarray(abs_gradients, dtype=float)
    delta = deficits(abs_gradients)
    distinct = np.unique(delta[delta > 0])
    thresholds: list[Threshold] = []
    for index, value in enumerate(sorted(distinct, reverse=True)):
        active = tuple(int(i) for i in np.flatnonzero(delta <= value))
        eliminated = tuple(int(i) for i in np.flatnonzero(delta == value))
        thresholds.append(Threshold(index, float(value) / 2.0, active, eliminated))
    return thresholds


@dataclass(frozen=True)
class EliminationRound:
    """One round of successive elimination."""

    index: int
    radius: float
    active_before: tuple[int, ...]
    eliminated: tuple[int, ...]

    @property
    def active_after(self) -> tuple[int, ...]:
        removed = set(self.eliminated)
        return tuple(i for i in self.active_before if i not in removed)


def survivors(abs_gradients: np.ndarray, active: Sequence[int], radius: float) -> list[int]:
    """Arms of ``active`` that survive one elimination step at ``radius``."""
    best = max(abs_gradients[i] for i in active)
    threshold = best - 2.0 * radius
    return [i for i in active if abs_gradients[i] >= threshold]


def elimination_schedule(
    abs_gradients: np.ndarray,
    *,
    initial_radius: float | None = None,
    shrink: float = DEFAULT_SHRINK,
    minimum_radius: float = MINIMUM_RADIUS,
    maximum_rounds: int = MAXIMUM_ROUNDS,
) -> Iterator[EliminationRound]:
    """Yield successive-elimination rounds until one arm survives.

    The radius starts wide enough that no arm is eliminated for free and is
    multiplied by ``shrink`` each round.  Ties never separate, so the iteration
    also stops at ``minimum_radius`` or ``maximum_rounds`` and the caller should
    check how many arms remain.
    """
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must lie in (0, 1)")
    abs_gradients = np.asarray(abs_gradients, dtype=float)
    active = tuple(range(abs_gradients.size))
    radius = (
        float(abs_gradients.max()) if initial_radius is None else float(initial_radius)
    )
    if radius <= 0.0:
        radius = 1.0

    for index in range(maximum_rounds):
        if len(active) <= 1 or radius < minimum_radius:
            return
        remaining = survivors(abs_gradients, active, radius)
        eliminated = tuple(i for i in active if i not in set(remaining))
        round_result = EliminationRound(index, radius, active, eliminated)
        yield round_result
        active = round_result.active_after
        radius *= shrink
