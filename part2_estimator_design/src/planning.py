"""Oracle planning bounds for the Part II designs (Step 3).

These are Part I's planning bounds with Part I's fragment standard deviations
replaced by those of an optimised design.  Everything is oracle: exact gradients
decide the elimination trajectory and exact covariances drive the design, so the
numbers are *ceilings* on what a level could deliver, not estimates of what a
non-oracle run will realise.

Three trajectories are provided.

``m1``
    one allocation resolving every generator to the common radius ``gap/2``.
``m3_common_radius``
    Part I's exact-limit M3: the allocation at each breakpoint of
    :func:`bai.elimination_thresholds`, with every active arm charged the common
    radius.  ``redesign=False`` optimises one design for the full pool and keeps
    it; ``redesign=True`` re-optimises the design for the surviving arms at every
    breakpoint (oracle II-D).  ``accounting="maxima"`` is Part I's convention, each
    breakpoint solved from scratch and each context charged its largest request;
    ``accounting="topup"`` solves each breakpoint above the shots already held.
``actual_precision``
    Part I's event-driven M3 that eliminates on each arm's actual variance
    (:func:`m3_baifcug.run_actual_radii`), for a fixed design or with the design
    re-optimised at every event.

Redesigning is legitimate at every stage because the coefficients are
post-processing: shots are taken in contexts, and any design satisfying
``A = BC`` can be applied to all of them afterwards.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import part1_bridge  # noqa: F401  (Part I's src on the path)
from allocation import allocate
from bai import elimination_thresholds  # Part I
from part1_bridge import epsilon_from_radius

ACCOUNTINGS = ("maxima", "topup")


def m1(design, arms: Sequence[int], radius: float, z: float):
    """Static full-vector estimation to the common ``radius``."""
    return design.solve(list(arms), epsilon_from_radius(radius, z))


def m3_common_radius(
    design,
    abs_gradients: np.ndarray,
    z: float,
    *,
    redesign: bool = False,
    accounting: str = "maxima",
    static_solution=None,
) -> dict:
    """Part I's exact-limit M3 trajectory on a design (see module docstring)."""
    if accounting not in ACCOUNTINGS:
        raise ValueError(f"accounting must be one of {ACCOUNTINGS}")
    n_arms = len(design.problems)
    committed = np.zeros(design.n_contexts)
    history = []
    sigmas = None
    if not redesign:
        if static_solution is None:
            design.solve(range(n_arms), 1.0)
        sigmas = design.sigmas(range(n_arms))
    for threshold in elimination_thresholds(abs_gradients):
        epsilon = epsilon_from_radius(threshold.radius, z)
        arms = list(threshold.active)
        lower = committed if accounting == "topup" else None
        if redesign:
            solution = design.solve(arms, epsilon, lower)
            request = solution.shots
            iterations = solution.iterations
        else:
            request = allocate(sigmas[arms], epsilon, lower)
            iterations = 0
        committed = np.maximum(committed, request)
        history.append(
            {
                "round": threshold.index,
                "radius": threshold.radius,
                "n_active": len(arms),
                "n_eliminated": len(threshold.eliminated),
                "requested_shots": float(request.sum()),
                "cumulative_shots": float(committed.sum()),
                "design_iterations": iterations,
                "constraint_residual": design.max_constraint_residual(),
            }
        )
    return {"total": float(committed.sum()), "shots": committed, "history": history}


def actual_precision(
    abs_gradients: np.ndarray,
    z: float,
    *,
    sigmas: np.ndarray | None = None,
    design=None,
    bisection_steps: int = 80,
) -> dict:
    """Event-driven M3 eliminating on each arm's actual precision.

    A port of Part I's :func:`m3_baifcug.run_actual_radii` that takes the fragment
    standard deviations as input.  Give ``sigmas`` for a fixed design, or
    ``design`` to re-optimise the design for the surviving arms at every event;
    the variances used for elimination are then those of the current design
    applied to every shot taken so far.
    """
    if (sigmas is None) == (design is None):
        raise ValueError("give exactly one of sigmas and design")
    absg = np.asarray(abs_gradients, dtype=float)
    active = list(range(absg.size))
    n_contexts = sigmas.shape[1] if sigmas is not None else design.n_contexts
    shots = np.zeros(n_contexts)
    events = []

    def state(arms, sq, shape, radius):
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.maximum(shots, shape / epsilon_from_radius(radius, z) ** 2)
            v = np.where(sq[arms] > 0, sq[arms] / n, 0.0).sum(axis=1)
        half = z * np.sqrt(v)
        g = absg[arms]
        best = np.max(g - half)
        return n, best - (g + half)

    upper = 1e3 * float(absg.max())
    while len(active) > 1:
        if design is not None:
            solution = design.solve(active, 1.0)
            shape = solution.shots
            current = design.sigmas(range(absg.size))
        else:
            current = sigmas
            shape = allocate(current[active, :], 1.0)
        sq = current ** 2
        _, high_margin = state(active, sq, shape, upper)
        if high_margin.max() > 0:
            r_star = upper
        else:
            lo, hi = 0.0, upper
            _, low_margin = state(active, sq, shape, max(lo, 1e-300))
            if low_margin.max() <= 0:  # exact ties never separate
                break
            for _ in range(bisection_steps):
                mid = 0.5 * (lo + hi)
                _, margin = state(active, sq, shape, mid)
                if margin.max() > 0:
                    lo = mid
                else:
                    hi = mid
            r_star = lo
        n, margin = state(active, sq, shape, r_star)
        shots = n
        gone = [active[k] for k in np.flatnonzero(margin > 0)]
        events.append(
            {
                "radius": r_star,
                "n_active": len(active),
                "n_eliminated": len(gone),
                "cumulative_shots": float(shots.sum()),
            }
        )
        upper = r_star
        active = [i for i in active if i not in gone]
    return {"total": float(shots.sum()), "shots": shots, "events": events, "survivors": active}
