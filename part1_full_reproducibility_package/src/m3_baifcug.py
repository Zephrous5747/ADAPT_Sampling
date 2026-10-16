"""M3: BAI-FC-UG -- Best-Arm Identification with Fully Commuting Universal Groups.

M3 is the intended combined method: it starts from exactly the same fixed parent
fully commuting contexts as M1, but eliminates generator candidates the way M2
does.  The parent contexts are never rebuilt during BAI.  At round ``r`` the
active Pauli support shrinks to ``B_r = union_{i in A_r} supp(C_i)`` and each
parent context is used in its restricted form ``M_alpha and B_r``, so every shot
taken in an earlier round stays compatible with later rounds.

Cost accounting: because the contexts are nested and the samples are reusable,
the shots a context has ever needed is the largest requirement it saw in any
round, and the reported total is the sum of those per-context maxima.  A context
that only distinguishes eliminated generators simply stops growing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bai import DEFAULT_SHRINK, elimination_schedule, elimination_thresholds
from gradients import GradientProblem
from io_utils import GROUPING_CONVENTIONS, environment_record, write_csv, write_json
from shot_models import (
    DEFAULT_DELTA,
    allocate_context_shots,
    epsilon_from_radius,
    z_for_selection_error,
    z_from_delta,
)

METHOD_CODE = "M3_BAIFCUG"
METHOD_NAME = "M3: BAI-FC-UG"


@dataclass
class M3Result:
    """Outcome of one M3 run."""

    case_id: str
    state_name: str
    z: float
    delta: float
    groups: list[list[str]] = field(repr=False)
    sigmas: np.ndarray = field(repr=False)
    shots_per_context: np.ndarray = field(repr=False)
    history: list[dict] = field(repr=False)
    remaining: tuple[int, ...] = ()

    @property
    def total_shots(self) -> int:
        return int(np.ceil(self.shots_per_context.sum()))

    @property
    def n_contexts(self) -> int:
        return len(self.groups)

    def summary(self) -> dict:
        return {
            "method_code": METHOD_CODE,
            "method_name": METHOD_NAME,
            "case_id": self.case_id,
            "state": self.state_name,
            "target": "winner identification on fixed parent FC universal contexts",
            "z": self.z,
            "delta_family_wise": self.delta,
            "n_parent_fc_contexts": self.n_contexts,
            "n_contexts_ever_sampled": int((self.shots_per_context > 0).sum()),
            "n_rounds": len(self.history),
            "n_remaining_arms": len(self.remaining),
            "selected_arms": list(self.remaining),
            "final_radius": self.history[-1]["radius"] if self.history else None,
            "total_context_shots": self.total_shots,
            "conventions": GROUPING_CONVENTIONS,
            "environment": environment_record(),
        }

    def save(self, directory: Path, labels: list[str]) -> list[Path]:
        directory = Path(directory)
        stem = f"{self.case_id}_{METHOD_CODE}"
        paths = [write_json(directory / f"{stem}_summary.json", self.summary())]
        paths.append(
            write_csv(
                directory / f"{stem}_contexts.csv",
                ["group_id", "n_terms", "envelope_sigma", "shots", "terms_preview"],
                (
                    {
                        "group_id": index,
                        "n_terms": len(group),
                        "envelope_sigma": float(self.sigmas[:, index].max()),
                        "shots": float(self.shots_per_context[index]),
                        "terms_preview": ";".join(group[:8]),
                    }
                    for index, group in enumerate(self.groups)
                ),
            )
        )
        paths.append(
            write_csv(
                directory / f"{stem}_fragment_sigmas.csv",
                ["generator_index", "label", "group_id", "sigma"],
                (
                    {
                        "generator_index": i,
                        "label": labels[i],
                        "group_id": alpha,
                        "sigma": float(self.sigmas[i, alpha]),
                    }
                    for i, alpha in zip(*np.nonzero(self.sigmas))
                ),
            )
        )
        paths.append(
            write_csv(
                directory / f"{stem}_bai_history.csv",
                [
                    "round",
                    "radius",
                    "n_active",
                    "n_active_contexts",
                    "n_eliminated",
                    "eliminated",
                    "cumulative_shots",
                ],
                self.history,
            )
        )
        return paths


def run(
    problem: GradientProblem,
    *,
    delta: float = DEFAULT_DELTA,
    calibration: str = "bonferroni",
    schedule: str = "exact",
    shrink: float = DEFAULT_SHRINK,
) -> M3Result:
    """Estimate the M3 context-shot cost for one gradient problem.

    ``schedule="exact"`` (the default) evaluates the shared-context allocation at
    the finitely many radii where the active set actually changes, which is the
    limit of a geometric schedule as its shrink factor approaches one.  Besides
    removing the arbitrary shrink parameter, it restores the ordering M3 <= M1
    that the method guarantees by construction: every active set is a subset of
    M1's, at a radius no tighter than M1's.  A coarse geometric schedule
    overshoots the elimination points and can violate that ordering.
    """
    if schedule not in ("exact", "geometric"):
        raise ValueError("schedule must be 'exact' or 'geometric'")
    groups = problem.parent_fc_groups()
    sigmas = problem.parent_fragment_sigmas()
    z = _confidence_factor(delta, problem.n_generators, calibration)

    shots_per_context = np.zeros(len(groups))
    history: list[dict] = []
    active: tuple[int, ...] = tuple(range(problem.n_generators))

    if schedule == "exact":
        for threshold in elimination_thresholds(problem.abs_gradients):
            requested = allocate_context_shots(
                sigmas[list(threshold.active), :],
                epsilon_from_radius(threshold.radius, z),
            )
            shots_per_context = np.maximum(shots_per_context, requested)
            history.append(
                {
                    "round": threshold.index,
                    "radius": threshold.radius,
                    "n_active": len(threshold.active),
                    "n_active_contexts": int((requested > 0).sum()),
                    "n_eliminated": len(threshold.eliminated),
                    "eliminated": " ".join(str(i) for i in threshold.eliminated),
                    "cumulative_shots": float(shots_per_context.sum()),
                }
            )
        leader = int(np.argmax(problem.abs_gradients))
        return M3Result(
            case_id=problem.case_id,
            state_name=problem.state_name,
            z=z,
            delta=delta,
            groups=groups,
            sigmas=sigmas,
            shots_per_context=shots_per_context,
            history=history,
            remaining=(leader,),
        )

    for round_result in elimination_schedule(problem.abs_gradients, shrink=shrink):
        active = round_result.active_before
        epsilon = epsilon_from_radius(round_result.radius, z)
        requested = allocate_context_shots(sigmas[list(active), :], epsilon)
        shots_per_context = np.maximum(shots_per_context, requested)
        history.append(
            {
                "round": round_result.index,
                "radius": round_result.radius,
                "n_active": len(active),
                "n_active_contexts": int((requested > 0).sum()),
                "n_eliminated": len(round_result.eliminated),
                "eliminated": " ".join(str(i) for i in round_result.eliminated),
                "cumulative_shots": float(shots_per_context.sum()),
            }
        )
        active = round_result.active_after

    return M3Result(
        case_id=problem.case_id,
        state_name=problem.state_name,
        z=z,
        delta=delta,
        groups=groups,
        sigmas=sigmas,
        shots_per_context=shots_per_context,
        history=history,
        remaining=active,
    )


def _confidence_factor(delta: float, n_arms: int, calibration: str) -> float:
    """Bonferroni over the pool, or calibrated to the realised selection error."""
    if calibration == "bonferroni":
        return z_from_delta(delta, n_arms)
    if calibration == "selection":
        return z_for_selection_error(delta)
    raise ValueError("calibration must be 'bonferroni' or 'selection'")


def run_actual_radii(problem: "GradientProblem", *, delta: float = 0.05,
                     bisection_steps: int = 80) -> dict:
    """M3 in the schedule-free limit, eliminating on each arm's *actual* precision.

    The planning bound of :func:`run` resolves every active arm to the common
    radius ``r`` and removes arm ``i`` once ``2r < Delta_i``.  But a shared
    allocation is sized by its binding arm, so every other arm is resolved more
    tightly than required, and the parent contexts keep every shot already taken,
    so shots spent separating arms that have since left keep sharpening the
    survivors.  A run that eliminates on each arm's actual reconstruction variance
    ``V_i = sum_a sigma_ia**2 / n_a`` -- as any real run computing its own
    variances would -- drops arm ``i`` as soon as some ``j`` satisfies

        |g_j| - z sqrt(V_j) > |g_i| + z sqrt(V_i).

    With ``n(r) = max(n_so_far, allocation(A, r))`` every ``V_i`` is non-increasing
    as ``r`` falls, so the dominance margin is monotone in ``r`` and the radius of
    the next elimination is found by bisection.  The trajectory is followed event by
    event with no schedule, and cost is the per-context maximum of Eq. (14), which
    is the running ``n`` itself.  This is the gamma -> 1 limit of the noiseless
    finite-shot trial.
    """
    from shot_models import allocate_context_shots, epsilon_from_radius, z_from_delta

    z = z_from_delta(delta, problem.n_generators)
    sigmas = problem.parent_fragment_sigmas()
    sq = sigmas ** 2
    absg = problem.abs_gradients
    active = list(range(problem.n_generators))
    shots = np.zeros(sigmas.shape[1])
    events = []

    def state(A, shape, r):
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.maximum(shots, shape / epsilon_from_radius(r, z) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(sq[A] > 0, sq[A] / n, 0.0).sum(axis=1)
        half = z * np.sqrt(v)
        g = absg[A]
        best = np.max(g - half)
        margin = best - (g + half)          # > 0 : dominated
        return n, margin

    upper = 1e3 * float(absg.max())
    while len(active) > 1:
        shape = allocate_context_shots(sigmas[active, :], 1.0)
        _, m_hi = state(active, shape, upper)
        if m_hi.max() > 0:                  # already dominated at the current radius
            r_star = upper
        else:
            lo, hi = 0.0, upper
            _, m_lo = state(active, shape, max(lo, 1e-300))
            if m_lo.max() <= 0:             # exact ties never separate
                break
            for _ in range(bisection_steps):
                mid = 0.5 * (lo + hi)
                _, m = state(active, shape, mid)
                if m.max() > 0:
                    lo = mid
                else:
                    hi = mid
            r_star = lo
        n, margin = state(active, shape, r_star)
        shots = n
        gone = [active[k] for k in np.flatnonzero(margin > 0)]
        events.append({"radius": r_star, "n_active": len(active), "eliminated": gone,
                       "cumulative": float(shots.sum())})
        upper = r_star
        active = [i for i in active if i not in gone]
    return {"total_shots": float(shots.sum()), "events": events, "survivors": active}
