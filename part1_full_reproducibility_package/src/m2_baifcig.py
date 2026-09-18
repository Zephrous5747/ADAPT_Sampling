"""M2: BAI-FC-IG -- Best-Arm Identification with Fully Commuting Individual Gradients.

Each generator gradient is one arm of a best-arm-identification problem, and
each arm is measured on its own: ``supp(C_i)`` is grouped into fully commuting
contexts independently of every other gradient, so there is no cross-gradient
sharing.  The statistical target is winner identification rather than
full-vector estimation, so an arm only ever has to be resolved well enough to be
separated from the leader.

Cost accounting: measurements are reusable across rounds, so an arm that is
eliminated at radius ``r`` has paid ``(z * sum_alpha sigma_{i,alpha} / r)**2``
shots in total, the optimal independent-fragment allocation at its final radius.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bai import DEFAULT_SHRINK, elimination_schedule, elimination_thresholds, final_radii
from gradients import GradientProblem
from io_utils import GROUPING_CONVENTIONS, environment_record, write_csv, write_json
from shot_models import (
    DEFAULT_DELTA,
    epsilon_from_radius,
    z_for_selection_error,
    z_from_delta,
)

METHOD_CODE = "M2_BAIFCIG"
METHOD_NAME = "M2: BAI-FC-IG"


def arm_shots(sigma_sum: float, radius: float, z: float) -> float:
    """Shots for one arm measured as independent FC fragments at ``radius``."""
    epsilon = epsilon_from_radius(radius, z)
    return (sigma_sum / epsilon) ** 2


@dataclass
class M2Result:
    """Outcome of one M2 run."""

    case_id: str
    state_name: str
    z: float
    delta: float
    groups: list[list[list[str]]] = field(repr=False)
    sigmas: list[np.ndarray] = field(repr=False)
    final_radius: np.ndarray = field(repr=False)
    shots_per_arm: np.ndarray = field(repr=False)
    history: list[dict] = field(repr=False)
    remaining: tuple[int, ...] = ()

    @property
    def total_shots(self) -> int:
        return int(np.ceil(self.shots_per_arm.sum()))

    def summary(self) -> dict:
        return {
            "method_code": METHOD_CODE,
            "method_name": METHOD_NAME,
            "case_id": self.case_id,
            "state": self.state_name,
            "target": "winner identification, per-gradient FC grouping, no sharing",
            "z": self.z,
            "delta_family_wise": self.delta,
            "n_rounds": len(self.history),
            "n_remaining_arms": len(self.remaining),
            "selected_arms": list(self.remaining),
            "final_radius": float(self.final_radius.min()),
            "total_context_shots": self.total_shots,
            "conventions": GROUPING_CONVENTIONS,
            "environment": environment_record(),
        }

    def save(self, directory: Path, labels: list[str], abs_gradients: np.ndarray) -> list[Path]:
        directory = Path(directory)
        stem = f"{self.case_id}_{METHOD_CODE}"
        paths = [write_json(directory / f"{stem}_summary.json", self.summary())]
        paths.append(
            write_csv(
                directory / f"{stem}_groups.csv",
                ["generator_index", "label", "group_id", "n_terms", "sigma", "terms_preview"],
                (
                    {
                        "generator_index": i,
                        "label": labels[i],
                        "group_id": alpha,
                        "n_terms": len(group),
                        "sigma": float(self.sigmas[i][alpha]),
                        "terms_preview": ";".join(group[:8]),
                    }
                    for i, groups in enumerate(self.groups)
                    for alpha, group in enumerate(groups)
                ),
            )
        )
        paths.append(
            write_csv(
                directory / f"{stem}_arms.csv",
                [
                    "generator_index",
                    "label",
                    "abs_gradient",
                    "n_fc_groups",
                    "sigma_sum",
                    "final_radius",
                    "shots",
                ],
                (
                    {
                        "generator_index": i,
                        "label": labels[i],
                        "abs_gradient": float(abs_gradients[i]),
                        "n_fc_groups": len(self.groups[i]),
                        "sigma_sum": float(self.sigmas[i].sum()),
                        "final_radius": float(self.final_radius[i]),
                        "shots": float(self.shots_per_arm[i]),
                    }
                    for i in range(len(labels))
                ),
            )
        )
        paths.append(
            write_csv(
                directory / f"{stem}_bai_history.csv",
                ["round", "radius", "n_active", "n_eliminated", "eliminated", "cumulative_shots"],
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
) -> M2Result:
    """Estimate the M2 context-shot cost for one gradient problem.

    ``schedule="exact"`` (the default) uses the schedule-free limit: each arm is
    resolved to exactly half its absolute-gradient deficit, which is where a
    geometric schedule converges as its shrink factor approaches one.  It removes
    the arbitrary shrink parameter and is a closed form in the deficits and the
    fragment sums.  ``schedule="geometric"`` reproduces the earlier behaviour.
    """
    if schedule not in ("exact", "geometric"):
        raise ValueError("schedule must be 'exact' or 'geometric'")
    groups = problem.individual_fc_groups()
    sigmas = problem.individual_fragment_sigmas(groups)
    sigma_sums = np.array([sigma.sum() for sigma in sigmas])
    z = _confidence_factor(delta, problem.n_generators, calibration)
    abs_gradients = problem.abs_gradients

    history: list[dict] = []

    if schedule == "exact":
        final_radius = final_radii(abs_gradients)
        for threshold in elimination_thresholds(abs_gradients):
            history.append(
                {
                    "round": threshold.index,
                    "radius": threshold.radius,
                    "n_active": len(threshold.active),
                    "n_eliminated": len(threshold.eliminated),
                    "eliminated": " ".join(str(i) for i in threshold.eliminated),
                    "cumulative_shots": float(
                        _shots(
                            sigma_sums,
                            np.maximum(final_radius, threshold.radius),
                            z,
                        ).sum()
                    ),
                }
            )
        leader = int(np.argmax(abs_gradients))
        return M2Result(
            case_id=problem.case_id,
            state_name=problem.state_name,
            z=z,
            delta=delta,
            groups=groups,
            sigmas=sigmas,
            final_radius=final_radius,
            shots_per_arm=_shots(sigma_sums, final_radius, z),
            history=history,
            remaining=(leader,),
        )

    final_radius = np.full(problem.n_generators, np.inf)
    active: tuple[int, ...] = tuple(range(problem.n_generators))

    for round_result in elimination_schedule(abs_gradients, shrink=shrink):
        active = round_result.active_before
        for arm in active:
            final_radius[arm] = round_result.radius
        shots = _shots(sigma_sums, final_radius, z)
        history.append(
            {
                "round": round_result.index,
                "radius": round_result.radius,
                "n_active": len(active),
                "n_eliminated": len(round_result.eliminated),
                "eliminated": " ".join(str(i) for i in round_result.eliminated),
                "cumulative_shots": float(shots.sum()),
            }
        )
        active = round_result.active_after

    shots = _shots(sigma_sums, final_radius, z)
    return M2Result(
        case_id=problem.case_id,
        state_name=problem.state_name,
        z=z,
        delta=delta,
        groups=groups,
        sigmas=sigmas,
        final_radius=final_radius,
        shots_per_arm=shots,
        history=history,
        remaining=active,
    )


def _shots(sigma_sums: np.ndarray, radii: np.ndarray, z: float) -> np.ndarray:
    """Shots paid by every arm at its current final radius."""
    shots = np.zeros_like(sigma_sums)
    measured = np.isfinite(radii) & (radii > 0)
    shots[measured] = (sigma_sums[measured] * z / radii[measured]) ** 2
    return shots


def _confidence_factor(delta: float, n_arms: int, calibration: str) -> float:
    """Bonferroni over the pool, or calibrated to the realised selection error."""
    if calibration == "bonferroni":
        return z_from_delta(delta, n_arms)
    if calibration == "selection":
        return z_for_selection_error(delta)
    raise ValueError("calibration must be 'bonferroni' or 'selection'")
