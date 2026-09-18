"""M1: FC-UG -- Fully Commuting Universal Groups.

One global fully commuting grouping of the parent support
``B_0 = union_i supp(C_i)`` is built once.  Measuring those contexts estimates
the shared Pauli expectations and reconstructs the *entire* gradient vector;
there is no candidate elimination.  The statistical target is therefore
full-vector estimation: every gradient must reach the same confidence radius.

Shots are spread over contexts with :func:`shot_models.allocate_context_shots`,
so the reported total is the number of circuit executions needed for every
gradient in the pool to satisfy the target simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gradients import GradientProblem
from io_utils import GROUPING_CONVENTIONS, environment_record, write_csv, write_json
from shot_models import (
    DEFAULT_DELTA,
    allocate_context_shots,
    confidence_radius_from_gap,
    epsilon_from_radius,
    reconstruction_variances,
    z_for_selection_error,
    z_from_delta,
)

METHOD_CODE = "M1_FCUG"
METHOD_NAME = "M1: FC-UG"

# How tight a common confidence radius M1 must reach, as gap / divisor.
#
# "identification" is the task-matched target: a single radius shared by every
# generator separates the leader from the field exactly when 2r < gap, so r =
# gap/2 is the loosest radius at which static full-vector estimation answers the
# same question M2 and M3 answer.  It is also the radius the BAI methods resolve
# their own leader to, so all three methods are compared at one criterion.
#
# "full_vector" is the legacy planning target of earlier revisions.  It is a
# strictly harder task -- four times the cost -- and is reported alongside rather
# than as the headline.
TARGETS = {"identification": 2.0, "full_vector": 4.0}
DEFAULT_TARGET = "identification"


@dataclass
class M1Result:
    """Outcome of one M1 run, with every intermediate object kept."""

    case_id: str
    state_name: str
    target: str
    radius: float
    epsilon: float
    z: float
    delta: float
    groups: list[list[str]]
    sigmas: np.ndarray = field(repr=False)
    shots_per_context: np.ndarray = field(repr=False)

    @property
    def total_shots(self) -> int:
        return int(np.ceil(self.shots_per_context.sum()))

    @property
    def n_contexts(self) -> int:
        return len(self.groups)

    @property
    def n_active_contexts(self) -> int:
        return int((self.shots_per_context > 0).sum())

    def summary(self) -> dict:
        variances = reconstruction_variances(self.sigmas, self.shots_per_context)
        return {
            "method_code": METHOD_CODE,
            "method_name": METHOD_NAME,
            "case_id": self.case_id,
            "state": self.state_name,
            "target": self.target,
            "target_description": (
                "one common confidence radius over the whole pool, tight enough to "
                "identify the largest-gradient generator"
                if self.target == "identification"
                else "full gradient vector estimated to a common confidence radius"
            ),
            "confidence_radius": self.radius,
            "epsilon_one_standard_error": self.epsilon,
            "z": self.z,
            "delta_family_wise": self.delta,
            "n_parent_fc_contexts": self.n_contexts,
            "n_contexts_receiving_shots": self.n_active_contexts,
            "total_context_shots": self.total_shots,
            "max_reconstruction_std": float(np.sqrt(variances.max())) if variances.size else 0.0,
            "constraint_satisfied": bool(
                variances.size == 0 or np.sqrt(variances.max()) <= self.epsilon * (1 + 1e-9)
            ),
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
        rows = (
            {
                "generator_index": i,
                "label": labels[i],
                "group_id": alpha,
                "sigma": float(self.sigmas[i, alpha]),
            }
            for i, alpha in zip(*np.nonzero(self.sigmas))
        )
        paths.append(
            write_csv(
                directory / f"{stem}_fragment_sigmas.csv",
                ["generator_index", "label", "group_id", "sigma"],
                rows,
            )
        )
        return paths


def run(
    problem: GradientProblem,
    *,
    target: str = DEFAULT_TARGET,
    radius: float | None = None,
    gap_divisor: float | None = None,
    delta: float = DEFAULT_DELTA,
    calibration: str = "bonferroni",
) -> M1Result:
    """Estimate the M1 context-shot cost for one gradient problem.

    ``target`` selects the common confidence radius from :data:`TARGETS`; pass
    ``gap_divisor`` to sweep it directly, or ``radius`` to set it outright.

    Every generator in the pool is resolved to the same radius, including those
    whose gradient turns out to be zero.  That is not an artefact: M1 is
    non-adaptive, so it cannot know in advance which generators are irrelevant,
    and the cost of bounding a high-variance zero-gradient operator is exactly the
    inefficiency that candidate elimination removes.
    """
    groups = problem.parent_fc_groups()
    sigmas = problem.parent_fragment_sigmas()
    if radius is None:
        divisor = gap_divisor if gap_divisor is not None else TARGETS[target]
        target_radius = confidence_radius_from_gap(problem.top_gap(), divisor)
    else:
        target_radius = radius
    z = _confidence_factor(delta, problem.n_generators, calibration)
    epsilon = epsilon_from_radius(target_radius, z)
    shots = allocate_context_shots(sigmas, epsilon)
    return M1Result(
        case_id=problem.case_id,
        state_name=problem.state_name,
        target=target if radius is None and gap_divisor is None else "custom",
        radius=target_radius,
        epsilon=epsilon,
        z=z,
        delta=delta,
        groups=groups,
        sigmas=sigmas,
        shots_per_context=shots,
    )


def _confidence_factor(delta: float, n_arms: int, calibration: str) -> float:
    """Bonferroni over the pool, or calibrated to the realised selection error."""
    if calibration == "bonferroni":
        return z_from_delta(delta, n_arms)
    if calibration == "selection":
        return z_for_selection_error(delta)
    raise ValueError("calibration must be 'bonferroni' or 'selection'")
