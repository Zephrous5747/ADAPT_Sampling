"""Appendix baseline: no sharing, no elimination.

This is the fourth cell of the two-by-two the three methods span. M1 shares
measurement contexts but does not eliminate; M2 eliminates but does not share;
M3 does both. This baseline does neither: every generator is measured in its own
fully commuting grouping, and every one of them is resolved to the same common
radius.

It is deliberately outside the method hierarchy. Its purpose is twofold.

It makes the two mechanisms separately attributable. Comparing the baseline with
M1 isolates the effect of sharing with elimination held off in both; comparing it
with M2 isolates the effect of elimination with the grouping held fixed. The
product of those two against the baseline-to-M3 ratio measures whether the
mechanisms reinforce or substitute for one another.

It is also a sanity bound. Measuring every generator separately is always
available as a fallback, so an allocation over shared contexts that costs more
than this baseline on a pool of any size signals a defect rather than a discovery.
That is not a theorem -- on a very small pool the fragmentation penalty of a
shared grouping can exceed the benefit of sharing -- which is why it is reported
as a diagnostic rather than asserted as an invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gradients import GradientProblem
from io_utils import GROUPING_CONVENTIONS, environment_record, write_csv, write_json
from m1_fcug import DEFAULT_TARGET, TARGETS
from shot_models import (
    DEFAULT_DELTA,
    confidence_radius_from_gap,
    epsilon_from_radius,
    z_for_selection_error,
    z_from_delta,
)

METHOD_CODE = "NOSHARE"
METHOD_NAME = "Baseline: no sharing, no elimination"


@dataclass
class NoShareResult:
    """Outcome of one baseline run."""

    case_id: str
    state_name: str
    target: str
    radius: float
    z: float
    delta: float
    groups: list[list[list[str]]] = field(repr=False)
    sigmas: list[np.ndarray] = field(repr=False)
    shots_per_arm: np.ndarray = field(repr=False)

    @property
    def total_shots(self) -> int:
        return int(np.ceil(self.shots_per_arm.sum()))

    def summary(self) -> dict:
        return {
            "method_code": METHOD_CODE,
            "method_name": METHOD_NAME,
            "case_id": self.case_id,
            "state": self.state_name,
            "target": self.target,
            "target_description": (
                "every generator measured in its own FC grouping and resolved to "
                "one common radius; no shared contexts and no elimination"
            ),
            "confidence_radius": self.radius,
            "z": self.z,
            "delta_family_wise": self.delta,
            "n_generators": int(self.shots_per_arm.size),
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
                directory / f"{stem}_arms.csv",
                ["generator_index", "label", "abs_gradient", "n_fc_groups", "sigma_sum", "shots"],
                (
                    {
                        "generator_index": i,
                        "label": labels[i],
                        "abs_gradient": float(abs_gradients[i]),
                        "n_fc_groups": len(self.groups[i]),
                        "sigma_sum": float(self.sigmas[i].sum()),
                        "shots": float(self.shots_per_arm[i]),
                    }
                    for i in range(len(labels))
                ),
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
) -> NoShareResult:
    """Cost of resolving every generator separately to one common radius.

    The radius convention matches :mod:`m1_fcug` so that the two are directly
    comparable; only the grouping differs.
    """
    groups = problem.individual_fc_groups()
    sigmas = problem.individual_fragment_sigmas(groups)
    if radius is None:
        divisor = gap_divisor if gap_divisor is not None else TARGETS[target]
        target_radius = confidence_radius_from_gap(problem.top_gap(), divisor)
    else:
        target_radius = radius

    z = _confidence_factor(delta, problem.n_generators, calibration)
    epsilon = epsilon_from_radius(target_radius, z)
    shots = np.array([(sigma.sum() / epsilon) ** 2 for sigma in sigmas])
    return NoShareResult(
        case_id=problem.case_id,
        state_name=problem.state_name,
        target=target if radius is None and gap_divisor is None else "custom",
        radius=target_radius,
        z=z,
        delta=delta,
        groups=groups,
        sigmas=sigmas,
        shots_per_arm=shots,
    )


def _confidence_factor(delta: float, n_arms: int, calibration: str) -> float:
    """Bonferroni over the pool, or calibrated to the realised selection error."""
    if calibration == "bonferroni":
        return z_from_delta(delta, n_arms)
    if calibration == "selection":
        return z_for_selection_error(delta)
    raise ValueError("calibration must be 'bonferroni' or 'selection'")
