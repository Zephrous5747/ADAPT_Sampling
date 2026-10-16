"""Access to the validated Part I pipeline.

Part II does not re-derive any chemistry.  The Jordan-Wigner Hamiltonian and its
validation gate, the fixed HF and CISD states, the UCCSD pool, the commutator
expansions ``C_i = [H, G_i]`` and the parent fully commuting grouping all come
from ``part1_full_reproducibility_package/src``, so a Part II number and the Part I
number it is compared with always rest on the same objects.

Problems are cached in Part II's own ``.cache`` directory and built with the
current Part I code.  Part I's own cache predates its SCF-instability fix, so it is
deliberately not reused; :func:`check_against_part1` compares every freshly built
problem with the gradients Part I committed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PART2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PART2_ROOT.parent
PART1_ROOT = REPO_ROOT / "part1_full_reproducibility_package"
PART1_SRC = PART1_ROOT / "src"
PART1_RUNS = PART1_ROOT / "runs"
DEFAULT_CACHE = PART2_ROOT / ".cache"

if str(PART1_SRC) not in sys.path:
    sys.path.insert(0, str(PART1_SRC))

from cases import CASES, SMALL_CASES, get_case  # noqa: E402  (Part I)
from cases_extra import register as _register_extra_cases  # noqa: E402

_register_extra_cases(CASES)  # LiH at equilibrium, BeH2, H6, N2 (Paper A scale checks)
from cases_window import install as _install_window_cases  # noqa: E402

_install_window_cases(CASES)  # H2O in cc-pVDZ / 6-31G orbital windows (larger-basis checks)
from gradients import GradientProblem  # noqa: E402
from problem_cache import load_or_build  # noqa: E402
from shot_models import (  # noqa: E402
    DEFAULT_DELTA,
    allocate_context_shots,
    epsilon_from_radius,
    z_for_selection_error,
    z_from_delta,
)

CONFIDENCE_MODES = ("bonferroni", "selection")


def confidence_z(delta: float, n_arms: int, mode: str = "bonferroni") -> float:
    """The confidence factor of the radii.

    ``"bonferroni"`` is the paper's convention (a two-sided normal quantile at ``delta / 2K`` over the
    ``K`` gradients of the pool, one look); ``"selection"`` is Part I's ``z`` calibrated to the realised
    selection error (``z = z_{delta/2} / sqrt(2)``, independent of ``K``), about 2.2 times smaller at
    ``delta = 0.05`` and ``K = 26``: an upper bound on how much of a cost is the price of the radius rule.
    """
    if mode == "bonferroni":
        return z_from_delta(delta, n_arms)
    if mode == "selection":
        return z_for_selection_error(delta)
    raise ValueError(f"confidence must be one of {CONFIDENCE_MODES}")


__all__ = [
    "CASES",
    "CONFIDENCE_MODES",
    "SMALL_CASES",
    "DEFAULT_CACHE",
    "DEFAULT_DELTA",
    "GradientProblem",
    "allocate_context_shots",
    "check_against_part1",
    "confidence_z",
    "epsilon_from_radius",
    "get_case",
    "load_problem",
    "part1_summary",
    "z_from_delta",
]


def load_problem(
    case_id: str, cache_dir: Path | None = DEFAULT_CACHE, *, validate: bool = True
) -> GradientProblem:
    """Build (or reload) the Part I gradient problem for one case.

    ``<geometry>_ADAPT<k>`` names the state after ``k`` steps of the saved ADAPT
    trajectory from ``<geometry>_HF`` (:mod:`trajectory`).
    """
    from trajectory import split_case  # local: trajectory imports this module

    if "@" in case_id:  # another operator pool: <case>@<pool>
        from pools import load_pool_problem

        return load_pool_problem(case_id, cache_dir if cache_dir is not None else DEFAULT_CACHE)
    if split_case(case_id) is not None:
        from trajectory import load_trajectory_problem

        return load_trajectory_problem(case_id, cache_dir if cache_dir is not None else DEFAULT_CACHE)
    return load_or_build(get_case(case_id), cache_dir, validate=validate)


def part1_summary(case_id: str) -> dict:
    """The committed Part I summary JSON of one case."""
    path = PART1_RUNS / case_id / f"{case_id}_summary.json"
    return json.loads(path.read_text())


def part1_gradients(case_id: str) -> dict[str, float]:
    """The committed Part I gradients of one case, keyed by generator label."""
    path = PART1_RUNS / case_id / f"{case_id}_gradients.csv"
    with path.open() as handle:
        return {row["label"]: float(row["gradient"]) for row in csv.DictReader(handle)}


def check_against_part1(problem: GradientProblem) -> dict:
    """Compare a freshly built problem with the numbers Part I committed.

    Signed gradients are compared as they stand: the SCF-instability fix pins the
    orbital solution, so the signs should agree too.  The absolute comparison is
    reported separately so that a pure orbital-phase change, which leaves every
    statistic of Part I invariant, can be told apart from a real discrepancy.
    """
    reference = part1_gradients(problem.case_id)
    signed = max(
        abs(problem.gradients[i] - reference[label])
        for i, label in enumerate(problem.labels)
    )
    absolute = max(
        abs(abs(problem.gradients[i]) - abs(reference[label]))
        for i, label in enumerate(problem.labels)
    )
    summary = part1_summary(problem.case_id)
    return {
        "case_id": problem.case_id,
        "max_signed_gradient_difference": float(signed),
        "max_abs_gradient_difference": float(absolute),
        "state_energy_difference": float(
            abs(problem.state_energy - summary["state_energy_hartree"])
        ),
        "universal_terms_match": len(problem.universal_support)
        == summary["universal_commutator_pauli_terms"],
        "parent_contexts_match": len(problem.parent_fc_groups())
        == summary["parent_fc_contexts"],
    }


def gradient_matrix(problem: GradientProblem, n_library: int | None = None):
    """The coefficient matrix ``A`` (generators x library Paulis), CSR.

    Columns follow ``problem.universal_support``, which is the order the Part II
    library uses for its required Paulis; auxiliary columns, if any, follow and
    are identically zero by construction.
    """
    from scipy.sparse import csr_matrix

    support = problem.universal_support
    index = {label: position for position, label in enumerate(support)}
    rows, cols, data = [], [], []
    for i, terms in enumerate(problem.commutator_terms):
        for label, coefficient in terms.items():
            rows.append(i)
            cols.append(index[label])
            data.append(coefficient)
    width = len(support) if n_library is None else n_library
    return csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(problem.n_generators, width),
    )
