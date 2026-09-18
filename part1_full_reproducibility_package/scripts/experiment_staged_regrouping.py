#!/usr/bin/env python3
"""Staged regrouping: M3's shared contexts for breadth, a fresh grouping for depth.

M3 fixes one fully commuting grouping of the whole parent support and never
rebuilds it, which keeps every shot compatible across the elimination but makes
each individual gradient more expensive to resolve than its own grouping would.
This experiment asks whether abandoning the parent contexts once the field has
narrowed pays for itself.

Phase one measures on the parent contexts while more than ``--switch-at``
generators are active.  Phase two builds a new grouping over the survivors'
union support and pays for it from scratch: those shots are taken in different
circuit settings and the earlier samples do not transfer.  Phase two is charged
the full cost of reaching the final radii, ignoring the head start phase one has
already provided, so the totals reported here are upper bounds on the real cost
of the procedure.

Results are discussed in the "An alternative: rebuild the grouping once the field
narrows" section of the Part I report.

Example::

    python scripts/experiment_staged_regrouping.py --case H4_square_eq_side1p0_HF \
        --switch-at 2 3 4 6 10 20 --cache .cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m2_baifcig
import m3_baifcug
from bai import elimination_thresholds
from cases import CASES, get_case
from gradients import GradientProblem
from pauli_fc import greedy_fc_groups
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA, allocate_context_shots, epsilon_from_radius, z_from_delta


def subset_fragment_sigmas(
    problem: GradientProblem, arms: list[int], groups: list[list[str]]
) -> np.ndarray:
    """Fragment standard deviations for a few generators against a new grouping."""
    index = {pauli: a for a, group in enumerate(groups) for pauli in group}
    sigmas = np.zeros((len(arms), len(groups)))
    for row, i in enumerate(arms):
        buckets: dict[int, dict[str, float]] = {}
        for pauli, coefficient in problem.commutator_terms[i].items():
            group_index = index.get(pauli)
            if group_index is not None:
                buckets.setdefault(group_index, {})[pauli] = coefficient
        for group_index, fragment in buckets.items():
            sigmas[row, group_index] = problem.evaluator.fragment_std(fragment)
    return sigmas


def staged_cost(
    problem: GradientProblem, parent_sigmas: np.ndarray, switch_at: int, z: float
) -> dict:
    """Cost of switching to a fresh grouping once at most ``switch_at`` arms remain."""
    thresholds = elimination_thresholds(problem.abs_gradients)
    parent_shots = np.zeros(parent_sigmas.shape[1])
    late = []
    for threshold in thresholds:
        if len(threshold.active) <= switch_at:
            late.append(threshold)
            continue
        parent_shots = np.maximum(
            parent_shots,
            allocate_context_shots(
                parent_sigmas[list(threshold.active), :],
                epsilon_from_radius(threshold.radius, z),
            ),
        )

    if not late:
        return {"switch_at": switch_at, "phase1": float(parent_shots.sum()),
                "phase2": 0.0, "new_groups": 0, "total": float(parent_shots.sum())}

    arms = sorted({i for threshold in late for i in threshold.active})
    support = sorted({p for i in arms for p in problem.commutator_terms[i]})
    groups = greedy_fc_groups(support)
    sigmas = subset_fragment_sigmas(problem, arms, groups)
    position = {arm: row for row, arm in enumerate(arms)}

    new_shots = np.zeros(len(groups))
    for threshold in late:
        rows = [position[i] for i in threshold.active]
        new_shots = np.maximum(
            new_shots,
            allocate_context_shots(sigmas[rows, :], epsilon_from_radius(threshold.radius, z)),
        )
    return {
        "switch_at": switch_at,
        "phase1": float(parent_shots.sum()),
        "phase2": float(new_shots.sum()),
        "new_groups": len(groups),
        "total": float(parent_shots.sum() + new_shots.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--switch-at", type=int, nargs="+", default=[2, 3, 4, 6, 10, 20])
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    problem = load_or_build(get_case(args.case), Path(args.cache) if args.cache else None)
    z = z_from_delta(args.delta, problem.n_generators)
    parent_sigmas = problem.parent_fragment_sigmas()

    m2 = m2_baifcig.run(problem, delta=args.delta).total_shots
    m3 = m3_baifcug.run(problem, delta=args.delta).total_shots
    baseline = min(m2, m3)
    print(f"{args.case}:  M2 = {m2:,}   M3 = {m3:,}   better of the two = {baseline:,}")
    print(f"{'switch at':>11}{'phase 1':>16}{'phase 2':>16}{'new groups':>12}{'total':>16}{'vs better':>11}")
    for switch_at in args.switch_at:
        row = staged_cost(problem, parent_sigmas, switch_at, z)
        print(f"{'|A|<=' + str(switch_at):>11}{row['phase1']:16,.0f}{row['phase2']:16,.0f}"
              f"{row['new_groups']:12d}{row['total']:16,.0f}{row['total'] / baseline:11.2f}")


if __name__ == "__main__":
    main()
