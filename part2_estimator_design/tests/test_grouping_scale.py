"""Critique-round-2 options: the coefficient-mass grouping (part1_bridge.apply_grouping), the first-radius scale and the default invariance of the labels."""
import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baselines import IndependentConfig  # noqa: E402
from learning import LearningConfig  # noqa: E402


def test_defaults_keep_the_labels_and_the_scale_enters_the_label():
    assert "start_scale" not in LearningConfig(start="bound").label
    assert "start_scale=5" in LearningConfig(start="bound", start_scale=5.0).label
    assert "crossfit" not in LearningConfig().label
    assert IndependentConfig(rule="safe", start="bound", start_scale=2.0).label != IndependentConfig(rule="safe", start="bound").label


def test_mass_grouping_is_a_valid_partition_with_commuting_groups():
    import part1_bridge as bridge

    try:
        problem = bridge.load_problem("H4_square_eq_side1p0_CISD")
    except Exception as error:  # the cached problem is not part of every checkout
        pytest.skip(f"problem not available: {error}")
    bridge.apply_grouping(problem, "mass")
    groups = problem.parent_fc_groups()
    flat = [p for g in groups for p in g]
    assert sorted(flat) == sorted(problem.universal_support) and len(flat) == len(set(flat))

    def commute(a, b):
        return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2 == 0

    assert all(commute(a, b) for g in groups for a, b in combinations(g, 2))
    for terms, arm_groups in zip(problem.commutator_terms, problem.individual_fc_groups()):
        assert sorted(p for g in arm_groups for p in g) == sorted(terms)
    # sorted insertion by mass puts the Pauli of the largest pooled coefficient mass into the first group
    mass = {}
    for terms in problem.commutator_terms:
        for label, value in terms.items():
            mass[label] = mass.get(label, 0.0) + abs(value)
    assert max(mass, key=lambda s: (mass[s], [-ord(c) for c in s])) in groups[0] or mass[groups[0][0]] == max(mass.values())
