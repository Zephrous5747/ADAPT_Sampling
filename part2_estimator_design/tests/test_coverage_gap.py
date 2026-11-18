"""The coverage test of the confidence intervals (scripts/coverage_gap_states.py, scripts/paper_a_coverage_table.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_a_coverage_table as table  # noqa: E402
from coverage_gap_states import relative_gap  # noqa: E402


def test_upper_bound_is_the_rule_of_three_without_events_and_one_with_all():
    # no event in n trials: 1 - 0.05**(1/n), about 3/n
    assert abs(table.upper_bound(0, 1884) - 0.0016) < 1e-4
    assert abs(table.upper_bound(0, 40) - (1 - 0.05 ** (1 / 40))) < 1e-12
    assert table.upper_bound(5, 5) == 1.0
    # k events: the bound exceeds the observed rate and falls with n
    assert table.upper_bound(3, 100) > 0.03
    assert table.upper_bound(3, 1000) < table.upper_bound(3, 100)


def test_clopper_pearson_interval_contains_the_rate_and_is_exact_at_the_ends():
    lo, hi = table.clopper_pearson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = table.clopper_pearson(10, 20)
    assert lo < 0.5 < hi


def test_relative_gap_of_the_two_largest_absolute_gradients():
    g = np.array([0.1, -0.5, 0.4, 0.05])
    rel, first, second = relative_gap(g)
    assert (first, second) == (1, 2)
    assert abs(rel - 0.2) < 1e-12
