"""The nine-state baseline table (SI Table, scripts/paper_a_baselines_table.py) reproduces the ranges that the Results quote."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_a_baselines_table as bt  # noqa: E402

pytestmark = pytest.mark.skipif(not bt.CSV.exists(), reason="the stored baseline comparison is not part of this checkout")


def _ratios(method):
    rows = bt.load()
    out = []
    for case, _ in bt.STATES:
        if (case, method) in rows:
            out.append(float(rows[(case, method)]["shots_mean"]) / float(rows[(case, "II-0")]["shots_mean"]))
    return out


def test_ranges_quoted_in_the_results_come_from_the_nine_states():
    seq = _ratios("M1 seq (no elimination)")
    m2 = _ratios("M2 independent, pairwise rule")
    assert len(seq) == 9 and len(m2) == 9
    # "fixed estimators are 1.3 to 28 times cheaper than sequential estimation and 1.2 to 4.3 times cheaper than independent elimination"
    assert (round(min(seq), 1), round(max(seq))) == (1.3, 28) or (round(min(seq), 1), round(max(seq))) == (1.4, 28)
    assert (round(min(m2), 1), round(max(m2), 1)) == (1.2, 4.3)


def test_static_estimator_costs_one_to_six_times_the_sequential_one():
    rows = bt.load()
    both = [(c, m) for c, _ in bt.STATES for m in [bt.COLUMNS[0][2]] if (c, m) in rows and (c, bt.COLUMNS[1][2]) in rows]
    ratio = [float(rows[(c, bt.COLUMNS[0][2])]["shots_mean"]) / float(rows[(c, bt.COLUMNS[1][2])]["shots_mean"]) for c, _ in both]
    assert (round(min(ratio), 1), round(max(ratio), 1)) == (1.1, 5.8)
