"""The grouping follow-up tables (covariance-aware grouping, oracle optimum, mass-grouping trajectories, radius scale) agree with the stored csv files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "paper_a"
GEN = ROOT.parent / "reports" / "generated"

pytestmark = pytest.mark.skipif(not (RUNS / "vaware_hf.csv").exists() or not (GEN / "tab_vaware.tex").exists(), reason="analysis outputs not generated")


def rows(name: str) -> list[str]:
    return [r for r in (GEN / f"{name}.tex").read_text(encoding="utf-8").splitlines() if r.endswith(r"\\") and "&" in r][1:]  # the first row is the header


def test_vaware_table_matches_csv():
    t = pd.read_csv(RUNS / "vaware_hf.csv")
    body = rows("tab_vaware")
    assert len(body) == len(t) == 7
    for line, (_, r) in zip(body, t.iterrows()):
        assert f"{r['II-A_ratio']:.2f} [{r['II-A_lo']:.2f}, {r['II-A_hi']:.2f}]" in line
    # the statement of the paper: the learned estimators gain on LiH and lose on H4
    lih, h4 = t[t.case.str.startswith("LiH")], t[t.case.str.startswith("H4")]
    assert lih["II-A_ratio"].max() < 1 and h4["II-A_ratio"].min() > 1
    assert (t["II-A_correct"] == 1).all()


def test_oracle_and_radius_ranges_are_those_of_the_text():
    o = pd.read_csv(RUNS / "oracle_grouping.csv")
    assert len(o) == 21
    assert round(o.best_over_weight.min(), 2) == 0.39 and round(o.best_over_weight.max(), 2) == 0.84
    assert round(o.vaware_hf_over_weight.min(), 2) == 0.40 and round(o.vaware_hf_over_weight.max(), 2) == 0.86
    r = pd.read_csv(RUNS / "r0_scale.csv")
    assert len(r) == 81 and round(r.ratio.min(), 2) == 0.90 and round(r.ratio.max(), 2) == 1.17
    assert (r.exact_best == 1).all()
    assert len(rows("tab_r0scale")) == 27
