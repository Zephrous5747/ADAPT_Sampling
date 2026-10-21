"""The planning and learning tables of the Supporting Information are written by scripts/paper_a_si_tables.py from the stored runs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_a_si_tables as t  # noqa: E402

pytestmark = pytest.mark.skipif(not (t.RUNS / "H4_square_eq_side1p0_CISD").exists(), reason="stored runs are not part of this checkout")


def test_cost_formats():
    assert t.thou(1234567) == "1{,}234{,}567"
    assert t.cost(2.5e7) == "$2.50\\times10^{7}$"
    assert t.cost(59641.5) == "59{,}642"
    assert t.sci1(7.78e4) == "$7.8\\times10^{4}$" and t.sci1(1.19e6) == "$1.19\\times10^{6}$"


def test_every_table_is_written_and_has_a_body(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "OUT", tmp_path)
    for fn in (t.si_planning_costs, t.si_surrogate, t.si_sign, t.si_ceilings, t.si_learned_fixed, t.si_learned_states, t.si_rounds, t.si_trajectory):
        fn()
    names = {p.stem for p in tmp_path.glob("*.tex")}
    assert names == {"si_planning_costs", "si_surrogate", "si_sign", "si_ceilings", "si_learned_fixed", "si_learned_states", "si_rounds", "si_trajectory"}
    for p in tmp_path.glob("*.tex"):
        body = p.read_text(encoding="utf-8")
        assert body.startswith("\\begin{tabular}") and body.rstrip().endswith("\\end{tabular}"), p.name
        assert "nan" not in body, p.name


def test_the_ceilings_of_h4_and_lih_are_those_of_the_study():
    rows = t.ceiling_rows("LiH_R3p0_HF")  # the run with the redesign columns, not the later single-case run
    base = float(rows[("mass", "II-0")]["m3_actual_static"])
    assert round(100 * (float(rows[("mass", "II-A")]["m3_actual_static"]) / base - 1)) == -73
