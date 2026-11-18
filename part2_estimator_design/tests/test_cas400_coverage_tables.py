"""The SI tables of the CAS(8e,8o) states at 400 shots and of the coverage test (paper_a_referee_tables.tab_window, paper_a_coverage_summary.latex_table)
agree with the stored per-trial summaries that the main text quotes."""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_a_coverage_summary as cov  # noqa: E402
import paper_a_referee_tables as rt  # noqa: E402

PAPER_A = Path(__file__).resolve().parents[1] / "runs" / "paper_a"
needs_runs = pytest.mark.skipif(not (PAPER_A / "coverage_summary.csv").exists() or not (PAPER_A / "cas400_facts.json").exists(),
                                reason="the stored coverage and CAS(8e,8o) summaries are not part of this checkout")


@needs_runs
def test_coverage_table_totals_are_the_sums_of_its_cells():
    table = pd.read_csv(PAPER_A / "coverage_summary.csv")
    text = cov.latex_table(table)
    last = [line for line in text.splitlines() if line.startswith("All states")][0]
    for method in cov.METHODS:
        cells = table[table.method == method]
        assert f"{int(cells.wrong.sum())} of {int(cells.trials.sum()):,}".replace(",", "{,}") in last
    # one row per state, the template state carries the footnote mark, and no cell shows a p-value
    assert text.count(r"\textsuperscript{a}") == 2
    assert "p<" not in text and "p-value" not in text


@needs_runs
def test_window_table_reproduces_the_factors_of_the_summary(tmp_path):
    rt.tab_window(tmp_path)
    rows = [line.split("&") for line in (tmp_path / "tab_window.tex").read_text(encoding="utf-8").splitlines() if line.startswith("H$_2$O") and " & II-A & " in line]
    assert len(rows) == 4
    facts = json.loads((PAPER_A / "cas400_facts.json").read_text(encoding="utf-8"))
    for row, state in zip(rows, ("eq CISD", "eq ADAPT10", "str CISD", "str ADAPT10")):
        factor = float(re.sub(r"[^0-9.]", "", row[-1].replace(r"\\", "")))
        assert abs(factor - facts[state]["IIA_400_over_50"][0]) < 0.006
        assert row[3].strip() == "1.00" and row[8].strip() == "1.00"  # II-A over itself at both minimums
