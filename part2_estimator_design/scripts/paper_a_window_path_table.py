#!/usr/bin/env python3
"""Paper A, SI: certified II-A on states along the exact ADAPT path of the 16-qubit cc-pVDZ window of H2O (jobs win_sel_*, 2026-10-19/20).

Reads ``runs/H2O_dz8o_{eq,str}_ADAPT{5,10,20,30,40}/*_step5_external_baselines.csv`` (II-A, a priori start, sign-aware, rho 0.1, 16 trials)
and writes the tabular body ``reports/generated/tab_window_path.tex`` (caption and note stand in ``si_parts/s_window_path.tex``).

    python scripts/paper_a_window_path_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "reports" / "generated" / "tab_window_path.tex"
GEOMETRIES = [("eq", "equilibrium"), ("str", "stretched")]
STATES = (5, 10, 20, 30, 40)


def sci(x: float) -> str:
    e = len(str(int(x))) - 1
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


def main() -> None:
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Geometry & Iteration $k$ & Selection shots (median) & Within $\rho$ & Best & Design and statistics (h) & Trials \\", r"\midrule"]
    for g, (key, name) in enumerate(GEOMETRIES):
        for i, k in enumerate(STATES):
            case = f"H2O_dz8o_{key}_ADAPT{k}"
            path = ROOT / "runs" / case / f"{case}_step5_external_baselines.csv"
            rows = [r for r in csv.DictReader(path.open()) if r["config"].startswith("II-A")]
            r = rows[0]
            hours = (float(r["design_seconds_mean"]) + float(r["statistics_seconds_mean"])) / 3600
            lines.append(f"{name if i == 0 else ''} & {k} & {sci(float(r['shots_median']))} & {100 * float(r['good_10pct_rate']):.0f}\\% & "
                         f"{100 * float(r['correct_rate']):.0f}\\% & {hours:.1f} & {r['n_trials']} \\\\")
        if g < len(GEOMETRIES) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
