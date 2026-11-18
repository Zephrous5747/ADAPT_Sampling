#!/usr/bin/env python3
"""Paper A (SI rebuild, evidence block D1): what the shot-limited selections did to the ADAPT trajectories.

For every system and method of the measured trajectories (sign-aware rule, a priori first radius, rho = 0.1, default minimum of 50 shots per fold; the
first runs ``trajectories.csv`` of the folder ``runs/<case>/phase4/`` and the runs added to them, ``extra1_*``; for H2O the files ``II_*`` and ``extra1_*``):
the number of trajectories, how many took the number of iterations of exact ADAPT, how many needed more, the share of the selections that were the exact
best generator (ties included) and the final energy error.  The sensitivity runs (``sens_*``), the runs at other minima of shots (``minshots*``) and the
runs of the classical-prior study (``cp_*``) are not part of the set.

The Results (Sec. "The trajectory absorbs selection errors") quote "684 of the 690 measured trajectories on H4 and LiH took the iterations of exact ADAPT":
the set here is II-0, II-A and II-E (``II-A data, contrast``): H4 96, 96 and 48 per geometry, LiH 90, 90 and 30.

Outputs: ``runs/paper_a/trajectory_counts.csv`` and ``reports/generated/tab_trajectory_counts.tex`` (tabular only).

    python scripts/paper_a_trajectory_counts.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
GEN = ROOT.parent / "reports" / "generated"

# (case, label, iterations of exact ADAPT)
SYSTEMS = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0~\AA", 10), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0~\AA", 10), ("LiH_R3p0_HF", r"LiH 3.0~\AA", 6),
           ("H2O_eq_HF", r"H$_2$O eq", 17), ("H2O_stretch_HF", r"H$_2$O str", 18)]
METHODS = [("II-0 safe", "II-0"), ("II-A data, safe", "II-A"), ("II-A data, contrast", "II-E")]


def headline_files(case: str) -> list[Path]:
    out = []
    for f in sorted((RUNS / case / "phase4").glob("*_trajectories.csv")):
        short = f.name[len(case) + len("_phase4_"):]
        if short == "trajectories.csv" or short.startswith(("extra1", "II_")):
            out.append(f)
    return out


def table() -> list[dict]:
    rows = []
    for case, label, exact in SYSTEMS:
        frames = [pd.read_csv(f) for f in headline_files(case)]
        d = pd.concat([f for f in frames if len(f)])
        for method, short in METHODS:
            g = d[d.method == method]
            if not len(g):
                continue
            same = g[g.iterations == exact]
            rows.append({"system": label, "case": case, "exact_iterations": exact, "method": short, "trajectories": len(g), "same_iterations": len(same),
                         "more_iterations": int((g.iterations > exact).sum()), "fewer_iterations": int((g.iterations < exact).sum()),
                         "exact_best_share": float(g.exact_best_selections.sum() / g.iterations.sum()),
                         "final_error_min_mHa": float(same.final_error.min() * 1e3), "final_error_max_mHa": float(same.final_error.max() * 1e3)})
    return rows


def main() -> None:
    rows = table()
    out = RUNS / "paper_a" / "trajectory_counts.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"System (exact ADAPT: iterations) & Method & Trajectories & Same iterations & More iterations & Exact-best selections & Final error (mHa) \\", r"\midrule"]
    last = None
    for r in rows:
        name = f"{r['system']} ({r['exact_iterations']})" if r["system"] != last else ""
        last = r["system"]
        lo, hi = r["final_error_min_mHa"], r["final_error_max_mHa"]
        err = f"{lo:.2f}" if round(lo, 2) == round(hi, 2) else f"{lo:.2f}--{hi:.2f}"
        lines.append(f"{name} & {r['method']} & {r['trajectories']} & {r['same_iterations']} & {r['more_iterations']} & {100 * r['exact_best_share']:.1f}\\% & {err} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (GEN / "tab_trajectory_counts.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    n_all = sum(r["trajectories"] for r in rows if r["case"].startswith(("H4", "LiH")))
    n_same = sum(r["same_iterations"] for r in rows if r["case"].startswith(("H4", "LiH")))
    print(f"H4 and LiH: {n_same} of {n_all} trajectories took the iterations of exact ADAPT")
    h4 = [r["exact_best_share"] for r in rows if r["case"].startswith("H4")]
    print(f"exact-best share on H4: {100 * min(h4):.1f} to {100 * max(h4):.1f}%")
    for r in rows:
        print(f"  {r['case']:30s} {r['method']:5s} n={r['trajectories']:3d} same={r['same_iterations']:3d} more={r['more_iterations']} exact-best {100 * r['exact_best_share']:.1f}%")
    print("wrote", out, "and", GEN / "tab_trajectory_counts.tex")


if __name__ == "__main__":
    main()
