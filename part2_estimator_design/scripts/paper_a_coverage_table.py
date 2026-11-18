#!/usr/bin/env python3
"""Paper A: coverage of the selection on states with a set gap (round-1 review, D5).

Reads the trials of ``scripts/step5_external_baselines.py`` on the synthetic-gap cases written by
``scripts/coverage_gap_states.py`` (``<geometry>_<tag>_ADAPT<j>``) and tabulates, per state and method:

* wrong selections out of the trials, with the exact one-sided 95 % upper bound of the error rate (as in the main text: 0 of 1884 -> 0.16 %);
* the share of trials in which some interval of some round missed the exact value (``miscovered_rounds > 0``), with its 95 % interval;
* the share of trials in which the best arm was eliminated (``best_eliminated``);
* the mean context-shots.

The guarantee that is tested is ``P(wrong) <= delta`` (0.05); the per-round intervals are expected to miss in a sizeable share of
the trials without a wrong selection (one look per round, a selection looks 60 to 90 times), the anytime intervals not at all.

    python scripts/paper_a_coverage_table.py --geometry H4_square_eq_side1p0_GAPS [--runs runs] [--latex ../reports/generated/tab_coverage_means.tex]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallel import read_trials  # noqa: E402

METHODS = [("II-A data, safe, bound start, min shots=400, diagnose", "II-A, 400"),
           ("II-0, safe, bound start, min shots=400, diagnose", "II-0, 400"),
           ("II-A data, safe, bound start, min shots=50, diagnose", "II-A, 50"),
           ("II-A data, safe, bound start, anytime, min shots=200, diagnose", "II-A anytime, 200")]


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def upper_bound(k: int, n: int, level: float = 0.95) -> float:
    """One-sided exact upper bound of a rate with k events in n trials (k = 0: 1 - (1 - level)^(1/n), the 'rule of three')."""
    return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))


def pct(x: float) -> str:
    return f"{100 * x:.2g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", nargs="+", default=["H4_square_eq_side1p0_GAPS"], help="<geometry>_<tag> of the state families")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--latex", type=Path, default=None)
    args = parser.parse_args()

    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"State & Method & Trials & Wrong & Upper bound (\%) & Interval missed (\%) & Best arm eliminated (\%) & Mean shots \\", r"\midrule"]
    for geometry in args.geometry:
        gaps = {}
        table = args.runs / "coverage_gap" / f"{geometry}_states.csv"
        if table.exists():
            with table.open() as handle:
                gaps = {r["case"]: r for r in csv.DictReader(handle)}
        cases = sorted((p.name for p in args.runs.glob(f"{geometry}_ADAPT*") if p.is_dir()), key=lambda c: int(c.rsplit("ADAPT", 1)[1]))
        for case in cases:
            trials = read_trials(args.runs, case, "step5")
            first = True
            for label, short in METHODS:
                rows = trials.get(label)
                if not rows:
                    continue
                n = len(rows)
                wrong = sum(1 - int(r["correct"]) for r in rows)
                missed = sum(int(float(r["miscovered_rounds"]) > 0) for r in rows)
                eliminated = sum(str(r["best_eliminated"]) == "True" for r in rows)
                shots = float(np.mean([float(r["shots"]) for r in rows]))
                upper = upper_bound(wrong, n)
                mlo, mhi = clopper_pearson(missed, n)
                gap = gaps.get(case, {}).get("achieved")
                name = f"{float(gap) * 100:.2g}\\% gap" if gap not in (None, "") else case.rsplit("_", 1)[-1]
                lines.append(f"{name if first else ''} & {short} & {n} & {wrong} & {pct(upper)} & {pct(missed / n)} [{pct(mlo)}, {pct(mhi)}] & "
                             f"{pct(eliminated / n)} & {shots:.3g} \\\\")
                print(f"{case:36s} {short:18s} n={n:5d} wrong {wrong:3d} (upper {100 * upper:.2f}%)  missed {100 * missed / n:5.1f}% [{100 * mlo:.1f}, {100 * mhi:.1f}]  "
                      f"best eliminated {eliminated}  mean shots {shots:,.0f}")
                first = False
            if case != cases[-1]:
                lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print("wrote", args.latex)


if __name__ == "__main__":
    main()
