#!/usr/bin/env python3
"""Paper A (round-1 review, B3): the headline ratios at the two minimum numbers of shots, for every method, in one table.

The headline comparison of the main text (Table IV) gives every method a minimum of ``n_min = 400`` shots per context and fold.  A
reader asks what the ratios are at the default of 50 and whether the ranking depends on the choice, so this table puts the two side by
side for the 21 states of Table IV (the 17 states of the a priori study and four ADAPT snapshots of H4):

* ``n_min = 50``: II-A, II-0 and every baseline at its default (a priori start, sign-aware rule, per-round intervals); II-0 / II-A and
  best baseline / II-A as ratios of mean context-shots with 95 % bootstrap intervals (10,000 resamples of numerator and denominator),
  the best baseline being the cheapest of M1 seq, M2, M2 marginal and the reuse baseline whose selections are at least 95 % correct;
* ``n_min = 400``: the same ratios as in Table IV (read from ``runs/paper_a/minshots_headline_n400.csv``, written by
  ``scripts/paper_a_minshots_headline.py``);
* the share of correct selections of II-A at 50.

A minimum of 200 shots was run for the three late states of Table S17 only and is not in this table.

Outputs: ``reports/generated/tab_nmin_ratios.tex`` (tabular only) and ``runs/paper_a/nmin_ratios.csv``.

    python scripts/paper_a_nmin_table.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_a_bootstrap import ratio_interval  # noqa: E402
from paper_a_minshots_headline import (  # noqa: E402
    BASELINES,
    II_0_DEFAULT,
    II_A_DEFAULT,
    SEED,
    STATES,
    VALID,
    cell,
    load,
)

REPORTS = ROOT.parent / "reports"


def main() -> None:
    rng = np.random.default_rng(SEED + 1)
    n400 = {r["case"]: r for r in csv.DictReader((ROOT / "runs" / "paper_a" / "minshots_headline_n400.csv").open())}
    rows, lines = [], []
    for case, name in STATES:
        t = load(case)
        a, a_correct = t[II_A_DEFAULT]
        r0, _ = t[II_0_DEFAULT]
        valid = [(b, t[(step, lab)][0]) for b, step, lab in BASELINES if (step, lab) in t and t[(step, lab)][1] >= VALID]
        best_name, best = min(valid, key=lambda item: item[1].mean())
        vs0 = ratio_interval(r0, a, "mean", rng)
        vsb = ratio_interval(best, a, "mean", rng)
        m = n400[case]
        row = {"case": case, "iia50_correct": a_correct, "ii0_over_iia_50": vs0[0], "ii0_lo_50": vs0[1], "ii0_hi_50": vs0[2],
               "best_over_iia_50": vsb[0], "best_lo_50": vsb[1], "best_hi_50": vsb[2], "best_baseline_50": best_name,
               "ii0_over_iia_400": float(m["ii0_over_iia"]), "ii0_lo_400": float(m["ii0_lo"]), "ii0_hi_400": float(m["ii0_hi"]),
               "best_over_iia_400": float(m["best_over_iia"]), "best_lo_400": float(m["best_lo"]), "best_hi_400": float(m["best_hi"]),
               "best_baseline_400": m["best_baseline"]}
        rows.append(row)
        lines.append(f"{name} & {cell(*vs0)} & {cell(*vsb)} & {best_name} & {100 * a_correct:.0f}\\% & "
                     f"{cell(row['ii0_over_iia_400'], row['ii0_lo_400'], row['ii0_hi_400'])} & "
                     f"{cell(row['best_over_iia_400'], row['best_lo_400'], row['best_hi_400'])} & {m['best_baseline']} \\\\")
        print(f"{case:34s} n=50: II-0 {vs0[0]:.2f}  best {vsb[0]:.2f} ({best_name}); n=400: II-0 {row['ii0_over_iia_400']:.2f}  best "
              f"{row['best_over_iia_400']:.2f} ({m['best_baseline']})")
        if case == STATES[16][0]:  # the 17 states of the a priori study above the line, the four ADAPT snapshots of H4 below
            lines.append(r"\midrule")
    body = [r"\begin{tabular}{@{}lrrlrrrl@{}}", r"\toprule",
            r" & \multicolumn{4}{c}{$n_{\min}=50$ (default)} & \multicolumn{3}{c}{$n_{\min}=400$ (Table~\ref{M-tab:minshots})} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
            r"State & II-0 / II-A & best baseline / II-A & baseline & II-A correct & II-0 / II-A & best baseline / II-A & baseline \\", r"\midrule",
            *lines, r"\bottomrule", r"\end{tabular}", ""]
    out = REPORTS / "generated" / "tab_nmin_ratios.tex"
    out.write_text("\n".join(body), encoding="utf-8", newline="\n")
    csv_path = ROOT / "runs" / "paper_a" / "nmin_ratios.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out, "and", csv_path)
    for key, label in (("ii0_over_iia", "II-0 / II-A"), ("best_over_iia", "best baseline / II-A")):
        for n in (50, 400):
            v = [r[f"{key}_{n}"] for r in rows]
            print(f"{label:22s} n_min={n:3d}: range {min(v):.2f} to {max(v):.2f}, median {np.median(v):.2f}")
    for n in (50, 400):
        lo = [r["case"] for r in rows if r[f"best_lo_{n}"] <= 1]
        print(f"n_min={n}: states where the interval of best baseline / II-A does not lie above one:", lo)


if __name__ == "__main__":
    main()
