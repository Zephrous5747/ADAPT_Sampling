#!/usr/bin/env python3
"""Coverage test summary with medians and the agreement of two clusters.

Per state and method (``runs/<geometry>_GAPS_ADAPT<j>/trials/*.csv``; the primary data, run on one cluster) and, when ``runs_trillium/`` holds the same job
from the other cluster, its counts next to them: trials, wrong selections (= best arm eliminated) with a one-sided 95% Clopper-Pearson upper bound, trials with
at least one missed interval (95% interval), mean and median shots, and the p-values of the comparison of the two clusters (Mann-Whitney on the shots, Fisher
on the two counts).  Writes ``runs/paper_a/coverage_summary.csv``.  The mean shots of a cell can be one trial (3e11 shots in 1 of 1000 trials = 99% of a
total): quote medians.

    python scripts/paper_a_coverage_summary.py

It also writes the SI table of the coverage test, ``reports/generated/tab_coverage.tex`` (one row per state, sorted by the relative gap; the counts of wrong
selections and the share of trials with a missed interval for the four methods, the median shots of II-A at 400; a last row with the totals).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
from scipy.stats import beta, fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
PRIMARY, OTHER = ROOT / "runs", ROOT / "runs_trillium"
LABELS = {"II-A_data_safe_bound_start_min_shots=400_diagnose": "II-A, 400", "II-0_safe_bound_start_min_shots=400_diagnose": "II-0, 400",
          "II-A_data_safe_bound_start_min_shots=50_diagnose": "II-A, 50", "II-A_data_safe_bound_start_anytime_min_shots=200_diagnose": "II-A anytime, 200"}


def upper(k: int, n: int) -> float:
    return 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))


def wilson(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


FAMILIES = [("H4_square_eq_side1p0_GAPS", "H$_4$, 1.0~\\AA"), ("H4_square_stretch_side2p0_GAPS", "H$_4$, 2.0~\\AA"), ("LiH_R3p0_GAPS", "LiH, 3.0~\\AA")]
METHODS = ["II-A, 400", "II-A, 50", "II-0, 400", "II-A anytime, 200"]


def sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return f"${float(m):.2f}\\times10^{{{int(e)}}}$"


def latex_table(table: pd.DataFrame) -> str:
    """One row per state (relative gap), four methods: wrong selections and trials with a missed interval; totals in the last row."""
    lines = [r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}", r"\toprule",
             r"State & relative gap & trials & \multicolumn{2}{c}{II-A, $n_{\min}=400$} & \multicolumn{2}{c}{II-A, $n_{\min}=50$} & \multicolumn{2}{c}{II-0, $n_{\min}=400$} "
             r"& \multicolumn{2}{c}{II-A anytime, 200} & median shots \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r" & & & wrong & missed (\%) & wrong & missed (\%) & wrong & missed (\%) & wrong & missed (\%) & II-A, 400 \\", r"\midrule"]
    tot = {m: [0, 0] for m in METHODS}
    for fam, name in FAMILIES:
        states = pd.read_csv(PRIMARY / "coverage_gap" / f"{fam}_states.csv").set_index("case")
        cases = sorted([c for c in table.case.unique() if c.startswith(fam + "_ADAPT")], key=lambda c: -float(states.loc[c, "achieved"]))
        for k, case in enumerate(cases):
            cells = []
            for m in METHODS:
                r = table[(table.case == case) & (table.method == m)]
                if r.empty:
                    cells += ["--", "--"]
                    continue
                r = r.iloc[0]
                cells += [str(int(r.wrong)), f"{100 * r.missed_share:.1f}"]
                tot[m][0] += int(r.wrong)
                tot[m][1] += int(r.trials)
            ref = table[(table.case == case) & (table.method == "II-A, 400")].iloc[0]
            gap = f"{float(states.loc[case, 'achieved']):.2g}" + (r"\textsuperscript{a}" if case.endswith("ADAPT0") else "")
            lines.append(f"{name if k == 0 else ''} & {gap} & {int(ref.trials)} & " + " & ".join(cells) + f" & {sci(float(ref.shots_median))} \\\\")
        lines.append(r"\midrule")
    lines.append(r"All states & & & " + " & ".join(f"{tot[m][0]} of {tot[m][1]:,}".replace(",", "{,}") + " & " for m in METHODS)[:-3] + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def clusters_sentence(table: pd.DataFrame) -> str:
    """The agreement of the two clusters (same seeds, two CPU types) as a sentence of the SI, with every number computed."""
    c = table[table.p_shots.notna()]
    n, ident = len(c), int(c.identical.sum())
    ratio = c.other_shots_median / c.shots_median
    return (f"Run on two clusters with the same seeds, {ident} of the {n} cells are identical trial by trial (the templates and the states whose rotation angle is the same on both CPU types); "
            f"in the others the angle differs by $10^{{-12}}$ or less, which changes the outcome of single trials and not their distribution: "
            f"{int((c.p_shots < 0.01).sum())} test of the shots (Mann--Whitney) and {int((c.p_wrong < 0.01).sum() + (c.p_missed < 0.01).sum())} of the counts of wrong selections and of missed intervals (Fisher) "
            f"gave $p<0.01$ ({0.01 * n:.1f} expected by chance in each), the median shots of the two clusters differ by a factor of ${ratio.min():.2f}$ to ${ratio.max():.2f}$, "
            f"and the cells hold {int(c.wrong.sum())} wrong selections on the primary cluster and {int(c.other_wrong.sum())} on the other.\n")


def main() -> None:
    rows = []
    for d in sorted(glob.glob(str(PRIMARY / "*GAPS_ADAPT*"))):
        case = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "trials", "*.csv"))):
            name = os.path.basename(f).split("step5__")[1].split("__shard")[0]
            t = pd.read_csv(f)
            n, wrong, missed = len(t), int((~t.correct.astype(bool)).sum()), int((t.miscovered_rounds > 0).sum())
            lo, hi = wilson(missed, n)
            row = {"case": case, "method": LABELS.get(name, name), "trials": n, "wrong": wrong, "wrong_upper95": upper(wrong, n), "best_eliminated": int(t.best_eliminated.sum()),
                   "missed_trials": missed, "missed_share": missed / n, "missed_lo": lo, "missed_hi": hi, "shots_mean": float(t.shots.mean()), "shots_median": float(t.shots.median()),
                   "top_trial_share_of_shots": float(t.shots.max() / t.shots.sum())}
            g = OTHER / case / "trials"
            other = glob.glob(str(g / f"step5__{name}*.csv"))
            if other:
                o = pd.read_csv(other[0])
                ow, om = int((~o.correct.astype(bool)).sum()), int((o.miscovered_rounds > 0).sum())
                row.update(other_wrong=ow, other_missed_trials=om, other_shots_median=float(o.shots.median()),
                           identical=bool(len(t) == len(o) and (t.shots.to_numpy() == o.shots.to_numpy()).all() and (t.correct.to_numpy() == o.correct.to_numpy()).all()),
                           p_shots=float(mannwhitneyu(t.shots, o.shots).pvalue),
                           p_wrong=float(fisher_exact([[wrong, n - wrong], [ow, len(o) - ow]])[1]), p_missed=float(fisher_exact([[missed, n - missed], [om, len(o) - om]])[1]))
            rows.append(row)
    table = pd.DataFrame(rows)
    out = ROOT / "runs" / "paper_a"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "coverage_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(table[["case", "method", "trials", "wrong", "wrong_upper95", "missed_share", "shots_median", "top_trial_share_of_shots"]].round(4).to_string(index=False))
    if "p_shots" in table:
        n = int(table.p_shots.notna().sum())
        print(f"\ncluster comparison, {n} cells: p<0.01 in shots {int((table.p_shots < 0.01).sum())}, wrong {int((table.p_wrong < 0.01).sum())}, missed {int((table.p_missed < 0.01).sum())} "
              f"(about {0.01 * n:.1f} each by chance)")
    gen = ROOT.parent / "reports" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "tab_coverage.tex").write_text(latex_table(table), encoding="utf-8", newline="\n")
    print("wrote", gen / "tab_coverage.tex")
    if "p_shots" in table:
        (gen / "coverage_clusters.tex").write_text(clusters_sentence(table), encoding="utf-8", newline="\n")
        print("wrote", gen / "coverage_clusters.tex")
    k = table
    print("worst-case wrong share: II-A,400 %.2f%%  II-0,400 %.2f%%  II-A,50 %.2f%%; anytime wrong max %d; anytime missed max %.1f%%" % (
        100 * (k[k.method == "II-A, 400"].wrong / k[k.method == "II-A, 400"].trials).max(), 100 * (k[k.method == "II-0, 400"].wrong / k[k.method == "II-0, 400"].trials).max(),
        100 * (k[k.method == "II-A, 50"].wrong / k[k.method == "II-A, 50"].trials).max(), k[k.method == "II-A anytime, 200"].wrong.max(), 100 * k[k.method == "II-A anytime, 200"].missed_share.max()))


if __name__ == "__main__":
    main()
