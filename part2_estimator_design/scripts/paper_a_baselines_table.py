#!/usr/bin/env python3
"""Paper A (SI rebuild, evidence block A2/C2): every method against II-0 on the nine fixed states, in one table.

The Results quote ranges over the nine fixed states of the shot-level comparison ("1.3 to 28 times cheaper than the sequential estimator, 1.2 to 4.3 times
cheaper than independent elimination", "the static estimator costs 1.1 to 5.8 times the sequential one"); the earlier tables of the SI showed five of the
nine states (all methods) or only the strongest baseline of each state.  This table has all nine, in the condition of Table III of the main text (finite
shots, pairwise rule, estimated radii, first radius from the exact largest gradient, default minimum of 50 shots per context and fold, ``oracle start`` of
``runs/paper_a/sota_fixed_state.csv``): mean context-shots as a ratio to II-0, so a ratio above one means that the method is dearer than II-0.

* baselines: M1 static (handed the exact gap and variances), M1 seq (sequential estimation, no elimination), M2 (independent elimination, marginal rule =
  Successive Elimination, and the pairwise rule);
* this work: II-A (learned splitting);
* with the energy data of the last VQE evaluation (fully commuting groups): the reuse baseline of Ikhtiarudin et al. and II-0 and II-A (home start)
  with the same data.

A bracket gives the share of correct selections where it is below 100 %.  The ranges that the main text quotes are printed.

Output: ``reports/generated/tab_baselines.tex`` (tabular only).

    python scripts/paper_a_baselines_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs" / "paper_a" / "sota_fixed_state.csv"
GEN = ROOT.parent / "reports" / "generated"

STATES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0~\AA, CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0~\AA, CISD"), ("LiH_R3p0_HF", r"LiH 3.0~\AA, HF"),
          ("LiH_R3p0_ADAPT3", r"LiH 3.0~\AA, ADAPT 3"), ("LiH_R3p0_ADAPT5", r"LiH 3.0~\AA, ADAPT 5"), ("H2O_eq_CISD", r"H$_2$O eq, CISD"),
          ("H2O_stretch_CISD", r"H$_2$O str, CISD"), ("H2O_eq_ADAPT11", r"H$_2$O eq, ADAPT 11"), ("H2O_stretch_ADAPT8", r"H$_2$O str, ADAPT 8")]
# (key, header, method name in the CSV)
COLUMNS = [("m1static", "M1 static", "M1 static (exact gap, exact variances)"), ("m1seq", "M1 seq", "M1 seq (no elimination)"),
           ("m2se", "M2, SE", "M2 independent, marginal rule"), ("m2pw", "M2, pairwise", "M2 independent, pairwise rule"), ("iia", "II-A", "II-A data"),
           ("ikh", "Ikh", "Ikh reuse, FC"), ("ii0r", "II-0", "II-0 + reuse"), ("iiar", "II-A", "II-A data + reuse (home)")]


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def num(x: float) -> str:
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def load() -> dict[tuple[str, str], dict]:
    rows = {}
    for r in csv.DictReader(CSV.open(encoding="utf-8")):
        if r["setting"] == "oracle start" and float(r["rho"]) == 0.0:
            rows[(r["case"], r["method"])] = r
    return rows


def main() -> None:
    rows = load()
    lines, ratios = [], {k: [] for k, _, _ in COLUMNS}
    static_over_seq = []
    for case, name in STATES:
        base = rows[(case, "II-0")]
        cells = [name, sci(float(base["shots_mean"]))]
        for key, _, method in COLUMNS:
            r = rows.get((case, method))
            if r is None:
                cells.append("--")
                continue
            ratio = float(r["shots_mean"]) / float(base["shots_mean"])
            ratios[key].append(ratio)
            cell = num(ratio)
            if float(r["correct_rate"]) < 0.995:
                cell += rf"{{\,\scriptsize[{100 * float(r['correct_rate']):.0f}]}}"
            cells.append(cell)
        if (case, COLUMNS[0][2]) in rows and (case, COLUMNS[1][2]) in rows:
            static_over_seq.append(float(rows[(case, COLUMNS[0][2])]["shots_mean"]) / float(rows[(case, COLUMNS[1][2])]["shots_mean"]))
        lines.append(" & ".join(cells) + r" \\")
        if case == "H4_square_stretch_side2p0_CISD":
            pass
    head = [
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Baselines} & This work & \multicolumn{3}{c}{With the energy data (FC)} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){8-10}",
        r"State & II-0 (shots) & " + " & ".join(h for _, h, _ in COLUMNS[:5]) + " & " + " & ".join(h for _, h, _ in COLUMNS[5:]) + r" \\", r"\midrule"]
    (GEN / "tab_baselines.tex").write_text("\n".join([*head, *lines, r"\bottomrule", r"\end{tabular}", ""]), encoding="utf-8", newline="\n")
    print("wrote", GEN / "tab_baselines.tex")
    for key, header, _ in COLUMNS:
        v = ratios[key]
        if v:
            print(f"{key:8s} n={len(v)} range {min(v):.2f} .. {max(v):.2f}")
    if static_over_seq:
        print(f"M1 static / M1 seq: n={len(static_over_seq)} range {min(static_over_seq):.2f} .. {max(static_over_seq):.2f}")
    first4 = [s[0] for s in STATES[:4]]
    for label, cases in (("first four", first4), ("other five", [s[0] for s in STATES[4:]])):
        for key in ("m1seq", "m2pw"):
            m = [float(rows[(c, dict((k, mm) for k, _, mm in COLUMNS)[key])]["shots_mean"]) / float(rows[(c, "II-0")]["shots_mean"]) for c in cases
                 if (c, dict((k, mm) for k, _, mm in COLUMNS)[key]) in rows]
            print(f"{label} {key}: {min(m):.2f} .. {max(m):.2f}")


if __name__ == "__main__":
    main()
