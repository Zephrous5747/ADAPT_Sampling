#!/usr/bin/env python3
"""Paper A (round-3 review, Q2): how far is the learned design from the oracle, and how much of the way from the floor does it cover?

Block A, the condition of Table III of the main text (finite shots, pairwise rule, estimated radii, first radius from the exact largest gradient,
default minimum of 50 shots per context and fold; 100 to 200 trials), on the five fixed states of that table.  Mean context-shots as a ratio to II-0
with estimated radii:

* floor: independent elimination (M2) and sequential estimation without elimination (M1 seq), which use no learned covariance;
* II-0 (fixed coefficients), II-A (coefficients learned from the shots);
* ceiling: II-A with exact covariances and exact radii (``II-A oracle`` of the step-4 runs).

"Learned share of the gap" is (II-0 - II-A) / (II-0 - oracle) and, from the floor, (floor - II-A) / (floor - oracle) with the cheaper of M2 and M1 seq as the
floor.  It exceeds 100 % where the learned design is cheaper than the oracle.

Block B, the headline condition (n_min = 400, sign-aware rule, a priori first radius), on the 17 states of the classical-prior study: II-0 / II-A and
"oracle design" / II-A, the oracle design being the coefficient split designed with the exact covariance of the state (``true state, static``) while radii
and planning come from the data, so that the intervals stay certified (``runs/classical_prior/fixed_states.csv``, ratios of mean shots with 95 %
bootstrap intervals).

Outputs: ``reports/generated/tab_ladder_a.tex`` and ``tab_ladder_b.tex`` (tabular only), ``runs/paper_a/ladder.csv``.

    python scripts/paper_a_ladder_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
GEN = ROOT.parent / "reports" / "generated"

A_STATES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0~\AA, CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0~\AA, CISD"), ("LiH_R3p0_HF", r"LiH 3.0~\AA, HF"),
            ("H2O_eq_CISD", r"H$_2$O eq, CISD"), ("H2O_stretch_CISD", r"H$_2$O str, CISD")]


def rows_of(path: Path) -> dict[str, dict]:
    return {r["config"]: r for r in csv.DictReader(path.open())}


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"


def main() -> None:
    out_rows, lines, lines_b = [], [], []
    sem = []
    for case, name in A_STATES:
        r4 = rows_of(RUNS / case / f"{case}_step4_learned_designs.csv")
        r5 = rows_of(RUNS / case / f"{case}_step5_external_baselines.csv")
        pick = {"II-0": r4["II-0 estimated"], "II-A": r4["II-A data"], "orc": r4["II-A oracle"], "M2": r5["M2 pairwise"], "M1s": r5["M1 seq"]}
        v = {k: float(r["shots_mean"]) for k, r in pick.items()}
        sem += [float(r["shots_sem"]) / float(r["shots_mean"]) for r in pick.values()]
        floor = min(v["M2"], v["M1s"])
        from_ii0 = (v["II-0"] - v["II-A"]) / (v["II-0"] - v["orc"])
        from_floor = (floor - v["II-A"]) / (floor - v["orc"])
        out_rows.append({"block": "A", "case": case, **{f"{k}_shots": x for k, x in v.items()}, "gap_recovered_from_II0": from_ii0, "gap_recovered_from_floor": from_floor})
        lines.append(f"{name} & {sci(v['II-0'])} & {v['M2'] / v['II-0']:.2f} & {v['M1s'] / v['II-0']:.2f} & 1 & {v['II-A'] / v['II-0']:.2f} & {v['orc'] / v['II-0']:.2f} & "
                     f"{100 * from_ii0:.0f}\\% & {100 * from_floor:.0f}\\% \\\\")
        print(f"{case:34s} M2 {v['M2'] / v['II-0']:.2f}  M1seq {v['M1s'] / v['II-0']:.2f}  II-A {v['II-A'] / v['II-0']:.2f}  oracle {v['orc'] / v['II-0']:.2f}  "
              f"recovered {100 * from_ii0:.0f}% (from II-0), {100 * from_floor:.0f}% (from the floor)")
    print(f"largest relative standard error of the means in block A: {100 * max(sem):.1f}%  (median {100 * float(np.median(sem)):.1f}%)")

    fixed = list(csv.DictReader((RUNS / "classical_prior" / "fixed_states.csv").open()))
    ref = {}
    for r in fixed:
        if r["family"] != "CPD":
            continue
        ref.setdefault(r["case"], {})[r["arm"]] = r
    nmin = {r["case"]: r for r in csv.DictReader((RUNS / "paper_a" / "minshots_headline_n400.csv").open())}
    orc_ratios = []
    for case, d in ref.items():
        o, z = d.get("true state, static (ceiling)"), d.get("(i) II-0, data only")
        if o is None or z is None:
            continue
        name = z["state"].replace(" A,", "~\\AA,").replace("H2O", r"H$_2$O").replace("H4", r"H$_4$").replace("BeH2", r"BeH$_2$")
        ro, rz = float(o["ratio_to_ref"]), float(z["ratio_to_ref"])
        orc_ratios.append(ro)
        out_rows.append({"block": "B", "case": case, "iia_400_shots": float(o["reference_shots_mean"]), "ii0_over_iia": rz, "oracle_over_iia": ro,
                         "oracle_lo": float(o["ratio_low"]), "oracle_hi": float(o["ratio_high"])})
        lines_b.append(f"{name} & {sci(float(o['reference_shots_mean']))} & {rz:.2f} & {ro:.2f} [{float(o['ratio_low']):.2f}, {float(o['ratio_high']):.2f}] \\\\")
    gm = float(np.exp(np.mean(np.log(orc_ratios))))
    print(f"block B: oracle design / II-A over {len(orc_ratios)} states: range {min(orc_ratios):.2f} to {max(orc_ratios):.2f}, geometric mean {gm:.2f}; "
          f"below 1: {sum(r < 1 for r in orc_ratios)}, above 1: {sum(r > 1 for r in orc_ratios)}")
    head = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
            r" & II-0 & \multicolumn{2}{c}{Floor: no learned covariance} & & Learned & Ceiling: exact & \multicolumn{2}{c}{Learned share of the gap} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){8-9}",
            r"State & (shots) & M2 & M1 seq & II-0 & II-A & covariances & from II-0 & from the floor \\", r"\midrule"]
    head_b = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"State & II-A (shots) & II-0 / II-A & Oracle design / II-A \\", r"\midrule"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_ladder_a.tex").write_text("\n".join([*head, *lines, r"\bottomrule", r"\end{tabular}", ""]), encoding="utf-8", newline="\n")
    (GEN / "tab_ladder_b.tex").write_text("\n".join([*head_b, *lines_b, r"\bottomrule", r"\end{tabular}", ""]), encoding="utf-8", newline="\n")
    keys = sorted({k for r in out_rows for k in r})
    with (RUNS / "paper_a" / "ladder.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out_rows)
    print("wrote", GEN / "tab_ladder_a.tex", "and", GEN / "tab_ladder_b.tex")


if __name__ == "__main__":
    main()
