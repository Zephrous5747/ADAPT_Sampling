#!/usr/bin/env python3
"""Tabular bodies (captions live in the SI) for the results of the cluster round of 2026-10-27.

Reads ``runs/paper_a/minshots_headline_n400_rho0.1.csv`` and ``minshots_headline_n400.csv`` (``paper_a_minshots_headline.py``), ``ablations_400.csv``
(``paper_a_ablations_400.py``) and ``yen_reproduction.csv`` (``reproduce_yen_splitting.py``) and writes to ``reports/generated``:

* ``tab_rho01_headline.tex``: the 21 states at rho = 0.1, every method at n_min = 400: II-A, share of selections inside the rho-set, II-0 / II-A and best baseline / II-A
  (ratios of means with 95% intervals), and II-A at rho = 0.1 over II-A at rho = 0;
* ``tab_crossfit.tex``, ``tab_gamma.tex``, ``tab_reuse400.tex``: the ablations at n_min = 400;
* ``tab_yen.tex``: the Hamiltonian-variance reduction of Yen et al. reproduced with the splitting code.

    python scripts/paper_a_round4_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parent / "reports" / "generated"
RUNS = ROOT / "runs" / "paper_a"
NAMES = {"H4_square_eq_side1p0": r"H$_4$ 1.0~\AA", "H4_square_stretch_side2p0": r"H$_4$ 2.0~\AA", "LiH_R3p0": r"LiH 3.0~\AA", "LiH_R1p6": r"LiH 1.6~\AA", "BeH2": r"BeH$_2$",
         "H2O_eq": r"H$_2$O eq", "H2O_stretch": r"H$_2$O str"}


def state(case: str) -> str:
    for key, name in NAMES.items():
        if case.startswith(key):
            rest = case[len(key):].lstrip("_")
            return f"{name}, " + ("CISD" if rest == "CISD" else "HF" if rest in ("HF", "") else "ADAPT " + rest.replace("ADAPT", ""))
    return case


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10 ** e:.2f}\times10^{{{e}}}$"


def iv(r: float, lo: float, hi: float) -> str:
    return f"{r:.2f} [{lo:.2f}, {hi:.2f}]"


def write(name: str, header: list[str], body: list[str], spec: str) -> None:
    text = "\n".join([rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])
    (GEN / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", name)


def headline() -> None:
    a, b = pd.read_csv(RUNS / "minshots_headline_n400_rho0.1.csv"), pd.read_csv(RUNS / "minshots_headline_n400.csv")
    m = a.merge(b[["case", "ii_a_mean"]], on="case", suffixes=("", "_rho0"))
    body = []
    for r in m.itertuples():
        body.append(f"{state(r.case)} & {sci(r.ii_a_mean)} & {100 * r.ii_a_correct:.0f}\\% & {iv(r.ii0_over_iia, r.ii0_lo, r.ii0_hi)} & {iv(r.best_over_iia, r.best_lo, r.best_hi)} & {r.best_baseline} & "
                    f"{r.ii_a_mean / r.ii_a_mean_rho0:.2f} \\\\")
    write("tab_rho01_headline", ["State", "II-A", "inside $\\rho$-set", "II-0 / II-A", "best baseline / II-A", "baseline", r"II-A, $\rho=0.1$ / $\rho=0$"], body, "@{}lrrllrr@{}")


def ablations() -> None:
    t = pd.read_csv(RUNS / "ablations_400.csv")
    c = t[t.study == "crossfit"]
    write("tab_crossfit", ["State", "no cross-fit / cross-fit", "runs", "exact best", "interval missed in a trial"],
          [f"{state(r.case)} & {iv(r.ratio, r.lo, r.hi)} & {r.n} & {100 * r.exact_best_variant:.0f}\\% / {100 * r.exact_best_reference:.0f}\\% & {100 * r.missed_variant:.0f}\\% / {100 * r.missed_reference:.0f}\\% \\\\" for r in c.itertuples()],
          "@{}lrrrr@{}")
    g = t[t.study == "gamma"]
    cols = [("II-A", 0.8), ("II-A", 0.95), ("II-0", 0.8), ("II-0", 0.95), ("M2", 0.8), ("M2", 0.95)]
    body = []
    for case in dict.fromkeys(g.case):
        cells = []
        for method, gam in cols:
            r = g[(g.case == case) & (g.variant == f"{method} gamma={gam:g} / gamma=0.9")].iloc[0]
            cells.append(iv(r.ratio, r.lo, r.hi))
        body.append(f"{state(case)} & " + " & ".join(cells) + r" \\")
    write("tab_gamma", ["State"] + [f"{m}, $\\gamma={x:g}$" for m, x in cols], body, "@{}lrrrrrr@{}")
    u = t[t.study == "reuse"]
    body = []
    for r in u.itertuples():
        body.append(f"{state(r.case)} & {r.variant.split(' / ')[0]} & {sci(r.mean_variant)} & {iv(r.ratio, r.lo, r.hi)} & {iv(r.ikh_over_variant, r.ikh_lo, r.ikh_hi)} \\\\")
    write("tab_reuse400", ["State", "estimator with the free data", "shots", "over the same without", "reuse baseline / estimator"], body, "@{}llrll@{}")


def yen() -> None:
    d = pd.read_csv(RUNS / "yen_reproduction.csv")
    body = []
    for mol in dict.fromkeys(d.molecule):
        s = d[d.molecule == mol]
        r = s.iloc[0]
        lab = mol.replace("H2O", r"H$_2$O (107.6$^\circ$)").replace("BeH2", r"BeH$_2$").replace("NH3", r"NH$_3$").replace("H2", r"H$_2$") if mol != "H2" else r"H$_2$"
        rng = lambda col: f"{s[col].min():.3g}" if abs(s[col].min() - s[col].max()) < 5e-3 * abs(s[col].max()) else f"{s[col].min():.3g}--{s[col].max():.3g}"  # noqa: E731
        body.append(f"{lab} & {r.n_qubits} & {r.pauli_terms} & {rng('SI')} & {r.published_SI:.3g} & {rng('split_SI')} & {r.published_ICS:.3g} & {rng('ratio_split_over_SI')} & {r.published_ratio:.2f} & {rng('FF')} \\\\")
    write("tab_yen", ["Molecule", "qubits", "Pauli terms", "SI", "SI, published", "split", "ICS, published", "split / SI", "published", "first fit by weight"], body, "@{}lrrrrrrrrr@{}")


def norm(t: pd.DataFrame) -> pd.DataFrame:
    t.columns = [c.replace(" ", "_").replace("-", "_") for c in t.columns]
    return t


def grouping() -> None:
    t = norm(pd.read_csv(RUNS / "q1_grouping.csv"))
    body = []
    for r in t.itertuples():
        body.append(f"{state(r.case)} & {iv(r.best_over_IIA_weight, r.best_lo_weight, r.best_hi_weight)} & {iv(r.best_over_IIA_mass, r.best_lo_mass, r.best_hi_mass)} & {r.best_mass} & "
                    f"{r.II_A_mass_over_weight:.2f} & {r.II_0_mass_over_weight:.2f} \\\\")
    write("tab_grouping", ["State", "best / II-A, by weight", "best / II-A, by mass", "baseline", "II-A, mass / weight", "II-0, mass / weight"], body, "@{}lllrrr@{}")


def trajectories400() -> None:
    t = pd.read_csv(RUNS / "q2_methods400.csv")
    r4 = pd.read_csv(RUNS / "q2_trajectories400.csv").set_index("case")
    names = {"II-A": "II-A", "II-0": "II-0", "M2": "M2", "M1 seq": "M1 seq"}
    body = []
    for case in dict.fromkeys(t.case):
        s = t[t.case == case]
        cells = []
        for key in names:
            r = s[s.method.str.startswith(key)]
            cells.append(f"{r.iloc[0].median_over_IIA:.2f}" if len(r) else "--")  # M1 seq was not run on H$_2$O
        a = s[s.method.str.startswith("II-A")].iloc[0]
        body.append(f"{state(case)} & {a.iterations:.0f} / {a.exact_iterations} & {sci(a.shots_median)} & " + (f"{r4.loc[case, 'ratio_400_over_50']:.2f}" if case in r4.index and r4.loc[case, 'ratio_400_over_50'] == r4.loc[case, 'ratio_400_over_50'] else "--") + " & " + " & ".join(cells[1:]) + f" & {100 * s.rho_good.min():.1f}\% \\\\")
    write("tab_traj400", ["State", "iterations / exact", "II-A shots", "II-A 400 / 50", "II-0 / II-A", "M2 / II-A", "M1 seq / II-A", r"inside $\rho$-set"], body, "@{}lrrrrrrr@{}")


def pools400() -> None:
    t = norm(pd.read_csv(RUNS / "q9_pools400.csv"))
    body = []
    for r in t.itertuples():
        base, pool = r.case.split("@")
        body.append(f"{state(base)} & {pool} & {sci(r.IIA_mean)} & {100 * r.IIA_within_rho:.0f}\% & {iv(r.best_over_IIA, r.best_lo, r.best_hi)} & {r.best} & {r.M2_over_IIA:.2f} & {r.M1_seq_over_IIA:.2f} \\\\")
    write("tab_pools400", ["State", "pool", "II-A", r"inside $\rho$-set", "best baseline / II-A", "baseline", "M2 / II-A", "M1 seq / II-A"], body, "@{}llrrlrrr@{}")


def main() -> None:
    headline()
    ablations()
    yen()
    grouping()
    trajectories400()
    pools400()


if __name__ == "__main__":
    main()
