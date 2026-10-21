#!/usr/bin/env python3
"""Tables, figure and quoted numbers of the trajectory results of Paper A (Q7, Q9-Q12 of ``reports/draft.tex``).

The captions live in the draft; this script writes only the tabular bodies, so ``scripts/paper_a_inline.py`` can paste
them between the ``% BEGIN/END INLINE`` markers:

* ``reports/generated/tab_q7_cost.tex``        selection and modelled optimization of complete runs at three optimizer thresholds
* ``reports/generated/tab_q7_baselines.tex``   totals of the other certified methods relative to II-A on H2O at the three thresholds
* ``reports/generated/tab_premium_total.tex``  premium of certification in selection shots and in total shots (Q10)
* ``reports/generated/tab_confidence.tex``     confidence level of the selection against the optimization cost (Q11)
* ``reports/figures/fig_dose.pdf``             iterations (energy, for the 16-qubit windows) against the size of injected selection errors

Every number that the text of the draft quotes from these experiments is printed and written to
``runs/paper_a/trajectory_facts.json``, so each sentence can be checked against the stored runs.

Inputs (under ``part2_estimator_design/runs``): ``features{,_gtol1e-5_cl,_gtol1e-3_cl}/report/{cost_totals,conf_noninferiority}.csv``
(complete measured runs, replayed at the looser thresholds), ``features_q10{,_gtol1e-5,_gtol1e-3}`` (replays of the certified and
fixed-budget runs of Q10), ``paper_a/fixed_budget_summary.csv``, ``energy_policies{,_gtol1e-5_cl,_gtol1e-3_cl}`` (injected errors),
``energy_policies_smoke`` and ``energy_policies_gtol1e-{5,3}`` (second build, for the build dependence only),
``features/report/rho_map{,_full}.csv``, ``features{,_ext}`` replays and the Phase 4 files of the LiH minimum-shot runs.

    python scripts/paper_a_trajectory_results.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT.parent / "reports"
GEN = REPORTS / "generated"
FIG = REPORTS / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))

ORDER5 = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
ORDER_Q10 = ["H2O_eq_HF", "H2O_stretch_HF", "H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R1p6_HF", "LiH_R3p0_HF"]
LABEL = {"H4_square_eq_side1p0_HF": r"H$_4$ 1.0~\AA", "H4_square_stretch_side2p0_HF": r"H$_4$ 2.0~\AA", "LiH_R3p0_HF": r"LiH 3.0~\AA",
         "LiH_R1p6_HF": r"LiH 1.6~\AA", "BeH2_HF": r"BeH$_2$", "H2O_eq_HF": r"H$_2$O eq", "H2O_stretch_HF": r"H$_2$O str"}
POLICY_SYSTEMS = ["H4 1.0", "H4 2.0", "LiH 3.0", "BeH2", "H2O eq", "H2O str"]
WINDOWS = ["H2O dz8o eq", "H2O dz8o str"]
COST = {"9": "features", "5": "features_gtol1e-5_cl", "3": "features_gtol1e-3_cl"}
Q10 = {"9": "features_q10", "5": "features_q10_gtol1e-5", "3": "features_q10_gtol1e-3"}
POL = {"9": "energy_policies", "5": "energy_policies_gtol1e-5_cl", "3": "energy_policies_gtol1e-3_cl"}
EXP = {"9": r"$10^{-9}$", "5": r"$10^{-5}$", "3": r"$10^{-3}$"}
FACTS: dict = {}


def fact(key: str, value) -> None:
    FACTS[key] = value
    print(f"  {key}: {value}")


def sci(x, digits: int = 1) -> str:
    if x is None or not np.isfinite(x) or x == 0:
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return r"$" + f"{m:.{digits}f}" + r"\times10^{" + str(e) + r"}$"


def pct2(x: float) -> str:
    return f"{x:.2g}" + r"\%"


def ci(r: float, lo: float, hi: float) -> str:
    return f"{r:.2f} [{lo:.2f}, {hi:.2f}]"


def write(name: str, colspec: str, lines: list[str]) -> None:
    text = "\n".join([r"\begin{tabular}{" + colspec + "}", r"\toprule", *lines, r"\bottomrule", r"\end{tabular}", ""])
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", GEN / f"{name}.tex")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


# ------------------------------------------------------------------------------------------------ Q7

def cost_tables() -> None:
    print("\n[Q7] selection and optimization of complete runs")
    tabs = {k: pd.read_csv(RUNS / v / "report" / "cost_totals.csv") for k, v in COST.items()}

    def pick(tab, case, method, conf=95.0):
        s = tab[(tab.case == case) & (tab.method == method) & ((tab.confidence_pct - conf).abs() < 0.5)]
        return s.iloc[0] if len(s) else None

    lines = [r"System & runs & $C_{\mathrm{sel}}$ & \multicolumn{3}{c}{$C_{\mathrm{opt}}$} & \multicolumn{3}{c}{$C_{\mathrm{sel}}/C_{\mathrm{ADAPT}}$} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r" & & & " + " & ".join(EXP[k] for k in "953") + " & " + " & ".join(EXP[k] for k in "953") + r" \\", r"\midrule"]
    red5, red3, shares = [], [], {}
    for case in ORDER5:
        r = {k: pick(tabs[k], case, "II-A data, safe") for k in "953"}
        lines.append(f"{LABEL[case]} & {int(r['9'].n)} & {sci(r['9'].selection_median)} & " + " & ".join(sci(r[k].optimisation_median) for k in "953")
                     + " & " + " & ".join(pct2(r[k].selection_share_pct) for k in "953") + r" \\")
        red5.append(r["9"].optimisation_median / r["5"].optimisation_median)
        red3.append(r["9"].optimisation_median / r["3"].optimisation_median)
        shares[case] = [round(float(r[k].selection_share_pct), 3) for k in "953"]
        fact(f"q7.astar.{case}", [round(float(r[k].optimisation_median / r[k].selection_median), 1) for k in "953"])
    write("tab_q7_cost", "@{}lrrrrrrrr@{}", lines)
    fact("q7.copt_reduction_1e-5", [round(min(red5), 2), round(max(red5), 2)])
    fact("q7.copt_reduction_1e-3", [round(min(red3), 2), round(max(red3), 2)])
    fact("q7.shares_pct_9_5_3", shares)

    lines = [r"System & method & runs & \multicolumn{3}{c}{$C_{\mathrm{sel}}/C_{\mathrm{ADAPT}}$} & \multicolumn{3}{c}{total / II-A} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r" & & & " + " & ".join(EXP[k] for k in "953") + " & " + " & ".join(EXP[k] for k in "953") + r" \\", r"\midrule"]
    names = {"II-0 safe": "II-0", "M2 safe": "M2", "M1 seq, safe": "M1 seq"}
    for case in ("H2O_eq_HF", "H2O_stretch_HF"):
        ref = {k: pick(tabs[k], case, "II-A data, safe") for k in "953"}
        first = True
        for method, short in names.items():
            r = {k: pick(tabs[k], case, method) for k in "953"}
            if any(v is None for v in r.values()):
                continue
            ratios = [float(r[k].total_median / ref[k].total_median) for k in "953"]
            lines.append(f"{LABEL[case] if first else ''} & {short} & {int(r['9'].n)} & " + " & ".join(pct2(r[k].selection_share_pct) for k in "953")
                         + " & " + " & ".join(f"{x:.2f}" for x in ratios) + r" \\")
            fact(f"q7.total_over_IIA.{case}.{short}", [round(x, 2) for x in ratios])
            first = False
    write("tab_q7_baselines", "@{}llrrrrrrr@{}", lines)

    # uncertified pilot budgets of the confidence sweeps, in total shots (Q10 text)
    for case in ("H4_square_eq_side1p0_HF", "H2O_eq_HF", "H2O_stretch_HF"):
        for k in "953":
            t = tabs[k]
            s = t[(t.case == case) & t.method.str.startswith("Fixed")]
            ref = pick(t, case, "II-A data, safe")
            out = []
            for row in s.itertuples():
                out.append((int(row.method.split()[1].rstrip(",")), int(row.n), round(float(row.reached_pct), 1), round(float(row.total_median / ref.total_median), 2)))
            fact(f"q10.pilot_budgets.{case}.{k} (budget, runs, reach %, total/II-A)", sorted(out))

    # lowering the confidence from 99 to 68 % in total shots
    for case in ORDER5:
        vals = []
        for k in "953":
            a, b = pick(tabs[k], case, "II-A data, safe", 68.0), pick(tabs[k], case, "II-A data, safe", 99.0)
            vals.append(round(float(1 - a.total_median / b.total_median), 3) if a is not None and b is not None else None)
        fact(f"q11.total_saving_68_vs_99.{case}", vals)


# ------------------------------------------------------------------------------------------------ Q10

def premium_table() -> None:
    print("\n[Q10] premium of certification in total shots")
    summary = pd.read_csv(RUNS / "paper_a" / "fixed_budget_summary.csv")
    lines = [r"System & budget & \multicolumn{2}{c}{iterations} & \multicolumn{5}{c}{premium (certified / fixed)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-9}",
             r" & per selection & fixed & certified & selection & total, " + EXP["9"] + " & total, " + EXP["5"] + " & total, " + EXP["3"] + r" & iterations only \\", r"\midrule"]
    lows = []
    loose, tight = [], []
    for case in ORDER_Q10:
        row = summary[(summary.case == case) & (summary.allocation == "designed")].iloc[0]
        budget = int(round(float(row.budget)))
        totals = {}
        fx9 = jsonl(RUNS / Q10["9"] / case / f"ci_Fixed_{budget}_designed_d0.05.jsonl")
        ce9 = jsonl(RUNS / Q10["9"] / case / "ci_II-A_data_safe_d0.05.jsonl")
        for k, root in Q10.items():
            fx = jsonl(RUNS / root / case / f"ci_Fixed_{budget}_designed_d0.05.jsonl")
            ce = jsonl(RUNS / root / case / "ci_II-A_data_safe_d0.05.jsonl")
            ref = json.loads((RUNS / root / case / "exact_reference.json").read_text())
            target = 1.6e-3 if ref["final_error"] < 1.6e-3 else ref["final_error"] + 1e-4
            reach_f = 100 * float(np.mean([r["final_error"] <= target for r in fx]))
            reach_c = 100 * float(np.mean([r["final_error"] <= target for r in ce]))
            if reach_f < 99.5 or reach_c < 99.5:
                lows.append((case, k, round(reach_f, 1), round(reach_c, 1)))
            tf = np.array([r["selection_shots"] + r["c_opt_eta2"] for r in fx])
            tc = np.array([r["selection_shots"] + r["c_opt_eta2"] for r in ce])
            rng = np.random.default_rng(1)
            ia, ib = rng.integers(0, len(tc), (20000, len(tc))), rng.integers(0, len(tf), (20000, len(tf)))
            lo, hi = np.percentile(np.median(tc[ia], axis=1) / np.median(tf[ib], axis=1), [2.5, 97.5])
            totals[k] = (float(np.median(tc) / np.median(tf)), float(lo), float(hi))
        sel_f = np.median([r["selection_shots"] for r in fx9])
        sel_c = np.median([r["selection_shots"] for r in ce9])
        it_f = float(np.mean([r["iterations"] for r in fx9]))
        it_c = float(np.mean([r["iterations"] for r in ce9]))
        lines.append(f"{LABEL[case]} & {sci(budget)} & {it_f:.2f} & {it_c:.2f} & {sel_c / sel_f:.2f} & " + " & ".join(ci(*totals[k]) for k in "953")
                     + f" & {it_c / it_f:.2f} \\\\")
        fact(f"q10.{case} (runs fixed/certified, sel premium, total 9/5/3, iterations ratio)",
             [len(fx9), len(ce9), round(sel_c / sel_f, 3), [tuple(round(v, 2) for v in totals[k]) for k in "953"], round(it_c / it_f, 3), round(it_f - it_c, 2)])
        tight.append(totals["9"][0])
        loose += [totals["5"][0], totals["3"][0]]
    write("tab_premium_total", "@{}lrrrrlllr@{}", lines)
    fact("q10.total_premium_1e-9_range", [round(min(tight), 2), round(max(tight), 2)])
    fact("q10.total_premium_loose_range", [round(min(loose), 2), round(max(loose), 2)])
    fact("q10.reach_below_100 (case, threshold, fixed %, certified %)", lows)


# ------------------------------------------------------------------------------------------------ Q11: confidence level

def confidence_table() -> None:
    print("\n[Q11] confidence level")
    tabs = {k: pd.read_csv(RUNS / v / "report" / "conf_noninferiority.csv") for k, v in COST.items()}
    key = {"H4 1.0": "H4_square_eq_side1p0_HF", "H4 2.0": "H4_square_stretch_side2p0_HF", "LiH 3.0": "LiH_R3p0_HF", "H2O eq": "H2O_eq_HF", "H2O str": "H2O_stretch_HF"}
    lines = [r"System & runs per level & selection shots, $99\%$ & $99\%$ / $68\%$ & \multicolumn{3}{c}{$C_{\mathrm{opt}}$ at $68\%$ / at $99\%$} \\",
             r"\cmidrule(lr){5-7}", r" & & & & " + " & ".join(EXP[k] for k in "953") + r" \\", r"\midrule"]
    ratios, maxdev = [], {"5": 0.0, "3": 0.0}
    for system, case in key.items():
        t9 = tabs["9"][tabs["9"].system == system]
        top, low = t9[t9.confidence_pct == t9.confidence_pct.max()].iloc[0], t9[(t9.confidence_pct - 68).abs() < 0.5].iloc[0]
        cells = []
        for k in "953":
            t = tabs[k][tabs[k].system == system]
            r = t[(t.confidence_pct - 68).abs() < 0.5].iloc[0]
            cells.append(ci(r.c_opt_ratio_top, r.c_opt_lo, r.c_opt_hi))
            if k != "9":
                maxdev[k] = max(maxdev[k], float((t.c_opt_ratio_top - 1).abs().max()))
                fact(f"q11.conf.{system}.{k}.hi_max", round(float(t.c_opt_hi.max()), 3))
        sel_ratio = float(top.selection_median / low.selection_median)
        ratios.append(sel_ratio)
        steps = (round(float(t9.steps_mean.min()), 3), round(float(t9.steps_mean.max()), 3))
        lines.append(f"{LABEL[case]} & {int(low.n)} & {sci(top.selection_median)} & {sel_ratio:.2f} & " + " & ".join(cells) + r" \\")
        fact(f"q11.conf.{system} (levels, n top, sel ratio 99/68, steps range, Copt68 9/5/3)",
             [sorted(t9.confidence_pct.tolist()), int(top.n), round(sel_ratio, 2), steps, cells])
        if system == "H4 2.0":
            fact("q11.conf.H4 2.0 at 1e-9 by level", [(float(c), round(float(v), 2)) for c, v in zip(t9.confidence_pct, t9.c_opt_ratio_top)])
    write("tab_confidence", "@{}lrrrlll@{}", lines)
    fact("q11.conf.sel_ratio_range", [round(min(ratios), 2), round(max(ratios), 2)])
    fact("q11.conf.max_abs_dev_any_level (1e-5, 1e-3)", [round(maxdev["5"], 3), round(maxdev["3"], 3)])


# ------------------------------------------------------------------------------------------------ Q11: injected errors

def policy_facts_and_figure() -> None:
    print("\n[Q11] injected selection errors")
    tabs = {k: pd.read_csv(RUNS / v / "report" / "policies_all.csv") for k, v in POL.items()}
    t = tabs["9"]

    def val(tab, system, policy, col):
        r = tab[(tab.system == system) & (tab.policy == policy)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    for system in POLICY_SYSTEMS:
        grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        steps = [val(t, system, f"rho_random:{x:g}", "steps_ratio") for x in grid]
        star, good = None, True
        for x, v in zip(grid, steps):
            if v <= 1.03 and good:
                star = x
            else:
                good = False
        fact(f"q11.rho.{system} (exact iterations, ratios at 0.05..0.9, rho*)", [int(val(t, system, "exact", "exact_steps")), [round(v, 2) for v in steps], star])
        fact(f"q11.tau.{system} (ratios at 0.05..0.5)", [round(val(t, system, f"tau_random:{x:g}", "steps_ratio"), 2) for x in (0.05, 0.1, 0.2, 0.3, 0.5)])
        fact(f"q11.random_adapt_reach_pct.{system} (1e-9, 1e-5, 1e-3)", [val(tabs[k], system, "rho_random:1", "reached_pct") for k in "953"])
        fact(f"q11.greedy.{system} (onep, newton_hf, newton_exact, top3, top10, all)",
             [round(val(t, system, p, "steps_ratio"), 2) for p in ("onep_greedy", "newton_hf", "newton_exact", "reopt_top:3", "reopt_top:10", "reopt_all")])
        fact(f"q11.greedy_all_steps.{system} (1e-9, 1e-5, 1e-3)", [round(val(tabs[k], system, "reopt_all", "steps_ratio"), 3) for k in "953"])
        fact(f"q11.worst_energy.{system} (rho 0.1, 0.3, 0.5)", [round(val(t, system, f"rho_worst_e:{x:g}", "steps_ratio"), 2) for x in (0.1, 0.3, 0.5)])
        fact(f"q11.more_steps_pct.{system} (rho 0.2, 0.5)", [val(t, system, f"rho_random:{x:g}", "more_steps_pct") for x in (0.2, 0.5)])
        for k in "953":
            fact(f"q11.evals_vs_tie.{system}.{k} (rho 0.05, 0.1, 0.3, 0.5, tau 0.5)",
                 [round(val(tabs[k], system, p, "evals_ratio_tie_avg"), 2) for p in ("rho_random:0.05", "rho_random:0.1", "rho_random:0.3", "rho_random:0.5", "tau_random:0.5")])
        same = [abs(val(tabs[k], system, f"rho_random:{x:g}", "steps_ratio") - val(t, system, f"rho_random:{x:g}", "steps_ratio")) for k in "53" for x in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)]
        fact(f"q11.steps_change_at_loose_thresholds.{system} (max abs over rho<=0.7)", round(max(same), 3))
    for system in WINDOWS:
        cols = ("exact", "onep_greedy", "newton_hf", "reopt_top:3", "tau_random:0.1", "tau_random:0.3", "tau_random:0.5", "rho_random:0.1", "rho_random:0.3", "rho_random:0.5", "rho_worst_e:0.1")
        fact(f"q11.window.{system} err at 20 (mHa)", {c: round(val(t, system, c, "err20_mHa"), 2) for c in cols})
        fact(f"q11.window.{system} err at 40 (mHa)", {c: round(val(t, system, c, "err_exact_steps_mHa"), 2) for c in cols})

    # figure: iterations (energy error after 40 iterations for the windows) against the tolerance
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.8, "lines.markersize": 4.5,
                         "mathtext.fontset": "cm", "font.family": "serif"})
    titles = {"H4 1.0": "(a) H$_4$, 1.0 Å", "H4 2.0": "(b) H$_4$, 2.0 Å", "LiH 3.0": "(c) LiH, 3.0 Å", "BeH2": "(d) BeH$_2$",
              "H2O eq": "(e) H$_2$O, eq.", "H2O str": "(f) H$_2$O, str.", "H2O dz8o eq": "(g) cc-pVDZ, eq.",
              "H2O dz8o str": "(h) cc-pVDZ, str."}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.3), sharex=True)
    for ax, system in zip(axes.flat, POLICY_SYSTEMS + WINDOWS):
        window = system in WINDOWS
        col = "err_ratio_exact_steps" if window else "steps_ratio"

        def series(prefix, params):
            xs, ys, es = [], [], []
            for x in params:
                r = t[(t.system == system) & (t.policy == f"{prefix}:{x:g}")]
                if r.empty:
                    continue
                r = r.iloc[0]
                xs.append(x)
                ys.append(float(r[col]))
                es.append(0.0 if window or not np.isfinite(r.steps_se) else 1.96 * float(r.steps_se) / float(r.exact_steps))
            return xs, ys, es

        x, y, e = series("rho_random", [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1])
        ax.errorbar(x, y, yerr=e, color="blue", marker="o", capsize=2, label=r"random generator in the $\rho$-set")
        x, y, e = series("rho_worst_e", [0.1, 0.3, 0.5])
        ax.errorbar(x, y, yerr=e, color="red", marker="s", capsize=2, label=r"smallest energy lowering in the $\rho$-set")
        x, y, e = series("tau_random", [0.05, 0.1, 0.2, 0.3, 0.5])
        ax.errorbar(x, y, yerr=e, color="black", marker="^", capsize=2, label=r"random generator in the $\tau$-set")
        g = val(t, system, "onep_greedy", col)
        if np.isfinite(g):
            ax.axhline(g, color="magenta", ls="--", lw=1.5, label="largest one-parameter lowering")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_title(titles[system], fontsize=9)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"tolerance $\rho$ or $\tau$")
    axes[0, 0].set_ylabel("iterations / exact ADAPT")
    axes[1, 0].set_ylabel("iterations / exact ADAPT")
    axes[1, 2].set_ylabel("energy error at 40 iter.\n/ exact ADAPT", fontsize=8)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=0.6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_dose.pdf")
    fig.savefig(FIG / "fig_dose.png", dpi=200)
    plt.close(fig)
    print("wrote", FIG / "fig_dose.pdf")


# ------------------------------------------------------------------------------------------------ Q9, Q12, Discussion: quoted numbers

def other_facts() -> None:
    print("\n[Q9] minimum number of shots along LiH trajectories")
    import energy_e5_compare as e5m

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        exact = None
        for label, tags, method in (("50", [""], "II-A data, safe"), ("200", ["minshots200"], "II-A data, safe, min shots=200"),
                                    ("400", ["minshots400"], "II-A data, safe, min shots=400")):
            traj, sel, exact_steps = e5m.rows_of("LiH_R3p0_HF", tags, method)
            exact = exact_steps or exact
            steps = np.array([float(r["iterations"]) for r in traj])
            shots = np.array([float(r["total_shots"]) for r in traj])
            sh = np.array([float(r["shortfall"]) for r in sel])
            lo, hi = e5m.boot_ci(shots, np.random.default_rng(0))
            fact(f"q9.minshots{label} (trajectories, mean iterations, % above exact, selections, not rho-good, shortfalls of those, mean shots, CI)",
                 [len(traj), round(steps.mean(), 3), round(100 * float(np.mean(steps > exact)), 1), len(sel), int(np.sum(sh > 0.1 + 1e-12)),
                  sorted(round(float(v), 3) for v in sh[sh > 0.1 + 1e-12]), float(f"{shots.mean():.3g}"), [float(f"{lo:.3g}"), float(f"{hi:.3g}")]])
    finally:
        os.chdir(cwd)

    print("\n[Q12] optimizer evaluations of the exact path")
    sets = {"cluster 1e-9": "energy_policies", "cluster 1e-5": "energy_policies_gtol1e-5_cl", "cluster 1e-3": "energy_policies_gtol1e-3_cl",
            "second build 1e-9": "energy_policies_smoke", "second build 1e-5": "energy_policies_gtol1e-5", "second build 1e-3": "energy_policies_gtol1e-3"}
    cases = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "BeH2_HF", "H2O_eq_HF", "H2O_stretch_HF"]
    counts = {}
    for name, root in sets.items():
        row = []
        for case in cases:
            p = RUNS / root / case / "exact_reference.json"
            if name == "second build 1e-5" and case == "BeH2_HF":  # the second build ran BeH2 at 1e-5 only with the curvature-scaled rules
                p = RUNS / "energy_policies_nrho" / case / "exact_reference.json"
            ref = json.loads(p.read_text()) if p.exists() else None
            row.append((sum(ref["evaluations"]), ref["iterations"]) if ref else None)
            counts[(name, case)] = row[-1]
        fact(f"q12.exact_path_evaluations.{name} (H4 1.0, H4 2.0, LiH, BeH2, H2O eq, H2O str)", row)
    agree = [(c, g) for c in cases for g in ("1e-5", "1e-3") if counts[(f"second build {g}", c)] and counts[(f"cluster {g}", c)]]
    fact("q12.builds_agree_at_loose (agree, compared)", [sum(counts[(f"second build {g}", c)][0] == counts[(f"cluster {g}", c)][0] for c, g in agree), len(agree)])
    lih = {}
    for root, label in (("energy_policies", "1e-9"), ("energy_policies_gtol1e-5_cl", "1e-5"), ("energy_policies_gtol1e-3_cl", "1e-3")):
        rows = jsonl(RUNS / root / "LiH_R3p0_HF" / "policy_rho_random_0.05.jsonl") + jsonl(RUNS / root / "LiH_R3p0_HF" / "policy_rho_random_0.jsonl")
        six = [r["evals_total"] for r in rows if r["iterations"] == 6]
        lih[label] = (min(six), max(six), len(rows))
    fact("q12.LiH_orders_six_steps (min, max, trajectories)", lih)

    print("\n[Q12] final energies at the looser thresholds (replays, run by run)")
    for g in ("1e-5", "1e-3"):
        cert, fixed = [], []
        for root, tail in (("features", f"features_gtol{g}_cl"), ("features_ext", f"features_ext_gtol{g}_cl")):
            for cdir in sorted((RUNS / root).glob("*_HF")):
                for f in sorted(cdir.glob("ci_*.jsonl")):
                    lf = RUNS / tail / cdir.name / f.name
                    if not lf.exists():
                        continue
                    for ra, rb in zip(jsonl(f), jsonl(lf)):
                        d = 1e3 * (rb["final_error"] - ra["final_error"])
                        (fixed if "Fixed" in f.name else cert).append((d, cdir.name))
        big = [x for x in fixed if abs(x[0]) > 0.1]
        fact(f"q12.replay_{g} (certified runs, max |dE| mHa; uncertified runs, changed >0.1 mHa, of these H4 1.0, max dE)",
             [len(cert), float(f"{max(abs(d) for d, _ in cert):.2g}"), len(fixed), len(big), sum(1 for _, c in big if c == "H4_square_eq_side1p0_HF"),
              float(f"{max((d for d, _ in big), default=0):.3g}")])

    print("\n[Discussion] gradient against energy lowering")
    for name in ("rho_map", "rho_map_full"):
        t = pd.read_csv(RUNS / "features" / "report" / f"{name}.csv")
        for i, r in t.iterrows():
            fact(f"disc.{name}.{r['system']} (states, best in rho=0.1 set %, worst rho-good mean, worst rho-good min, top-|g| share mean)",
                 [int(r["states"]), round(float(r["covered_0.1_pct"]), 1), round(float(r["worst_0.1_mean"]), 2), round(float(r["worst_0.1_min"]), 2),
                  round(float(r["top_g_share_mean"]), 2)])


def main() -> None:
    cost_tables()
    premium_table()
    confidence_table()
    policy_facts_and_figure()
    other_facts()
    out = RUNS / "paper_a" / "trajectory_facts.json"
    out.write_text(json.dumps(FACTS, indent=1, default=str), encoding="utf-8", newline="\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
