#!/usr/bin/env python3
"""Paper A, Supporting Information: the tables of the planning and learning studies that had no generator, from the stored runs.

Before the compression of the paper these tables were typed into the draft from console output.  Each is now written as a
tabular body to ``reports/generated/<name>.tex``; ``paper_a_si_clean.py`` pastes the body between ``% BEGIN/END INLINE`` markers
(captions and notes stand in the SI).

``si_planning_costs``    absolute planning-bound costs and their ratios (M1, M2, M3 with exact variances)
``si_surrogate``         M3 in the Gaussian surrogate against II-0 with sampled outcomes
``si_sign``              failure of the plug-in signs on stretched H4 after two ADAPT iterations
``si_ceilings``          oracle ceilings of estimator design (II-A, II-B, redesign)
``si_learned_fixed``     finite-shot study on three fixed states (II-0, II-A variants, II-E)
``si_learned_states``    the same on H2O CISD and on states of ADAPT trajectories
``si_trajectory``        median cumulative selection shots of the measured ADAPT trajectories of H4 at 1.0 Angstrom
``si_rounds``            one selection on H4 1.0 CISD with fixed and learned coefficients, round by round

    python scripts/paper_a_si_tables.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PART1 = ROOT.parent / "part1_full_reproducibility_package" / "runs"
OUT = ROOT.parent / "reports" / "generated"


def rows_of(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def thou(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def sci(x: float) -> str:
    e = int(math.floor(math.log10(x)))
    m = x / 10 ** e
    if round(m, 2) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.2f}\\times10^{{{e}}}$"


def sci1(x: float) -> str:
    """One decimal below 10^6, two from there on (the format of the trajectory table)."""
    e = int(math.floor(math.log10(x)))
    d = 1 if e < 6 else 2
    m = round(x / 10 ** e, d)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.{d}f}\\times10^{{{e}}}$"


def cost(x: float, limit: float = 1e7) -> str:
    return thou(x) if x < limit else sci(x)


def pct(x: float) -> str:
    return f"${100 * x:+.0f}\\%$".replace("+", "+").replace("-", "-") if x >= 0 else f"${100 * x:.0f}\\%$"


def change(x: float, base: float) -> str:
    d = 100 * (x / base - 1)
    return f"${d:+.0f}\\%$"


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", name)


def step4(case: str) -> dict[str, dict]:
    return {r["config"]: r for r in rows_of(RUNS / case / f"{case}_step4_learned_designs.csv")}


# ------------------------------------------------------------------------------------------------ planning bounds

PLAN = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0 HF"), ("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0 HF"),
        ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"), ("LiH_R3p0_HF", "LiH HF"), ("H2O_eq_HF", r"H$_2$O eq HF"), ("H2O_eq_CISD", r"H$_2$O eq CISD"),
        ("H2O_stretch_HF", r"H$_2$O str HF"), ("H2O_stretch_CISD", r"H$_2$O str CISD")]


def planning(case: str) -> dict:
    ceil = [r for r in rows_of(RUNS / case / f"{case}_step3_oracle_ceiling.csv") if r["strategy"] == "mass" and r["level"] == "II-0"][0]
    summ = json.loads((PART1 / case / f"{case}_summary.json").read_text())
    ns = json.loads((PART1 / case / f"{case}_NOSHARE_summary.json").read_text())
    return {"m1": f(ceil["part1_m1"]), "m2": f(ceil["part1_m2"]), "m3": f(ceil["m3_actual_static"]), "base": float(ns["total_context_shots"]),
            "gap": f(summ["top_abs_gradient"]) - f(summ["second_abs_gradient"])}


def ratio(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def si_planning_costs() -> None:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Case & gap & M1 & M2 & M3 \\", r"\midrule"]
    second = [r"\midrule", r"Case & M1/M2 & M2/M3 & M1/M3 & interaction \\", r"\midrule"]
    for case, name in PLAN:
        d = planning(case)
        lines.append(f"{name} & {d['gap']:.4f} & {cost(math.ceil(d['m1']))} & {cost(d['m2'])} & {cost(d['m3'])} \\\\")
        s, e, b = d["base"] / d["m1"], d["base"] / d["m2"], d["base"] / d["m3"]
        second.append(f"{name} & {ratio(d['m1'] / d['m2'])} & {ratio(d['m2'] / d['m3'])} & {ratio(d['m1'] / d['m3'])} & {b / (s * e):.2f} \\\\")
    write("si_planning_costs", lines + second + [r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------------------------------------ surrogate

SURR = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0 HF"), ("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0 HF"),
        ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"), ("LiH_R3p0_HF", "LiH HF")]


def si_surrogate() -> None:
    reuse = {(r["case_id"], r["config"]): r for r in rows_of(RUNS / "paper_a" / "reuse.csv")}
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Case & M3, surrogate & II-0, shot level & change & wrong & reuse \\", r"\midrule"]
    for case, name in SURR:
        rows = rows_of(RUNS / case / f"{case}_step1_online_baseline.csv")
        sur = [r for r in rows if r["check"] == "noisy" and r["rule"] == "marginal" and r["accounting"] == "maxima" and r["gamma"] == "0.5"][0]
        shot = [r for r in rows if r["check"] == "baseline" and r["rule"] == "pairwise" and r["accounting"] == "topup" and r["gamma"] == "0.9"][0]
        a, b = f(sur["part2_shots_median"]), f(shot["part2_shots_median"])
        wrong = round((1 - f(shot["part2_correct_rate"])) * f(shot["part2_n_trials"]))
        ru = reuse.get((case, "II-0 estimated")) or reuse.get((case, "II-A data"))
        reuse_txt = f"{f(ru['reuse_mean']):.1f} of {int(f(ru['pool']))}" if ru else "--"
        lines.append(f"{name} & {thou(a)} & {thou(b)} & {change(b, a)} & {wrong} & {reuse_txt} \\\\")
    write("si_surrogate", lines + [r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------------------------------------ plug-in signs

def si_sign() -> None:
    cfg = step4("H4_square_stretch_side2p0_ADAPT2")

    def row(label: str, key: str) -> str:
        r = cfg[key]
        return (f"\\quad {label} & {100 * f(r['correct_rate']):.0f}\\% & {f(r['miscovered_rate']):.2f} & {cost(f(r['shots_median']), 1e5)} & "
                f"{cost(f(r['shots_mean']), 1e5) if f(r['shots_mean']) < 1e5 else thou(f(r['shots_mean']))} \\\\")

    def row2(label: str, key: str) -> str:
        r = cfg[key]
        med, mean = f(r["shots_median"]), f(r["shots_mean"])
        return (f"\\quad {label} & {100 * f(r['correct_rate']):.0f}\\% & {f(r['miscovered_rate']):.2f} & "
                f"{sci(med) if med >= 1e6 else thou(med)} & {sci(mean) if mean >= 1e6 else thou(mean)} \\\\")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Rule / Design & correct & missed & median & mean \\", r"\midrule",
             r"\multicolumn{5}{l}{\emph{pairwise (plug-in signs)}} \\",
             row2("II-0, exact variances", "II-0 oracle"), row2("II-0", "II-0 estimated"), row2("II-A, learned", "II-A data"),
             row2("II-A, HF prior only", "II-A HF prior"),
             r"\multicolumn{5}{l}{\emph{sign-aware}} \\",
             row2("II-0", "II-0 safe"), row2("II-E, learned", "II-A data, contrast"),
             r"\multicolumn{5}{l}{\emph{sign-aware, anytime}} \\",
             row2("II-E, learned", "II-A data, contrast, bound start, anytime"), r"\bottomrule", r"\end{tabular}"]
    write("si_sign", lines)


# ------------------------------------------------------------------------------------------------ oracle ceilings

CEIL = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0 HF"), ("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0 HF"),
        ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"), ("LiH_R3p0_HF", "LiH HF"), ("H2O_eq_HF", r"H$_2$O eq HF"), ("H2O_eq_CISD", r"H$_2$O eq CISD"),
        ("H2O_stretch_HF", r"H$_2$O str HF"), ("H2O_stretch_CISD", r"H$_2$O str CISD")]


def ceiling_rows(case: str) -> dict:
    """The ceilings of LiH and H2O are those of the run that has the redesign columns (``step3_oracle_ceiling_table5.csv``)."""
    table5 = [r for r in rows_of(RUNS / "step3_oracle_ceiling_table5.csv") if r["case_id"] == case]
    src = table5 or rows_of(RUNS / case / f"{case}_step3_oracle_ceiling.csv")
    return {(r["strategy"], r["level"]): r for r in src}


def si_ceilings() -> None:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Case & II-0 & II-A & II-B & redesign (II-A / II-B) \\", r"\midrule"]
    for case, name in CEIL:
        rows = ceiling_rows(case)
        z = rows[("mass", "II-0")]
        base, bt = f(z["m3_actual_static"]), f(z["m3_bound_static_topup"])
        cells = []
        for lvl in ("II-A", "II-B"):
            r = rows[("mass", lvl)]
            cells.append((f"${100 * (f(r['m3_actual_static']) / base - 1):.0f}\\%$", f"${100 * (f(r['m3_bound_redesign_topup']) / bt - 1):.0f}\\%$"))
        lines.append(f"{name} & {cost(base)} & {cells[0][0]} & {cells[1][0]} & {cells[0][1]} / {cells[1][1]} \\\\")
    write("si_ceilings", lines + [r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------------------------------------ learned estimators, finite shots

FIXED = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"), ("LiH_R3p0_HF", "LiH HF")]


def si_learned_fixed() -> None:
    cfg = {case: step4(case) for case, _ in FIXED}
    # LiH: the flat prior stands for the Hartree-Fock prior (the prior is the state itself), and the sign-aware row is the 200-trial run of the
    # comparison with published strategies
    cfg["LiH_R3p0_HF"]["II-A HF nu=100"] = cfg["LiH_R3p0_HF"]["II-A flat nu=100"]
    cfg["LiH_R3p0_HF"]["II-A data, safe"] = {r["config"]: r for r in rows_of(RUNS / "LiH_R3p0_HF" / "LiH_R3p0_HF_step5_external_baselines.csv")}["II-A data, safe"]

    def cell(case: str, key: str, base_key: str = "II-0 estimated", mark: str = "", bold: bool = False, count: bool = False) -> str:
        r, b = cfg[case].get(key), cfg[case][base_key]
        if r is None:
            return "--"
        s = f"{thou(f(r['shots_mean']))} ({change(f(r['shots_mean']), f(b['shots_mean']))}){mark}"
        if count and f(r["correct_rate"]) < 1:
            s += f", {round(f(r['correct_rate']) * f(r['n_trials']))}"
        return f"\\textbf{{{s}}}" if bold else s

    def block(title: str, key: str, **kw) -> list[str]:
        cells = [cell(case, key, mark=(kw.get("marks", {}).get(case, "")), bold=kw.get("bold", False), count=kw.get("count", False)) for case, _ in FIXED]
        return [f"\\multicolumn{{4}}{{@{{}}l}}{{{title}}} \\\\", " & " + " & ".join(cells) + " \\\\"]

    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"\multicolumn{1}{@{}l}{} & " + " & ".join(n for _, n in FIXED) + r" \\", r"\midrule",
             r"\multicolumn{4}{@{}l}{II-0; rule: pairwise} \\",
             " & " + " & ".join(thou(f(cfg[c]["II-0 estimated"]["shots_mean"])) for c, _ in FIXED) + r" \\"]
    exact = [f"{thou(f(cfg[c]['II-A oracle']['shots_mean']))} ({change(f(cfg[c]['II-A oracle']['shots_mean']), f(cfg[c]['II-0 oracle']['shots_mean']))})$^a$" for c, _ in FIXED]
    lines += [r"\multicolumn{4}{@{}l}{II-A, exact covariances$^a$; rule: pairwise} \\", " & " + " & ".join(exact) + r" \\"]
    prior = [cell(c, "II-A HF prior", count=True) if c != "LiH_R3p0_HF" else "--" for c, _ in FIXED]
    lines += [r"\multicolumn{4}{@{}l}{II-A, HF prior only; rule: pairwise} \\", " & " + " & ".join(prior) + r" \\"]
    lines += block(r"II-A, HF prior, $\nu=100$$^c$; rule: pairwise", "II-A HF nu=100")
    lines += block(r"\textbf{II-A, learned from data ($\nu=0$)}; rule: pairwise", "II-A data", bold=True)
    lines += block(r"II-A, learned; rule: sign-aware", "II-A data, safe", marks={"LiH_R3p0_HF": "$^d$"})
    lines += block(r"II-E, learned; rule: sign-aware", "II-A data, contrast")
    lines += block(r"II-E, learned, a priori $R_0$; rule: sign-aware", "II-A data, contrast, bound start", marks={"LiH_R3p0_HF": "$^b$"})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("si_learned_fixed", lines)
    oracle = [thou(f(cfg[c]["II-0 oracle"]["shots_mean"])) for c, _ in FIXED]
    print("II-0 with exact variances:", ", ".join(oracle))


STATES = [("H2O_eq_CISD", r"H$_2$O eq CISD"), ("H2O_stretch_CISD", r"H$_2$O str CISD"), ("H4_square_eq_side1p0_ADAPT5", r"H$_4$ 1.0, $k=5$ (gap 15\%)"),
          ("LiH_R3p0_ADAPT3", r"LiH, $k=3$ (gap 75\%)"), ("LiH_R3p0_ADAPT5", r"LiH, $k=5$ (gap 15\%)"), ("H2O_eq_ADAPT11", r"H$_2$O eq, $k=11$ (gap 11\%)"),
          ("H2O_stretch_ADAPT8", r"H$_2$O str, $k=8$ (gap 15\%)")]


def si_learned_states() -> None:
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"State & II-0 & II-A, learned & II-E, learned & II-E, a priori $R_0$ \\", r"\midrule"]
    for case, name in STATES:
        cfg = step4(case)
        base = f(cfg["II-0 estimated"]["shots_mean"])

        def cell(key: str) -> str:
            r = cfg[key]
            s = change(f(r["shots_mean"]), base)
            c = round(100 * f(r["correct_rate"]))
            return s + (f" [{c}]" if c < 100 else "")

        lines.append(f"{name} & {sci(base)} & {cell('II-A data')} & {cell('II-A data, contrast')} & {cell('II-A data, contrast, bound start')} \\\\")
    write("si_learned_states", lines + [r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------------------------------------ one selection, round by round

def si_rounds() -> None:
    case = "H4_square_eq_side1p0_CISD"
    fixed = rows_of(RUNS / "paper_a" / f"{case}_II-0_estimated_evolution.csv")
    learned = rows_of(RUNS / "paper_a" / f"{case}_II-A_data_evolution.csv")
    lines = [r"\begin{tabular}{rrrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{II-0} & \multicolumn{4}{c}{II-A, learned} & \multicolumn{3}{c}{refit} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-11}",
             r"$r$ & radius & $\lvert\mathcal A_r\rvert$ & shots & $\lvert\mathcal A_r\rvert$ & $\lvert\mathcal B_r\rvert$ & ctx & shots & cov.\ err. & moved & var.\ ratio \\",
             r"\midrule"]
    last = None
    for a, b in zip(fixed, learned):
        if b["refit"] == "True":
            last = (f(b["leader_cov_error_median"]), f(b["leader_mass_moved"]), f(b["leader_variance_ratio"]))
            stats = " & ".join(f"{x:.2f}" for x in last)
        else:
            stats = " & ".join(f"({x:.2f})" for x in last)
        lines.append(f"{b['round']} & {f(b['radius']):.3f} & {int(f(a['active']))} & {thou(f(a['cumulative_shots']))} & {int(f(b['active']))} & {thou(f(b['support']))} & "
                     f"{int(f(b['contexts_receiving_shots']))} & {thou(f(b['cumulative_shots']))} & {stats} \\\\")
    write("si_rounds", lines + [r"\bottomrule", r"\end{tabular}"])


# ------------------------------------------------------------------------------------------------ measured trajectory on H4

def si_trajectory() -> None:
    """Median cumulative selection shots of the measured ADAPT trajectories of H4 at 1.0 Angstrom (48 per method), and the exact path."""
    import statistics

    case = "H4_square_eq_side1p0_HF"
    path = rows_of(RUNS / case / f"{case}_adapt_trajectory.csv")
    sel = rows_of(RUNS / case / "phase4" / f"{case}_phase4_selections.csv")
    methods = ["II-0 safe", "II-A data, safe", "II-A data, contrast"]
    med = {}
    for m in methods:
        by_k: dict[int, list[float]] = {}
        for r in sel:
            if r["method"] == m:
                by_k.setdefault(int(r["iteration"]), []).append(f(r["cumulative_shots"]))
        med[m] = {k: statistics.median(v) for k, v in by_k.items()}
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule", r"$k$ & $\max_i\lvert g_i\rvert$ & rel.\ gap & $\Delta E$ (mHa) & II-0 & II-A & II-E \\", r"\midrule"]
    for k in sorted(med[methods[0]]):
        r, nxt = path[k], path[k + 1]
        gap = "tie" if f(r["relative_gap"]) < 1e-6 else f"{100 * f(r['relative_gap']):.1f}\\%"
        lines.append(f"{k} & {f(r['max_abs_gradient']):.3f} & {gap} & {1000 * f(nxt['energy_error']):.1f} & "
                     + " & ".join(sci1(med[m][k]) for m in methods) + r" \\")
    write("si_trajectory", lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    for fn in (si_planning_costs, si_surrogate, si_sign, si_ceilings, si_learned_fixed, si_learned_states, si_rounds, si_trajectory):
        try:
            fn()
        except Exception as e:  # report and go on: one missing run must not hide the others
            print("FAILED", fn.__name__, repr(e), file=sys.stderr)
