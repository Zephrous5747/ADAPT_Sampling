#!/usr/bin/env python3
"""Paper A: the tables the referee runs add (checklist A3, A6, A8, A9, A10, A11), from the stored runs.

Each table is written as a LaTeX fragment (``--out``, default ``../reports/generated``) and printed in plain text:

``tab_apriori``       the headline with no oracle input: sign-aware rule, a priori starting radius, per state: II-0, II-A,
                      sequential M1, independent arms (M2), the reuse baseline, correct shares, ratios with 95% intervals;
``tab_anytime``       per-round against anytime intervals (II-0 and II-A, a priori start): cost, ratio, missed-interval share;
``tab_sensitivity``   the thresholds (minimum shots, refit factor) and rho: II-A's cost relative to its default setting;
``tab_scaling``       structure and design time against molecule size (``runs/paper_a/scaling.csv``);
``tab_reproduction``  Huang and Izmaylov's published savings against the reproduction (``huang_izmaylov_reproduction_*.csv``);
``tab_sampled_optimizer``  the sampled optimiser against the cost model C_opt.

    python scripts/paper_a_referee_tables.py [--only tab_anytime tab_scaling]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_a_bootstrap import CASES, ratio_interval, load_trials  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SHORT = {"H4_square_eq_side1p0_CISD": "H$_4$ 1.0~\\AA, CISD", "H4_square_stretch_side2p0_CISD": "H$_4$ 2.0~\\AA, CISD",
         "LiH_R3p0_HF": "LiH 3.0~\\AA, HF", "LiH_R3p0_ADAPT3": "LiH 3.0~\\AA, ADAPT 3", "LiH_R3p0_ADAPT5": "LiH 3.0~\\AA, ADAPT 5",
         "H2O_eq_CISD": "H$_2$O eq, CISD", "H2O_stretch_CISD": "H$_2$O str, CISD", "H2O_eq_ADAPT11": "H$_2$O eq, ADAPT 11",
         "H2O_stretch_ADAPT8": "H$_2$O str, ADAPT 8", "LiH_R1p6_HF": "LiH 1.6~\\AA, HF", "LiH_R1p6_ADAPT2": "LiH 1.6~\\AA, ADAPT 2",
         "LiH_R1p6_ADAPT3": "LiH 1.6~\\AA, ADAPT 3", "LiH_R1p6_ADAPT4": "LiH 1.6~\\AA, ADAPT 4", "BeH2_ADAPT2": "BeH$_2$, ADAPT 2",
         "BeH2_ADAPT4": "BeH$_2$, ADAPT 4", "BeH2_ADAPT7": "BeH$_2$, ADAPT 7", "BeH2_ADAPT9": "BeH$_2$, ADAPT 9"}
NEW_STATES = ["LiH_R1p6_HF", "LiH_R1p6_ADAPT2", "LiH_R1p6_ADAPT3", "LiH_R1p6_ADAPT4", "BeH2_ADAPT2", "BeH2_ADAPT4", "BeH2_ADAPT7", "BeH2_ADAPT9"]
APRIORI = [("II-0", ("step4", "II-0 safe, bound start")), ("II-A", ("step5", "II-A data, safe, bound start")),
           ("M1 seq", ("step5", "M1 seq, safe, bound start")), ("M2", ("step5", "M2 safe, bound start")),
           ("M2 marg.", ("step5", "M2 marginal, bound start")), ("Ikh", ("step5", "Ikh reuse, FC, safe, bound start")),
           ("II-A + reuse", ("step5", "II-A data, safe + reuse, FC, bound start"))]


def sci(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if x < 1e4:
        return f"{x:,.0f}".replace(",", "{,}")
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def step_rows(case: str) -> dict[str, dict]:
    out = {}
    for stem in ("step4_learned_designs", "step5_external_baselines"):
        for r in read_rows(RUNS / case / f"{case}_{stem}.csv"):
            out[r["config"]] = r
    return out


def write(out: Path, name: str, text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text(text, newline="\n")
    print(f"wrote {out / name}.tex")


# --- a priori sign-aware headline -------------------------------------------------------------------------------------------------


def tab_apriori(out: Path) -> None:
    rng = np.random.default_rng(11)
    lines = [r"\begin{tabular}{@{}l" + "r" * 7 + r"@{}}", r"\toprule",
             "State & " + " & ".join(n for n, _ in APRIORI) + r" \\", r"\midrule"]
    print("\nA priori sign-aware family (mean context-shots; correct share in brackets when below 100%)")
    for case in CASES + NEW_STATES:
        trials = load_trials(RUNS, case)
        cells, means = [], {}
        for name, key in APRIORI:
            if key not in trials:
                cells.append("--")
                continue
            shots, correct = trials[key]
            means[name] = shots
            cells.append(sci(float(shots.mean())) + ("" if correct >= 0.995 else f" [{100 * correct:.0f}]"))
        if "II-A" not in means:
            continue
        lines.append(f"{SHORT.get(case, case)} & " + " & ".join(cells) + r" \\")
        extra = ""
        if "II-0" in means:
            point, lo, hi = ratio_interval(means["II-A"], means["II-0"], "mean", rng)
            extra = f"II-A/II-0 {point:.3f} [{lo:.3f}, {hi:.3f}]"
        base = [(n, means[n]) for n in ("M1 seq", "M2", "M2 marg.", "Ikh") if n in means]
        if base:
            n, best = min(base, key=lambda t: t[1].mean())
            point, lo, hi = ratio_interval(best, means["II-A"], "mean", rng)
            extra += f"; best baseline ({n}) / II-A {point:.2f} [{lo:.2f}, {hi:.2f}]"
        print(f"  {case:32s} " + "  ".join(f"{c:>18s}" for c in cells) + "   " + extra)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_apriori", "\n".join(lines) + "\n")


# --- anytime ---------------------------------------------------------------------------------------------------------------------


def tab_anytime(out: Path) -> None:
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"State & II-0 & II-0 anytime & II-A & II-A anytime & ratio II-A & missed (II-A, any.) \\", r"\midrule"]
    print("\nAnytime against per-round intervals, a priori start, sign-aware")
    for case in CASES + NEW_STATES:
        rows = step_rows(case)
        get = lambda label: rows.get(label)  # noqa: E731
        a0, a0t = get("II-0 safe, bound start"), get("II-0, safe, bound start, anytime")
        a1, a1t = get("II-A data, safe, bound start"), get("II-A data, safe, bound start, anytime")
        if not (a1 and a1t):
            continue
        ratio = float(a1t["shots_mean"]) / float(a1["shots_mean"])
        miss = f"{100 * float(a1['miscovered_rate']):.0f}\\%, {100 * float(a1t['miscovered_rate']):.0f}\\%" if a1.get("miscovered_rate") else "--"
        cell0 = sci(float(a0["shots_mean"])) if a0 else "--"
        cell0t = sci(float(a0t["shots_mean"])) if a0t else "--"
        lines.append(f"{SHORT.get(case, case)} & {cell0} & {cell0t} & {sci(float(a1['shots_mean']))} & {sci(float(a1t['shots_mean']))} & "
                     f"{ratio:.2f} & {miss} \\\\")
        print(f"  {case:32s} II-A {float(a1['shots_mean']):12,.0f} -> any. {float(a1t['shots_mean']):12,.0f} (x{ratio:.2f}); correct {a1t['correct_rate']}; "
              f"missed {a1.get('miscovered_rate')} -> {a1t.get('miscovered_rate')}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_anytime", "\n".join(lines) + "\n")


# --- sensitivity -----------------------------------------------------------------------------------------------------------------


def tab_sensitivity(out: Path) -> None:
    cases = ["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD", "LiH_R3p0_HF"]
    default = "II-A data, safe, bound start"
    variants = [("min shots 20", "II-A data, safe, bound start, min shots=20"), ("100", "II-A data, safe, bound start, min shots=100"),
                ("200", "II-A data, safe, bound start, min shots=200"), ("refit 1.25", "II-A data, safe, bound start, refit factor=1.25"),
                ("2", "II-A data, safe, bound start, refit factor=2"), ("3", "II-A data, safe, bound start, refit factor=3")]
    lines = [r"\begin{tabular}{@{}l" + "r" * (len(variants) + 2) + r"@{}}", r"\toprule",
             "State & default 50 / 1.5 & " + " & ".join(n for n, _ in variants) + r" & correct, all (\%) \\", r"\midrule"]
    print("\nThresholds (II-A, a priori start, sign-aware): mean cost relative to the default (50 shots, refit factor 1.5), correct share in brackets if < 100%")
    for case in cases:
        rows = step_rows(case)
        if default not in rows:
            continue
        base = float(rows[default]["shots_mean"])
        cells = []
        for _, label in variants:
            r = rows.get(label)
            cells.append("--" if r is None else f"{float(r['shots_mean']) / base:.2f}" + ("" if float(r["correct_rate"]) >= 0.995 else f" [{100 * float(r['correct_rate']):.0f}]"))
        shares = [float(rows[l]["correct_rate"]) for _, l in variants if l in rows] + [float(rows[default]["correct_rate"])]
        lines.append(f"{SHORT.get(case, case)} & {sci(base)} & " + " & ".join(cells) + f" & {100 * min(shares):.0f}" + r" \\")
        print(f"  {case:32s} default {base:12,.0f}  " + "  ".join(f"{c:>12s}" for c in cells) + f"   min correct {100 * min(shares):.0f}%")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_sensitivity_thresholds", "\n".join(lines) + "\n")

    # rho: fixed states (step5 rows stored as "<config>, rho=<rho>") and trajectories (phase4, tag sens_rho<r>)
    rhos = ["0.05", "0.1", "0.2", "0.3"]
    fixed_rhos = ["0", "0.05", "0.2", "0.3"]  # fixed states: 0 is the exact identification of the main study
    methods = [("II-0", "II-0, safe, bound start", "II-0 safe, bound start"), ("II-A", "II-A data, safe, bound start", None),
               ("M1 seq", "M1 seq, safe, bound start", None), ("M2", "M2 safe, bound start", None)]
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(fixed_rhos) + r"@{}}", r"\toprule", "State & method & " + " & ".join(f"$\\rho={r}$" for r in fixed_rhos) + r" \\", r"\midrule"]
    print("\nrho, fixed states (mean context-shots; rho = 0 is exact identification)")
    for case in cases:
        rows = step_rows(case)
        for name, label, exact_label in methods:
            cells = []
            for rho in fixed_rhos:
                r = rows.get(exact_label or label) if rho == "0" else rows.get(f"{label}, rho={rho}")
                cells.append("--" if r is None else sci(float(r["shots_mean"])))
            if all(c == "--" for c in cells):
                continue
            lines.append(f"{SHORT.get(case, case)} & {name} & " + " & ".join(cells) + r" \\")
            print(f"  {case:32s} {name:8s} " + "  ".join(f"{c:>16s}" for c in cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_sensitivity_rho_fixed", "\n".join(lines) + "\n")

    print("\nrho, trajectories (median total selection shots to chemical accuracy; share reaching it)")
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(rhos) + r"@{}}", r"\toprule", "System & method & " + " & ".join(f"$\\rho={r}$" for r in rhos) + r" \\", r"\midrule"]
    for case in ("H4_square_eq_side1p0_HF", "LiH_R3p0_HF"):  # stretched H4 stalls by symmetry before chemical accuracy: left out
        for method in ("II-0 safe", "II-A data, safe"):
            cells = []
            for rho in rhos:
                files = ([RUNS / case / "phase4" / f"{case}_phase4_trajectories.csv", *sorted((RUNS / case / "phase4").glob(f"{case}_phase4_*_trajectories.csv"))]
                         if rho == "0.1" else sorted((RUNS / case / "phase4").glob(f"{case}_phase4_sens_rho{rho}*_trajectories.csv")))
                totals, reached = [], []
                for path in files:
                    if rho == "0.1" and ("sens_" in path.name or "fixed_" in path.name):
                        continue
                    for r in read_rows(path):
                        if r["method"] == method:
                            totals.append(float(r["total_shots"]))
                            reached.append(r["reached"] == "True")
                    if totals:
                        break
                cells.append("--" if not totals else f"{sci(float(np.median(totals)))} ({100 * np.mean(reached):.0f}\\%)")
            if all(c == "--" for c in cells):
                continue
            lines.append(f"{OPT_NAMES.get(case, case)} & {method} & " + " & ".join(cells) + r" \\")
            print(f"  {case:32s} {method:16s} " + "  ".join(f"{c:>24s}" for c in cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_sensitivity_rho_trajectories", "\n".join(lines) + "\n")


# --- scaling ---------------------------------------------------------------------------------------------------------------------


CLASSICAL_STATE = {"H4_square_eq_side1p0_CISD": "H4_square_eq_side1p0_CISD", "LiH_R3p0_HF": "LiH_R3p0_HF", "LiH_R1p6_HF": "LiH_R1p6_HF",
                   "BeH2_HF": "BeH2_ADAPT4", "H2O_eq_HF": "H2O_eq_CISD"}


def classical_seconds(case: str) -> float:
    """Mean classical seconds of one II-A selection (designs, statistics, allocation) on a fixed state of the molecule, from the trial files."""
    from merge_trials import read_trials

    state = CLASSICAL_STATE.get(case)
    if state is None:
        return float("nan")
    for step, label in (("step5", "II-A data, safe, bound start"), ("step5", "II-A data, safe"), ("step4", "II-A data")):
        try:
            rows = read_trials(RUNS, state, step).get(label)
        except (SystemExit, FileNotFoundError):
            rows = None
        if rows and "design_seconds" in rows[0]:
            return float(np.mean([float(r["design_seconds"]) + float(r.get("statistics_seconds", 0) or 0) + float(r.get("allocation_seconds", 0) or 0) for r in rows]))
    return float("nan")


def dash(x: float) -> str:
    return "--" if not np.isfinite(x) else f"{x:.0f}"


def tab_scaling(out: Path) -> None:
    rows = []
    for folder in ("paper_a", "paper_a_n2", "paper_a_dz", "paper_a_dz11o", "paper_a_631g"):
        rows += read_rows(RUNS / folder / "scaling.csv")
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    from outputs import write_csv
    write_csv(RUNS / "paper_a" / "scaling_all.csv", columns, [{k: r.get(k, "") for k in columns} for r in rows])
    order = {"H4_square_eq_side1p0_CISD": 0, "LiH_R3p0_HF": 1, "LiH_R1p6_HF": 2, "H6_chain_HF": 3, "BeH2_HF": 4, "H2O_eq_HF": 5,
             "H2O_dz8o_eq_HF": 6, "N2_HF": 7, "H2O_dz11o_eq_HF": 8, "H2O_631g_eq_HF": 9}
    rows.sort(key=lambda r: order.get(r["case"], 99))
    names = {"H4_square_eq_side1p0_CISD": "H$_4$", "LiH_R3p0_HF": "LiH 3.0", "LiH_R1p6_HF": "LiH 1.6", "H6_chain_HF": "H$_6$ chain",
             "BeH2_HF": "BeH$_2$", "H2O_eq_HF": "H$_2$O STO-3G", "N2_HF": "N$_2$ STO-3G",
             "H2O_dz8o_eq_HF": "H$_2$O cc-pVDZ (8 orbitals)", "H2O_dz11o_eq_HF": "H$_2$O cc-pVDZ (11 orbitals)", "H2O_631g_eq_HF": "H$_2$O 6-31G"}
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"System & qubits & $K$ & Pauli products & contexts & CZ per circuit & coordinates per generator & II-A, one selection (s) \\", r"\midrule"]
    print("\nScaling (structure only)")
    for r in rows:
        f = lambda k: float(r[k])  # noqa: E731
        lines.append(f"{names.get(r['case'], r['case'])} & {int(f('qubits'))} & {int(f('generators'))} & {int(f('universal_support')):,} & "
                     f"{int(f('library_contexts')):,} & {f('two_qubit_mean'):.1f} & {f('coordinates_per_generator_mean'):,.0f} & "
                     f"{dash(classical_seconds(r['case']))} \\\\".replace(",", "{,}"))
        print(f"  {r['case']:28s} {int(f('qubits')):3d}q  K {int(f('generators')):4d}  support {int(f('universal_support')):9,d}  contexts {int(f('library_contexts')):7,d}  "
              f"CZ {f('two_qubit_mean'):5.1f}  coords/gen {f('coordinates_per_generator_mean'):10,.0f}  II-A selection {classical_seconds(r['case']):7.0f} s  "
              f"II-A coordinates {f('design_iia_coordinates_s'):7.1f}s  rss {f('peak_rss_gb'):.1f} GB")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_scaling", "\n".join(lines) + "\n")


# --- reproduction and sampled optimiser ----------------------------------------------------------------------------------------


def tab_reproduction(out: Path) -> None:
    rows = {}
    for tag in ("reuse", "noreuse"):
        for r in read_rows(RUNS / "paper_a" / f"huang_izmaylov_reproduction_{tag}.csv"):
            rows.setdefault(r["case"], {})[tag] = r
    if not rows:
        return
    names = {"H4_chain1p0_HF": "H$_4$", "LiH_R1p0_HF": "LiH", "BeH2_R1p0_HF": "BeH$_2$"}
    fresh = any("noreuse" in d for d in rows.values())  # the column appears only once the no-reuse run exists
    lines = [r"\begin{tabular}{@{}l" + "r" * (2 + fresh) + r"@{}}", r"\toprule",
             r"System & published (\%) & reproduced, samples kept (\%)" + (r" & reproduced, fresh each round (\%)" if fresh else "") + r" \\", r"\midrule"]
    print("\nReproduction of Huang and Izmaylov's Table I (UCCSD)")
    for case, d in rows.items():
        pub = float(next(iter(d.values()))["published_reduction_pct"])
        cell = lambda t: "--" if t not in d else f"{float(d[t]['reduction_pct_mean']):.1f} $\\pm$ {float(d[t]['reduction_pct_sd']):.1f}"  # noqa: E731
        lines.append(f"{names.get(case, case)} & {pub:.1f} & {cell('reuse')}" + (f" & {cell('noreuse')}" if fresh else "") + r" \\")
        print(f"  {case:16s} published {pub:5.1f}   kept {cell('reuse'):>14s}   fresh {cell('noreuse'):>14s}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_reproduction", "\n".join(lines) + "\n")


OPT_NAMES = {"H4_square_eq_side1p0_HF": "H$_4$ 1.0~\\AA", "LiH_R3p0_HF": "LiH 3.0~\\AA"}


def tab_sampled_optimizer(out: Path) -> None:
    rows = read_rows(RUNS / "paper_a" / "sampled_optimizer_summary.csv")
    if not rows:
        return
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"System & $\varepsilon_E$ (mHa) & energies & sampled shots & model $\eta=2$ / $\eta=4$ & final $\Delta E$ (mHa) \\", r"\midrule"]
    print("\nSampled optimiser against C_opt")
    for r in rows:
        eps = float(r["eps_energy"]) * 1e3
        lines.append(f"{OPT_NAMES.get(r['case'], r['case'])} & {eps:g} & {float(r['evaluations_mean']):.0f} & {sci(float(r['sampled_shots_mean']))} & "
                     f"{float(r['ratio_to_model_r2']):.2f} / {float(r['ratio_to_model_r4']):.2f} & {1e3 * float(r['final_error_mean']):.2f} \\\\")
        print(f"  {r['case']:28s} eps_E {eps:5.2f} mHa: {float(r['sampled_shots_mean']):.3e} shots = {float(r['ratio_to_model_r2']):.2f} x model(r=2), "
              f"{float(r['ratio_to_model_r4']):.2f} x model(r=4); final dE {1e3 * float(r['final_error_mean']):.2f} mHa")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_sampled_optimizer", "\n".join(lines) + "\n")


DIAG_STATES = {"LiH_R3p0_ADAPT5": "LiH 3.0~\\AA, ADAPT 5", "LiH_R1p6_ADAPT4": "LiH 1.6~\\AA, ADAPT 4", "H2O_eq_CISD": "H$_2$O eq, CISD"}


def tab_interval_diag(out: Path) -> None:
    """Why do intervals miss: per state and method, the minimum shots, the cost, the share of correct selections and of trials with
    a missed interval, the first missed round, the share with a zero-width interval, and the worst intervals' ratio of estimated to
    true standard deviation and thinnest context (the miss_* fields of the trial files)."""
    from merge_trials import read_trials

    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r"State & method & min shots & trials & mean shots & correct (\%) & missed (\%) & first round & zero-width (\%) & sd ratio \\", r"\midrule"]
    print("\nInterval diagnostics (a priori start, sign-aware)")
    for case, name in DIAG_STATES.items():
        try:
            trials = read_trials(RUNS, case, "step5")
        except (SystemExit, FileNotFoundError):
            continue
        first = True
        # the full set (every minimum from 50 to 1000, II-0 and II-A, per round and anytime) is in the printed log and the trial files
        for family, tmpl, shots in (("II-A", "II-A data, safe, bound start, min shots={n}, diagnose", (50, 200, 1000)),
                                    ("II-A anytime", "II-A data, safe, bound start, anytime, min shots={n}, diagnose", (200, 1000)),
                                    ("II-0", "II-0, safe, bound start, min shots={n}, diagnose", (50, 1000)),
                                    ("II-0 anytime", "II-0, safe, bound start, anytime, min shots={n}, diagnose", (200,))):
            for n in (50, 100, 200, 400, 1000):
                if n not in shots:
                    rows = trials.get(tmpl.format(n=n))
                    if rows and "miss_arms" in rows[0]:
                        f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
                        missed = f("miscovered_rounds") > 0
                        ratio = f("miss_min_sdratio")[missed & (f("miss_min_sdratio") >= 0)]
                        print(f"  (not in the table) {case:18s} {family:13s} n={n:5d} trials {len(rows):3d}  shots {f('shots').mean():12,.0f}  correct {100 * f('correct').mean():3.0f}%  "
                              f"missed {100 * missed.mean():3.0f}%  sd ratio {np.median(ratio) if ratio.size else float('nan'):.3f}")
                    continue
                rows = trials.get(tmpl.format(n=n))
                if not rows or "miss_arms" not in rows[0]:
                    continue
                f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
                missed = f("miscovered_rounds") > 0
                ratio = f("miss_min_sdratio")[missed & (f("miss_min_sdratio") >= 0)]
                first_round = f("miss_first_round")[missed]
                zero = f("miss_zero_sd")[missed]
                zero_cell = "--" if not missed.any() else f"{100 * (zero > 0).mean():.0f}"
                first_cell = "--" if not missed.any() else f"{np.median(first_round):.0f}"
                ratio_cell = "--" if ratio.size == 0 else f"{np.median(ratio):.2f}"
                lines.append(f"{name if first else ''} & {family} & {n} & {len(rows)} & {sci(float(f('shots').mean()))} & {100 * f('correct').mean():.0f} & "
                             f"{100 * missed.mean():.0f} & {first_cell} & {zero_cell} & {ratio_cell} \\\\")
                first = False
                print(f"  {case:18s} {family:13s} n={n:5d} trials {len(rows):3d}  shots {f('shots').mean():12,.0f}  correct {100 * f('correct').mean():3.0f}%  "
                      f"missed {100 * missed.mean():3.0f}%  first round {np.median(first_round) if missed.any() else float('nan'):5.1f}  "
                      f"zero-sd {100 * (f('miss_zero_sd')[missed] > 0).mean() if missed.any() else float('nan'):4.0f}%  "
                      f"sd ratio {np.median(ratio) if ratio.size else float('nan'):.3f}  thin ctx {np.median(f('miss_min_shots')[missed]) if missed.any() else float('nan'):6.0f}  "
                      f"best arm {100 * f('miss_best_arm')[missed].mean() if missed.any() else float('nan'):3.0f}%")
        if not first:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out, "tab_interval_diag", "\n".join(lines) + "\n")


TABLES = {"tab_apriori": tab_apriori, "tab_anytime": tab_anytime, "tab_sensitivity": tab_sensitivity, "tab_scaling": tab_scaling,
          "tab_reproduction": tab_reproduction, "tab_sampled_optimizer": tab_sampled_optimizer, "tab_interval_diag": tab_interval_diag}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "reports" / "generated")
    parser.add_argument("--only", nargs="+", choices=list(TABLES), default=list(TABLES))
    args = parser.parse_args()
    for name in args.only:
        TABLES[name](args.out)


if __name__ == "__main__":
    main()
