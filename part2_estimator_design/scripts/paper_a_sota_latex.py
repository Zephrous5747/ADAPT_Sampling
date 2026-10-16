#!/usr/bin/env python3
"""LaTeX tables of the SOTA-baseline comparison, generated from ``runs/paper_a/sota_*.csv``.

Run ``paper_a_sota_tables.py`` first (it rebuilds the CSVs from the stored trial summaries).  The
tables are written to ``reports/generated/sota_tables.tex`` (``\\input`` them, or paste them) and a
plain-text copy is printed; nothing is copied by hand.  A cell is the mean cost with its ratio to
II-0 in brackets; a dash means the configuration was not run (or was refused: the reuse rows of a
cached problem whose orbitals do not match a rebuilt Hamiltonian).

    python scripts/paper_a_sota_latex.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"),
         ("LiH_R3p0_HF", "LiH HF"), ("H2O_eq_CISD", r"H$_2$O eq CISD"), ("H2O_stretch_CISD", r"H$_2$O str CISD")]
FIXED_ROWS = [
    ("external baselines", None),
    ("M1 static", "M1 static (exact gap, exact variances)"),
    ("M1 seq (no elimination)", "M1 seq (no elimination)"),
    ("M2, Successive Elimination", "M2 independent, marginal rule"),
    ("M2, pairwise rule", "M2 independent, pairwise rule"),
    ("this work", None),
    ("II-0 (shared BAI)", "II-0"),
    ("II-A (learned splitting)", "II-A data"),
    ("with the energy data of the last VQE evaluation (FC)", None),
    ("shot reuse of Ikhtiarudin \\textit{et al.}", "Ikh reuse, FC"),
    ("II-0 + reuse", "II-0 + reuse"),
    ("II-A + reuse, forced assignment", "II-A data + reuse"),
    ("II-A + reuse, home start", "II-A data + reuse (home)"),
    ("sign-aware rule (ratios to II-0 safe)", None),
    ("M1 seq", "M1 seq safe (oracle start)"),
    ("II-0", "II-0 safe"),
    ("II-A", "II-A safe"),
    ("shot reuse of Ikhtiarudin \\textit{et al.}", "Ikh reuse safe"),
    ("II-0 + reuse", "II-0 safe + reuse"),
    ("II-A + reuse, forced assignment", "II-A safe + reuse"),
    ("II-A + reuse, home start", "II-A safe + reuse (home)"),
]
TRAJ_CASES = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0"), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0"),
              ("LiH_R3p0_HF", "LiH"), ("H2O_eq_HF", r"H$_2$O eq"), ("H2O_stretch_HF", r"H$_2$O str")]
TRAJ_ROWS = [
    ("external baselines", None),
    ("M1 static", "M1 static"), ("M1 seq", "M1 seq, safe"), ("M2, Successive Elimination", "M2 marginal"),
    ("M2, sign-aware", "M2 safe"),
    ("this work", None),
    ("II-0", "II-0 safe"), ("II-A", "II-A data, safe"), ("II-E", "II-A data, contrast"),
    ("with the energy data of the last VQE evaluation (FC)", None),
    ("shot reuse of Ikhtiarudin \\textit{et al.}", "Ikh reuse, FC"),
    ("II-0 + reuse", "II-0 safe + reuse, FC"), ("II-A + reuse", "II-A data, safe + reuse, FC"),
]


def sci(x: float) -> str:
    """Thousands-separated integer below 1e5, otherwise mantissa times a power of ten (LaTeX)."""
    if x != x:
        return "--"
    if x < 1e5:
        return f"{x:,.0f}".replace(",", r"\,")
    exponent = len(f"{x:.0f}") - 1
    return rf"${x / 10 ** exponent:.2f}\times10^{{{exponent}}}$"


# Not stored with the runs: the molecular basis, the pool of the main tables and the confidence level (``delta`` of Part I).
SETUP = r"STO-3G, UCCSD pool, $\delta=0.05$"
MOLECULES = (("H4", r"H$_4$"), ("LiH", "LiH"), ("H2O", r"H$_2$O"))


def count_text(counts) -> str:
    """``100`` or ``100 to 200`` for the numbers in ``counts``."""
    counts = sorted(set(counts))
    return f"{counts[0]}" if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"


def per_molecule(counts_by_case: dict, unit: str) -> str:
    """``200 <unit> on H$_4$, 100 to 200 on LiH, 100 on H$_2$O`` from {case: counts} (the molecule is the case name up to ``_``)."""
    groups = collections.defaultdict(list)
    for case, counts in counts_by_case.items():
        groups[case.split("_")[0]] += list(counts)
    parts = [f"{count_text(groups[m])} on {name}" for m, name in MOLECULES if groups.get(m)]
    parts[0] = parts[0].replace(" on ", f" {unit} on ", 1)
    return ", ".join(parts)


def fixed_trials(runs: Path, cases, rho: str = "0", settings=("oracle start", "sign-aware")) -> dict:
    """{case: set of trial counts} of ``sota_fixed_state.csv`` for the cases at this rho and these settings."""
    counts = collections.defaultdict(set)
    path = runs / "paper_a" / "sota_fixed_state.csv"
    if path.exists():
        for r in csv.DictReader(path.open()):
            if r["case"] in cases and r["rho"] == rho and r["setting"] in settings:
                counts[r["case"]].add(int(r["trials"]))
    return dict(counts)


def trajectory_counts(runs: Path, cases) -> dict:
    """{case: set of trajectory counts per method} of ``sota_trajectories.csv``."""
    counts = collections.defaultdict(set)
    path = runs / "paper_a" / "sota_trajectories.csv"
    if path.exists():
        for r in csv.DictReader(path.open()):
            if r["case"] in cases:
                counts[r["case"]].add(int(r["trajectories"]))
    return dict(counts)


def table(rows, columns, cells, caption, label, header_note, tabcolsep: str = "4pt") -> str:
    cols = "l" + "r" * len(columns)
    out = ["\\begin{table*}[!htb]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\footnotesize\\setlength{{\\tabcolsep}}{{{tabcolsep}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
           f"{header_note} & " + " & ".join(c[1] for c in columns) + " \\\\", "\\midrule"]
    for name, key in rows:
        if key is None:
            out.append(f"\\multicolumn{{{len(columns) + 1}}}{{l}}{{\\emph{{{name}}}}} \\\\")
            continue
        out.append(name + " & " + " & ".join(cells.get((c[0], key), "--") for c in columns) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out)


POOL_ROWS = [("uccsd", "UCCSD"), ("qeb", "QEB"), ("qubit", "qubit-ADAPT"), ("gsd_qeb", "generalized QEB"),
             ("gsd_qubit", "generalized qubit"), ("ceo", "OVP-CEO")]
POOL_CASES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_eq_side1p0_HF", r"H$_4$ 1.0 HF"),
              ("LiH_R3p0_HF", "LiH HF")]


def pools_table(runs: Path) -> str | None:
    census_path = runs / "step6_pool_census.csv"
    if not census_path.exists():
        return None
    census = {(r["case"], r["pool"]): r for r in csv.DictReader(census_path.open())}
    shots = {}
    fixed_path = runs / "paper_a" / "sota_fixed_state.csv"
    if fixed_path.exists():
        for r in csv.DictReader(fixed_path.open()):
            if r["rho"] == "0.1" and r["setting"] == "oracle start":
                base, _, pool = r["case"].partition("@")
                shots[(base, pool, r["method"])] = float(r["shots_mean"])
    lines = [r"\begin{table*}[!htb]", r"\centering\small",
             r"\caption{Operator pools (CISD and HF states of H$_4$ at 1.0~\AA, HF state of LiH; STO-3G, $\delta=0.05$, "
             + count_text(c for v in fixed_trials(runs, [f"{case}@{pool}" for case, _ in POOL_CASES for pool, _ in POOL_ROWS], "0.1",
                                                   ("oracle start",)).values() for c in v)
             + r" trials per row). $K$: generators; $\lvert\mathcal B_0\rvert$: "
             r"parent support; $N$: parent FC contexts; uses: gradients per Pauli product. Planning bounds (exact "
             r"gradients and variances) are given as ratios M1/M3 and M2/M3 where the leader is unique; qubit-type pools have exactly "
             r"tied leaders. Right: sampled outcomes with estimated variances and a $\rho=0.1$ stop, mean context-shots; "
             r"a dash: no planning bound, because the leaders are tied.}", r"\label{tab:q8pools}", r"\footnotesize\setlength{\tabcolsep}{3pt}"]
    lines.append(r"\begin{tabular}{llrrrrrrrrrr}")
    lines += [r"\toprule", r"state & pool & $K$ & $\lvert\mathcal B_0\rvert$ & $N$ & uses & M1/M3 & M2/M3 & II-0 & II-A & M1 seq & M2 pairw. \\", r"\midrule"]
    for case, case_label in POOL_CASES:
        for pool, pool_label in POOL_ROWS:
            row = census.get((case, pool))
            if row is None:
                continue
            ratio = (f"{float(row['M1/M3']):.2f}", f"{float(row['M2/M3']):.2f}") if row.get("M1/M3") else ("--", "--")

            def cell(method):
                value = shots.get((case, pool, method))
                return sci(value) if value is not None else "--"

            lines.append(" & ".join([case_label, pool_label, row["K"], sci(float(row["support_B0"])), row["parent_contexts"],
                                     f"{float(row['uses_per_pauli']):.2f}", *ratio, cell("II-0"), cell("II-A data"),
                                     cell("M1 seq (no elimination)"), cell("M2 independent, pairwise rule")]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


BEST_STATES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0 CISD"),
               ("LiH_R3p0_HF", "LiH HF"), ("LiH_R3p0_ADAPT3", r"LiH ADAPT$_3$"), ("LiH_R3p0_ADAPT5", r"LiH ADAPT$_5$"),
               ("H2O_eq_CISD", r"H$_2$O eq CISD"), ("H2O_stretch_CISD", r"H$_2$O str CISD"),
               ("H2O_eq_ADAPT11", r"H$_2$O eq ADAPT$_{11}$"), ("H2O_stretch_ADAPT8", r"H$_2$O str ADAPT$_8$")]
NO_FREE = {"oracle start": ["M1 static (exact gap, exact variances)", "M1 seq (no elimination)", "M2 independent, marginal rule",
                            "M2 independent, pairwise rule"],
           "sign-aware": ["M1 seq safe (oracle start)", "M2 safe (oracle start)"]}
FREE = {"oracle start": ["Ikh reuse, FC", "Ikh reuse, QWC"], "sign-aware": ["Ikh reuse safe"]}
OURS_NOFREE = {"oracle start": ("II-0", "II-A data"), "sign-aware": ("II-0 safe", "II-A safe")}
OURS_FREE = {"oracle start": ("II-0 + reuse", "II-A data + reuse", "II-A data + reuse (home)"),
             "sign-aware": ("II-0 safe + reuse", "II-A safe + reuse", "II-A safe + reuse (home)")}
SHORT = {"M1 static (exact gap, exact variances)": "M1 static", "M1 seq (no elimination)": "M1 seq",
         "M2 independent, marginal rule": "M2 SE", "M2 independent, pairwise rule": "M2 pairw.", "M1 seq safe (oracle start)": "M1 seq", "M2 safe (oracle start)": "M2 sign-aware",
         "Ikh reuse, FC": "reuse FC", "Ikh reuse, QWC": "reuse QWC", "Ikh reuse safe": "reuse FC"}


def best_table(runs: Path) -> str | None:
    fixed_path = runs / "paper_a" / "sota_fixed_state.csv"
    traj_path = runs / "paper_a" / "sota_trajectories.csv"
    if not fixed_path.exists():
        return None
    data = {}
    for r in csv.DictReader(fixed_path.open()):
        if r["rho"] == "0" and r["setting"] in NO_FREE and float(r["correct_rate"]) >= 0.95:
            data[(r["case"], r["setting"], r["method"])] = float(r["shots_mean"])

    def pick(case, setting, names):
        values = {n: data[(case, setting, n)] for n in names if (case, setting, n) in data}
        if not values:
            return None, None
        name = min(values, key=values.get)
        return name, values[name]

    lines = [r"\begin{table*}[!htb]", r"\centering\small",
             r"\caption{The strongest baseline against our methods, per state (mean context-shots; ratios of the baseline's cost to ours, "
             r"so a value above one is a win). " + SETUP + "; " + count_text(c for v in fixed_trials(runs, [s for s, _ in BEST_STATES]).values() for c in v)
             + " trials per state and " + count_text(c for v in trajectory_counts(runs, [s for s, _ in TRAJ_CASES]).values() for c in v)
             + r" trajectories per method. Without free data: the best of M1 static, M1 seq and M2 against II-0 and II-A. With the energy "
             r"data of the last VQE evaluation (1~mHa): the best of all baselines including the reuse baseline against the better of II-0 + "
             r"reuse and II-A + reuse. Rows marked sign-aware use the sign-aware rule throughout. Rows whose selections were correct in fewer "
             r"than 95\% of the trials are excluded.}", r"\label{tab:q8best}", r"\footnotesize\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lllrrrlrr}", r"\toprule",
             r"state & rule & best baseline & cost & II-0 & II-A & best, with data & cost & ours + data \\", r"\midrule"]
    for case, label in BEST_STATES:
        for setting, rule in (("oracle start", "pairwise"), ("sign-aware", "sign-aware")):
            nb, vb = pick(case, setting, NO_FREE[setting])
            nf, vf = pick(case, setting, NO_FREE[setting] + FREE[setting])
            if nb is None:
                continue
            o0, oa = OURS_NOFREE[setting]
            r0, ra = data.get((case, setting, o0)), data.get((case, setting, oa))
            fo = [data.get((case, setting, n)) for n in OURS_FREE[setting]]
            fo = [x for x in fo if x]
            if setting == "sign-aware" and ra is None:
                continue
            ratio = lambda a, b: f"{a / b:.2f}" if a and b else "--"  # noqa: E731
            lines.append(" & ".join([label, rule, SHORT[nb], sci(vb), ratio(vb, r0), ratio(vb, ra),
                                     SHORT[nf] if nf else "--", sci(vf) if vf else "--", ratio(vf, min(fo)) if fo and vf else "--"]) + r" \\")
    if traj_path.exists():
        lines.append(r"\midrule")
        tdata = collections.defaultdict(dict) if False else {}
        for r in csv.DictReader(traj_path.open()):
            tdata[(r["case"], r["method"])] = float(r["total_shots_median"])
        for case, label in TRAJ_CASES:
            base = {m: tdata[(case, m)] for m in ("M1 static", "M1 seq, safe", "M2 marginal", "M2 safe") if (case, m) in tdata}
            if not base:
                continue
            nb = min(base, key=base.get)
            free = dict(base)
            if (case, "Ikh reuse, FC") in tdata:
                free["Ikh reuse, FC"] = tdata[(case, "Ikh reuse, FC")]
            nf = min(free, key=free.get)
            ours = [tdata.get((case, m)) for m in ("II-0 safe + reuse, FC", "II-A data, safe + reuse, FC")]
            ours = [x for x in ours if x]
            short = {"M1 static": "M1 static", "M1 seq, safe": "M1 seq", "M2 marginal": "M2 SE", "M2 safe": "M2 sign-aware",
                     "Ikh reuse, FC": "reuse FC"}
            lines.append(" & ".join([label + " trajectory", "sign-aware", short[nb], sci(base[nb]),
                                     f"{base[nb] / tdata[(case, 'II-0 safe')]:.2f}" if (case, "II-0 safe") in tdata else "--",
                                     f"{base[nb] / tdata[(case, 'II-A data, safe')]:.2f}" if (case, "II-A data, safe") in tdata else "--",
                                     short[nf], sci(free[nf]), f"{free[nf] / min(ours):.2f}" if ours else "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


EPS_CASE = "H4_square_eq_side1p0_CISD"


def eps_table(runs: Path) -> str | None:
    """Reuse rows against the precision of the last energy evaluation (H4 1.0 CISD)."""
    import json

    def read(directory: Path) -> dict:
        path = directory / EPS_CASE / f"{EPS_CASE}_step5_external_baselines.csv"
        return {r["config"]: float(r["shots_mean"]) for r in csv.DictReader(path.open())} if path.exists() else {}

    def credit(directory: Path):
        path = directory / EPS_CASE / f"{EPS_CASE}_step5_meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("reuse_libraries", {}).get("fc", {}).get("credit_shots")

    sweep = []
    for directory in sorted(runs.parent.glob("runs_energy_error/e*")):
        eps = float(directory.name[1:])
        sweep.append((eps, read(directory), credit(directory)))
    main = read(runs)
    if main and (runs / EPS_CASE / f"{EPS_CASE}_step5_meta.json").exists():
        meta = json.loads((runs / EPS_CASE / f"{EPS_CASE}_step5_meta.json").read_text())
        sweep.append((meta.get("energy_error", 1e-3), main, meta.get("reuse_libraries", {}).get("fc", {}).get("credit_shots")))
    if not sweep:
        return None
    sweep = sorted({e: (e, r, c) for e, r, c in sweep}.values())
    lines = [r"\begin{table}[!htb]", r"\centering\small",
             r"\caption{Reuse of the energy measurement against the standard error $\varepsilon_E$ of the last energy evaluation "
             r"(H$_4$ 1.0~\AA, CISD; mean context-shots, 100 trials, 200 at 1~mHa). The data held for free scale as "
             r"$\varepsilon_E^{-2}$. Last row: the same methods without free data.}", r"\label{tab:q8eps}",
             r"\footnotesize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$\varepsilon_E$ (mHa) & free shots & Ikhtiarudin FC & II-0 + reuse & II-A + reuse \\", r"\midrule"]
    for eps, rows, held in sweep:
        lines.append(" & ".join([f"{1000 * eps:g}", sci(float(held)) if held else "--",
                                 sci(rows.get("Ikh reuse, FC", float("nan"))), sci(rows.get("II-0 + reuse, FC", float("nan"))),
                                 sci(rows.get("II-A data + reuse, FC", float("nan")))]) + r" \\")
    step4_path = runs / EPS_CASE / f"{EPS_CASE}_step4_learned_designs.csv"
    none = {**read(runs), **({r["config"]: float(r["shots_mean"]) for r in csv.DictReader(step4_path.open())}
                              if step4_path.exists() else {})}
    lines += [r"\midrule", " & ".join(["none", "0", sci(none.get("M1 seq", float("nan"))), sci(none.get("II-0 estimated", float("nan"))),
                                         sci(none.get("II-A data", float("nan")))]) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def splice(paper: Path, text: str) -> None:
    begin, end = "% BEGIN GENERATED sota_tables", "% END GENERATED sota_tables"
    source = paper.read_text(encoding="utf-8")
    if begin not in source or end not in source:
        raise SystemExit(f"{paper}: markers {begin!r} / {end!r} not found")
    head, rest = source.split(begin, 1)
    _, tail = rest.split(end, 1)
    with paper.open("w", encoding="utf-8", newline="\n") as handle:  # the paper is LF-only; write_text would give CRLF on Windows
        handle.write(head + begin + "\n" + text + "\n" + end + tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("../reports/generated/sota_tables.tex"))
    parser.add_argument("--splice", type=Path, default=None,
                        help="replace the text between the BEGIN/END GENERATED sota_tables markers of this .tex file")
    args = parser.parse_args()
    fixed_path = args.runs / "paper_a" / "sota_fixed_state.csv"
    traj_path = args.runs / "paper_a" / "sota_trajectories.csv"
    parts = []
    if fixed_path.exists():
        cells = {}
        for r in csv.DictReader(fixed_path.open()):
            if r["setting"] not in ("oracle start", "sign-aware") or r["rho"] != "0":
                continue
            ratio = float(r["vs_II0"])
            wrong = "" if float(r["correct_rate"]) >= 0.995 else f"{{\\,\\scriptsize[{100 * float(r['correct_rate']):.0f}\\%]}}"
            cells[(r["case"], r["method"])] = f"{sci(float(r['shots_mean']))} ({ratio:.2f}){wrong}"
        parts.append(table(FIXED_ROWS, CASES, cells,
                           "Fixed states, sampled outcomes, covariances and radii estimated from the shots (exact "
                           "identification, starting radius $\\max_i\\lvert g_i\\rvert$): mean context-shots and, in "
                           "brackets, the ratio to II-0. " + SETUP + "; "
                           + per_molecule(fixed_trials(args.runs, [c for c, _ in CASES]), "trials") + ". M1 static is handed the exact gap and variances. The "
                           "``reuse'' rows hold the shots of one energy evaluation to 1~mHa for free (they are part "
                           "of $C_{\\mathrm{opt}}$) and charge only new shots. In square brackets, the percentage of "
                           "correct selections when it is below 100\\%.", "tab:q8fixed", "Method", tabcolsep="2.4pt"))
    if traj_path.exists():
        cells = {}
        for r in csv.DictReader(traj_path.open()):
            cells[(r["case"], r["method"])] = f"{sci(float(r['total_shots_median']))} ({float(r['vs_II0_safe']):.2f})"
        parts.append(table(TRAJ_ROWS, TRAJ_CASES, cells,
                           "Measured ADAPT trajectories to chemical accuracy ($\\rho=0.1$, a-priori starting radius, "
                           "sign-aware rule): median cumulative selection context-shots and, in brackets, the ratio to "
                           "II-0. " + SETUP + "; " + per_molecule(trajectory_counts(args.runs, [c for c, _ in TRAJ_CASES]), "trajectories per method")
                           + ". M1 static uses the exact gradient scale and variances."
                           + (" A dash means that the configuration was not run."
                              if any(key is not None and (case[0], key) not in cells for _, key in TRAJ_ROWS for case in TRAJ_CASES)
                              else ""), "tab:q8traj", "Method"))
    best = best_table(args.runs)
    if best:
        parts.insert(0, best)
    pools = pools_table(args.runs)
    if pools:
        parts.append(pools)
    eps = eps_table(args.runs)
    if eps:
        parts.append(eps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("% generated by scripts/paper_a_sota_latex.py from runs/paper_a/sota_*.csv; do not edit\n"
                        + "\n\n".join(parts) + "\n")
    if args.splice is not None:
        splice(args.splice, "\n\n".join(parts))
        print(f"spliced the tables into {args.splice}")
    print(f"wrote {args.out}")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
