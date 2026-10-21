#!/usr/bin/env python3
"""Tables of ``reports/energy_trajectory_findings.tex`` (written to ``reports/generated/findings_*.tex``).

Every number of the tables is read from the stored runs, so the tables follow the data:

    python scripts/energy_findings_tables.py          # then: python scripts/paper_a_inline.py --draft ../reports/energy_trajectory_findings.tex

Inputs (all under ``part2_estimator_design``): ``runs/features/report`` (comparison with the colleague's report, cost totals, non-inferiority, rho maps),
``runs/features_ext``, ``runs/energy_policies`` (policy report), ``runs/features_q10`` (replays of certified and fixed-budget runs), the local tolerance tests
``runs/energy_policies_gtol*`` and ``runs/energy_policies_smoke`` (WSL build; only used when present), the cluster tolerance jobs ``runs/*_gtol1e-5_cl`` and
``runs/*_gtol1e-3_cl`` (policy grid and replays of the sweeps), the local replays ``runs/features_q10_gtol*`` of the Q10 runs, the Phase 4 files of the LiH min-shots runs,
and the logs of the gate job (``cluster/logs_feat``).  A missing input leaves its cells empty (``--``).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent / "reports"
GEN = REPORTS / "generated"
RUNS = ROOT / "runs"
sys.path.insert(0, str(Path(__file__).resolve().parent))

SYS = {"H4_square_eq_side1p0_HF": r"H$_4$ 1.0~\AA", "H4_square_stretch_side2p0_HF": r"H$_4$ 2.0~\AA", "LiH_R3p0_HF": r"LiH 3.0~\AA",
       "LiH_R1p6_HF": r"LiH 1.6~\AA", "BeH2_HF": r"BeH$_2$", "H2O_eq_HF": r"H$_2$O eq", "H2O_stretch_HF": r"H$_2$O str",
       "H2O_dz8o_eq_HF": r"H$_2$O cc-pVDZ window eq", "H2O_dz8o_str_HF": r"H$_2$O cc-pVDZ window str"}
SHORT = {"H4 1.0": "H4_square_eq_side1p0_HF", "H4 2.0": "H4_square_stretch_side2p0_HF", "LiH 3.0": "LiH_R3p0_HF", "BeH2": "BeH2_HF",
         "H2O eq": "H2O_eq_HF", "H2O str": "H2O_stretch_HF", "H2O dz8o eq": "H2O_dz8o_eq_HF", "H2O dz8o str": "H2O_dz8o_str_HF"}
CASE_OF_LABEL = dict(SHORT)
LABEL_OF_CASE = {v: k for k, v in SHORT.items()}


def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and not np.isfinite(x))


def sci(x, digits: int = 1) -> str:
    if not ok(x) or x == 0:
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return r"$" + f"{m:.{digits}f}" + r"\times10^{" + str(e) + r"}$"


def num(x, d: int = 2) -> str:
    return f"{x:.{d}f}" if ok(x) else "--"


def pct(x, d: int = 0) -> str:
    return (f"{x:.{d}f}" + r"\%") if ok(x) else "--"


def interval(r, lo, hi, d: int = 2) -> str:
    return f"{r:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if ok(r) and ok(lo) and ok(hi) else "--"


def table(name: str, caption: str, label: str, colspec: str, lines: list[str], note: str | None = None) -> None:
    """One table float: a short title (caption), the tabular, and a table note that defines the columns, the statistics and the runs.

    Publication format (ACS): the caption is one phrase or sentence that is understandable alone, the details are footnotes.  The note is
    typeset with ``\\tabnote`` (defined in the preamble of the report and of the SI).  Labels and symbols: Table ``tab:notation`` of the SI.
    """
    text = "\n".join([r"\begin{table}[!htb]", r"\centering\small", r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\fitcol{%", r"\begin{tabular}{" + colspec + "}", r"\toprule",
                      *lines, r"\bottomrule", r"\end{tabular}}", *([r"\tabnote{" + note + "}"] if note else []), r"\end{table}", ""])
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", name)


def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# ------------------------------------------------------------------------------------------------ reproduction scorecard and gate

def repro() -> None:
    txt = (RUNS / "features/report/compare_tzs.txt").read_text() if (RUNS / "features/report/compare_tzs.txt").exists() else ""
    cells = {k: (int(a), int(b)) for k, a, b in re.findall(r"^(S1|Table 1|Table 2|Table 3): (\d+)/(\d+) numbers agree", txt, re.M)}
    st = read_csv(RUNS / "features/report/compare_stats.csv")
    lines = [r"Table of her report & cells & agree & what differs \\", r"\midrule"]

    def row(label, key, note):
        a, b = cells.get(key, (None, None))
        lines.append(f"{label} & {b if b is not None else '--'} & {a if a is not None else '--'} & {note} \\\\")

    row(r"S1: gradient against one-parameter and full re-optimization lowering, five systems", "S1",
        r"Spearman coefficients, at most $0.03$ (H$_4$ 2.0~\AA: $0.95$ against $0.97$, full $0.83$ against $0.86$)")
    row(r"Table 1: correlation of the gradient estimates, five systems, II-0 and II-A", "Table 1",
        r"pairs 75 against 73 (H$_4$ 2.0~\AA); $r$, $\chi$ and $\rho_s$ within $0.04$")
    lines.append(r"\midrule")
    lines.append(r"Table 2, statistical comparison & \multicolumn{3}{l}{her number inside the 95\% range of $n$-run subsets of our 48 (H$_4$) and 24 (LiH) runs} \\")
    if st is not None:
        names = {"shots": "median selection shots", "exact_best": "share of selections that were the exact best", "evals": "mean optimizer evaluations (1e-9)"}
        for q in ("shots", "exact_best", "evals"):
            g = st[st.quantity == q]
            lines.append(f"\\quad {names[q]} & {len(g)} & {int(g.inside.sum())} & "
                         f"median of hers / ours $= {float((g.hers / g.ours_all).median()):.2f}$ \\\\")
    lines.append(r"\midrule")
    row(r"Table 3, digit by digit (strict tolerance)", "Table 3", r"almost all in the evaluation and $C_\mathrm{opt}$ ratios")
    row(r"Table 2, digit by digit (strict tolerance)", "Table 2", r"evaluation counts and their ratios to exact ADAPT; shots differ by the sampling error of 12 runs")
    cap = (r"Agreement of our re-run of the colleague's study with the numbers of her report. Digit by digit, cells agree when they match at the printed precision; "
           r"the sampled tables cannot match digit by digit because the random streams do not carry over between builds of the Hamiltonian (Sec.~\ref{sec:evalenv}). "
           r"The statistical rows test instead whether her statistic lies within the central 95\% of the same statistic over random subsets of her number of runs. "
           r"Shots and the exact-best share reproduce; the evaluation counts do not.")
    table("findings_repro", cap, "tab:find_repro", "p{6.2cm}rrp{6.0cm}", lines)


def corr() -> None:
    t = read_csv(RUNS / "features/report/table1_gradient_correlation_vs_energy.csv")
    if t is None:
        return
    lines = [r"System & Estimator & Pairs & Median $r$ & $\lvert r\rvert_{90\%}$ & Median $\chi$ (range) & Top pair: $r$ / $\chi$ & Spearman $r_{\mathrm S}$ \\", r"\midrule"]
    last = None
    for r in t.itertuples():
        label = SYS.get(SHORT.get(r.system, r.system), r.system)
        lines.append(f"{label if last != r.system else ''} & {r.design} & {r.pairs} & {r.r_median:+.2f} & {r.abs_r_p90:.2f} & {r.chi_median:.2f} ({r.chi_min:.2f}--{r.chi_max:.2f}) & "
                     f"{r.top_pair_r_median:+.2f} / {r.top_pair_chi_median:.2f} & {r.spearman_r_vs_lowering_ratio:+.2f} \\\\")
        last = r.system
    cap = r"Correlation between the estimates of two gradients that are built from shared measurements."
    note = (r"Pairs of the leader and each challenger at the states of the exact ADAPT trajectories (H$_2$O: 819 and 864 pairs from 18 and 19 states). "
            r"$r$: correlation coefficient of the estimates of $\lvert g\rvert$ of the two generators; $\lvert r\rvert_{90\%}$: its 90th percentile of the absolute value; "
            r"$\chi=\operatorname{Var}(\hat g_i-s_{ij}\hat g_j)/[\operatorname{Var}\hat g_i+\operatorname{Var}\hat g_j]$: the factor by which the shared data change the variance of the "
            r"comparison of the pair ($\chi=1$: no effect); top pair: the leader and the runner-up. Spearman $r_{\mathrm S}$: rank correlation of $r$ with the ratio of the "
            r"one-parameter energy lowering of the challenger to that of the leader. Estimators II-0 and II-A (\notationref{}).")
    table("findings_corr", cap, "tab:find_corr", "llrrrrrr", lines, note)


def gate() -> None:
    logdir = ROOT / "cluster/logs_feat"
    rows = []
    for i in range(1, 9):
        p = logdir / f"feat_check_2582602_r0_line{i}.log"
        if not p.exists():
            continue
        text = p.read_text()
        m = re.search(r"^(\S+): (\d+) states x (\d+) methods x (\d+) seeds; time original (\d+)s, fast (\d+)s", text, re.M)
        if not m:
            continue
        case, n_states, n_meth, n_seed, t_o, t_f = m.group(1), *map(int, m.groups()[1:])
        steps = sorted({int(s) for s in re.findall(r"step\s+(\d+)\s+seed", text)})
        n_ok, n_bad = len(re.findall(r" OK$", text, re.M)), len(re.findall(r"DIFFERENT", text))
        rows.append((case, steps, n_meth, n_seed, n_ok, n_bad, t_o, t_f))
    lines = [r"System & ADAPT iterations tested & Methods $\times$ streams & Selections compared & Identical & Time, reference / fast (s) \\", r"\midrule"]
    for case, steps, n_meth, n_seed, n_ok, n_bad, t_o, t_f in rows:
        label = SYS.get(case.replace("_HF", "_HF"), case)
        lines.append(f"{label} & {', '.join(map(str, steps))} & {n_meth} $\\times$ {n_seed} & {n_ok + n_bad} & {n_ok} & {t_o} / {t_f} \\\\")
    cap = r"Fast implementation of the selection statistics compared with the reference implementation."
    note = (r"Same states and random streams for both implementations; methods: II-A and, for LiH and H$_4$, also II-0 (\notationref{}). "
            r"Iterations tested: the states of the exact ADAPT trajectory at which the selections were compared. A selection is identical if the generator, the number of rounds, the stopping reason, "
            r"the shortfall and the shots agree (relative tolerance $10^{-9}$). Time: wall-clock seconds of the reference and of the fast implementation on the cluster.")
    table("findings_gate", cap, "tab:find_gate", "lllrrr", lines, note)


# ------------------------------------------------------------------------------------------------ evaluation counts by environment and tolerance

def evals_env() -> None:
    def total(path: Path):
        return sum(json.loads(path.read_text())["evaluations"]) if path.exists() else None

    hers = {"H4_square_eq_side1p0_HF": 285, "H4_square_stretch_side2p0_HF": 242, "LiH_R3p0_HF": 96}
    dirs = {"wsl9": "energy_policies_smoke", "cl9": "energy_policies", "wsl5": "energy_policies_gtol1e-5", "cl5": "energy_policies_gtol1e-5_cl",
            "wsl3": "energy_policies_gtol1e-3", "cl3": "energy_policies_gtol1e-3_cl"}
    keys = ("wsl9", "cl9", "wsl5", "cl5", "wsl3", "cl3")
    cases = ("H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "BeH2_HF", "H2O_eq_HF", "H2O_stretch_HF")
    stretched = "H4_square_stretch_side2p0_HF"
    lines = [r" & Independent study & \multicolumn{2}{c}{$\epsilon_{\mathrm{opt}}=10^{-9}$} & \multicolumn{2}{c}{$\epsilon_{\mathrm{opt}}=10^{-5}$} & \multicolumn{2}{c}{$\epsilon_{\mathrm{opt}}=10^{-3}$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"System & ($10^{-9}$) & Workstation & Cluster & Workstation & Cluster & Workstation & Cluster \\", r"\midrule"]
    val: dict = {}
    for case in cases:
        cells = [str(hers[case]) if case in hers else "--"]
        for key in keys:
            p = RUNS / dirs[key] / case / "exact_reference.json"
            if key == "wsl5" and case == "BeH2_HF":
                p = RUNS / "energy_policies_nrho" / case / "exact_reference.json"
            v = total(p)
            val[(case, key)] = v
            mark = r"$^\dagger$" if (case == stretched and key in ("wsl5", "cl5", "wsl3", "cl3")) else ""
            cells.append(f"{v}{mark}" if v is not None else "--")
        lines.append(f"{SYS[case]} & " + " & ".join(cells) + r" \\")

    def span(tight, loose):
        r = [val[(c, tight)] / val[(c, loose)] for c in cases if c != stretched and val.get((c, tight)) and val.get((c, loose))]
        return f"{min(r):.1f}--{max(r):.1f}" if r else "--"

    pairs = [(c, a, b) for c in cases for a, b in (("wsl5", "cl5"), ("wsl3", "cl3")) if val.get((c, a)) and val.get((c, b))]
    same = sum(1 for c, a, b in pairs if val[(c, a)] == val[(c, b)])
    cap = r"Optimizer evaluations of the exact-ADAPT path in two builds and at three optimizer tolerances."
    note = (r"Exact ADAPT appends the generator of largest $\lvert g\rvert$ in every iteration (ties to the lowest index); the count is the number of energy-and-gradient evaluations of the BFGS "
            r"reoptimizations, summed over all iterations up to chemical accuracy, at the tolerance $\epsilon_{\mathrm{opt}}$ of the header. Workstation and cluster: two builds of the code "
            r"(\notationref{}); independent study: the counts at $10^{-9}$ of an earlier independent study on another machine. At the tight tolerance the count depends on the build, "
            f"because the orbitals of degenerate subspaces differ and with them the Pauli expansion of the Hamiltonian, so only ratios within one build are meaningful. "
            f"At $10^{{-5}}$ and $10^{{-3}}$ the two builds give the same count in {same} of {len(pairs)} comparisons. "
            f"Within one build the count falls by {span('wsl9', 'wsl5')} times (workstation) and {span('cl9', 'cl5')} times (cluster) at $10^{{-5}}$ and by {span('wsl9', 'wsl3')} (workstation) and {span('cl9', 'cl3')} (cluster) at $10^{{-3}}$ "
            r"(stretched H$_4$ left out). "
            r"$^\dagger$With the looser tolerance, stretched H$_4$ no longer stalls on the gradient test, and the run stops on the energy gain after 13 iterations instead of 10.")
    table("findings_evals", cap, "tab:find_evals", "lrrrrrrr", lines, note)


def gtol_paths() -> None:
    d = RUNS / "energy_policies" / "LiH_R3p0_HF"
    lines = [r"Order in which the generators are added & Choices allowed & Runs & Evaluations (cluster, $10^{-9}$) \\", r"\midrule"]
    paths: dict[tuple, list] = {}
    for tag, label in (("policy_rho_random_0.jsonl", "tied $|g|$"), ("policy_rho_random_0.05.jsonl", r"within 5\% of $\max|g|$")):
        p = d / tag
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = (label, tuple(r["selected"]), r["evals_total"])
            paths[key] = paths.get(key, 0) + 1
    def abbrev(path):
        return " ".join(s.replace("D 2,3 -> ", "D").replace("S ", "S").replace(" -> ", "$\\to$") for s in path)
    for (label, path, ev), n in sorted(paths.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        lines.append(f"{abbrev(path)} & {label} & {n} & {ev} \\\\")
    extra = []
    for build, tag in (("workstation build", "energy_policies_gtol{g}"), ("cluster build", "energy_policies_gtol{g}_cl")):
        cells = []
        n = 0
        for gt in ("1e-5", "1e-3"):
            w = RUNS / tag.format(g=gt) / "LiH_R3p0_HF" / "policy_rho_random_0.05.jsonl"
            if w.exists():
                rows = [json.loads(x) for x in w.read_text().splitlines() if x.strip()]
                n = max(n, len(rows))
                cells.append(f"{', '.join(map(str, sorted({r['evals_total'] for r in rows})))} at $10^{{{'-5' if gt == '1e-5' else '-3'}}}$")
        if cells:
            extra.append(f"all orders ({build}) & within 5\\% of $\\max\\lvert g\\rvert$ & {n} & {'; '.join(cells)} \\\\")
    if extra:
        lines.append(r"\midrule")
        lines.extend(extra)
    cap = r"Path dependence of the optimizer cost at the tight tolerance, LiH 3.0~\AA."
    note = (r"The same six generators (up to symmetry partners) added in different orders end at the same energy but need between $54$ and $252$ BFGS evaluations at $\epsilon_{\mathrm{opt}}=10^{-9}$ "
            r"(cluster build; $83$ or $156$ in the workstation build). D: double excitation, S: single excitation, followed by the orbital indices of the pool. "
            r"Choices allowed: the generator is drawn from the generators with exactly tied $\lvert g\rvert$, or from those within $5\%$ of the largest $\lvert g\rvert$. "
            r"At $\epsilon_{\mathrm{opt}}=10^{-5}$ every order, tied or near-tied, needs $39$ or $40$ evaluations and at $10^{-3}$ $30$ or $31$ (both builds, 100 and 200 random orders).")
    # plain columns: a p-column inside \fitcol does not compile under revtex4-2 (the table is also used in the SI); \fitcol shrinks the table
    table("findings_gtol_paths", cap, "tab:find_gtol_paths", "llrl", lines, note)


# ------------------------------------------------------------------------------------------------ confidence levels, totals and premium

def conf() -> None:
    t9 = read_csv(RUNS / "features/report/conf_noninferiority.csv")
    t5 = read_csv(RUNS / "features_gtol1e-5_cl/report/conf_noninferiority.csv")
    t3 = read_csv(RUNS / "features_gtol1e-3_cl/report/conf_noninferiority.csv")
    if t9 is None:
        return

    def cell(tab, r):
        if tab is None:
            return "--"
        m = tab[(tab.system == r.system) & ((tab.confidence_pct - r.confidence_pct).abs() < 1e-6)]
        if m.empty:
            return "--"
        m = m.iloc[0]
        return interval(m.c_opt_ratio_top, m.c_opt_lo, m.c_opt_hi)

    lines = [r" & & & & & \multicolumn{3}{c}{$C_{\mathrm{opt}}/C_{\mathrm{opt}}^{\mathrm{top}}$ [95\% CI] at $\epsilon_{\mathrm{opt}}=$} \\", r"\cmidrule(lr){6-8}",
             r"System & Confidence $1-\delta$ & Runs & Selection shots (median) & Iterations (mean) & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]
    last = None
    for r in t9.itertuples():
        label = SYS.get(SHORT.get(r.system, r.system), r.system)
        if last is not None and last != r.system:
            lines.append(r"\midrule")
        last = r.system
        lines.append(f"{label if r.confidence_pct == t9[t9.system == r.system].confidence_pct.max() else ''} & {r.confidence_pct:.0f}\\% & {r.n} & {sci(r.selection_median)} & "
                     f"{r.steps_mean:.2f} & {interval(r.c_opt_ratio_top, r.c_opt_lo, r.c_opt_hi)} & {cell(t5, r)} & {cell(t3, r)} \\\\")
    cap = r"Optimization cost of complete II-A runs at different confidence levels of the selection."
    note = (r"II-A with the sign-aware rule, a priori first radius and $\rho=0.1$ (\notationref{}). $C_{\mathrm{opt}}$: modeled optimization cost ($\eta=2$), relative to the highest confidence level "
            r"of the system (top); the interval is a 95\% bootstrap interval of the ratio of the means. $\epsilon_{\mathrm{opt}}=10^{-9}$ as recorded; $10^{-5}$ and $10^{-3}$ by replaying the recorded choices "
            r"(cluster build). Runs: 48 per level for H$_4$ and 24 for LiH 3.0~\AA; H$_2$O: 12 at each of 99, 90 and 68\% and the 36 recorded runs at 95\%, replayed with their optimization "
            r"(the only level that was not run afresh).")
    table("findings_conf", cap, "tab:find_conf", "lrrrrlll", lines, note)


def cost_totals() -> None:
    t = read_csv(RUNS / "features/report/cost_totals.csv")
    if t is None:
        return
    order = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
    lines = [r"System & Method & Runs & Iterations (mean) & Reach & Selection share & Total / total of II-A, 95\% \\", r"\midrule"]
    for case in order:
        s = t[t.case == case].copy()
        if s.empty:
            continue
        ref = s[(s.method.str.startswith("II-A")) & ((s.confidence_pct - 95).abs() < 0.5)]
        ref = (ref if len(ref) else s[s.method.str.startswith("II-A")].sort_values("confidence_pct").iloc[[-1]]).iloc[0]
        s["rank"] = s.method.map(lambda m: 0 if m.startswith("II-A") else 1 if m.startswith("II-0") else 2 if m.startswith("M2") else 3 if m.startswith("M1") else 4)
        s["budget"] = s.method.map(lambda m: int(m.split()[1].rstrip(",")) if m.startswith("Fixed") else 0)
        s = s.sort_values(["rank", "confidence_pct", "budget", "method"], ascending=[True, False, True, True])
        first = True
        for r in s.itertuples():
            if r.method.startswith("II-A"):
                name = f"II-A {r.confidence_pct:.0f}\\%"
            elif r.method.startswith("Fixed"):
                b = int(r.method.split()[1].rstrip(","))
                name = f"fixed {sci(b, 1)} (pilot)"
            else:
                name = r.method.replace(", safe", "").replace(" safe", "")
            lines.append(f"{SYS[case] if first else ''} & {name} & {r.n} & {r.steps_mean:.1f} & {pct(r.reached_pct)} & {r.selection_share_pct:.2g}\\% & "
                         f"{r.total_median / ref.total_median:.2f} \\\\")
            first = False
        lines.append(r"\midrule")
    lines.pop()
    cap = r"Total cost of complete measured runs at the tight optimizer tolerance, relative to II-A at 95\%."
    note = (r"Total: selection shots plus modeled optimization cost $C_{\mathrm{opt}}$ ($\eta=2$, $\epsilon_{\mathrm{opt}}=10^{-9}$), median over the runs; II-A at a confidence level is II-A at $1-\delta$ "
            r"(\notationref{}). Selection share: selection shots over the total. Reach: share of the runs that reach chemical accuracy; a run that does not is stopped ten iterations above "
            r"the count of exact ADAPT, so its total is a lower bound. Fixed $N$ (pilot): uncertified selection with a budget of $N$ context-shots per iteration and the pilot allocation. "
            r"Runs: II-A, 48 (H$_4$) and 24 (LiH 3.0~\AA) per level from the larger sets, 12 (H$_2$O eq) at 99, 90 and 68\% and 36 at 95\%; the other methods, the 36 recorded runs (H$_2$O) and 12 runs per fixed budget. "
            r"At this tolerance a median total fluctuates by $\pm20$--$30\%$ between sets of 24 runs (tie-order noise of the optimizer) and by $\pm7\%$ for 36 runs of H$_2$O.")
    table("findings_cost", cap, "tab:find_cost", "llrrrrr", lines, note)


def cost_tolerances() -> None:
    srcs = {"9": RUNS / "features/report/cost_totals.csv", "5": RUNS / "features_gtol1e-5_cl/report/cost_totals.csv", "3": RUNS / "features_gtol1e-3_cl/report/cost_totals.csv"}
    tabs = {k: read_csv(p) for k, p in srcs.items()}
    if tabs["9"] is None or tabs["5"] is None or tabs["3"] is None:
        return
    order = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
    lines = [r" & & & Reach at $\epsilon_{\mathrm{opt}}=$ & \multicolumn{3}{c}{Selection share at $\epsilon_{\mathrm{opt}}=$} & \multicolumn{3}{c}{Total / total of II-A, 95\%, at $\epsilon_{\mathrm{opt}}=$} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"System & Method & Runs & $10^{-9}$ / $10^{-3}$ & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]

    def pick(tab, case, method, conf):
        s = tab[(tab.case == case) & (tab.method == method) & ((tab.confidence_pct - conf).abs() < 0.5)]
        return s.iloc[0] if len(s) else None

    for case in order:
        s9 = tabs["9"][(tabs["9"].case == case) & (((tabs["9"].confidence_pct - 95).abs() < 0.5))]
        if s9.empty:
            continue
        s9 = s9.assign(rank=s9.method.map(lambda m: 0 if m.startswith("II-A") else 1 if m.startswith("II-0") else 2 if m.startswith("M2") else 3 if m.startswith("M1") else 4),
                       budget=s9.method.map(lambda m: int(m.split()[1].rstrip(",")) if m.startswith("Fixed") else 0)).sort_values(["rank", "budget"])
        refs = {k: pick(tabs[k], case, "II-A data, safe", 95) for k in tabs}
        first = True
        for r in s9.itertuples():
            cells = {k: pick(tabs[k], case, r.method, 95) for k in tabs}
            if any(c is None for c in cells.values()):
                continue
            if r.method.startswith("II-A"):
                name = "II-A 95\\%"
            elif r.method.startswith("Fixed"):
                name = f"fixed {sci(int(r.method.split()[1].rstrip(',')), 1)} (pilot)"
            else:
                name = r.method.replace(", safe", "").replace(" safe", "")
            lines.append(f"{SYS[case] if first else ''} & {name} & {int(cells['9'].n)} & {pct(cells['9'].reached_pct)} / {pct(cells['3'].reached_pct)} & "
                         + " & ".join(f"{cells[k].selection_share_pct:.2g}\\%" for k in ("9", "5", "3")) + " & "
                         + " & ".join(f"{cells[k].total_median / refs[k].total_median:.2f}" for k in ("9", "5", "3")) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines.pop()
    cap = r"Selection share and total cost of complete measured runs at three optimizer tolerances."
    note = (r"Replay of the recorded runs: the selections and the selection shots are kept and the optimization is rerun at $\epsilon_{\mathrm{opt}}$ (cluster build). Total: selection shots plus modeled "
            r"optimization cost, median over the runs, relative to II-A at 95\% at the same tolerance; selection share: selection shots over the total. Reach: share of the runs that end at chemical accuracy "
            r"at the tight and at the loosest tolerance; a run that does not is stopped ten iterations above the count of exact ADAPT, so its total is a lower bound. Methods and fixed budgets as in "
            r"Table~\ref{tab:find_cost}. Runs: the other methods are the 36 recorded runs of H$_2$O, the fixed budgets 12 runs each.")
    table("findings_cost_tol", cap, "tab:find_cost_tol", "llrrrrrrrr", lines, note)


def tol_energy() -> None:
    """Final energy of the replays at the looser tolerances against the tight one, run by run."""
    pairs = []
    for root, tail in (("features", "features_gtol{g}_cl"), ("features_ext", "features_ext_gtol{g}_cl")):
        for g in ("1e-5", "1e-3"):
            ldir = RUNS / tail.format(g=g)
            for cdir in sorted((RUNS / root).glob("*_HF")):
                for f in sorted(cdir.glob("ci_*.jsonl")):
                    lf = ldir / cdir.name / f.name
                    if not lf.exists():
                        continue
                    a = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                    b = [json.loads(x) for x in lf.read_text().splitlines() if x.strip()]
                    kind = "fixed" if "Fixed" in f.name else "certified"
                    for ra, rb in zip(a, b):
                        pairs.append((g, kind, cdir.name, 1e3 * (rb["final_error"] - ra["final_error"])))
    if not pairs:
        return
    d = pd.DataFrame(pairs, columns=["gtol", "kind", "case", "d"])
    def fmt(v: float) -> str:
        return sci(v, 1) if v < 0.01 else f"{v:.2g}"

    lines = [r"Tolerance $\epsilon_{\mathrm{opt}}$ and runs & Runs & Largest change (mHa) & Runs changing by more than $0.1$~mHa & of these, H$_4$ 1.0~\AA \\", r"\midrule"]
    for g, name in (("1e-5", "$10^{-5}$"), ("1e-3", "$10^{-3}$")):
        for kind, label in (("certified", "certified and baselines (II-A, II-0, M1 seq, M2)"), ("fixed", "uncertified pilot budgets")):
            s = d[(d.gtol == g) & (d.kind == kind)]
            big = s[s.d.abs() > 0.1]
            lines.append(f"{name}, {label} & {len(s)} & {fmt(s.d.abs().max())} & {len(big)} & {int((big.case == 'H4_square_eq_side1p0_HF').sum())} \\\\")
    cap = r"Change of the final energy when the optimizer stops at a looser tolerance."
    note = (r"Replay of the recorded choices, run by run, against the same run at $\epsilon_{\mathrm{opt}}=10^{-9}$ (cluster build); the change is the energy at the looser tolerance minus the energy at $10^{-9}$. "
            r"Certified and baselines: II-A, II-0, M1 seq and M2; uncertified: fixed budgets with the pilot allocation (\notationref{}). The certified runs end within a few $\mu$Ha of "
            r"their tight-tolerance energy; the uncertified runs with poor choices can stop on a plateau where $\lvert g\rvert$ is below the tolerance although the energy is still far above the minimum "
            r"(H$_4$ 1.0~\AA, $10^{-3}$: eight of the 23 pilot runs that reach chemical accuracy at $10^{-9}$ end $4$--$76$~mHa higher).")
    table("findings_tol_energy", cap, "tab:find_tol_energy", "lrrrr", lines, note)


def rho_evals() -> None:
    tabs = {"9": read_csv(RUNS / "energy_policies/report/policies_all.csv"), "5": read_csv(RUNS / "energy_policies_gtol1e-5_cl/report/policies_all.csv"),
            "3": read_csv(RUNS / "energy_policies_gtol1e-3_cl/report/policies_all.csv")}
    if any(v is None for v in tabs.values()):
        return
    pols = [("rho_random:0.05", r"$\rho=0.05$"), ("rho_random:0.1", r"$\rho=0.1$"), ("rho_random:0.3", r"$\rho=0.3$"), ("rho_random:0.5", r"$\rho=0.5$"), ("tau_random:0.5", r"$\tau=0.5$")]
    lines = [r"System & " + " & ".join(h for _, h in pols) + r" \\", r"\midrule"]
    for s in ["H4 1.0", "H4 2.0", "LiH 3.0", "BeH2", "H2O eq", "H2O str"]:
        cells = []
        for p, _ in pols:
            vals = []
            for k in ("9", "5", "3"):
                r = tabs[k][(tabs[k].system == s) & (tabs[k].policy == p)]
                vals.append(num(float(r.evals_ratio_tie_avg.iloc[0])) if len(r) else "--")
            cells.append(" / ".join(vals))
        lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(cells) + r" \\")
    cap = r"Optimizer evaluations of ADAPT with injected selection errors, relative to exact ties, at three optimizer tolerances."
    note = (r"Each entry gives three ratios, for $\epsilon_{\mathrm{opt}}=10^{-9}$ / $10^{-5}$ / $10^{-3}$: the mean number of optimizer evaluations of ADAPT with the selection rule of the header, "
            r"divided by that of ADAPT with random choices among exactly tied generators in the same build. $\rho$: uniform draw from the $\rho$-set; $\tau$: uniform draw from the $\tau$-set "
            r"(\notationref{}). Cluster build, 200 trajectories per entry. At the tight tolerance a near-tie costs up to $+73\%$ (LiH 3.0~\AA) because the polishing depends on the order of the generators; "
            r"at $10^{-5}$ and $10^{-3}$ the count follows the number of iterations: tolerances up to $\rho^\ast$ cost at most a few percent, larger tolerances cost what their extra iterations cost.")
    table("findings_policies_rho_evals", cap, "tab:find_rho_evals", "lrrrrr", lines, note)


def nrho_tables() -> None:
    t = read_csv(RUNS / "energy_policies_nrho_gtol1e-5_cl/report/policies_all.csv")
    if t is None:
        return

    def val(s: str, p: str, col: str) -> str:
        r = t[(t.system == s) & (t.policy == p)]
        return num(float(r[col].iloc[0])) if len(r) else "--"

    def pair(s: str, r: str, col: str) -> str:
        return f"{val(s, 'rho_random:' + r, col)} / {val(s, 'nrho_random:' + r, col)}"

    systems = ["H4 1.0", "H4 2.0", "LiH 3.0", "BeH2", "H2O eq", "H2O str"]
    radii = ["0.1", "0.2", "0.3", "0.5", "0.7"]
    blocks = (("steps_ratio", "Iterations relative to exact ADAPT"), ("evals_ratio_tie_avg", "Optimizer evaluations relative to exact ADAPT"))
    lines = [r"System & " + " & ".join(rf"$R={r}$" for r in radii) + r" & $\tau$-set, $\tau=0.3$ \\", r"\midrule"]
    for col, title in blocks:
        lines.append(rf"\multicolumn{{{len(radii) + 2}}}{{l}}{{\emph{{{title}}}}} \\")
        for s in systems:
            lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(pair(s, r, col) for r in radii) + f" & {val(s, 'tau_random:0.3', col)} \\\\")
    cap = r"Iterations and optimizer evaluations of ADAPT with random choices from the plain and from the curvature-scaled gradient set."
    note = (r"Each entry is \emph{plain set} / \emph{curvature-scaled set} for the tolerance $R$ of the header: a uniform draw from the generators with $\lvert g_i\rvert\ge(1-R)\max_j\lvert g_j\rvert$ "
            r"(plain), respectively with $\lvert g_i\rvert/\sqrt{h_i}\ge(1-R)\max_j\lvert g_j\rvert/\sqrt{h_j}$ (curvature-scaled), where $h_i$ is the diagonal curvature at the Hartree--Fock determinant "
            r"(no shots). Last column: uniform draw from the $\tau$-set with $\tau=0.3$, which needs the one-parameter lowering at every iteration. Ratios are relative to exact ADAPT; the evaluation counts "
            r"are relative to the exact path of the same build (cluster build, $\epsilon_{\mathrm{opt}}=10^{-5}$). 200 trajectories per entry.")
    table("findings_policies_nrho", cap, "tab:find_nrho", "lrrrrrr", lines, note)

    w = ["H2O dz8o eq", "H2O dz8o str"]
    cols = [("rho_random:0.1", r"plain $0.1$"), ("nrho_random:0.1", r"scaled $0.1$"), ("rho_random:0.3", r"plain $0.3$"), ("nrho_random:0.3", r"scaled $0.3$"),
            ("rho_random:0.5", r"plain $0.5$"), ("nrho_random:0.5", r"scaled $0.5$"), ("tau_random:0.3", r"$\tau=0.3$"), ("onep_greedy", "greedy")]
    lines = [r" & & \multicolumn{6}{c}{Gradient set, tolerance $R$} & \multicolumn{2}{c}{Energy-based} \\", r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}",
             r"System & Quantity & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for s in w:
        for k, (col, name) in enumerate((("err_ratio_exact_steps", r"energy error, iteration 40"), ("reached_pct", r"reaching the target (\%)"), ("evals_ratio_tie_avg", "optimizer evaluations"))):
            fmt = (lambda p, c=col: f"{float(t[(t.system == s) & (t.policy == p)][c].iloc[0]):.0f}") if col == "reached_pct" else (lambda p, c=col: val(s, p, c))
            lines.append(f"{SYS[SHORT[s]] if k == 0 else ''} & {name} & " + " & ".join(fmt(p) for p, _ in cols) + r" \\")
    cap = r"Curvature-scaled gradient set on the 16-qubit cc-pVDZ windows of H$_2$O."
    note = (r"Cluster build, $\epsilon_{\mathrm{opt}}=10^{-5}$, 40 iterations, 48 trajectories per random rule. Plain: uniform draw from the generators with $\lvert g_i\rvert\ge(1-R)\max_j\lvert g_j\rvert$; "
            r"scaled: the same with $\lvert g_i\rvert/\sqrt{h_i}$, $h_i$ the curvature at the Hartree--Fock determinant (no shots); $\tau=0.3$: uniform draw from the $\tau$-set; greedy: the generator with the "
            r"largest one-parameter energy lowering. Energy error at iteration 40: mean over the trajectories relative to that of exact ADAPT. Reaching the target: share of the trajectories whose energy error "
            r"after 40 iterations is at most the error of exact ADAPT plus $0.1$~mHa. Optimizer evaluations: relative to exact ADAPT.")
    table("findings_policies_nrho_window", cap, "tab:find_nrho_window", "llrrrrrrrr", lines, note)


def conf_direct() -> None:
    """Confidence level at BFGS 1e-5: the replays pooled with the direct runs of 2026-10-19 (seed 11), and the direct runs alone."""
    pooled = read_csv(RUNS / "features_conf_gtol1e-5_cl/report_pooled/conf_noninferiority.csv")
    direct = read_csv(RUNS / "features_conf_gtol1e-5_cl/report/conf_noninferiority.csv")
    if pooled is None or direct is None:
        return
    lines = [r"System & Confidence $1-\delta$ & Runs (pooled / new) & Selection shots (median) & Iterations & Evaluations & $C_{\mathrm{opt}}$ ratio, pooled & $C_{\mathrm{opt}}$ ratio, new runs \\", r"\midrule"]
    falls = []
    for system in ("H4 1.0", "H4 2.0", "LiH 3.0", "H2O eq", "H2O str"):
        p = pooled[pooled.system == system]
        if p.empty:
            continue
        top, low = p[p.confidence_pct == p.confidence_pct.max()].iloc[0], p[(p.confidence_pct - 68).abs() < 1e-6].iloc[0]
        falls.append(top.selection_median / low.selection_median)
        levels = [c for c in (99, 95, 68) if ((p.confidence_pct - c).abs() < 1e-6).any()]
        for k, c in enumerate(levels):
            r = p[(p.confidence_pct - c).abs() < 1e-6].iloc[0]
            d = direct[(direct.system == system) & ((direct.confidence_pct - c).abs() < 1e-6)]
            n_new = int(d.iloc[0].n) if len(d) else 0
            new = interval(d.iloc[0].c_opt_ratio_top, d.iloc[0].c_opt_lo, d.iloc[0].c_opt_hi) if len(d) else "--"
            lines.append(f"{SYS[SHORT[system]] if k == 0 else ''} & {c}\\% & {int(r.n)} / {n_new} & {sci(r.selection_median)} & {r.steps_mean:.2f} & {r.evals_mean:.0f} & "
                         f"{interval(r.c_opt_ratio_top, r.c_opt_lo, r.c_opt_hi)} & {new} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    cap = r"Confidence level of the selection and the optimization cost of complete runs at $\epsilon_{\mathrm{opt}}=10^{-5}$, direct runs pooled with replays."
    note = (r"II-A, sign-aware rule, a priori first radius, $\rho=0.1$. $\epsilon_{\mathrm{opt}}$: BFGS gradient tolerance, here $10^{-5}$ in the runs themselves (new runs, seed 11, cluster build). "
            r"Pooled: the earlier sweeps replayed at $10^{-5}$ plus the new runs (95\% level of H$_2$O: the 36 replayed runs plus 16). "
            r"$C_{\mathrm{opt}}$ ratio: modeled optimization cost relative to the highest level, ratio of means with a 95\% bootstrap interval. "
            rf"The selection shots of the highest level over those of 68\% are {min(falls):.1f}--{max(falls):.1f}.")
    table("findings_conf_direct", cap, "tab:find_conf_direct", "llrrrrll", lines, note)


def noise_tables() -> None:
    """The optimizer on shot-noise energies and gradients (set_noisy_optimizer): exact-gradient choices on six systems and measured II-A."""
    summary = read_csv(RUNS / "energy_policies_noisy_summary.csv")
    if summary is None:
        return
    cfg = [(1e-3, "1e-5"), (1e-3, "3e-3"), (1e-4, "1e-5"), (1e-4, "3e-4")]
    heads = [r"$10^{-3}$ / $10^{-5}$", r"$10^{-3}$ / $3\times10^{-3}$", r"$10^{-4}$ / $10^{-5}$", r"$10^{-4}$ / $3\times10^{-4}$"]
    lines = [r"System & " + " & ".join(heads) + r" \\", r"\midrule"]
    s = summary[summary.policy == "exact"]
    blocks = (("evals_ratio", r"Evaluations relative to noise-free BFGS at $10^{-5}$", 2), ("steps", r"Iterations (noise free in brackets)", 1), ("reached_pct", r"Runs reaching the target (\%)", 0))
    for col, title, d in blocks:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\")
        for system, case in (("H4 1.0", "H4_square_eq_side1p0_HF"), ("H4 2.0", "H4_square_stretch_side2p0_HF"), ("LiH 3.0", "LiH_R3p0_HF"), ("BeH2", "BeH2_HF"),
                             ("H2O eq", "H2O_eq_HF"), ("H2O str", "H2O_stretch_HF")):
            cells = []
            for e, g in cfg:
                r = s[(s.system == system) & ((s.eps - e).abs() < 1e-12) & ((s.gtol.astype(float) - float(g)).abs() < 1e-12)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                cells.append(f"{r[col]:.{d}f}" + (f" ({r.steps_free:.0f})" if col == "steps" else ""))
            lines.append(f"{SYS[case]} & " + " & ".join(cells) + r" \\")
    both = summary.pivot_table(index=["system", "eps", "gtol"], columns="policy", values="evals_ratio")
    diff = float((both["exact"] - both["rho_random_0.1"]).abs().max())
    cap = r"Optimizer on shot-noise energies and gradients: evaluations, iterations and success of ADAPT with exact choices."
    note = (r"Cluster build; 100 noise streams per entry; exact-gradient choices (the optimizer is the only noisy part). Each evaluation returns the energy with a normal error of standard deviation $\epsilon$ (Ha) "
            r"and each gradient component with $\epsilon/\sqrt2$ (a parameter-shift component is half the difference of two shifted energies); BFGS stops at the gradient tolerance $g_{\mathrm{tol}}$, "
            r"after a failed line search, or after 300 iterations. The energy reported is the exact energy of the parameters returned. Header: $\epsilon$ / $g_{\mathrm{tol}}$. "
            r"The shots of one evaluation grow with $(1\,\mathrm{mHa}/\epsilon)^2$: a factor 100 in the last two columns. "
            rf"Random choices inside the $\rho=0.1$ set (not shown) change the evaluation ratio by at most {diff:.2f}. "
            r"On stretched H$_4$ the noise-free path stops after 13 iterations through the zero gradients of symmetry-forbidden generators; with noisy parameters those gradients are not zero and the energy test needs more iterations.")
    table("findings_noise_policy", cap, "tab:find_noise_policy", "lrrrr", lines, note)

    def rows(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

    lines = [r"System & Optimizer & Confidence $1-\delta$ & Runs & Iterations & Evaluations & Selection shots & $C_{\mathrm{opt}}$ & Selection share \\", r"\midrule"]
    sources = [("noise free, $10^{-5}$", None), (r"noisy, $g_{\mathrm{tol}}=10^{-5}$", "eps1e-3_gtol1e-5"), (r"noisy, $g_{\mathrm{tol}}=3\times10^{-3}$", "eps1e-3_gtol3e-3")]
    for case, free_root in (("H4_square_eq_side1p0_HF", "features_ext_gtol1e-5_cl"), ("LiH_R3p0_HF", "features_conf_gtol1e-5_cl")):
        for si, (label, tag) in enumerate(sources):
            for li, delta in enumerate((0.01, 0.32)):
                root = RUNS / (free_root if tag is None else f"features_noisy_{tag}_cl")
                r = rows(root / case / f"ci_II-A_data_safe_d{delta:g}.jsonl")
                if not r:
                    continue
                sel, c = np.array([x["selection_shots"] for x in r]), np.array([x["c_opt_eta2"] for x in r])
                first = si == 0 and li == 0
                lines.append(f"{SYS[case] if first else ''} & {label if li == 0 else ''} & {100 * (1 - delta):.0f}\\% & {len(r)} & {np.mean([x['iterations'] for x in r]):.2f} & "
                             f"{np.mean([x['optimizer_evaluations'] for x in r]):.0f} & {sci(np.median(sel))} & {sci(np.median(c))} & {np.median(sel) / np.median(sel + c):.3f} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    cap2 = r"Measured II-A selection with the noisy optimizer: iterations and modeled cost at the highest and at the lowest confidence level."
    note2 = (r"II-A, sign-aware rule, a priori first radius, $\rho=0.1$; noise $\epsilon=10^{-3}$~Ha per energy as in the preceding table; medians of selection shots and of $C_{\mathrm{opt}}$ "
             r"(modeled with one evaluation of 1~mHa per energy and $\eta=2$, i.e.\ the shots of the noise level itself); selection share: median selection over the median of selection plus $C_{\mathrm{opt}}$. "
             r"Noise free: BFGS at $10^{-5}$ without noise. Stretched H$_4$ is left out: with noisy parameters the zero gradients of its symmetry-forbidden generators do not vanish, "
             r"and its runs continue to the 50 iterations allowed.")
    table("findings_noise_measured", cap2, "tab:find_noise_measured", "llrrrrrrr", lines, note2)


def window_share() -> None:
    a, b = read_csv(RUNS / "window_share_gtol1e-5.csv"), read_csv(RUNS / "window_share_gtol1e-9.csv")
    if a is None or b is None:
        return
    lines = [r"Geometry & Step $k$ & $\max\lvert g\rvert$ & Selection shots (median) & Exact best & $n_{\mathrm{ev}}$, $10^{-5}$ & $m_E$ & $C_{\mathrm{opt}}$ of the next step & Share, $10^{-5}$ & Share, $10^{-9}$ \\", r"\midrule"]
    for g, name in (("eq", "equilibrium"), ("str", "stretched")):
        for k in sorted(a[a.geometry == g].k):
            r, r9 = a[(a.geometry == g) & (a.k == k)].iloc[0], b[(b.geometry == g) & (b.k == k)].iloc[0]
            lines.append(f"{name if k == 5 else ''} & {k} & {r.max_gradient:.3f} & {sci(r.selection_median)} & {pct(100 * r.correct_rate)} & {int(r.n_ev)} & {sci(r.m_E)} & {sci(r.c_opt)} & "
                         f"{pct(100 * r.share_median, 1)} & {pct(100 * r9.share_median, 1)} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    cap = r"Share of the certified selection in the total shots on the 16-qubit cc-pVDZ window of H$_2$O, along the exact ADAPT path."
    note = (r"II-A, a priori start, sign-aware rule, $\rho=0.1$, 16 trials at the state after $k$ steps of exact ADAPT; every trial chose a generator inside the $\rho$-set. Exact best: share of trials that chose the exact largest gradient. "
            r"$C_{\mathrm{opt}}$: $n_{\mathrm{ev}}(1+2n)m_E$ for the re-optimization after adding the next generator ($n=k+1$ parameters; $n_{\mathrm{ev}}$ from the exact path with BFGS at the tolerance of the header; "
            r"$m_E$ the shots of one energy to 1~mHa at the state after the step; for $k=40$ the 40th step stands in for the 41st). Share: median selection over median selection plus $C_{\mathrm{opt}}$. "
            r"A complete measured trajectory was not run (3--4 h of classical time per selection).")
    table("findings_window_share", cap, "tab:find_window_share", "lrrrrrrrrr", lines, note)


def premium() -> None:
    bases ={"9": RUNS / "features_q10", "5": RUNS / "features_q10_gtol1e-5", "3": RUNS / "features_q10_gtol1e-3"}
    base = bases["9"]
    if not base.exists():
        return
    adequate = {"H4_square_eq_side1p0_HF": 13335, "H4_square_stretch_side2p0_HF": 133352, "LiH_R1p6_HF": 3162278, "LiH_R3p0_HF": 7498942,
                "H2O_eq_HF": 133352143, "H2O_stretch_HF": 421696503}
    lines = [r" & Fixed budget: & Iterations, & & \multicolumn{3}{c}{Premium in total shots at $\epsilon_{\mathrm{opt}}=$} & \\", r"\cmidrule(lr){5-7}",
             r"System & per iteration / selection shots & fixed (certified) & Premium in selection shots & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & Iterations only \\", r"\midrule"]

    def load(path: Path):
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

    def reach(root: Path, case: str, rows: list):
        refp = root / case / "exact_reference.json"
        if not refp.exists() or not rows:
            return None
        ref = json.loads(refp.read_text())
        target = 1.6e-3 if ref["final_error"] < 1.6e-3 else ref["final_error"] + 1e-4
        return 100 * float(np.mean([r["final_error"] <= target for r in rows]))

    missing, low = [], []
    for case, b in adequate.items():
        fx9 = load(base / case / f"ci_Fixed_{b}_designed_d0.05.jsonl")
        ce9 = load(base / case / "ci_II-A_data_safe_d0.05.jsonl")
        if not fx9 or not ce9:
            missing.append(SYS[case])
            continue
        totals = {}
        for k, root in bases.items():
            fx = load(root / case / f"ci_Fixed_{b}_designed_d0.05.jsonl")
            ce = load(root / case / "ci_II-A_data_safe_d0.05.jsonl")
            if fx and ce:
                rk = reach(root, case, fx)
                if k != "9" and rk is not None and rk < 99.5:
                    low.append((SYS[case], k, rk))
                tf = np.array([r["selection_shots"] + r["c_opt_eta2"] for r in fx])
                tc = np.array([r["selection_shots"] + r["c_opt_eta2"] for r in ce])
                rng = np.random.default_rng(1)
                ia, ib = rng.integers(0, len(tc), (20000, len(tc))), rng.integers(0, len(tf), (20000, len(tf)))
                lo, hi = np.percentile(np.median(tc[ia], axis=1) / np.median(tf[ib], axis=1), [2.5, 97.5])
                totals[k] = (float(np.median(tc) / np.median(tf)), float(lo), float(hi))
        sel_f = np.array([r["selection_shots"] for r in fx9])
        sel_c = np.array([r["selection_shots"] for r in ce9])
        steps_f = np.mean([r["iterations"] for r in fx9])
        steps_c = np.mean([r["iterations"] for r in ce9])
        lines.append(f"{SYS[case]} & {sci(b)} / {sci(np.median(sel_f))} & {steps_f:.2f} ({steps_c:.2f}) & {np.median(sel_c) / np.median(sel_f):.2f} & "
                     + " & ".join(f"{totals[k][0]:.2f} ({totals[k][1]:.2f}--{totals[k][2]:.2f})" if k in totals else "--" for k in ("9", "5", "3")) + f" & {steps_c / steps_f:.2f} \\\\")
    cap = r"Certification premium in selection shots and in total shots."
    note = (r"Premium: certified II-A at 95\% over the cheapest adequate fixed budget (designed allocation; at least 95\% of the trajectories reach chemical accuracy within one iteration of the count "
            r"of exact ADAPT), from the recorded runs with their optimization recomputed (modeled $C_{\mathrm{opt}}$, $\eta=2$; $\epsilon_{\mathrm{opt}}=10^{-9}$ as recorded, $10^{-5}$ and $10^{-3}$ by replaying "
            r"the recorded choices). Premium in selection shots: ratio of the median selection shots. Premium in total shots: ratio of the medians of selection plus optimization, with the 95\% bootstrap "
            r"interval in parentheses; the fixed budget pays for its longer trajectories. Iterations only: ratio of the mean numbers of iterations, the part of the premium that is due to the longer trajectories alone; "
            r"at $10^{-9}$ the rest is mostly the path dependence of the evaluation count of the optimizer (Sec.~\ref{sec:gtol}). Methods and symbols: \notationref{}.")
    if low:
        note += (" The designed fixed budgets are adequate at $10^{-9}$ by construction; at the looser tolerances some of their runs stall on a plateau, and the share that reaches chemical accuracy falls to "
                 + ", ".join(f"{v:.0f}\\% ({s}, $10^{{{'-5' if k == '5' else '-3'}}}$)" for s, k, v in low) + " (II-A: all runs).")
    if missing:
        note += " Rows still to come (replay of the recorded runs running): " + ", ".join(missing) + "."
    table("findings_premium", cap, "tab:find_premium", "llrrrrrr", lines, note)


# ------------------------------------------------------------------------------------------------ policies

def policies() -> None:
    t = read_csv(RUNS / "energy_policies/report/policies_all.csv")
    if t is None:
        return
    order = ["H4 1.0", "H4 2.0", "LiH 3.0", "BeH2", "H2O eq", "H2O str"]
    rhos = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    def val(system, policy, col):
        r = t[(t.system == system) & (t.policy == policy)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    lines = [r" & Exact ADAPT & \multicolumn{" + str(len(rhos)) + r"}{c}{Iterations relative to exact ADAPT, $\rho$-set} & & Reach at \\",
             r"\cmidrule(lr){3-" + str(2 + len(rhos)) + r"}",
             r"System & iterations & " + " & ".join(f"$\\rho={x:g}$" for x in rhos) + r" & $\rho^\ast$ & $\rho=0.9$ \\", r"\midrule"]
    for s in order:
        cells = []
        star = None
        good = True
        for x in rhos:
            v = val(s, f"rho_random:{x:g}", "steps_ratio")
            cells.append(f"{v:.2f}" if ok(v) else "--")
            if ok(v) and v <= 1.03 and good:
                star = x
            elif ok(v):
                good = False
        lines.append(f"{SYS[SHORT[s]]} & {int(val(s, 'exact', 'exact_steps'))} & " + " & ".join(cells) + f" & {star if star is not None else '--'} & "
                     f"{pct(val(s, 'rho_random:0.9', 'reached_pct'))} \\\\")
    cap = r"Iterations of ADAPT with a random choice from the $\rho$-set in every iteration, relative to exact ADAPT."
    note = (r"Every selection is a uniform draw from the $\rho$-set $\{i:\lvert g_i\rvert\ge(1-\rho)\max_j\lvert g_j\rvert\}$; exact gradients and exact optimization at $\epsilon_{\mathrm{opt}}=10^{-9}$, "
            r"200 trajectories per entry, common random numbers over $\rho$. Entries: mean number of iterations divided by that of exact ADAPT. $\rho^\ast$: the largest $\rho$ of the grid up to which the entries stay "
            r"within $3\%$ of one. Reach: share of the trajectories that end at chemical accuracy (for stretched H$_4$: within $0.1$~mHa of the final energy of exact ADAPT) at $\rho=0.9$. "
            r"Random ADAPT ($\rho=1$) still reaches it on H$_4$ (a pool of 26 generators), but in $10\%$ of the LiH 3.0~\AA\ runs and in none of the BeH$_2$ and H$_2$O runs. The iterations are the same at "
            r"$10^{-5}$ and $10^{-3}$ for $\rho\le0.7$, but random ADAPT then reaches chemical accuracy in only about half of the H$_4$ 1.0~\AA\ runs ($51\%$, $48\%$) and in $6\%$ and $2\%$ of the LiH runs, "
            r"because a plateau stops the optimizer (Sec.~\ref{sec:gtol}).")
    table("findings_policies_rho", cap, "tab:find_rho", "lrrrrrrrrrrr", lines, note)

    taus = [0.05, 0.1, 0.2, 0.3, 0.5]
    lines = [r" & \multicolumn{" + str(len(taus)) + r"}{c}{Iterations relative to exact ADAPT, $\tau$-set} & $\rho$-set \\", r"\cmidrule(lr){2-" + str(1 + len(taus)) + r"}",
             r"System & " + " & ".join(f"$\\tau={x:g}$" for x in taus) + r" & " + r"$\rho=0.5$ \\", r"\midrule"]
    for s in order:
        cells = [num(val(s, f"tau_random:{x:g}", "steps_ratio")) for x in taus]
        lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(cells) + f" & {num(val(s, 'rho_random:0.5', 'steps_ratio'))} \\\\")
    cap = r"Iterations of ADAPT with a random choice from the $\tau$-set in every iteration, relative to exact ADAPT."
    note = (r"Every selection is a uniform draw from the $\tau$-set $\{i:\Delta E_i\ge(1-\tau)\max_j\Delta E_j\}$ of the generators whose one-parameter energy lowering $\Delta E_i=E(\psi)-\min_\theta E(e^{\theta G_i}\psi)$ "
            r"is within a fraction $\tau$ of the best one. Conditions and entries as in Table~\ref{tab:find_rho}; the last column repeats the entry of the $\rho$-set at $\rho=0.5$ for comparison. "
            r"A tolerance on the energy lowering leaves the number of iterations unchanged up to $\tau=0.5$ on every system; on H$_4$ 1.0~\AA\ it saves two iterations.")
    table("findings_policies_tau", cap, "tab:find_tau", "lrrrrrr", lines, note)

    greedy = ["onep_greedy", "newton_hf", "newton_exact", "reopt_top:3", "reopt_top:10", "reopt_all"]
    head = ["One-parameter lowering", "Newton, HF curvature", "Newton, current curvature", "Reoptimized, top 3", "Reoptimized, top 10", "Reoptimized, all"]
    lines = [r"System & " + " & ".join(head) + r" \\", r"\midrule"]
    for s in order:
        lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(num(val(s, g, "steps_ratio")) for g in greedy) + r" \\")
    cap = r"Iterations of ADAPT with energy-greedy choices, relative to exact ADAPT."
    note = (r"Every selection is the generator with the largest one-parameter energy lowering; with the largest Newton estimate $g_i^2/(2h_i)$, the curvature $h_i$ taken at the Hartree--Fock determinant "
            r"(no shots) or at the current state; or with the lowest energy after reoptimizing all parameters, among the 3 or 10 generators of largest $\lvert g_i\rvert$ or among all generators with a nonzero "
            r"gradient. Exact gradients and optimization, $\epsilon_{\mathrm{opt}}=10^{-9}$. Only H$_4$ 1.0~\AA\ gains iterations (10 to 8; H$_2$O eq 17 to 16 with all candidates reoptimized). "
            r"The numbers of iterations are the same at $10^{-5}$ and $10^{-3}$, except that the H$_2$O eq gain disappears at $10^{-3}$.")
    table("findings_policies_greedy", cap, "tab:find_greedy", "lrrrrrr", lines, note)

    cols = ["exact", "onep_greedy", "newton_hf", "reopt_top:3", "tau_random:0.1", "tau_random:0.3", "rho_random:0.1", "rho_random:0.3", "rho_worst_e:0.1"]
    head = ["Exact ADAPT", "One-parameter lowering", "Newton, HF curvature", "Reoptimized, top 3", r"$\tau=0.1$", r"$\tau=0.3$", r"$\rho=0.1$", r"$\rho=0.3$", r"$\rho=0.1$, worst"]
    lines = [r"System & Iteration & " + " & ".join(head) + r" \\", r"\midrule"]
    for s in ("H2O dz8o eq", "H2O dz8o str"):
        for col, name in (("err20_mHa", "20"), ("err_exact_steps_mHa", "40")):
            lines.append(f"{SYS[SHORT[s]] if name == '20' else ''} & {name} & " + " & ".join(num(val(s, c, col), 2) for c in cols) + r" \\")
    cap = r"Energy error of ADAPT on the 16-qubit cc-pVDZ windows of H$_2$O after 20 and 40 iterations (mHa), for different selection rules."
    note = (r"Windows of H$_2$O (360 generators), Hartree--Fock start; energy error: mean energy above the full configuration interaction energy of the window, in mHa; 48 trajectories for the random rules; "
            r"exact ADAPT is still at $5.9$ and $3.7$~mHa after 40 iterations. Columns: exact ADAPT; the generator with the largest one-parameter lowering; the largest Newton estimate with the curvature at the "
            r"Hartree--Fock determinant; the lowest energy after reoptimization among the 3 generators of largest $\lvert g_i\rvert$; uniform draws from the $\tau$-set and the $\rho$-set; and, in the last column, "
            r"the member of the $\rho$-set with the smallest one-parameter lowering (worst).")
    table("findings_policies_window", cap, "tab:find_window", "llrrrrrrrrr", lines, note)


def rho_map() -> None:
    a, b = read_csv(RUNS / "features/report/rho_map.csv"), read_csv(RUNS / "features/report/rho_map_full.csv")
    if a is None or b is None:
        return
    lines = [r" & & \multicolumn{4}{c}{Lowering by the one-parameter move} & \multicolumn{4}{c}{Lowering after reoptimizing all parameters} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"System & States & \shortstack{Top-$\lvert g\rvert$\\share} & \shortstack{Energy-best\\in $\rho$-set} & \shortstack{Worst $\rho$-good\\share} & $\rho_{\mathrm{needed}}$ & "
             r"\shortstack{Top-$\lvert g\rvert$\\share} & \shortstack{Energy-best\\in $\rho$-set} & \shortstack{Worst $\rho$-good\\share} & $\rho_{\mathrm{needed}}$ \\",
             r"\midrule"]
    for system in ("H4 1.0", "H4 2.0", "LiH 3.0", "H2O eq", "H2O str"):
        ra, rb = a[a.system == system], b[b.system == system]
        if ra.empty or rb.empty:
            continue
        ra, rb = ra.iloc[0], rb.iloc[0]

        def block(r):
            return (f"{r.top_g_share_mean:.2f} ({r.top_g_share_min:.2f}) & {r['covered_0.1_pct']:.0f}\\% & {r['worst_0.1_mean']:.2f} ({r['worst_0.1_min']:.2f}) & "
                    f"{r.rho_cover_median:.2f} ({r.rho_cover_max:.2f})")
        lines.append(f"{SYS[SHORT[system]]} & {int(ra.states)}/{int(rb.states)} & {block(ra)} & {block(rb)} \\\\")
    cap = r"What a tolerance on the gradient guarantees for the energy lowering."
    note = (r"States: the Hartree--Fock determinant, the CISD ground state and every state of the exact ADAPT trajectory; the two numbers are the states analysed for the one-parameter move and for the full "
            r"reoptimization after the generator is appended. Top-$\lvert g\rvert$ share: energy lowering of the generator of largest $\lvert g\rvert$ divided by the best lowering of any generator "
            r"(mean, with the minimum in parentheses). Energy-best in $\rho$-set: share of the states whose generator of best lowering lies in the $\rho$-set with $\rho=0.1$, that is, has $\lvert g\rvert\ge0.9\max_j\lvert g_j\rvert$. "
            r"Worst $\rho$-good share: lowering of the worst generator with $\lvert g\rvert\ge0.9\max_j\lvert g_j\rvert$ divided by the best lowering (mean, minimum). "
            r"$\rho_{\mathrm{needed}}=1-\lvert g_{\mathrm{best}}\rvert/\max_j\lvert g_j\rvert$: the tolerance whose $\rho$-set contains the generator of best lowering (median, maximum).")
    table("findings_rhomap", cap, "tab:find_rhomap", "lrrrrrrrrr", lines, note)


def e5() -> None:
    import energy_e5_compare as e5m

    rng = np.random.default_rng(0)
    lines = [r"Minimum shots per fold & Runs & Iterations (mean) & Runs above exact & $\rho$-good selections & Exact-best selections & Selection shots, mean [95\% CI] \\", r"\midrule"]
    for label, tags, method in (("50 (default)", [""], "II-A data, safe"), ("200", ["minshots200"], "II-A data, safe, min shots=200"),
                                ("400", ["minshots400"], "II-A data, safe, min shots=400")):
        import os

        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            traj, sel, exact_steps = e5m.rows_of("LiH_R3p0_HF", tags, method)
        finally:
            os.chdir(cwd)
        if not traj:
            continue
        steps = np.array([float(r["iterations"]) for r in traj])
        shots = np.array([float(r["total_shots"]) for r in traj])
        sh = np.array([float(r["shortfall"]) for r in sel])
        lo, hi = e5m.boot_ci(shots, rng)
        lines.append(f"{label} & {len(traj)} & {steps.mean():.3f} & {100 * np.mean(steps > exact_steps):.1f}\\% & {100 * np.mean(sh <= 0.1 + 1e-12):.1f}\\% & "
                     f"{100 * np.mean(sh < 1e-9):.1f}\\% & {sci(shots.mean(), 2)} [{sci(lo, 2)}, {sci(hi, 2)}] \\\\")
    cap = r"Complete II-A trajectories on LiH 3.0~\AA\ with different minimum numbers of shots per fold."
    note = (r"Exact ADAPT needs 6 iterations. II-A with a priori first radius, $\rho=0.1$ and per-round intervals (\notationref{}); the minimum number of shots per context and fold is also the "
            r"floor of the learned estimators. Default (50): the 90 runs of the main set (seeds 7 and 8). Runs above exact: share of the trajectories that need more iterations than exact ADAPT; the extra "
            r"iteration is the only effect. $\rho$-good and exact-best selections: share of all selections of the runs. Selection shots: mean over the runs with a 95\% bootstrap interval.")
    table("findings_e5", cap, "tab:find_e5", "lrrrrrl", lines, note)


def measured_shares() -> None:
    srcs = {"9": RUNS / "features/report/cost_totals.csv", "5": RUNS / "features_gtol1e-5_cl/report/cost_totals.csv", "3": RUNS / "features_gtol1e-3_cl/report/cost_totals.csv"}
    tabs = {k: read_csv(p) for k, p in srcs.items()}
    if any(v is None for v in tabs.values()):
        return
    lines = [r" & & & \multicolumn{3}{c}{$C_{\mathrm{opt}}$ at $\epsilon_{\mathrm{opt}}=$} & \multicolumn{3}{c}{Selection share at $\epsilon_{\mathrm{opt}}=$} \\", r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"System & Runs & Selection shots & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]
    for case in ("H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"):
        rows = {}
        for k, tab in tabs.items():
            s = tab[(tab.case == case) & (tab.method.str.startswith("II-A")) & ((tab.confidence_pct - 95).abs() < 0.5)]
            if s.empty:
                break
            rows[k] = s.iloc[0]
        if len(rows) < 3:
            continue
        lines.append(f"{SYS[case]} & {int(rows['9'].n)} & {sci(rows['9'].selection_median)} & " + " & ".join(sci(rows[k].optimisation_median) for k in ("9", "5", "3")) + " & "
                     + " & ".join(f"{rows[k].selection_share_pct:.2g}\\%" for k in ("9", "5", "3")) + r" \\")
    cap = r"Share of the selection in the total cost of complete II-A runs at three optimizer tolerances."
    note = (r"II-A at 95\%. Replay of the recorded runs (cluster build): the selection shots are unchanged, the optimization is rerun at $\epsilon_{\mathrm{opt}}$; medians. "
            r"$C_{\mathrm{opt}}$: modeled optimization cost, with the energy shots per evaluation of the model. Selection share: selection shots over the sum of selection shots and $C_{\mathrm{opt}}$.")
    table("findings_shares", cap, "tab:find_shares", "lrrrrrrrr", lines, note)


def main() -> None:
    for fn in (repro, corr, gate, evals_env, gtol_paths, conf, cost_totals, cost_tolerances, tol_energy, premium, policies, rho_evals, nrho_tables, conf_direct, noise_tables, window_share, rho_map, e5, measured_shares):
        try:
            fn()
        except Exception as exc:  # a missing input must not stop the other tables
            print(f"{fn.__name__}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
