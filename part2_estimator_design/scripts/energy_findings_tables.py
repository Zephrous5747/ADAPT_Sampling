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


def table(name: str, caption: str, label: str, colspec: str, lines: list[str]) -> None:
    text = "\n".join([r"\begin{table}[!htb]", r"\centering\small", r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\fitcol{%", r"\begin{tabular}{" + colspec + "}", r"\toprule",
                      *lines, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
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
    lines = [r"System & design & pairs & median $r$ & $|r|_{90\%}$ & $\chi$ median (range) & top pair $r$ / $\chi$ & $\rho_s(r,\Delta E)$ \\", r"\midrule"]
    last = None
    for r in t.itertuples():
        label = SYS.get(SHORT.get(r.system, r.system), r.system)
        lines.append(f"{label if last != r.system else ''} & {r.design} & {r.pairs} & {r.r_median:+.2f} & {r.abs_r_p90:.2f} & {r.chi_median:.2f} ({r.chi_min:.2f}--{r.chi_max:.2f}) & "
                     f"{r.top_pair_r_median:+.2f} / {r.top_pair_chi_median:.2f} & {r.spearman_r_vs_lowering_ratio:+.2f} \\\\")
        last = r.system
    cap = (r"Correlation of the gradient estimates from shared measurements against energy lowering (her Table 1), now with the H$_2$O rows: 819 and 864 pairs from the 18 and 19 states of the exact ADAPT "
           r"trajectories. $r$: correlation of the estimates of $|g|$ of the leader and each challenger; $\chi$: factor by which the shared data change the shots of their comparison; $\rho_s$: Spearman correlation "
           r"of $r$ with the challenger's lowering over the leader's.")
    table("findings_corr", cap, "tab:find_corr", "llrrrrrr", lines)


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
    lines = [r"System & steps tested & methods $\times$ streams & selections compared & identical & time original / fast (s) \\", r"\midrule"]
    for case, steps, n_meth, n_seed, n_ok, n_bad, t_o, t_f in rows:
        label = SYS.get(case.replace("_HF", "_HF"), case)
        lines.append(f"{label} & {', '.join(map(str, steps))} & {n_meth} $\\times$ {n_seed} & {n_ok + n_bad} & {n_ok} & {t_o} / {t_f} \\\\")
    cap = (r"Gate job: the fast statistics of \texttt{fastpath.py} against the original code of \texttt{LearnedM3}, on the same states and random streams "
           r"(II-A and, for LiH and H$_4$, also II-0). ``Identical'' means the same generator, rounds, stopping reason, shortfall and shots (relative $10^{-9}$). "
           r"Replaying the recorded H$_2$O trajectories reproduced their final energies to $0.0$~Ha on the cluster ($1.6\times10^{-13}$~Ha locally).")
    table("findings_gate", cap, "tab:find_gate", "lllrrr", lines)


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
    lines = [r"System & her machine & WSL, $10^{-9}$ & cluster, $10^{-9}$ & WSL, $10^{-5}$ & cluster, $10^{-5}$ & WSL, $10^{-3}$ & cluster, $10^{-3}$ \\", r"\midrule"]
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
    cap = (r"Optimizer evaluations of the exact-ADAPT path (largest $|g|$, ties to the lowest index; BFGS, gradient tolerance in the header, to chemical accuracy) "
           r"on three machines/builds. At the tight tolerance the count depends on the build of the Hamiltonian (the SCF orbitals of degenerate subspaces differ), so only ratios within one build mean anything; "
           f"at $10^{{-5}}$ and $10^{{-3}}$ the WSL and the cluster build give the same count in {same} of {len(pairs)} comparisons. "
           f"In one build the count falls by {span('wsl9', 'wsl5')} times (WSL) and {span('cl9', 'cl5')} times (cluster) at $10^{{-5}}$ and by {span('wsl9', 'wsl3')} (WSL) and {span('cl9', 'cl3')} (cluster) at $10^{{-3}}$ "
           r"(stretched H$_4$ left out). "
           r"$^\dagger$With the looser tolerance stretched H$_4$ no longer stalls on the gradient test and the run stops on the energy gain after 13 steps instead of 10.")
    table("findings_evals", cap, "tab:find_evals", "lrrrrrrr", lines)


def gtol_paths() -> None:
    d = RUNS / "energy_policies" / "LiH_R3p0_HF"
    lines = [r"Order of the generators (LiH 3.0~\AA, 6 steps) & choices allowed & runs & evaluations (cluster, $10^{-9}$) \\", r"\midrule"]
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
    for build, tag in (("WSL build", "energy_policies_gtol{g}"), ("cluster build", "energy_policies_gtol{g}_cl")):
        cells = []
        n = 0
        for gt in ("1e-5", "1e-3"):
            w = RUNS / tag.format(g=gt) / "LiH_R3p0_HF" / "policy_rho_random_0.05.jsonl"
            if w.exists():
                rows = [json.loads(x) for x in w.read_text().splitlines() if x.strip()]
                n = max(n, len(rows))
                cells.append(f"{', '.join(map(str, sorted({r['evals_total'] for r in rows})))} at $10^{{{'-5' if gt == '1e-5' else '-3'}}}$")
        if cells:
            extra.append(f"all orders ({build}) & within 5\\% & {n} & {'; '.join(cells)} \\\\")
    if extra:
        lines.append(r"\midrule")
        lines.extend(extra)
    cap = (r"The path dependence of the optimizer cost at the tight tolerance, LiH 3.0~\AA: the same six generators (up to symmetry partners; D: double, S: single, orbital labels as in the pool) "
           r"added in different orders end at the same energy but need between $54$ and $252$ BFGS evaluations at $10^{-9}$ (cluster build; $83$ or $156$ in the WSL build). "
           r"At $10^{-5}$ every order, tied or near-tied, needs $39$ or $40$ and at $10^{-3}$ $30$ or $31$ (both builds, 100 and 200 random orders).")
    table("findings_gtol_paths", cap, "tab:find_gtol_paths", "p{6.5cm}lrl", lines)


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

    lines = [r"System & conf. & runs & selection shots & steps & $C_\mathrm{opt}$ / top [95\% CI], $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]
    last = None
    for r in t9.itertuples():
        label = SYS.get(SHORT.get(r.system, r.system), r.system)
        if last is not None and last != r.system:
            lines.append(r"\midrule")
        last = r.system
        lines.append(f"{label if r.confidence_pct == t9[t9.system == r.system].confidence_pct.max() else ''} & {r.confidence_pct:.0f}\\% & {r.n} & {sci(r.selection_median)} & "
                     f"{r.steps_mean:.2f} & {interval(r.c_opt_ratio_top, r.c_opt_lo, r.c_opt_hi)} & {cell(t5, r)} & {cell(t3, r)} \\\\")
    cap = (r"Confidence level of the selection against the optimization cost of the complete run (II-A, sign-aware rule, a priori start, $\rho=0.1$): $C_\mathrm{opt}$ relative to the highest level of the system, "
           r"at the BFGS gradient tolerance $10^{-9}$ of the draft and, by replaying the recorded choices, at $10^{-5}$ and $10^{-3}$ (cluster build). "
           r"Runs: 48 per level for H$_4$, 24 for LiH; H$_2$O: 12 at each of 99, 90 and 68\% (new) and the 36 recorded runs at 95\% replayed with their optimization (the only level that was not run afresh). "
           r"``Top'' is the highest level of the system and the interval is a bootstrap interval of the ratio of the means. "
           r"At $10^{-9}$ the cost of H$_4$ 2.0~\AA\ rises monotonically as the confidence falls; at $10^{-5}$ and $10^{-3}$ no level differs from the top one by more than a few percent (Sec.~\ref{sec:gtol}).")
    table("findings_conf", cap, "tab:find_conf", "lrrrrlll", lines)


def cost_totals() -> None:
    t = read_csv(RUNS / "features/report/cost_totals.csv")
    if t is None:
        return
    order = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
    lines = [r"System & method & runs & steps & reach & selection share & total / II-A (95\%) \\", r"\midrule"]
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
                name = r.method.replace(" safe", "").replace(", safe", "").replace("M1 seq", "M1 seq.")
            lines.append(f"{SYS[case] if first else ''} & {name} & {r.n} & {r.steps_mean:.1f} & {pct(r.reached_pct)} & {r.selection_share_pct:.2g}\\% & "
                         f"{r.total_median / ref.total_median:.2f} \\\\")
            first = False
        lines.append(r"\midrule")
    lines.pop()
    cap = (r"Selection and optimization together: median total shots (selection plus modelled $C_\mathrm{opt}$, $\eta=2$, BFGS tolerance $10^{-9}$) of complete measured runs "
           r"relative to II-A at 95\%. Selection share: selection shots over the total. Reach: share of runs that reach chemical accuracy "
           r"(a run that does not is stopped at ten steps above exact ADAPT's count, so its total is a lower bound). The fixed budgets are the uncertified pilot-allocation selections. "
           r"II-A rows: 48 (H$_4$), 24 (LiH) runs per level from the larger sets, 12 (H$_2$O eq) at 99, 90 and 68\%, 36 at 95\%; the other methods are the 36 recorded runs (H$_2$O), 12 runs per budget. "
           r"At this tolerance a median total fluctuates by $\pm20$--$30\%$ between sets of 24 runs (tie-order noise of the optimizer), by $\pm7\%$ for 36 runs of H$_2$O.")
    table("findings_cost", cap, "tab:find_cost", "llrrrrr", lines)


def cost_tolerances() -> None:
    srcs = {"9": RUNS / "features/report/cost_totals.csv", "5": RUNS / "features_gtol1e-5_cl/report/cost_totals.csv", "3": RUNS / "features_gtol1e-3_cl/report/cost_totals.csv"}
    tabs = {k: read_csv(p) for k, p in srcs.items()}
    if tabs["9"] is None or tabs["5"] is None or tabs["3"] is None:
        return
    order = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
    lines = [r"System & method & runs & reach, $10^{-9}$ / $10^{-3}$ & share, $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & total / II-A (95\%), $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]

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
                name = r.method.replace(" safe", "").replace(", safe", "").replace("M1 seq", "M1 seq.")
            lines.append(f"{SYS[case] if first else ''} & {name} & {int(cells['9'].n)} & {pct(cells['9'].reached_pct)} / {pct(cells['3'].reached_pct)} & "
                         + " & ".join(f"{cells[k].selection_share_pct:.2g}\\%" for k in ("9", "5", "3")) + " & "
                         + " & ".join(f"{cells[k].total_median / refs[k].total_median:.2f}" for k in ("9", "5", "3")) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines.pop()
    cap = (r"Selection and optimization together at three BFGS gradient tolerances: share of the selection in the total and median total relative to II-A at 95\%, by replaying the recorded runs "
           r"(selections and selection shots kept, optimization re-run; cluster build; the other methods are the 36 recorded runs of H$_2$O, the fixed budgets 12 runs each). "
           r"Reach is the share of runs that end at chemical accuracy at the tight and at the loosest tolerance; a run that does not reach it is stopped at ten steps above exact ADAPT's count, so its total is a lower bound. "
           r"At the looser tolerances the optimization is cheaper and every difference in selection shots weighs more: II-0 and M2 cost more than II-A, sequential M1 several times as much.")
    table("findings_cost_tol", cap, "tab:find_cost_tol", "llrrrrrrrr", lines)


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

    lines = [r"tolerance & runs & largest change (mHa) & runs changing by $>0.1$~mHa & of these, H$_4$ 1.0~\AA \\", r"\midrule"]
    for g, name in (("1e-5", "$10^{-5}$"), ("1e-3", "$10^{-3}$")):
        for kind, label in (("certified", "certified and baselines (II-A, II-0, M1, M2)"), ("fixed", "uncertified pilot budgets")):
            s = d[(d.gtol == g) & (d.kind == kind)]
            big = s[s.d.abs() > 0.1]
            lines.append(f"{name}, {label} & {len(s)} & {fmt(s.d.abs().max())} & {len(big)} & {int((big.case == 'H4_square_eq_side1p0_HF').sum())} \\\\")
    cap = (r"Change of the final energy when the optimizer stops at a looser gradient tolerance (replay of the recorded choices, run by run, against the same run at $10^{-9}$; cluster build). "
           r"The certified runs end within a few $\mu$Ha of their tight-tolerance energy; the uncertified runs with poor choices can stop on a plateau where $|g|$ is below the tolerance "
           r"although the energy is still far above the minimum (H$_4$ 1.0~\AA, $10^{-3}$: eight of the 23 pilot runs that reach chemical accuracy at $10^{-9}$ end $4$--$76$~mHa higher).")
    table("findings_tol_energy", cap, "tab:find_tol_energy", "lrrrr", lines)


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
    cap = (r"Optimizer evaluations of ADAPT with injected selection errors, relative to random choices among exactly tied generators in the same build (cluster build, 200 trajectories per entry), "
           r"at the BFGS gradient tolerances $10^{-9}$ / $10^{-5}$ / $10^{-3}$. At the tight tolerance a near-tie costs up to $+73\%$ (LiH) because the polishing depends on the order of the generators; "
           r"at $10^{-5}$ and $10^{-3}$ the count follows the number of steps: harmless tolerances cost at most a few percent, tolerances beyond $\rho^\ast$ cost what the extra steps cost.")
    table("findings_policies_rho_evals", cap, "tab:find_rho_evals", "lrrrrr", lines)


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
    blocks = (("steps_ratio", "Steps relative to exact ADAPT"), ("evals_ratio_tie_avg", "Optimizer evaluations relative to exact ADAPT"))
    lines = [r"System & " + " & ".join(rf"$R={r}$" for r in radii) + r" & $\tau=0.3$ \\", r"\midrule"]
    for col, title in blocks:
        lines.append(rf"\multicolumn{{{len(radii) + 2}}}{{l}}{{\emph{{{title}}}}} \\")
        for s in systems:
            lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(pair(s, r, col) for r in radii) + f" & {val(s, 'tau_random:0.3', col)} \\\\")
    cap = (r"Curvature-scaled tolerance, cluster build, BFGS gradient tolerance $10^{-5}$, 200 trajectories per entry. Each cell is \emph{plain gradient set} / \emph{curvature-scaled set}: the draw is uniform among "
           r"the generators with $|g_i|\ge(1-R)\max|g|$, respectively with $|g_i|/\sqrt{h_i}\ge(1-R)\max|g_j|/\sqrt{h_j}$, $h_i$ the diagonal curvature at the Hartree--Fock reference (no extra shots). "
           r"The last column is the draw inside the set of one-parameter energy lowerings within $30\%$ of the best (needs the lowering at every step). "
           r"The evaluation ratios are relative to the exact path of the same build, as in the preceding tables.")
    table("findings_policies_nrho", cap, "tab:find_nrho", "lrrrrrr", lines)

    w = ["H2O dz8o eq", "H2O dz8o str"]
    cols = [("rho_random:0.1", r"$\rho=0.1$"), ("nrho_random:0.1", r"curv.\ $0.1$"), ("rho_random:0.3", r"$\rho=0.3$"), ("nrho_random:0.3", r"curv.\ $0.3$"),
            ("rho_random:0.5", r"$\rho=0.5$"), ("nrho_random:0.5", r"curv.\ $0.5$"), ("tau_random:0.3", r"$\tau=0.3$"), ("onep_greedy", "one-param.")]
    lines = [r"System & quantity & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for s in w:
        for k, (col, name) in enumerate((("err_ratio_exact_steps", "error, step 40"), ("reached_pct", r"within target (\%)"), ("evals_ratio_tie_avg", "evaluations"))):
            fmt = (lambda p, c=col: f"{float(t[(t.system == s) & (t.policy == p)][c].iloc[0]):.0f}") if col == "reached_pct" else (lambda p, c=col: val(s, p, c))
            lines.append(f"{SYS[SHORT[s]] if k == 0 else ''} & {name} & " + " & ".join(fmt(p) for p, _ in cols) + r" \\")
    cap = (r"The 16-qubit cc-pVDZ windows of H$_2$O at the BFGS gradient tolerance $10^{-5}$ (cluster build, 48 trajectories per random policy, 40 steps). Error: mean energy error after 40 steps over that of "
           r"exact ADAPT; ``within target'': share of the trajectories ending below the target error of the analysis script (exact ADAPT's own final error plus its tolerance); "
           r"evaluations relative to exact ADAPT. ``curv.'' is the curvature-scaled gradient set.")
    table("findings_policies_nrho_window", cap, "tab:find_nrho_window", "llrrrrrrrr", lines)


def premium() -> None:
    bases ={"9": RUNS / "features_q10", "5": RUNS / "features_q10_gtol1e-5", "3": RUNS / "features_q10_gtol1e-3"}
    base = bases["9"]
    if not base.exists():
        return
    adequate = {"H4_square_eq_side1p0_HF": 13335, "H4_square_stretch_side2p0_HF": 133352, "LiH_R1p6_HF": 3162278, "LiH_R3p0_HF": 7498942,
                "H2O_eq_HF": 133352143, "H2O_stretch_HF": 421696503}
    lines = [r"System & fixed: budget per step / selection shots & steps, fixed (II-A) & premium, selection only & premium, total, $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & from steps alone \\", r"\midrule"]

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
        lines.append(f"{SYS[case]} & {sci(b)} / {sci(np.median(sel_f))} & {steps_f:.2f} ({steps_c:.2f}) & {np.median(sel_c) / np.median(sel_f):.1f} & "
                     + " & ".join(f"{totals[k][0]:.2f} ({totals[k][1]:.2f}--{totals[k][2]:.2f})" if k in totals else "--" for k in ("9", "5", "3")) + f" & {steps_c / steps_f:.2f} \\\\")
    cap = (r"Certification premium of Q10 in selection shots and in total shots. The fixed budget is the cheapest adequate one of Q10 (designed allocation: at least 95\% of the trajectories reach chemical accuracy "
           r"within one step of exact ADAPT's count); II-A is the certified selection at 95\%, both from the recorded Phase~4 runs with their optimization recomputed ($C_\mathrm{opt}$ model, $\eta=2$, "
           r"BFGS gradient tolerance $10^{-9}$ as in the draft, and $10^{-5}$ and $10^{-3}$ by replaying the recorded choices). The premium is the ratio II-A over fixed. Selection-only premium as in the draft; "
           r"in total shots the premium is the ratio of the medians of the sums (bootstrap 95\% interval in brackets), where the fixed budget pays for its longer trajectories. The last column is the ratio of the mean numbers of steps, the part of the premium that is due to the longer trajectories alone; at $10^{-9}$ the rest is mostly the path dependence of the optimizer's evaluation count (Sec.~\ref{sec:gtol}), which is larger for the runs with more near-tied choices.")
    if low:
        cap += (" The designed fixed budgets are adequate at $10^{-9}$ by construction; at the looser tolerances some of their runs stall on a plateau, and the share that reaches chemical accuracy falls to "
                + ", ".join(f"{v:.0f}\\% ({s}, $10^{{{'-5' if k == '5' else '-3'}}}$)" for s, k, v in low) + " (II-A: all runs).")
    if missing:
        cap += " Rows still to come (replay of the recorded runs running): " + ", ".join(missing) + "."
    table("findings_premium", cap, "tab:find_premium", "llrrrrrr", lines)


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

    lines = [r"System & exact steps & " + " & ".join(f"$\\rho={x:g}$" for x in rhos) + r" & $\rho^\ast$ & reach at $0.9$ \\", r"\midrule"]
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
    cap = (r"Steps of ADAPT when every selection is a uniform draw from the set $\{i:|g_i|\ge(1-\rho)\max|g|\}$, as a multiple of exact ADAPT's steps (exact gradients, exact optimization, "
           r"200 trajectories per entry, common random numbers over $\rho$). $\rho^\ast$: the largest $\rho$ of the grid up to which the steps stay within 3\% of exact ADAPT's. "
           r"Reach: share of the trajectories that end at chemical accuracy (or at what exact ADAPT reaches, for stretched H$_4$). $\rho=1$ (random ADAPT) still reaches it on H$_4$ (a pool of 26 generators) but in 10\% of the "
           r"LiH runs and in none of the BeH$_2$ and H$_2$O runs. All entries are at the optimizer tolerance $10^{-9}$; the steps are the same at $10^{-5}$ and $10^{-3}$ for $\rho\le0.7$, "
           r"but random ADAPT then reaches chemical accuracy in only about half of the H$_4$ 1.0~\AA\ runs ($51\%$, $48\%$) and in $6\%$ and $2\%$ of the LiH runs, because a plateau stops the optimizer (Sec.~\ref{sec:gtol}).")
    table("findings_policies_rho", cap, "tab:find_rho", "lrrrrrrrrrrr", lines)

    taus = [0.05, 0.1, 0.2, 0.3, 0.5]
    lines = [r"System & " + " & ".join(f"$\\tau={x:g}$" for x in taus) + r" & " + r"$\rho=0.5$ for comparison \\", r"\midrule"]
    for s in order:
        cells = [num(val(s, f"tau_random:{x:g}", "steps_ratio")) for x in taus]
        lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(cells) + f" & {num(val(s, 'rho_random:0.5', 'steps_ratio'))} \\\\")
    cap = (r"As the preceding table, with the draw from the set of generators whose one-parameter energy lowering is within a fraction $\tau$ of the best one "
           r"($\{i: \Delta E_i\ge(1-\tau)\max\Delta E\}$, exact lowering $\min_\theta E(e^{\theta G_i}\psi)-E(\psi)$). A tolerance on the energy lowering leaves the number of steps unchanged up to $\tau=0.5$ "
           r"in every system; on H$_4$ 1.0~\AA\ it saves two steps.")
    table("findings_policies_tau", cap, "tab:find_tau", "lrrrrrr", lines)

    greedy = ["onep_greedy", "newton_hf", "newton_exact", "reopt_top:3", "reopt_top:10", "reopt_all"]
    head = ["one-param.", "Newton, HF curv.", "Newton, exact curv.", "re-opt., top 3", "re-opt., top 10", "re-opt., all"]
    lines = [r"System & " + " & ".join(head) + r" \\", r"\midrule"]
    for s in order:
        lines.append(f"{SYS[SHORT[s]]} & " + " & ".join(num(val(s, g, "steps_ratio")) for g in greedy) + r" \\")
    cap = (r"Energy-greedy choices: steps relative to exact ADAPT when every selection is the generator with the largest one-parameter lowering (one-param.), the largest Newton estimate "
           r"$g_i^2/2h_i$ with the curvature $h_i$ of the Hartree--Fock reference (no shots) or of the current state, or the lowest energy after re-optimizing all parameters among the top 3 / top 10 gradients / all "
           r"generators with a gradient. Only H$_4$ 1.0~\AA\ gains steps (10 to 8; H$_2$O eq 17 to 16 with all candidates re-optimized). The steps are the same at $10^{-5}$ and $10^{-3}$, except that the H$_2$O eq gain disappears at $10^{-3}$.")
    table("findings_policies_greedy", cap, "tab:find_greedy", "lrrrrrr", lines)

    cols = ["exact", "onep_greedy", "newton_hf", "reopt_top:3", "tau_random:0.1", "tau_random:0.3", "rho_random:0.1", "rho_random:0.3", "rho_worst_e:0.1"]
    head = ["exact", "one-param.", "Newton HF", "re-opt. top 3", r"$\tau=0.1$", r"$\tau=0.3$", r"$\rho=0.1$", r"$\rho=0.3$", r"$\rho=0.1$, worst"]
    lines = [r"System & step & " + " & ".join(head) + r" \\", r"\midrule"]
    for s in ("H2O dz8o eq", "H2O dz8o str"):
        for col, name in (("err20_mHa", "20"), ("err_exact_steps_mHa", "40")):
            lines.append(f"{SYS[SHORT[s]] if name == '20' else ''} & {name} & " + " & ".join(num(val(s, c, col), 2) for c in cols) + r" \\")
    cap = (r"Sixteen-qubit cc-pVDZ window of H$_2$O (360 generators, Hartree--Fock start): mean energy error (mHa) after 20 and after 40 ADAPT steps for the choices of the header "
           r"(48 trajectories for the random ones; exact ADAPT is still at $5.9$ / $3.7$~mHa after 40 steps; the energies of the random choices are means). On the stretched geometry the energy-based choices are lower by $14$\% (step 40) and $24$\% (step 20); "
           r"on the equilibrium geometry a gradient tolerance of $0.1$ costs $3$\% and $0.3$ costs $9$\% in the energy at step 40, an adversarial choice inside the $\rho=0.1$ set $10$\%.")
    table("findings_policies_window", cap, "tab:find_window", "llrrrrrrrrr", lines)


def rho_map() -> None:
    a, b = read_csv(RUNS / "features/report/rho_map.csv"), read_csv(RUNS / "features/report/rho_map_full.csv")
    if a is None or b is None:
        return
    lines = [r" & & \multicolumn{4}{c}{one-parameter move} & \multicolumn{4}{c}{after re-optimizing all parameters} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"System & states & top $|g|$ share & energy-best in $\rho=0.1$ set & worst $\rho$-good & rho needed & top $|g|$ share & energy-best in $\rho=0.1$ set & worst $\rho$-good & rho needed \\",
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
    cap = (r"What the gradient tolerance guarantees for the energy lowering (HF, CISD and every state of the exact ADAPT trajectory; one-parameter move and full re-optimization after the generator is appended). "
           r"Top $|g|$ share: lowering of the largest-$|g|$ generator over the best lowering (mean, minimum in brackets). Energy-best in the set: share of states whose best generator has $|g|\ge0.9\max|g|$. "
           r"Worst $\rho$-good: the worst generator with $|g|\ge0.9\max|g|$ over the best (mean, minimum). Rho needed: $1-|g_\text{best}|/\max|g|$, the tolerance that contains the energy-best generator (median, maximum).")
    table("findings_rhomap", cap, "tab:find_rhomap", "lrrrrrrrrr", lines)


def e5() -> None:
    import energy_e5_compare as e5m

    rng = np.random.default_rng(0)
    lines = [r"Minimum shots per fold & runs & steps & runs above exact & $\rho$-good & exact best & selection shots, mean [95\% CI] \\", r"\midrule"]
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
    cap = (r"Repairing the small-sample intervals at trajectory level: complete II-A trajectories on LiH 3.0~\AA\ (exact ADAPT: 6 steps), a priori start, $\rho=0.1$, per-round intervals, "
           r"with the default minimum of 50 shots per fold and with 200 and 400 (the same minimum is the floor of the learned designs). Default: the 90 runs of the main set (seeds 7 and 8). "
           r"A trajectory is above exact if it needs more steps than exact ADAPT; the extra step is the only effect.")
    table("findings_e5", cap, "tab:find_e5", "lrrrrrl", lines)


def measured_shares() -> None:
    srcs = {"9": RUNS / "features/report/cost_totals.csv", "5": RUNS / "features_gtol1e-5_cl/report/cost_totals.csv", "3": RUNS / "features_gtol1e-3_cl/report/cost_totals.csv"}
    tabs = {k: read_csv(p) for k, p in srcs.items()}
    if any(v is None for v in tabs.values()):
        return
    lines = [r"System & runs & selection (II-A, 95\%) & $C_\mathrm{opt}$, $10^{-9}$ & $10^{-5}$ & $10^{-3}$ & share, $10^{-9}$ & $10^{-5}$ & $10^{-3}$ \\", r"\midrule"]
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
    cap = (r"Share of the selection in the total cost of the complete runs (II-A, 95\%) when the optimizer stops at a looser BFGS gradient tolerance, measured by replaying the recorded runs "
           r"(cluster build; selection shots unchanged, medians). The modelled $C_\mathrm{opt}$ of the draft is the $10^{-9}$ column; the energy shots per evaluation are those of the model.")
    table("findings_shares", cap, "tab:find_shares", "lrrrrrrrr", lines)


def main() -> None:
    for fn in (repro, corr, gate, evals_env, gtol_paths, conf, cost_totals, cost_tolerances, tol_energy, premium, policies, rho_evals, nrho_tables, rho_map, e5, measured_shares):
        try:
            fn()
        except Exception as exc:  # a missing input must not stop the other tables
            print(f"{fn.__name__}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
