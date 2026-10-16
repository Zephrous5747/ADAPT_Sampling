#!/usr/bin/env python3
"""Compare the numbers of the colleague's report (reports/TZS_three experiments.tex) with a re-run of her pipeline.

The report was produced by ``features_report.py`` from ``runs/features``; after the cluster re-run
(``cluster/jobs/make_feat_jobs.sh``) the same script gives new tables in ``runs/features/report``.  Here every number of
her tables is read back from the LaTeX and set against ours, cell by cell:

* S1 (``tab:S_grad_energy``), Table 1 (``tab:corr_energy``), Table 2 (``tab:ci_opt``), Table 3 (``tab:err_opt``);
* the deterministic parts (exact ADAPT, energies, correlations) must agree to the printed precision;
* the runs are deterministic in (seed, trajectory index), so with the same code and the same indices the sampled parts
  must agree as well; a difference is reported with its size and is either a different number of runs or a difference
  of the code.

    python scripts/features_compare_tzs.py --tex "../reports/TZS_three experiments.tex" --report runs/features/report
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SYSTEMS = {"H$_4$ 1.0~\\AA": "H4 1.0", "H$_4$ 2.0~\\AA": "H4 2.0", "LiH 3.0~\\AA": "LiH 3.0",
           "H$_2$O eq": "H2O eq", "H$_2$O str": "H2O str"}


def table_body(tex: str, label: str) -> list[list[str]]:
    """Cells of every data row of the table with this label (rules and multicolumn headers dropped)."""
    start = tex.index(f"\\label{{{label}}}")
    body = tex[tex.index("\\begin{tabular}", start): tex.index("\\end{tabular}", start)]
    body = body[body.index("\n") + 1:]
    rows = []
    for raw in re.split(r"\\\\", body):
        text = "\n".join(line for line in raw.splitlines()
                         if not line.strip().startswith(("\\toprule", "\\midrule", "\\cmidrule", "\\bottomrule")))
        text = re.sub(r"\\(toprule|midrule|bottomrule)", "", text).strip()
        if not text or text.startswith(("\\multicolumn{9}", " &  & \\multicolumn")) or "\\multicolumn{3}" in text[:40]:
            continue
        rows.append([c.strip() for c in text.split("&")])
    return rows


def numbers(cell: str) -> list[float]:
    """All numbers of a cell, ``1.5\\times10^{6}`` read as 1.5e6."""
    out = []
    cell = re.sub(r"(\d+\.?\d*)\s*\\times10\^\{(-?\d+)\}", lambda m: f"{float(m.group(1)) * 10 ** int(m.group(2)):.6e}", cell)
    for m in re.finditer(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", cell.replace("--", " ").replace("\\%", "")):
        out.append(float(m.group(0)))
    return out


def system_of(cell: str, current: str | None) -> str | None:
    cell = cell.strip()
    return SYSTEMS.get(cell, current if cell == "" else None) if cell in SYSTEMS or cell == "" else current


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def add(self, table: str, key: str, quantity: str, theirs: float, ours: float, tol: float, rel: bool = False) -> None:
        diff = ours - theirs
        scale = abs(theirs) if rel and theirs else 1.0
        ok = abs(diff) / scale <= tol
        self.rows.append((table, key, quantity, theirs, ours, diff, ok))

    def show(self) -> int:
        bad = [r for r in self.rows if not r[-1]]
        by_table = {}
        for r in self.rows:
            by_table.setdefault(r[0], [0, 0])
            by_table[r[0]][0] += 1
            by_table[r[0]][1] += r[-1]
        for t, (n, ok) in by_table.items():
            print(f"{t}: {ok}/{n} numbers agree")
        if bad:
            print("\ndifferences:")
            for table, key, q, a, b, d, _ in bad:
                print(f"  {table:8s} {key:34s} {q:24s} theirs {a:12.5g} ours {b:12.5g} diff {d:+.4g}")
        return len(bad)


def compare_s1(tex: str, report: Path, rep: Report) -> None:
    ours = pd.read_csv(report / "tableS1_gradient_vs_energy.csv").set_index("system")
    system = None
    for cells in table_body(tex, "tab:S_grad_energy"):
        if cells[0] in SYSTEMS:
            system = SYSTEMS[cells[0]]
        if system is None or system not in ours.index or len(cells) < 8:
            continue
        o = ours.loc[system]
        v = [numbers(c) for c in cells]
        rep.add("S1", system, "states", v[1][0], o.states, 0)
        rep.add("S1", system, "spearman median", v[2][0], o.spearman_median, 0.011)
        rep.add("S1", system, "spearman min", v[2][1], o.spearman_min, 0.011)
        rep.add("S1", system, "top |g| is best %", v[3][0], o.top_g_is_best_dE_pct, 0.6)
        rep.add("S1", system, "share top g mean", v[4][0], o.share_top_g_mean, 0.011)
        rep.add("S1", system, "share top g min", v[4][1], o.share_top_g_min, 0.011)
        rep.add("S1", system, "share worst rho-good", v[5][0], o.share_worst_rho_good_mean, 0.011)
        rep.add("S1", system, "share worst rho-good min", v[5][1], o.share_worst_rho_good_min, 0.011)
        if v[6]:
            rep.add("S1", system, "spearman full", v[6][0], o.spearman_full_median, 0.011)
            rep.add("S1", system, "share top g full", v[7][0], o.share_top_g_full_mean, 0.011)


def compare_t1(tex: str, report: Path, rep: Report) -> None:
    ours = pd.read_csv(report / "table1_gradient_correlation_vs_energy.csv")
    system = None
    for cells in table_body(tex, "tab:corr_energy"):
        if cells[0] in SYSTEMS:
            system = SYSTEMS[cells[0]]
        if len(cells) < 9 or cells[1] not in ("II-0", "II-A") or system is None:
            continue
        sel = ours[(ours.system == system) & (ours.design == cells[1])]
        if sel.empty:
            continue
        o = sel.iloc[0]
        v = [numbers(c) for c in cells]
        key = f"{system} {cells[1]}"
        if not v[2]:
            continue
        rep.add("Table 1", key, "pairs", v[2][0], o.pairs, 1)
        rep.add("Table 1", key, "r median", v[3][0], o.r_median, 0.011)
        rep.add("Table 1", key, "|r| p90", v[4][0], o.abs_r_p90, 0.011)
        rep.add("Table 1", key, "chi median", v[5][0], o.chi_median, 0.011)
        rep.add("Table 1", key, "chi min", v[5][1], o.chi_min, 0.011)
        rep.add("Table 1", key, "chi max", v[5][2], o.chi_max, 0.011)
        rep.add("Table 1", key, "top-pair r", v[6][0], o.top_pair_r_median, 0.011)
        rep.add("Table 1", key, "top-pair chi", v[7][0], o.top_pair_chi_median, 0.011)
        rep.add("Table 1", key, "spearman(r, dE)", v[8][0], o.spearman_r_vs_lowering_ratio, 0.021)


def compare_t2(tex: str, report: Path, rep: Report) -> None:
    ours = pd.read_csv(report / "table2_confidence_vs_optimization.csv")
    system = None
    for cells in table_body(tex, "tab:ci_opt"):
        if cells[0] in SYSTEMS:
            system = SYSTEMS[cells[0]]
        if system is None or len(cells) < 12 or "exact" in cells[1] or not numbers(cells[1]):
            continue
        conf = numbers(cells[1])[0]
        sel = ours[(ours.system == system) & ((ours.confidence_pct - conf).abs() < 0.6)]
        if sel.empty:
            continue
        o = sel.iloc[0]
        v = [numbers(c) for c in cells]
        key = f"{system} {conf:.0f}%"
        if not v[3]:
            continue
        rep.add("Table 2", key, "runs", v[2][0], o.trajectories, 0)
        rep.add("Table 2", key, "median shots", v[3][0], o.selection_shots_median, 0.02, rel=True)
        rep.add("Table 2", key, "steps mean", v[4][0], o.iterations_mean, 0.06)
        rep.add("Table 2", key, "opt evals mean", v[5][0], o.optimizer_evaluations_mean, 0.6)
        rep.add("Table 2", key, "evals ratio", v[6][0], o.optimizer_evaluations_ratio, 0.011)
        rep.add("Table 2", key, "C_opt ratio", v[7][0], o.c_opt_ratio, 0.011)
        rep.add("Table 2", key, "dE median mHa", v[8][0], o.final_error_median_mHa, 0.011)
        rep.add("Table 2", key, "same E %", v[9][0], o.reached_pct, 0.6)
        rep.add("Table 2", key, "exact best %", v[10][0], o.exact_best_pct, 0.6)
        rep.add("Table 2", key, "rho-good %", v[11][0], o.rho_good_pct, 0.6)


def compare_t3(tex: str, report: Path, rep: Report) -> None:
    ours = pd.read_csv(report / "table3_errors_vs_optimization_cost.csv")
    system = None
    for cells in table_body(tex, "tab:err_opt"):
        if cells[0] in SYSTEMS:
            system = SYSTEMS[cells[0]]
        if system is None or len(cells) < 11 or "exact" in cells[1] or not numbers(cells[2]):
            continue
        label = cells[1]
        if label.startswith("II-A"):
            bucket = "3 or more" if "3 or more" in label else re.search(r"II-A, (\d)", label).group(1)
            sel = ours[(ours.system == system) & ours.group.str.startswith("certified") & (ours.label == bucket)]
        else:
            m = re.search(r"fixed ([\d.]+)e(\d+)", label)
            if not m:
                continue
            target = float(m.group(1)) * 10 ** int(m.group(2))
            sel = ours[(ours.system == system) & ours.group.str.startswith("uncertified")]
            sel = sel[[abs(float(re.search(r"([\d.]+)e(\d+)", s).group(1)) * 10 ** int(re.search(r"([\d.]+)e(\d+)", s).group(2)) - target)
                       < 0.06 * target for s in sel.label]]
        if sel.empty:
            rep.rows.append(("Table 3", f"{system} {label}", "row missing in re-run", 0, 0, 0, False))
            continue
        o = sel.iloc[0]
        v = [numbers(c) for c in cells]
        key = f"{system} {label}"
        rep.add("Table 3", key, "runs", v[2][0], o.trajectories, 0)
        rep.add("Table 3", key, "non-best/run", v[3][0], o.non_best_per_run, 0.06)
        rep.add("Table 3", key, "not rho-good/run", v[4][0], o.not_rho_good_per_run, 0.06)
        rep.add("Table 3", key, "steps mean", v[5][0], o.iterations_mean, 0.06)
        rep.add("Table 3", key, "evals ratio", v[6][0], o.optimizer_evaluations_ratio, 0.011)
        rep.add("Table 3", key, "C_opt ratio", v[7][0], o.c_opt_ratio, 0.011)
        rep.add("Table 3", key, "dE median mHa", v[8][0], o.final_error_median_mHa, 0.011)
        rep.add("Table 3", key, "same E %", v[9][0], o.same_energy_pct, 0.6)
        rep.add("Table 3", key, "CA %", v[10][0], o.reached_ca_pct, 0.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=Path("../reports/TZS_three experiments.tex"))
    parser.add_argument("--report", type=Path, default=Path("runs/features/report"))
    parser.add_argument("--only", nargs="*", default=["S1", "T1", "T2", "T3"])
    args = parser.parse_args()
    tex = args.tex.read_text(encoding="utf-8")
    rep = Report()
    for name, fn, csv in (("S1", compare_s1, "tableS1_gradient_vs_energy.csv"),
                          ("T1", compare_t1, "table1_gradient_correlation_vs_energy.csv"),
                          ("T2", compare_t2, "table2_confidence_vs_optimization.csv"),
                          ("T3", compare_t3, "table3_errors_vs_optimization_cost.csv")):
        if name in args.only and (args.report / csv).exists():
            fn(tex, args.report, rep)
        elif name in args.only:
            print(f"{name}: {csv} not in {args.report}, skipped")
    return 1 if rep.show() else 0


if __name__ == "__main__":
    raise SystemExit(main())
