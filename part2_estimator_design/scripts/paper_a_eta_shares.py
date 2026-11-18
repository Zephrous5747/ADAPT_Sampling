#!/usr/bin/env python3
"""Paper A: selection share of the modeled total cost for other numbers of energies per parameter (round-1 review, D3).

The optimization cost of the draft is ``C_opt = sum_k n_ev_k (1 + eta n_k) m_E_k`` with eta = 2 energies per parameter.  The stored runs hold the
per-iteration terms for eta = 2 (``c_opt_steps``), so the cost for another eta is the same terms multiplied by ``(1 + eta n_k) / (1 + 2 n_k)``.
This script prints the share ``median C_sel / median (C_sel + C_opt)`` of II-A at 95 % on the five systems of Fig. 5a for eta = 2, 4 and 8 at the
three optimizer thresholds, with the same runs as ``energy_cost_curves.py`` (the roots of the first and the extension sets, the later root
winning), checks the eta = 2 column against the stored ``cost_totals.csv`` and writes ``reports/generated/tab_eta_shares.tex`` for the SI.

    python scripts/paper_a_eta_shares.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
GEN = ROOT.parent / "reports" / "generated"

# threshold -> (roots in the order of energy_cost_curves.py, report directory of the stored table)
SETS = {"9": (["features", "features_ext"], "features"),
        "5": (["features_gtol1e-5_cl", "features_ext_gtol1e-5_cl"], "features_gtol1e-5_cl"),
        "3": (["features_gtol1e-3_cl", "features_ext_gtol1e-3_cl"], "features_gtol1e-3_cl")}
CASES = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0~\AA"), ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0~\AA"), ("LiH_R3p0_HF", r"LiH 3.0~\AA"),
         ("H2O_eq_HF", r"H$_2$O eq"), ("H2O_stretch_HF", r"H$_2$O str")]
ETAS = (2, 4, 8)
METHOD, DELTA = "II-A data, safe", 0.05


def load(case: str, roots: list[str]) -> list[dict]:
    found = {}
    for root in roots:
        case_dir = RUNS / root / case
        for path in sorted(case_dir.glob("ci_*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    found[(r["method"], round(r["delta"], 6), int(r["index"]), r.get("source", "measured"))] = r
    return [r for (method, delta, _, _), r in found.items() if method == METHOD and abs(delta - DELTA) < 1e-9]


def terms(run: dict, eta: float) -> tuple[float, float]:
    sel = float(sum(s["shots"] for s in run["selections"]))
    steps = np.asarray(run["c_opt_steps"], float)
    n = np.asarray(run["n_params"], float)[: len(steps)]
    assert len(n) == len(steps)
    return sel, float(np.sum(steps * (1 + eta * n) / (1 + 2 * n)))


def pct(x: float) -> str:
    return f"{x:.2g}" + r"\%"


def bootstrap(sel: np.ndarray, tot: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
    """95 % percentile interval of median(sel) / median(tot) over resamples of the runs."""
    idx = rng.integers(0, len(sel), (n, len(sel)))
    share = 100 * np.median(sel[idx], axis=1) / np.median(tot[idx], axis=1)
    lo, hi = np.percentile(share, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    lines = [r"System & \multicolumn{3}{c}{$\epsilon_{\mathrm{opt}}=10^{-9}$} & \multicolumn{3}{c}{$\epsilon_{\mathrm{opt}}=10^{-5}$} & \multicolumn{3}{c}{$\epsilon_{\mathrm{opt}}=10^{-3}$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             " & " + " & ".join(f"$\\eta={e}$" for _ in SETS for e in ETAS) + r" \\", r"\midrule"]
    ci_lines = [r"System & Runs & $\epsilon_{\mathrm{opt}}=10^{-9}$ & $\epsilon_{\mathrm{opt}}=10^{-5}$ & $\epsilon_{\mathrm{opt}}=10^{-3}$ \\", r"\midrule"]
    rng = np.random.default_rng(3)
    for case, label in CASES:
        cells, ci_cells, n_runs = [], [], 0
        for k, (roots, _) in SETS.items():
            runs = load(case, roots)
            n_runs = len(runs)
            sel = np.array([terms(r, 2)[0] for r in runs])
            tot = np.array([sum(terms(r, 2)) for r in runs])
            lo, hi = bootstrap(sel, tot, rng)
            ci_cells.append(f"{100 * np.median(sel) / np.median(tot):.2g}\\% [{lo:.2g}, {hi:.2g}]")
        ci_lines.append(f"{label} & {n_runs} & " + " & ".join(ci_cells) + r" \\")
        print(case, "bootstrap", ci_cells)
        for k, (roots, report) in SETS.items():
            runs = load(case, roots)
            stored = pd.read_csv(RUNS / report / "report" / "cost_totals.csv")
            row = stored[(stored.case == case) & (stored.method == METHOD) & ((stored.confidence_pct - 95.0).abs() < 0.5)].iloc[0]
            assert len(runs) == int(row.n), (case, k, len(runs), int(row.n))
            for eta in ETAS:
                sel = np.array([terms(r, eta)[0] for r in runs])
                tot = np.array([sum(terms(r, eta)) for r in runs])
                share = 100 * float(np.median(sel) / np.median(tot))
                if eta == 2:
                    assert abs(share - float(row.selection_share_pct)) < 1e-6 * max(1.0, share), (case, k, share, float(row.selection_share_pct))
                cells.append(share)
        print(case, [round(c, 3) for c in cells])
        lines.append(f"{label} & " + " & ".join(pct(c) for c in cells) + r" \\")
    text = "\n".join([r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule", *lines, r"\bottomrule", r"\end{tabular}", ""])
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_eta_shares.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", GEN / "tab_eta_shares.tex")
    text = "\n".join([r"\begin{tabular}{@{}lrlll@{}}", r"\toprule", *ci_lines, r"\bottomrule", r"\end{tabular}", ""])
    (GEN / "tab_share_ci.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", GEN / "tab_share_ci.tex")


if __name__ == "__main__":
    main()
