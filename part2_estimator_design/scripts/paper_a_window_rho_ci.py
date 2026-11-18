#!/usr/bin/env python3
"""Paper A (round-2 review, M12): intervals for the effect of the tolerance on the energy of the CAS(8e,8o) active space.

``runs/energy_policies/H2O_dz8o_{eq,str}_HF/policy_*.jsonl`` hold 48 trajectories per random rule and the single exact-ADAPT path, with the
energy error after every iteration.  The Results quote, after 40 iterations, how much a uniform draw from the rho-set (rho = 0.1, 0.3) and
the worst member of the rho-set change the energy error against exact ADAPT.  This script gives the mean energy error after 40 iterations of
each rule, the relative change against exact ADAPT and its 95 % percentile bootstrap interval (resampling the 48 trajectories, 10,000
draws), and writes ``reports/generated/tab_window_rho_ci.tex`` for the SI.

    python scripts/paper_a_window_rho_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "energy_policies"
GEN = ROOT.parent / "reports" / "generated"
SYSTEMS = [("H2O_dz8o_eq_HF", "equilibrium"), ("H2O_dz8o_str_HF", "stretched")]
RULES = [("policy_rho_random_0.1.jsonl", r"$\rho=0.1$, uniform draw"), ("policy_rho_random_0.3.jsonl", r"$\rho=0.3$, uniform draw"),
         ("policy_rho_worst_e_0.1.jsonl", r"$\rho=0.1$, worst member"), ("policy_tau_random_0.3.jsonl", r"$\tau=0.3$, uniform draw")]
ITERATION = 40


def errors(path: Path) -> np.ndarray:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return np.array([r["energy_errors"][ITERATION] for r in rows], float) * 1e3  # mHa


def main() -> None:
    rng = np.random.default_rng(5)
    lines = [r"System & Rule & Trajectories & Energy error (mHa) & Change against exact ADAPT \\", r"\midrule"]
    for case, label in SYSTEMS:
        exact = errors(RUNS / case / "policy_exact.jsonl")
        assert exact.size == 1
        first = True
        for file, name in RULES:
            e = errors(RUNS / case / file)
            idx = rng.integers(0, e.size, (10000, e.size))
            change = 100 * (e[idx].mean(axis=1) / exact[0] - 1)
            lo, hi = np.percentile(change, [2.5, 97.5])
            point = 100 * (e.mean() / exact[0] - 1)
            sign = lambda v: f"{v:+.1f}"  # noqa: E731
            interval = " (all trajectories identical)" if np.ptp(e) < 1e-6 * abs(e.mean()) else f" [{sign(lo)}, {sign(hi)}]"
            lines.append(f"{label if first else ''} & {name} & {e.size} & {e.mean():.2f} & ${sign(point)}\\%${interval} \\\\")
            print(f"{case:18s} {name:28s} n={e.size}  mean error {e.mean():.3f} mHa (exact {exact[0]:.3f})  change {sign(point)}% [{sign(lo)}, {sign(hi)}]")
            first = False
        if case == SYSTEMS[0][0]:
            lines.append(r"\midrule")
    text = "\n".join([r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", *lines, r"\bottomrule", r"\end{tabular}", ""])
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "tab_window_rho_ci.tex").write_text(text, encoding="utf-8", newline="\n")
    print("wrote", GEN / "tab_window_rho_ci.tex")


if __name__ == "__main__":
    main()
