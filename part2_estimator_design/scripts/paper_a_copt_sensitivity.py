#!/usr/bin/env python3
"""Paper A Q7 sensitivity: how much cheaper must parameter optimisation get before selection matters?

Q7's conclusion that reoptimisation dominates rests on a modelled ``C_opt`` (exact optimiser
evaluation counts, parameter-shift gradients, no reuse of shots between evaluations).  The
literature has methods that lower exactly this cost -- Hessian recycling (Ramoa et al.,
arXiv:2401.05172) reports a reduction of the total measurement cost of ADAPT-VQE by an order
of magnitude, and the CEO-ADAPT workflow stacks it with other subroutines.  Rather than
model a specific optimiser, this script divides the modelled ``C_opt`` by a factor ``f`` and
reports, for every selection method with a measured (Phase 4) or planned cost ``C_sel``:

* the share ``C_sel / (C_sel + C_opt / f)``;
* the end-to-end speedup of the method over static M1 at the same ``f``;
* the breakeven ``f* = C_opt / C_sel``: beyond it selection is the larger part of the cost.

``C_opt`` comes from ``runs/paper_a/cost_summary.csv`` (Q7); ``C_sel`` from the median total
of the Phase 4 trajectories of every method present, and the static M1 planning bound of
``cost_summary.csv``.

    python scripts/paper_a_copt_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outputs import write_csv  # noqa: E402

FACTORS = (1, 3, 10, 30, 100)


def selection_costs(runs: Path, case: str, summary: dict) -> dict[str, float]:
    costs = {"M1 static (planning)": float(summary["c_sel_m1"])}
    phase4 = runs / case / "phase4"
    # the original run and every tagged run (``--tag``): <case>_phase4_trajectories.csv, <case>_phase4_<tag>_trajectories.csv
    for path in sorted(phase4.glob(f"{case}_phase4*_trajectories.csv")) if phase4.exists() else []:
        rows = list(csv.DictReader(path.open()))
        for method in dict.fromkeys(r["method"] for r in rows):
            costs[method] = float(np.median([float(r["total_shots"]) for r in rows if r["method"] == method]))
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    summaries = {r["case_id"]: r for r in csv.DictReader((args.runs / "paper_a" / "cost_summary.csv").open())}
    rows = []
    for case, summary in summaries.items():
        costs = selection_costs(args.runs, case, summary)
        reference = costs["M1 static (planning)"]
        for r_label in ("r2", "r4"):
            c_opt = float(summary[f"c_opt_{r_label}"])
            for method, c_sel in costs.items():
                row = {"case": case, "gradient_cost": r_label, "method": method, "c_sel": c_sel, "c_opt": c_opt,
                       "breakeven_f": c_opt / c_sel}
                for f in FACTORS:
                    total = c_sel + c_opt / f
                    row[f"share_f{f}"] = c_sel / total
                    row[f"S_vs_M1static_f{f}"] = (reference + c_opt / f) / total
                rows.append(row)
    write_csv(args.runs / "paper_a" / "copt_sensitivity.csv", list(rows[0]), rows)
    print(f"{'case':30s} {'r':3s} {'method':34s} {'C_sel':>10s} {'f*':>8s}  share at f = " + " ".join(f"{f:>5d}" for f in FACTORS))
    for r in rows:
        if r["gradient_cost"] != "r2":
            continue
        print(f"{r['case']:30s} {r['gradient_cost']:3s} {r['method']:34s} {r['c_sel']:10.3g} {r['breakeven_f']:8.1f}  "
              + "              " + " ".join(f"{r[f'share_f{f}']:5.2f}" for f in FACTORS))


if __name__ == "__main__":
    main()
