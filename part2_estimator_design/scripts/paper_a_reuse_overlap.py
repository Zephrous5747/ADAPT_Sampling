#!/usr/bin/env python3
"""Overlap of the energy contexts of H with the gradient support, for the shot-reuse baseline (Paper A, Q8).

How much of the gradient support (and of its coefficient mass) the groups of ``H`` measure, for fully commuting groups
and for QWC cliques measured in a product basis.  A deterministic function of the Hamiltonian and the problem, so it
is computed here and not read back from the trial meta files (which concurrent runs overwrite).  Writes
``runs/paper_a/reuse_overlap.csv``.

    python scripts/paper_a_reuse_overlap.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402
from reuse import ReuseLibrary, hamiltonian_terms  # noqa: E402

CASES = ["H4_square_eq_side1p0_CISD", "LiH_R3p0_HF"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a/reuse_overlap.csv"))
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        problem = part1_bridge.load_problem(case)
        terms = hamiltonian_terms(problem, case)
        for grouping in ("fc", "qwc"):
            rl = ReuseLibrary(problem, terms, "mass", grouping)
            o = rl.overlap(problem)
            rows.append({"case": case, "grouping": grouping, "energy_groups": len(rl.energy), "parent_contexts": rl.n_parent,
                         "covered_fraction": o["covered_fraction"], "covered_mass_fraction": o["covered_mass_fraction"],
                         "two_qubit_mean": float(rl.library.two_qubit_counts().mean())})
            print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
