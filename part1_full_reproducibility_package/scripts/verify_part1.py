#!/usr/bin/env python3
"""Verify that the Part I results are robust, trustworthy and reproducible.

This checks the shipped ``runs/`` directory and the live code together.  It is
deliberately separate from the unit tests: the tests pin the implementation,
whereas this asks whether the *published numbers* still stand.

Five families of check:

Integrity      every published total is the sum of the per-context or per-arm
               file that claims to explain it, so no table entry is orphaned.
Physics        each run's validation gate reproduced the PySCF RHF and FCI
               energies, and the fragment decomposition sums back to the gradient.
Feasibility    the allocation actually meets its confidence target.
Invariants     properties the methods guarantee by construction: M3 never costs
               more than M1, and all three select the same generator.
Robustness     conclusions do not move under the free conventions -- the ratios
               are invariant to the confidence level, and the exact-limit
               schedule is the limit a geometric schedule converges onto.

Usage::

    python scripts/verify_part1.py --runs runs
    python scripts/verify_part1.py --runs runs --live-case H4_square_eq_side1p0_HF
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ENERGY_TOLERANCE = 1e-8
METHOD_KEYS = ("M1_FCUG", "M2_BAIFCIG", "M3_BAIFCUG")


class Report:
    """Collects pass/fail lines and decides the exit status."""

    def __init__(self) -> None:
        self.failures = 0
        self.checks = 0

    def record(self, ok: bool, family: str, message: str) -> None:
        self.checks += 1
        if not ok:
            self.failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {family:<12} {message}")

    def section(self, title: str) -> None:
        print(f"\n{title}")


def _sum_column(path: Path, column: str) -> float:
    with path.open() as handle:
        return sum(float(row[column]) for row in csv.DictReader(handle))


def check_published_runs(runs: Path, report: Report) -> None:
    cases = sorted(d for d in runs.iterdir() if d.is_dir())
    if not cases:
        report.record(False, "integrity", f"no case directories under {runs}")
        return

    report.section(f"Published runs ({len(cases)} cases under {runs})")
    for case in cases:
        name = case.name
        summary = json.loads((case / f"{name}_summary.json").read_text())

        # --- integrity: totals equal the files that explain them ---------
        m1 = math.ceil(_sum_column(case / f"{name}_M1_FCUG_contexts.csv", "shots"))
        m3 = math.ceil(_sum_column(case / f"{name}_M3_BAIFCUG_contexts.csv", "shots"))
        m2 = math.ceil(_sum_column(case / f"{name}_M2_BAIFCIG_arms.csv", "shots"))
        totals_match = (
            abs(m1 - summary["M1_FCUG"]) <= 1
            and abs(m2 - summary["M2_BAIFCIG"]) <= 1
            and abs(m3 - summary["M3_BAIFCUG"]) <= 1
        )
        report.record(totals_match, "integrity", f"{name}: totals reconstruct from saved files")

        # --- physics: the validation gate passed --------------------------
        validation = summary.get("validation", {})
        gate = (
            validation.get("hf_energy_error", 1.0) < ENERGY_TOLERANCE
            and validation.get("fci_energy_error", 1.0) < ENERGY_TOLERANCE
        )
        report.record(gate, "physics", f"{name}: RHF and FCI reproduced to < {ENERGY_TOLERANCE:g} Ha")

        # --- feasibility: M1 meets its stated confidence target -----------
        m1_summary = json.loads((case / f"{name}_M1_FCUG_summary.json").read_text())
        report.record(
            bool(m1_summary["constraint_satisfied"]),
            "feasibility",
            f"{name}: M1 allocation reaches radius {m1_summary['confidence_radius']:.3e}",
        )

        # --- invariants ---------------------------------------------------
        if "NOSHARE" in summary:
            report.record(
                summary["M2_BAIFCIG"] <= summary["NOSHARE"],
                "invariant",
                f"{name}: M2 <= no-sharing baseline "
                f"({summary['M2_BAIFCIG']:,} <= {summary['NOSHARE']:,})",
            )
            report.record(
                summary["M3_BAIFCUG"] <= summary["NOSHARE"],
                "diagnostic",
                f"{name}: M3 below the no-sharing baseline "
                f"(x{summary['NOSHARE'] / summary['M3_BAIFCUG']:.1f})",
            )
        report.record(
            summary["M3_BAIFCUG"] <= summary["M1_FCUG"],
            "invariant",
            f"{name}: M3 <= M1 ({summary['M3_BAIFCUG']:,} <= {summary['M1_FCUG']:,})",
        )
        winners = {tuple(summary.get("M2_selected", [])), tuple(summary.get("M3_selected", []))}
        report.record(
            winners == {(summary["top_generator"],)},
            "invariant",
            f"{name}: M2 and M3 both select {summary['top_generator']}",
        )


def check_live_code(case_id: str, cache: Path | None, report: Report) -> None:
    from cases import get_case
    from problem_cache import load_or_build
    import m1_fcug
    import m2_baifcig
    import m3_baifcug

    report.section(f"Live code ({case_id})")
    problem = load_or_build(get_case(case_id), cache)

    # --- physics: fragments sum back to the gradient ---------------------
    groups = problem.parent_fc_groups()
    index = {p: a for a, g in enumerate(groups) for p in g}
    worst = 0.0
    for i in problem.ranking()[:5]:
        buckets: dict[int, dict[str, float]] = {}
        for pauli, coefficient in problem.commutator_terms[i].items():
            buckets.setdefault(index[pauli], {})[pauli] = coefficient
        rebuilt = sum(problem.evaluator.fragment_mean_std(f)[0] for f in buckets.values())
        worst = max(worst, abs(rebuilt - problem.gradients[i]))
    report.record(worst < 1e-12, "physics", f"fragments sum back to the gradient (worst {worst:.1e})")

    # --- reproducibility: the pipeline is deterministic ------------------
    first = [m1_fcug.run(problem).total_shots, m2_baifcig.run(problem).total_shots,
             m3_baifcug.run(problem).total_shots]
    second = [m1_fcug.run(problem).total_shots, m2_baifcig.run(problem).total_shots,
              m3_baifcug.run(problem).total_shots]
    report.record(first == second, "reproducible", f"repeated runs agree exactly {tuple(first)}")

    # --- robustness: ratios are invariant to the confidence level --------
    def ratios(delta: float) -> tuple[float, float]:
        a = m1_fcug.run(problem, delta=delta).total_shots
        b = m2_baifcig.run(problem, delta=delta).total_shots
        c = m3_baifcug.run(problem, delta=delta).total_shots
        return a / b, b / c

    base = ratios(0.05)
    stable = all(
        abs(r - b) / b < 1e-3 for delta in (0.2, 0.01, 0.001) for r, b in zip(ratios(delta), base)
    )
    report.record(stable, "robustness", f"M1/M2 and M2/M3 invariant to delta (base {base[0]:.3f}, {base[1]:.3f})")

    # --- robustness: the exact schedule is the geometric limit -----------
    for module in (m2_baifcig, m3_baifcug):
        exact = module.run(problem).total_shots
        coarse = module.run(problem, schedule="geometric", shrink=0.5).total_shots
        fine = module.run(problem, schedule="geometric", shrink=0.97).total_shots
        converged = exact <= fine <= coarse and abs(fine - exact) / exact < 0.05
        report.record(
            converged,
            "robustness",
            f"{module.METHOD_CODE}: geometric schedule converges onto the exact limit "
            f"({coarse:,} -> {fine:,} -> {exact:,})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs", help="directory of published runs")
    parser.add_argument("--live-case", default="H4_square_eq_side1p0_HF")
    parser.add_argument("--cache", default=None, help="gradient-problem cache directory")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    report = Report()
    runs = Path(args.runs)
    if runs.exists():
        check_published_runs(runs, report)
    else:
        report.record(False, "integrity", f"{runs} does not exist")
    if not args.skip_live:
        check_live_code(args.live_case, Path(args.cache) if args.cache else None, report)

    print(f"\n{report.checks - report.failures}/{report.checks} checks passed")
    if report.failures:
        print("VERIFICATION FAILED")
        raise SystemExit(1)
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
