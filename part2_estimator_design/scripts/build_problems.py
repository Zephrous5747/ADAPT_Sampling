#!/usr/bin/env python3
"""Build (or reload) every Part I problem with the current Part I code and check it.

Part II keeps its own problem cache, built by the current Part I pipeline (with its
validation gate and SCF-instability fix), and compares each build against the
gradients Part I committed.  A case whose absolute gradients disagree beyond 1e-9
is a hard failure: every Part II number is meant to sit on exactly the Part I
objects.

Signed gradients are compared too but only reported.  The SCF fix pins the RHF
solution, not the phases of its orbitals, and a phase flip changes the sign of
every gradient that excites from that orbital.  Part I shows (and tests) that
every statistic it reports -- ``|g_i|``, fragment variances, method totals and the
sign-corrected pairwise variances -- is invariant under such flips, so a signed
difference with matching ``|g_i|`` is a convention, not a discrepancy.

Example::

    python scripts/build_problems.py --cases H4_square_eq_side1p0_HF LiH_R3p0_HF
    python scripts/build_problems.py            # all nine cases
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1_bridge import CASES, DEFAULT_CACHE, check_against_part1, load_problem  # noqa: E402
from outputs import write_json  # noqa: E402

TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path("runs") / "problems_check.json")
    args = parser.parse_args()

    records = []
    failed = []
    for case in args.cases:
        started = time.perf_counter()
        problem = load_problem(case, args.cache)
        problem.parent_fc_groups()
        record = check_against_part1(problem)
        record["seconds"] = round(time.perf_counter() - started, 1)
        records.append(record)
        record["orbital_phase_convention_differs"] = (
            record["max_signed_gradient_difference"] >= TOLERANCE
        )
        ok = (
            record["max_abs_gradient_difference"] < TOLERANCE
            and record["state_energy_difference"] < TOLERANCE
            and record["universal_terms_match"]
            and record["parent_contexts_match"]
        )
        if not ok:
            failed.append(case)
        print(
            f"{case:34s} grad diff {record['max_signed_gradient_difference']:.1e} "
            f"(abs {record['max_abs_gradient_difference']:.1e})  energy diff "
            f"{record['state_energy_difference']:.1e}  terms {record['universal_terms_match']}  "
            f"contexts {record['parent_contexts_match']}  {record['seconds']:.0f}s  "
            f"{'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
    write_json(args.out, {"tolerance": TOLERANCE, "cases": records, "failed": failed})
    if failed:
        raise SystemExit(f"problems disagree with Part I: {failed}")


if __name__ == "__main__":
    main()
