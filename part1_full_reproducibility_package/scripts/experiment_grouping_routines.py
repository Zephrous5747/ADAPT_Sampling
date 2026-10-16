#!/usr/bin/env python3
"""Routine independence: do the conclusions depend on the FC grouping routine?

Every number in the Part I tables comes from one deterministic first-fit grouping
with one insertion order. First-fit is deterministic given an order, but the order
is a free convention, and group counts are known to be sensitive to it -- the
cached data of an earlier revision reported 9,874 parent contexts for H2O where
the project's routine gives 2,366.

This sweeps the insertion order over three deterministic conventions and a number
of random ones, recomputes the baseline and all three methods for each, and reports
the spread. The physics is untouched; only the grouping changes. What matters is
not whether the absolute costs move -- they will -- but whether the ordering of
the methods and the ratios between them survive.

Example::

    python scripts/experiment_grouping_routines.py --case LiH_R3p0_HF \
        --random-seeds 4 --cache .cache
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline_noshare
import m1_fcug
import m2_baifcig
import m3_baifcug
from cases import CASES, get_case
from io_utils import write_csv
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA

DETERMINISTIC = ("weight", "reverse-weight", "lexicographic")


def evaluate(problem, delta: float, actual: bool = False) -> dict:
    """Baseline and all three methods under the problem's current grouping.

    With ``actual`` set, also M3 eliminating on each arm's actual precision
    (:func:`m3_baifcug.run_actual_radii`) and the M2/M3 ratio it implies.
    """
    baseline = baseline_noshare.run(problem, delta=delta).total_shots
    m1 = m1_fcug.run(problem, delta=delta)
    m2 = m2_baifcig.run(problem, delta=delta).total_shots
    m3 = m3_baifcug.run(problem, delta=delta).total_shots
    extra = {}
    if actual:
        m3a = m3_baifcug.run_actual_radii(problem, delta=delta)["total_shots"]
        extra = {"M3_actual": m3a, "M2_over_M3_actual": m2 / m3a}
    return {**extra, 
        "parent_contexts": len(problem.parent_fc_groups()),
        "NOSHARE": baseline,
        "M1_FCUG": m1.total_shots,
        "M2_BAIFCIG": m2,
        "M3_BAIFCUG": m3,
        "M1_over_M2": m1.total_shots / m2,
        "M2_over_M3": m2 / m3,
        "M1_over_M3": m1.total_shots / m3,
        "NOSHARE_over_M3": baseline / m3,
    }


def _read_rows(path: Path) -> list[dict]:
    import csv as _csv

    if not path.exists():
        return []
    with path.open() as handle:
        return list(_csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--random-seeds", type=int, default=4)
    parser.add_argument("--orderings", nargs="+", default=None,
                        help="deterministic orderings to run (default: all three)")
    parser.add_argument("--resume", action="store_true",
                        help="skip groupings already in the CSV and save after each one; "
                             "an interrupted sweep loses at most the grouping in flight")
    parser.add_argument("--append", action="store_true", help="alias for --resume")
    parser.add_argument("--actual", action="store_true",
                        help="also evaluate M3 on actual precision (Sec. spillover)")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default=None, help="directory for a CSV of the sweep")
    args = parser.parse_args()
    resume = args.resume or args.append
    if resume and not args.out:
        parser.error("--resume needs --out")

    path = Path(args.out) / args.case / f"{args.case}_grouping_routines.csv" if args.out else None
    rows = _read_rows(path) if (resume and path) else []
    done = {(r["ordering"], int(r["seed"])) for r in rows}

    routines = [(name, 0) for name in (args.orderings or DETERMINISTIC)]
    routines += [("random", seed) for seed in range(args.random_seeds)]
    pending = [r for r in routines if r not in done]

    print(f"{args.case}: {len(routines)} groupings, {len(routines) - len(pending)} already done")
    print(f"{'ordering':>16}{'seed':>5}{'contexts':>10}{'M1/M2':>9}{'M2/M3':>9}"
          f"{'M1/M3':>9}{'base/M3':>10}{'seconds':>9}")
    for r in rows:
        print(f"{r['ordering']:>16}{int(r['seed']):>5}{int(float(r['parent_contexts'])):>10,}"
              f"{float(r['M1_over_M2']):9.2f}{float(r['M2_over_M3']):9.2f}"
              f"{float(r['M1_over_M3']):9.2f}{float(r['NOSHARE_over_M3']):10.2f}   (saved)")

    if pending:
        problem = load_or_build(get_case(args.case), Path(args.cache) if args.cache else None)
    for ordering, seed in pending:
        started = time.perf_counter()
        problem.set_fc_routine(ordering, seed)
        row = {"case_id": args.case, "ordering": ordering, "seed": seed,
               **evaluate(problem, args.delta, args.actual)}
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"{ordering:>16}{seed:>5}{row['parent_contexts']:>10,}{row['M1_over_M2']:9.2f}"
              f"{row['M2_over_M3']:9.2f}{row['M1_over_M3']:9.2f}{row['NOSHARE_over_M3']:10.2f}"
              f"{row['runtime_seconds']:9.1f}"
              + (f"   M2/M3 actual {row['M2_over_M3_actual']:.2f}" if args.actual else ""),
              flush=True)
        if path is not None:
            write_csv(path, list(row), rows)

    if not rows:
        return
    print("\nspread across groupings")
    for key in ("parent_contexts", "M1_over_M2", "M2_over_M3", "M1_over_M3", "NOSHARE_over_M3"):
        values = [float(r[key]) for r in rows]
        low, high = min(values), max(values)
        spread = high / low if low > 0 else float("nan")
        print(f"   {key:>16}  min {low:12,.2f}   max {high:12,.2f}   max/min {spread:6.2f}"
              f"   median {statistics.median(values):12,.2f}")

    first = float(rows[0]["M2_over_M3"]) > 1
    consistent = all((float(r["M2_over_M3"]) > 1) == first and float(r["M1_over_M3"]) > 1
                     for r in rows)
    print(f"\n   method ordering identical across every grouping: {consistent}")
    if path is not None:
        print(f"   wrote {path} ({len(rows)} groupings)")


if __name__ == "__main__":
    main()
