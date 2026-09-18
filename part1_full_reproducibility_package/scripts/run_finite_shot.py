#!/usr/bin/env python3
"""Finite-shot study: do the planning bounds survive actual measurement noise?

The main Part I tables are oracle bounds. This driver re-runs each method with
estimates built from a finite number of shots and reports what a real run would
see: how often the largest-gradient generator is actually selected, how often the
selection is within a tolerance of the best, and how many shots it really takes.

Example::

    python scripts/run_finite_shot.py --case H4_square_eq_side1p0_HF --trials 400 \
        --out runs --cache .cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import finite_shot
from cases import CASES, get_case
from io_utils import environment_record, write_csv, write_json
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA

BOUND_KEYS = {
    "noshare": "NOSHARE",
    "m1": "M1_FCUG",
    "m2": "M2_BAIFCIG",
    "m3": "M3_BAIFCUG",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--methods", nargs="+", default=list(finite_shot.METHODS),
                        choices=list(finite_shot.METHODS))
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--tolerance", type=float, default=None,
                        help="good-enough tolerance in absolute gradient; "
                             "default is a quarter of the top gap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    outdir = Path(args.out) / args.case
    bounds = json.loads((outdir / f"{args.case}_summary.json").read_text())
    problem = load_or_build(get_case(args.case), Path(args.cache) if args.cache else None)
    tolerance = args.tolerance if args.tolerance is not None else problem.top_gap() / 4.0

    print(f"{args.case}: {args.trials} trials per method, delta={args.delta}, "
          f"tolerance={tolerance:.6f}")
    print(f"{'method':>9}{'correct':>10}{'within tol':>12}{'bound':>18}"
          f"{'realised mean':>18}{'p90':>18}{'inflation':>11}")

    rows = []
    for method in args.methods:
        key = BOUND_KEYS[method]
        if key not in bounds:
            print(f"{method:>9}  no planning bound in the summary; run the methods first")
            continue
        started = time.perf_counter()
        summary = finite_shot.simulate(
            problem, method,
            planning_bound=bounds[key],
            n_trials=args.trials,
            delta=args.delta,
            tolerance=tolerance,
            seed=args.seed,
        )
        row = summary.as_row()
        row["runtime_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"{method:>9}{summary.correct_rate:10.1%}{summary.within_tolerance_rate:12.1%}"
              f"{summary.planning_bound:18,.0f}{summary.shots_mean:18,.0f}"
              f"{summary.shots_p90:18,.0f}{summary.inflation:11.2f}")

    if rows:
        write_csv(outdir / f"{args.case}_finite_shot.csv", list(rows[0]), rows)
        write_json(outdir / f"{args.case}_finite_shot_summary.json", {
            "case_id": args.case,
            "n_trials": args.trials,
            "delta_family_wise": args.delta,
            "tolerance": tolerance,
            "seed": args.seed,
            "noise_model": (
                "normal-limit estimator fluctuations with exact fragment variances; "
                "shots accumulate across rounds; fluctuations drawn independently per "
                "generator, which is exact for the baseline and M2 and neglects "
                "within-context correlation for M1 and M3"
            ),
            "results": rows,
            "environment": environment_record(),
        })
        print(f"wrote {outdir / f'{args.case}_finite_shot.csv'}")


if __name__ == "__main__":
    main()
