#!/usr/bin/env python3
"""Q7: is the finite-shot inflation the schedule, the stopping rule, or noise?

Three quantities are separated here, on the same trials.

``gamma`` is the geometric shrink of the radius schedule.  As it approaches one
the schedule stops overshooting each elimination point, so whatever inflation
survives is not discretisation.

``match_stopping_rule`` replaces "halt when one arm remains" with "halt at the
radius the planning bound assumes", gap/2.  The default rule is weaker, so it can
stop early and make a method look cheaper than its own bound.

Noise is then the residue: the same protocol run with the noise draws zeroed
gives the schedule-and-stopping cost alone, and the ratio of the two is the part
that measurement noise is actually responsible for.

Example::

    python scripts/experiment_q7_decomposition.py --case LiH_R3p0_HF --trials 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m1_fcug
import m2_baifcig
import m3_baifcug
from cases import CASES, get_case
from finite_shot import simulate
from io_utils import write_csv
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem = load_or_build(get_case(args.case), Path(args.cache) if args.cache else None)
    bounds = {
        "m1": m1_fcug.run(problem, delta=args.delta).total_shots,
        "m2": m2_baifcig.run(problem, delta=args.delta).total_shots,
        "m3": m3_baifcug.run(problem, delta=args.delta).total_shots,
    }
    print(f"{args.case}: planning bounds  M1={bounds['m1']:,.0f}  "
          f"M2={bounds['m2']:,.0f}  M3={bounds['m3']:,.0f}  "
          f"M2/M3={bounds['m2']/bounds['m3']:.2f}")

    rows = []
    for matched in (False, True):
        label = "matched" if matched else "one-arm"
        print(f"\n   stopping rule: {label}")
        print(f"{'gamma':>7}{'M2 real':>14}{'/bound':>8}{'M3 real':>14}{'/bound':>8}"
              f"{'M2/M3':>8}{'correct M3':>12}")
        for gamma in args.gammas:
            out = {}
            for method in ("m2", "m3"):
                r = simulate(problem, method, planning_bound=bounds[method],
                             n_trials=args.trials, seed=args.seed, shrink=gamma,
                             delta=args.delta, match_stopping_rule=matched)
                out[method] = r
            row = {
                "case_id": args.case, "stopping_rule": label, "gamma": gamma,
                "m1_bound": bounds["m1"], "m2_bound": bounds["m2"], "m3_bound": bounds["m3"],
                "m2_realised": out["m2"].shots_mean, "m3_realised": out["m3"].shots_mean,
                "m2_inflation": out["m2"].inflation, "m3_inflation": out["m3"].inflation,
                "m2_over_m3_realised": out["m2"].shots_mean / out["m3"].shots_mean,
                "m3_correct": out["m3"].correct_rate,
                "m3_beats_m1": out["m3"].shots_mean < bounds["m1"],
            }
            rows.append(row)
            print(f"{gamma:>7.2f}{row['m2_realised']:>14,.0f}{row['m2_inflation']:>8.2f}"
                  f"{row['m3_realised']:>14,.0f}{row['m3_inflation']:>8.2f}"
                  f"{row['m2_over_m3_realised']:>8.2f}{row['m3_correct']:>12.1%}")

    print(f"\n   bound-level M2/M3 = {bounds['m2']/bounds['m3']:.2f}")
    for label in ("one-arm", "matched"):
        sub = [r for r in rows if r["stopping_rule"] == label]
        finest = max(sub, key=lambda r: r["gamma"])
        print(f"   {label:>8}: at gamma={finest['gamma']}, realised M2/M3 = "
              f"{finest['m2_over_m3_realised']:.2f}, "
              f"M3 inflation = {finest['m3_inflation']:.2f}, "
              f"M3 beats M1: {finest['m3_beats_m1']}")

    if args.out:
        path = Path(args.out) / args.case / f"{args.case}_q7_decomposition.csv"
        write_csv(path, list(rows[0]), rows)
        print(f"   wrote {path}")


if __name__ == "__main__":
    main()
