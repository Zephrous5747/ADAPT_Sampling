#!/usr/bin/env python3
"""Does the confidence convention control the selection-error rate, beyond one case?

Two studies, each on M2 and M3 at the default schedule so the numbers are
comparable with the H4 side 1.0 HF table of the report.

* **delta sweep.**  The family-wise failure probability is swept over nearly its
  whole range under the Bonferroni convention, and the realised selection-error
  rate is measured with a Wilson 95% interval.  If delta controlled the rate, the
  rate would track it.
* **Good-enough stopping.**  With tolerance tau = gap/4 the run halts once every
  survivor is certified within tau of the best.  Reported: the shot saving against
  exact selection and the rate at which the returned generator really is within tau.

Example::

    python scripts/experiment_calibration.py --case LiH_R3p0_HF --trials 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m2_baifcig
import m3_baifcug
from cases import CASES, get_case
from finite_shot import simulate, wilson_interval
from io_utils import write_csv
from problem_cache import load_or_build
from shot_models import z_from_delta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.5, 0.9, 0.99])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = load_or_build(get_case(args.case), Path(args.cache))
    bounds = {"m2": m2_baifcig.run(p).total_shots, "m3": m3_baifcug.run(p).total_shots}
    rows = []
    print(f"{args.case}: {args.trials} trials per entry\n")
    print(f"{'study':>12}{'delta':>7}{'z':>6}{'meth':>5}{'error':>8}{'95% CI':>16}"
          f"{'shots':>14}{'within tau':>12}")
    for delta in args.deltas:
        for m in ("m2", "m3"):
            r = simulate(p, m, planning_bound=bounds[m], n_trials=args.trials,
                         seed=args.seed, delta=delta)
            wrong = round((1 - r.correct_rate) * args.trials)
            lo, hi = wilson_interval(wrong, args.trials)
            row = dict(case_id=args.case, study="delta sweep", delta=delta,
                       z=z_from_delta(delta, p.n_generators), method=m,
                       error_rate=1 - r.correct_rate, error_low=lo, error_high=hi,
                       shots_mean=r.shots_mean, within_tolerance=r.within_tolerance_rate)
            rows.append(row)
            print(f"{'delta sweep':>12}{delta:>7.2f}{row['z']:>6.2f}{m:>5}"
                  f"{row['error_rate']:>8.1%}{f'[{lo:.3f},{hi:.3f}]':>16}"
                  f"{row['shots_mean']:>14,.0f}{'':>12}", flush=True)

    tau = p.top_gap() / 4.0
    for m in ("m2", "m3"):
        exact = next(r for r in rows if r["method"] == m and r["delta"] == 0.05)
        r = simulate(p, m, planning_bound=bounds[m], n_trials=args.trials, seed=args.seed,
                     tolerance=tau, stop_at_tolerance=True)
        row = dict(case_id=args.case, study="good-enough", delta=0.05,
                   z=z_from_delta(0.05, p.n_generators), method=m,
                   error_rate=1 - r.correct_rate, error_low=float("nan"),
                   error_high=float("nan"), shots_mean=r.shots_mean,
                   within_tolerance=r.within_tolerance_rate,
                   saving=1 - r.shots_mean / exact["shots_mean"])
        rows.append(row)
        print(f"{'good-enough':>12}{0.05:>7.2f}{row['z']:>6.2f}{m:>5}"
              f"{row['error_rate']:>8.1%}{'':>16}{row['shots_mean']:>14,.0f}"
              f"{row['within_tolerance']:>12.1%}   saving {row['saving']:.0%}", flush=True)

    if args.out:
        path = Path(args.out) / args.case / f"{args.case}_calibration_sweep.csv"
        keys = sorted({k for r in rows for k in r})
        write_csv(path, keys, [{k: r.get(k, "") for k in keys} for r in rows])
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
