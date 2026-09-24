#!/usr/bin/env python3
"""Finite-shot rows for M2 and M3 at a given schedule, bounds computed in place.

``run_finite_shot.py`` reads its planning bounds from a methods summary; this
computes them directly, so it can fill rows for cases whose summary is not at hand
(the two H2O CISD rows, whose bounds are the largest in the set).

Example::

    python scripts/experiment_finite_shot_rows.py --case H2O_eq_CISD --gammas 0.5 0.95
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m2_baifcig
import m3_baifcug
from cases import CASES, get_case
from finite_shot import simulate
from io_utils import write_csv
from problem_cache import load_or_build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    p = load_or_build(get_case(args.case), Path(args.cache))
    bounds = {"m2": m2_baifcig.run(p).total_shots, "m3": m3_baifcug.run(p).total_shots}
    print(f"{args.case}: bounds M2={bounds['m2']:,.0f} M3={bounds['m3']:,.0f}", flush=True)
    path = Path(args.out) / args.case / f"{args.case}_finite_shot_rows.csv"
    rows = []
    if path.exists():                      # resume: keep and skip schedules already done
        import csv
        rows = list(csv.DictReader(path.open()))
    done = {float(r["gamma"]) for r in rows}
    for gamma in args.gammas:
        if gamma in done:
            print(f"  gamma={gamma:.2f} already saved", flush=True)
            continue
        out = {}
        for m in ("m2", "m3"):
            r = simulate(p, m, planning_bound=bounds[m], n_trials=args.trials,
                         seed=args.seed, shrink=gamma)
            out[m] = r
            row = r.as_row() | {"gamma": gamma, "method": m}
            rows.append(row)
            print(f"  gamma={gamma:.2f} {m}: realised {r.shots_mean:,.0f} +- {r.shots_sem:,.0f}"
                  f"  inflation {r.inflation:.2f} +- {r.inflation_sem:.2f}"
                  f"  correct {r.correct_rate:.1%}", flush=True)
        print(f"  gamma={gamma:.2f} realised M2/M3 = {out['m2'].shots_mean / out['m3'].shots_mean:.2f}",
              flush=True)
        write_csv(path, list(rows[0]), rows)


if __name__ == "__main__":
    main()
