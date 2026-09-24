#!/usr/bin/env python3
"""M3 planning bound against M3 eliminating on each arm's actual precision.

For every case: the common-radius planning bound of Eq. (14), the exact
event-driven cost of :func:`m3_baifcug.run_actual_radii`, their ratio, and what the
second does to the M2/M3 comparison and to the attribution against the no-sharing
baseline.  M2 needs no second column: every M2 arm is resolved to exactly the
common radius, so its bound already uses its actual precision.

Example::

    python scripts/experiment_spillover.py --cases LiH_R3p0_HF H2O_eq_HF --out runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline_noshare
import m1_fcug
import m2_baifcig
import m3_baifcug
from bai import elimination_thresholds
from cases import CASES, get_case
from io_utils import write_csv
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA, allocate_context_shots, epsilon_from_radius, z_from_delta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=sorted(CASES))
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"{'case':32s}{'breadth':>8}{'M3 bound':>13}{'M3 actual':>13}{'ratio':>7}"
          f"{'M2/M3 bnd':>10}{'M2/M3 act':>10}{'interact bnd':>13}{'interact act':>13}",
          flush=True)
    for case in args.cases:
        p = load_or_build(get_case(case), Path(args.cache))
        z = z_from_delta(DEFAULT_DELTA, p.n_generators)
        base = baseline_noshare.run(p).total_shots
        m1 = m1_fcug.run(p).total_shots
        m2 = m2_baifcig.run(p).total_shots
        m3 = m3_baifcug.run(p).total_shots
        act = m3_baifcug.run_actual_radii(p)["total_shots"]
        t0 = elimination_thresholds(p.abs_gradients)[0]
        breadth = allocate_context_shots(p.parent_fragment_sigmas()[list(t0.active), :],
                                         epsilon_from_radius(t0.radius, z)).sum() / m3
        inter_b = (base / m3) / ((base / m1) * (base / m2))
        inter_a = (base / act) / ((base / m1) * (base / m2))
        row = dict(case_id=case, breadth_share=breadth, baseline=base, m1=m1, m2=m2,
                   m3_bound=m3, m3_actual=act, actual_over_bound=act / m3,
                   m2_over_m3_bound=m2 / m3, m2_over_m3_actual=m2 / act,
                   m1_over_m3_actual=m1 / act,
                   interaction_bound=inter_b, interaction_actual=inter_a)
        print(f"{case:32s}{breadth:>8.1%}{m3:>13,.0f}{act:>13,.0f}{act/m3:>7.3f}"
              f"{m2/m3:>10.2f}{m2/act:>10.2f}{inter_b:>13.2f}{inter_a:>13.2f}", flush=True)
        if args.out:
            path = Path(args.out) / case / f"{case}_spillover.csv"
            write_csv(path, list(row), [row])


if __name__ == "__main__":
    main()
