#!/usr/bin/env python3
"""Paper A, Discussion: what does it cost to certify that ADAPT may stop?

The measured trajectories of Q6--Q8 stop at an oracle: chemical accuracy against the FCI
energy, or ``max |g_i| < 1e-6`` on the exact gradients.  A real run stops on the gradient,
``max_i |g_i| < tau``, and has to decide that from shot data.  Elimination does not help
there.  To certify ``max_i |g_i| < tau`` every gradient needs an upper confidence bound below
``tau``, i.e. a radius ``r_i <= tau - |g_i|``, and no gradient can be dropped before that.

Planning bound.  With the Part I fully commuting contexts and the exact fragment variances,
the cheapest allocation that gives every gradient the standard error ``eps_i = (tau - |g_i|) / z``
is the programme of Part I's M1 with a per-gradient target (scaling row ``i`` of the sigma
matrix by ``1/eps_i`` turns it into the common-target programme).  This is a lower bound for
any sequential procedure that has to learn ``g_i`` as well, like the other planning bounds of
the paper.  For reference, the same states' selection costs are reported at ``rho``-good
precision, ``rho max|g| / 2`` for all gradients.

    python scripts/paper_a_termination_cost.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402
from allocation import allocate  # noqa: E402
from part1_bridge import epsilon_from_radius, z_from_delta  # noqa: E402

# the last saved state of every exact ADAPT trajectory of Phase 2, and the states before it
DEFAULT_CASES = [
    "H4_square_eq_side1p0_ADAPT3", "H4_square_eq_side1p0_ADAPT10",
    "H4_square_stretch_side2p0_ADAPT3", "H4_square_stretch_side2p0_ADAPT10",
    "LiH_R3p0_ADAPT3", "LiH_R3p0_ADAPT6",
    "H2O_eq_ADAPT11", "H2O_eq_ADAPT17",
    "H2O_stretch_ADAPT8", "H2O_stretch_ADAPT18",
]


def certify(problem, sigmas: np.ndarray, tau: float, z: float) -> float:
    """Planning shots that give every gradient the radius ``tau - |g_i|`` (``nan`` if ``tau <= max|g|``)."""
    g = problem.abs_gradients
    if tau <= g.max():
        return float("nan")
    eps = (tau - g) / z
    return float(allocate(sigmas / eps[:, None], 1.0).sum())


TRAJECTORY_METHODS = {"II-A": "II-A data, safe", "M2": "M2 safe", "M1 static": "M1 static"}


def trajectory_totals(path: Path) -> dict[str, dict[str, float]]:
    """Median total selection cost of the measured trajectories, by system and method (Q8)."""
    if not path.exists():
        return {}
    table: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for short, name in TRAJECTORY_METHODS.items():
                if row["method"] == name:
                    table.setdefault(row["case"], {})[short] = float(row["total_shots_median"])
    return table


def system_of(case: str) -> str:
    """``H4_square_eq_side1p0_ADAPT10`` -> ``H4_square_eq_side1p0_HF`` (the start of its trajectory)."""
    base = case.split("_ADAPT")[0]
    return f"{base}_HF"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--taus", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3],
                        help="absolute thresholds on max |g_i|")
    parser.add_argument("--relative", nargs="+", type=float, default=[1.25, 2.0],
                        help="thresholds tau = kappa max|g| (only where max|g| > 1e-4)")
    parser.add_argument("--rho", type=float, default=0.1, help="rho of the reference selection")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a/termination_cost.csv"))
    args = parser.parse_args()

    rows = []
    totals = trajectory_totals(Path("runs/paper_a/sota_trajectories.csv"))
    for case in args.cases:
        problem = part1_bridge.load_problem(case)
        sigmas = problem.parent_fragment_sigmas()
        z = z_from_delta(args.delta, problem.n_generators)
        peak = float(problem.abs_gradients.max())
        reference = (float(allocate(sigmas, epsilon_from_radius(args.rho * peak / 2.0, z)).sum())
                     if peak > 1e-4 else float("nan"))
        trajectory = totals.get(system_of(case), {})
        thresholds = [("absolute", t) for t in args.taus]
        if peak > 1e-4:
            thresholds += [("relative", k * peak) for k in args.relative]
        for kind, tau in thresholds:
            cost = certify(problem, sigmas, tau, z)
            rows.append({"case": case, "n_generators": problem.n_generators, "max_abs_gradient": peak,
                         "threshold_kind": kind, "tau": tau, "tau_over_max": tau / peak if peak > 0 else float("inf"),
                         "certify_shots": cost, "rho_good_selection_shots": reference,
                         "ratio": cost / reference if reference == reference and cost == cost else float("nan"),
                         **{f"trajectory_{k}": v for k, v in trajectory.items()},
                         "over_trajectory_II-A": cost / trajectory["II-A"] if "II-A" in trajectory and cost == cost else float("nan")})
            print(f"{case:34s} max|g| {peak:9.2e}  tau {tau:9.2e} ({kind:8s})  certify {cost:12,.0f}  "
                  f"one selection (M1, rho={args.rho:g}) {reference:12,.0f}  "
                  f"whole trajectory II-A {trajectory.get('II-A', float('nan')):12,.0f}  "
                  f"ratio {rows[-1]['over_trajectory_II-A']:7.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
