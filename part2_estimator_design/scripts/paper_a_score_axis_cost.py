#!/usr/bin/env python3
"""Paper A, Discussion: what would an energy-based score cost to measure?

The paper fixes the gradient criterion.  Energy-based scores (greedy gradient-free ADAPT,
ExcitationSolve, fermionic Rotoselect) rank a candidate by the energy lowering of a one-parameter
move, which needs the one-dimensional energy landscape of every candidate.  The energy of an
excitation generator is a low-order trigonometric polynomial in the angle (three unknowns for
a single-frequency landscape, five for two frequencies), so each candidate needs that many
energy evaluations.  The gradient, in contrast, is the expectation of one observable on the
*one* state that all candidates share, which is what lets all ``K`` gradients share shots.  Each
energy evaluation is a different state (a different circuit) per candidate, so shots cannot be shared across
candidates, and the cost grows linearly in ``K``.

This script puts a number on that structural factor, from stored quantities only.  One energy
evaluation to standard error ``eps`` costs ``M_E = (sum_g sigma_g)^2 / eps^2`` context-shots with the
fully commuting groups of ``H`` (the ``C_opt`` model of Q7), evaluated on the state itself; the
landscape of a candidate is taken at its own state, which differs from the current one by the
small rotation, so ``sigma_g`` is the current state's.  The energy-score cost is
``K n_E M_E(eps)``; the number of landscape points ``n_E`` and the precision ``eps`` are inputs.
The comparison is with the stored mean cost of the learned design (II-A) and of the strongest
baseline (``M1`` seq or ``M2``) on the same state.  Besides the two fixed precisions given, one more
is added per state, ``gap_matched``: ranking the candidates by energy lowering needs a standard error
below the difference between the lowerings of the two best candidates, ``(g_1^2 - g_2^2) / (2h)``
with an assumed curvature ``h = 1`` Ha of a normalised excitation generator, divided by ``2 z``.  At
a small gradient this is orders of magnitude below 1 mHa.  This is a cost-side statement only: an
energy score can choose better generators and so need fewer ADAPT iterations (Rossi et al.), which
is not modelled.

    python scripts/paper_a_score_axis_cost.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402
from part1_bridge import z_from_delta  # noqa: E402
from reuse import energy_groups, hamiltonian_terms  # noqa: E402

HESSIAN = 1.0  # Ha, assumed curvature of the energy along a normalised excitation generator

DEFAULT_CASES = ["H4_square_eq_side1p0_CISD", "LiH_R3p0_HF", "LiH_R3p0_ADAPT3", "LiH_R3p0_ADAPT5",
                 "H2O_eq_CISD", "H2O_stretch_CISD", "H2O_eq_ADAPT11", "H2O_stretch_ADAPT8"]


def energy_cost(problem, terms: dict[str, float], epsilon: float) -> tuple[float, int]:
    """Context-shots of one energy evaluation to standard error ``epsilon`` (FC groups of H, optimal allocation)."""
    groups = energy_groups(terms, "fc")
    sigma = sum(problem.evaluator.fragment_std({p: terms[p] for p in group}) for group in groups)
    return float(sigma ** 2 / epsilon ** 2), len(groups)


def stored_costs(path: Path) -> dict[str, dict[str, float]]:
    """Mean cost per state: the learned design and the cheapest of M1 / M2 (exact identification, oracle start)."""
    table: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if float(row["rho"]) != 0 or row["setting"] != "oracle start":
                continue
            entry = table.setdefault(row["case"], {})
            cost = float(row["shots_mean"])
            if row["method"] == "II-A data":
                entry["II-A"] = cost
            elif row["method"].startswith(("M1", "M2")):
                entry["baseline"] = min(entry.get("baseline", cost), cost)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-3, 1e-4], help="standard error (Ha) of each energy")
    parser.add_argument("--landscape-points", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--costs", type=Path, default=Path("runs/paper_a/sota_fixed_state.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a/score_axis_cost.csv"))
    args = parser.parse_args()

    stored = stored_costs(args.costs)
    rows = []
    for case in args.cases:
        problem = part1_bridge.load_problem(case)
        try:
            terms = hamiltonian_terms(problem, case)
        except ValueError as error:
            print(f"{case}: skipped ({error})")
            continue
        known = stored.get(case, {})
        # the precision an energy score needs: the energy lowerings of the two best candidates, g^2 / (2 h) with a curvature of
        # h = 1 Ha (the order of the diagonal Hessian of a normalised excitation generator), must be resolved to one z.
        top = sorted(problem.abs_gradients, reverse=True)[:2]
        z = z_from_delta(0.05, problem.n_generators)
        matched = max((top[0] ** 2 - top[1] ** 2) / (2.0 * HESSIAN) / (2.0 * z), 1e-12) if len(top) == 2 else float("nan")
        for epsilon in list(args.epsilons) + [matched]:
            m_e, n_groups = energy_cost(problem, terms, epsilon)
            for n_e in args.landscape_points:
                total = problem.n_generators * n_e * m_e
                row = {"case": case, "n_generators": problem.n_generators, "energy_groups": n_groups,
                       "epsilon": epsilon, "gap_matched": bool(epsilon == matched), "landscape_points": n_e, "energy_cost": m_e, "energy_score_cost": total,
                       "II-A": known.get("II-A", float("nan")), "strongest_baseline": known.get("baseline", float("nan"))}
                row["over_II-A"] = total / row["II-A"] if row["II-A"] == row["II-A"] else float("nan")
                row["over_baseline"] = total / row["strongest_baseline"] if row["strongest_baseline"] == row["strongest_baseline"] else float("nan")
                rows.append(row)
                print(f"{case:22s} K={row['n_generators']:4d} eps={epsilon:7.0e} n_E={n_e}  M_E={m_e:12,.0f}  energy scores {total:14,.0f}  "
                      f"II-A {row['II-A']:12,.0f} (x{row['over_II-A']:9,.0f})  strongest baseline {row['strongest_baseline']:12,.0f} "
                      f"(x{row['over_baseline']:9,.0f})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
