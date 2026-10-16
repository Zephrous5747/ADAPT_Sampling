#!/usr/bin/env python3
"""M3-ext: regroup on the finalist support, with the earlier shots credited.

Two changes to the staged regrouping of the Part I report.

**The earlier shots are credited.**  That experiment charged phase two the full
cost of reaching the final radii and ignored the head start, so its totals were
upper bounds.  The parent-context data give an unbiased estimate of every
survivor at reconstruction variance ``V_par``, the regrouped data an independent
one at ``V_new``, and the inverse-variance combination is unbiased at
``1/V = 1/V_par + 1/V_new``.  Phase two therefore only has to supply the residual
precision, arm by arm:

    1/eps_i^2 = 1/eps^2 - 1/V_par(i),

which is imposed by scaling row ``i`` of the new fragment sigmas by ``1/eps_i``
and solving the ordinary allocation at unit epsilon.  An arm whose parent data
already meet the target drops out of phase two entirely.

**The trigger is the overlap, not the active-set size.**  The report swept switch
points on ``|A|`` and found the best one does not transfer between systems.  The
structural condition is different: once no Pauli product is shared by two active
generators, the universal grouping can only fragment and can no longer share, so
it has stopped paying.  ``--eta`` sets the shared-support fraction at which to
switch.

Example::

    python scripts/experiment_m3ext.py --case H4_square_eq_side1p0_HF --cache .cache
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m2_baifcig
import m3_baifcug
from bai import elimination_thresholds
from cases import CASES, get_case
from gradients import GradientProblem
from io_utils import write_csv
from pauli_fc import greedy_fc_groups
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA, allocate_context_shots, epsilon_from_radius, z_from_delta


def shared_fraction(problem: GradientProblem, active) -> float:
    """Fraction of the active support carried by more than one active generator."""
    counts = Counter()
    for i in active:
        counts.update(problem.commutator_terms[i].keys())
    if not counts:
        return 0.0
    return sum(1 for c in counts.values() if c > 1) / len(counts)


def subset_fragment_sigmas(problem, arms, groups) -> np.ndarray:
    index = {pauli: a for a, group in enumerate(groups) for pauli in group}
    sigmas = np.zeros((len(arms), len(groups)))
    for row, i in enumerate(arms):
        buckets: dict[int, dict[str, float]] = {}
        for pauli, coefficient in problem.commutator_terms[i].items():
            a = index.get(pauli)
            if a is not None:
                buckets.setdefault(a, {})[pauli] = coefficient
        for a, fragment in buckets.items():
            sigmas[row, a] = problem.evaluator.fragment_std(fragment)
    return sigmas


def staged(problem, parent_sigmas, z, switch_index: int, reuse: bool) -> dict:
    """Cost when phase two starts at threshold ``switch_index`` of the trajectory."""
    thresholds = elimination_thresholds(problem.abs_gradients)
    early = thresholds[:switch_index]
    late = thresholds[switch_index:]

    parent_shots = np.zeros(parent_sigmas.shape[1])
    for t in early:
        parent_shots = np.maximum(
            parent_shots,
            allocate_context_shots(parent_sigmas[list(t.active), :],
                                   epsilon_from_radius(t.radius, z)),
        )
    phase1 = float(parent_shots.sum())
    if not late:
        return {"phase1": phase1, "phase2": 0.0, "new_groups": 0, "total": phase1,
                "arms_dropped": 0}

    arms = sorted({i for t in late for i in t.active})
    support = sorted({p for i in arms for p in problem.commutator_terms[i]})
    groups = greedy_fc_groups(support)
    sigmas = subset_fragment_sigmas(problem, arms, groups)
    position = {arm: row for row, arm in enumerate(arms)}

    live = parent_shots > 0
    if live.any():
        var_par = (parent_sigmas[:, live] ** 2 / parent_shots[live]).sum(axis=1)
    else:
        var_par = np.full(problem.n_generators, np.inf)

    new_shots = np.zeros(len(groups))
    dropped = 0
    for t in late:
        epsilon = epsilon_from_radius(t.radius, z)
        rows, targets = [], []
        for i in t.active:
            if reuse:
                residual = 1.0 / epsilon ** 2 - (1.0 / var_par[i] if var_par[i] > 0 else 0.0)
                if residual <= 0.0:          # parent data already meet this target
                    dropped += 1
                    continue
                target = (1.0 / residual) ** 0.5
            else:
                target = epsilon
            rows.append(position[i])
            targets.append(target)
        if rows:
            scaled = sigmas[rows, :] / np.asarray(targets)[:, None]
            new_shots = np.maximum(new_shots, allocate_context_shots(scaled, 1.0))
    phase2 = float(new_shots.sum())
    return {"phase1": phase1, "phase2": phase2, "new_groups": len(groups),
            "total": phase1 + phase2, "arms_dropped": dropped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--eta", type=float, default=0.0,
                        help="shared-support fraction at or below which to regroup")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem = load_or_build(get_case(args.case), Path(args.cache) if args.cache else None)
    z = z_from_delta(args.delta, problem.n_generators)
    parent_sigmas = problem.parent_fragment_sigmas()
    thresholds = elimination_thresholds(problem.abs_gradients)

    m2 = m2_baifcig.run(problem, delta=args.delta).total_shots
    m3 = m3_baifcug.run(problem, delta=args.delta).total_shots
    better = min(m2, m3)
    print(f"{args.case}:  M2 = {m2:,.0f}   M3 = {m3:,.0f}   better = {better:,.0f}")

    print(f"\n{'switch':>7}{'|A|':>6}{'shared':>9}{'discard':>15}{'vs':>7}"
          f"{'reuse':>15}{'vs':>7}{'groups':>8}{'dropped':>9}")
    rows = []
    trigger = None
    for k, t in enumerate(thresholds):
        frac = shared_fraction(problem, t.active)
        if trigger is None and frac <= args.eta:
            trigger = k
        d = staged(problem, parent_sigmas, z, k, reuse=False)
        r = staged(problem, parent_sigmas, z, k, reuse=True)
        mark = "  <-- overlap trigger" if trigger == k else ""
        print(f"{k:>7}{len(t.active):>6}{frac:>9.3f}{d['total']:>15,.0f}"
              f"{d['total']/better:>7.2f}{r['total']:>15,.0f}{r['total']/better:>7.2f}"
              f"{r['new_groups']:>8}{r['arms_dropped']:>9}{mark}")
        rows.append({"case_id": args.case, "switch_index": k, "n_active": len(t.active),
                     "shared_fraction": frac, "m2": m2, "m3": m3,
                     "discard_total": d["total"], "reuse_total": r["total"],
                     "new_groups": r["new_groups"], "arms_dropped": r["arms_dropped"],
                     "is_overlap_trigger": trigger == k})

    best_d = min(rows, key=lambda r: r["discard_total"])
    best_r = min(rows, key=lambda r: r["reuse_total"])
    print(f"\n   best discard  : {best_d['discard_total']:,.0f} at |A|={best_d['n_active']}"
          f"  ({best_d['discard_total']/better:.2f} of the better method)")
    print(f"   best reuse    : {best_r['reuse_total']:,.0f} at |A|={best_r['n_active']}"
          f"  ({best_r['reuse_total']/better:.2f})")
    if trigger is not None:
        row = rows[trigger]
        print(f"   M3-ext (eta={args.eta}): {row['reuse_total']:,.0f} at |A|={row['n_active']}"
              f"  ({row['reuse_total']/better:.2f})"
              f"   -- {row['reuse_total']/best_r['reuse_total']:.3f} of the best switch point")
    else:
        print(f"   M3-ext (eta={args.eta}): never triggers; the active support stays shared")

    if args.out:
        path = Path(args.out) / args.case / f"{args.case}_m3ext.csv"
        write_csv(path, list(rows[0]), rows)
        print(f"   wrote {path}")


if __name__ == "__main__":
    main()
