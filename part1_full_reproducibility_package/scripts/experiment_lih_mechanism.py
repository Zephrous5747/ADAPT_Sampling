#!/usr/bin/env python3
"""Why does M3 realise below its planning bound on LiH while M2 realises above it?

The trial loop of :mod:`finite_shot` is reproduced here with a trace: for every
arm, the radius at which it left the active set, and for every round, the radius,
the active-set size and the cumulative cost.  Each configuration is run twice, with
the noise on and with every draw set to zero, so that what the schedule and the
stopping rule do on their own is separated from what noise does.

The quantity that decides the mechanism is the elimination radius of each arm
relative to its exact value ``Delta_i / 2``: a ratio above one means the arm left
*earlier* than the exact trajectory would have removed it.

Example::

    python scripts/experiment_lih_mechanism.py --case LiH_R3p0_HF --trials 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import m2_baifcig
import m3_baifcug
from bai import deficits
from cases import CASES, get_case
from finite_shot import MINIMUM_RADIUS_FRACTION, MAXIMUM_ROUNDS, _AllocationCache, _eliminate
from io_utils import write_csv
from problem_cache import load_or_build
from shot_models import DEFAULT_DELTA, epsilon_from_radius, z_from_delta


class ZeroRNG:
    def normal(self, loc=0.0, scale=1.0, size=None):
        a = np.asarray(scale, dtype=float)
        return np.zeros(a.shape if size is None else size)

    def standard_normal(self, size=None):
        return np.zeros(() if size is None else size)


def traced_trial(problem, method, z, gamma, rng, sigmas, cache, sigma_sums, matched):
    """One M2 or M3 trial, returning cost and per-arm elimination radii."""
    truth = problem.abs_gradients
    n = problem.n_generators
    shared = method == "m3"
    active = list(range(n))
    radius = float(truth.max())
    floor = radius * MINIMUM_RADIUS_FRACTION
    terminal = problem.top_gap() / 2.0 if matched else None
    left_at = np.full(n, np.nan)
    if shared:
        spent = np.zeros(sigmas.shape[1]); noise = np.zeros_like(sigmas)
    else:
        spent = np.zeros(n); noise = np.zeros(n)
    breadth_cost = None
    for _ in range(MAXIMUM_ROUNDS):
        if terminal is None:
            if len(active) <= 1 or radius < floor:
                break
        elif radius < terminal or radius < floor:
            break
        eps = epsilon_from_radius(radius, z)
        if shared:
            wanted = np.maximum(spent, cache.allocation(active, eps))
            added = wanted - spent; grew = added > 0
            if grew.any():
                noise[:, grew] += rng.normal(
                    0.0, sigmas[:, grew] * np.sqrt(added[grew]), size=(n, int(grew.sum())))
            spent = wanted; used = spent > 0
            est = problem.gradients + (noise[:, used] / spent[used]).sum(axis=1)
            var = (sigmas[:, used] ** 2 / spent[used]).sum(axis=1)
        else:
            wanted = spent.copy()
            wanted[active] = np.maximum(spent[active], (sigma_sums[active] / eps) ** 2)
            added = wanted - spent; grew = added > 0
            if grew.any():
                noise[grew] += rng.normal(0.0, sigma_sums[grew] * np.sqrt(added[grew]))
            spent = wanted; cnt = spent > 0
            est = problem.gradients.copy(); est[cnt] += noise[cnt] / spent[cnt]
            var = np.full(n, np.inf); var[cnt] = sigma_sums[cnt] ** 2 / spent[cnt]
        radii = z * np.sqrt(np.where(np.isfinite(var), var, 0.0))
        before = set(active)
        active = _eliminate(est, radii, active)
        gone = before - set(active)
        for i in gone:
            left_at[i] = radius
        if gone and breadth_cost is None:
            breadth_cost = float(spent.sum())
        radius *= gamma
    return float(spent.sum()), left_at, breadth_cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="LiH_R3p0_HF", choices=sorted(CASES))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.95, 0.99])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = load_or_build(get_case(args.case), Path(args.cache))
    n = p.n_generators
    z = z_from_delta(DEFAULT_DELTA, n)
    sigmas = p.parent_fragment_sigmas()
    cache = _AllocationCache(sigmas)
    groups = p.individual_fc_groups()
    sigma_sums = np.array([s.sum() for s in p.individual_fragment_sigmas(groups)])
    bound = {"m2": m2_baifcig.run(p).total_shots, "m3": m3_baifcug.run(p).total_shots}
    exact_half = deficits(p.abs_gradients) / 2.0
    leader = int(np.argmax(p.abs_gradients))
    nonzero = p.abs_gradients > 1e-6
    print(f"{args.case}: {n} arms, {int(nonzero.sum())} non-zero; "
          f"bounds M2={bound['m2']:,.0f} M3={bound['m3']:,.0f}")
    print(f"\n{'meth':>5}{'gamma':>6}{'stop':>9}{'noise':>7}{'cost/bound':>12}"
          f"{'early zero':>12}{'early nz':>10}{'late nz':>9}{'1st elim r/r0':>15}")
    rows = []
    r0 = exact_half.max()
    for method in ("m2", "m3"):
        for gamma in args.gammas:
            for matched in (False, True):
                for noisy in (False, True):
                    rng = np.random.default_rng(args.seed) if noisy else ZeroRNG()
                    trials = args.trials if noisy else 1
                    costs, ratios_z, ratios_e, ratios_l, first = [], [], [], [], []
                    for _ in range(trials):
                        c, left, _b = traced_trial(p, method, z, gamma, rng, sigmas, cache,
                                                   sigma_sums, matched)
                        costs.append(c)
                        ratio = left / exact_half      # >1 : left earlier than exact
                        m = ~np.isnan(ratio)
                        m[leader] = False
                        zmask = m & ~nonzero
                        nzmask = m & nonzero
                        ratios_z.append(np.mean(ratio[zmask] > 1.0) if zmask.any() else np.nan)
                        ratios_e.append(np.mean(ratio[nzmask] > 1.0) if nzmask.any() else np.nan)
                        ratios_l.append(np.mean(ratio[nzmask] < gamma) if nzmask.any()
                                        else np.nan)
                        first.append(np.nanmax(left) / r0 if np.isfinite(np.nanmax(left)) else np.nan)
                    row = dict(case_id=args.case, method=method, gamma=gamma,
                               stopping="matched" if matched else "one-arm",
                               noise="on" if noisy else "zero",
                               cost_over_bound=float(np.mean(costs)) / bound[method],
                               frac_zero_arms_early=float(np.nanmean(ratios_z)),
                               frac_nonzero_arms_early=float(np.nanmean(ratios_e)),
                               frac_nonzero_arms_late=float(np.nanmean(ratios_l)),
                               first_elim_radius_over_r0=float(np.nanmean(first)))
                    rows.append(row)
                    print(f"{method:>5}{gamma:>6.2f}{row['stopping']:>9}{row['noise']:>7}"
                          f"{row['cost_over_bound']:>12.3f}{row['frac_zero_arms_early']:>12.2f}"
                          f"{row['frac_nonzero_arms_early']:>10.2f}"
                          f"{row['frac_nonzero_arms_late']:>9.2f}"
                          f"{row['first_elim_radius_over_r0']:>15.3f}", flush=True)
    if args.out:
        path = Path(args.out) / args.case / f"{args.case}_mechanism.csv"
        write_csv(path, list(rows[0]), rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
