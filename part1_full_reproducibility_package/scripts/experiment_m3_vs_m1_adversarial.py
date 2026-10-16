#!/usr/bin/env python3
"""Adversarial search for a counterexample to M3 <= M1 on the planning bounds.

M1 resolves every arm to gap/2 in one allocation.  M3 solves one allocation per
breakpoint of the exact trajectory, each over a subset of arms at a looser radius,
and is charged the per-context *maximum* over breakpoints (Eq. 14).  Each
breakpoint's allocation is cheaper than M1's in total, but the maxima are taken
context by context, and two breakpoints can load different contexts.  Whether the
sum of maxima can exceed M1 is therefore not settled by the obvious argument.

The search draws small random problems -- sparse fragment standard deviations,
random gradients -- and hill-climbs on M3/M1.  The confidence factor cancels from
the ratio and is set to one.

Example::

    python scripts/experiment_m3_vs_m1_adversarial.py --starts 400 --steps 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bai import elimination_thresholds
from shot_models import allocate_context_shots


def m1_cost(absg: np.ndarray, sig: np.ndarray) -> float:
    top = np.sort(absg)[::-1]
    return float(allocate_context_shots(sig, (top[0] - top[1]) / 2.0).sum())


def m3_cost(absg: np.ndarray, sig: np.ndarray) -> float:
    shots = np.zeros(sig.shape[1])
    for t in elimination_thresholds(absg):
        shots = np.maximum(shots, allocate_context_shots(sig[list(t.active), :], t.radius))
    return float(shots.sum())


def ratio(absg, sig) -> float:
    top = np.sort(absg)[::-1]
    if top[0] - top[1] < 1e-6 or not np.all(sig.sum(axis=1) > 0):
        return 0.0
    return m3_cost(absg, sig) / m1_cost(absg, sig)


def random_problem(rng, n, k, compressed=False):
    sig = np.exp(rng.normal(0.0, 1.5, size=(n, k))) * (rng.random((n, k)) < 0.6)
    for i in range(n):                     # every arm needs at least one fragment
        if not sig[i].any():
            sig[i, rng.integers(k)] = 1.0
    if compressed:
        # leader at one, every other arm just below the runner-up: max deficit / gap
        # close to one, the regime in which the first and last breakpoints nearly
        # coincide and can load different contexts
        gap = 0.1
        absg = np.concatenate([[1.0], 1.0 - gap * (1.0 + 0.5 * rng.random(n - 1))])
    else:
        absg = rng.random(n)
    return absg, sig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compressed", action="store_true",
                    help="start from fields with max deficit / gap near one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    best = (0.0, None, None)
    above = 0
    for _ in range(args.starts):
        n, k = int(rng.integers(3, 7)), int(rng.integers(2, 6))
        absg, sig = random_problem(rng, n, k, args.compressed)
        cur = ratio(absg, sig)
        for _ in range(args.steps):
            g2 = np.clip(absg * np.exp(rng.normal(0, 0.15, n)), 1e-3, None)
            s2 = sig * np.exp(rng.normal(0, 0.3, sig.shape))
            if rng.random() < 0.1:              # occasionally toggle a fragment
                i, a = rng.integers(n), rng.integers(k)
                s2[i, a] = 0.0 if s2[i, a] > 0 else float(np.exp(rng.normal()))
            r2 = ratio(g2, s2)
            if r2 > cur:
                absg, sig, cur = g2, s2, r2
        above += cur > 1.0 + 1e-9
        if cur > best[0]:
            best = (cur, absg.copy(), sig.copy())

    r, absg, sig = best
    top = np.sort(absg)[::-1]
    spread = (top[0] - top[-1]) / (top[0] - top[1])
    print(f"starts with a final ratio above one: {above} of {args.starts}")
    print(f"worst M3/M1 found: {r:.4f}  ({sig.shape[0]} arms, {sig.shape[1]} contexts, "
          f"max deficit / gap = {spread:.3f})")
    np.set_printoptions(precision=4, suppress=True)
    print("abs gradients:", absg)
    print("fragment sigmas:\n", sig)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.out, f"m3_vs_m1_adversarial{'_compressed' if args.compressed else ''}.json").write_text(json.dumps({
            "worst_ratio": r, "starts_above_one": above, "starts": args.starts,
            "abs_gradients": absg.tolist(), "sigmas": sig.tolist(),
            "max_deficit_over_gap": spread}, indent=1))


if __name__ == "__main__":
    main()
