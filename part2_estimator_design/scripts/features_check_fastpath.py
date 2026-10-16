#!/usr/bin/env python3
"""Audit of ``src/fastpath.py`` (the colleague's faster statistics for ``learning.LearnedM3``).

The module claims to give the same selections, shots and rounds as the original code up to floating-point
round-off.  Here the claim is tested directly: for a few states of a case (the HF state and states of the exact
ADAPT trajectory) and a few random streams, the same selection is run once with the original methods and once
with the fast ones, and the outcomes are compared (generator, shots, rounds, shortfall, stopping reason).

    python scripts/features_check_fastpath.py --case H4_square_eq_side1p0_HF --steps 0 3 6 --seeds 3

Exit code 0 only if every selection agrees exactly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
import fastpath  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, problem_at  # noqa: E402

from phase4_adapt_selection import METHODS, Phase4  # noqa: E402


def outcome_key(o) -> tuple:
    return (int(o.selected), float(o.shots), int(o.rounds), float(o.extra["shortfall"]), o.extra.get("stopped_rho"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 3],
                        help="exact-ADAPT iteration whose state is used (0: the HF state)")
    parser.add_argument("--methods", nargs="+", default=["II-A data, safe", "II-0 safe"])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.1)
    args = parser.parse_args()

    ph = Phase4(args.case, "mass", CHEMICAL_ACCURACY, args.rho)
    system = ph.system
    exact = ph.exact
    cases = []
    for k in args.steps:
        if k >= len(exact):
            print(f"step {k}: beyond the exact trajectory ({len(exact) - 1} steps), skipped")
            continue
        step = exact[k]
        psi = system.state(step.ops, step.params)
        g = system.gradients(psi)
        full = system.full_state(psi)
        problem = problem_at(system.problem, full, f"{args.case}_check{k}", energy=system.energy(psi),
                             sector_gradients=g)
        cases.append((k, problem, full))

    results = {}
    timing = {"original": 0.0, "fast": 0.0}
    for mode in ("original", "fast"):
        if mode == "fast":
            fastpath.install()
        for method in args.methods:
            spec = METHODS[method]
            for k, problem, full in cases:
                for s in range(args.seeds):
                    rng = np.random.default_rng(np.random.SeedSequence(7).spawn(args.seeds)[s])
                    t0 = time.perf_counter()
                    out = ph._select_with(spec, problem, full, rng)
                    timing[mode] += time.perf_counter() - t0
                    results[(mode, method, k, s)] = outcome_key(out)
    fastpath.uninstall()

    bad = 0
    print(f"{args.case}: {len(cases)} states x {len(args.methods)} methods x {args.seeds} seeds; "
          f"time original {timing['original']:.0f}s, fast {timing['fast']:.0f}s")
    for method in args.methods:
        for k, _, _ in cases:
            for s in range(args.seeds):
                a, b = results[("original", method, k, s)], results[("fast", method, k, s)]
                same = a[0] == b[0] and a[2] == b[2] and a[4] == b[4] and np.isclose(a[1], b[1], rtol=1e-9, atol=0) \
                    and np.isclose(a[3], b[3], rtol=0, atol=1e-12)
                bad += not same
                print(f"  {method:16s} step {k:2d} seed {s}: arm {a[0]:3d}/{b[0]:3d} shots {a[1]:.6e}/{b[1]:.6e} "
                      f"rounds {a[2]}/{b[2]} shortfall {a[3]:.3e}/{b[3]:.3e} "
                      f"{'OK' if same else 'DIFFERENT'}")
    print("fastpath agrees with the original on every selection" if not bad else f"{bad} selections differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
