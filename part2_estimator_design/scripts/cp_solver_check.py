#!/usr/bin/env python3
"""Validation of the alternating design solver (src/cp_order.CPSelector.alternating) against the conic program (CVXPY/Clarabel).

The alternating solver replaces the conic program where that cannot be built (LiH: 365,000 coefficients), so its distance from the
conic optimum must be measured where both run: all arms of the small states, and the first ``--subset`` arms (by exact gradient) of
LiH, whose program is then small enough.  ``--no-conic`` runs the alternating solver alone on all arms to show its convergence with the
number of sweeps (the trace is the predicted objective after the reference reconstruction and after each sweep).

Objectives: mode A2 the largest predicted variance at a budget of 128 shots per context, mode B the total shots needed for a standard
error of ``--epsilon``.  Both are evaluated under the design covariance (what the program optimizes); in A2 the true largest variance
(evaluator only) at the rounded counts is recorded as well.

    python scripts/cp_solver_check.py --state H4_a1_m1 --reps 3 --out runs/cp_solver
    python scripts/cp_solver_check.py --state LiH_r30_m3 --subset 6 --seconds 900 --out runs/cp_solver
    python scripts/cp_solver_check.py --state LiH_r30_m3 --no-conic --sweeps 10 --out runs/cp_solver
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
from cp_states import arm_inputs, build_state, design_covariance, make_record, nested_counts  # noqa: E402
from note_protocol import integer_counts  # noqa: E402


def rng_of(*key) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, ("CP-SOLVER",) + key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True)
    parser.add_argument("--subset", type=int, default=0, help="use only the first k arms by |exact gradient| (0: all arms)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=900.0, help="time limit of one conic solve")
    parser.add_argument("--epsilon", type=float, default=5e-3)
    parser.add_argument("--no-conic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/cp_solver"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sel = build_state(args.state, solve_seconds=args.seconds)
    sel.max_coords = 10 ** 9  # the conic program is attempted whatever its size (the guard exists for the production runs)
    arms = sorted(sel.candidates, key=lambda j: -abs(sel.truth[j]))
    if args.subset:
        arms = arms[:args.subset]
    L = sel.n_contexts
    budget = 128.0 * L
    coords = sum(sel._split_problems()[j].n_coordinates for j in arms)
    print(f"{args.state}: {len(arms)} arms, {coords} coefficients, {L} contexts", flush=True)
    path = args.out / f"{args.state}{'_s%d' % args.subset if args.subset else ''}{'_alt' if args.no_conic else ''}.jsonl"
    with open(path, "a") as handle:
        for rep in range(args.rep_start, args.rep_start + args.reps):
            record = make_record(sel, rng_of("pre", args.state, rep))
            counts = nested_counts(sel, rng_of("cur", args.state, rep), prefixes=(32,))[32]
            for source, params in (("NEUTRAL", None), ("CLASSICAL", {"const": 128.0})):
                prior, tau, _ = arm_inputs(sel, source, record, params)
                cov = design_covariance(sel, counts, prior, tau)
                for mode in ("A2", "B"):
                    kw = {"budget": budget} if mode == "A2" else {"epsilon": args.epsilon}
                    row = {"state": args.state, "rep": rep, "source": source, "mode": mode, "arms": len(arms), "coords": coords}

                    def evaluate(designs, plan):
                        pred = sel.predicted(designs, cov, plan)
                        out = {"objective": float(max(pred.values())) if mode == "A2" else float(plan.sum())}
                        if mode == "A2":
                            n = integer_counts(plan, int(round(budget)), 32).astype(float)
                            out["true_vmax"] = float(max(sel.true_variances(designs, n).values()))
                        return out

                    started = time.perf_counter()
                    designs, plan, info = sel.alternating(arms, cov, mode, coords=coords, sweeps=args.sweeps, **kw)
                    row["alternating"] = {**evaluate(designs, plan), "trace": info["objective_trace"], "seconds": info["seconds"]}
                    if not args.no_conic:
                        started = time.perf_counter()
                        designs, plan, info = sel.joint(arms, cov, mode, **kw)
                        row["conic"] = {"status": info["status"], "tol": info.get("tol"), "seconds": round(time.perf_counter() - started, 1)}
                        if plan is not None:
                            row["conic"].update(evaluate(designs, plan))
                    handle.write(json.dumps(row, default=float) + "\n")
                    handle.flush()
                    c, a = row.get("conic", {}), row["alternating"]
                    gap = f"{a['objective'] / c['objective'] - 1:+.2%}" if c.get("objective") else "n/a"
                    print(f"  rep {rep} {source:9s} {mode}: alternating {a['objective']:.5g} (frozen start {a['trace'][0]:.5g}, {len(a['trace']) - 1} sweeps) "
                          f"conic {c.get('objective', float('nan')):.5g} [{c.get('status', '-')}] gap {gap}", flush=True)


if __name__ == "__main__":
    main()
