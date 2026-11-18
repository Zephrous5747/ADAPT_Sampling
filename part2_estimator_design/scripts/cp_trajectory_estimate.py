"""Cumulative selection-shot ratio along ADAPT trajectories from the measured Test B states, and the implied end-to-end gain.

Cumulative ratio over the measured steps of one trajectory: sum over steps of the mean shots of the data-only design divided by the same sum for
the arm (steps are independent runs; 95% interval by bootstrapping the replicates of every step independently, 4000 resamples).  Steps that were not
measured (and steps beyond the last measured one) are not included; the share of the last measured step in the data-only total is printed because
it decides the result.  End-to-end gain: S_total = 1 / ((1 - f) + f / s) for a selection share f of the total cost and selection ratio s.
    python scripts/cp_trajectory_estimate.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(7)


def shots(d, state, src):
    path = os.path.join(ROOT, "runs", d, "testb", f"{state}__{src}.jsonl")
    return {json.loads(l)["rep"]: json.loads(l) for l in open(path)}


def arr(rec, reps):
    return np.array([rec[r]["selection_shots"] for r in reps], float)


TRAJ = {  # name: list of (step, dir, state) ; arms: label -> source
    "LiH 3.0 A, rho=0 (steps 3, 5, 6)": [(3, "cp_v5", "LiH_r30_m3"), (5, "cp_v6", "LiH_r30_m5"), (6, "cp_v7", "LiH_r30_m6")],
    "LiH 3.0 A, rho=0.1 (steps 3, 5, 6)": [(3, "cp_v9", "LiH_r30_m3"), (5, "cp_v9", "LiH_r30_m5"), (6, "cp_v9", "LiH_r30_m6")],
    "H2O eq., rho=0.1 (steps 3, 7, 11)": [(3, "cp_v10", "H2O_eq_m3"), (7, "cp_v10", "H2O_eq_m7"), (11, "cp_v10", "H2O_eq_m11")],
    "H2O str., rho=0.1 (steps 3, 8, 13)": [(3, "cp_v10", "H2O_str_m3"), (8, "cp_v10", "H2O_str_m8"), (13, "cp_v10", "H2O_str_m13")],
}
ARMS = {"updated, trust 128": "CLASSICAL", "static": "CLASSICAL_INF"}


def main():
    print("trajectory | arm | data-only total | arm total | cumulative R [95% CI] | share of last measured step in data-only total")
    results = {}
    for name, steps in TRAJ.items():
        for label, src in ARMS.items():
            base_sum, arm_sum, per = 0.0, 0.0, []
            data = []
            for step, d, state in steps:
                b, a = shots(d, state, "NEUTRAL"), shots(d, state, src)
                reps = sorted(set(a) & set(b))
                data.append((arr(b, reps), arr(a, reps)))
                base_sum += data[-1][0].mean()
                arm_sum += data[-1][1].mean()
                per.append(data[-1][0].mean())
            boots = []
            for _ in range(4000):
                bs = sum(b[RNG.integers(0, len(b), len(b))].mean() for b, a in data)
                as_ = sum(a[RNG.integers(0, len(a), len(a))].mean() for b, a in data)
                boots.append(bs / as_)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"{name} | {label} | {base_sum:.3e} | {arm_sum:.3e} | {base_sum/arm_sum:.3f} [{lo:.3f}, {hi:.3f}] | {per[-1]/base_sum:.0%}")
            results[(name, label)] = base_sum / arm_sum
    print("\nEnd-to-end gain S_total = 1/((1-f)+f/s), s = cumulative selection ratio above and the best single-step ratios seen")
    for s in (1.00, 1.08, 1.28, 1.74):
        row = []
        for f in (0.0005, 0.05, 0.14, 0.30, 0.50):
            row.append(f"f={f:.2%}: {1/((1-f)+f/s):.3f}")
        print(f"s={s:.2f}  " + "  ".join(row))


if __name__ == "__main__":
    main()
