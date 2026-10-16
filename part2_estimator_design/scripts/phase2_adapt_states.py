#!/usr/bin/env python3
"""Phase 2: run exact ADAPT-VQE from each HF case and save every state on the way.

The trajectory is written to ``.cache/trajectory/<case>_adapt.npz`` (states, ansatz,
parameters, energies, exact gradients), and a summary table to
``runs/<case>/<case>_adapt_trajectory.csv``.  Each saved state is then available to
every Part II script as the case ``<geometry>_ADAPT<k>``; the script checks that
the gradients rebuilt from the commutator expansions agree with the sector
computation for every step.

Example::

    python scripts/phase2_adapt_states.py --cases H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from outputs import run_record, write_csv, write_json  # noqa: E402
from trajectory import (  # noqa: E402
    CHEMICAL_ACCURACY,
    AdaptSystem,
    load_trajectory_problem,
    run_adapt,
    save_trajectory,
    trajectory_path,
)

HF_CASES = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF",
            "H2O_eq_HF", "H2O_stretch_HF"]


def trajectory_id(base_case: str, k: int) -> str:
    return f"{base_case[: -len('_HF')]}_ADAPT{k}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=HF_CASES[:3])
    parser.add_argument("--tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    for case in args.cases:
        started = time.perf_counter()
        system = AdaptSystem.build(case)
        steps = run_adapt(system, tolerance=args.tolerance, max_iterations=args.max_iterations)
        path = save_trajectory(system, steps, trajectory_path(case))
        rows = []
        for s in steps:
            absg = np.abs(s.gradients)
            order = np.argsort(-absg, kind="stable")
            problem = load_trajectory_problem(trajectory_id(case, s.iteration))  # checks gradients
            rows.append({
                "case_id": trajectory_id(case, s.iteration),
                "iteration": s.iteration,
                "energy": s.energy,
                "energy_error": s.error,
                "max_abs_gradient": float(absg[order[0]]),
                "second_abs_gradient": float(absg[order[1]]),
                "relative_gap": float((absg[order[0]] - absg[order[1]]) / absg[order[0]]),
                "n_nonzero_gradients": int(problem.count_nonzero_gradients()),
                "best": system.labels[order[0]],
                "added_next": system.labels[s.selected] if s.selected is not None else "",
            })
            print(f"{rows[-1]['case_id']:32s} dE {s.error:10.3e}  max|g| {absg[order[0]]:.4e}  "
                  f"gap {rows[-1]['relative_gap']:6.1%}  nonzero {rows[-1]['n_nonzero_gradients']:3d}  "
                  f"next {rows[-1]['added_next']}", flush=True)
        write_csv(args.out / case / f"{case}_adapt_trajectory.csv", list(rows[0]), rows)
        write_json(args.out / case / f"{case}_adapt_trajectory_meta.json",
                   {"tolerance": args.tolerance, "fci_energy": system.fci_energy,
                    "sector_dimension": int(system.indices.size), "pool_size": len(system.labels),
                    "converged": bool(steps[-1].error < args.tolerance), "steps": len(steps) - 1,
                    "states_file": str(path), "seconds": round(time.perf_counter() - started, 1),
                    "run": run_record()})
        print(f"{case}: {len(steps) - 1} generators to |dE| < {args.tolerance:g} "
              f"({time.perf_counter() - started:.0f}s), saved {path}", flush=True)


if __name__ == "__main__":
    main()
