#!/usr/bin/env python3
"""Selection versus parameter-optimisation cost along exact ADAPT trajectories.

Paper A, question Q7.  For each geometry the exact ADAPT trajectory is rerun with
the optimiser's evaluation count recorded at every iteration, and three costs are
assembled in context-shots:

``C_opt``
    The cost model of the manuscript: iteration ``k`` optimises ``n_k`` parameters
    with ``n_k^ev`` energy-and-gradient evaluations, each counted as one energy plus
    a parameter-shift gradient of ``r`` energies per parameter, and one energy to
    standard error ``eps = 1 mHa`` costs ``M_E = (sum_alpha sigma_alpha)^2 / eps^2``
    over sorted-insertion fully commuting groups of H, evaluated at the optimised
    state.  ``C_opt = sum_k n_k^ev (1 + r n_k) M_E(k)`` for ``r = 2`` and ``4``.
``C_sel, M1``
    Static all-gradient selection at its best (Part I's M1): at every step one
    allocation over the whole pool on the parent contexts, at the common radius
    ``rho max_i |g_i| / 2``, which certifies a ``rho``-good choice.  It uses the
    exact gradient scale and fragment variances, so it is an oracle lower bound for
    any static method.
``C_sel, measured``
    The median cumulative selection cost of the Phase 4 trajectories
    (``runs/<case>/phase4``), where those exist.

Output: ``runs/paper_a/<case>_cost_steps.csv`` and ``runs/paper_a/cost_summary.csv``.

    python scripts/paper_a_cost_decomposition.py --cases H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import allocate_context_shots, z_from_delta  # noqa: E402
from pauli_fc import greedy_fc_groups  # noqa: E402  (Part I)
from pauli_ops import PauliEvaluator  # noqa: E402  (Part I)
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, problem_at, run_adapt  # noqa: E402

ENERGY_ERROR = 1e-3  # hartree, one standard error per energy estimate


def sorted_insertion_groups(terms: dict[str, float]) -> list[list[str]]:
    """Fully commuting groups of H by sorted insertion (decreasing |h_p|, first fit)."""
    order = sorted(terms, key=lambda p: (-abs(terms[p]), p))
    return greedy_fc_groups(order, sort=False)


def energy_cost(groups, terms, psi_full) -> float:
    """``(sum_alpha sigma_alpha)^2 / eps^2`` at the state, optimal allocation over groups."""
    evaluator = PauliEvaluator(psi_full)
    total = sum(evaluator.fragment_std({p: terms[p] for p in group}) for group in groups)
    return total ** 2 / ENERGY_ERROR ** 2


def m1_cost(problem, rho: float, delta: float) -> float:
    """Static all-gradient estimation to the common radius ``rho max|g| / 2``."""
    sigmas = problem.fragment_sigmas(problem.parent_fc_groups())
    radius = rho * float(problem.abs_gradients.max()) / 2.0
    z = z_from_delta(delta, problem.n_generators)
    return float(np.sum(allocate_context_shots(sigmas, radius / z)))


def measured_selection(out: Path, case: str) -> dict[str, float]:
    path = out / case / "phase4" / f"{case}_phase4_trajectories.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    result = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        totals = [float(r["total_shots"]) for r in rows if r["method"] == method]
        result[method] = float(np.median(totals))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        default=["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF"])
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    summary = []
    for case in args.cases:
        started = time.perf_counter()
        system = AdaptSystem.build(case)
        steps = run_adapt(system, tolerance=CHEMICAL_ACCURACY)
        terms = {p: float(np.real(c)) for p, c in system.problem.metadata["hamiltonian_terms"].items()
                 if set(p) != {"I"}}
        groups = sorted_insertion_groups(terms)
        rows = []
        for k, step in enumerate(steps):
            if step.selected is None:
                continue
            after = steps[k + 1]
            psi_after = system.full_state(system.state(after.ops, after.params))
            psi_now = system.full_state(system.state(step.ops, step.params))
            problem = problem_at(system.problem, psi_now, f"{case}_step{k}", energy=step.energy,
                                 sector_gradients=step.gradients)
            n_params = len(after.ops)
            m_energy = energy_cost(groups, terms, psi_after)
            rows.append({
                "case_id": case, "iteration": k, "energy_error": step.error,
                "max_abs_gradient": float(np.abs(step.gradients).max()),
                "added": system.labels[step.selected], "parameters": n_params,
                "evaluations": step.evaluations, "energy_cost": m_energy,
                "c_opt_r2": step.evaluations * (1 + 2 * n_params) * m_energy,
                "c_opt_r4": step.evaluations * (1 + 4 * n_params) * m_energy,
                "c_sel_m1": m1_cost(problem, args.rho, args.delta),
            })
            print(f"{case} step {k}: params {n_params:2d} evals {step.evaluations:3d} "
                  f"M_E {m_energy:.3e} M1 {rows[-1]['c_sel_m1']:.3e}", flush=True)
        write_csv(args.out / "paper_a" / f"{case}_cost_steps.csv", list(rows[0]), rows)
        measured = measured_selection(args.out, case)
        record = {
            "case_id": case, "steps": len(rows), "final_error": steps[-1].error,
            "hamiltonian_terms": len(terms), "energy_groups": len(groups),
            "c_opt_r2": sum(r["c_opt_r2"] for r in rows), "c_opt_r4": sum(r["c_opt_r4"] for r in rows),
            "c_sel_m1": sum(r["c_sel_m1"] for r in rows),
            **{f"c_sel_{m}": v for m, v in measured.items()},
            "seconds": round(time.perf_counter() - started, 1),
        }
        summary.append(record)
        print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in record.items()}, flush=True)
    columns = list(dict.fromkeys(k for r in summary for k in r))
    write_csv(args.out / "paper_a" / "cost_summary.csv", columns, summary)
    write_json(args.out / "paper_a" / "cost_meta.json",
               {"rho": args.rho, "delta": args.delta, "energy_error": ENERGY_ERROR,
                "grouping": "sorted insertion of H (decreasing |h_p|), first-fit FC",
                "energy_state": "optimised state of each iteration", "run": run_record()})


if __name__ == "__main__":
    main()
