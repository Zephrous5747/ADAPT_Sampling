#!/usr/bin/env python3
"""Paper A (referee point 6): the optimisation cost *sampled*, against the cost model ``C_opt``.

The manuscript models the cost of the parameter optimisation as ``n_ev (1 + r n_k) M_E`` energy-shots (exact BFGS
evaluation counts, a parameter-shift gradient of ``r`` energies per parameter, one energy to 1 mHa costing ``M_E``).
That is a model, not a run.  Here the optimisation is run with sampled energies:

* the pool generators of the UCCSD pool have the spectrum ``{0, +-i}``, so the energy is a trigonometric polynomial of
  degree two in each parameter, ``E(t) = a0 + a1 cos t + b1 sin t + a2 cos 2t + b2 sin 2t`` (ExcitationSolve /
  Rotosolve);
* a coordinate update evaluates the energy at five equispaced angles, each *with sampling noise of standard error
  ``eps_E``* (Gaussian, the central-limit model of the rest of the paper), fits the five coefficients, and sets the
  parameter to the minimiser of the fitted curve;
* parameters are swept in turn, and the optimisation stops when the fitted minimum of a sweep improved on the one
  before by less than ``eps_E`` (``--min-sweeps`` at least), or after ``--max-sweeps``;
* the cost is the number of energy evaluations times the shots one energy needs for ``eps_E``: ``M_E (1 mHa / eps_E)^2``,
  with ``M_E`` of the model (optimal allocation over sorted-insertion fully commuting groups of H, evaluated on the
  optimised state).

The generators are those of exact ADAPT (the trajectory is not changed by the optimiser, so the comparison with
``C_opt`` is step for step); the exact energy error of the sampled-optimiser state is reported next to the exact BFGS one.

Output: ``runs/paper_a/sampled_optimizer.csv`` (per case, ``eps_E`` and step) and ``sampled_optimizer_summary.csv``.

    python scripts/paper_a_sampled_optimizer.py --cases H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import part1_bridge  # noqa: E402,F401
from outputs import run_record, write_csv, write_json  # noqa: E402
from paper_a_cost_decomposition import ENERGY_ERROR, energy_cost, sorted_insertion_groups  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, run_adapt  # noqa: E402

ANGLES = 2.0 * np.pi * np.arange(5) / 5.0
GRID = np.linspace(-np.pi, np.pi, 4001)


def basis(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], axis=-1)


def fit_minimum(values: np.ndarray, offset: float) -> tuple[float, float]:
    """Minimiser and minimum of the degree-two trigonometric polynomial through five equispaced energies."""
    coefficients = np.linalg.solve(basis(ANGLES), values)
    curve = basis(GRID) @ coefficients
    i = int(np.argmin(curve))
    return float(offset + GRID[i]), float(curve[i])


def sampled_optimise(system: AdaptSystem, ops: list[int], params: np.ndarray, eps_e: float, rng, *,
                     min_sweeps: int, max_sweeps: int) -> tuple[np.ndarray, int, int]:
    """Coordinate-wise ExcitationSolve with noisy energies; returns the parameters, energy evaluations, sweeps."""
    params = params.copy()
    evaluations = 0
    previous = np.inf
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        minimum = np.inf
        for j in range(len(ops)):
            base = params[j]
            values = np.empty(5)
            for a, angle in enumerate(ANGLES):
                trial = params.copy()
                trial[j] = base + angle
                values[a] = system.energy(system.state(ops, trial)) + eps_e * rng.standard_normal()
            evaluations += 5
            # the fit is in the shifted variable t = theta - base
            shift, minimum = fit_minimum(values, 0.0)
            params[j] = base + shift
        if sweeps >= min_sweeps and previous - minimum < eps_e:
            break
        previous = minimum
    return params, evaluations, sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_HF", "LiH_R3p0_HF"])
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-3, 3e-4, 1e-4], help="standard error of one energy (Ha)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--min-sweeps", type=int, default=2)
    parser.add_argument("--max-sweeps", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    out = args.out / "paper_a"
    rows, summary = [], []
    for case in args.cases:
        system = AdaptSystem.build(case)
        steps = run_adapt(system, tolerance=CHEMICAL_ACCURACY)
        terms = {p: float(np.real(c)) for p, c in system.problem.metadata["hamiltonian_terms"].items() if set(p) != {"I"}}
        groups = sorted_insertion_groups(terms)
        model = {int(r["iteration"]): r for r in csv.DictReader((out / f"{case}_cost_steps.csv").open())} \
            if (out / f"{case}_cost_steps.csv").exists() else {}
        final_ops = steps[-1].ops
        for eps_e in args.eps:
            per_seed = []
            for seed in range(args.seeds):
                rng = np.random.default_rng([seed, int(eps_e * 1e6)])
                params = np.zeros(0)
                cost = 0.0
                total_evals = 0
                errors = []
                for k in range(1, len(final_ops) + 1):
                    ops = final_ops[:k]
                    params, evals, sweeps = sampled_optimise(system, ops, np.append(params, 0.0), eps_e, rng,
                                                             min_sweeps=args.min_sweeps, max_sweeps=args.max_sweeps)
                    psi = system.state(ops, params)
                    error = system.energy(psi) - system.fci_energy
                    m_e = energy_cost(groups, terms, system.full_state(psi)) * (ENERGY_ERROR / eps_e) ** 2
                    cost += evals * m_e
                    total_evals += evals
                    errors.append(error)
                    rows.append({"case": case, "eps_energy": eps_e, "seed": seed, "step": k, "parameters": k, "sweeps": sweeps,
                                 "energy_evaluations": evals, "shots_per_energy": m_e, "shots": evals * m_e, "energy_error": error,
                                 "energy_error_exact_bfgs": steps[k].error,
                                 "model_c_opt_r2": float(model[k - 1]["c_opt_r2"]) if (k - 1) in model else float("nan"),
                                 "model_c_opt_r4": float(model[k - 1]["c_opt_r4"]) if (k - 1) in model else float("nan")})
                per_seed.append((cost, total_evals, errors[-1], max(e - steps[i + 1].error for i, e in enumerate(errors))))
            shots = np.array([p[0] for p in per_seed])
            r2 = sum(float(model[k]["c_opt_r2"]) for k in model) if model else float("nan")
            r4 = sum(float(model[k]["c_opt_r4"]) for k in model) if model else float("nan")
            record = {"case": case, "eps_energy": eps_e, "steps": len(final_ops), "seeds": args.seeds,
                      "sampled_shots_mean": float(shots.mean()), "sampled_shots_sd": float(shots.std(ddof=1)),
                      "evaluations_mean": float(np.mean([p[1] for p in per_seed])),
                      "final_error_mean": float(np.mean([p[2] for p in per_seed])),
                      "worst_excess_over_bfgs_mean": float(np.mean([p[3] for p in per_seed])),
                      "reached_chemical_accuracy": float(np.mean([p[2] < CHEMICAL_ACCURACY for p in per_seed])),
                      "model_c_opt_r2": r2, "model_c_opt_r4": r4,
                      "ratio_to_model_r2": float(shots.mean() / r2) if model else float("nan"),
                      "ratio_to_model_r4": float(shots.mean() / r4) if model else float("nan")}
            summary.append(record)
            print(f"{case:26s} eps_E {eps_e:7.0e}: sampled {record['sampled_shots_mean']:.3e} +- {record['sampled_shots_sd']:.1e} shots "
                  f"({record['evaluations_mean']:.0f} energies), final dE {record['final_error_mean']:.2e}, reached "
                  f"{record['reached_chemical_accuracy']:.2f}; model r=2 {r2:.3e} (x{record['ratio_to_model_r2']:.2f}), r=4 {r4:.3e} "
                  f"(x{record['ratio_to_model_r4']:.2f})", flush=True)
    write_csv(out / "sampled_optimizer.csv", list(rows[0]), rows)
    write_csv(out / "sampled_optimizer_summary.csv", list(summary[0]), summary)
    write_json(out / "sampled_optimizer_meta.json", {"eps": args.eps, "seeds": args.seeds, "min_sweeps": args.min_sweeps,
                                                     "max_sweeps": args.max_sweeps, "run": run_record()})


if __name__ == "__main__":
    main()
