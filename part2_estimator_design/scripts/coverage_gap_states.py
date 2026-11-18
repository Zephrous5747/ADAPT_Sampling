#!/usr/bin/env python3
"""Synthetic-gap states for the coverage test of the confidence intervals (round-1 review, D5).

The real states of the study have the gaps that chemistry gave them.  To test whether the selection error stays below
``delta`` and how often an interval misses, one wants states whose gap is *set*: here a template state of a saved exact
ADAPT trajectory (default ``H4_square_eq_side1p0_ADAPT3``) is rotated by the generator of its runner-up,

    psi(t) = exp(t G_b) psi_template ,      b = second largest |g_i| of the template,

and for every target relative gap ``(|g|_(1) - |g|_(2)) / |g|_(1)`` the smallest ``|t|`` is taken at which the *same* generator
still leads and the gap has that value (root of a bracketed scan, ``brentq``).  If a target is wider than the template's gap, the leader's own generator widens it instead. The states keep the support, the Pauli variances
and the rare outcomes of the template, which is what a hand-made Gaussian toy would lose.

The states are written as a trajectory file of a fictitious geometry ``<geometry>_<tag>``, so every script of the study reads
them as the cases ``<geometry>_<tag>_ADAPT<j>`` (j = 0 is the template itself, j = 1.. the targets in the order given):

    python scripts/coverage_gap_states.py                                     # H4, targets 0.5 ... 0.005
    python scripts/step5_external_baselines.py --cases H4_square_eq_side1p0_GAPS_ADAPT1 --configs "II-0, safe, bound start, min shots=400, diagnose" ...

Output: ``.cache/trajectory/<geometry>_<tag>_HF_adapt.npz`` and ``_problem.pickle`` (the cache is not archived; the script
rebuilds it in seconds) and ``runs/coverage_gap/<geometry>_<tag>_states.csv`` (target and achieved gaps, leaders, energies).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from trajectory import (  # noqa: E402
    DEFAULT_CACHE,
    AdaptSystem,
    load_trajectory,
    problem_cache_path,
    split_case,
    trajectory_path,
)

DEFAULT_TARGETS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)


def system_from_cache(base_case: str, cache: Path) -> AdaptSystem:
    """The AdaptSystem of the cached trajectory, with the Hamiltonian stored in its problem pickle instead of a fresh SCF.

    A fresh SCF of stretched square H4 returns orbitals of other sign than the ones the saved trajectory was built with (on Trillium and
    Fir, not in the WSL build that made the cache): the stored states then give other gradients (differences of 0.57).  The Hamiltonian terms
    stored with the problem are exactly the ones of the saved states, so the system is built from them (same construction as ``AdaptSystem.build``)."""
    import pickle

    from pauli_fc import dict_to_openfermion_qubitop  # Part I
    from pool import uccsd_pool
    from states import basis_index, sector_block, sector_indices

    with problem_cache_path(base_case, cache).open("rb") as handle:
        problem = pickle.load(handle)
    n_qubits, n_electrons = problem.n_qubits, int(problem.metadata["n_electrons"])
    operator = dict_to_openfermion_qubitop(problem.metadata["hamiltonian_terms"])
    n_alpha = n_electrons // 2  # closed-shell singlets (H4, LiH)
    indices = np.asarray(sector_indices(n_qubits, n_alpha, n_electrons - n_alpha), dtype=np.int64)
    block = sector_block(operator, n_qubits, indices).real.tocsr()
    pool = uccsd_pool(n_qubits, n_electrons)
    generators = [sector_block(g.qubit_operator, n_qubits, indices).tocsr().real.tocsr() for g in pool]
    hf = np.zeros(indices.size)
    hf[np.searchsorted(indices, basis_index(range(n_electrons), n_qubits))] = 1.0
    fci = float(np.linalg.eigvalsh(block.toarray())[0])
    system = AdaptSystem(base_case, n_qubits, n_electrons, indices, block, generators, [g.label for g in pool], hf, fci,
                         {"rhf_energy": problem.metadata.get("rhf_energy_hartree"), "sector_dimension": int(indices.size)})
    system.problem = problem
    return system


def relative_gap(g: np.ndarray) -> tuple[float, int, int]:
    """(relative gap of the two largest |g|, index of the leader, index of the runner-up)."""
    a = np.abs(g)
    order = np.argsort(-a)
    first, second = int(order[0]), int(order[1])
    return float((a[first] - a[second]) / a[first]), first, second


def find_rotation(system: AdaptSystem, psi0: np.ndarray, rotor: int, leader: int, target: float,
                  span: float = 0.8, points: int = 801) -> float:
    """Smallest |t| with the leader unchanged and relative gap ``target`` along exp(t G_rotor) psi0."""
    generator = system.generators[rotor]

    def gap_at(t: float) -> float:
        g = system.gradients(expm_multiply(t * generator, psi0))
        rel, first, _ = relative_gap(g)
        return rel - target if first == leader else np.nan

    grid = np.linspace(-span, span, points)
    values = np.array([gap_at(t) for t in grid])
    roots = []
    for i in range(points - 1):
        a, b = values[i], values[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0:
            roots.append(brentq(gap_at, grid[i], grid[i + 1], xtol=1e-12) if a * b < 0 else grid[i])
    if not roots:
        raise ValueError(f"no rotation by generator {rotor} reaches a relative gap of {target} with the leader unchanged")
    return float(min(roots, key=abs))


def best_rotation(system: AdaptSystem, psi0: np.ndarray, leader: int, runner_up: int, target: float) -> tuple[int, float]:
    """The generator (runner-up: closes the gap, leader: widens it) and angle of the smallest rotation that reaches ``target``."""
    found = []
    for rotor in (runner_up, leader):
        try:
            found.append((rotor, find_rotation(system, psi0, rotor, leader, target)))
        except ValueError:
            continue
    if not found:
        raise ValueError(f"no rotation reaches a relative gap of {target} with the leader unchanged")
    return min(found, key=lambda item: abs(item[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", default="H4_square_eq_side1p0_ADAPT3", help="a saved exact-ADAPT state, <geometry>_ADAPT<k>")
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS), help="relative gaps")
    parser.add_argument("--tag", default="GAPS")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path("runs") / "coverage_gap")
    args = parser.parse_args()

    base_case, k = split_case(args.template)
    geometry = base_case[: -len("_HF")]
    new_geometry = f"{geometry}_{args.tag}"
    new_base = f"{new_geometry}_HF"
    system = system_from_cache(base_case, args.cache)
    data = load_trajectory(base_case, args.cache)
    psi0 = data["states"][k]
    worst = float(np.abs(system.gradients(psi0) - data["gradients"][k]).max())
    assert worst < 1e-8, f"the cached Hamiltonian does not reproduce the stored gradients of {args.template} (max difference {worst:.2e})"
    g0 = system.gradients(psi0)
    rel0, leader, runner_up = relative_gap(g0)
    print(f"template {args.template}: relative gap {rel0:.4f}, leader {system.labels[leader]}, runner-up {system.labels[runner_up]}")

    states, rows = [psi0], [{"case": f"{new_geometry}_ADAPT0", "target": rel0, "t": 0.0, "rotor": ""}]
    for target in args.targets:
        rotor, t = best_rotation(system, psi0, leader, runner_up, target)
        states.append(expm_multiply(t * system.generators[rotor], psi0))
        rows.append({"case": f"{new_geometry}_ADAPT{len(states) - 1}", "target": target, "t": t, "rotor": system.labels[rotor]})
    states = np.stack(states)
    gradients = np.stack([system.gradients(s) for s in states])
    energies = np.array([system.energy(s) for s in states])
    for row, g, e in zip(rows, gradients, energies):
        rel, first, second = relative_gap(g)
        row.update(achieved=rel, gap=float(abs(g[first]) - abs(g[second])), max_gradient=float(abs(g[first])), leader=system.labels[first],
                   runner_up=system.labels[second], leader_unchanged=bool(first == leader), energy=float(e),
                   energy_error=float(e - system.fci_energy))
        print(f"  {row['case']:40s} target {row['target']:.4f}  achieved {rel:.4f}  gap {row['gap']:.3e}  t={row['t']:+.4f}  "
              f"leader unchanged: {row['leader_unchanged']}")
        assert row["leader_unchanged"], row

    path = trajectory_path(new_base, args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(problem_cache_path(base_case, args.cache), problem_cache_path(new_base, args.cache))
    np.savez_compressed(path, base_case=new_base, indices=data["indices"], states=states, ops=np.full((len(states), 1), -1, dtype=np.int64),
                        params=np.zeros((len(states), 1)), energies=energies, fci_energy=system.fci_energy, gradients=gradients,
                        labels=np.array(system.labels))
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / f"{new_geometry}_states.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path, "and", table)


if __name__ == "__main__":
    main()
