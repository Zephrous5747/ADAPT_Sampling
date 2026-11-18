#!/usr/bin/env python3
"""Reproduce the Hamiltonian-variance reductions of Yen et al. (arXiv:2201.01471) with the splitting code of this repository.

For a molecule (STO-3G, 1 A bond lengths) the variance of the energy estimator for a normalised budget (sum of shots = 1, optimal allocation) is
computed on the exact ground state for

* **SI**: fully commuting groups by sorted insertion (decreasing |h_p|, first fit), fixed coefficients: V = (sum_a sigma_a)^2;
* **FF**: the first-fit grouping of this repository (decreasing Pauli weight), fixed coefficients;
* **split**: the splitting of the coefficients over all contexts that measure a Pauli (``design.build_fragment_problems`` at level II-A, alternating
  design and allocation: ``DesignSet.solve_ics``), the counterpart of their ICS.

Published values (their Tables I and IV, exact covariances, FC groups): H2 0.136; LiH SI 0.882 / ICS 0.232; BeH2 1.11 / 0.459; H2O 7.59 / 1.50; NH3 18.8 / 3.32
(Table IV mixes CISD covariances for the design with the exact state for the evaluation; this script has the exact-state column only).  Compare the ratios
split / SI (LiH 0.26, BeH2 0.41, H2O 0.20, NH3 0.18), not the absolute numbers: the mapping (their Bravyi-Kitaev, ``--mapping bk``), the geometry, the grouping
heuristic and the solver (theirs iterates) all enter.

    python scripts/reproduce_yen_splitting.py --molecules H2 LiH BeH2 --mapping bk jw
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401  (puts Part I's src on the path)

GEOMETRIES = {  # Angstrom; R = 1 for every bond, H2O at 107.6 degrees (their text); NH3 at 107 degrees (their reference 27 is not in the text we have)
    "H2": "H 0 0 0; H 0 0 1.0",
    "LiH": "Li 0 0 0; H 0 0 1.0",
    "BeH2": "Be 0 0 0; H 0 0 1.0; H 0 0 -1.0",
    "H2O": f"O 0 0 0; H {math.sin(math.radians(107.6 / 2)):.10f} 0 {math.cos(math.radians(107.6 / 2)):.10f}; H {-math.sin(math.radians(107.6 / 2)):.10f} 0 {math.cos(math.radians(107.6 / 2)):.10f}",
    "NH3": "N 0 0 0; " + "; ".join(
        f"H {math.sin(math.radians(a)) * math.sin(math.radians(112.0)):.10f} {math.cos(math.radians(a)) * math.sin(math.radians(112.0)):.10f} {math.cos(math.radians(112.0)):.10f}"
        for a in (0, 120, 240)),
}
PUBLISHED = {"H2": (0.136, 0.136), "LiH": (0.882, 0.232), "BeH2": (1.11, 0.459), "H2O": (7.59, 1.50), "NH3": (18.8, 3.32)}  # (SI, ICS), Tables IV and I


def qubit_hamiltonian(geometry: str, mapping: str):
    """(qubit operator, n_qubits, n_electrons, FCI energy of the molecule) from PySCF RHF (stability followed) and OpenFermion."""
    from openfermion import InteractionOperator, bravyi_kitaev, get_fermion_operator, jordan_wigner
    from pyscf import ao2mo, fci, gto, scf

    from chemistry import _spin_orbital_tensors, converge_stable_rhf  # Part I

    molecule = gto.M(atom=geometry, basis="sto-3g", unit="Angstrom", verbose=0)
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    converge_stable_rhf(mean_field)
    orbitals = mean_field.mo_coeff
    one_body = orbitals.T @ mean_field.get_hcore() @ orbitals
    eri = ao2mo.restore(1, ao2mo.kernel(molecule, orbitals), orbitals.shape[1])
    h1, h2 = _spin_orbital_tensors(one_body, eri)
    fermion = get_fermion_operator(InteractionOperator(float(molecule.energy_nuc()), h1, h2))
    n_qubits = 2 * orbitals.shape[1]
    operator = jordan_wigner(fermion) if mapping == "jw" else bravyi_kitaev(fermion, n_qubits)
    e_fci = float(fci.FCI(mean_field).kernel()[0])
    return operator, n_qubits, int(molecule.nelectron), e_fci


def ground_state(operator, n_qubits: int, n_electrons: int, mapping: str, e_fci: float) -> tuple[np.ndarray, float]:
    """Lowest eigenvector of H in the sector of ``n_electrons`` (a particle-number penalty keeps other sectors out)."""
    from openfermion import bravyi_kitaev, get_sparse_operator, jordan_wigner, number_operator
    from scipy.sparse.linalg import eigsh

    number = number_operator(n_qubits)
    number = jordan_wigner(number) if mapping == "jw" else bravyi_kitaev(number, n_qubits)
    n_op = get_sparse_operator(number, n_qubits)
    h = get_sparse_operator(operator, n_qubits)
    shifted = n_op - n_electrons * type(n_op)(np.eye(2 ** n_qubits)) if False else n_op
    penalty = (n_op - n_electrons * _identity(n_qubits)) @ (n_op - n_electrons * _identity(n_qubits))
    values, vectors = eigsh(h + 10.0 * penalty, k=1, which="SA")
    state = vectors[:, 0].astype(complex)
    energy = float(np.vdot(state, h @ state).real)
    if abs(energy - e_fci) > 1e-7:
        print(f"  warning: ground-state energy {energy:.9f} differs from the FCI energy {e_fci:.9f}", flush=True)
    return state, energy


def _identity(n_qubits: int):
    import scipy.sparse as sp

    return sp.identity(2 ** n_qubits, format="csr", dtype=complex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--molecules", nargs="+", default=["H2", "LiH"])
    parser.add_argument("--mapping", nargs="+", default=["bk"], choices=["bk", "jw"])
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a/yen_reproduction.csv"))
    args = parser.parse_args()

    from openfermion import count_qubits  # noqa: F401

    from contexts import build_context_library
    from design import DesignSet, build_fragment_problems
    from gradients import GradientProblem
    from paper_a_cost_decomposition import sorted_insertion_groups
    from pauli_fc import greedy_fc_groups, openfermion_qubitop_to_dict
    from pauli_ops import PauliEvaluator
    from sampler import OracleMoments

    rows = []
    for name in args.molecules:
        for mapping in args.mapping:
            started = time.perf_counter()
            operator, n_qubits, n_electrons, e_fci = qubit_hamiltonian(GEOMETRIES[name], mapping)
            terms = {p: float(np.real(c)) for p, c in openfermion_qubitop_to_dict(operator, n_qubits).items()}
            identity = terms.pop("I" * n_qubits, 0.0)
            state, energy = ground_state(operator, n_qubits, n_electrons, mapping, e_fci)
            evaluator = PauliEvaluator(state)
            result = {"molecule": name, "mapping": mapping, "n_qubits": n_qubits, "pauli_terms": len(terms), "energy": energy}

            def variance(groups) -> tuple[float, int]:
                return sum(evaluator.fragment_std({p: terms[p] for p in g}) for g in groups) ** 2, len(groups)

            si_groups = sorted_insertion_groups(terms)
            ff_groups = greedy_fc_groups(sorted(terms))
            result["SI"], result["SI_groups"] = variance(si_groups)
            result["FF"], result["FF_groups"] = variance(ff_groups)
            problem = GradientProblem(case_id=f"{name}_H", state_name="exact", n_qubits=n_qubits, state_energy=energy, labels=["H"], kinds=["H"],
                                      gradients=np.array([1.0]), commutator_terms=[terms], evaluator=evaluator)
            for tag, groups in (("split_SI", si_groups), ("split_FF", ff_groups)):
                problem._parent_groups = [list(g) for g in groups]
                library = build_context_library(problem, "canonical", groups=groups)
                moments = OracleMoments(library, state)
                problems = build_fragment_problems(problem, library, moments, "II-A")
                design = DesignSet(problems, len(library.contexts))
                solution = design.solve_ics([0], epsilon=1.0, max_iter=args.max_iter)
                result[tag] = float(solution.total)
                result[f"{tag}_contexts"] = len(library.contexts)
            published = PUBLISHED.get(name)
            result["published_SI"], result["published_ICS"] = published if published else (float("nan"), float("nan"))
            result["ratio_split_over_SI"] = result["split_SI"] / result["SI"]
            result["published_ratio"] = published[1] / published[0] if published else float("nan")
            result["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(result)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
