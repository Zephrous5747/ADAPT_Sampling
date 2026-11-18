"""Hamiltonian-variance reduction of Yen et al. (arXiv:2201.01471): H2 and LiH with the splitting code of this repository."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reproduce_yen_splitting as yen  # noqa: E402


def _terms(operator, n_qubits):
    from pauli_fc import openfermion_qubitop_to_dict

    return {p: float(np.real(c)) for p, c in openfermion_qubitop_to_dict(operator, n_qubits).items()}


@pytest.mark.parametrize("mapping", ["bk", "jw"])
def test_h2_variance_of_the_hamiltonian_is_their_value_and_independent_of_the_grouping(mapping):
    from paper_a_cost_decomposition import sorted_insertion_groups
    from pauli_ops import PauliEvaluator

    operator, n_qubits, n_electrons, e_fci = yen.qubit_hamiltonian(yen.GEOMETRIES["H2"], mapping)
    terms = _terms(operator, n_qubits)
    terms.pop("I" * n_qubits)
    state, energy = yen.ground_state(operator, n_qubits, n_electrons, mapping, e_fci)
    assert energy == pytest.approx(e_fci, abs=1e-8)
    evaluator = PauliEvaluator(state)
    variance = sum(evaluator.fragment_std({p: terms[p] for p in g}) for g in sorted_insertion_groups(terms)) ** 2
    assert variance == pytest.approx(0.136, abs=5e-4)  # their Tables I and IV, H2 at 1 A


def test_the_hamiltonian_coefficients_are_reconstructed_by_the_splitting_and_it_does_not_lose_to_the_fixed_design():
    from contexts import build_context_library
    from design import DesignSet, build_fragment_problems
    from gradients import GradientProblem
    from paper_a_cost_decomposition import sorted_insertion_groups
    from pauli_ops import PauliEvaluator
    from sampler import OracleMoments

    operator, n_qubits, n_electrons, e_fci = yen.qubit_hamiltonian(yen.GEOMETRIES["LiH"], "jw")
    terms = _terms(operator, n_qubits)
    terms.pop("I" * n_qubits)
    state, energy = yen.ground_state(operator, n_qubits, n_electrons, "jw", e_fci)
    evaluator = PauliEvaluator(state)
    groups = sorted_insertion_groups(terms)
    fixed = sum(evaluator.fragment_std({p: terms[p] for p in g}) for g in groups) ** 2
    problem = GradientProblem(case_id="LiH_H", state_name="exact", n_qubits=n_qubits, state_energy=energy, labels=["H"], kinds=["H"],
                              gradients=np.array([1.0]), commutator_terms=[terms], evaluator=evaluator)
    problem._parent_groups = [list(g) for g in groups]
    library = build_context_library(problem, "canonical", groups=groups)
    problems = build_fragment_problems(problem, library, OracleMoments(library, state), "II-A")
    design = DesignSet(problems, len(library.contexts))
    solution = design.solve_ics([0], epsilon=1.0, max_iter=40)
    assert design.max_constraint_residual() < 1e-8  # the split coefficients add up to the Hamiltonian
    assert solution.total <= fixed * (1 + 1e-9)
    assert solution.total / fixed < 0.4  # their LiH ratio ICS / SI is 0.26; this code 0.24 to 0.26 on the exact state
