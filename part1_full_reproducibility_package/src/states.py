"""Computational-basis states in the OpenFermion qubit ordering.

Qubit ``i`` occupies bit ``n_qubits - 1 - i`` of the basis index, matching
:func:`openfermion.linalg.get_sparse_operator`.  Spin orbitals are interleaved
(even index alpha, odd index beta) and the closed-shell Hartree-Fock reference
occupies spin orbitals ``0 .. n_electrons - 1``.
"""
from __future__ import annotations

import gc
from itertools import combinations
from typing import Sequence

import numpy as np

DENSE_DIAGONALISATION_LIMIT = 512


def expectation_without_matrix(hamiltonian, state: np.ndarray) -> float:
    """Energy expectation evaluated term by term, never forming an operator matrix."""
    from pauli_fc import openfermion_qubitop_to_dict
    from pauli_ops import PauliEvaluator

    terms = openfermion_qubitop_to_dict(hamiltonian.operator, hamiltonian.n_qubits)
    mean, _ = PauliEvaluator(state).fragment_mean_std(
        {pauli: coefficient.real for pauli, coefficient in terms.items()}
    )
    return mean


def basis_index(occupied: Sequence[int], n_qubits: int) -> int:
    """Computational-basis index for a set of occupied spin orbitals."""
    index = 0
    for orbital in occupied:
        if not 0 <= orbital < n_qubits:
            raise ValueError(f"spin orbital {orbital} outside 0..{n_qubits - 1}")
        index |= 1 << (n_qubits - 1 - orbital)
    return index


def hartree_fock_state(n_qubits: int, n_electrons: int) -> np.ndarray:
    """Closed-shell Hartree-Fock determinant as a state vector."""
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[basis_index(range(n_electrons), n_qubits)] = 1.0
    return state


def sector_indices(
    n_qubits: int,
    n_alpha: int,
    n_beta: int,
    *,
    max_excitation: int | None = None,
    n_reference_electrons: int | None = None,
) -> list[int]:
    """Basis indices in a fixed ``(N_alpha, N_beta)`` sector.

    With ``max_excitation`` set, the sector is truncated to determinants within
    that excitation rank of the closed-shell reference, which gives the CISD
    space for ``max_excitation=2``.
    """
    alpha_orbitals = range(0, n_qubits, 2)
    beta_orbitals = range(1, n_qubits, 2)
    n_reference = (
        n_reference_electrons if n_reference_electrons is not None else n_alpha + n_beta
    )
    reference = set(range(n_reference))

    indices = []
    for alpha_occupied in combinations(alpha_orbitals, n_alpha):
        for beta_occupied in combinations(beta_orbitals, n_beta):
            occupied = set(alpha_occupied) | set(beta_occupied)
            if max_excitation is not None and len(reference - occupied) > max_excitation:
                continue
            indices.append(basis_index(occupied, n_qubits))
    return sorted(indices)


def sector_block(operator, n_qubits: int, indices: Sequence[int]):
    """Restriction of ``operator`` to a set of basis indices.

    The full sparse operator is 2**n_qubits square and is the largest object in
    the whole pipeline, so it is released as soon as the (much smaller) block has
    been extracted.  Holding on to it is what pushes 14-qubit cases out of memory.
    """
    from openfermion.linalg import get_sparse_operator

    matrix = get_sparse_operator(operator, n_qubits=n_qubits).tocsr()
    try:
        return matrix[indices, :][:, indices]
    finally:
        del matrix
        gc.collect()


def restricted_ground_state(
    operator, n_qubits: int, indices: Sequence[int]
) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of ``operator`` restricted to the given basis indices."""
    from scipy.sparse.linalg import eigsh

    block = sector_block(operator, n_qubits, indices)
    if block.shape[0] <= DENSE_DIAGONALISATION_LIMIT:
        eigenvalues, eigenvectors = np.linalg.eigh(block.toarray())
        energy, sub_vector = float(eigenvalues[0].real), eigenvectors[:, 0]
    else:
        eigenvalues, eigenvectors = eigsh(block, k=1, which="SA", tol=1e-12)
        energy, sub_vector = float(eigenvalues[0].real), eigenvectors[:, 0]

    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[list(indices)] = sub_vector
    state /= np.linalg.norm(state)
    return energy, state


def prepare_state(hamiltonian, state_name: str) -> tuple[float, np.ndarray]:
    """Build the fixed ADAPT-proxy state named by a :class:`CaseSpec`.

    ``HF`` is the Hartree-Fock determinant.  ``CISD`` is the lowest eigenstate of
    the Hamiltonian inside the singles-and-doubles space, used as a correlated
    proxy for a later ADAPT state.
    """
    name = state_name.upper()
    if name == "HF":
        state = hartree_fock_state(hamiltonian.n_qubits, hamiltonian.n_electrons)
        return expectation_without_matrix(hamiltonian, state), state
    if name == "CISD":
        indices = sector_indices(
            hamiltonian.n_qubits,
            hamiltonian.n_alpha,
            hamiltonian.n_beta,
            max_excitation=2,
            n_reference_electrons=hamiltonian.n_electrons,
        )
        return restricted_ground_state(
            hamiltonian.operator, hamiltonian.n_qubits, indices
        )
    raise ValueError(f"unknown state {state_name!r}")
