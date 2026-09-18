"""Standard spin-orbital UCCSD generator pool.

The pool contains anti-Hermitian singles and doubles from occupied to virtual
spin orbitals of the closed-shell Hartree-Fock reference.  It conserves electron
number and ``S_z`` but not generally ``S^2``, which is the generator universe
fixed by the Part I report.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Generator:
    """One pool generator ``G_i`` and its Jordan-Wigner image."""

    index: int
    label: str
    kind: str  # "S" for a single excitation, "D" for a double
    qubit_operator: object  # openfermion.QubitOperator


def _anti_hermitian(term) -> object:
    from openfermion import FermionOperator, hermitian_conjugated, jordan_wigner

    excitation = FermionOperator(term, 1.0)
    return jordan_wigner(excitation - hermitian_conjugated(excitation))


def uccsd_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    """Build the spin-conserving singles-and-doubles pool.

    Singles keep the spin of the excited electron.  Doubles keep the number of
    alpha and beta electrons separately.  Ordering is deterministic: all singles
    in ascending ``(i, a)``, then all doubles in ascending ``(i, j, a, b)``.
    """
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    generators: list[Generator] = []

    for i in occupied:
        for a in virtual:
            if i % 2 != a % 2:
                continue
            generators.append(
                Generator(
                    index=len(generators),
                    label=f"S {i} -> {a}",
                    kind="S",
                    qubit_operator=_anti_hermitian(((a, 1), (i, 0))),
                )
            )

    for i, j in combinations(occupied, 2):
        occupied_alpha = (i % 2 == 0) + (j % 2 == 0)
        for a, b in combinations(virtual, 2):
            virtual_alpha = (a % 2 == 0) + (b % 2 == 0)
            if occupied_alpha != virtual_alpha:
                continue
            generators.append(
                Generator(
                    index=len(generators),
                    label=f"D {i},{j} -> {a},{b}",
                    kind="D",
                    qubit_operator=_anti_hermitian(((a, 1), (b, 1), (j, 0), (i, 0))),
                )
            )
    return generators
