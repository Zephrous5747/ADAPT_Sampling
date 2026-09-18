"""Molecular input and qubit Hamiltonian construction.

Conventions fixed here and relied on by every other module:

* Spin orbitals are interleaved: spatial orbital ``p`` gives spin orbitals
  ``2p`` (alpha) and ``2p + 1`` (beta).
* PySCF returns two-electron integrals in chemist notation ``(pq|rs)``.
  OpenFermion's :class:`~openfermion.InteractionOperator` expects physicist
  notation ``<pq|rs> = (pr|qs)`` with the spin pattern ``(a, b, b, a)`` and the
  one-half prefactor folded into the tensor.
* Jordan-Wigner mapping, no frozen core, no tapering.

:func:`validate_hamiltonian` is a hard gate.  A Hamiltonian that does not
reproduce the PySCF RHF energy on the Hartree-Fock determinant and the FCI
energy as its lowest eigenvalue in the target particle-number sector is wrong,
and every gradient and shot count derived from it is meaningless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

Atom = tuple[str, tuple[float, float, float]]

INTEGRAL_TOLERANCE = 1e-14
ENERGY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class CaseSpec:
    """A single Part I benchmark case."""

    case_id: str
    atoms: Sequence[Atom]
    state: str = "HF"
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0

    def __post_init__(self) -> None:
        if self.state.upper() not in ("HF", "CISD"):
            raise ValueError(f"state must be HF or CISD, got {self.state!r}")

    @property
    def geometry(self) -> str:
        return "; ".join(f"{s} {x} {y} {z}" for s, (x, y, z) in self.atoms)


@dataclass
class QubitHamiltonian:
    """Jordan-Wigner Hamiltonian together with the data needed to validate it."""

    operator: object  # openfermion.QubitOperator
    n_qubits: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    n_spatial_orbitals: int
    rhf_energy: float
    nuclear_repulsion: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_pauli_terms(self) -> int:
        return len(self.operator.terms)


def water_geometry(r_oh: float, angle_deg: float = 104.52) -> list[Atom]:
    """Water in the xz plane with the oxygen at the origin."""
    half_angle = math.radians(angle_deg / 2.0)
    dx, dz = r_oh * math.sin(half_angle), r_oh * math.cos(half_angle)
    return [("O", (0.0, 0.0, 0.0)), ("H", (dx, 0.0, dz)), ("H", (-dx, 0.0, dz))]


def _spin_orbital_tensors(
    one_body_mo: np.ndarray, eri_chemist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build OpenFermion spin-orbital one- and two-body tensors.

    ``eri_chemist[p, q, r, s]`` is ``(pq|rs)``.  The physicist-ordered tensor is
    ``<pq|rs> = (pr|qs)``, obtained by transposing to ``(0, 2, 3, 1)``.
    """
    n_spatial = one_body_mo.shape[0]
    n_spin = 2 * n_spatial
    one_body = np.zeros((n_spin, n_spin))
    two_body = np.zeros((n_spin,) * 4)

    for p in range(n_spatial):
        for q in range(n_spatial):
            value = one_body_mo[p, q]
            if abs(value) < INTEGRAL_TOLERANCE:
                continue
            for spin in (0, 1):
                one_body[2 * p + spin, 2 * q + spin] = value

    eri_physicist = np.asarray(eri_chemist.transpose(0, 2, 3, 1), order="C")
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    value = eri_physicist[p, q, r, s]
                    if abs(value) < INTEGRAL_TOLERANCE:
                        continue
                    half = value / 2.0
                    for spin_a in (0, 1):
                        for spin_b in (0, 1):
                            two_body[
                                2 * p + spin_a,
                                2 * q + spin_b,
                                2 * r + spin_b,
                                2 * s + spin_a,
                            ] = half
    return one_body, two_body


def build_qubit_hamiltonian(spec: CaseSpec) -> QubitHamiltonian:
    """Run RHF and return the Jordan-Wigner qubit Hamiltonian."""
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
    from pyscf import ao2mo, gto, scf

    molecule = gto.M(
        atom=spec.geometry,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    rhf_energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"RHF did not converge for {spec.case_id}")

    orbitals = mean_field.mo_coeff
    one_body_mo = orbitals.T @ mean_field.get_hcore() @ orbitals
    n_spatial = orbitals.shape[1]
    eri_chemist = ao2mo.restore(1, ao2mo.kernel(molecule, orbitals), n_spatial)

    one_body, two_body = _spin_orbital_tensors(one_body_mo, eri_chemist)
    nuclear_repulsion = float(molecule.energy_nuc())
    interaction = InteractionOperator(nuclear_repulsion, one_body, two_body)
    operator = jordan_wigner(get_fermion_operator(interaction))
    operator.compress(abs_tol=1e-12)

    n_electrons = int(molecule.nelectron)
    n_alpha = (n_electrons + int(molecule.spin)) // 2
    return QubitHamiltonian(
        operator=operator,
        n_qubits=2 * n_spatial,
        n_electrons=n_electrons,
        n_alpha=n_alpha,
        n_beta=n_electrons - n_alpha,
        n_spatial_orbitals=n_spatial,
        rhf_energy=rhf_energy,
        nuclear_repulsion=nuclear_repulsion,
        metadata={"basis": spec.basis, "mapping": "Jordan-Wigner", "frozen_core": False},
    )


def reference_fci_energy(spec: CaseSpec) -> float:
    """FCI energy from PySCF, used only as an independent check."""
    from pyscf import fci, gto, scf

    molecule = gto.M(
        atom=spec.geometry,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.kernel()
    return float(fci.FCI(mean_field).kernel()[0])


def validate_hamiltonian(
    spec: CaseSpec,
    hamiltonian: QubitHamiltonian,
    *,
    tolerance: float = ENERGY_TOLERANCE,
    check_fci: bool = True,
) -> dict:
    """Assert that the qubit Hamiltonian reproduces RHF and FCI energies.

    Raises :class:`ValueError` if either check fails.  Returns the measured
    energies so callers can record them.
    """
    from states import (
        expectation_without_matrix,
        hartree_fock_state,
        restricted_ground_state,
        sector_indices,
    )

    reference = hartree_fock_state(hamiltonian.n_qubits, hamiltonian.n_electrons)
    jw_hf_energy = expectation_without_matrix(hamiltonian, reference)
    report = {
        "rhf_energy_hartree": hamiltonian.rhf_energy,
        "jw_hf_energy_hartree": jw_hf_energy,
        "hf_energy_error": abs(jw_hf_energy - hamiltonian.rhf_energy),
    }
    if report["hf_energy_error"] > tolerance:
        raise ValueError(
            f"{spec.case_id}: Jordan-Wigner Hartree-Fock energy {jw_hf_energy:.12f} "
            f"does not match RHF energy {hamiltonian.rhf_energy:.12f} "
            f"(error {report['hf_energy_error']:.2e}). The qubit Hamiltonian or the "
            "computational-basis ordering is wrong."
        )

    if check_fci:
        indices = sector_indices(
            hamiltonian.n_qubits, hamiltonian.n_alpha, hamiltonian.n_beta
        )
        sector_energy, _ = restricted_ground_state(
            hamiltonian.operator, hamiltonian.n_qubits, indices
        )
        pyscf_fci = reference_fci_energy(spec)
        report["jw_fci_energy_hartree"] = sector_energy
        report["pyscf_fci_energy_hartree"] = pyscf_fci
        report["fci_energy_error"] = abs(sector_energy - pyscf_fci)
        if report["fci_energy_error"] > tolerance:
            raise ValueError(
                f"{spec.case_id}: lowest eigenvalue in the (N_alpha, N_beta) sector "
                f"{sector_energy:.12f} does not match the PySCF FCI energy "
                f"{pyscf_fci:.12f} (error {report['fci_energy_error']:.2e})."
            )
    return report
