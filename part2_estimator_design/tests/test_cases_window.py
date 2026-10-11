"""The H2O orbital-window cases (cc-pVDZ, 8 orbitals): energies against the colleagues' archive, and Part I's validation gate."""
import pytest

import part1_bridge  # noqa: F401  (registers the window cases and patches Part I's chemistry module)
from cases_window import WindowSpec, window_problem

# archive (larger-basis note, Tables 2 and 3): RHF and window-FCI energies of H2O cc-pVDZ 8o
ARCHIVE = {"eq": (-76.026741428, -76.059281484), "str": (-75.669624909, -75.826835961)}


@pytest.mark.parametrize("tag", ["eq", "str"])
def test_window_energies_match_archive(tag):
    w = window_problem("dz8o", tag)
    assert w["norb"] == 8 and w["nelec"] == 8
    assert w["e_hf"] == pytest.approx(ARCHIVE[tag][0], abs=1e-7)
    assert w["e_fci"] == pytest.approx(ARCHIVE[tag][1], abs=1e-7)


def test_part1_validation_gate_accepts_window():
    from chemistry import build_qubit_hamiltonian, validate_hamiltonian
    from cases import get_case

    spec = get_case("H2O_dz8o_eq_HF")
    assert isinstance(spec, WindowSpec)
    hamiltonian = build_qubit_hamiltonian(spec)
    report = validate_hamiltonian(spec, hamiltonian)  # HF energy and sector FCI energy of the qubit operator
    assert report["hf_energy_error"] < 1e-8 and report["fci_energy_error"] < 1e-8
    assert hamiltonian.n_qubits == 16 and hamiltonian.n_alpha == 4


def test_cache_key_depends_on_window_and_geometry():
    from cases import get_case
    from problem_cache import cache_key

    keys = {cache_key(get_case(f"H2O_dz8o_{g}_HF")) for g in ("eq", "str")} | {cache_key(get_case("H2O_dz11o_eq_HF"))}
    assert len(keys) == 3
