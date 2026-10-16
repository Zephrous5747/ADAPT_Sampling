"""Other operator pools: qubit-ADAPT and qubit-excitation (QEB), built on the same case."""
from __future__ import annotations

import numpy as np
import pytest

import part1_bridge
from pools import POOL_BUILDERS, build_pool_problem, split_pool_case


def test_case_id_parsing():
    assert split_pool_case("LiH_R3p0_HF@qubit") == ("LiH_R3p0_HF", "qubit")
    assert split_pool_case("LiH_R3p0_HF") is None
    with pytest.raises(ValueError):
        split_pool_case("LiH_R3p0_HF@nope")


@pytest.mark.parametrize("pool", ["qubit", "qeb"])
def test_generators_are_anti_hermitian_and_conserve_number(pool):
    from openfermion import get_sparse_operator, hermitian_conjugated, number_operator

    n, ne = 6, 2
    generators = POOL_BUILDERS[pool](n, ne)
    assert generators
    number = get_sparse_operator(number_operator(n), n).toarray()
    for g in generators:
        dense = get_sparse_operator(g.qubit_operator, n).toarray()
        assert np.allclose(dense + dense.conj().T, 0.0, atol=1e-12)
        if pool == "qeb":  # a qubit excitation conserves the number of particles exactly
            assert np.allclose(dense @ number - number @ dense, 0.0, atol=1e-12)


def test_qubit_pool_is_single_pauli_strings_with_odd_y():
    for g in POOL_BUILDERS["qubit"](8, 4):
        from pauli_fc import openfermion_qubitop_to_dict

        terms = openfermion_qubitop_to_dict(g.qubit_operator, 8)
        assert len(terms) == 1
        (label,) = terms
        assert label.count("Y") % 2 == 1 and "Z" not in label


def test_qeb_pool_has_uccsd_index_sets_and_eight_term_doubles():
    from pauli_fc import openfermion_qubitop_to_dict
    from pool import uccsd_pool

    qeb, uccsd = POOL_BUILDERS["qeb"](8, 4), uccsd_pool(8, 4)
    assert [g.label for g in qeb] == [g.label for g in uccsd]
    sizes = {g.kind: len(openfermion_qubitop_to_dict(g.qubit_operator, 8)) for g in qeb}
    assert sizes == {"S": 2, "D": 8}


@pytest.mark.parametrize("pool", ["qubit", "qeb"])
def test_problem_gradients_match_matrix_algebra(pool):
    from chemistry import build_qubit_hamiltonian
    from openfermion import get_sparse_operator

    problem = build_pool_problem("H4_square_eq_side1p0_CISD", pool)
    assert problem.case_id == f"H4_square_eq_side1p0_CISD@{pool}"
    assert problem.metadata["pool_size"] == problem.n_generators > 10
    spec = part1_bridge.get_case("H4_square_eq_side1p0_CISD")
    ham = build_qubit_hamiltonian(spec)
    h = get_sparse_operator(ham.operator, ham.n_qubits).toarray()
    state = problem.evaluator.state
    generators = {g.label: g for g in POOL_BUILDERS[pool](ham.n_qubits, ham.n_electrons)}
    for i in sorted({0, problem.n_generators // 2, problem.n_generators - 1}):
        g = get_sparse_operator(generators[problem.labels[i]].qubit_operator, ham.n_qubits).toarray()
        expected = np.vdot(state, (h @ g - g @ h) @ state)
        assert abs(expected.imag) < 1e-9
        assert problem.gradients[i] == pytest.approx(expected.real, abs=1e-9)


@pytest.mark.parametrize("pool", ["gsd_qeb", "ceo"])
def test_generalised_pools_are_anti_hermitian_and_conserve_number_and_sz(pool):
    from openfermion import get_sparse_operator, number_operator, QubitOperator

    n = 6
    generators = POOL_BUILDERS[pool](n, 2)
    number = get_sparse_operator(number_operator(n), n).toarray()
    sz = get_sparse_operator(
        sum((0.5 * QubitOperator("") - 0.5 * QubitOperator(f"Z{q}")) * (1 if q % 2 == 0 else -1)
            for q in range(n)), n).toarray()
    for g in generators:
        dense = get_sparse_operator(g.qubit_operator, n).toarray()
        assert np.allclose(dense + dense.conj().T, 0.0, atol=1e-12)
        assert np.allclose(dense @ number - number @ dense, 0.0, atol=1e-12)
        assert np.allclose(dense @ sz - sz @ dense, 0.0, atol=1e-12)


def test_generalised_pool_sizes_on_four_spatial_orbitals():
    # alpha and beta orbitals: 4 each.  singles 2*C(4,2) = 12; opposite-spin sets C(4,2)^2 = 36;
    # same-spin sets 2*C(4,4) = 2.
    assert len(POOL_BUILDERS["gsd_qeb"](8, 4)) == 12 + 36 * 2 + 2 * 3
    assert len(POOL_BUILDERS["ceo"](8, 4)) == 12 + 36 * 2 + 2 * 6


def test_ceo_is_sum_and_difference_of_the_two_qes_on_an_opposite_spin_set():
    from openfermion import get_sparse_operator

    n = 4  # one opposite-spin set: alpha {0, 2}, beta {1, 3}
    qeb = {g.label: g for g in POOL_BUILDERS["gsd_qeb"](n, 2)}
    ceo = {g.label: g for g in POOL_BUILDERS["ceo"](n, 2)}
    e1, e2 = qeb["D 0,1 -> 2,3"], qeb["D 2,1 -> 0,3"]
    plus = ceo["C+ D 0,1 -> 2,3 + D 2,1 -> 0,3"]
    minus = ceo["C- D 0,1 -> 2,3 - D 2,1 -> 0,3"]
    d = lambda g: get_sparse_operator(g.qubit_operator, n).toarray()  # noqa: E731
    assert np.allclose(d(plus), d(e1) + d(e2)) and np.allclose(d(minus), d(e1) - d(e2))


def test_ceo_doubles_have_at_most_four_pauli_strings():
    from pauli_fc import openfermion_qubitop_to_dict

    sizes = {len(openfermion_qubitop_to_dict(g.qubit_operator, 8))
             for g in POOL_BUILDERS["ceo"](8, 4) if g.kind == "C"}
    assert sizes <= {4, 8} and 4 in sizes  # the paper: OVP-CEOs are sums of four Pauli strings (Eqs. 25-26)
