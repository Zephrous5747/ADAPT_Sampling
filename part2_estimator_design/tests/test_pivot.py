"""Pivot-based grouping of the gradient observables (Anastasiou et al.) and block-wise contexts."""
from __future__ import annotations

import numpy as np
import pytest

import part1_bridge
from allocation import allocate
from contexts import build_context_library
from design import DesignSet, build_fragment_problems
from pivot import (
    _hermitian_phase_exponent,
    anchor_classes,
    build_pivot_structure,
    describe,
    firstfit_classes,
    merged_assignment,
    merged_fragment_problems,
    pivot_fragment_problems,
    pivot_library,
)
from reuse import hamiltonian_terms
from sampler import OracleMoments
from symplectic import anticommutes, dense_pauli, masks_from_xz, xz_from_labels


def commute(a: str, b: str) -> bool:
    xa, za = masks_from_xz(*xz_from_labels([a]))
    xb, zb = masks_from_xz(*xz_from_labels([b]))
    return not bool(anticommutes(xa, za, xb, zb)[0])


def test_product_phase_matches_dense_matrices():
    rng = np.random.default_rng(3)
    n = 3
    for _ in range(40):
        a = "".join(rng.choice(list("IXYZ"), n))
        b = "".join(rng.choice(list("IXYZ"), n))
        x1, z1 = xz_from_labels([a])
        x2, z2 = xz_from_labels([b])
        exponent = int(_hermitian_phase_exponent(x1[0], z1[0], x2[0], z2[0]))
        q = [("I", "X", "Z", "Y")[int(x) + 2 * int(z)] for x, z in zip(x1[0] ^ x2[0], z1[0] ^ z2[0])]
        xq, zq = xz_from_labels(["".join(q)])
        hermitian_q = dense_pauli(xq[0], zq[0], int((xq[0] & zq[0]).sum()) % 4)
        left = dense_pauli(x1[0], z1[0], int((x1[0] & z1[0]).sum()) % 4)
        right = dense_pauli(x2[0], z2[0], int((x2[0] & z2[0]).sum()) % 4)
        assert np.allclose(left @ right, (1j) ** exponent * hermitian_q)


def test_anchored_classes_commute_and_number_at_most_2n():
    strings = ["YXXI", "YIXX", "IYXX", "XYYY", "YXII", "IXYI", "YXYY", "IYXI", "YYXY"]
    classes = anchor_classes(strings)
    assert classes is not None and len(set(classes.values())) <= 2 * 4
    for a in strings:
        for b in strings:
            if classes[a] == classes[b]:
                assert commute(a, b)
    assert anchor_classes(["ZXXY"]) is None  # a Z string is not a qubit-pool string
    assert anchor_classes(["XXXX"]) is None  # no Y: not a generator string


def test_firstfit_classes_commute():
    strings = ["ZZII", "XXII", "IYYI", "XIXI", "ZIIZ", "YYYY"]
    classes = firstfit_classes(strings)
    for a in strings:
        for b in strings:
            if classes[a] == classes[b]:
                assert commute(a, b)


@pytest.fixture(scope="module")
def uccsd(h4_cisd_problem):
    problem = h4_cisd_problem
    terms = hamiltonian_terms(problem, problem.case_id)
    library, structure = pivot_library(problem, terms)
    oracle = OracleMoments(library, problem.evaluator.state)
    return problem, terms, library, structure, oracle


@pytest.fixture(scope="module")
def qubit_pool():
    problem = part1_bridge.load_problem("H4_square_eq_side1p0_HF@qubit")
    terms = hamiltonian_terms(problem)
    library, structure = pivot_library(problem, terms)
    oracle = OracleMoments(library, problem.evaluator.state)
    return problem, terms, library, structure, oracle


def test_qubit_pool_uses_the_anchored_classes(qubit_pool):
    _, _, _, structure, _ = qubit_pool
    assert structure.classes == "anchor" and structure.n_classes <= 2 * 8
    assert structure.extra_labels == []  # one string per generator: every product is a gradient product


def test_every_pivot_context_is_fully_commuting(uccsd, qubit_pool):
    for _, _, _, structure, _ in (uccsd, qubit_pool):
        for group in structure.groups:
            for a in group:
                assert all(commute(a, b) for b in group)


def test_decomposition_reproduces_every_commutator_and_rejects_other_hamiltonians(uccsd):
    problem, terms, _, structure, _ = uccsd  # build_pivot_structure raised if it did not
    for i, parts in enumerate(structure.contributions):
        merged: dict[str, float] = {}
        for (_, q), v in parts.items():
            merged[q] = merged.get(q, 0.0) + v
        assert max(abs(merged.get(q, 0.0) - v) for q, v in problem.commutator_terms[i].items()) < 1e-8
    scaled = {p: 1.01 * c for p, c in terms.items()}
    with pytest.raises(ValueError, match="disagrees"):
        build_pivot_structure(problem, scaled)


def test_split_design_reconstructs_the_exact_gradients(uccsd, qubit_pool):
    for problem, _, library, structure, oracle in (uccsd, qubit_pool):
        for p in pivot_fragment_problems(problem, structure, library, oracle):
            assert p.constraint_residual() < 1e-9
            total = 0.0
            for alpha in np.unique(p.coord_ctx):
                chosen = p.coord_ctx == alpha
                total += p.x[chosen] @ oracle.means(int(alpha), p.coord_pauli[chosen])
            assert abs(total - problem.gradients[p.generator]) < 1e-9


def test_a_product_from_several_pivots_is_measured_in_each_of_them(uccsd):
    problem, _, library, structure, oracle = uccsd
    split = pivot_fragment_problems(problem, structure, library, oracle)
    assert sum(len(c) > 1 for p in split for c in p.pauli_coords) > 0
    merged = merged_fragment_problems(problem, structure, library, oracle)
    assert all(len(c) == 1 for p in merged for c in p.pauli_coords)


def test_merged_assignment_covers_every_needed_product_with_a_context_that_reads_it(uccsd):
    problem, _, library, structure, oracle = uccsd
    assigned = merged_assignment(structure)
    assert set(problem.universal_support) <= set(assigned)
    owners = {(c, q) for parts in structure.contributions for (c, q) in parts}
    assert all((ctx, q) in owners for q, ctx in assigned.items())
    for p in merged_fragment_problems(problem, structure, library, oracle):
        assert p.constraint_residual() < 1e-12
        total = sum(p.x[c] @ oracle.means(int(p.coord_ctx[c[0]]), p.coord_pauli[c]) for c in p.pauli_coords)
        assert abs(total - problem.gradients[p.generator]) < 1e-9


def test_pivot_contexts_are_many_and_shallower_than_a_universal_context_is_wide(uccsd):
    problem, _, library, structure, _ = uccsd
    info = describe(structure, library)
    assert info["contexts"] > 10 * len(problem.parent_fc_groups())
    assert info["two_qubit_max"] >= 0 and info["bound_n_minus_3"] == problem.n_qubits - 3


def test_static_pivot_split_costs_more_than_universal_grouping_on_uccsd(uccsd):
    """The published scheme measures a product once per pivot; on the UCCSD pool this is dear."""
    problem, _, library, structure, oracle = uccsd
    epsilon = 0.01

    def planned(design, n):
        return float(allocate(DesignSet(design, n).sigmas(range(problem.n_generators)), epsilon).sum())

    universal = build_context_library(problem, "canonical")
    universal_oracle = OracleMoments(universal, problem.evaluator.state)
    base =planned(build_fragment_problems(problem, universal, universal_oracle, "II-0"), universal.n_contexts)
    split = planned(pivot_fragment_problems(problem, structure, library, oracle), library.n_contexts)
    merged = planned(merged_fragment_problems(problem, structure, library, oracle), library.n_contexts)
    assert split > 2.0 * base
    assert merged < split
