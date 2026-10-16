"""Reuse of the last energy evaluation: QWC groups, energy contexts, free shots."""
from __future__ import annotations

import numpy as np
import pytest

from contexts import build_context_library, qwc_groups
from design import build_fragment_problems
from learning import LearnedM3, LearningConfig
from reuse import ENERGY_ERROR, ReuseLibrary, hamiltonian_terms, reuse_fragment_problems
from sampler import OracleMoments
from states import hartree_fock_state  # Part I
from symplectic import anticommutes, masks_from_xz, xz_from_labels


def test_qwc_groups_are_qubit_wise_commuting_and_cover_everything():
    labels = ["XXII", "XIZI", "IYZZ", "ZZII", "IIZX", "YIII", "XXXX", "IIII"]
    groups = qwc_groups(labels)
    assert sorted(p for g in groups for p in g) == sorted(labels)
    for group in groups:
        for a in group:
            for b in group:
                assert all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def test_qwc_needs_at_least_as_many_groups_as_fc(h4_cisd_problem):
    assert len(qwc_groups(h4_cisd_problem.universal_support)) >= len(h4_cisd_problem.parent_fc_groups())


def test_qwc_contexts_are_product_measurements(h4_cisd_problem):
    """A QWC clique is measured in a product basis: no entangling gates, and only the Paulis of that basis."""
    problem = h4_cisd_problem
    terms = hamiltonian_terms(problem, problem.case_id)
    library = ReuseLibrary(problem, terms, "mass", "qwc").library
    assert (library.two_qubit_counts() == 0).all()
    for context in library.contexts[::7]:
        labels = [library.labels[m] for m in context.members[:60]]
        for a in labels:
            for b in labels:
                assert all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


@pytest.fixture(scope="module")
def reuse(h4_cisd_problem):
    problem = h4_cisd_problem
    terms = hamiltonian_terms(problem, problem.case_id)
    rl = ReuseLibrary(problem, terms, "mass", "fc")
    oracle = OracleMoments(rl.library, problem.evaluator.state)
    return problem, terms, rl, oracle


def test_hamiltonian_terms_reproduce_the_state_energy(reuse):
    problem, terms, _, _ = reuse
    from pauli_ops import PauliEvaluator

    assert abs(PauliEvaluator(problem.evaluator.state).fragment_mean_std(terms)[0]
               + 0.0 - (problem.state_energy - _identity(problem))) < 1e-7


def _identity(problem):
    from chemistry import build_qubit_hamiltonian  # Part I
    import part1_bridge
    from pauli_fc import openfermion_qubitop_to_dict

    ham = build_qubit_hamiltonian(part1_bridge.get_case(problem.case_id))
    return float(np.real(openfermion_qubitop_to_dict(ham.operator, ham.n_qubits)["I" * problem.n_qubits]))


def test_energy_contexts_are_appended_and_measure_their_groups(reuse):
    _, terms, rl, _ = reuse
    library = rl.library
    assert library.n_contexts == rl.n_parent + len(rl.energy)
    index = {label: i for i, label in enumerate(library.labels)}
    for k, group in enumerate(rl.energy):
        context = library.contexts[rl.n_parent + k]
        assert np.isin([index[p] for p in group], context.members).all()
    # parents keep their home Paulis: no required Pauli is homed in an energy context
    assert (library.home[: library.n_required] < rl.n_parent).all()


def test_credit_reaches_the_energy_error(reuse):
    problem, terms, rl, oracle = reuse
    credit = rl.credit(oracle, ENERGY_ERROR)
    assert (credit[: rl.n_parent] == 0).all() and credit[rl.n_parent:].sum() > 0
    index = {label: i for i, label in enumerate(rl.library.labels)}
    variance = 0.0
    for k, group in enumerate(rl.energy):
        ids = np.array([index[p] for p in group])
        x = np.array([terms[p] for p in group])
        n = credit[rl.n_parent + k]
        if n:
            variance += float(x @ oracle.covariance(rl.n_parent + k, ids) @ x) / n
    assert variance <= ENERGY_ERROR ** 2 * (1 + 1e-6)


def test_reuse_assignment_prefers_funded_energy_contexts_and_reconstructs(reuse):
    problem, _, rl, oracle = reuse
    credit = rl.credit(oracle)
    problems = reuse_fragment_problems(problem, rl, oracle, credit)
    owners = rl.library.contexts_of()
    moved = 0
    for p in problems:
        assert p.constraint_residual() < 1e-12  # one coordinate per Pauli: the target itself
        for pauli, coords in zip(p.pauli_ids, p.pauli_coords):
            energy = owners[pauli][owners[pauli] >= rl.n_parent]
            if energy.size:
                assert p.coord_ctx[coords[0]] >= rl.n_parent
                moved += 1
    assert moved > 0


def test_free_data_lowers_the_charged_cost_and_costs_nothing_when_absent(reuse):
    problem, _, rl, oracle = reuse
    credit = rl.credit(oracle)
    base = reuse_fragment_problems(problem, rl, oracle, credit)
    split = build_fragment_problems(problem, rl.library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    prior = OracleMoments(rl.library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    config = LearningConfig(level="II-0", prior="none", nu=0.0, elimination="off")
    free = LearnedM3(problem, rl.library, oracle, prior, base, coords, config, credit=credit)
    paid = LearnedM3(problem, rl.library, oracle, prior, base, coords, config)
    a = [free.run(np.random.default_rng(i)).shots for i in range(3)]
    b = [paid.run(np.random.default_rng(i)).shots for i in range(3)]
    assert np.mean(a) < np.mean(b)
    again = [free.run(np.random.default_rng(i)).shots for i in range(3)]
    assert a == again  # reproducible with the credit draw


def test_refits_are_counted_on_new_shots_not_on_the_credit(reuse):
    """With a large free credit the learner must still refit as shots accumulate (it once froze the design)."""
    problem, _, rl, oracle = reuse
    credit = rl.credit(oracle)
    base = reuse_fragment_problems(problem, rl, oracle, credit)
    split = build_fragment_problems(problem, rl.library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    prior = OracleMoments(rl.library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    config = LearningConfig(prior="none", nu=0.0)
    free = LearnedM3(problem, rl.library, oracle, prior, base, coords, config, credit=credit)
    paid = LearnedM3(problem, rl.library, oracle, prior, base, coords, config)
    refits_free = [free.run(np.random.default_rng(i)).extra["refits"] for i in range(2)]
    refits_paid = [paid.run(np.random.default_rng(i)).extra["refits"] for i in range(2)]
    assert min(refits_free) >= 2
    assert min(refits_free) >= 0.5 * min(refits_paid)
