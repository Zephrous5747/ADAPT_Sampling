"""External baselines at shot level: static M1, independent-arm BAI (M2), sequential M1."""
from __future__ import annotations

import numpy as np
import pytest

from baselines import IndependentBAI, IndependentConfig, IndependentContexts, StaticM1, neyman_topup
from contexts import build_context_library
from design import DesignSet, build_fragment_problems
from learning import LearnedM3, LearningConfig
from sampler import OracleMoments
from states import hartree_fock_state  # Part I


def test_neyman_matches_closed_form_without_lower_bounds():
    sigma = np.array([1.0, 2.0, 0.5, 0.0])
    n = neyman_topup(sigma, 0.1, np.zeros(4))
    assert n[3] == 0
    assert np.sum(sigma[:3] ** 2 / n[:3]) == pytest.approx(0.01, rel=1e-9)
    assert n.sum() == pytest.approx(sigma.sum() ** 2 / 0.01, rel=1e-9)  # (sum sigma)^2 / eps^2


def test_neyman_topup_respects_lower_bounds_and_is_feasible():
    sigma = np.array([1.0, 2.0, 0.5])
    lower = np.array([50000.0, 0.0, 10.0])
    n = neyman_topup(sigma, 0.1, lower)
    assert (n >= lower - 1e-9).all()
    assert np.sum(sigma ** 2 / n) <= 0.01 * (1 + 1e-9)
    # Never worse than the unconstrained optimum plus the shots already held.
    assert n.sum() <= sigma.sum() ** 2 / 0.01 + lower.sum()


def test_neyman_returns_lower_when_already_feasible():
    n = neyman_topup(np.array([1.0, 1.0]), 1.0, np.array([10.0, 10.0]))
    assert np.array_equal(n, [10.0, 10.0])


@pytest.fixture(scope="module")
def independent(h4_cisd_problem):
    contexts = IndependentContexts(h4_cisd_problem)
    contexts.set_state(h4_cisd_problem.evaluator.state)
    return h4_cisd_problem, contexts


def test_independent_fragments_reproduce_every_gradient(independent):
    problem, contexts = independent
    assert np.allclose(contexts.exact_gradients(), problem.gradients, atol=1e-10)


def test_independent_group_count_matches_part1(independent):
    problem, contexts = independent
    assert contexts.n_contexts == sum(len(g) for g in problem.individual_fc_groups())


def test_independent_bound_dominates_exact_sd(independent):
    _, contexts = independent
    p, f = contexts.distributions, contexts.fvalues
    mean = (p * f).sum(axis=1)
    sd = np.sqrt(np.maximum((p * f * f).sum(axis=1) - mean ** 2, 0.0))
    assert (sd <= contexts.bound + 1e-12).all()


def test_independent_bai_selects_and_is_reproducible(independent):
    problem, contexts = independent
    model = IndependentBAI(problem, contexts, IndependentConfig(rule="safe", rho=0.1, shrink=0.8))
    first = [model.run(np.random.default_rng(3)) for _ in range(2)]
    again = [model.run(np.random.default_rng(3)) for _ in range(2)]
    assert [o.shots for o in first] == [o.shots for o in again]
    assert all(o.extra["shortfall"] <= 0.1 + 1e-12 for o in first)
    assert all(o.shots > 0 for o in first)


def test_static_m1_selects_best_arm_at_gap_radius(h4_cisd_problem):
    problem = h4_cisd_problem
    library = build_context_library(problem, "canonical")
    moments = OracleMoments(library, problem.evaluator.state)
    design = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
    model = StaticM1(problem, library, moments, design, radius=problem.top_gap() / 2.0)
    outcomes = [model.run(np.random.default_rng(i)) for i in range(20)]
    assert all(o.correct for o in outcomes)
    assert {o.shots for o in outcomes} == {float(model.planned.sum())}
    # The planned allocation reaches the radius for every arm.
    assert (model.z * model.standard_deviations <= model.radius * (1 + 1e-6)).all()


@pytest.fixture(scope="module")
def learner(h4_cisd_problem):
    problem = h4_cisd_problem
    library = build_context_library(problem, "mass")
    oracle = OracleMoments(library, problem.evaluator.state)
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    base = build_fragment_problems(problem, library, oracle, "II-0")
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    return problem, library, oracle, prior, base, coords


def test_sequential_m1_keeps_every_arm_in_the_allocation_and_costs_more(learner):
    problem, library, oracle, prior, base, coords = learner
    common = dict(level="II-0", prior="none", nu=0.0, rule="safe", start="bound", radius_min_shots=50, rho=0.1)
    on = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig(**common))
    off = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig(elimination="off", **common))
    assert "no-elimination" in off.config.label
    a = [on.run(np.random.default_rng(i)) for i in range(3)]
    b = [off.run(np.random.default_rng(i)) for i in range(3)]
    assert all(o.extra["shortfall"] <= 0.1 + 1e-12 for o in b)
    assert np.mean([o.shots for o in b]) > np.mean([o.shots for o in a])


def test_no_elimination_rejects_contrast_objective():
    with pytest.raises(ValueError):
        LearningConfig(elimination="off", objective="contrast", rule="safe")
