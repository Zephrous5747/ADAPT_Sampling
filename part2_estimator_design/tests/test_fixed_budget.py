"""The uncertified fixed-budget selection (the practical default of ADAPT-VQE)."""
from __future__ import annotations

import numpy as np
import pytest

from baselines import FixedBudgetM1, FixedBudgetPilotM1, FixedBudgetSpec
from contexts import build_context_library
from design import DesignSet, build_fragment_problems
from sampler import OracleMoments


@pytest.fixture(scope="module")
def setup(h4_cisd_problem):
    problem = h4_cisd_problem
    library = build_context_library(problem, "canonical")
    moments = OracleMoments(library, problem.evaluator.state)
    design = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
    return problem, library, moments, design


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        FixedBudgetSpec(0)
    with pytest.raises(ValueError):
        FixedBudgetSpec(100, "greedy")


@pytest.mark.parametrize("allocation", ["designed", "uniform"])
def test_budget_is_spent_up_to_rounding_and_does_not_depend_on_the_draw(setup, allocation):
    problem, library, moments, design = setup
    spec = FixedBudgetSpec(20_000, allocation)
    model = FixedBudgetM1(problem, library, moments, design, spec)
    outcomes = [model.run(np.random.default_rng(i)) for i in range(5)]
    assert {o.shots for o in outcomes} == {float(model.planned.sum())}
    # ceilings and the one-shot minimum of every funded context add at most about one shot per context
    assert 20_000 <= model.planned.sum() <= 20_000 + 2 * library.n_contexts
    assert all(o.rounds == 1 and not o.extra["stopped_rho"] for o in outcomes)


def test_large_budget_finds_the_best_arm_and_small_one_does_not_always(setup):
    problem, library, moments, design = setup
    big = FixedBudgetM1(problem, library, moments, design, FixedBudgetSpec(10_000_000))
    assert all(big.run(np.random.default_rng(i)).correct for i in range(20))
    small = FixedBudgetM1(problem, library, moments, design, FixedBudgetSpec(50, "uniform"))
    shortfalls = [small.run(np.random.default_rng(i)).extra["shortfall"] for i in range(40)]
    assert max(shortfalls) > 0  # an uncertified, thin selection can pick a worse generator


def test_designed_allocation_beats_uniform_in_variance(setup):
    problem, library, moments, design = setup
    designed = FixedBudgetM1(problem, library, moments, design, FixedBudgetSpec(100_000, "designed"))
    uniform = FixedBudgetM1(problem, library, moments, design, FixedBudgetSpec(100_000, "uniform"))
    # the worst standard deviation over all gradients is what the minimax allocation minimises
    assert designed.standard_deviations.max() <= uniform.standard_deviations.max() * (1 + 1e-9)


def test_pilot_allocation_uses_no_exact_variance_and_spends_the_budget(setup):
    problem, library, moments, design = setup
    spec = FixedBudgetSpec(200_000, "pilot")
    model = FixedBudgetPilotM1(problem, library, moments, design, spec)
    outcomes = [model.run(np.random.default_rng(i)) for i in range(4)]
    # the budget is spent up to the ceilings of the allocation, and the design is left with its exact covariance model
    assert all(200_000 <= o.shots <= 200_000 + 3 * library.n_contexts for o in outcomes)
    assert np.allclose(design.sigmas([0, 1, 2]), model.sigmas[:3])
    assert all(o.rounds == 1 for o in outcomes)


def test_pilot_allocation_finds_the_best_arm_with_a_large_budget(setup):
    problem, library, moments, design = setup
    model = FixedBudgetPilotM1(problem, library, moments, design, FixedBudgetSpec(20_000_000, "pilot"))
    assert all(model.run(np.random.default_rng(i)).correct for i in range(10))
