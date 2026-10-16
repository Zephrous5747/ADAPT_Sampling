"""ADAPT trajectory states (Phase 2) and their gradient problems."""
from __future__ import annotations

import numpy as np
import pytest

from trajectory import AdaptSystem, problem_at, run_adapt


@pytest.fixture(scope="module")
def system():
    return AdaptSystem.build("H4_square_eq_side1p0_HF")


def test_sector_gradients_match_the_commutator_expansion(system, h4_problem):
    g = system.gradients(system.hf)
    assert np.allclose(g, system.problem.gradients, atol=1e-10)
    # The cached HF problem may use orbitals of opposite sign, never other magnitudes.
    assert np.allclose(np.abs(g), h4_problem.abs_gradients, atol=1e-10)


def test_parameter_gradient_matches_finite_differences(system):
    ops = [int(i) for i in np.argsort(-np.abs(system.gradients(system.hf)))[:3]]
    params = np.array([0.1, -0.2, 0.05])
    _, grad = system._energy_and_gradient(params, ops)
    h = 1e-6
    numeric = [(system.energy(system.state(ops, params + h * e)) - system.energy(system.state(ops, params - h * e)))
               / (2 * h) for e in np.eye(3)]
    assert np.allclose(grad, numeric, atol=1e-7)


def test_adapt_lowers_the_energy_and_states_become_problems(system):
    steps = run_adapt(system, max_iterations=3)
    energies = [s.energy for s in steps]
    assert all(b < a for a, b in zip(energies, energies[1:]))
    assert steps[0].selected == int(np.argmax(np.abs(steps[0].gradients)))
    last = steps[-1]
    psi = system.full_state(system.state(last.ops, last.params))
    problem = problem_at(system.problem, psi, "H4_square_eq_side1p0_ADAPT3", energy=last.energy,
                         sector_gradients=last.gradients)
    assert problem.state_name == "ADAPT3"
    # The pool gradient is the derivative for a new outermost exponential, so at the
    # optimum it vanishes for the generator added last (and not, in general, for
    # earlier ones, whose parameters act further inside the ansatz).
    assert abs(problem.gradients[last.ops[-1]]) < 1e-5
