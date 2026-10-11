"""The dual (one unknown per Pauli) solves a generator's design problem like the primal (one per extra copy)."""
from __future__ import annotations

import numpy as np
import pytest

import design
from contexts import block_commuting_groups, build_context_library, contiguous_blocks
from design import build_fragment_problems
from sampler import OracleMoments


@pytest.fixture(scope="module")
def qwc(h4_cisd_problem):
    problem = h4_cisd_problem
    blocks = contiguous_blocks(problem.n_qubits, 1)
    library = build_context_library(problem, "canonical", groups=block_commuting_groups(problem.universal_support, blocks),
                                    blocks=blocks)
    oracle = OracleMoments(library, problem.evaluator.state)
    return library, build_fragment_problems(problem, library, oracle, "II-A")


def _shots(library, fraction, seed):
    rng = np.random.default_rng(seed)
    shots = np.zeros(library.n_contexts)
    pick = rng.random(library.n_contexts) < fraction
    shots[pick] = rng.integers(50, 20000, pick.sum())
    return shots


def _usable_variance(p, shots, x):
    block_shots = shots[p.ctx_ids]
    inverse = np.zeros(p.ctx_ids.size)
    inverse[block_shots > 0] = 1.0 / block_shots[block_shots > 0]
    return p._usable_variance(inverse, x)


@pytest.mark.parametrize("fraction", [0.25, 0.7])
def test_dual_reaches_the_primal_minimum(qwc, fraction):
    library, fragments = qwc
    shots = _shots(library, fraction, 5)
    compared = 0
    for p in fragments[::3]:
        home = p.x.copy()
        p.solver = "primal"
        p.optimise(shots)
        primal = p.x.copy()
        p.x = home.copy()
        p.solver = "dual"
        p.optimise(shots)
        dual = p.x.copy()
        p.x = home
        p.solver = "auto"
        if np.array_equal(primal, home):
            continue  # nothing to improve with these shots
        compared += 1
        vp, vd = _usable_variance(p, shots, primal), _usable_variance(p, shots, dual)
        assert vd <= vp * (1 + 1e-6)
        assert vd >= vp * (1 - 1e-6)  # both are minima of the same quadratic
        # the sum of the copies of every Pauli is the target, to rounding
        p.x = dual
        assert p.constraint_residual() < 1e-12
        p.x = home
    assert compared >= 3


def test_dual_leaves_unusable_and_single_copies_alone(qwc):
    library, fragments = qwc
    shots = _shots(library, 0.2, 9)
    for p in fragments[::5]:
        home = p.x.copy()
        p.solver = "dual"
        p.optimise(shots)
        usable_coord = (shots[p.ctx_ids] > 0)[p.coord_block]
        count = np.bincount(p._pauli_of[usable_coord], minlength=p.pauli_ids.size)
        movable = usable_coord & (count[p._pauli_of] >= 2)
        assert np.array_equal(p.x[~movable], home[~movable])
        p.x = home
        p.solver = "auto"


def test_auto_switches_on_size(qwc, monkeypatch):
    library, fragments = qwc
    shots = _shots(library, 0.5, 2)
    p = max(fragments, key=lambda q: q.n_coordinates)
    home = p.x.copy()
    p.solver = "dual"
    p.optimise(shots)
    dual = p.x.copy()
    p.x = home.copy()
    p.solver = "auto"
    monkeypatch.setattr(design, "DUAL_ABOVE", 0)  # every problem counts as large
    p.optimise(shots)
    assert np.array_equal(p.x, dual)
    p.x = home.copy()
    monkeypatch.setattr(design, "DUAL_ABOVE", 10 ** 9)  # none does: the primal
    p.optimise(shots)
    p.solver = "primal"
    q = p.x.copy()
    p.x = home.copy()
    p.optimise(shots)
    assert np.array_equal(p.x, q)


def test_primal_cost_is_the_size_of_the_structure_it_builds(qwc):
    library, fragments = qwc
    for seed, fraction in ((1, 0.15), (2, 0.6)):
        shots = _shots(library, fraction, seed)
        for p in fragments[::4]:
            usable = shots[p.ctx_ids] > 0
            variables, triplets = p.primal_cost(usable)
            structure = p._structure(usable)
            if structure is None:
                assert variables == 0
            else:
                assert variables == structure["n_vars"]
                assert triplets == structure["vals"].size
            p._structures.clear()
