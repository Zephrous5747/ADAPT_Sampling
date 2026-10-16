"""Design layer: allocation with committed shots, fragment designs, joint solver.

The checks follow Phase 0 of the Part II work order:
(a) ``A = BC`` holds exactly for every design produced;
(b) auxiliary coefficients cancel in every reconstructed gradient;
(c) analytic variances agree with an independent state-vector computation;
(d) II-A is recovered from II-B when the auxiliary library is empty;
(e) a hand-checkable ghost-Pauli example reproduces its analytic variance reduction.
The joint solver is checked against SLSQP on instances small enough for it.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize

from allocation import allocate, allocate_topup, reconstruction_variances
from contexts import build_context_library
from design import (
    DesignSet,
    FragmentProblem,
    assert_exact,
    build_fragment_problems,
    reconstruction_residual,
)
from part1_bridge import allocate_context_shots, gradient_matrix
from sampler import OracleMoments
from symplectic import commutes_with_all, masks_from_labels


# --- allocation ----------------------------------------------------------------------


def _random_sigmas(rng, arms=4, contexts=7):
    sigmas = rng.random((arms, contexts)) * (rng.random((arms, contexts)) < 0.6)
    sigmas[np.arange(arms), rng.integers(0, contexts, arms)] += 0.1
    return sigmas


def test_topup_without_commitments_matches_part1():
    rng = np.random.default_rng(1)
    for _ in range(10):
        sigmas = _random_sigmas(rng)
        reference = allocate_context_shots(sigmas, 0.1).sum()
        topped = allocate_topup(sigmas, 0.1, np.zeros(sigmas.shape[1])).sum()
        assert topped == pytest.approx(reference, rel=1e-5)


def test_topup_is_feasible_respects_commitments_and_matches_slsqp():
    rng = np.random.default_rng(2)
    for _ in range(8):
        sigmas = _random_sigmas(rng)
        free = allocate_context_shots(sigmas, 0.2)
        lower = free * rng.random(free.size) * 1.5
        shots = allocate_topup(sigmas, 0.2, lower)
        assert (shots >= lower - 1e-12).all()
        assert reconstruction_variances(sigmas ** 2, shots).max() <= 0.2 ** 2 * (1 + 1e-9)

        squared = sigmas ** 2
        constraints = [
            {"type": "ineq", "fun": lambda n, s=s: 0.04 - (s / n).sum()} for s in squared
        ]
        result = scipy.optimize.minimize(
            lambda n: n.sum(), np.maximum(shots * 1.1, lower + 1e-3), method="SLSQP",
            bounds=[(l, None) for l in lower + 1e-9], constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        assert shots.sum() <= result.fun * (1 + 1e-4)


# --- designs on H4 -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def h4_setup(h4_cisd_problem):
    library = build_context_library(h4_cisd_problem, "mass")
    moments = OracleMoments(library, h4_cisd_problem.evaluator.state)
    return h4_cisd_problem, library, moments


def test_level_zero_reproduces_part1_sigmas(h4_setup):
    problem, library, moments = h4_setup
    design = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
    assert np.allclose(design.sigmas(range(problem.n_generators)), problem.parent_fragment_sigmas(), atol=1e-10)


def test_split_designs_are_exact_measurable_and_independently_verified(h4_setup):
    problem, library, moments = h4_setup
    A = gradient_matrix(problem, library.n_library)
    design = DesignSet(build_fragment_problems(problem, library, moments, "II-A"), library.n_contexts)
    order = problem.ranking()
    solution = design.solve(order[:3] + [order[-1]], 1.0, max_outer=30)
    assert_exact(design, A)  # (a)
    evaluator = problem.evaluator
    for i in order[:3]:
        p = design.problems[i]
        second = p.context_second_moments()
        mean_total = 0.0
        for k, alpha in enumerate(p.ctx_ids):
            lo, hi = p.ctx_ptr[k], p.ctx_ptr[k + 1]
            terms = {library.labels[l]: v for l, v in zip(p.coord_pauli[lo:hi], p.x[lo:hi]) if v != 0.0}
            if not terms:
                continue
            xm, zm = masks_from_labels(list(terms))
            gx, gz = masks_from_labels(list(terms))
            assert commutes_with_all(xm, zm, gx, gz).all()  # jointly measurable
            mean, std = evaluator.fragment_mean_std(terms)
            mean_total += mean
            assert std ** 2 == pytest.approx(second[k], abs=1e-10)  # (c)
        assert mean_total == pytest.approx(problem.gradients[i], abs=1e-10)
    assert solution.total <= solution.history[0] * (1 + 1e-12)


def test_split_never_worse_than_level_zero_at_fixed_shots(h4_setup):
    problem, library, moments = h4_setup
    base = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
    split = DesignSet(build_fragment_problems(problem, library, moments, "II-A"), library.n_contexts)
    shots = allocate(base.sigmas(range(problem.n_generators)), 1.0)
    for i in range(problem.n_generators):
        before = base.problems[i].variance(shots)
        after = split.problems[i].optimise(shots)
        assert after <= before * (1 + 1e-9) + 1e-15


def test_auxiliaries_cancel_and_empty_library_recovers_level_a(h4_setup):
    problem, library, moments = h4_setup
    A = gradient_matrix(problem, library.n_library)
    level_a = build_fragment_problems(problem, library, moments, "II-A")
    level_b_empty = build_fragment_problems(problem, library, moments, "II-B", aux_cap=0)
    for pa, pb in zip(level_a, level_b_empty):  # (d)
        assert np.array_equal(pa.coord_ctx, pb.coord_ctx)
        assert np.array_equal(pa.coord_pauli, pb.coord_pauli)
    level_b = build_fragment_problems(problem, library, moments, "II-B", aux_cap=40)
    design = DesignSet(level_b, library.n_contexts)
    winner = problem.ranking()[0]
    design.solve([winner], 1.0, max_outer=30)
    assert_exact(design, A)  # (b): auxiliary columns of A are zero, so BC must cancel them
    p = design.problems[winner]
    aux = [k for k, pauli in enumerate(p.pauli_ids) if p.pauli_target[k] == 0.0]
    assert aux, "expected auxiliary candidates for the winner"
    assert max(abs(p.x[p.pauli_coords[k]].sum()) for k in aux) < 1e-12


def test_parity_is_useless_as_a_ghost_on_a_symmetry_eigenstate(h4_cisd_problem):
    """An operator the state is an eigenstate of has zero covariance with everything."""
    parity = "Z" * h4_cisd_problem.n_qubits
    library = build_context_library(h4_cisd_problem, "canonical", auxiliary_labels=[parity])
    moments = OracleMoments(library, h4_cisd_problem.evaluator.state)
    index = library.labels.index(parity)
    measuring = library.contexts_of()[index]
    assert measuring.size > 0
    for alpha in measuring:
        members = library.contexts[alpha].members
        covariance = moments.covariance(int(alpha), members)
        row = covariance[list(members).index(index)]
        assert np.abs(row).max() < 1e-12


# --- the ghost example of the work order (Phase 0 (e)) ----------------------------------------


def test_two_context_ghost_matches_the_analytic_reduction():
    """Target ``D = P0`` measured only in context 0; ghost ``Q`` in both contexts.

    With ``F_0 = P0 + t Q`` and ``F_1 = -t Q`` the variance is
    ``(V0 + 2 t C + t**2 VQ) / n0 + t**2 VQ / n1``, minimised at
    ``t* = -C n1 / (VQ (n0 + n1))`` with reduction ``C**2 n1 / (n0 VQ (n0 + n1))``.
    """
    V0, VQ, C = 1.0, 0.8, 0.6
    blocks = {0: np.array([[V0, C], [C, VQ]]), 1: np.array([[VQ]])}
    paulis = {0: [0, 1], 1: [1]}

    def covariance(alpha, ids):
        position = [paulis[alpha].index(int(i)) for i in ids]
        return blocks[alpha][np.ix_(position, position)]

    home = np.array([0, -1])
    problem = FragmentProblem(0, np.array([0, 0, 1]), np.array([0, 1, 1]), {0: 1.0}, home, covariance)
    shots = np.array([3.0, 2.0])
    before = problem.variance(shots)
    after = problem.optimise(shots)
    n0, n1 = shots
    assert before == pytest.approx(V0 / n0)
    assert before - after == pytest.approx(C ** 2 * n1 / (n0 * VQ * (n0 + n1)), rel=1e-9)
    t = problem.x[(problem.coord_ctx == 0) & (problem.coord_pauli == 1)][0]
    assert t == pytest.approx(-C * n1 / (VQ * (n0 + n1)), rel=1e-9)


# --- the joint solver against independent solvers ---------------------------------------------------


def test_single_arm_solve_matches_an_independent_smoothed_minimisation(h4_problem):
    """One arm: the optimum is ``min_x (sum_alpha ||x_alpha||_Sigma)**2``, a sum of norms.

    Solved independently by L-BFGS on its smoothed form with a continuation to
    ``delta = 1e-8``; ICS (alternating) is known to stall above it.
    """
    library = build_context_library(h4_problem, "canonical")
    moments = OracleMoments(library, h4_problem.evaluator.state)
    design = DesignSet(build_fragment_problems(h4_problem, library, moments, "II-A"), library.n_contexts)
    arm = h4_problem.ranking()[0]
    p = design.problems[arm]
    structure = p._structure(np.ones(p.ctx_ids.size, dtype=bool))
    N, B, x0, starts = structure["N"], p._block_matrix, p.x.copy(), p.ctx_ptr[:-1]

    def smoothed(y, delta):
        x = x0 + N @ y
        Bx = B @ x
        root = np.sqrt(np.maximum(np.add.reduceat(x * Bx, starts), 0) + delta ** 2)
        return root.sum(), N.T @ (Bx * (1.0 / root)[p.coord_block])

    y = np.zeros(N.shape[1])
    for delta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        y = scipy.optimize.minimize(smoothed, y, args=(delta,), jac=True, method="L-BFGS-B",
                                    options={"maxiter": 20000, "maxcor": 50, "ftol": 1e-15, "gtol": 1e-12}).x
    x = x0 + N @ y
    reference = np.sqrt(np.maximum(np.add.reduceat(x * (B @ x), starts), 0)).sum() ** 2

    ours = design.solve([arm], 1.0).total
    assert ours == pytest.approx(reference, rel=1e-4)
    design.reset()
    stalled = design.solve_ics([arm], 1.0, max_iter=300, rtol=0.0).total
    assert stalled > reference * 1.02  # the documented failure of alternating ICS


# --- the joint solver against SLSQP -------------------------------------------------------------


def _tiny_instance(seed):
    rng = np.random.default_rng(seed)
    n_contexts, n_paulis = 3, 5
    measured = {0: [0, 1, 2], 1: [1, 2, 3], 2: [2, 3, 4]}
    blocks = {}
    for alpha, ids in measured.items():
        factor = rng.normal(size=(len(ids), len(ids) + 1))
        blocks[alpha] = factor @ factor.T / len(ids)

    def covariance(alpha, ids):
        position = [measured[alpha].index(int(i)) for i in ids]
        return blocks[alpha][np.ix_(position, position)]

    home = np.array([0, 0, 1, 2, 2])
    owners = {l: [a for a in range(n_contexts) if l in measured[a]] for l in range(n_paulis)}
    targets = [{0: 1.0, 1: -0.5, 2: 0.7}, {2: 0.4, 3: 1.2, 4: -0.3}]
    problems = []
    for i, target in enumerate(targets):
        ctx = np.concatenate([owners[l] for l in target])
        pauli = np.repeat(list(target), [len(owners[l]) for l in target])
        problems.append(FragmentProblem(i, ctx, pauli, target, home, covariance))
    return problems, n_contexts


def _slsqp_joint(problems, n_contexts):
    sizes = [p.n_coordinates for p in problems]
    offsets = np.cumsum([0] + sizes)

    def unpack(v):
        return [v[offsets[k] : offsets[k + 1]] for k in range(len(problems))], v[offsets[-1] :]

    constraints = []
    for k, p in enumerate(problems):
        for coords, target in zip(p.pauli_coords, p.pauli_target):
            constraints.append({"type": "eq", "fun": lambda v, k=k, c=coords, t=target: unpack(v)[0][k][c].sum() - t})
        constraints.append({"type": "ineq", "fun": lambda v, k=k, p=p: 1.0 - p.variance(unpack(v)[1], unpack(v)[0][k])})
    start = np.concatenate([p.x for p in problems] + [np.full(n_contexts, 5.0)])
    result = scipy.optimize.minimize(
        lambda v: unpack(v)[1].sum(), start, method="SLSQP", constraints=constraints,
        bounds=[(None, None)] * offsets[-1] + [(1e-9, None)] * n_contexts,
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    return result.fun


@pytest.mark.parametrize("seed", range(4))
def test_joint_solver_matches_slsqp_on_tiny_instances(seed):
    problems, n_contexts = _tiny_instance(seed)
    reference = _slsqp_joint(problems, n_contexts)
    problems, n_contexts = _tiny_instance(seed)
    solution = DesignSet(problems, n_contexts).solve([0, 1], 1.0)
    assert solution.total == pytest.approx(reference, rel=2e-4)
