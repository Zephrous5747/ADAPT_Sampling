"""Tests for the Part I measurement pipeline.

Run with::

    python -m pytest tests -q

The chemistry tests are the important ones: they pin the conventions that were
previously wrong (two-electron integral ordering and computational-basis
endianness) to energies that PySCF computes independently.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline_noshare
import finite_shot
import m1_fcug
import m2_baifcig
import m3_baifcug
from bai import elimination_schedule, survivors
from cases import get_case
from chemistry import build_qubit_hamiltonian, reference_fci_energy, validate_hamiltonian
from gradients import build_gradient_problem
from pauli_fc import greedy_fc_groups, pauli_commutes
from pauli_ops import PauliEvaluator, pauli_masks
from shot_models import allocate_context_shots, fragment_sum_shots, reconstruction_variances
from states import basis_index, hartree_fock_state

CASE_ID = "H4_square_eq_side1p0_HF"
TOLERANCE = 1e-8


@pytest.fixture(scope="module")
def problem():
    return build_gradient_problem(get_case(CASE_ID))


# --- Pauli algebra -----------------------------------------------------------


def test_pauli_masks_round_trip():
    mask = pauli_masks("XYZI")
    assert mask.n_y == 1
    assert mask.x_mask == 0b1100
    assert mask.z_mask == 0b0110


def test_pauli_expectation_matches_openfermion():
    from openfermion import QubitOperator
    from openfermion.linalg import get_sparse_operator

    rng = np.random.default_rng(7)
    state = rng.normal(size=2 ** 4) + 1j * rng.normal(size=2 ** 4)
    state /= np.linalg.norm(state)
    evaluator = PauliEvaluator(state)
    for pauli in ("XYZI", "IIZZ", "YYYY", "ZIXI"):
        term = tuple((i, c) for i, c in enumerate(pauli) if c != "I")
        matrix = get_sparse_operator(QubitOperator(term), n_qubits=4).tocsr()
        expected = complex(np.vdot(state, matrix @ state)).real
        assert evaluator.expectation(pauli) == pytest.approx(expected, abs=1e-12)


def test_greedy_groups_are_fully_commuting():
    paulis = ["XXII", "IIZZ", "ZZII", "XYXY", "YIYI", "IZIZ"]
    for group in greedy_fc_groups(paulis):
        for a in group:
            for b in group:
                assert pauli_commutes(a, b)


# --- Conventions (the previously broken part) --------------------------------


def test_hartree_fock_index_is_big_endian():
    # Occupying spin orbitals 0..3 of 8 sets the four most significant bits.
    assert basis_index(range(4), 8) == 0b11110000
    assert np.count_nonzero(hartree_fock_state(8, 4)) == 1


def test_hamiltonian_reproduces_rhf_and_fci():
    spec = get_case(CASE_ID)
    hamiltonian = build_qubit_hamiltonian(spec)
    report = validate_hamiltonian(spec, hamiltonian)
    assert report["hf_energy_error"] < TOLERANCE
    assert report["fci_energy_error"] < TOLERANCE
    assert report["jw_fci_energy_hartree"] == pytest.approx(
        reference_fci_energy(spec), abs=TOLERANCE
    )


def test_gradient_matches_energy_derivative(problem):
    """<[H, G]> must equal dE/dtheta of exp(theta G) applied to the state.

    The propagator is built with a dense ``expm`` so that the check does not
    depend on the adaptive error control of a sparse exponential.
    """
    from openfermion.linalg import get_sparse_operator
    from scipy.linalg import expm

    from pool import uccsd_pool

    hamiltonian = build_qubit_hamiltonian(get_case(CASE_ID))
    matrix = get_sparse_operator(
        hamiltonian.operator, n_qubits=hamiltonian.n_qubits
    ).tocsr()
    generators = uccsd_pool(hamiltonian.n_qubits, hamiltonian.n_electrons)
    best = problem.ranking()[0]
    assert generators[best].label == problem.labels[best]
    generator = get_sparse_operator(
        generators[best].qubit_operator, n_qubits=hamiltonian.n_qubits
    ).toarray()

    state = problem.evaluator.state
    step = 1e-4

    def energy(theta: float) -> float:
        evolved = expm(theta * generator) @ state
        return float(np.vdot(evolved, matrix @ evolved).real)

    derivative = (energy(step) - energy(-step)) / (2 * step)
    assert derivative == pytest.approx(problem.gradients[best], abs=1e-6)


def test_commutator_expansion_reproduces_gradient(problem):
    """Summing A_il * <R_l> must reproduce the directly computed gradient."""
    index = problem.ranking()[0]
    terms = problem.commutator_terms[index]
    reconstructed = sum(
        coefficient * problem.evaluator.expectation(pauli)
        for pauli, coefficient in terms.items()
    )
    assert reconstructed == pytest.approx(problem.gradients[index], abs=1e-10)


# --- Shot model --------------------------------------------------------------


def test_allocation_matches_closed_form_for_one_gradient():
    sigmas = np.array([[1.0, 2.0, 0.5]])
    shots = allocate_context_shots(sigmas, 0.01)
    assert shots.sum() == pytest.approx(fragment_sum_shots(sigmas[0], 0.01), rel=1e-9)


def test_allocation_is_feasible_and_not_worse_than_envelope():
    rng = np.random.default_rng(3)
    for _ in range(10):
        sigmas = rng.random((4, 7)) * (rng.random((4, 7)) < 0.6)
        if sigmas.sum(axis=1).min() == 0:
            continue
        epsilon = 0.05
        shots = allocate_context_shots(sigmas, epsilon)
        variances = reconstruction_variances(sigmas, shots)
        assert variances.max() <= epsilon ** 2 * (1 + 1e-8)
        envelope = sigmas.max(axis=0)
        assert shots.sum() <= (envelope.sum() / epsilon) ** 2 * (1 + 1e-8)


# --- BAI ---------------------------------------------------------------------


def test_elimination_keeps_the_true_winner():
    abs_gradients = np.array([0.30, 0.20, 0.05, 0.01])
    active = tuple(range(4))
    for round_result in elimination_schedule(abs_gradients):
        active = round_result.active_after
    assert active == (0,)


def test_survivors_are_within_two_radii_of_the_leader():
    abs_gradients = np.array([1.0, 0.9, 0.2])
    assert survivors(abs_gradients, [0, 1, 2], 0.1) == [0, 1]
    assert survivors(abs_gradients, [0, 1, 2], 0.5) == [0, 1, 2]


# --- Methods -----------------------------------------------------------------


def test_all_three_methods_select_the_true_winner(problem):
    winner = problem.ranking()[0]
    assert m2_baifcig.run(problem).remaining == (winner,)
    assert m3_baifcug.run(problem).remaining == (winner,)


def test_m1_meets_its_confidence_target(problem):
    result = m1_fcug.run(problem)
    assert result.summary()["constraint_satisfied"]
    assert result.total_shots > 0


def test_m1_and_m3_share_the_same_parent_contexts(problem):
    assert m1_fcug.run(problem).groups is m3_baifcug.run(problem).groups


def test_m3_never_pays_more_than_m1(problem):
    """M3 uses M1's contexts with elimination, so it cannot cost more.

    This holds only for the schedule-free trajectory: every M3 active set is a
    subset of M1's whole pool at a radius no tighter than M1's target.  A coarse
    geometric schedule overshoots the elimination points and can break it.
    """
    assert m3_baifcug.run(problem).total_shots <= m1_fcug.run(problem).total_shots


def test_m2_exact_matches_its_closed_form(problem):
    """The exact-limit M2 cost equals sum_i (z * sum_a sigma_ia / (Delta_i/2))^2."""
    from bai import final_radii
    from shot_models import z_from_delta

    result = m2_baifcig.run(problem)
    sigma_sums = np.array([s.sum() for s in result.sigmas])
    z = z_from_delta(0.05, problem.n_generators)
    closed_form = ((z * sigma_sums / final_radii(problem.abs_gradients)) ** 2).sum()
    assert result.total_shots == pytest.approx(math.ceil(closed_form), rel=1e-9)


@pytest.mark.parametrize("method", [m2_baifcig, m3_baifcug])
def test_exact_schedule_is_the_limit_of_the_geometric_one(problem, method):
    """A geometric schedule converges onto the exact limit from above."""
    exact = method.run(problem).total_shots
    coarse = method.run(problem, schedule="geometric", shrink=0.5).total_shots
    fine = method.run(problem, schedule="geometric", shrink=0.97).total_shots
    assert exact <= fine <= coarse
    assert fine == pytest.approx(exact, rel=0.05)


def test_m1_identification_target_is_four_times_cheaper_than_full_vector(problem):
    """gap/2 against gap/4 is a factor of four, since cost goes as 1/radius^2."""
    identification = m1_fcug.run(problem, target="identification").total_shots
    full_vector = m1_fcug.run(problem, target="full_vector").total_shots
    assert full_vector == pytest.approx(4 * identification, rel=1e-6)


# --- baseline and finite-shot ------------------------------------------------


def test_baseline_costs_at_least_as_much_as_m2(problem):
    """Same per-gradient groupings, but every arm resolved to the tightest radius.

    M2 resolves arm i only to Delta_i/2, which is never tighter than the gap/2 the
    baseline demands of every arm, so the baseline cannot be cheaper.
    """
    baseline = baseline_noshare.run(problem).total_shots
    assert baseline >= m2_baifcig.run(problem).total_shots


def test_baseline_uses_the_same_radius_as_m1(problem):
    """The two non-adaptive methods must be compared at one precision target."""
    assert baseline_noshare.run(problem).radius == pytest.approx(m1_fcug.run(problem).radius)


def test_finite_shot_is_reproducible_from_its_seed(problem):
    kwargs = dict(planning_bound=1.0, n_trials=20, seed=7)
    first = finite_shot.simulate(problem, "m3", **kwargs)
    second = finite_shot.simulate(problem, "m3", **kwargs)
    assert first.shots_mean == second.shots_mean
    assert first.correct_rate == second.correct_rate


@pytest.mark.parametrize("method", ["noshare", "m1"])
def test_non_adaptive_methods_spend_exactly_their_planning_bound(problem, method):
    """Without elimination there is no schedule to overshoot, so cost is the bound."""
    bound = (baseline_noshare if method == "noshare" else m1_fcug).run(problem).total_shots
    result = finite_shot.simulate(problem, method, planning_bound=bound, n_trials=5, seed=3)
    assert result.shots_mean == pytest.approx(bound, rel=1e-6)


@pytest.mark.parametrize("method", ["m2", "m3"])
def test_finite_shot_costs_at_least_the_planning_bound(problem, method):
    """A real run cannot beat a bound computed with knowledge it does not have."""
    module = m2_baifcig if method == "m2" else m3_baifcug
    bound = module.run(problem).total_shots
    result = finite_shot.simulate(problem, method, planning_bound=bound, n_trials=30, seed=5)
    assert result.shots_mean >= bound
    assert result.correct_rate >= 0.9


def test_context_covariances_reproduce_the_fragment_variances(problem):
    """The diagonal of each context covariance must be the fragment variance."""
    import numpy as np

    groups = problem.parent_fc_groups()
    matrices = finite_shot.context_covariances(problem, groups)
    assert matrices is not None
    diagonal = np.array([np.diag(c) for c in matrices]).T
    assert np.abs(diagonal - problem.parent_fragment_sigmas() ** 2).max() < 1e-12


def test_correlated_sampler_reproduces_the_marginal_variances(problem):
    """Correlated draws must still give each generator its own intended variance.

    This is the check that caught the original defect: a ridge added to make a
    Cholesky succeed injected variance into null directions, which the division by
    a small shot count then amplified without limit.
    """
    import numpy as np

    from shot_models import allocate_context_shots, epsilon_from_radius, z_from_delta

    sigmas = problem.parent_fragment_sigmas()
    n = problem.n_generators
    factors = finite_shot._noise_factors(
        finite_shot.context_covariances(problem, problem.parent_fc_groups())
    )
    z = z_from_delta(0.05, n)
    shots = allocate_context_shots(
        sigmas, epsilon_from_radius(float(problem.abs_gradients.max()), z)
    )
    used = shots > 0
    target = (sigmas[:, used] ** 2 / shots[used]).sum(axis=1)

    rng = np.random.default_rng(0)
    draws = np.zeros((3000, n))
    for trial in range(draws.shape[0]):
        column = np.zeros((n, int(used.sum())))
        for k, alpha in enumerate(np.flatnonzero(used)):
            column[:, k] = np.sqrt(shots[alpha]) * (factors[alpha] @ rng.standard_normal(n))
        draws[trial] = (column / shots[used]).sum(axis=1)
    ratio = draws.var(axis=0) / target
    assert 0.85 < ratio.min() and ratio.max() < 1.15


def test_context_covariances_are_positive_semidefinite():
    """Each context covariance must be a genuine covariance matrix.

    The polarisation identity Cov = (Var(A+B) - Var(A) - Var(B))/2 is exact in
    exact arithmetic but cancels catastrophically in the symmetry-forbidden
    fragments that dominate a parent context.  It used to return matrices with
    relative eigenvalues as low as -6e264 and correlations as large as 136,
    which fed negative pairwise variances into Eq. (12).
    """
    problem = build_gradient_problem(get_case("H4_square_stretch_side2p0_HF"))
    matrices = finite_shot.context_covariances(problem, problem.parent_fc_groups())
    assert matrices is not None
    for covariance in matrices:
        scale = max(abs(np.diag(covariance)).max(), 1e-300)
        eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        assert eigenvalues.min() / scale > -1e-12
        diagonal = np.diag(covariance)
        bound = np.sqrt(np.outer(diagonal, diagonal))
        off = ~np.eye(len(diagonal), dtype=bool)
        assert np.all(np.abs(covariance[off]) <= bound[off] * (1.0 + 1e-9))


def test_pairwise_variances_are_never_negative():
    """Eq. (12) applied to accumulated data must stay non-negative.

    ``_eliminate_pairwise`` clamps the variance at zero before taking a square
    root, so a negative value does not raise -- it silently produces a zero
    confidence radius and eliminates the arm on any lead at all.  That is how the
    true leader was being eliminated on H4 side 2.0.
    """
    for case in ("H4_square_stretch_side2p0_HF", "H4_square_eq_side1p0_HF"):
        problem = build_gradient_problem(get_case(case))
        matrices = finite_shot.context_covariances(problem, problem.parent_fc_groups())
        sigmas = problem.parent_fragment_sigmas()
        shots = allocate_context_shots(sigmas, 1e-3)
        used = np.flatnonzero(shots > 0)
        block = sum(matrices[alpha] / shots[alpha] for alpha in used)
        diagonal = np.diag(block)
        for signs in (np.ones(len(diagonal)), np.sign(problem.gradients)):
            relative = np.outer(signs, signs)
            pairwise = diagonal[:, None] + diagonal[None, :] - 2.0 * relative * block
            assert pairwise.min() > -1e-18 * max(diagonal.max(), 1e-300)


def test_fragment_variance_is_stable_for_near_eigenstate_fragments():
    """<F^2> - <F>^2 loses every digit when the state nearly diagonalises F."""
    problem = build_gradient_problem(get_case("H4_square_stretch_side2p0_HF"))
    evaluator = problem.evaluator
    groups = problem.parent_fc_groups()
    index = {pauli: a for a, group in enumerate(groups) for pauli in group}
    worst = 0.0
    for terms in problem.commutator_terms:
        buckets = {}
        for pauli, coefficient in terms.items():
            buckets.setdefault(index[pauli], {})[pauli] = coefficient
        for fragment in buckets.values():
            vector = evaluator.fragment_vector(fragment)
            mean = float(np.vdot(evaluator.state, vector).real)
            naive = float(np.vdot(vector, vector).real) - mean * mean
            stable = evaluator.fragment_std(fragment) ** 2
            if stable > 1e-24:
                worst = max(worst, abs(naive - stable) / stable)
            else:
                # the unstable form returns noise of either sign here
                assert stable >= 0.0
    assert worst < 1e-8


# --- Brute-force checks against dense linear algebra -------------------------


def _dense(terms, n_qubits):
    """Sparse matrix of sum_l c_l R_l, built independently of PauliEvaluator."""
    from openfermion import QubitOperator
    from openfermion.linalg import get_sparse_operator

    operator = QubitOperator()
    for pauli, coefficient in terms.items():
        operator += QubitOperator(
            tuple((q, ch) for q, ch in enumerate(pauli) if ch != "I"), coefficient
        )
    return get_sparse_operator(operator, n_qubits=n_qubits).tocsr()


@pytest.fixture(scope="module")
def dense_setup():
    from openfermion.linalg import get_sparse_operator

    from pool import uccsd_pool

    hamiltonian = build_qubit_hamiltonian(get_case(CASE_ID))
    n = hamiltonian.n_qubits
    h = get_sparse_operator(hamiltonian.operator, n_qubits=n).tocsr()
    generators = [
        get_sparse_operator(g.qubit_operator, n_qubits=n).tocsr()
        for g in uccsd_pool(n, hamiltonian.n_electrons)
    ]
    return n, h, generators


def test_every_commutator_expansion_matches_the_dense_commutator(problem, dense_setup):
    """C_i = sum_l A_il R_l must equal HG_i - G_iH as a matrix, for every generator."""
    n, h, generators = dense_setup
    assert len(generators) == problem.n_generators
    for i, g in enumerate(generators):
        exact = (h @ g - g @ h).toarray()
        rebuilt = _dense(problem.commutator_terms[i], n).toarray()
        assert np.abs(exact - rebuilt).max() < 1e-10, problem.labels[i]


def test_every_gradient_matches_its_energy_derivative(problem, dense_setup):
    """All gradients, not only the largest, against a central finite difference."""
    from scipy.linalg import expm

    n, h, generators = dense_setup
    state = problem.evaluator.state
    step = 1e-4
    for i, g in enumerate(generators):
        dense_g = g.toarray()
        plus = expm(step * dense_g) @ state
        minus = expm(-step * dense_g) @ state
        derivative = (np.vdot(plus, h @ plus).real - np.vdot(minus, h @ minus).real) / (2 * step)
        assert derivative == pytest.approx(problem.gradients[i], abs=1e-6), problem.labels[i]


def test_fragment_variances_and_covariances_match_dense_matrices(problem):
    """Evaluator variances and context covariances against dense fragment matrices."""
    n = problem.n_qubits
    state = problem.evaluator.state
    groups = problem.parent_fc_groups()
    sigmas = problem.parent_fragment_sigmas()
    matrices = finite_shot.context_covariances(problem, groups)
    index = {pauli: a for a, group in enumerate(groups) for pauli in group}
    centred: dict[tuple[int, int], np.ndarray] = {}
    for i, terms in enumerate(problem.commutator_terms):
        buckets = {}
        for pauli, coefficient in terms.items():
            buckets.setdefault(index[pauli], {})[pauli] = coefficient
        for alpha, fragment in buckets.items():
            vector = _dense(fragment, n) @ state
            centred[(i, alpha)] = vector - np.vdot(state, vector) * state
    for (i, alpha), c in centred.items():
        variance = float(np.vdot(c, c).real)
        assert sigmas[i, alpha] ** 2 == pytest.approx(variance, rel=1e-9, abs=1e-14)
    for (i, alpha), ci in centred.items():
        for (j, beta), cj in centred.items():
            if beta == alpha and j > i:
                covariance = float(np.vdot(ci, cj).real)
                assert matrices[alpha][i, j] == pytest.approx(covariance, rel=1e-9, abs=1e-14)


# --- Grouping validity ----------------------------------------------------------


def test_parent_contexts_partition_the_support_into_commuting_sets(problem):
    groups = problem.parent_fc_groups()
    flat = [pauli for group in groups for pauli in group]
    assert len(flat) == len(set(flat)), "a Pauli product appears in two contexts"
    assert set(flat) == set(problem.universal_support)
    for group in groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                assert pauli_commutes(group[a], group[b])


def test_individual_groupings_partition_each_commutator(problem):
    for terms, groups in zip(problem.commutator_terms, problem.individual_fc_groups()):
        flat = [pauli for group in groups for pauli in group]
        assert sorted(flat) == sorted(terms)
        for group in groups:
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    assert pauli_commutes(group[a], group[b])


# --- Phase invariance -------------------------------------------------------------


def test_costs_and_pairwise_variances_are_invariant_under_generator_sign_flips(problem):
    """An orbital phase flip negates a subset of generators; nothing reported may move.

    Conjugating by Z on one spin orbital maps G_i to -G_i for every generator that
    touches that orbital an odd number of times, and leaves H and the state
    physically unchanged.  So negating an arbitrary subset of commutators and their
    gradients is a superset of every phase convention, and the costs, the absolute
    gradients and the sign-corrected pairwise variances of Eq. (30) must be
    unchanged by it.
    """
    import dataclasses

    rng = np.random.default_rng(7)
    flips = np.where(rng.random(problem.n_generators) < 0.5, -1.0, 1.0)
    flipped = dataclasses.replace(
        problem,
        gradients=problem.gradients * flips,
        commutator_terms=[
            {pauli: s * c for pauli, c in terms.items()}
            for s, terms in zip(flips, problem.commutator_terms)
        ],
    )
    assert np.allclose(problem.abs_gradients, flipped.abs_gradients, atol=1e-15)
    for module in (m1_fcug, m2_baifcig, m3_baifcug, baseline_noshare):
        assert module.run(problem).total_shots == module.run(flipped).total_shots

    def pairwise(p):
        matrices = finite_shot.context_covariances(p, p.parent_fc_groups())
        shots = allocate_context_shots(p.parent_fragment_sigmas(), 1e-3)
        block = sum(matrices[a] / shots[a] for a in np.flatnonzero(shots > 0))
        d = np.diag(block)
        s = np.sign(p.gradients)
        return d[:, None] + d[None, :] - 2.0 * np.outer(s, s) * block

    before, after = pairwise(problem), pairwise(flipped)
    scale = np.abs(before).max()
    assert np.abs(before - after).max() < 1e-12 * scale


# --- The RHF solution is pinned ------------------------------------------------


@pytest.mark.parametrize(
    "case_id, energy",
    [
        ("H4_square_eq_side1p0_HF", -1.7610750541),
        ("H4_square_stretch_side2p0_HF", -1.5412552626),
    ],
)
def test_square_h4_uses_the_stable_rhf_solution(case_id, energy):
    """The default guess on square H4 at side 1.0 converges, in some numerical
    environments, to an internally unstable solution 66 mHa higher whose gradients
    differ by up to 0.25.  The build must follow the instability to the stable
    solution the published numbers use, whatever the BLAS build."""
    hamiltonian = build_qubit_hamiltonian(get_case(case_id))
    assert hamiltonian.rhf_energy == pytest.approx(energy, abs=1e-8)


# --- M3 accounting ------------------------------------------------------------------


def test_m3_bound_can_exceed_m1_when_two_breakpoints_nearly_coincide():
    """M3 <= M1 is not a theorem under the sum-of-maxima accounting of Eq. (14).

    Found by scripts/experiment_m3_vs_m1_adversarial.py and checked against a
    brute-force optimum of every allocation.  Two runners-up are nearly tied, so the
    last two breakpoints differ in radius by 0.2%, but dropping arm 3 moves about
    seventy shots from context A to context B; the per-context maxima then pay for
    both loadings.
    """
    from bai import elimination_thresholds

    absg = np.array([0.4683, 0.8566, 1.3082, 0.8555])
    sig = np.array([[0.3935, 0.0], [0.5714, 0.0], [0.6418, 2.2803], [2.2768, 0.5]])
    m1 = allocate_context_shots(sig, (1.3082 - 0.8566) / 2).sum()
    shots = np.zeros(2)
    for t in elimination_thresholds(absg):
        shots = np.maximum(shots, allocate_context_shots(sig[list(t.active)], t.radius))
    assert shots.sum() / m1 == pytest.approx(1.0914, abs=2e-3)


def test_actual_radius_m3_is_the_limit_of_the_noiseless_trial(monkeypatch):
    """run_actual_radii must agree with the finite-shot loop run without noise on
    a nearly continuous schedule; the trial overshoots each event by at most one
    schedule step, so it may only be slightly dearer."""
    problem = build_gradient_problem(get_case("H4_square_eq_side1p0_CISD"))
    exact = m3_baifcug.run_actual_radii(problem)["total_shots"]

    class ZeroRNG:
        def normal(self, loc=0.0, scale=1.0, size=None):
            a = np.asarray(scale, dtype=float)
            return np.zeros(a.shape if size is None else size)

        def standard_normal(self, size=None):
            return np.zeros(() if size is None else size)

    monkeypatch.setattr(finite_shot.np.random, "default_rng", lambda *a, **k: ZeroRNG())
    trial = finite_shot.simulate(problem, "m3", planning_bound=1.0, n_trials=1,
                                 shrink=0.99).shots_mean
    assert exact <= trial <= 1.03 * exact
