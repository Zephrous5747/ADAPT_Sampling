"""Classical priors for measurement design: the sparse state, the full-support mixing, the moment update."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from classical_prior import MixedMoments, apply_rotation, sparse_state, sparse_state_for_case, truncate
from contexts import build_context_library
from design import build_fragment_problems
from learning import CovarianceModel, LearnedM3, LearningConfig
from sampler import OracleMoments
from states import hartree_fock_state  # Part I


# --- the sparse state ------------------------------------------------------------------------------


def test_truncate_keeps_the_largest_weights_with_index_ties_and_normalises():
    v = np.array([0.5, -0.5, 0.5, 0.5, 0.1])
    out = truncate(v, 2)
    assert np.count_nonzero(out) == 2 and out[0] != 0 and out[1] != 0  # the tie among four equal weights goes to the lowest indices
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out[1] < 0 < out[0]  # phases are retained
    assert np.allclose(truncate(v, 5), v / np.linalg.norm(v))


def _excitation(n: int, i: int, j: int) -> sp.csr_matrix:
    g = sp.lil_matrix((n, n))
    g[j, i], g[i, j] = 1.0, -1.0  # a single excitation minus de-excitation: G^3 = -G
    return g.tocsr()


def test_rotation_matches_the_matrix_exponential_and_falls_back_off_the_two_level_form():
    n = 6
    rng = np.random.default_rng(0)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    g = _excitation(n, 1, 4)
    assert np.allclose(apply_rotation(g, 0.7, v), expm_multiply(0.7 * g, v))
    both = (_excitation(n, 0, 2) + _excitation(n, 2, 5)).tocsr()  # G^3 != -G: the exponential is used instead
    assert np.allclose(apply_rotation(both, 0.7, v), expm_multiply(0.7 * both, v))


def test_sparse_state_is_exact_when_nothing_is_truncated_and_obeys_the_cap():
    n = 8
    gens = [_excitation(n, 0, 3), _excitation(n, 3, 5), _excitation(n, 0, 6), _excitation(n, 5, 7)]
    hf = np.zeros(n)
    hf[0] = 1.0
    ops, params = [0, 1, 2, 3], [0.4, -0.7, 0.2, 0.5]
    full = hf.astype(complex)
    for o, t in zip(ops, params):
        full = expm_multiply(t * gens[o], full)
    assert np.allclose(sparse_state(gens, ops, params, hf, keep=n), full)
    c = sparse_state(gens, ops, params, hf)  # K = m + 1 = 5
    assert np.count_nonzero(c) <= 5 and np.linalg.norm(c) == pytest.approx(1.0)
    assert sparse_state(gens, [], [], hf)[0] == 1.0  # m = 0: the HF determinant


def test_saved_adapt_state_gives_a_close_sparse_state_and_reproduces_the_trajectory():
    full, info = sparse_state_for_case("H4_square_eq_side1p0_ADAPT3")  # also checks that the pool rebuilds the saved state
    assert info["m"] == 3 and info["K"] <= 4
    assert 0.9 < info["overlap"] <= 1.0 + 1e-12
    assert np.linalg.norm(full) == pytest.approx(1.0)
    again, _ = sparse_state_for_case("H4_square_eq_side1p0_ADAPT3")
    assert np.array_equal(full, again)
    with pytest.raises(ValueError):
        sparse_state_for_case("H4_square_eq_side1p0_CISD")


# --- the mixed state and the moment update ----------------------------------------------------------


@pytest.fixture(scope="module")
def setup(h4_cisd_problem):
    problem = h4_cisd_problem
    library = build_context_library(problem, "mass")
    oracle = OracleMoments(library, problem.evaluator.state)
    hf = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    base = build_fragment_problems(problem, library, oracle, "II-0")
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    return problem, library, oracle, hf, base, coords


def test_mixed_moments_are_those_of_the_mixed_state(setup):
    """Eq. 19: mu_0 = (1 - eps) mu_c and R_0 = (1 - eps) R_c + eps I for distinct non-identity Paulis."""
    _, library, _, hf, _, _ = setup
    eps = 0.2
    mixed = MixedMoments(hf, eps)
    context = library.contexts[0]
    paulis = context.members[: min(12, len(context.members))]
    p = hf.distribution(0)
    assert np.allclose(mixed.distribution(0), (1 - eps) * p + eps / p.size)
    assert mixed.group_means(0)[0] == pytest.approx(1.0)
    mu_c = hf.means(0, paulis)
    mu_0 = mixed.means(0, paulis)
    assert np.allclose(mu_0, (1 - eps) * mu_c)
    r_c = hf.covariance(0, paulis) + np.outer(mu_c, mu_c)
    r_0 = mixed.covariance(0, paulis) + np.outer(mu_0, mu_0)
    assert np.allclose(r_0, (1 - eps) * r_c + eps * np.eye(len(paulis)))
    assert np.linalg.eigvalsh(mixed.covariance(0, paulis)).min() > -1e-12
    assert np.allclose(MixedMoments(hf, 0.0).covariance(0, paulis), hf.covariance(0, paulis))
    # a prior that predicts zero variance in some direction now predicts a positive one in every direction
    for alpha in range(library.n_contexts):
        members = library.contexts[alpha].members
        if np.diag(hf.covariance(alpha, members)).min() < 1e-12:
            assert np.diag(MixedMoments(hf, eps).covariance(alpha, members)).min() >= eps * (1 - eps) - 1e-12
            break
    else:
        pytest.skip("no context of this library has a zero-variance direction under HF")


def _outcome_vectors(model, alpha, paulis):
    zmask, sign = model._images(alpha, paulis)
    outcomes = np.arange(model.counts.shape[2])
    parity = np.array([[bin(int(z) & int(b)).count("1") & 1 for b in outcomes] for z in zmask])
    return sign[:, None] * (1 - 2 * parity)  # (n_paulis, n_outcomes): the eigenvalue each Pauli reports on each outcome


def test_moment_update_is_the_raw_moment_update(setup):
    """Eqs. 20-23: S, T and the update of mu and R, then Sigma = R - mu mu^T."""
    _, library, oracle, hf, _, _ = setup
    distributions = oracle.distributions_matrix()
    distributions /= distributions.sum(axis=1, keepdims=True)
    mixed = MixedMoments(hf, 0.1)
    tau = 40.0
    model = CovarianceModel(library, distributions.shape[1], mixed, tau, update="moments")
    rng = np.random.default_rng(5)
    shots = 60
    contexts = np.arange(library.n_contexts)
    model.add(contexts, (rng.multinomial(shots // 2, distributions), rng.multinomial(shots - shots // 2, distributions)))
    alpha = 3
    paulis = library.contexts[alpha].members[:8]
    z = _outcome_vectors(model, alpha, paulis)
    counts = model.counts[:, alpha].sum(axis=0).astype(float)
    n = counts.sum()
    s = z @ counts
    t = (z * counts) @ z.T
    mu0 = mixed.means(alpha, paulis)
    r0 = mixed.covariance(alpha, paulis) + np.outer(mu0, mu0)
    mu = (tau * mu0 + s) / (tau + n)
    r = (tau * r0 + t) / (tau + n)
    assert np.allclose(model.covariance(alpha, paulis), r - np.outer(mu, mu))
    assert np.linalg.eigvalsh(model.covariance(alpha, paulis)).min() > -1e-12
    # no shots: the classical design covariance; very many shots (static prior excluded): the data
    empty = CovarianceModel(library, distributions.shape[1], mixed, tau, update="moments")
    assert np.allclose(empty.covariance(alpha, paulis), mixed.covariance(alpha, paulis))
    static = CovarianceModel(library, distributions.shape[1], mixed, math.inf, update="moments")
    static.counts = model.counts.copy()
    assert np.allclose(static.covariance(alpha, paulis), mixed.covariance(alpha, paulis))
    # the radii never see the prior: data only (unbiased), and the bound k I with fewer than two shots
    unbiased = model.covariance(alpha, paulis, data_only=True)
    plug_in = CovarianceModel(library, distributions.shape[1], None, 0.0)
    plug_in.counts = model.counts.copy()
    assert np.allclose(unbiased, plug_in.covariance(alpha, paulis))
    assert np.allclose(empty.covariance(alpha, paulis, data_only=True), len(paulis) * np.eye(len(paulis)))


def test_radii_and_planning_see_the_data_or_the_design_covariance(setup):
    """prior_radii="data": the held-out covariance of the radii and the planner is the measured one; "design": the updated prior."""
    problem, library, oracle, hf, base, coords = setup
    rng = np.random.default_rng(9)
    results = {}
    for radii in ("data", "design"):
        config = LearningConfig(prior="hf", nu=100.0, prior_update="moments", prior_eps=0.1, rule="safe", start="bound",
                                radius_min_shots=0, min_fold_shots=0, prior_radii=radii)
        learner = LearnedM3(problem, library, oracle, hf, base, coords, config)
        model = learner._model()
        contexts = np.arange(library.n_contexts)
        distributions = learner.distributions
        model.add(contexts, (np.random.default_rng(1).multinomial(80, distributions), np.random.default_rng(2).multinomial(80, distributions)))
        paulis = library.contexts[3].members[:8]
        shots = model.shots(1)
        results[radii] = learner._held_out(model, 3, paulis, 1, shots)
        data = model.covariance(3, paulis, 1, data_only=True)
        design = model.covariance(3, paulis, 1)
        assert np.allclose(results[radii], data if radii == "data" else design)
        assert not np.allclose(data, design)
    with pytest.raises(ValueError):
        LearningConfig(prior_radii="never")


def test_configuration_checks_and_labels():
    with pytest.raises(ValueError):
        LearningConfig(prior="flat", prior_update="moments")
    with pytest.raises(ValueError):
        LearningConfig(prior="none", prior_eps=0.1)
    with pytest.raises(ValueError):
        LearningConfig(prior="hf", prior_eps=1.0)
    with pytest.raises(ValueError):
        LearningConfig(prior="hf", prior_update="moments", radii="oracle")
    old = LearningConfig(prior="hf", nu=100.0)
    assert "update=" not in old.label and "eps=" not in old.label  # existing labels (and trial files) are unchanged
    new = LearningConfig(level="II-0", prior="sparse", nu=math.inf, prior_update="moments", prior_eps=0.1)
    assert "prior=sparse" in new.label and "update=moments" in new.label and "eps=0.1" in new.label
    assert LearningConfig(level="II-0", prior="none", nu=0.0).label == "II-0/estimated radii"


def test_sparse_prior_requires_the_classical_state(setup):
    problem, library, oracle, hf, base, coords = setup
    config = LearningConfig(prior="sparse", nu=100.0, prior_update="moments", prior_eps=0.1)
    with pytest.raises(ValueError):
        LearnedM3(problem, library, oracle, hf, base, coords, config)._model()


@pytest.mark.parametrize("level,nu", [("II-0", math.inf), ("II-A", math.inf), ("II-A", 100.0)])
def test_runs_with_a_classical_prior_select_the_winner_and_keep_the_estimate_unbiased(setup, level, nu):
    problem, library, oracle, hf, base, coords = setup
    config = LearningConfig(level=level, prior="sparse", nu=nu, prior_update="moments", prior_eps=0.1, rule="safe", start="bound",
                            radius_min_shots=50, min_fold_shots=0)
    learner = LearnedM3(problem, library, oracle, hf, base, coords, config, classical=hf)  # any state works as the classical one
    outcome = learner.run(np.random.default_rng(2))
    assert outcome.correct and outcome.shots > 0
    # the cross-fitted estimate stays unbiased whatever the (here deliberately wrong) prior: fixed shots, designs from the prior
    rng = np.random.default_rng(17)
    shots = np.full(library.n_contexts, 150)
    replicates = 120
    values = np.empty((replicates, problem.n_generators))
    for r in range(replicates):
        model = learner._model()
        contexts = np.arange(library.n_contexts)
        half = shots // 2
        model.add(contexts, (rng.multinomial(half, learner.distributions), rng.multinomial(shots - half, learner.distributions)))
        designs = learner._refit(model, set(range(problem.n_generators))) if level != "II-0" else learner._base_designs()
        values[r] = learner._estimates(model, designs)
    error = values - problem.gradients
    z = np.abs(error.mean(axis=0)) / np.maximum(error.std(axis=0, ddof=1) / math.sqrt(replicates), 1e-12)
    assert np.nanmax(z[error.std(axis=0) > 0]) < 4.5
