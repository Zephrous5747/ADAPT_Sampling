"""Step 4: the covariance model and the cross-fitted learned estimator."""
from __future__ import annotations

import math

import numpy as np
import pytest

from contexts import build_context_library
from design import build_fragment_problems
from learning import CovarianceModel, LearnedM3, LearningConfig
from sampler import OracleMoments
from states import hartree_fock_state  # Part I


@pytest.fixture(scope="module")
def setup(h4_cisd_problem):
    problem = h4_cisd_problem
    library = build_context_library(problem, "mass")
    oracle = OracleMoments(library, problem.evaluator.state)
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    base = build_fragment_problems(problem, library, oracle, "II-0")
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    return problem, library, oracle, prior, base, coords


def _sample(model, distributions, shots, rng):
    contexts = np.arange(distributions.shape[0])
    half = shots // 2
    model.add(contexts, (rng.multinomial(half, distributions), rng.multinomial(shots - half, distributions)))


def test_shrinkage_limits_and_psd(setup):
    problem, library, oracle, prior, _, _ = setup
    distributions = oracle.distributions_matrix()
    distributions /= distributions.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(3)
    context = library.contexts[0]
    paulis = context.members[:10]
    for nu, expected in ((math.inf, prior.covariance(0, paulis)), (0.0, oracle.covariance(0, paulis))):
        model = CovarianceModel(library, distributions.shape[1], prior, nu)
        _sample(model, distributions, np.full(library.n_contexts, 200_000), rng)
        sigma = model.covariance(0, paulis)
        assert np.linalg.eigvalsh(sigma).min() > -1e-12
        assert np.allclose(sigma, expected, atol=1e-2)
    model = CovarianceModel(library, distributions.shape[1], prior, 100.0)
    _sample(model, distributions, np.full(library.n_contexts, 100), rng)
    mixed = model.covariance(0, paulis)
    only_data = CovarianceModel(library, distributions.shape[1], prior, 0.0)
    only_data.counts = model.counts.copy()
    weight = 100.0 / (100.0 + 100.0)
    assert np.allclose(mixed, weight * prior.covariance(0, paulis) + (1 - weight) * only_data.covariance(0, paulis))


def test_cross_fitted_estimate_is_unbiased_at_fixed_shots(setup):
    problem, library, oracle, prior, base, coords = setup
    learner = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig(prior="none", nu=0.0))
    distributions = learner.distributions
    rng = np.random.default_rng(11)
    shots = np.full(library.n_contexts, 400)
    order = problem.ranking()[:4]
    replicates = 150
    estimates = np.empty((replicates, len(order)))
    for r in range(replicates):
        model = learner._model()
        _sample(model, distributions, shots, rng)
        designs = learner._refit(model, set(range(problem.n_generators)))
        estimates[r] = learner._estimates(model, designs)[order]
    error = estimates - problem.gradients[order]
    z = np.abs(error.mean(axis=0)) / (error.std(axis=0, ddof=1) / math.sqrt(replicates))
    assert z.max() < 4.0


def test_estimated_variance_matches_the_actual_spread(setup):
    """The radii must not be flattered by the fold the design was fitted to.

    With few shots per context the fitted design overfits its own fold; its
    variance estimated with a covariance that includes that fold comes out too
    small, which on LiH produced 54% correct selections.  The held-out estimate
    must match the replicate-to-replicate spread of the cross-fitted estimate.
    """
    problem, library, oracle, prior, base, coords = setup
    learner = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig(prior="none", nu=0.0))
    rng = np.random.default_rng(17)
    shots = np.full(library.n_contexts, 120)  # 60 per fold: above min_fold_shots, still few
    arms = problem.ranking()[:4]
    replicates = 200
    estimates = np.empty((replicates, len(arms)))
    predicted = np.empty((replicates, len(arms)))
    for r in range(replicates):
        model = learner._model()
        _sample(model, learner.distributions, shots, rng)
        designs = learner._refit(model, set(range(problem.n_generators)))
        estimates[r] = learner._estimates(model, designs)[arms]
        predicted[r] = np.diag(learner._covariance_matrix(model, designs, arms))
    assert learner.guard_kept > 0, "the test must exercise learned designs"
    ratio = predicted.mean(axis=0) / estimates.var(axis=0, ddof=1)
    assert (ratio > 0.75).all() and (ratio < 1.35).all(), ratio


def test_learned_run_selects_the_winner(setup):
    problem, library, oracle, prior, base, coords = setup
    learner = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig(prior="none", nu=0.0))
    outcome = learner.run(np.random.default_rng(1))
    assert outcome.correct and outcome.shots > 0


def test_contrast_designs_are_unbiased_with_honest_variances(setup):
    """II-E: a directly designed contrast is unbiased, and its held-out variance is honest."""
    problem, library, oracle, prior, base, coords = setup
    config = LearningConfig(prior="none", nu=0.0, rule="safe", objective="contrast")
    learner = LearnedM3(problem, library, oracle, prior, base, coords, config)
    rng = np.random.default_rng(23)
    shots = np.full(library.n_contexts, 120)
    lead, *others = problem.ranking()[:4]
    sign = float(np.sign(problem.gradients[lead]))
    keys = [(lead, sign, i, t) for i in others for t in (1.0, -1.0)]
    exact = np.array([sign * problem.gradients[lead] - t * problem.gradients[i] for _, _, i, t in keys])
    replicates = 150
    values = np.empty((replicates, len(keys)))
    predicted = np.empty((replicates, len(keys)))
    for r in range(replicates):
        model = learner._model()
        _sample(model, learner.distributions, shots, rng)
        designs = learner._refit(model, set(range(problem.n_generators)))
        contrast = learner._design_contrasts(model, designs, keys)
        values[r] = learner._estimates(model, contrast)
        predicted[r] = np.diag(learner._covariance_matrix(model, contrast, list(range(len(keys)))))
    assert learner.contrast_kept > 0
    error = values - exact
    z = np.abs(error.mean(axis=0)) / (error.std(axis=0, ddof=1) / math.sqrt(replicates))
    assert z.max() < 4.0
    ratio = predicted.mean(axis=0) / values.var(axis=0, ddof=1)
    assert (ratio > 0.7).all() and (ratio < 1.4).all(), ratio


def test_contrast_run_selects_the_winner_and_logs_everything(setup, tmp_path):
    import json

    from runlog import RunLog, write_static

    problem, library, oracle, prior, base, coords = setup
    config = LearningConfig(prior="none", nu=0.0, rule="safe", objective="contrast", start="bound")
    learner = LearnedM3(problem, library, oracle, prior, base, coords, config)
    log = RunLog()
    outcome = learner.run(np.random.default_rng(1), log=log)
    assert outcome.correct and outcome.shots > 0
    paths = log.save(tmp_path, "run")
    write_static(tmp_path / "static", problem, library)
    rounds = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert rounds[-1]["cumulative_shots"] == outcome.shots
    assert sum(sum(v) for r in rounds for v in r["shots_added"].values()) == outcome.shots
    assert any(r["contrasts"] for r in rounds) and any(r["eliminated"] for r in rounds)
    summary = json.loads(paths[4].read_text())
    assert "exact_gradients" not in summary  # oracle values only in the validation file
    assert "exact_gradients" in json.loads(paths[5].read_text())
    designs = np.load(paths[2])
    assert any(k.endswith("_x") for k in designs.files)


def test_reconstruction_residual_fails_hard(setup):
    problem, library, oracle, prior, base, coords = setup
    learner = LearnedM3(problem, library, oracle, prior, base, coords, LearningConfig())
    ctx, pauli, targets = coords[problem.ranking()[0]]
    home = library.home[pauli]
    first = np.unique(pauli, return_index=True)[1]
    x = np.array([targets[int(l)] for l in pauli[first]])
    learner._check_row((home[first], pauli[first], x), targets, "exact")
    x[0] += 1e-6
    with pytest.raises(AssertionError):
        learner._check_row((home[first], pauli[first], x), targets, "perturbed")


def test_contrast_objective_requires_the_safe_rule():
    with pytest.raises(ValueError):
        LearningConfig(objective="contrast", rule="pairwise")
