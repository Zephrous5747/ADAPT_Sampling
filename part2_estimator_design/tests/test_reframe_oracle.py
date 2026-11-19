"""JCTC round 5: the exact-covariance bounds ``FC_SE_ORACLE`` and ``FC_SE_CS_ORACLE`` (design with the covariance of the true state).
H4 only (a few seconds)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402
from reframe import LAMBDA, ORACLE_METHODS, Counts, DesignCovariance, epsilon_round  # noqa: E402

CASE = "H4_square_eq_side1p0_ADAPT3"
KW = dict(restart_ratio=0.0, batch_fix=True)


@pytest.fixture(scope="module")
def selector():
    built, _, _ = reframe_run.build(CASE, 1e9, need_prior=False, need_oracle=True)
    return built


def test_oracle_methods_need_the_true_moments():
    built, _, _ = reframe_run.build(CASE, 1e9, need_prior=False)
    assert built.truth_moments is None
    with pytest.raises(ValueError):
        built.run(np.random.default_rng(0), "FC_SE_ORACLE", 0.1, 0.05, **KW)


def test_exact_design_covariance_ignores_the_data(selector):
    """The oracle design covariance is the covariance of the true state plus the regularizer, before and after any outcomes."""
    counts = Counts(selector.n_contexts)
    cov = DesignCovariance(selector.library, counts, selector.truth_moments, exact=True)
    alpha = int(np.argmax([len(c.members) for c in selector.library.contexts]))
    paulis = np.asarray(selector.library.contexts[alpha].members[:12])
    before = cov(alpha, paulis)
    expected = selector.truth_moments.covariance(alpha, paulis)
    assert np.allclose(before, 0.5 * (expected + expected.T) + LAMBDA * np.eye(len(paulis)), atol=1e-14, rtol=0)
    assert np.linalg.eigvalsh(before).min() >= LAMBDA - 1e-12  # covariance is positive semidefinite
    counts.add(np.array([alpha]), np.round(selector.dist[alpha][None, :] * 400.0))  # some outcomes arrive
    assert counts.n[alpha] > 1 and np.array_equal(before, cov(alpha, paulis))


@pytest.mark.parametrize("method", ORACLE_METHODS)
def test_oracle_plan_meets_the_variance_targets_under_the_true_covariance(selector, method):
    """Whatever design is chosen (optimized or the fixed fallback), every requested gradient has true variance at most epsilon^2."""
    counts = Counts(selector.n_contexts)
    cov = DesignCovariance(selector.library, counts, selector.truth_moments, exact=True)
    requested = list(selector.candidates)
    epsilon = epsilon_round(1)
    stats = {"plan_seconds": 0.0, "solve_seconds": 0.0}
    designs, plan, _, choice = selector._plan(method, requested, epsilon, cov, stats)
    assert choice in ("fixed", "optimized", "fixed_invalid")
    for i in requested:
        variance = 0.0
        for a, p, x in designs[i]:
            assert plan[a] > 0.0
            variance += float(x @ selector.truth_moments.covariance(a, p) @ x) / plan[a]
        assert variance <= epsilon ** 2 * (1.0 + 1e-6), (i, variance, epsilon ** 2)


@pytest.mark.parametrize("method", ORACLE_METHODS)
def test_oracle_arms_certify_a_rho_good_selection(selector, method):
    for seed in range(3):
        r = selector.run(np.random.default_rng(seed), method, 0.1, 0.05, **KW)
        assert r["status"] == "certified" and r["rho_good"] and not r["maximizer_eliminated"]
        assert r["block_failures"] == 0 and r["history_failures"] == 0


def test_oracle_records_are_marked(selector):
    import json

    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    tasks = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE_CS_ORACLE", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True)
    plain = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE_CS_DATA", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True)
    assert all(t["method"] in ORACLE_METHODS for t in tasks)
    assert not {t["task_id"] for t in tasks} & {t["task_id"] for t in plain}
