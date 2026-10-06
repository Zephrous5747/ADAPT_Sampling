"""Informationally complete (SIC product POVM) measurement data model."""
from __future__ import annotations

import numpy as np
import pytest

from ic import (F_MATRIX, N_MATRIX, ICConfig, ICMeasurement, ICSelection, estimator_values,
                outcome_distribution, pauli_expectations)


def test_dual_frame_reproduces_pauli_expectations_of_a_single_qubit():
    # p_m = (1 + n_m . r) / 4 and sum_m f_a(m) p_m = r_a for a state with Bloch vector r
    r = np.array([0.3, -0.5, 0.6])
    p = (1.0 + (N_MATRIX[:, 1:] @ r)) / 4.0
    assert p.sum() == pytest.approx(1.0)
    assert (F_MATRIX[1:] @ p == pytest.approx(r))
    assert F_MATRIX[0] @ p == pytest.approx(1.0)


def test_pauli_expectations_and_outcome_distribution_on_a_bell_state():
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1 / np.sqrt(2)  # (|00> + |11>)/sqrt 2
    mu = pauli_expectations(psi, 2).reshape(4, 4)
    assert mu[0, 0] == pytest.approx(1) and mu[1, 1] == pytest.approx(1)  # XX
    assert mu[3, 3] == pytest.approx(1) and mu[2, 2] == pytest.approx(-1)  # ZZ, YY
    assert abs(mu[1, 0]) < 1e-12 and abs(mu[0, 3]) < 1e-12
    p = outcome_distribution(mu.reshape(-1), 2)
    assert p.min() >= -1e-12 and p.sum() == pytest.approx(1.0)


def test_estimator_is_unbiased_with_second_moment_three_to_the_weight():
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1 / np.sqrt(2)
    p = outcome_distribution(pauli_expectations(psi, 2), 2)
    for label, expected, weight in (("XX", 1.0, 2), ("ZZ", 1.0, 2), ("YY", -1.0, 2), ("ZI", 0.0, 1)):
        g = estimator_values({label: 1.0}, 2)
        assert g @ p == pytest.approx(expected)
        assert (p * g ** 2).sum() == pytest.approx(3.0 ** weight)  # every non-identity factor squares to 3


@pytest.fixture(scope="module")
def measurement(h4_cisd_problem):
    return ICMeasurement(h4_cisd_problem)


def test_ic_means_are_the_exact_gradients(measurement, h4_cisd_problem):
    assert np.allclose(measurement.means, h4_cisd_problem.gradients, atol=1e-9)


def test_exact_covariance_is_psd_and_matches_a_monte_carlo_estimate(measurement):
    cov = measurement.covariance
    assert np.allclose(cov, cov.T, atol=1e-9) and np.linalg.eigvalsh(cov).min() > -1e-8
    rng = np.random.default_rng(0)
    counts = rng.multinomial(2_000_000, measurement.p)
    est = measurement.apply(counts / counts.sum())
    second = measurement._weighted_gram(counts / counts.sum())
    sample = second - np.outer(est, est)
    scale = np.abs(cov).max()
    assert np.abs(sample - cov).max() < 0.05 * scale


def test_ic_selection_runs_and_is_reproducible(measurement):
    selection = ICSelection(measurement, ICConfig(rule="safe", rho=0.1, shrink=0.8, start_shots=1000))
    a = [selection.run(np.random.default_rng(i)) for i in range(2)]
    b = [selection.run(np.random.default_rng(i)) for i in range(2)]
    assert [o.shots for o in a] == [o.shots for o in b]
    assert all(o.extra["shortfall"] <= 0.1 + 1e-12 for o in a)
