"""Step 1: the shot-level online loop against Part I's trial loop."""
from __future__ import annotations

import numpy as np
import pytest

import finite_shot  # Part I
from contexts import build_context_library
from design import DesignSet, build_fragment_problems
from online import OnlineConfig, OnlineM3, summarise
from part1_bridge import DEFAULT_DELTA, z_from_delta
from sampler import OracleMoments


class _ZeroNoise:
    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.zeros(size if size is not None else np.shape(scale))

    def standard_normal(self, size=None):
        return np.zeros(size)


@pytest.fixture(scope="module")
def baseline(h4_cisd_problem):
    library = build_context_library(h4_cisd_problem, "canonical")
    moments = OracleMoments(library, h4_cisd_problem.evaluator.state)
    design = DesignSet(build_fragment_problems(h4_cisd_problem, library, moments, "II-0"), library.n_contexts)
    return h4_cisd_problem, library, moments, design


@pytest.mark.parametrize("rule", ["marginal", "pairwise"])
def test_noiseless_loop_reproduces_part1(baseline, rule):
    problem, library, moments, design = baseline
    sigmas = problem.parent_fragment_sigmas()
    covariances = finite_shot.context_covariances(problem, problem.parent_fc_groups()) if rule == "pairwise" else None
    reference = finite_shot._sequential_trial(
        problem, None, sigmas, z_from_delta(DEFAULT_DELTA, problem.n_generators), 0.5, 0.0,
        _ZeroNoise(), True, finite_shot._AllocationCache(sigmas), False, None, covariances,
    ).shots
    exact = OnlineM3(problem, library, moments, design,
                     OnlineConfig(rule=rule, sampling="none", integer_shots=False))
    assert exact.run(np.random.default_rng(0)).shots == pytest.approx(reference, rel=1e-9)
    integer = OnlineM3(problem, library, moments, design, OnlineConfig(rule=rule, sampling="none"))
    assert integer.run(np.random.default_rng(0)).shots == pytest.approx(reference, rel=2e-2)


def test_pairwise_covariances_match_part1(baseline):
    problem, library, moments, design = baseline
    model = OnlineM3(problem, library, moments, design, OnlineConfig(rule="pairwise"))
    reference = finite_shot.context_covariances(problem, problem.parent_fc_groups())
    assert np.allclose(model.covariances, np.stack(reference), atol=1e-12)


def test_sampled_run_selects_and_is_reproducible(baseline):
    problem, library, moments, design = baseline
    model = OnlineM3(problem, library, moments, design, OnlineConfig(rule="pairwise", shrink=0.7))
    first = [model.run(np.random.default_rng(5)) for _ in range(3)]
    again = [model.run(np.random.default_rng(5)) for _ in range(3)]
    assert [o.shots for o in first] == [o.shots for o in again]
    summary = summarise(first)
    assert summary["correct_rate"] == 1.0 and summary["shots_mean"] > 0
