"""Trial results depend only on (seed, trial index): not on workers or shards."""
from __future__ import annotations

import numpy as np
import pytest

from contexts import build_context_library
from design import DesignSet, build_fragment_problems
from online import OnlineConfig, OnlineM3
from parallel import parse_shard, run_trials, shard_indices
from sampler import OracleMoments


@pytest.fixture(scope="module")
def model(h4_cisd_problem):
    library = build_context_library(h4_cisd_problem, "canonical")
    moments = OracleMoments(library, h4_cisd_problem.evaluator.state)
    design = DesignSet(build_fragment_problems(h4_cisd_problem, library, moments, "II-0"), library.n_contexts)
    return OnlineM3(h4_cisd_problem, library, moments, design, OnlineConfig(rule="pairwise", shrink=0.7))


def _shots(results):
    return [(i, o.shots, o.selected) for i, o in results]


def test_workers_and_shards_give_identical_trials(model):
    serial = run_trials(model.run, 12, seed=5)
    parallel = run_trials(model.run, 12, seed=5, workers=4)
    assert _shots(serial) == _shots(parallel)
    shards = []
    for k in (1, 2, 3):
        shards.extend(run_trials(model.run, 12, seed=5, workers=2, shard=(k, 3)))
    assert _shots(sorted(shards, key=lambda item: item[0])) == _shots(serial)


def test_shard_parsing_and_cover():
    assert parse_shard("2/5") == (2, 5)
    with pytest.raises(ValueError):
        parse_shard("0/3")
    union = sorted(i for k in range(1, 5) for i in shard_indices(10, (k, 4)))
    assert union == list(range(10))
