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


def test_checkpoint_resume_reproduces_an_uninterrupted_run(tmp_path):
    """Trials checkpointed by an interrupted run plus the rest equal one full run."""
    from online import OnlineOutcome
    from parallel import Checkpoint, completed_trials, run_trials, write_trials

    def runner(rng):
        value = float(rng.normal())
        return OnlineOutcome(int(value > 0), value > 0, 100 * abs(value), 3, extra={"v": value}), {"v": value}

    full = run_trials(runner, 10, 4, workers=2)
    checkpoint = Checkpoint(tmp_path / "trials.csv")
    first = run_trials(runner, 10, 4, shard=(1, 1), skip=set(range(5, 10)), on_result=checkpoint.add)
    assert [i for i, _ in first] == list(range(5))
    done = checkpoint.load()
    assert sorted(done) == list(range(5))
    rest = run_trials(runner, 10, 4, workers=2, skip=done, on_result=checkpoint.add)
    merged = sorted(list(done.items()) + rest, key=lambda item: item[0])
    assert [r[1][0].shots for r in merged] == [r[1][0].shots for r in full]
    write_trials(tmp_path / "trials.csv", "x", 10, 4, [(i, o, e) for i, (o, e) in merged])
    assert completed_trials(tmp_path / "trials.csv", 10, (1, 1)) is not None
    assert completed_trials(tmp_path / "trials.csv", 10, (1, 2)) is None
