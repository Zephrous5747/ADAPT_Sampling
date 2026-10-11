"""Phase 4 trajectories run in stages (selection checkpoints, ``--max-selections``, ``--max-hours``) give the direct run's result."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase4_adapt_selection as p4  # noqa: E402


@pytest.fixture(scope="module")
def study():
    return p4.Phase4("H4_square_eq_side1p0_HF", "mass", p4.CHEMICAL_ACCURACY)


def _rng():
    return np.random.default_rng(np.random.SeedSequence(7).spawn(3)[1])


def test_staged_trajectory_equals_direct(study, tmp_path):
    config = p4.METHODS["M2 safe"]
    direct = study.trajectory(_rng(), config)
    stem = tmp_path / "t"
    first = study.trajectory(_rng(), config, stem, budget=3)
    assert first["paused"] and first["selections_done"] == 3
    second = study.trajectory(_rng(), config, stem, budget=3)
    assert second["paused"] and second["selections_done"] == 6
    last = study.trajectory(_rng(), config, stem)
    assert not last.get("paused") and not list(tmp_path.glob("*.partial.jsonl"))  # checkpoint removed when finished
    key = ("selected", "shots", "rounds", "shortfall")
    assert [[s[k] for k in key] for s in last["selections"]] == [[s[k] for k in key] for s in direct["selections"]]
    assert last["total_shots"] == direct["total_shots"]


def test_time_budget_stops_before_a_selection_that_would_not_fit(study, tmp_path):
    config = p4.METHODS["M2 safe"]
    direct = study.trajectory(_rng(), config)
    stem = tmp_path / "t"
    now = time.time()
    assert study.trajectory(_rng(), config, stem, deadline=now - 1.0) == {"paused": True, "selections_done": 0}
    first = study.trajectory(_rng(), config, stem, budget=2)
    assert first["paused"] and first["selections_done"] == 2
    (path,) = tmp_path.glob("*.partial.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:  # the recorded selections took 10^5 s each
        record["info"]["seconds"] = 1e5
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    # the next selection would take about 2 x 10^5 s and the job has 10^4 s left: it is left to the next stage
    assert study.trajectory(_rng(), config, stem, deadline=time.time() + 1e4) == {"paused": True, "selections_done": 2}
    # with time enough the trajectory finishes and equals the direct run
    last = study.trajectory(_rng(), config, stem, deadline=time.time() + 1e9)
    assert not last.get("paused") and not list(tmp_path.glob("*.partial.jsonl"))
    key = ("selected", "shots", "rounds", "shortfall")
    assert [[s[k] for k in key] for s in last["selections"]] == [[s[k] for k in key] for s in direct["selections"]]
