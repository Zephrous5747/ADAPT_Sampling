"""Task T25: the leader-out relative certificate of GS-1.0 (``run(..., leader_out=True)``): L_k >= (1 - rho) max_{i != k} U_i.
H4 only (a few seconds)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402

CASES = ["H4_square_eq_side1p0_CISD", "H4_square_eq_side1p0_ADAPT3"]
KW = dict(restart_ratio=0.0, batch_fix=True)


@pytest.fixture(scope="module", params=CASES)
def selector(request):
    built, _, _ = reframe_run.build(request.param, 1e9, need_prior=False)
    return built


@pytest.mark.parametrize("method", ["FC_SE", "FC_SE_CS_DATA"])
def test_leader_out_does_not_change_strict(selector, method):
    a = selector.run(np.random.default_rng(3), method, 0.0, 0.05, **KW)
    b = selector.run(np.random.default_rng(3), method, 0.0, 0.05, leader_out=True, **KW)
    assert a["selection_shots"] == b["selection_shots"] and a["selected_index"] == b["selected_index"]


@pytest.mark.parametrize("method", ["FC_SE", "FC_SE_CS_DATA"])
def test_leader_out_never_waits_longer_and_stays_rho_good(selector, method):
    """Same random stream: the leader-out certificate is implied by the default one at every checkpoint, so it stops at the same checkpoint or earlier."""
    for seed in range(4):
        old = selector.run(np.random.default_rng(seed), method, 0.1, 0.05, **KW)
        new = selector.run(np.random.default_rng(seed), method, 0.1, 0.05, leader_out=True, **KW)
        assert old["status"] == new["status"] == "certified"
        assert new["selection_shots"] <= old["selection_shots"]
        assert new["rho_good"] and new["leader_out"] is True and old["leader_out"] is False


def test_task_ids_differ_but_seeds_do_not():
    import json

    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    a = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True)
    b = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True, leader_out=True)
    assert [t["seed"] for t in a] == [t["seed"] for t in b]
    assert not set(t["task_id"] for t in a) & set(t["task_id"] for t in b)


def test_trajectory_task_ids_differ_but_seeds_do_not():
    import json

    import reframe_traj_run

    protocol = json.loads(reframe_traj_run.PROTOCOL.read_text())
    a = reframe_traj_run.tasks_for(protocol, "LiH_r16", "FC_SE", "rho010", 3, 0)
    b = reframe_traj_run.tasks_for(protocol, "LiH_r16", "FC_SE", "rho010", 3, 0, leader_out=True)
    assert [t["seed"] for t in a] == [t["seed"] for t in b]
    assert not set(t["task_id"] for t in a) & set(t["task_id"] for t in b)


def test_rho_and_delta_setting_changes_only_delta():
    """T26 setting rho010risk020: rho 0.1 with delta 0.2; every other setting keeps delta 0.05; the seeds differ because the setting id is part of the key."""
    import json

    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    a = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True)
    b = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", "FC_SE", 2, 0, setting_id="rho010risk020", restart_ratio=0.0, batch_fix=True)
    assert {t["rho"] for t in a} == {t["rho"] for t in b} == {0.1}
    assert {t["delta"] for t in a} == {0.05} and {t["delta"] for t in b} == {0.2}
    assert not {t["seed"] for t in a} & {t["seed"] for t in b}
