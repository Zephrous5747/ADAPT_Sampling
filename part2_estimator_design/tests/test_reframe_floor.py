"""JCTC round 5: the shot floor of the design (``run(..., floor=...)``, ``reframe_run.py --floor``).  The default is the protocol's 32 and must leave every
record unchanged; another floor changes only the rounding of the plans.  H4 only (a few seconds)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402
from reframe import FLOOR, rounded_total  # noqa: E402

CASE = "H4_square_eq_side1p0_ADAPT3"
KW = dict(restart_ratio=0.0, batch_fix=True)


@pytest.fixture(scope="module")
def selector():
    built, _, _ = reframe_run.build(CASE, 1e9, need_prior=False)
    return built


def test_default_floor_is_the_protocols():
    assert FLOOR == 32
    plan = np.array([0.0, 1.0, 10.0, 100.0])
    assert rounded_total(plan) == rounded_total(plan, 32) == 32 + 32 + 100  # zero entries are not active
    assert rounded_total(plan, 8) == 8 + 10 + 100


def test_explicit_default_floor_changes_nothing(selector):
    a = selector.run(np.random.default_rng(3), "FC_SE_CS_DATA", 0.1, 0.05, **KW)
    b = selector.run(np.random.default_rng(3), "FC_SE_CS_DATA", 0.1, 0.05, floor=32, **KW)
    for key in ("status", "selected_index", "selection_shots", "block_shots", "checkpoints"):
        assert a[key] == b[key]
    assert a["floor"] == b["floor"] == 32


@pytest.mark.parametrize("floor", [8, 64, 128])
def test_every_planned_context_gets_at_least_the_floor(selector, floor):
    result = selector.run(np.random.default_rng(5), "FC_SE_CS_DATA", 0.1, 0.05, floor=floor, **KW)
    assert result["floor"] == floor
    assert result["blocks"], "a selection that stops before its first design has nothing to check"
    for block in result["blocks"]:
        assert block["plan_total"] >= block["contexts"] * floor


def test_a_floor_below_two_outcomes_is_refused(selector):
    with pytest.raises(ValueError):
        selector.run(np.random.default_rng(0), "FC_SE", 0.1, 0.05, floor=1, **KW)


def test_the_runner_keeps_the_record_id_of_the_default_and_separates_the_others():
    protocol = reframe_run.json.loads((ROOT / "reframe" / "protocol.json").read_text())
    digest = reframe_run.sha(protocol)
    default = reframe_run.task_list(protocol, digest, "H4_a1_ADAPT3", "FC_SE_CS_DATA", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True)
    same = reframe_run.task_list(protocol, digest, "H4_a1_ADAPT3", "FC_SE_CS_DATA", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True, floor=32)
    other = reframe_run.task_list(protocol, digest, "H4_a1_ADAPT3", "FC_SE_CS_DATA", 2, 0, setting_id="rho010", restart_ratio=0.0, batch_fix=True, floor=128)
    assert [t["task_id"] for t in default] == [t["task_id"] for t in same]
    assert all(d["task_id"] != o["task_id"] for d, o in zip(default, other))
    assert [t["seed"] for t in default] == [t["seed"] for t in other]  # paired streams
    assert all(t["floor"] == 128 for t in other)
