"""Task T24 additions to GS-1.0 (``src/reframe.py``, ``scripts/reframe_run.py``): the full-pool dominance rule, the checkpoint growth factor and
the individual-grouping contexts.  H4 only (a few seconds)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402

CASE = "H4_square_eq_side1p0_CISD"


@pytest.fixture(scope="module")
def selector():
    built, _, _ = reframe_run.build(CASE, 1e9)
    return built


def test_full_pool_dominance_never_waits_longer_at_rho_positive(selector):
    """With the dominance rule FC_FULL stops at the first checkpoint where the old rule stops, or earlier (same random stream)."""
    for seed in range(3):
        old = selector.run(np.random.default_rng(seed), "FC_FULL", 0.1, 0.05, restart_ratio=0.0, batch_fix=True)
        new = selector.run(np.random.default_rng(seed), "FC_FULL", 0.1, 0.05, restart_ratio=0.0, batch_fix=True, full_dominance=True)
        assert old["status"] == new["status"] == "certified"
        assert new["selection_shots"] <= old["selection_shots"]
        assert new["rho_good"]


def test_full_pool_dominance_does_not_change_strict(selector):
    a = selector.run(np.random.default_rng(5), "FC_FULL", 0.0, 0.05, restart_ratio=0.0, batch_fix=True)
    b = selector.run(np.random.default_rng(5), "FC_FULL", 0.0, 0.05, restart_ratio=0.0, batch_fix=True, full_dominance=True)
    assert a["selection_shots"] == b["selection_shots"] and a["selected_index"] == b["selected_index"]


def test_dominance_does_not_touch_elimination_arms(selector):
    a = selector.run(np.random.default_rng(6), "FC_SE", 0.1, 0.05, restart_ratio=0.0, batch_fix=True)
    b = selector.run(np.random.default_rng(6), "FC_SE", 0.1, 0.05, restart_ratio=0.0, batch_fix=True, full_dominance=True)
    assert a["selection_shots"] == b["selection_shots"]


@pytest.mark.parametrize("growth", [1.25, 1.5])
def test_finer_checkpoints_still_certify_correctly(selector, growth):
    result = selector.run(np.random.default_rng(7), "FC_SE", 0.0, 0.05, restart_ratio=0.0, batch_fix=True, growth=growth)
    assert result["status"] in ("certified", "unresolved")
    if result["status"] == "certified":
        assert result["correct"]
    assert result["checkpoints"] > 0


def test_individual_contexts_reproduce_every_gradient():
    built, _, info = reframe_run.build("H4_square_eq_side1p0_ADAPT3", 1e9, grouping="individual")
    from sampler import OracleMoments

    oracle = OracleMoments(built.library, built.problem.evaluator.state)
    for i in range(built.n_arms):
        total = sum(float(x @ oracle.means(a, p)) for a, p, x in built.x0[i])
        assert total == pytest.approx(built.truth[i], abs=1e-12)
    # nothing is shared: every context belongs to exactly one gradient
    owners = [a for fragments in built.x0 for a, _, _ in fragments]
    assert len(owners) == len(set(owners)) == built.n_contexts
