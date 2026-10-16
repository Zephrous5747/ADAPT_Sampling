"""The trajectory policies of ``scripts/energy_trajectory_policies.py`` (selection with injected errors, exact gradients)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
from trajectory import CHEMICAL_ACCURACY, run_adapt  # noqa: E402

import energy_trajectory_policies as etp  # noqa: E402
from features_energy_lowering import light_system  # noqa: E402

CASE = "H4_square_eq_side1p0_HF"


@pytest.fixture(scope="module")
def system():
    return light_system(CASE)


@pytest.fixture(scope="module")
def exact(system):
    return run_adapt(system, tolerance=CHEMICAL_ACCURACY)


def run(system, spec, seed=0, **kw):
    return etp.adapt_with_policy(system, etp.Policy.parse(spec), np.random.default_rng(seed),
                                 tolerance=CHEMICAL_ACCURACY, max_iterations=25, **kw)


def test_exact_policy_is_run_adapt(system, exact):
    r = run(system, "exact")
    assert r["iterations"] == len(exact) - 1
    assert r["selected"] == [system.labels[s.selected] for s in exact[:-1]]
    assert r["evals_total"] == sum(int(s.evaluations) for s in exact[:-1])
    assert r["final_error"] == pytest.approx(exact[-1].error, abs=1e-12)
    assert r["energy_errors"] == pytest.approx([s.error for s in exact], abs=1e-12)


def test_rho_zero_only_picks_tied_generators(system):
    for seed in range(3):
        r = run(system, "rho_random:0", seed)
        assert max(r["g_shortfall"]) < 1e-8


def test_rho_random_stays_inside_the_set(system):
    for seed in range(3):
        r = run(system, "rho_random:0.3", seed)
        assert max(r["g_shortfall"]) <= 0.3 + 1e-9


def test_tau_random_stays_inside_the_energy_set(system):
    for seed in range(3):
        r = run(system, "tau_random:0.2", seed)
        assert min(r["lowering_ratio"]) >= 0.8 - 1e-9


def test_energy_greedy_is_at_least_as_good_per_step(system):
    r = run(system, "onep_greedy")
    assert min(r["lowering_ratio"]) >= 1.0 - 1e-9


def test_worst_energy_choice_is_not_better_than_random(system):
    worst = run(system, "rho_worst_e:0.5")
    best = run(system, "rho_random:0")
    assert np.mean(worst["lowering_ratio"]) <= np.mean(best["lowering_ratio"]) + 1e-9


def test_same_seed_same_trajectory(system):
    a, b = run(system, "rho_random:0.2", 5), run(system, "rho_random:0.2", 5)
    assert a["selected"] == b["selected"] and a["evals_total"] == b["evals_total"]
