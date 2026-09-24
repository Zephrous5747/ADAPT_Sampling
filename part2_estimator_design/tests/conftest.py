"""Shared fixtures: Part II's ``src`` on the path and one small built problem."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402  (puts Part I's src on the path as well)

SMALL_CASE = "H4_square_eq_side1p0_HF"


@pytest.fixture(scope="session")
def h4_problem():
    """H4 side 1.0 HF, built by the Part I pipeline (validation gate included)."""
    return part1_bridge.load_problem(SMALL_CASE)


@pytest.fixture(scope="session")
def h4_cisd_problem():
    return part1_bridge.load_problem("H4_square_eq_side1p0_CISD")
