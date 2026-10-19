"""Shared fixtures: Part II's ``src`` on the path and one small built problem.

BLAS threads.  On square H4 the HOMO shell is degenerate, and a multithreaded BLAS lets the SCF return one of several equivalent
orbital sets from run to run (the same process settings gave E = -1.910 and E = -1.638 on the cached CISD state in two consecutive
runs).  The cached problems were built single-threaded, as the cluster jobs run, and the tests that rebuild a Hamiltonian next to a
cached state need the same orbitals, so the suite runs with one BLAS thread.  The variables must be set before numpy is imported.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "FLEXIBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

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
