"""Benchmark cases added in Part II (Paper A scale checks), registered next to Part I's.

Part I's registry holds H4, LiH at 3.0 A and H2O.  A referee asked for a case nearer to
chemistry's standard benchmarks and for the cost of the design on larger molecules, so
Part II adds, with Part I's own builders and validation gate (nothing is re-derived):

* ``LiH_R1p6_HF`` / ``LiH_R1p6_CISD``: LiH at (about) its equilibrium bond length, 12 qubits;
* ``BeH2_HF`` / ``BeH2_CISD``: linear BeH2 at 1.33 A, 14 qubits;
* ``H6_chain_HF`` / ``H6_chain_CISD``: a linear H6 chain, 1.0 A spacing, 12 qubits;
* ``N2_HF``: N2 at 1.098 A, 20 qubits (structure studies only: no state-vector CISD);
* ``H4_chain1p0_HF``, ``LiH_R1p0_HF``, ``BeH2_R1p0_HF``: the linear 1 A geometries of Huang and Izmaylov's Table I.

The registry is Part I's ``cases.CASES`` dictionary, extended in place at import of
:mod:`part1_bridge`, so ``get_case`` and every cache function find the new ids.
"""
from __future__ import annotations

from chemistry import CaseSpec


def _chain(n: int, spacing: float):
    return [("H", (0.0, 0.0, spacing * i)) for i in range(n)]


EXTRA_CASES = tuple(
    CaseSpec(f"{name}_{state}", atoms, state)
    for name, atoms, states in (
        ("LiH_R1p6", [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))], ("HF", "CISD")),
        ("BeH2", [("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.33)), ("H", (0.0, 0.0, -1.33))], ("HF", "CISD")),
        ("H6_chain", _chain(6, 1.0), ("HF", "CISD")),
        ("N2", [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.098))], ("HF",)),
        # Huang and Izmaylov (arXiv:2509.14917): linear geometries with 1 A between adjacent atoms
        ("H4_chain1p0", _chain(4, 1.0), ("HF",)),
        ("LiH_R1p0", [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0))], ("HF",)),
        ("BeH2_R1p0", [("H", (0.0, 0.0, -1.0)), ("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0))], ("HF",)),
    )
    for state in states
)


def register(cases: dict) -> None:
    """Add the extra cases to Part I's registry (idempotent)."""
    for spec in EXTRA_CASES:
        cases.setdefault(spec.case_id, spec)
