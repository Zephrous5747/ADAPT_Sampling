"""Pauli-string utilities for ADAPT gradient measurement diagnostics.

Representation: a Pauli string is a string over 'I', 'X', 'Y', 'Z'. Coefficients are
Python complex numbers. Fully commuting means every pair of Pauli products commutes.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np

PAULI_ORDER = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_weight(p: str) -> int:
    return sum(ch != "I" for ch in p)


def pauli_commutes(a: str, b: str) -> bool:
    """Return True if two same-length Pauli strings commute."""
    if len(a) != len(b):
        raise ValueError("Pauli strings must have same length")
    anticomm = 0
    for x, y in zip(a, b):
        if x == "I" or y == "I" or x == y:
            continue
        anticomm += 1
    return (anticomm % 2) == 0


class _FirstFitGrouper:
    """First-fit fully-commuting grouping over symplectic bit masks.

    Two Pauli strings commute when the symplectic form
    ``popcount(x1 & z2 XOR z1 & x2)`` is even.  All placed Pauli strings are
    kept in flat arrays tagged with their group index, so testing a candidate
    against *every* existing group costs one vectorised pass instead of a Python
    loop over groups and members.  The candidate joins the lowest-numbered group
    that no anticommuting member blocks, which is exactly first-fit.
    """

    __slots__ = ("groups", "_x", "_z", "_group_of", "_size")

    def __init__(self, capacity: int = 1024) -> None:
        self.groups: List[List[str]] = []
        self._x = np.empty(capacity, dtype=np.int64)
        self._z = np.empty(capacity, dtype=np.int64)
        self._group_of = np.empty(capacity, dtype=np.int64)
        self._size = 0

    def add(self, pauli: str, x_mask: int, z_mask: int) -> None:
        index = self._first_free_group(x_mask, z_mask)
        if index == len(self.groups):
            self.groups.append([])
        self.groups[index].append(pauli)
        self._store(x_mask, z_mask, index)

    def _first_free_group(self, x_mask: int, z_mask: int) -> int:
        size = self._size
        if size == 0:
            return 0
        anticommuting = _parity(
            (x_mask & self._z[:size]) ^ (z_mask & self._x[:size])
        ).astype(bool)
        if not anticommuting.any():
            return 0
        blocked = np.bincount(
            self._group_of[:size][anticommuting], minlength=len(self.groups)
        ) > 0
        free = np.flatnonzero(~blocked)
        return int(free[0]) if free.size else len(self.groups)

    def _store(self, x_mask: int, z_mask: int, group_index: int) -> None:
        if self._size == self._x.size:
            new_capacity = self._size * 2
            self._x = np.resize(self._x, new_capacity)
            self._z = np.resize(self._z, new_capacity)
            self._group_of = np.resize(self._group_of, new_capacity)
        self._x[self._size] = x_mask
        self._z[self._size] = z_mask
        self._group_of[self._size] = group_index
        self._size += 1


def _parity(values: np.ndarray) -> np.ndarray:
    """Parity of the population count of each entry."""
    bitwise_count = getattr(np, "bitwise_count", None)
    if bitwise_count is not None:
        return bitwise_count(values) & 1
    parity = np.zeros_like(values)
    remaining = values.copy()
    while remaining.any():
        parity ^= remaining & 1
        remaining >>= 1
    return parity


def symplectic_masks(pauli: str) -> Tuple[int, int]:
    """Return ``(x_mask, z_mask)`` for a Pauli string, ignoring its phase."""
    x_mask = z_mask = 0
    for position, char in enumerate(pauli):
        if char == "I":
            continue
        if char not in PAULI_ORDER:
            raise ValueError(f"invalid Pauli character {char!r} in {pauli!r}")
        bit = 1 << position
        if char in ("X", "Y"):
            x_mask |= bit
        if char in ("Z", "Y"):
            z_mask |= bit
    return x_mask, z_mask


ORDERINGS = ("weight", "reverse-weight", "lexicographic", "random")
DEFAULT_ORDERING = "weight"


def order_paulis(
    items: List[str], ordering: str = DEFAULT_ORDERING, seed: int = 0
) -> List[str]:
    """Put Pauli strings into the insertion order a first-fit grouping will use.

    First-fit grouping is deterministic given an order, but the order itself is a
    free choice and the resulting group count depends on it. Exposing the choice
    lets the conclusions be tested against it rather than resting on one
    arbitrary convention: ``weight`` is the project default, ``reverse-weight``
    and ``lexicographic`` are two other deterministic conventions, and ``random``
    draws an order from a seed so that a whole family of groupings can be sampled.
    """
    items = sorted(items)
    if ordering == "weight":
        items.sort(key=lambda s: (-pauli_weight(s), s))
    elif ordering == "reverse-weight":
        items.sort(key=lambda s: (pauli_weight(s), s))
    elif ordering == "lexicographic":
        pass
    elif ordering == "random":
        np.random.default_rng(seed).shuffle(items)
    else:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    return items


def greedy_fc_groups(
    paulis: Iterable[str],
    *,
    sort: bool = True,
    ordering: str = DEFAULT_ORDERING,
    seed: int = 0,
) -> List[List[str]]:
    """Deterministic first-fit fully-commuting grouping.

    The insertion order is set by ``ordering``; see :func:`order_paulis`. The
    project default sorts by decreasing weight and then lexicographically. This
    is not a minimum coloring algorithm, and all reported group counts depend on
    the ordering, which is why it is a parameter.

    The result is the same first-fit grouping a naive pairwise implementation
    produces; only the commutation test is vectorised, which is what makes the
    large universal supports of Part I tractable.
    """
    items = list(dict.fromkeys(paulis))
    if sort:
        items = order_paulis(items, ordering, seed)
    grouper = _FirstFitGrouper()
    for pauli in items:
        grouper.add(pauli, *symplectic_masks(pauli))
    return grouper.groups


def openfermion_qubitop_to_dict(qop, n_qubits: int, coeff_tol: float = 1e-12) -> Dict[str, complex]:
    """Convert an OpenFermion QubitOperator to dict string -> complex coefficient."""
    out: Dict[str, complex] = OrderedDict()
    for term, coeff in qop.terms.items():
        chars = ["I"] * n_qubits
        for idx, op in term:
            chars[idx] = op
        key = "".join(chars)
        val = out.get(key, 0.0) + complex(coeff)
        if abs(val) > coeff_tol:
            out[key] = val
        elif key in out:
            del out[key]
    return dict(out)


def dict_to_openfermion_qubitop(terms: Dict[str, complex]):
    from openfermion import QubitOperator
    qop = QubitOperator()
    for p, c in terms.items():
        term = tuple((i, ch) for i, ch in enumerate(p) if ch != "I")
        qop += QubitOperator(term, c)
    return qop
