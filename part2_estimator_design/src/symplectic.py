"""Binary (symplectic) representation of Pauli strings and GF(2) linear algebra.

A Pauli operator on ``n`` qubits is stored as two bit vectors ``x`` and ``z`` and a
phase exponent ``k``,

    P = i**k * X**x * Z**z ,      X**x = X_0**x_0 ... X_{n-1}**x_{n-1},

with the ``X`` factor to the left of the ``Z`` factor on every qubit.  A Hermitian
Pauli string with a ``+`` sign has ``k = x . z``: one factor of ``i`` for every
``Y = i X Z``.

Two encodings are used and converted between freely.

* Boolean arrays of shape ``(N, n)`` indexed by qubit, convenient for Clifford
  propagation.
* Integer masks in the Part I state-vector convention of ``pauli_ops``: qubit
  ``q`` is bit ``n - 1 - q`` of a computational-basis index.  The eigenvalue of
  ``Z**z`` on basis state ``b`` is then ``(-1)**popcount(zmask & b)``, with no
  re-indexing anywhere between Pauli labels, circuits and sampled outcomes.

Packing both masks into one integer, ``(xmask << n) | zmask``, gives the GF(2)
vectors used for rank and span computations; ``2n <= 28`` bits for every system
in this project, so they fit an ``int64``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

PAULI_LETTERS = frozenset("IXYZ")


def popcount(values: np.ndarray) -> np.ndarray:
    """Population count of a non-negative integer array."""
    return np.bitwise_count(np.asarray(values, dtype=np.int64)).astype(np.int64)


def xz_from_labels(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(N, n)`` arrays ``x`` and ``z`` for a list of Pauli strings."""
    labels = list(labels)
    if not labels:
        raise ValueError("no Pauli strings given")
    n = len(labels[0])
    if any(len(label) != n for label in labels):
        raise ValueError("Pauli strings must all have the same length")
    buffer = np.frombuffer("".join(labels).encode("ascii"), dtype=np.uint8)
    chars = buffer.reshape(len(labels), n)
    valid = np.isin(chars, np.frombuffer(b"IXYZ", dtype=np.uint8))
    if not valid.all():
        raise ValueError("Pauli strings may contain only I, X, Y and Z")
    x = (chars == ord("X")) | (chars == ord("Y"))
    z = (chars == ord("Z")) | (chars == ord("Y"))
    return x, z


def labels_from_xz(x: np.ndarray, z: np.ndarray) -> list[str]:
    """Pauli strings for boolean ``x`` and ``z`` arrays (signs are not represented)."""
    x = np.atleast_2d(np.asarray(x, dtype=bool))
    z = np.atleast_2d(np.asarray(z, dtype=bool))
    letters = np.array(["I", "X", "Z", "Y"])[x.astype(int) + 2 * z.astype(int)]
    return ["".join(row) for row in letters]


def bit_weights(n: int) -> np.ndarray:
    """``2**(n - 1 - q)`` for every qubit ``q``: the Part I big-endian convention."""
    return (np.int64(1) << (n - 1 - np.arange(n, dtype=np.int64))).astype(np.int64)


def masks_from_xz(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer masks ``(xmask, zmask)`` for boolean arrays, big-endian in the qubit."""
    x = np.atleast_2d(np.asarray(x, dtype=np.int64))
    z = np.atleast_2d(np.asarray(z, dtype=np.int64))
    weights = bit_weights(x.shape[1])
    return x @ weights, z @ weights


def xz_from_masks(xmask: np.ndarray, zmask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays for integer masks, inverse of :func:`masks_from_xz`."""
    shifts = n - 1 - np.arange(n, dtype=np.int64)
    xmask = np.atleast_1d(np.asarray(xmask, dtype=np.int64))
    zmask = np.atleast_1d(np.asarray(zmask, dtype=np.int64))
    x = ((xmask[:, None] >> shifts) & 1).astype(bool)
    z = ((zmask[:, None] >> shifts) & 1).astype(bool)
    return x, z


def masks_from_labels(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    return masks_from_xz(*xz_from_labels(labels))


def pack(xmask: np.ndarray, zmask: np.ndarray, n: int) -> np.ndarray:
    """GF(2) vector ``(xmask << n) | zmask``."""
    return (np.asarray(xmask, dtype=np.int64) << n) | np.asarray(zmask, dtype=np.int64)


def unpack(vectors: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.int64)
    low = (np.int64(1) << n) - 1
    return vectors >> n, vectors & low


def anticommutes(
    xmask_a: np.ndarray, zmask_a: np.ndarray, xmask_b, zmask_b
) -> np.ndarray:
    """Symplectic form: ``True`` where the two Paulis anticommute (broadcasts)."""
    form = (np.asarray(xmask_a) & np.asarray(zmask_b)) ^ (
        np.asarray(zmask_a) & np.asarray(xmask_b)
    )
    return (popcount(form) & 1).astype(bool)


def commutes_with_all(
    xmask: np.ndarray, zmask: np.ndarray, gen_xmask: Iterable[int], gen_zmask: Iterable[int]
) -> np.ndarray:
    """``True`` for every Pauli that commutes with all the given generators."""
    ok = np.ones(np.shape(xmask), dtype=bool)
    for gx, gz in zip(gen_xmask, gen_zmask):
        ok &= ~anticommutes(xmask, zmask, np.int64(gx), np.int64(gz))
    return ok


def hermitian_phase(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Phase exponent ``k = x . z (mod 4)`` of a ``+``-signed Hermitian Pauli string."""
    return (np.asarray(x, dtype=np.int64) & np.asarray(z, dtype=np.int64)).sum(axis=-1) % 4


def pauli_product(
    x1: np.ndarray, z1: np.ndarray, k1, x2: np.ndarray, z2: np.ndarray, k2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(i**k1 X**x1 Z**z1)(i**k2 X**x2 Z**z2)`` in the same representation.

    Moving ``Z**z1`` to the right of ``X**x2`` costs ``(-1)**(z1 . x2)``.
    """
    x1, z1 = np.asarray(x1, dtype=np.int64), np.asarray(z1, dtype=np.int64)
    x2, z2 = np.asarray(x2, dtype=np.int64), np.asarray(z2, dtype=np.int64)
    k = (np.asarray(k1) + np.asarray(k2) + 2 * (z1 * x2).sum(axis=-1)) % 4
    return (x1 ^ x2).astype(bool), (z1 ^ z2).astype(bool), k


def dense_pauli(x: np.ndarray, z: np.ndarray, k: int = 0) -> np.ndarray:
    """Dense matrix of ``i**k X**x Z**z`` in the big-endian basis (tests only)."""
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    matrix = np.array([[1.0 + 0j]])
    for xq, zq in zip(np.asarray(x, dtype=bool), np.asarray(z, dtype=bool)):
        factor = np.eye(2, dtype=complex)
        if xq:
            factor = factor @ pauli_x
        if zq:
            factor = factor @ pauli_z
        matrix = np.kron(matrix, factor)
    return (1j ** int(k)) * matrix


class GF2Basis:
    """Incremental GF(2) basis on packed integer vectors.

    Rows are kept with distinct leading bits.  Reducing a vector against the rows
    in descending order of their leading bit clears every leading bit, so a vector
    lies in the span exactly when it reduces to zero.  A newly added row is stored
    already reduced, which lets :meth:`absorb_into` update vectors reduced against
    the old basis with a single pass.
    """

    def __init__(self) -> None:
        self._rows: list[tuple[int, int]] = []  # (leading bit, row), descending

    @property
    def rank(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[int]:
        return [row for _, row in self._rows]

    def reduce(self, vector: int) -> int:
        vector = int(vector)
        for lead, row in self._rows:
            if (vector >> lead) & 1:
                vector ^= row
        return vector

    def add(self, vector: int) -> int | None:
        """Add ``vector`` if independent; return the stored reduced row or ``None``."""
        reduced = self.reduce(vector)
        if reduced == 0:
            return None
        lead = reduced.bit_length() - 1
        self._rows.append((lead, reduced))
        self._rows.sort(reverse=True)
        return reduced

    def reduce_many(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.array(vectors, dtype=np.int64, copy=True)
        for lead, row in self._rows:
            hit = ((vectors >> lead) & 1).astype(bool)
            vectors[hit] ^= np.int64(row)
        return vectors

    @staticmethod
    def absorb_into(reduced_vectors: np.ndarray, new_row: int) -> None:
        """Update, in place, vectors already reduced against the basis before
        ``new_row`` (the value returned by :meth:`add`) was added."""
        lead = int(new_row).bit_length() - 1
        hit = ((reduced_vectors >> lead) & 1).astype(bool)
        reduced_vectors[hit] ^= np.int64(new_row)


def gf2_nullspace(rows: Sequence[int], width: int) -> list[int]:
    """Basis of ``{v : popcount(v & row) even for every row}``, as packed integers."""
    pivots: dict[int, int] = {}  # pivot column -> row with that pivot, fully reduced
    for row in rows:
        row = int(row)
        for col, pivot_row in pivots.items():
            if (row >> col) & 1:
                row ^= pivot_row
        if row == 0:
            continue
        col = row.bit_length() - 1
        for other in list(pivots):
            if (pivots[other] >> col) & 1:
                pivots[other] ^= row
        pivots[col] = row
    basis = []
    for free in range(width):
        if free in pivots:
            continue
        vector = 1 << free
        for col, pivot_row in pivots.items():
            if (pivot_row >> free) & 1:
                vector |= 1 << col
        basis.append(vector)
    return basis
