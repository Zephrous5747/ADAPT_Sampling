"""Measurement circuits for fully commuting contexts.

Measuring a context means applying a Clifford ``U`` and reading out all ``n``
qubits in the computational basis.  That yields the eigenvalue of every
``U^dagger Z_j U``, and these generate a *maximal* abelian group of ``2**n``
Pauli operators.  Every shot therefore measures the whole group, not just the
Pauli strings the context was built from, and which maximal group a context
measures is fixed only once its generators are completed to ``n`` independent
commuting ones.  This module does that completion and synthesises ``U``.

Synthesis follows the standard graph-state route for a maximal isotropic set of
``n`` generators, tracked only in the binary tableau ``[X | Z]``:

1. row-reduce ``X``; Hadamards on the non-pivot qubits make ``X`` invertible
   (the zero-``X`` rows then have an invertible ``Z`` block on those qubits);
2. row-reduce to ``X = I``, which forces ``Z`` symmetric;
3. ``S`` on every qubit with ``Z_qq = 1`` and ``CZ`` on every pair with
   ``Z_ab = 1`` clear ``Z``, leaving generators ``+-X_j``;
4. Hadamards on all qubits turn them into ``+-Z_j``.

Row operations only change which generators describe the group, so signs are not
tracked during synthesis.  Instead every Pauli of interest is pushed through the
finished gate list with exact phase bookkeeping (:meth:`MeasurementCircuit.conjugate`),
which gives its sign and its ``Z``-string image directly and is checked against
dense matrices in the tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from symplectic import (
    GF2Basis,
    anticommutes,
    bit_weights,
    gf2_nullspace,
    hermitian_phase,
    unpack,
    xz_from_masks,
)

INV_SQRT2 = 1.0 / math.sqrt(2.0)

Gate = tuple  # ("H", q) | ("S", q) | ("CZ", a, b)


@dataclass(frozen=True)
class MeasurementCircuit:
    """A diagonalising Clifford as an ordered gate list over ``H``, ``S``, ``CZ``."""

    n_qubits: int
    gates: tuple[Gate, ...]

    @property
    def two_qubit_count(self) -> int:
        """Number of ``CZ`` gates; each costs one ``CNOT`` plus single-qubit gates."""
        return sum(1 for gate in self.gates if gate[0] == "CZ")

    @property
    def single_qubit_count(self) -> int:
        return sum(1 for gate in self.gates if gate[0] != "CZ")

    def conjugate(
        self, x: np.ndarray, z: np.ndarray, k: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Push ``i**k X**x Z**z`` through the circuit: returns ``U P U^dagger``.

        Update rules, derived in the representation of :mod:`symplectic`:

        * ``H``:  ``(x, z) -> (z, x)``,       ``k += 2 x z``;
        * ``S``:  ``z ^= x``,                  ``k += x``        (``X -> Y``);
        * ``CZ(a, b)``: ``z_a ^= x_b``, ``z_b ^= x_a``, ``k += 2 x_a x_b``.
        """
        x = np.array(np.atleast_2d(x), dtype=np.int64, copy=True)
        z = np.array(np.atleast_2d(z), dtype=np.int64, copy=True)
        k = hermitian_phase(x, z) if k is None else np.array(k, dtype=np.int64, copy=True)
        for gate in self.gates:
            kind = gate[0]
            if kind == "H":
                q = gate[1]
                xq, zq = x[:, q].copy(), z[:, q].copy()
                k += 2 * (xq & zq)
                x[:, q], z[:, q] = zq, xq
            elif kind == "S":
                q = gate[1]
                k += x[:, q]
                z[:, q] ^= x[:, q]
            elif kind == "CZ":
                a, b = gate[1], gate[2]
                k += 2 * (x[:, a] & x[:, b])
                z[:, a] ^= x[:, b]
                z[:, b] ^= x[:, a]
            else:  # pragma: no cover - construction never emits anything else
                raise ValueError(f"unknown gate {gate!r}")
        return x.astype(bool), z.astype(bool), k % 4

    def diagonal_images(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(zmask, sign)`` with ``U P U^dagger = sign * Z**zmask`` for Hermitian ``P``.

        Raises if any ``P`` is not mapped to a diagonal operator, which means it is
        not in the group this circuit measures.
        """
        xo, zo, k = self.conjugate(x, z)
        if xo.any():
            bad = int(np.flatnonzero(xo.any(axis=1))[0])
            raise ValueError(f"Pauli {bad} is not diagonalised by this circuit")
        if (k % 2).any():  # pragma: no cover - impossible for Hermitian input
            raise ValueError("non-Hermitian image; phase bookkeeping is broken")
        zmask = zo.astype(np.int64) @ bit_weights(self.n_qubits)
        return zmask, np.where(k == 0, 1.0, -1.0)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """``U |state>`` for a big-endian state vector."""
        n = self.n_qubits
        psi = np.array(state, dtype=complex, copy=True).reshape((2,) * n)
        for gate in self.gates:
            kind = gate[0]
            if kind == "H":
                q = gate[1]
                moved = np.moveaxis(psi, q, 0)
                a, b = moved[0], moved[1]
                psi = np.moveaxis(np.stack((a + b, a - b)) * INV_SQRT2, 0, q)
            elif kind == "S":
                index = [slice(None)] * n
                index[gate[1]] = 1
                psi[tuple(index)] *= 1j
            else:
                index = [slice(None)] * n
                index[gate[1]] = 1
                index[gate[2]] = 1
                psi[tuple(index)] *= -1.0
            psi = np.ascontiguousarray(psi)
        return psi.reshape(-1)

    def outcome_distribution(self, state: np.ndarray) -> np.ndarray:
        """Probability of every computational-basis outcome after ``U``."""
        probabilities = np.abs(self.apply(state)) ** 2
        return probabilities / probabilities.sum()

    def dense(self) -> np.ndarray:
        """Dense unitary (tests only)."""
        dim = 2 ** self.n_qubits
        return np.column_stack([self.apply(column) for column in np.eye(dim, dtype=complex)])


def independent_subset(vectors: Sequence[int]) -> list[int]:
    """Positions of a maximal independent subset, chosen greedily in order."""
    basis = GF2Basis()
    return [position for position, v in enumerate(vectors) if basis.add(int(v)) is not None]


def _symplectic_rows(generators: Sequence[int], n: int) -> list[int]:
    """Rows whose GF(2) dot product with ``v`` is the symplectic form ``w(v, g)``."""
    rows = []
    low = (1 << n) - 1
    for g in generators:
        g = int(g)
        rows.append(((g & low) << n) | (g >> n))
    return rows


def single_qubit_candidates(n: int) -> np.ndarray:
    """Packed ``Z_q``, then ``X_q``, then ``Y_q``: the canonical completion order."""
    weights = [1 << (n - 1 - q) for q in range(n)]
    zs = [w for w in weights]
    xs = [w << n for w in weights]
    ys = [(w << n) | w for w in weights]
    return np.array(zs + xs + ys, dtype=np.int64)


def complete_generators(
    generators: Sequence[int],
    n: int,
    candidates: np.ndarray | None = None,
) -> tuple[list[int], int]:
    """Complete independent commuting generators to a maximal set of ``n``.

    ``candidates`` are packed Paulis in order of preference.  Each step adds the
    first candidate that commutes with every current generator and is not already
    in their span; single-qubit ``Z``, ``X`` and ``Y`` are tried after the given
    candidates, and a symplectic-complement basis vector is the last resort, so the
    completion always succeeds.  Returns the generators and how many came from the
    preferred list.
    """
    gens = [int(g) for g in generators]
    basis = GF2Basis()
    for g in gens:
        if basis.add(g) is None:
            raise ValueError("generators are not independent")
    xm, zm = unpack(np.array(gens, dtype=np.int64), n) if gens else (np.zeros(0), np.zeros(0))
    for gx, gz in zip(xm, zm):
        if anticommutes(xm, zm, gx, gz).any():
            raise ValueError("generators do not commute")

    from_preferred = 0
    for pool_index, pool in enumerate((candidates, single_qubit_candidates(n))):
        if pool is None or len(gens) >= n:
            continue
        pool = np.asarray(pool, dtype=np.int64)
        cx, cz = unpack(pool, n)
        ok = np.ones(pool.size, dtype=bool)
        for g in gens:
            gx, gz = unpack(np.int64(g), n)
            ok &= ~anticommutes(cx, cz, gx, gz)
        reduced = basis.reduce_many(pool)
        while len(gens) < n:
            valid = np.flatnonzero(ok & (reduced != 0))
            if valid.size == 0:
                break
            chosen = int(pool[valid[0]])
            row = basis.add(chosen)
            gens.append(chosen)
            if pool_index == 0:
                from_preferred += 1
            GF2Basis.absorb_into(reduced, row)
            gx, gz = unpack(np.int64(chosen), n)
            ok &= ~anticommutes(cx, cz, gx, gz)

    while len(gens) < n:
        commutant = gf2_nullspace(_symplectic_rows(gens, n), 2 * n)
        chosen = next(v for v in commutant if basis.reduce(v) != 0)
        basis.add(chosen)
        gens.append(int(chosen))
    return gens, from_preferred


def synthesise_measurement_circuit(generators: Sequence[int], n: int) -> MeasurementCircuit:
    """Diagonalising circuit for ``n`` independent commuting generators."""
    if len(generators) != n:
        raise ValueError("synthesis needs a maximal set of n generators")
    gx, gz = unpack(np.array([int(g) for g in generators], dtype=np.int64), n)
    x_bits, z_bits = xz_from_masks(gx, gz, n)
    X = x_bits.astype(np.uint8)
    Z = z_bits.astype(np.uint8)
    gates: list[Gate] = []

    # 1. pivot columns of X; Hadamards elsewhere make X invertible.
    Xr, Zr = X.copy(), Z.copy()
    pivots = []
    row = 0
    for col in range(n):
        hits = np.flatnonzero(Xr[row:, col]) + row if row < n else np.array([], dtype=int)
        if hits.size == 0:
            continue
        r = int(hits[0])
        Xr[[row, r]] = Xr[[r, row]]
        Zr[[row, r]] = Zr[[r, row]]
        for other in np.flatnonzero(Xr[:, col]):
            if other != row:
                Xr[other] ^= Xr[row]
                Zr[other] ^= Zr[row]
        pivots.append(col)
        row += 1
    for q in range(n):
        if q not in pivots:
            gates.append(("H", q))
            X[:, q], Z[:, q] = Z[:, q].copy(), X[:, q].copy()

    # 2. Gauss-Jordan to X = I (row operations only; the group is unchanged).
    for col in range(n):
        hits = np.flatnonzero(X[col:, col]) + col
        if hits.size == 0:  # pragma: no cover - excluded by step 1
            raise ValueError("X block is singular after the Hadamard layer")
        r = int(hits[0])
        X[[col, r]] = X[[r, col]]
        Z[[col, r]] = Z[[r, col]]
        for other in np.flatnonzero(X[:, col]):
            if other != col:
                X[other] ^= X[col]
                Z[other] ^= Z[col]
    if not np.array_equal(Z, Z.T):
        raise ValueError("generators do not commute (Z block not symmetric)")

    # 3. S clears the diagonal, CZ the off-diagonal.
    for q in range(n):
        if Z[q, q]:
            gates.append(("S", q))
    for a in range(n):
        for b in range(a + 1, n):
            if Z[a, b]:
                gates.append(("CZ", a, b))

    # 4. X_j -> Z_j.
    gates.extend(("H", q) for q in range(n))
    circuit = MeasurementCircuit(n, tuple(gates))
    circuit.diagonal_images(x_bits, z_bits)  # hard check: every generator is diagonal
    return circuit
