"""Numeric action of Pauli strings on state vectors.

A Pauli string is a text string over ``I``, ``X``, ``Y``, ``Z`` whose character
``i`` describes qubit ``i``.  Qubit ``i`` is stored in bit ``n_qubits - 1 - i``
of the computational-basis index, which is the convention used by
:func:`openfermion.linalg.get_sparse_operator`.  Mixing this up with the
little-endian convention silently produces the wrong state, so every module in
this package goes through :class:`PauliEvaluator` rather than indexing by hand.

The evaluator never materialises an operator matrix.  A Pauli string factorises
as ``P = i**n_y * X(x_mask) * Z(z_mask)``, so ``P|psi>`` is one permutation and
one sign flip of the state vector, which keeps fragment variances affordable for
the large universal supports in Part I.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

PAULI_CHARS = frozenset("IXYZ")


def _popcount(values: np.ndarray) -> np.ndarray:
    """Population count for an integer array, with a fallback for older NumPy."""
    bitwise_count = getattr(np, "bitwise_count", None)
    if bitwise_count is not None:
        return bitwise_count(values)
    counts = np.zeros_like(values)
    remaining = values.copy()
    while remaining.any():
        counts += remaining & 1
        remaining >>= 1
    return counts


@dataclass(frozen=True)
class PauliMask:
    """Bit-mask form of a Pauli string: ``P = i**n_y * X(x_mask) * Z(z_mask)``."""

    x_mask: int
    z_mask: int
    n_y: int


def pauli_masks(pauli: str) -> PauliMask:
    """Convert a Pauli string to its bit-mask form."""
    n_qubits = len(pauli)
    x_mask = z_mask = n_y = 0
    for position, char in enumerate(pauli):
        if char not in PAULI_CHARS:
            raise ValueError(f"invalid Pauli character {char!r} in {pauli!r}")
        if char == "I":
            continue
        bit = 1 << (n_qubits - 1 - position)
        if char in ("X", "Y"):
            x_mask |= bit
        if char in ("Z", "Y"):
            z_mask |= bit
        if char == "Y":
            n_y += 1
    return PauliMask(x_mask, z_mask, n_y)


class PauliEvaluator:
    """Expectation values and fragment variances in one fixed state."""

    def __init__(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=complex)
        if self.state.ndim != 1:
            raise ValueError("state must be a one-dimensional vector")
        self.dim = int(self.state.size)
        self.n_qubits = int(round(math.log2(self.dim)))
        if 2 ** self.n_qubits != self.dim:
            raise ValueError("state dimension must be a power of two")
        self._indices = np.arange(self.dim, dtype=np.int64)

    def apply(self, pauli: str) -> np.ndarray:
        """Return ``P|psi>`` for a single Pauli string."""
        self._check_length(pauli)
        mask = pauli_masks(pauli)
        source = self._indices ^ mask.x_mask
        parity = _popcount(source & mask.z_mask) & 1
        signs = np.where(parity.astype(bool), -1.0, 1.0)
        return (1j ** mask.n_y) * signs * self.state[source]

    def expectation(self, pauli: str) -> float:
        """Return ``<psi|P|psi>``, which is real for a Hermitian Pauli string."""
        return float(np.vdot(self.state, self.apply(pauli)).real)

    def fragment_vector(self, terms: Mapping[str, float]) -> np.ndarray:
        """Return ``F|psi>`` for ``F = sum_l c_l R_l``."""
        result = np.zeros(self.dim, dtype=complex)
        for pauli, coefficient in terms.items():
            result += coefficient * self.apply(pauli)
        return result

    def fragment_mean_std(self, terms: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(<F>, sqrt(Var F))`` for ``F = sum_l c_l R_l``.

        ``F`` must be Hermitian; the caller guarantees this by passing real
        coefficients of a Hermitian operator.
        """
        if not terms:
            return 0.0, 0.0
        vector = self.fragment_vector(terms)
        mean = float(np.vdot(self.state, vector).real)
        # Centre before squaring.  Forming <F^2> - <F>^2 subtracts two large
        # nearly equal numbers whenever the state is close to an eigenstate of
        # F, which is exactly the case for the symmetry-forbidden fragments that
        # dominate a parent context.  The cancellation there is total: the
        # result is floating-point noise of either sign, and the max(0, .)
        # clamp that used to guard it hid the failure rather than fixing it.
        # ||(F - <F>)|psi>||^2 is the same quantity with no cancellation.
        centred = vector - mean * self.state
        variance = float(np.vdot(centred, centred).real)
        return mean, math.sqrt(max(0.0, variance))

    def fragment_std(self, terms: Mapping[str, float]) -> float:
        """Return the one-shot standard deviation of ``F = sum_l c_l R_l``."""
        return self.fragment_mean_std(terms)[1]

    def _check_length(self, pauli: str) -> None:
        if len(pauli) != self.n_qubits:
            raise ValueError(
                f"Pauli string of length {len(pauli)} does not match "
                f"{self.n_qubits} qubits"
            )
