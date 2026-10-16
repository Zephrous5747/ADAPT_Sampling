"""Noisy measurement circuits: a depolarising error after every entangling gate, and readout error.

The shot-level studies so far sample the *ideal* outcome distribution of every context.  The
circuits differ a lot in depth (a QWC context has no entangling gate, a fully commuting one 7 to 30), and
Part II's intervals cover shot noise only.  This module makes the measurement circuits noisy, in the
way that stays exact and cheap for Clifford circuits: a Pauli-twirled channel scales the expectation of
every Pauli by a factor that depends only on which gates it passes through.

Model.
* After every ``CZ`` a two-qubit depolarising channel of total error probability ``p2`` acts (every one
  of the 15 non-identity Paulis with probability ``p2 / 15``).  Its action on a Pauli observable is a
  factor ``1 - 16 p2 / 15`` if the observable, propagated back to that point of the circuit, is not the
  identity on the gate's two qubits, and 1 otherwise.
* Readout error: every qubit's outcome is flipped with probability ``pr``, which multiplies the
  expectation of a ``Z`` string of weight ``w`` by ``(1 - 2 pr)**w``.
* Single-qubit gates are ideal and the state preparation is ideal (the state is the one whose
  gradients are being selected).

For a context with outcome distribution ``p`` the noisy distribution has the group means
``E'[z] = d(z) E[z]`` with ``d(z)`` the factor of the Pauli whose image is ``Z**z``;
:func:`damping` computes ``d`` for all ``2**n`` elements of the measured group at once, and
:class:`NoisyMoments` replaces ``p`` by the distribution with those means.  Everything downstream (the
sampling, the estimated covariances, the designs) then sees the noisy data, while the exact gradients of
the problem stay the noiseless ones, so a selection is correct only if it picks the best *true* generator.
The bias of the estimates is the point: the confidence radii are built for shot noise and do not know
about it.
"""
from __future__ import annotations

import numpy as np

from sampler import OracleMoments, walsh_hadamard

TWO_QUBIT_DEPOLARISING = 16.0 / 15.0


def damping(circuit, two_qubit_error: float, readout_error: float) -> np.ndarray:
    """Factor ``d(z)`` for every ``Z**z`` at the output of the circuit (index ``z`` in the big-endian convention)."""
    n = circuit.n_qubits
    size = 1 << n
    index = np.arange(size, dtype=np.int64)
    z = ((index[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)  # (2^n, n): the output Pauli Z**z
    x = np.zeros_like(z)
    factor = np.ones(size)
    gate_factor = 1.0 - TWO_QUBIT_DEPOLARISING * two_qubit_error
    for gate in reversed(circuit.gates):  # Heisenberg picture: from the readout back to the preparation
        kind = gate[0]
        if kind == "CZ":
            a, b = gate[1], gate[2]
            touched = (x[:, a] | z[:, a] | x[:, b] | z[:, b]).astype(bool)
            factor = np.where(touched, factor * gate_factor, factor)
            z[:, a] ^= x[:, b]
            z[:, b] ^= x[:, a]
        elif kind == "H":
            q = gate[1]
            x[:, q], z[:, q] = z[:, q].copy(), x[:, q].copy()
        else:  # S: the support is all that matters here
            q = gate[1]
            z[:, q] ^= x[:, q]
    if readout_error:
        weight = np.bitwise_count(index).astype(float)
        factor = factor * (1.0 - 2.0 * readout_error) ** weight
    return factor


def noisy_distribution(distribution: np.ndarray, damp: np.ndarray) -> np.ndarray:
    """The outcome distribution whose group means are ``damp * WHT(distribution)``."""
    means = walsh_hadamard(distribution) * damp
    out = walsh_hadamard(means) / distribution.size
    out = np.clip(out, 0.0, None)
    return out / out.sum()


class NoisyMoments(OracleMoments):
    """Oracle moments of the noisy circuits: same interface, damped distributions."""

    def __init__(self, library, state, two_qubit_error: float = 0.0, readout_error: float = 0.0) -> None:
        super().__init__(library, state)
        self.two_qubit_error = float(two_qubit_error)
        self.readout_error = float(readout_error)

    def distribution(self, alpha: int) -> np.ndarray:
        if alpha not in self._distributions:
            circuit = self.library.contexts[alpha].circuit
            ideal = circuit.outcome_distribution(self.state)
            damp = damping(circuit, self.two_qubit_error, self.readout_error)
            self._distributions[alpha] = noisy_distribution(ideal, damp)
        return self._distributions[alpha]
