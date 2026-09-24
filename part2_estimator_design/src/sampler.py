"""Outcome distributions, group moments and shot-level sampling.

After the measurement circuit ``U_alpha`` of a context, one shot is one
computational-basis outcome ``b`` drawn from ``p_alpha(b) = |<b|U_alpha|psi>|**2``.
Every element of the measured group is ``sign * U^dagger Z**zmask U``, so its
outcome on that shot is ``sign * (-1)**popcount(zmask & b)`` and the mean of *all*
``2**n`` group elements is one Walsh-Hadamard transform of ``p_alpha``:

    E_alpha[z] = sum_b p_alpha(b) (-1)**popcount(z & b).

Covariances follow without further work, because the product of two group
elements is again a group element:

    Cov(P_l, P_m) = s_l s_m (E[z_l ^ z_m] - E[z_l] E[z_m]).

The same two formulas give the oracle moments (exact ``p_alpha``) and the
empirical ones (a histogram of sampled outcomes), so the two can differ only in
their input.  Sampling a context ``m`` times is one multinomial draw over its
``2**n`` outcomes, which is exact shot-level sampling at a cost independent of
``m``; the histogram is a sufficient statistic for everything estimated here.
"""
from __future__ import annotations

import math

import numpy as np


def walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform along the last axis (length ``2**n``)."""
    values = np.asarray(values, dtype=float)
    size = values.shape[-1]
    n = int(round(math.log2(size))) if size else 0
    if 2 ** n != size:
        raise ValueError("last axis must have length 2**n")
    lead = values.shape[:-1]
    array = values.reshape(lead + (2,) * n)
    for axis in range(len(lead), len(lead) + n):
        first = np.take(array, 0, axis=axis)
        second = np.take(array, 1, axis=axis)
        array = np.stack((first + second, first - second), axis=axis)
    return array.reshape(values.shape)


def pauli_means(group_means: np.ndarray, zmask: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return sign * group_means[..., zmask]


def pauli_covariance(group_means: np.ndarray, zmask: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """One-shot covariance of the given group elements (exact or plug-in)."""
    means = group_means[zmask]
    products = group_means[zmask[:, None] ^ zmask[None, :]]
    return np.outer(sign, sign) * (products - np.outer(means, means))


class OracleMoments:
    """Exact outcome distributions and group means of every context (cached)."""

    def __init__(self, library, state: np.ndarray) -> None:
        self.library = library
        self.state = np.asarray(state, dtype=complex)
        self._distributions: dict[int, np.ndarray] = {}
        self._means: dict[int, np.ndarray] = {}

    def distribution(self, alpha: int) -> np.ndarray:
        if alpha not in self._distributions:
            circuit = self.library.contexts[alpha].circuit
            self._distributions[alpha] = circuit.outcome_distribution(self.state)
        return self._distributions[alpha]

    def group_means(self, alpha: int) -> np.ndarray:
        if alpha not in self._means:
            self._means[alpha] = walsh_hadamard(self.distribution(alpha))
        return self._means[alpha]

    def _images(self, alpha: int, paulis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        context = self.library.contexts[alpha]
        positions = context.positions(paulis)
        return context.member_zmask[positions], context.member_sign[positions]

    def means(self, alpha: int, paulis: np.ndarray) -> np.ndarray:
        zmask, sign = self._images(alpha, paulis)
        return pauli_means(self.group_means(alpha), zmask, sign)

    def covariance(self, alpha: int, paulis: np.ndarray) -> np.ndarray:
        zmask, sign = self._images(alpha, paulis)
        return pauli_covariance(self.group_means(alpha), zmask, sign)

    def distributions_matrix(self) -> np.ndarray:
        """``(n_contexts, 2**n)`` array of every outcome distribution."""
        return np.stack([self.distribution(a) for a in range(self.library.n_contexts)])


def empirical_group_means(counts: np.ndarray) -> np.ndarray:
    """Sample means of every group element from outcome histograms (last axis)."""
    counts = np.asarray(counts, dtype=float)
    shots = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(shots > 0, walsh_hadamard(counts) / np.maximum(shots, 1), 0.0)


def empirical_covariance(counts: np.ndarray, zmask: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of group elements from one outcome histogram."""
    shots = float(np.asarray(counts).sum())
    if shots < 2:
        raise ValueError("need at least two shots for a sample covariance")
    plug_in = pauli_covariance(empirical_group_means(counts), zmask, sign)
    return plug_in * shots / (shots - 1.0)


def sample_counts(
    rng: np.random.Generator, shots: np.ndarray, distributions: np.ndarray
) -> np.ndarray:
    """One multinomial histogram per context: exact shot-level sampling."""
    shots = np.asarray(shots, dtype=np.int64)
    if shots.ndim == 0:
        return rng.multinomial(int(shots), _normalised(distributions))
    return rng.multinomial(shots, _normalised(distributions))


def _normalised(distributions: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(distributions, dtype=float), 0.0, None)
    return p / p.sum(axis=-1, keepdims=True)
