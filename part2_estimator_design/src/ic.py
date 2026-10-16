"""Informationally complete (IC) measurement baseline: AIM-ADAPT-VQE's data model.

Nykanen et al. (Phys. Rev. Research 7, 043114; arXiv:2212.09719) measure every qubit with a
single-qubit IC POVM (four outcomes per qubit, ``4**n`` outcomes in all) and estimate *any*
observable ``O = sum_m w_m Pi_m`` as the sample mean of ``w_m`` over the outcomes.  The
energy and every commutator ``[H, G_i]`` of the pool come from the same shots by classical
post-processing; the POVM parameters are adapted to lower the variance of the energy, and the
gradient-based variant stops when the commutators' standard errors are small enough.

This module simulates that data model exactly for small systems:

* the POVM is the product of tetrahedral (SIC) single-qubit POVMs,
  ``Pi_m = (I + n_m . sigma) / 4`` with ``n_m`` the four tetrahedron vertices; the dual
  frame gives every Pauli ``sigma_a`` the single-shot estimator ``f_a(m) = 3 n_{m,a}``
  (``+-sqrt 3``; the identity gets 1), so a Pauli string of weight ``w`` has single-shot
  second moment ``3**w``.  The optimisation of the POVM parameters of the paper (which
  lowers the energy variance) is not modelled: this is the *unoptimised* member of the
  family, a stand-in with the paper's scaling, and is labelled so wherever it is used;
* the outcome distribution ``p(m) = <psi| Pi_m |psi>`` and the single-shot value
  ``G_i(m) = sum_l A_il prod_q f_{l_q}(m_q)`` of every gradient are computed exactly by
  per-qubit contractions (``O(n 4^n)``), so the exact covariance of the gradient estimators
  ``Sigma_ij = sum_m p(m) G_i(m) G_j(m) - g_i g_j`` is available (``oracle`` radii) and shots
  can be sampled from the exact multinomial (``estimated`` radii use the sample
  covariance);
* the cost is the number of IC shots ``S`` (one state preparation, all qubits measured in
  the IC POVM).  An IC shot needs an ancilla-assisted or randomised implementation, so shot
  counts are not directly comparable with context-shots; the comparison is of statistical
  efficiency, not of circuit cost.

Memory is ``K 4^n`` floats (``K = 92, n = 12``: 6 GB in float32); H2O (n = 14) is out of reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from part1_bridge import z_from_delta
from rules import RULES, eliminate, rho_good_stop

PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
VERTICES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / math.sqrt(3.0)
# N[m, a] = (1, n_m) ; F[a, m] = 1 for the identity and 3 n_{m,a} for X, Y, Z
N_MATRIX = np.hstack([np.ones((4, 1)), VERTICES])
F_MATRIX = np.vstack([np.ones((1, 4)), 3.0 * VERTICES.T])
SIGMA = np.array([np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
# Tr(sigma_a |i><j|) = sigma_a[j, i], indexed [a, 2 i + j]
PAULI_PROJECTION = np.stack([SIGMA[a].T.reshape(4) for a in range(4)])


def apply_each_axis(tensor: np.ndarray, matrix: np.ndarray, n: int) -> np.ndarray:
    """``(matrix tensor-product n)`` applied to a tensor of shape ``(4,) * n`` (flattened input ok)."""
    out = np.asarray(tensor).reshape((4,) * n)
    for axis in range(n):
        out = np.moveaxis(np.tensordot(matrix, out, axes=(1, axis)), 0, axis)
    return out.reshape(-1)


def pauli_expectations(state: np.ndarray, n: int) -> np.ndarray:
    """``mu[a_1 .. a_n] = <psi| sigma_{a_1} x ... x sigma_{a_n} |psi>`` as a flat real array of ``4**n``."""
    psi = np.asarray(state, dtype=complex).reshape((2,) * n)
    rho = np.multiply.outer(psi, psi.conj())  # axes i_1..i_n, j_1..j_n
    order = [axis for q in range(n) for axis in (q, n + q)]
    pairs = np.transpose(rho, order).reshape((4,) * n)  # axis q: 2 i_q + j_q
    mu = apply_each_axis(pairs, PAULI_PROJECTION, n)
    return mu.real


def outcome_distribution(mu: np.ndarray, n: int) -> np.ndarray:
    """``p(m) = 4^{-n} sum_a prod_q N[m_q, a_q] mu[a]``."""
    p = apply_each_axis(mu, N_MATRIX, n) / 4.0 ** n
    return p


def estimator_values(terms: dict[str, float], n: int) -> np.ndarray:
    """Single-shot values ``G(m) = sum_l A_l prod_q f_{l_q}(m_q)`` of an observable's estimator."""
    coefficients = np.zeros(4 ** n)
    for label, value in terms.items():
        index = 0
        for char in label:
            index = 4 * index + PAULI_CODE[char]
        coefficients[index] += value
    return apply_each_axis(coefficients, F_MATRIX.T, n)


class ICMeasurement:
    """One problem on one state: exact outcome distribution, estimator values and covariances."""

    def __init__(self, problem, *, hamiltonian_terms: dict[str, float] | None = None,
                 dtype=None) -> None:
        n = problem.n_qubits
        dtype = (np.float64 if n <= 10 else np.float32) if dtype is None else dtype
        if 4 ** n * problem.n_generators * np.dtype(dtype).itemsize > 40e9:
            raise MemoryError(f"IC data for K={problem.n_generators}, n={n} need more than 40 GB")
        self.n = n
        self.problem = problem
        mu = pauli_expectations(problem.evaluator.state, n)
        p = outcome_distribution(mu, n)
        if p.min() < -1e-12 or abs(p.sum() - 1.0) > 1e-9:
            raise AssertionError("IC outcome distribution is not a distribution")
        self.p = np.clip(p, 0.0, None)
        self.p /= self.p.sum()
        self.G = np.empty((problem.n_generators, 4 ** n), dtype=dtype)
        for i, terms in enumerate(problem.commutator_terms):
            self.G[i] = estimator_values(terms, n)
        self.means = self.apply(self.p)
        self.covariance = self._weighted_gram(self.p) - np.outer(self.means, self.means)
        self.energy_values = None
        if hamiltonian_terms is not None:
            self.energy_values = estimator_values(hamiltonian_terms, n)
            mean = float(self.energy_values @ self.p)
            self.energy_variance = float((self.p * self.energy_values ** 2).sum() - mean ** 2)

    def apply(self, weights: np.ndarray, chunk: int = 1 << 22) -> np.ndarray:
        """``G @ weights`` accumulated in float64 over chunks of outcomes (``G`` may be float32)."""
        out = np.zeros(self.G.shape[0])
        for start in range(0, weights.size, chunk):
            out += self.G[:, start:start + chunk].astype(np.float64) @ weights[start:start + chunk]
        return out

    def _weighted_gram(self, weights: np.ndarray, chunk: int = 1 << 22) -> np.ndarray:
        K = self.G.shape[0]
        gram = np.zeros((K, K))
        for start in range(0, weights.size, chunk):
            g = self.G[:, start:start + chunk].astype(np.float64)
            gram += (g * weights[start:start + chunk]) @ g.T
        return gram

    def energy_shots(self, epsilon: float) -> float:
        """IC shots for the energy to standard error ``epsilon`` (the data AIM already holds)."""
        if self.energy_values is None:
            raise ValueError("pass hamiltonian_terms to ICMeasurement")
        return self.energy_variance / epsilon ** 2


@dataclass(frozen=True)
class ICConfig:
    rule: str = "safe"
    shrink: float = 0.9  # the radius schedule of the other methods: shots grow by 1 / shrink**2 per round
    delta: float = 0.05
    rho: float = 0.0
    radii: str = "oracle"  # "oracle": exact covariance / S; "estimated": sample covariance
    start_shots: int = 100
    max_shots: float = 1e13
    credit: float = 0.0  # IC shots already held (the last energy evaluation); not charged
    anytime: bool = False  # delta_r = 6 delta / (pi^2 r^2) per round: valid over all looks

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.radii not in ("oracle", "estimated"):
            raise ValueError("radii must be 'oracle' or 'estimated'")

    @property
    def label(self) -> str:
        extras = [f"rule={self.rule}", f"{self.radii} radii"]
        extras += [f"rho={self.rho:g}"] if self.rho > 0 else []
        extras += ["anytime"] if self.anytime else []
        extras += [f"credit={self.credit:.3g}"] if self.credit else []
        return "IC (SIC product POVM, unoptimised)/" + "/".join(extras)


class ICSelection:
    """Best-arm identification on IC data: one shot informs every gradient, nothing to allocate."""

    def __init__(self, measurement: ICMeasurement, config: ICConfig) -> None:
        self.m = measurement
        self.config = config
        self.K = measurement.problem.n_generators
        self.z = z_from_delta(config.delta, self.K)

    def run(self, rng: np.random.Generator):
        from online import OnlineOutcome

        cfg, m = self.config, self.m
        truth = m.problem.abs_gradients
        leader = int(np.argmax(truth))
        alive = list(range(self.K))
        counts = np.zeros(m.p.size, dtype=np.int64)
        shots = 0
        held = int(cfg.credit)
        target = max(cfg.start_shots, held)
        estimates = np.zeros(self.K)
        stopped = False
        miscovered = 0
        best_eliminated = False
        rounds = 0
        growth = 1.0 / cfg.shrink ** 2
        while len(alive) > 1 and not stopped and target <= cfg.max_shots:
            rounds += 1
            z = (z_from_delta(cfg.delta * 6.0 / (math.pi ** 2 * rounds ** 2), self.K) if cfg.anytime else self.z)
            add = int(target) - shots
            if add > 0:
                counts += rng.multinomial(add, m.p)
                shots += add
            estimates = self._estimates(counts, shots)
            covariance = self._covariance(counts, shots)
            sub = covariance[np.ix_(alive, alive)]
            decision = eliminate(alive, estimates, sub, z, cfg.rule)
            radius = z * np.sqrt(np.maximum(np.diag(covariance), 0.0))
            miscovered += int(bool((np.abs(estimates - m.problem.gradients) > radius).any()))
            best_eliminated |= any(arm == leader for arm, _, _ in decision.eliminated)
            alive = decision.survivors
            if cfg.rho > 0 and len(alive) > 1:
                keep = alive
                stopped = rho_good_stop(alive, estimates, covariance[np.ix_(keep, keep)], z, cfg.rho)
            target = math.ceil(target * growth)
        selected = max(alive, key=lambda i: abs(estimates[i]))
        extra = {
            "shortfall": float(1.0 - truth[selected] / truth.max()) if truth.max() > 0 else 0.0,
            "miscovered_rounds": miscovered, "best_eliminated": bool(best_eliminated),
            "stopped_rho": bool(stopped), "gross_shots": float(shots),
        }
        return OnlineOutcome(selected, selected == leader, float(max(shots - held, 0)), rounds, extra=extra)

    def _estimates(self, counts: np.ndarray, shots: int) -> np.ndarray:
        return self.m.apply(counts.astype(np.float64)) / shots

    def _covariance(self, counts: np.ndarray, shots: int) -> np.ndarray:
        if self.config.radii == "oracle":
            return self.m.covariance / shots
        mean = self._estimates(counts, shots)
        second = self.m._weighted_gram(counts / shots)
        sample = (second - np.outer(mean, mean)) * shots / max(shots - 1, 1)
        return sample / shots
