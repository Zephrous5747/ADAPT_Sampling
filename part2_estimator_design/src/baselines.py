"""External baselines at shot level: static M1 and independent-arm BAI (M2).

Part I evaluates M1 (fully commuting universal groups, all-gradient estimation)
and M2 (best-arm identification with every gradient measured on its own) as
planning bounds with exact variances, or as a Gaussian surrogate.  Paper A's
headline comparisons against these baselines therefore never ran them the way
Part II's own methods run: from sampled measurement outcomes, with estimated
variances.  This module closes that gap.

``StaticM1``
    One allocation for the whole pool at a common radius (the "mild oracle"
    handed to M1 in Part I: the exact gap, or ``rho max|g| / 2`` along a
    trajectory), one multinomial draw per context, ``argmax |g_hat|``.  The
    allocation uses the exact fragment variances, as Part I's M1 does, so this is
    static estimation at its best.  The *sequential, non-oracle* M1 (uniform
    precision for every gradient, shrinking radius, the same rules and estimated
    variances as II-0) is ``LearnedM3`` with ``elimination="off"``.

``IndependentBAI``
    M2: every gradient is one arm with its own fully commuting groups
    (:meth:`GradientProblem.individual_fc_groups`) and its own shots, so nothing is
    shared.  Within an arm, the shots of the groups follow the optimal independent
    allocation ``n_alpha ~ sigma_alpha`` (total ``(sum sigma_alpha)^2 / eps^2``).
    The variances are estimated from the arm's own data; a group with fewer than
    ``min_shots`` shots uses the Popoviciu bound ``sd <= (max f - min f) / 2`` on its
    fragment value ``f(b)``, which is known without measuring (the circuit and the
    coefficients fix ``f``), the analogue of the ``k I`` bound of the shared learner.
    Elimination, stopping and the ``bound`` starting radius are the rules of
    :mod:`rules` and :mod:`learning`, so the comparison with II-0 isolates sharing.

A group's measurement is a Clifford circuit to a computational-basis readout, so an
arm's fragment on one shot is a function ``f_alpha(b)`` of the outcome ``b``:
``f_alpha = WHT(v)`` with ``v[zmask_l] += A_il sign_l``.  The sufficient statistics of a
group are ``sum f`` and ``sum f^2`` over its shots, which is all that is stored.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from allocation import allocate
from clifford import complete_generators, independent_subset, synthesise_measurement_circuit
from online import OnlineConfig, OnlineM3, OnlineOutcome
from part1_bridge import epsilon_from_radius, z_from_delta
from rules import RULES, eliminate, rho_good_stop
from sampler import walsh_hadamard
from symplectic import masks_from_xz, pack, xz_from_labels

CHUNK = 256  # contexts per multinomial draw: bounds the transient histogram memory


class StaticM1(OnlineM3):
    """M1 with sampled outcomes: one allocation at a common radius, one draw."""

    def __init__(self, problem, library, moments, design, *, radius: float, delta: float = 0.05) -> None:
        super().__init__(problem, library, moments, design, OnlineConfig(rule="marginal", delta=delta))
        self.cz = library.two_qubit_counts()
        self.radius = float(radius)
        wanted = allocate(self.sigmas, epsilon_from_radius(self.radius, self.z))
        self.planned = np.maximum(np.ceil(wanted - 1e-9).astype(np.int64), self._minimum)
        funded = self.planned > 0
        variance = (self.squared[:, funded] / self.planned[funded]).sum(axis=1)
        self.standard_deviations = np.sqrt(variance)

    def run(self, rng: np.random.Generator, *, record: bool = False) -> OnlineOutcome:
        truth = self.problem.abs_gradients
        shots = self.planned
        counts = rng.multinomial(shots, self.distributions)
        means = walsh_hadamard(counts) / np.maximum(shots, 1)[:, None]
        contributions = self._weight * means[self._ctx, self._zmask]
        estimates = np.bincount(self._gen, weights=contributions, minlength=self.n_arms)
        selected = int(np.argmax(np.abs(estimates)))
        miss = np.abs(estimates - self.problem.gradients) > self.z * self.standard_deviations
        extra = {
            "shortfall": float(1.0 - truth[selected] / truth.max()) if truth.max() > 0 else 0.0,
            "miscovered_rounds": int(miss.any()),
            "best_eliminated": False,
            "stopped_rho": False,
            "contexts_used": int((shots > 0).sum()),
            "cz_per_shot_mean": float((shots * self.cz).sum() / max(shots.sum(), 1)),
        }
        return OnlineOutcome(selected, selected == int(np.argmax(truth)), float(shots.sum()), 1, extra=extra)


# --- M2: independent arms ----------------------------------------------------------------


def neyman_topup(sigma: np.ndarray, epsilon: float, lower: np.ndarray) -> np.ndarray:
    """Cheapest ``n >= lower`` with ``sum sigma^2 / n <= epsilon^2``: ``n = max(lower, t sigma)``.

    The variance is monotone in ``t``, and ``t* = sum(sigma) / epsilon^2`` (the optimum
    without lower bounds) is always feasible, so the smallest feasible ``t`` is found by
    bisection in ``(0, t*]``.
    """
    sigma = np.asarray(sigma, dtype=float)
    lower = np.asarray(lower, dtype=float)
    positive = sigma > 0
    if not positive.any():
        return lower.copy()
    s = sigma[positive]
    low_bound = lower[positive]
    target = epsilon ** 2

    def variance(t: float) -> float:
        n = np.maximum(low_bound, t * s)
        return float(np.sum(s ** 2 / n))

    high = float(s.sum()) / target
    if np.all(low_bound > 0) and float(np.sum(s ** 2 / low_bound)) <= target:
        return lower.copy()
    low = 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if mid > 0 and variance(mid) > target:
            low = mid
        else:
            high = mid
        if high - low <= 1e-13 * high:
            break
    return np.maximum(lower, high * sigma)


class IndependentContexts:
    """Every arm's own measurement groups: circuits' fragment values, state-independent.

    ``fvalues[c, b]`` is the arm's fragment on outcome ``b`` of group ``c``; ``arm_of[c]``
    the arm; ``bound[c]`` the Popoviciu bound on the fragment's standard deviation.  The
    outcome distributions depend on the state and are computed by :meth:`set_state`.
    """

    def __init__(self, problem, *, progress: bool = False) -> None:
        n = problem.n_qubits
        self.n_qubits = n
        self.dim = 2 ** n
        self.n_arms = problem.n_generators
        groups = problem.individual_fc_groups()
        arm_of, fvalues, cz, circuits = [], [], [], []
        started = time.perf_counter()
        for arm, (terms, arm_groups) in enumerate(zip(problem.commutator_terms, groups)):
            for group in arm_groups:
                x, z = xz_from_labels(group)
                packed = pack(*masks_from_xz(x, z), n)
                generators = [int(packed[p]) for p in independent_subset(packed)]
                generators, _ = complete_generators(generators, n, None)
                circuit = synthesise_measurement_circuit(generators, n)
                zmask, sign = circuit.diagonal_images(x, z)
                coefficient = np.array([terms[label] for label in group])
                v = np.zeros(self.dim)
                np.add.at(v, zmask, coefficient * sign)
                fvalues.append(walsh_hadamard(v))
                arm_of.append(arm)
                cz.append(circuit.two_qubit_count)
                circuits.append(circuit)
            if progress and (arm + 1) % 20 == 0:
                print(f"  independent groups: {arm + 1}/{self.n_arms} arms "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        self.arm_of = np.array(arm_of, dtype=np.int64)
        self.fvalues = np.array(fvalues)
        self.cz = np.array(cz)
        self.circuits = circuits
        self.bound = 0.5 * (self.fvalues.max(axis=1) - self.fvalues.min(axis=1))
        self.arm_contexts = [np.flatnonzero(self.arm_of == a) for a in range(self.n_arms)]
        self.distributions: np.ndarray | None = None

    @property
    def n_contexts(self) -> int:
        return int(self.arm_of.size)

    def set_state(self, state: np.ndarray) -> None:
        """Outcome distribution of every group on ``state`` (replaces the previous state's)."""
        out = np.empty((self.n_contexts, self.dim))
        for c, circuit in enumerate(self.circuits):
            out[c] = circuit.outcome_distribution(state)
        self.distributions = out

    def exact_gradients(self) -> np.ndarray:
        """``sum_b p_c(b) f_c(b)`` summed over each arm's groups (a consistency check)."""
        means = (self.distributions * self.fvalues).sum(axis=1)
        return np.bincount(self.arm_of, weights=means, minlength=self.n_arms)


@dataclass(frozen=True)
class IndependentConfig:
    rule: str = "marginal"  # successive elimination on |g| (marginal) or the sign-aware pairwise rule
    shrink: float = 0.9
    delta: float = 0.05
    rho: float = 0.0
    start: str = "bound"  # or "oracle": max_i |g_i|
    min_shots: int = 50  # fewer shots in a group: use the Popoviciu bound
    anytime: bool = False

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.start not in ("oracle", "bound"):
            raise ValueError("start must be 'oracle' or 'bound'")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")

    @property
    def label(self) -> str:
        extras = [f"rule={self.rule}"]
        extras += [f"rho={self.rho:g}"] if self.rho > 0 else []
        extras += [f"start={self.start}"] if self.start != "bound" else []
        extras += ["anytime"] if self.anytime else []
        return "M2 independent/" + "/".join(extras)


class IndependentBAI:
    """Successive elimination with every gradient measured on its own (Part I's M2), sampled."""

    def __init__(self, problem, contexts: IndependentContexts, config: IndependentConfig) -> None:
        if contexts.distributions is None:
            raise ValueError("call contexts.set_state(state) first")
        self.problem = problem
        self.contexts = contexts
        self.config = config
        self.n_arms = problem.n_generators
        self.z = z_from_delta(config.delta, self.n_arms)
        self.bound = max(sum(abs(v) for v in terms.values()) for terms in problem.commutator_terms)

    def _sd(self, m, s1, s2) -> np.ndarray:
        """Per-group standard deviation: sample estimate, or the Popoviciu bound below ``min_shots``."""
        sd = self.contexts.bound.copy()
        enough = m >= max(self.config.min_shots, 2)
        mean = s1[enough] / m[enough]
        variance = np.maximum(s2[enough] / m[enough] - mean ** 2, 0.0) * m[enough] / (m[enough] - 1.0)
        sd[enough] = np.minimum(np.sqrt(variance), self.contexts.bound[enough] * 1.0 + 1e-300)
        return sd

    def run(self, rng: np.random.Generator) -> OnlineOutcome:
        cfg, ctx = self.config, self.contexts
        truth = self.problem.abs_gradients
        leader = int(np.argmax(truth))
        z = self.z
        active = list(range(self.n_arms))
        if cfg.start == "oracle":
            radius = float(truth.max())
            floor = radius * 1e-3
        else:
            radius = float(self.bound)
            floor = radius * 1e-6
        m = np.zeros(ctx.n_contexts, dtype=np.int64)
        s1 = np.zeros(ctx.n_contexts)
        s2 = np.zeros(ctx.n_contexts)
        estimates = np.zeros(self.n_arms)
        miscovered = 0
        best_eliminated = False
        stopped = False
        rounds = 0
        seconds = {"allocation": 0.0, "statistics": 0.0}
        for rounds in range(1, 401):
            if stopped or len(active) <= 1 or radius < floor:
                break
            if cfg.anytime:
                z = z_from_delta(cfg.delta * 6.0 / (math.pi ** 2 * rounds ** 2), self.n_arms)
            clock = time.perf_counter()
            epsilon = epsilon_from_radius(radius, z)
            sd = self._sd(m, s1, s2)
            wanted = m.astype(float)
            for arm in active:
                c = ctx.arm_contexts[arm]
                wanted[c] = np.maximum(wanted[c], neyman_topup(sd[c], epsilon, m[c].astype(float)))
            target = np.maximum(m, np.ceil(wanted - 1e-9).astype(np.int64))
            for arm in active:
                c = ctx.arm_contexts[arm]
                target[c] = np.maximum(target[c], 2)  # a variance needs two shots
            seconds["allocation"] += time.perf_counter() - clock

            added = target - m
            grew = np.flatnonzero(added > 0)
            for start in range(0, grew.size, CHUNK):
                part = grew[start:start + CHUNK]
                counts = rng.multinomial(added[part], ctx.distributions[part])
                f = ctx.fvalues[part]
                s1[part] += (counts * f).sum(axis=1)
                s2[part] += (counts * f * f).sum(axis=1)
            m = target

            clock = time.perf_counter()
            counted = m > 0
            mean = np.zeros(ctx.n_contexts)
            mean[counted] = s1[counted] / m[counted]
            estimates = np.bincount(ctx.arm_of, weights=mean, minlength=self.n_arms)
            sd = self._sd(m, s1, s2)
            variance = np.bincount(ctx.arm_of, weights=np.where(counted, sd ** 2 / np.maximum(m, 1), np.inf),
                                   minlength=self.n_arms)
            covariance = np.diag(variance[active])
            seconds["statistics"] += time.perf_counter() - clock

            decision = eliminate(active, estimates, covariance, z, cfg.rule)
            radii = z * np.sqrt(variance[active])
            miscovered += int(bool((np.abs(estimates[active] - self.problem.gradients[active]) > radii).any()))
            best_eliminated |= any(arm == leader for arm, _, _ in decision.eliminated)
            survivors = decision.survivors
            if cfg.rho > 0 and len(survivors) > 1:
                keep = [active.index(arm) for arm in survivors]
                stopped = rho_good_stop(survivors, estimates, covariance[np.ix_(keep, keep)], z, cfg.rho)
            active = survivors
            radius *= cfg.shrink

        selected = max(active, key=lambda i: abs(estimates[i]))
        extra = {
            "shortfall": float(1.0 - truth[selected] / truth.max()) if truth.max() > 0 else 0.0,
            "miscovered_rounds": miscovered,
            "best_eliminated": bool(best_eliminated),
            "stopped_rho": bool(stopped),
            "contexts_used": int((m > 0).sum()),
            "cz_per_shot_mean": float((m * ctx.cz).sum() / max(m.sum(), 1)),
            **{f"{k}_seconds": round(v, 3) for k, v in seconds.items()},
        }
        return OnlineOutcome(selected, selected == leader, float(m.sum()), rounds, extra=extra)
