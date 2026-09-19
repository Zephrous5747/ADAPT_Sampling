"""Finite-shot simulation of the Part I selection methods.

Every number in the main Part I tables is an oracle planning bound: the
elimination rule reads exact gradients, so it cannot misfire, and the family-wise
confidence is asserted rather than measured. This module replaces the exact
gradients with estimates built from a finite number of shots, runs the same
elimination rule on those estimates, and repeats, so that three things become
measurable rather than assumed:

* how often the method actually selects the largest-gradient generator;
* how many shots that really costs, which is not the planning bound because a
  real run does not know the deficits in advance and has to discover them;
* whether the normal, Bonferroni-corrected confidence model delivers the error
  rate it claims.

**Noise model and its limits.** Estimator fluctuations are drawn from the normal
limit: after ``m`` shots an estimator of one fragment has the fragment's exact
variance divided by ``m``. Shots accumulate, so the noise is carried as a running
sum rather than redrawn each round, which keeps successive rounds correlated the
way real data is.

Two approximations are deliberate and both are recorded with the results. The
normal limit is assumed rather than tested, so failures of that approximation at
very small shot counts are invisible here. And fluctuations are drawn
independently per generator: for the baseline and for M2 that is exact, because
those methods give every generator its own shots, but M1 and M3 read several
generators off the same shots in a shared context, and the resulting correlation
between their estimates is not modelled. Passing ``noise_model="correlated"`` models that correlation exactly, using the
per-context covariances of :func:`context_covariances` and the eigen square roots
of :func:`_noise_factors`. That correlation affects how errors on
different generators co-occur, not the per-generator error rate, because the
elimination rule of the Part I report uses individual radii.

The confidence radii use the exact fragment variances. Estimating variances from
the same data is a further source of cost that is not included here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bai import DEFAULT_SHRINK
from gradients import GradientProblem
from shot_models import (
    DEFAULT_DELTA,
    allocate_context_shots,
    epsilon_from_radius,
    z_for_selection_error,
    z_from_delta,
)

METHODS = ("noshare", "m1", "m2", "m3")
MINIMUM_RADIUS_FRACTION = 1e-3
MAXIMUM_ROUNDS = 200


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because the rates of interest here
    are at or near zero, where the normal interval collapses to a point and
    reports a precision the data do not support.
    """
    if trials <= 0:
        return 0.0, 1.0
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2.0 * trials)) / denominator
    margin = (
        z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass
class TrialOutcome:
    """One simulated run of one method."""

    selected: int
    correct: bool
    within_tolerance: bool
    shots: float
    rounds: int


@dataclass
class FiniteShotSummary:
    """Aggregate over many trials."""

    case_id: str
    method: str
    n_trials: int
    delta: float
    tolerance: float
    correct_rate: float
    within_tolerance_rate: float
    shots_mean: float
    shots_sem: float
    shots_median: float
    shots_p90: float
    planning_bound: float
    correct_low: float = 0.0
    correct_high: float = 1.0

    @property
    def inflation(self) -> float:
        """Realised mean cost divided by the oracle planning bound."""
        return self.shots_mean / self.planning_bound if self.planning_bound else float("nan")

    @property
    def inflation_sem(self) -> float:
        """Standard error of the realised inflation factor."""
        return self.shots_sem / self.planning_bound if self.planning_bound else float("nan")

    def as_row(self) -> dict:
        return {
            "case_id": self.case_id,
            "method": self.method,
            "n_trials": self.n_trials,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "correct_rate": self.correct_rate,
            "within_tolerance_rate": self.within_tolerance_rate,
            "planning_bound": self.planning_bound,
            "shots_mean": self.shots_mean,
            "shots_sem": self.shots_sem,
            "shots_median": self.shots_median,
            "shots_p90": self.shots_p90,
            "inflation_vs_bound": self.inflation,
            "inflation_sem": self.inflation_sem,
            "correct_low": self.correct_low,
            "correct_high": self.correct_high,
        }


def context_covariances(
    problem: GradientProblem, groups: list[list[str]], max_pairs: int = 400_000
) -> list[np.ndarray] | None:
    """Covariance between generators' fragments inside each shared context.

    Two generators read off the same shots in a shared context, so their estimates
    are correlated with covariance ``Cov(F_ia, F_ja)`` in that context. The default
    noise model ignores this; supplying these matrices makes the simulation exact
    for M1 and M3 as well.

    The covariance is obtained from variances alone, via
    ``Cov(A, B) = (Var(A + B) - Var(A) - Var(B)) / 2``, so it needs no operator
    products. Cost grows as the number of generator pairs sharing a context, and
    the function returns ``None`` rather than running for a long time when that
    exceeds ``max_pairs``; the caller then falls back to the independent model.
    """
    index = {pauli: a for a, group in enumerate(groups) for pauli in group}
    fragments: list[dict[int, dict[str, float]]] = []
    for terms in problem.commutator_terms:
        buckets: dict[int, dict[str, float]] = {}
        for pauli, coefficient in terms.items():
            buckets.setdefault(index[pauli], {})[pauli] = coefficient
        fragments.append(buckets)

    members = [[] for _ in groups]
    for i, buckets in enumerate(fragments):
        for alpha in buckets:
            members[alpha].append(i)
    if sum(len(m) * (len(m) - 1) // 2 for m in members) > max_pairs:
        return None

    n = problem.n_generators
    matrices = []
    for alpha, group in enumerate(groups):
        cov = np.zeros((n, n))
        present = members[alpha]
        variance = {
            i: problem.evaluator.fragment_std(fragments[i][alpha]) ** 2 for i in present
        }
        for i in present:
            cov[i, i] = variance[i]
        for a, i in enumerate(present):
            for j in present[a + 1:]:
                combined = dict(fragments[i][alpha])
                for pauli, coefficient in fragments[j][alpha].items():
                    combined[pauli] = combined.get(pauli, 0.0) + coefficient
                joint = problem.evaluator.fragment_std(combined) ** 2
                value = 0.5 * (joint - variance[i] - variance[j])
                cov[i, j] = cov[j, i] = value
        matrices.append(cov)
    return matrices


def _noise_factors(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Square roots of the per-context covariances, for sampling correlated noise.

    A Cholesky factorisation is not usable here. These matrices are singular by
    construction -- most generators have no support in any given context -- and
    the ridge needed to make Cholesky succeed injects a small variance into every
    direction, including the null ones. That is harmless in absolute terms but not
    after division by the shot count: contexts that the allocation gives almost no
    shots then amplify the injected variance without limit, and fragments whose
    true variance is denormal (values near 1e-36 occur) are swamped entirely.

    The symmetric eigendecomposition gives an exact square root of a positive
    semidefinite matrix without perturbing it: negative eigenvalues, which only
    ever appear at rounding level, are clipped to zero, and null directions stay
    null.
    """
    factors = []
    for covariance in matrices:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, vectors = np.linalg.eigh(symmetric)
        factors.append(vectors * np.sqrt(np.clip(eigenvalues, 0.0, None)))
    return factors


class _AllocationCache:
    """Context allocations for an active set, reused across rounds and trials.

    The allocation solves a convex programme whose solution scales exactly as
    ``1 / epsilon**2``: the stationarity condition fixes the shape and the final
    feasibility rescale carries all of the epsilon dependence. So the shape is
    computed once per distinct active set at a reference epsilon of one and then
    scaled, which is what makes a few hundred trials on a 92-generator pool
    tractable.
    """

    def __init__(self, sigmas: np.ndarray) -> None:
        self._sigmas = sigmas
        self._shapes: dict[frozenset, np.ndarray] = {}

    def allocation(self, active: list[int], epsilon: float) -> np.ndarray:
        key = frozenset(active)
        shape = self._shapes.get(key)
        if shape is None:
            shape = allocate_context_shots(self._sigmas[active, :], 1.0)
            self._shapes[key] = shape
        return shape / epsilon ** 2


def _eliminate(estimates: np.ndarray, radii: np.ndarray, active: list[int]) -> list[int]:
    """The Part I elimination rule applied to estimates rather than exact values.

    Each generator carries its own confidence radius, so the rule implicitly treats
    the estimates as independent. That holds when every generator has its own shots.
    It does not hold for shared contexts, where the estimates are correlated; see
    :func:`_eliminate_pairwise`.
    """
    best = max(abs(estimates[j]) - radii[j] for j in active)
    return [i for i in active if abs(estimates[i]) + radii[i] >= best]


def _eliminate_pairwise(
    estimates: np.ndarray, pairwise_variance: np.ndarray, active: list[int], z: float
) -> list[int]:
    """Covariance-aware elimination: compare each pair at its own radius.

    A generator is dropped only when some other generator leads it by more than the
    confidence radius of *their difference*, which is the quantity the comparison
    actually depends on. Where estimates are anticorrelated this radius is wider
    than the individual radii suggest, and the marginal rule is then anti-conservative.
    """
    magnitudes = np.abs(estimates[active])
    lead = magnitudes[None, :] - magnitudes[:, None]
    radii = z * np.sqrt(np.maximum(pairwise_variance, 0.0))
    dominated = (lead > radii).any(axis=1)
    return [i for k, i in enumerate(active) if not dominated[k]]


def _within_tolerance(
    estimates: np.ndarray, radii: np.ndarray, active: list[int], tolerance: float
) -> bool:
    """Can every surviving candidate be certified within ``tolerance`` of the best?

    Good-enough selection does not need the exact maximiser: a generator whose
    gradient is within a tolerance of the largest lowers the energy almost as much.
    The run may stop as soon as the worst case over survivors, taking each
    confidence interval at its least favourable end, is inside the tolerance.
    """
    if tolerance <= 0 or len(active) <= 1:
        return len(active) <= 1
    leader = max(active, key=lambda i: abs(estimates[i]))
    floor = abs(estimates[leader]) - radii[leader]
    return all(
        (abs(estimates[i]) + radii[i]) - floor <= tolerance
        for i in active
        if i != leader
    )


def _static_trial(
    problem: GradientProblem,
    sigma_sums: np.ndarray | None,
    sigmas: np.ndarray | None,
    radius: float,
    z: float,
    tolerance: float,
    rng: np.random.Generator,
    shared: bool,
) -> TrialOutcome:
    """Baseline and M1: measure everything once, then take the largest estimate."""
    truth = problem.abs_gradients
    leader = int(np.argmax(truth))
    if shared:
        shots = allocate_context_shots(sigmas, epsilon_from_radius(radius, z))
        used = shots > 0
        variance = (sigmas[:, used] ** 2 / shots[used]).sum(axis=1)
        total = float(shots.sum())
    else:
        counts = (sigma_sums * z / radius) ** 2
        variance = sigma_sums ** 2 / counts
        total = float(counts.sum())
    estimates = problem.gradients + rng.normal(0.0, np.sqrt(variance))
    selected = int(np.argmax(np.abs(estimates)))
    return TrialOutcome(
        selected=selected,
        correct=selected == leader,
        within_tolerance=bool(truth[leader] - truth[selected] <= tolerance),
        shots=total,
        rounds=1,
    )


def _sequential_trial(
    problem: GradientProblem,
    sigma_sums: np.ndarray | None,
    sigmas: np.ndarray | None,
    z: float,
    shrink: float,
    tolerance: float,
    rng: np.random.Generator,
    shared: bool,
    cache: "_AllocationCache | None" = None,
    stop_at_tolerance: bool = False,
    factors: list[np.ndarray] | None = None,
    covariances: list[np.ndarray] | None = None,
) -> TrialOutcome:
    """M2 and M3: batched successive elimination driven by the estimates."""
    truth = problem.abs_gradients
    leader = int(np.argmax(truth))
    n = problem.n_generators
    active = list(range(n))

    radius = float(truth.max())
    floor = radius * MINIMUM_RADIUS_FRACTION

    if shared:
        spent = np.zeros(sigmas.shape[1])
        noise = np.zeros_like(sigmas)
    else:
        spent = np.zeros(n)
        noise = np.zeros(n)

    rounds = 0
    for rounds in range(1, MAXIMUM_ROUNDS + 1):
        if len(active) <= 1 or radius < floor:
            break
        epsilon = epsilon_from_radius(radius, z)
        if shared:
            wanted = np.maximum(spent, cache.allocation(active, epsilon))
            added = wanted - spent
            grew = added > 0
            if grew.any():
                if factors is None:
                    noise[:, grew] += rng.normal(
                        0.0, sigmas[:, grew] * np.sqrt(added[grew]), size=(n, int(grew.sum()))
                    )
                else:
                    for alpha in np.flatnonzero(grew):
                        noise[:, alpha] += np.sqrt(added[alpha]) * (
                            factors[alpha] @ rng.standard_normal(n)
                        )
            spent = wanted
            used = spent > 0
            estimates = problem.gradients + (noise[:, used] / spent[used]).sum(axis=1)
            variance = (sigmas[:, used] ** 2 / spent[used]).sum(axis=1)
        else:
            wanted = spent.copy()
            wanted[active] = np.maximum(
                spent[active], (sigma_sums[active] / epsilon) ** 2
            )
            added = wanted - spent
            grew = added > 0
            if grew.any():
                noise[grew] += rng.normal(0.0, sigma_sums[grew] * np.sqrt(added[grew]))
            spent = wanted
            counted = spent > 0
            estimates = problem.gradients.copy()
            estimates[counted] += noise[counted] / spent[counted]
            variance = np.full(n, np.inf)
            variance[counted] = sigma_sums[counted] ** 2 / spent[counted]

        radii = z * np.sqrt(np.where(np.isfinite(variance), variance, 0.0))
        if covariances is None:
            active = _eliminate(estimates, radii, active)
        else:
            block = np.zeros((len(active), len(active)))
            index = np.ix_(active, active)
            for alpha in np.flatnonzero(spent > 0):
                block += covariances[alpha][index] / spent[alpha]
            diagonal = np.diag(block)
            # The rule compares |g_i| against |g_j|, so the relevant combination
            # is g_i - s_ij g_j with s_ij the relative sign of the two estimates.
            # Using the signed difference unconditionally is wrong whenever the
            # two gradients have opposite signs, which happens at the top of the
            # spectrum on the symmetric H4 geometries.
            signs = np.sign(estimates[active])
            relative = np.outer(signs, signs)
            pairwise = diagonal[:, None] + diagonal[None, :] - 2.0 * relative * block
            active = _eliminate_pairwise(estimates, pairwise, active, z)
        if stop_at_tolerance and _within_tolerance(estimates, radii, active, tolerance):
            break
        radius *= shrink

    selected = max(active, key=lambda i: abs(estimates[i]))
    return TrialOutcome(
        selected=selected,
        correct=selected == leader,
        within_tolerance=bool(truth[leader] - truth[selected] <= tolerance),
        shots=float(spent.sum()),
        rounds=rounds,
    )


def simulate(
    problem: GradientProblem,
    method: str,
    *,
    planning_bound: float,
    n_trials: int = 400,
    delta: float = DEFAULT_DELTA,
    shrink: float = DEFAULT_SHRINK,
    tolerance: float = 0.0,
    seed: int = 0,
    calibration: str = "bonferroni",
    stop_at_tolerance: bool = False,
    noise_model: str = "independent",
    rule: str = "marginal",
    m1_radius: float | None = None,
    progress: Callable[[int], None] | None = None,
) -> FiniteShotSummary:
    """Run ``n_trials`` finite-shot trials of one method on one problem."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(seed)
    if calibration == "bonferroni":
        z = z_from_delta(delta, problem.n_generators)
    elif calibration == "selection":
        z = z_for_selection_error(delta)
    else:
        raise ValueError("calibration must be 'bonferroni' or 'selection'")
    shared = method in ("m1", "m3")

    sigmas = problem.parent_fragment_sigmas() if shared else None
    cache = _AllocationCache(sigmas) if shared else None
    if noise_model not in ("independent", "correlated"):
        raise ValueError("noise_model must be 'independent' or 'correlated'")
    factors = None
    matrices = None
    if shared and (noise_model == "correlated" or rule == "pairwise"):
        matrices = context_covariances(problem, problem.parent_fc_groups())
        if matrices is not None and noise_model == "correlated":
            factors = _noise_factors(matrices)
    covariances = matrices if (shared and rule == "pairwise") else None
    sigma_sums = None
    if not shared:
        groups = problem.individual_fc_groups()
        sigma_sums = np.array([s.sum() for s in problem.individual_fragment_sigmas(groups)])

    radius = m1_radius if m1_radius is not None else problem.top_gap() / 2.0

    outcomes = []
    for trial in range(n_trials):
        if method in ("noshare", "m1"):
            outcomes.append(
                _static_trial(problem, sigma_sums, sigmas, radius, z, tolerance, rng, shared)
            )
        else:
            outcomes.append(
                _sequential_trial(
                    problem, sigma_sums, sigmas, z, shrink, tolerance, rng, shared, cache,
                    stop_at_tolerance, factors, covariances,
                )
            )
        if progress is not None and (trial + 1) % 50 == 0:
            progress(trial + 1)

    shots = np.array([o.shots for o in outcomes])
    wilson_low, wilson_high = wilson_interval(
        sum(1 for o in outcomes if o.correct), len(outcomes)
    )
    return FiniteShotSummary(
        case_id=problem.case_id,
        method=method,
        n_trials=n_trials,
        delta=delta,
        tolerance=tolerance,
        correct_rate=float(np.mean([o.correct for o in outcomes])),
        within_tolerance_rate=float(np.mean([o.within_tolerance for o in outcomes])),
        shots_mean=float(shots.mean()),
        shots_sem=float(shots.std(ddof=1) / math.sqrt(len(shots))) if len(shots) > 1 else 0.0,
        shots_median=float(np.median(shots)),
        shots_p90=float(np.percentile(shots, 90)),
        planning_bound=float(planning_bound),
        correct_low=wilson_low,
        correct_high=wilson_high,
    )
