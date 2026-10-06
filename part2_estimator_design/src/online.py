"""Shot-level online best-arm identification on fixed contexts (Step 1).

This is Part I's M3 run the way an experiment would run it.  Each round draws
*measurement outcomes*: the new shots of a context are one multinomial draw over
its ``2**n`` computational-basis outcomes after the measurement circuit
(:mod:`sampler`).  The gradient estimates are the design applied to the resulting
sample means of the measured group, and the elimination rule sees only those
estimates.  Part I's finite-shot study instead drew Gaussian estimator noise with
the exact covariance; the two agree in their first two moments by construction,
and Step 0 checks that the rest does not matter at the shot counts used here.

What is and is not learned.  As in Part I, the confidence radii use the *oracle*
fragment variances and covariances; estimating them from the same data is Step 4.
The starting radius ``max_i |g_i|`` and the floor ``1e-3 max_i |g_i|`` are Part I's
mild oracle quantities.  Everything else -- which arms survive, when to stop,
which generator is returned -- is decided from sampled outcomes only.

Options mirror Part I's ``finite_shot._sequential_trial`` so that the comparison is
like for like:

``rule``
    ``"marginal"`` (Part I Eq. 4), ``"pairwise"`` (covariance-aware, Part I
    Eq. 17, with the sign factor ``s_ij``) or ``"safe"`` (pairwise only where the
    signs are resolved; :mod:`rules`);
``accounting``
    ``"maxima"`` raises each context to Part I's allocation for the round and
    never lowers it; ``"topup"`` solves the round's allocation above the shots
    already held (:func:`allocation.allocate_topup`);
``sampling``
    ``"multinomial"`` or ``"none"``, the latter replacing every estimate by the
    exact gradient so that the schedule's own cost can be separated from noise.

Shots are integers: each context is raised to the ceiling of its cumulative
target, which differs from Part I's real-valued counts by less than one shot per
context.  Every context that carries a fragment also gets at least one shot in
the first round.  The allocation gives no shots to a context whose fragments all
have zero variance -- deterministic outcomes such as ``Z``-strings on a
determinant -- but their values still enter the gradients, and a real run has to
look once to learn them; the surrogate of Part I adds noise to exact gradients and
never needs to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from allocation import allocate, allocate_topup
from part1_bridge import confidence_z, epsilon_from_radius
from rules import RULES, eliminate
from sampler import walsh_hadamard

SAMPLING = ("multinomial", "none")
MINIMUM_RADIUS_FRACTION = 1e-3
MAXIMUM_ROUNDS = 400


@dataclass(frozen=True)
class OnlineConfig:
    shrink: float = 0.5
    rule: str = "pairwise"
    accounting: str = "maxima"
    sampling: str = "multinomial"
    delta: float = 0.05
    terminal_radius: float | None = None
    integer_shots: bool = True
    confidence: str = "bonferroni"  # or "selection": part1_bridge.confidence_z

    def __post_init__(self) -> None:
        confidence_z(self.delta, 1, self.confidence)  # validates the mode
        if not self.integer_shots and self.sampling != "none":
            raise ValueError("real-valued shot counts are only meaningful without sampling")
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {SAMPLING}")
        if self.accounting not in ("maxima", "topup"):
            raise ValueError("accounting must be 'maxima' or 'topup'")
        if not 0.0 < self.shrink < 1.0:
            raise ValueError("shrink must lie in (0, 1)")

    @property
    def label(self) -> str:
        return f"{self.rule}/{self.accounting}/gamma={self.shrink:g}/{self.sampling}"


@dataclass
class OnlineOutcome:
    selected: int
    correct: bool
    shots: float
    rounds: int
    history: list = field(default_factory=list, repr=False)
    extra: dict = field(default_factory=dict, repr=False)  # diagnostics and validation


class OnlineM3:
    """One problem, one context library, one fixed design, many trials."""

    def __init__(self, problem, library, moments, design, config: OnlineConfig) -> None:
        self.problem = problem
        self.config = config
        self.n_arms = problem.n_generators
        self.n_contexts = library.n_contexts
        self.z = confidence_z(config.delta, self.n_arms, config.confidence)
        arms = list(range(self.n_arms))
        self.sigmas = design.sigmas(arms)
        self.squared = self.sigmas ** 2
        self.distributions = moments.distributions_matrix()
        self.distributions /= self.distributions.sum(axis=1, keepdims=True)

        # Flattened design: g_hat_i = sum_c x_c * sign_c * E_hat[context_c, zmask_c].
        gen, ctx, zmask, weight = [], [], [], []
        for p in design.problems:
            keep = p.x != 0.0
            for alpha in np.unique(p.coord_ctx[keep]):
                context = library.contexts[int(alpha)]
                chosen = keep & (p.coord_ctx == alpha)
                positions = context.positions(p.coord_pauli[chosen])
                gen.append(np.full(positions.size, p.generator))
                ctx.append(np.full(positions.size, alpha))
                zmask.append(context.member_zmask[positions])
                weight.append(p.x[chosen] * context.member_sign[positions])
        self._gen = np.concatenate(gen)
        self._ctx = np.concatenate(ctx)
        self._zmask = np.concatenate(zmask)
        self._weight = np.concatenate(weight)
        self._minimum = np.zeros(self.n_contexts, dtype=np.int64)
        self._minimum[np.unique(self._ctx)] = 1

        self.covariances = None
        if config.rule != "marginal":
            self.covariances = self._generator_covariances(library, moments, design)
        self._shape_cache: dict[frozenset, np.ndarray] = {}

    def _generator_covariances(self, library, moments, design) -> np.ndarray:
        """``K[alpha, i, j] = Cov(F_{i,alpha}, F_{j,alpha})`` for the design (oracle)."""
        K = np.zeros((self.n_contexts, self.n_arms, self.n_arms))
        by_context: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for p in design.problems:
            keep = p.x != 0.0
            for alpha in np.unique(p.coord_ctx[keep]):
                chosen = keep & (p.coord_ctx == alpha)
                by_context.setdefault(int(alpha), []).append(
                    (p.generator, p.coord_pauli[chosen], p.x[chosen])
                )
        for alpha, fragments in by_context.items():
            paulis = np.unique(np.concatenate([f[1] for f in fragments]))
            X = np.zeros((paulis.size, len(fragments)))
            for column, (_, ids, values) in enumerate(fragments):
                X[np.searchsorted(paulis, ids), column] = values
            covariance = X.T @ moments.covariance(alpha, paulis) @ X
            gens = [f[0] for f in fragments]
            K[alpha][np.ix_(gens, gens)] = covariance
        return K

    def _allocation(self, active: list[int], epsilon: float, spent: np.ndarray) -> np.ndarray:
        if self.config.accounting == "topup":
            return allocate_topup(self.sigmas[active], epsilon, spent)
        key = frozenset(active)
        shape = self._shape_cache.get(key)
        if shape is None:
            shape = allocate(self.sigmas[active], 1.0)
            self._shape_cache[key] = shape
        return np.maximum(spent, shape / epsilon ** 2)

    def run(self, rng: np.random.Generator, *, record: bool = False) -> OnlineOutcome:
        truth = self.problem.abs_gradients
        leader = int(np.argmax(truth))
        z = self.z
        active = list(range(self.n_arms))
        radius = float(truth.max())
        floor = radius * MINIMUM_RADIUS_FRACTION
        spent = np.zeros(self.n_contexts, dtype=np.int64 if self.config.integer_shots else float)
        counts = np.zeros(self.distributions.shape, dtype=np.int64)
        means = np.zeros(self.distributions.shape)
        estimates = self.problem.gradients.copy()
        history = []

        rounds = 0
        for rounds in range(1, MAXIMUM_ROUNDS + 1):
            terminal = self.config.terminal_radius
            if terminal is None:
                if len(active) <= 1 or radius < floor:
                    break
            elif radius < terminal or radius < floor:
                break
            epsilon = epsilon_from_radius(radius, z)
            wanted = self._allocation(active, epsilon, spent.astype(float))
            if self.config.integer_shots:
                target = np.maximum(spent, np.ceil(wanted - 1e-9).astype(np.int64))
            else:
                target = np.maximum(spent, wanted)
            if self.config.sampling == "multinomial":
                target = np.maximum(target, self._minimum)
            added = target - spent
            grew = np.flatnonzero(added > 0)
            if grew.size and self.config.sampling == "multinomial":
                counts[grew] += rng.multinomial(added[grew], self.distributions[grew])
                means[grew] = walsh_hadamard(counts[grew]) / target[grew, None]
            spent = target
            if self.config.sampling == "multinomial":
                contributions = self._weight * means[self._ctx, self._zmask]
                estimates = np.bincount(self._gen, weights=contributions, minlength=self.n_arms)

            funded = spent > 0
            variance = (self.squared[:, funded] / spent[funded]).sum(axis=1)
            starved = (self.squared[:, ~funded] > 0).any(axis=1)
            variance[starved] = np.inf
            if self.config.rule == "marginal":
                # Part I zeroes a starved arm's radius here; kept for comparability.
                block = np.diag(np.where(np.isfinite(variance), variance, 0.0)[active])
            else:
                inverse = np.zeros(self.n_contexts)
                inverse[funded] = 1.0 / spent[funded]
                index = np.asarray(active)
                block = np.einsum("a,aij->ij", inverse, self.covariances[:, index][:, :, index])
            active = eliminate(active, estimates, block, z, self.config.rule).survivors
            if record:
                history.append({"round": rounds, "radius": radius, "n_active": len(active),
                                "cumulative_shots": int(spent.sum())})
            radius *= self.config.shrink

        selected = max(active, key=lambda i: abs(estimates[i]))
        return OnlineOutcome(selected, selected == leader, float(spent.sum()), rounds, history)


def summarise(outcomes: list[OnlineOutcome]) -> dict:
    """Median, IQR, P90 and mean of the cost; Wilson interval on correct selection."""
    from finite_shot import wilson_interval  # Part I

    shots = np.array([o.shots for o in outcomes])
    correct = sum(o.correct for o in outcomes)
    low, high = wilson_interval(correct, len(outcomes))
    return {
        "n_trials": len(outcomes),
        "correct_rate": correct / len(outcomes),
        "correct_low": low,
        "correct_high": high,
        "failures": len(outcomes) - correct,
        "shots_mean": float(shots.mean()),
        "shots_sem": float(shots.std(ddof=1) / math.sqrt(len(shots))) if len(shots) > 1 else 0.0,
        "shots_median": float(np.median(shots)),
        "shots_q25": float(np.percentile(shots, 25)),
        "shots_q75": float(np.percentile(shots, 75)),
        "shots_p90": float(np.percentile(shots, 90)),
        "rounds_mean": float(np.mean([o.rounds for o in outcomes])),
        **_extra_summary(outcomes),
    }


GOOD_FRACTIONS = (0.01, 0.05, 0.10)


def _extra_summary(outcomes: list[OnlineOutcome]) -> dict:
    """Delta-good rates, interval coverage and classical time, when recorded."""
    if not outcomes or not all("shortfall" in o.extra for o in outcomes):
        return {}
    n = len(outcomes)
    shortfall = np.array([o.extra["shortfall"] for o in outcomes])
    result = {f"good_{round(100 * rho)}pct_rate": float((shortfall <= rho + 1e-12).mean())
              for rho in GOOD_FRACTIONS}
    result["miscovered_rate"] = sum(o.extra["miscovered_rounds"] > 0 for o in outcomes) / n
    result["best_eliminated"] = int(sum(o.extra["best_eliminated"] for o in outcomes))
    result["stopped_rho_rate"] = sum(o.extra["stopped_rho"] for o in outcomes) / n
    for key in ("design_seconds", "statistics_seconds", "contexts_used", "cz_per_shot_mean"):
        if key in outcomes[0].extra:
            result[f"{key}_mean"] = float(np.mean([o.extra[key] for o in outcomes]))
    return result
