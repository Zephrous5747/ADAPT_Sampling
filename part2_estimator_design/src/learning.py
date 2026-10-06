"""Step 4: estimator designs learned from the shots themselves.

The online loop of :mod:`online` with the II-A design refitted from data.

Covariance model
    Per context and fold, the one-shot covariance of the measured Paulis is the
    shrinkage ``lam * prior + (1 - lam) * empirical``, with
    ``lam = nu / (nu + m)`` for ``m`` shots.  The empirical part is the unbiased
    sample covariance of the outcome histogram (:mod:`sampler`); being a sample
    covariance it is positive semidefinite, and so is every shrinkage of it with a
    PSD prior, so no projection is needed.  Priors: ``hf`` (the determinant, a
    classical proxy), ``flat`` (unit variances, no correlation) or ``oracle``
    (the true state; a diagnostic, not a method).  ``nu = inf`` is the prior
    alone, ``nu = 0`` the data alone.

Cross-fitting
    Every context's shots are split into two folds.  The design fitted on fold A's
    covariance is applied to fold B's sample means and vice versa, and the two
    estimates are averaged.  A fold's design depends only on that fold's data, so
    each half-estimate is unbiased whatever the design; this is what makes it
    legitimate to choose coefficients from the same experiment.

Guard
    A fitted design is kept for a generator only if the *other* fold's covariance
    predicts a lower variance for it than for the II-0 coefficients; otherwise
    that generator falls back to II-0.  This is the defence against a prior (or a
    noisy covariance) that fits cancellations the state does not have.

Radii
    ``radii="estimated"`` computes every variance and pairwise covariance from the
    covariance model of the *held-out* fold: fold A's design is evaluated with fold
    B's covariance, which is independent of it (fully non-oracle).  A pooled
    covariance would include the fold the design was fitted to and underestimate its
    variance.  ``radii="oracle"`` uses the exact covariances, as Part I and Step 1
    do, to isolate the design effect.

The first round is always an II-0 pilot at the starting radius; designs are
refitted when the shots held have grown by ``refit_growth`` since the last fit.
Allocation plans with the fold-A design and fold B's covariance and tops up from
the shots already held.

Objectives (``objective``)
    ``"arm"`` designs each generator's estimator for its own variance and plans
    the shots so that every active arm reaches the round's radius.  At fixed shots
    the arms decouple, so this is also the minimiser of the work order's
    ``U_avg`` for any weights.  ``"contrast"`` targets what BAI actually tests
    (``U_BAI`` and level II-E): once the leader ``l``'s sign is resolved, every
    other survivor ``i`` gets a *directly designed* estimator of
    ``s_l g_l - t g_i`` (``t = s_i`` when ``i``'s sign is resolved, both
    ``t = +-1`` otherwise), and the shots are planned so that each contrast's
    standard deviation reaches ``2 epsilon`` -- the separation a pair of arms at
    radius ``epsilon`` would certify -- and the leader's own reaches ``epsilon``.
    A contrast's coordinates are the union of both arms' II-A coordinates; Pauli
    products whose coefficients cancel become zero-sum control variates.  The
    guard compares each contrast design with the difference of the two arm
    designs under the held-out fold.

Rules and stopping (``rule``, ``rho``, ``start``)
    ``rule`` is one of :data:`rules.RULES`; ``"safe"`` is the sign-aware rule and
    is required by the contrast objective.  ``rho > 0`` stops as soon as the
    leader is certified ``rho``-good (:func:`rules.rho_good_stop`).
    ``start="oracle"`` begins at ``max_i |g_i|`` as Part I does; ``"bound"``
    begins at ``max_i sum_l |A_il|``, an a-priori bound on every ``|g_i|``, and
    lowers the floor accordingly, so no exact gradient enters the run.

Small-sample radii (``radius_min_shots``)
    A sample covariance from a handful of shots can be exactly zero (two equal
    outcomes), and an interval built on it has zero width.  With the oracle start
    the first eliminations come when contexts hold O(100) shots, but with the
    ``bound`` start they can come at 2-5 shots per fold: on a tied H4 state II-0
    then stopped at 9,141 shots on an arm with 6% of the best gradient.  A held-out
    fold with fewer than ``radius_min_shots`` shots in a context is therefore
    replaced, for radii and planning alike, by ``k I`` over its ``k`` Paulis, a
    valid upper bound (``Sigma <= tr(Sigma) I <= k I`` since each Pauli has
    variance at most one).  The planner then buys those contexts out of the regime.

Sequential validity (``anytime``)
    Part I's ``z`` is a Bonferroni bound over arms for one look.  The loop looks
    once per round, so the family-wise guarantee does not cover the whole run;
    ``anytime=True`` spends ``6 delta / (pi^2 r^2)`` in round ``r``, which sums to
    ``delta`` over all rounds.  The validation record measures the difference.

Logging and validation
    ``run(rng, log=...)`` hands a :class:`runlog.RunLog` every round's active set,
    shots, estimates, intervals, eliminations, redesigns and timings (Phase 3).
    Every refit checks ``A = BC`` for each designed row and fails hard beyond
    ``1e-9 max |A|``.  The truth enters only the validation record, which notes
    whether any interval used missed the exact value.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from allocation import allocate, allocate_topup
from design import FragmentProblem
from online import OnlineOutcome
from part1_bridge import epsilon_from_radius, z_from_delta
from rules import RULES, eliminate, rho_good_stop
from sampler import pauli_covariance, walsh_hadamard

PRIORS = ("hf", "flat", "oracle", "none")
OBJECTIVES = ("arm", "contrast")
STARTS = ("oracle", "bound")
ELIMINATIONS = ("on", "off")
RESIDUAL_TOLERANCE = 1e-9  # relative to max |A|


@dataclass(frozen=True)
class LearningConfig:
    level: str = "II-A"  # "II-0" disables learning
    prior: str = "hf"
    nu: float = 100.0
    radii: str = "estimated"  # or "oracle"
    guard: bool = True
    shrink: float = 0.9
    delta: float = 0.05
    refit_growth: float = 1.5
    min_fold_shots: int = 50  # a context enters a design only with this many shots per fold
    rule: str = "pairwise"
    objective: str = "arm"
    rho: float = 0.0
    start: str = "oracle"
    radius_min_shots: int = 0  # held-out folds with fewer shots use the bound Sigma <= k I
    anytime: bool = False  # union bound over rounds: delta_r = 6 delta / (pi^2 r^2)
    elimination: str = "on"  # "off": every arm stays in the allocation (sequential M1)

    def __post_init__(self) -> None:
        if self.elimination not in ELIMINATIONS:
            raise ValueError(f"elimination must be one of {ELIMINATIONS}")
        if self.elimination == "off" and self.objective != "arm":
            raise ValueError("without elimination there is no leader to contrast against: use objective='arm'")
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.objective == "contrast" and self.rule != "safe":
            raise ValueError("the contrast objective needs resolved signs: use rule='safe'")
        if self.start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")

    @property
    def label(self) -> str:
        if self.level == "II-0":
            label = f"II-0/{self.radii} radii"
        else:
            nu = "inf" if math.isinf(self.nu) else f"{self.nu:g}"
            label = f"{self.level}/prior={self.prior}/nu={nu}/{self.radii} radii/guard={'on' if self.guard else 'off'}"
        extras = [f"rule={self.rule}"] if self.rule != "pairwise" else []
        extras += [f"objective={self.objective}"] if self.objective != "arm" else []
        extras += [f"rho={self.rho:g}"] if self.rho > 0 else []
        extras += [f"start={self.start}"] if self.start != "oracle" else []
        extras += [f"radius_min_shots={self.radius_min_shots}"] if self.radius_min_shots else []
        extras += ["anytime"] if self.anytime else []
        extras += ["no-elimination"] if self.elimination == "off" else []
        return "/".join([label] + extras)


class CovarianceModel:
    """Shrinkage covariance per context and fold, from outcome histograms."""

    def __init__(self, library, dim: int, prior, nu: float, oracle=None) -> None:
        self.library = library
        self.prior = prior  # OracleMoments, "flat" or None
        self.nu = nu
        self.oracle = oracle  # OracleMoments when the model *is* the truth
        self.counts = np.zeros((2, library.n_contexts, dim), dtype=np.int64)
        self._means: dict[tuple, np.ndarray] = {}

    def add(self, contexts: np.ndarray, fold_counts: tuple[np.ndarray, np.ndarray]) -> None:
        for fold in (0, 1):
            self.counts[fold, contexts] += fold_counts[fold]
            for alpha in contexts:
                self._means.pop((fold, int(alpha)), None)
                self._means.pop((None, int(alpha)), None)

    def shots(self, fold=None) -> np.ndarray:
        counts = self.counts.sum(axis=0) if fold is None else self.counts[fold]
        return counts.sum(axis=1)

    def group_means(self, alpha: int, fold=None) -> np.ndarray:
        key = (fold, alpha)
        if key not in self._means:
            counts = self.counts[:, alpha].sum(axis=0) if fold is None else self.counts[fold, alpha]
            m = counts.sum()
            self._means[key] = walsh_hadamard(counts) / m if m else np.zeros(counts.size)
        return self._means[key]

    def _images(self, alpha, paulis):
        context = self.library.contexts[alpha]
        positions = context.positions(paulis)
        return context.member_zmask[positions], context.member_sign[positions]

    def _prior(self, alpha, paulis) -> np.ndarray:
        if self.prior == "flat" or self.prior is None:
            return np.eye(len(paulis))
        return self.prior.covariance(alpha, paulis)

    def covariance(self, alpha: int, paulis: np.ndarray, fold=None) -> np.ndarray:
        if self.oracle is not None:
            return self.oracle.covariance(alpha, paulis)
        counts = self.counts[:, alpha].sum(axis=0) if fold is None else self.counts[fold, alpha]
        m = float(counts.sum())
        prior = self._prior(alpha, paulis)
        if m < 2 or math.isinf(self.nu):
            return prior
        zmask, sign = self._images(alpha, paulis)
        empirical = pauli_covariance(self.group_means(alpha, fold), zmask, sign) * m / (m - 1.0)
        weight = self.nu / (self.nu + m)
        return weight * prior + (1.0 - weight) * empirical

    def means(self, alpha: int, paulis: np.ndarray, fold) -> np.ndarray:
        zmask, sign = self._images(alpha, paulis)
        return sign * self.group_means(alpha, fold)[zmask]


class LearnedM3:
    """Shot-level best-arm identification with a learned, cross-fitted design."""

    def __init__(self, problem, library, oracle_moments, prior_moments, base_problems, split_coords, config,
                 credit=None):
        self.problem = problem
        # Shots already held in each context when the selection starts (the last energy
        # evaluation of the VQE step, for the shot-reuse baseline).  They are sampled once
        # per trial, enter every estimate and allocation, and are not charged to the selection.
        self.credit = (np.zeros(library.n_contexts, dtype=np.int64) if credit is None
                       else np.asarray(credit, dtype=np.int64))
        self.library = library
        self.oracle = oracle_moments
        self.prior_moments = prior_moments
        self.config = config
        self.n_arms = problem.n_generators
        self.n_contexts = library.n_contexts
        self.z = z_from_delta(config.delta, self.n_arms)
        self.distributions = oracle_moments.distributions_matrix()
        self.distributions /= self.distributions.sum(axis=1, keepdims=True)
        self.dim = self.distributions.shape[1]
        self.base = base_problems  # II-0 FragmentProblems (oracle blocks; coefficients only)
        self.split_coords = split_coords  # per arm: (coord_ctx, coord_pauli, targets) of II-A
        carrying = np.unique(np.concatenate([p.coord_ctx for p in base_problems]))
        self._minimum = np.zeros(self.n_contexts, dtype=np.int64)
        self._minimum[carrying] = 2  # at least one shot in each fold
        self._pair_coords: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._n_library = library.n_library
        self._scale = max(max((abs(v) for v in t.values()), default=0.0) for _, _, t in split_coords)
        self.bound = max(sum(abs(v) for v in t.values()) for _, _, t in split_coords)
        self.cz = library.two_qubit_counts()
        self.guard_kept = 0
        self.contrast_kept = 0
        self.z_round = self.z

    # --- designs --------------------------------------------------------------

    def _model(self):
        prior = {"hf": self.prior_moments, "flat": "flat", "oracle": self.oracle, "none": None}[self.config.prior]
        oracle = self.oracle if self.config.prior == "oracle" and math.isinf(self.config.nu) else None
        return CovarianceModel(self.library, self.dim, prior, self.config.nu, oracle)

    def _fold_problems(self, model, fold) -> list[FragmentProblem]:
        home = self.library.home
        return [
            FragmentProblem(i, ctx, pauli, targets, home, lambda a, p, f=fold: model.covariance(a, p, f))
            for i, (ctx, pauli, targets) in enumerate(self.split_coords)
        ]

    def _home_x(self, p: FragmentProblem) -> np.ndarray:
        x = np.zeros(p.n_coordinates)
        x[p.reference] = p.pauli_target
        return x

    def _refit(self, model, active) -> list[list]:
        """Fold designs as (ctx, pauli, x) per arm; fallback to II-0 by the guard.

        Only contexts where *both* folds hold at least ``min_fold_shots`` shots may
        carry learned coefficients; elsewhere a generator keeps its II-0
        coefficients.  A covariance from a handful of samples has spurious
        zero-variance directions that a fitted design exploits.  On LiH, fitting
        straight after the pilot round (a few shots per context) eliminated
        almost every arm at once and gave 54% correct selections.
        """
        shots = [model.shots(0).astype(float), model.shots(1).astype(float)]
        supported = np.minimum(shots[0], shots[1]) >= self.config.min_fold_shots
        learnable = [np.where(supported, s, 0.0) for s in shots]
        folds = [self._fold_problems(model, 0), self._fold_problems(model, 1)]
        designs = [[None] * self.n_arms, [None] * self.n_arms]
        self.guard_kept = 0
        for i in range(self.n_arms):
            for f in (0, 1):
                p, check = folds[f][i], folds[1 - f][i]
                home = self._home_x(p)
                x = home
                if i in active and supported[p.ctx_ids].any():
                    p.x = home.copy()
                    p.optimise(learnable[f])
                    candidate = p.x
                    if not self.config.guard or (
                        check.variance(shots[1 - f], candidate) < check.variance(shots[1 - f], home)
                    ):
                        x = candidate
                        self.guard_kept += 1
                keep = x != 0.0
                designs[f][i] = (p.coord_ctx[keep], p.coord_pauli[keep], x[keep])
                self._check_row(designs[f][i], self.split_coords[i][2], f"arm {i}, fold {f}")
        return designs

    def _check_row(self, design, targets: dict, what: str) -> float:
        """Hard failure unless the copies of every Pauli sum to its target (A = BC)."""
        _, pauli, x = design
        ids, inverse = np.unique(pauli, return_inverse=True)
        totals = np.bincount(inverse, weights=x, minlength=ids.size)
        wanted = np.array([targets.get(int(l), 0.0) for l in ids])
        residual = float(np.abs(totals - wanted).max()) if ids.size else 0.0
        present = set(ids.tolist())
        residual = max([residual] + [abs(v) for l, v in targets.items() if l not in present])
        if residual > RESIDUAL_TOLERANCE * self._scale:
            raise AssertionError(f"reconstruction residual {residual:.3e} for {what}")
        return residual

    # --- contrasts (objective="contrast") -----------------------------------------

    def _coords_of_pair(self, lead: int, other: int):
        key = (lead, other)
        if key not in self._pair_coords:
            ctx = np.concatenate([self.split_coords[lead][0], self.split_coords[other][0]])
            pauli = np.concatenate([self.split_coords[lead][1], self.split_coords[other][1]])
            codes = np.unique(ctx.astype(np.int64) * self._n_library + pauli)
            self._pair_coords[key] = (codes // self._n_library, codes % self._n_library)
        return self._pair_coords[key]

    def _contrast_targets(self, lead, sign, other, t) -> dict:
        targets = {l: sign * v for l, v in self.split_coords[lead][2].items()}
        for l, v in self.split_coords[other][2].items():
            targets[l] = targets.get(l, 0.0) - t * v
        return targets

    def _embed(self, p: FragmentProblem, design, weight: float, out: np.ndarray) -> None:
        ctx, pauli, x = design
        codes = p.coord_ctx * self._n_library + p.coord_pauli  # sorted: (ctx, pauli) order
        out[np.searchsorted(codes, ctx.astype(np.int64) * self._n_library + pauli)] += weight * x

    def _design_contrasts(self, model, designs, keys) -> list[list]:
        """Cross-fitted designs of ``s_l g_l - t g_i`` for ``keys = [(l, s_l, i, t)]``."""
        shots = [model.shots(0).astype(float), model.shots(1).astype(float)]
        supported = np.minimum(shots[0], shots[1]) >= self.config.min_fold_shots
        learnable = [np.where(supported, v, 0.0) for v in shots]
        home = self.library.home
        result = [[None] * len(keys), [None] * len(keys)]
        for row, (lead, sign, other, t) in enumerate(keys):
            ctx, pauli = self._coords_of_pair(lead, other)
            targets = self._contrast_targets(lead, sign, other, t)
            folds = [FragmentProblem(-1, ctx, pauli, targets, home,
                                     lambda a, q, f=f: model.covariance(a, q, f)) for f in (0, 1)]
            for f in (0, 1):
                p, check = folds[f], folds[1 - f]
                start = np.zeros(p.n_coordinates)
                self._embed(p, designs[f][lead], sign, start)
                self._embed(p, designs[f][other], -t, start)
                x = start
                if self.config.level != "II-0" and supported[p.ctx_ids].any():
                    p.x = start.copy()
                    p.optimise(learnable[f])
                    if not self.config.guard or (
                        check.variance(shots[1 - f], p.x) < check.variance(shots[1 - f], start)
                    ):
                        x = p.x
                        self.contrast_kept += 1
                keep = x != 0.0
                result[f][row] = (p.coord_ctx[keep], p.coord_pauli[keep], x[keep])
                self._check_row(result[f][row], targets, f"contrast {lead}-{other} ({t:+g}), fold {f}")
        return result

    def _base_designs(self) -> list[list]:
        single = [(p.coord_ctx, p.coord_pauli, p.pauli_target[np.searchsorted(p.pauli_ids, p.coord_pauli)])
                  for p in self.base]
        return [single, single]

    # --- statistics -----------------------------------------------------------

    def _estimates(self, model, designs) -> np.ndarray:
        g = np.zeros(len(designs[0]))
        for f in (0, 1):  # design of fold f applied to the other fold's means
            for i, (ctx, pauli, x) in enumerate(designs[f]):
                total = 0.0
                for alpha in np.unique(ctx):
                    chosen = ctx == alpha
                    total += x[chosen] @ model.means(int(alpha), pauli[chosen], 1 - f)
                g[i] += 0.5 * total
        return g

    def _held_out(self, model, alpha: int, paulis: np.ndarray, fold: int, shots: np.ndarray) -> np.ndarray:
        """One-shot covariance of the held-out fold, or the bound ``k I`` below ``radius_min_shots``."""
        if shots[alpha] < self.config.radius_min_shots:
            return float(len(paulis)) * np.eye(len(paulis))
        return model.covariance(alpha, paulis, fold)

    def _covariance_matrix(self, model, designs, active) -> np.ndarray:
        """Cov(g_i, g_j) of the cross-fitted estimate over the active arms."""
        index = {arm: k for k, arm in enumerate(active)}
        result = np.zeros((len(active), len(active)))
        shots = [model.shots(0), model.shots(1)]
        for f in (0, 1):
            by_context: dict[int, list] = {}
            for arm in active:
                ctx, pauli, x = designs[f][arm]
                for alpha in np.unique(ctx):
                    chosen = ctx == alpha
                    by_context.setdefault(int(alpha), []).append((index[arm], pauli[chosen], x[chosen]))
            for alpha, parts in by_context.items():
                m = shots[1 - f][alpha]
                if m <= 0:
                    continue
                paulis = np.unique(np.concatenate([p for _, p, _ in parts]))
                X = np.zeros((paulis.size, len(parts)))
                for column, (_, ids, values) in enumerate(parts):
                    X[np.searchsorted(paulis, ids), column] = values
                if self.config.radii == "oracle":
                    sigma = self.oracle.covariance(alpha, paulis)
                else:
                    # The held-out fold only: fold f's design was fitted to fold f's
                    # noise, so any covariance containing fold f flatters it and the
                    # radii come out too small.  On LiH the pooled estimate gave 54%
                    # correct selections.  Fold 1-f is independent of the design,
                    # so x^T Sigma_hat x is an unbiased estimate of its variance.
                    sigma = self._held_out(model, alpha, paulis, 1 - f, shots[1 - f])
                K = X.T @ sigma @ X / m
                rows = [k for k, _, _ in parts]
                result[np.ix_(rows, rows)] += 0.25 * K
        return result

    def _variance_by_context(self, model, designs, arm) -> list[tuple]:
        """(context, fold shots, estimated variance, planning sigma) for one arm, largest first."""
        rows = []
        for f in (0, 1):
            ctx, pauli, x = designs[f][arm]
            for alpha in np.unique(ctx):
                chosen = ctx == alpha
                m = model.shots(1 - f)[alpha]
                sigma = model.covariance(int(alpha), pauli[chosen], 1 - f)
                variance = 0.25 * float(x[chosen] @ sigma @ x[chosen]) / max(m, 1)
                plan = model.covariance(int(alpha), pauli[chosen], 1) if f == 0 else None
                rows.append((int(alpha), f, int(model.shots(0)[alpha]), int(model.shots(1)[alpha]),
                             round(variance, 8),
                             None if plan is None else round(float(np.sqrt(max(x[chosen] @ plan @ x[chosen], 0))), 5)))
        return sorted(rows, key=lambda r: -r[4])

    def _plan_sigmas(self, model, designs, active) -> np.ndarray:
        """Per-shot fragment standard deviations, exactly as the radii will see them.

        With ``n`` shots split evenly, the cross-fitted variance in context ``a`` is
        ``(x_A' S_B x_A + x_B' S_A x_B) / (2 n)``, so the planning variance is the
        mean of the two held-out terms.  Planning with one design and one fold
        instead let the two disagree: on H4 side 2.0 a context whose 5-shot fold
        showed zero variance was never topped up, while the other fold's 6 shots
        kept a variance the radii could not shrink, and runs went to the radius
        floor (7.7e6 mean against a median of 1.3e4).
        """
        second = np.zeros((len(active), self.n_contexts))
        shots = [model.shots(0), model.shots(1)]
        for row, arm in enumerate(active):
            for f in (0, 1):
                ctx, pauli, x = designs[f][arm]
                for alpha in np.unique(ctx):
                    chosen = ctx == alpha
                    sigma = self._held_out(model, int(alpha), pauli[chosen], 1 - f, shots[1 - f])
                    second[row, alpha] += 0.5 * max(float(x[chosen] @ sigma @ x[chosen]), 0.0)
        return np.sqrt(second)

    # --- the run -----------------------------------------------------------------

    def _pilot_sigmas(self, active) -> np.ndarray:
        """II-0 pilot.  Before any shots only the flat scale ``||a_{i,alpha}||`` is
        known, unless the radii are oracle anyway."""
        sig = np.zeros((len(active), self.n_contexts))
        for row, i in enumerate(active):
            p = self.base[i]
            if self.config.radii == "oracle":
                sig[row, p.ctx_ids] = np.sqrt(np.maximum(p.context_second_moments(), 0.0))
            else:
                coefficients = p.pauli_target[np.searchsorted(p.pauli_ids, p.coord_pauli)]
                sig[row, p.ctx_ids] = np.sqrt(np.add.reduceat(coefficients ** 2, p.ctx_ptr[:-1]))
        return sig

    def _contrast_keys(self, active, estimates, covariance):
        """``[(l, s_l, i, t)]`` for the empirical leader, or ``None`` before its sign is resolved."""
        magnitude = np.abs(estimates[active])
        radius = self.z_round * np.sqrt(np.maximum(np.diag(covariance), 0.0))
        lead_row = int(np.argmax(magnitude))
        if not magnitude[lead_row] > radius[lead_row]:
            return None
        lead = active[lead_row]
        sign = float(np.sign(estimates[lead]))
        keys = []
        for row, arm in enumerate(active):
            if arm == lead:
                continue
            ts = (float(np.sign(estimates[arm])),) if magnitude[row] > radius[row] else (1.0, -1.0)
            keys.extend((lead, sign, arm, t) for t in ts)
        return keys

    def _validate(self, record: dict, active, estimates, covariance, decision, contrasts) -> None:
        """Oracle check of every interval the round used (validation output only)."""
        truth = self.problem.gradients
        radius = self.z_round * np.sqrt(np.maximum(np.diag(covariance), 0.0))
        missed = bool((np.abs(estimates[active] - truth[active]) > radius).any())
        for (lead, other, t), (value, var) in (contrasts or {}).items():
            exact = np.sign(estimates[lead]) * truth[lead] - t * truth[other]
            missed |= bool(abs(value - exact) > self.z_round * math.sqrt(max(var, 0.0)))
        record["miscovered_rounds"] += int(missed)
        best = int(np.argmax(self.problem.abs_gradients))
        record["best_eliminated"] |= any(arm == best for arm, _, _ in decision.eliminated)

    # --- the run -----------------------------------------------------------------

    def run(self, rng: np.random.Generator, trace: list | None = None, log=None) -> OnlineOutcome:
        """One trial.

        Pass a list as ``trace`` for the compact per-round diagnostics, or a
        :class:`runlog.RunLog` as ``log`` for the full Phase 3 record.
        """
        config = self.config
        truth = self.problem.abs_gradients
        leader = int(np.argmax(truth))
        z = self.z
        active = list(range(self.n_arms))
        # Arms the rule has not yet removed.  With elimination on these are the active
        # arms; with elimination off every arm stays in the allocation (sequential M1:
        # uniform precision for the whole pool) and `alive` only decides when to stop
        # and what to return.
        alive = list(active)
        if config.start == "oracle":
            radius = float(truth.max())
            floor = radius * 1e-3
        else:
            radius = float(self.bound)
            floor = radius * 1e-6
        model = self._model()
        spent = np.zeros(self.n_contexts, dtype=np.int64)
        held = np.flatnonzero(self.credit > 0)
        if held.size:
            first_held = (self.credit[held] + 1) // 2
            model.add(held, (rng.multinomial(first_held, self.distributions[held]),
                             rng.multinomial(self.credit[held] - first_held, self.distributions[held])))
            spent = self.credit.copy()
        designs = self._base_designs()
        fitted_at = None
        refits = 0
        estimates = np.zeros(self.n_arms)
        contrast_rows: dict[tuple, int] = {}
        contrast_designs: list[list] = [[], []]
        plan_keys = None
        self.guard_kept = 0
        self.contrast_kept = 0
        self.z_round = self.z
        check = {"miscovered_rounds": 0, "best_eliminated": False}
        seconds = {"design": 0.0, "statistics": 0.0, "allocation": 0.0}
        stopped = False
        rounds = 0
        for rounds in range(1, 401):
            if stopped or len(alive) <= 1 or radius < floor:
                break
            if config.anytime:
                z = z_from_delta(config.delta * 6.0 / (math.pi ** 2 * rounds ** 2), self.n_arms)
            self.z_round = z
            epsilon = epsilon_from_radius(radius, z)
            clock = time.perf_counter()
            if rounds == 1:
                wanted = allocate(self._pilot_sigmas(active), epsilon)
            else:
                sig = self._plan_sigmas(model, designs, active)
                if plan_keys:
                    rows = [contrast_rows[k] for k in plan_keys]
                    sub = [[contrast_designs[f][r] for r in rows] for f in (0, 1)]
                    lead_sig = sig[active.index(plan_keys[0][0])][None, :]
                    sig = np.vstack([lead_sig, 0.5 * self._plan_sigmas(model, sub, range(len(rows)))])
                wanted = allocate_topup(sig, epsilon, spent.astype(float))
            target = np.maximum(spent, np.ceil(wanted - 1e-9).astype(np.int64))
            target = np.maximum(target, self._minimum)
            seconds["allocation"] += time.perf_counter() - clock
            added = target - spent
            grew = np.flatnonzero(added > 0)
            first = (added[grew] + 1) // 2
            if grew.size:
                fold0 = rng.multinomial(first, self.distributions[grew])
                fold1 = rng.multinomial(added[grew] - first, self.distributions[grew])
                model.add(grew, (fold0, fold1))
            spent = target

            refitted = False
            clock = time.perf_counter()
            if config.level != "II-0":
                # New shots only: data held in advance (credit) must not postpone the refits.
                total = max(int(spent.sum() - self.credit.sum()), 1)
                if fitted_at is None or total >= config.refit_growth * fitted_at:
                    designs = self._refit(model, set(active))
                    fitted_at = total
                    refitted = True
                    refits += 1
                    contrast_rows, contrast_designs = {}, [[], []]
            seconds["design"] += time.perf_counter() - clock

            clock = time.perf_counter()
            estimates = self._estimates(model, designs)
            covariance = self._covariance_matrix(model, designs, active)
            seconds["statistics"] += time.perf_counter() - clock
            contrasts = None
            keys = self._contrast_keys(active, estimates, covariance) if config.objective == "contrast" else None
            if keys:
                missing = [k for k in keys if k not in contrast_rows]
                if missing:
                    clock = time.perf_counter()
                    fresh = self._design_contrasts(model, designs, missing)
                    for k, key in enumerate(missing):
                        contrast_rows[key] = len(contrast_designs[0])
                        for f in (0, 1):
                            contrast_designs[f].append(fresh[f][k])
                    seconds["design"] += time.perf_counter() - clock
                clock = time.perf_counter()
                rows = [contrast_rows[k] for k in keys]
                sub = [[contrast_designs[f][r] for r in rows] for f in (0, 1)]
                values = self._estimates(model, sub)
                variances = np.diag(self._covariance_matrix(model, sub, list(range(len(rows)))))
                contrasts = {(lead, other, t): (float(values[k]), float(variances[k]))
                             for k, (lead, _, other, t) in enumerate(keys)}
                seconds["statistics"] += time.perf_counter() - clock
            if config.elimination == "off":
                # Every arm is measured, but only the arms still alive may eliminate or be
                # eliminated; `active` is all arms, so positions in `covariance` are arm ids.
                decision = eliminate(alive, estimates, covariance[np.ix_(alive, alive)], z, config.rule)
            else:
                decision = eliminate(active, estimates, covariance, z, config.rule, contrasts)
            self._validate(check, active, estimates, covariance, decision, contrasts)

            if trace is not None:
                self._trace(trace, rounds, radius, active, spent, estimates, covariance, model, designs)
            if log is not None:
                log.round(self, rounds=rounds, radius=radius, epsilon=epsilon, active=active,
                          added=(grew, first, added[grew] - first), spent=spent, estimates=estimates,
                          covariance=covariance, decision=decision, contrasts=contrasts,
                          refitted=refitted, seconds=dict(seconds))
                if refitted:
                    log.refit(self, refits, rounds, model, designs)
            survivors = decision.survivors  # already a subset of `alive`
            plan_keys = [k for k in keys if k[2] in survivors] if keys else None
            if config.rho > 0 and len(survivors) > 1:
                keep = [active.index(arm) for arm in survivors]
                stopped = rho_good_stop(survivors, estimates, covariance[np.ix_(keep, keep)], z, config.rho)
            alive = survivors
            if config.elimination == "on":
                active = survivors
            radius *= config.shrink

        selected = max(alive, key=lambda i: abs(estimates[i]))
        extra = {
            "shortfall": float(1.0 - truth[selected] / truth.max()) if truth.max() > 0 else 0.0,
            "miscovered_rounds": check["miscovered_rounds"],
            "best_eliminated": bool(check["best_eliminated"]),
            "stopped_rho": bool(stopped),
            "refits": refits,
            "contexts_used": int((spent > self.credit).sum()),
            "cz_per_shot_mean": float(((spent - self.credit) * self.cz).sum() / max(spent.sum() - self.credit.sum(), 1)),
            **{f"{k}_seconds": round(v, 3) for k, v in seconds.items()},
        }
        if log is not None:
            log.final(self, model, spent, selected, extra)
        return OnlineOutcome(selected, selected == leader, float(spent.sum() - self.credit.sum()), rounds,
                             extra=extra)

    def _trace(self, trace, rounds, radius, active, spent, estimates, covariance, model, designs) -> None:
        diagonal = np.diag(covariance)
        magnitudes = np.abs(estimates[active])
        signs = np.sign(estimates[active])
        pairwise = diagonal[:, None] + diagonal[None, :] - 2.0 * np.outer(signs, signs) * covariance
        order = np.argsort(-magnitudes)[:2]
        top = [active[k] for k in order]
        record = {"round": rounds, "radius": radius, "n_active": len(active),
                  "shots": int(spent.sum()), "top": top,
                  "estimates": [float(estimates[i]) for i in top],
                  "truth": [float(self.problem.gradients[i]) for i in top],
                  "sd": [float(np.sqrt(max(diagonal[k], 0.0))) for k in order]}
        if len(order) == 2:
            a, b = order
            record["lead"] = float(magnitudes[a] - magnitudes[b])
            record["pair_radius"] = float(self.z * np.sqrt(max(pairwise[b, a], 0.0)))
            record["runner_up_contexts"] = self._variance_by_context(model, designs, active[b])[:4]
        trace.append(record)
