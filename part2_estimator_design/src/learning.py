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
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from allocation import allocate, allocate_topup
from design import FragmentProblem
from online import OnlineOutcome
from part1_bridge import epsilon_from_radius, z_from_delta
from sampler import pauli_covariance, walsh_hadamard

PRIORS = ("hf", "flat", "oracle", "none")


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

    @property
    def label(self) -> str:
        if self.level == "II-0":
            return f"II-0/{self.radii} radii"
        nu = "inf" if math.isinf(self.nu) else f"{self.nu:g}"
        return f"{self.level}/prior={self.prior}/nu={nu}/{self.radii} radii/guard={'on' if self.guard else 'off'}"


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

    def __init__(self, problem, library, oracle_moments, prior_moments, base_problems, split_coords, config):
        self.problem = problem
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
        return designs

    def _base_designs(self) -> list[list]:
        single = [(p.coord_ctx, p.coord_pauli, p.pauli_target[np.searchsorted(p.pauli_ids, p.coord_pauli)])
                  for p in self.base]
        return [single, single]

    # --- statistics -----------------------------------------------------------

    def _estimates(self, model, designs) -> np.ndarray:
        g = np.zeros(self.n_arms)
        for f in (0, 1):  # design of fold f applied to the other fold's means
            for i, (ctx, pauli, x) in enumerate(designs[f]):
                total = 0.0
                for alpha in np.unique(ctx):
                    chosen = ctx == alpha
                    total += x[chosen] @ model.means(int(alpha), pauli[chosen], 1 - f)
                g[i] += 0.5 * total
        return g

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
                    sigma = model.covariance(alpha, paulis, 1 - f)
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
        for row, arm in enumerate(active):
            for f in (0, 1):
                ctx, pauli, x = designs[f][arm]
                for alpha in np.unique(ctx):
                    chosen = ctx == alpha
                    sigma = model.covariance(int(alpha), pauli[chosen], 1 - f)
                    second[row, alpha] += 0.5 * max(float(x[chosen] @ sigma @ x[chosen]), 0.0)
        return np.sqrt(second)

    # --- the run -----------------------------------------------------------------

    def run(self, rng: np.random.Generator, trace: list | None = None) -> OnlineOutcome:
        """One trial; pass a list as ``trace`` to record every round (diagnostics)."""
        truth = self.problem.abs_gradients
        leader = int(np.argmax(truth))
        z = self.z
        active = list(range(self.n_arms))
        radius = float(truth.max())
        floor = radius * 1e-3
        model = self._model()
        spent = np.zeros(self.n_contexts, dtype=np.int64)
        designs = self._base_designs()
        fitted_at = None
        estimates = np.zeros(self.n_arms)
        rounds = 0
        self.guard_kept = 0
        for rounds in range(1, 401):
            if len(active) <= 1 or radius < floor:
                break
            epsilon = epsilon_from_radius(radius, z)
            if rounds == 1:
                # II-0 pilot.  Before any shots only the flat scale ||a_{i,alpha}||
                # is known, unless the radii are oracle anyway.
                sig = np.zeros((len(active), self.n_contexts))
                for row, i in enumerate(active):
                    p = self.base[i]
                    if self.config.radii == "oracle":
                        sig[row, p.ctx_ids] = np.sqrt(np.maximum(p.context_second_moments(), 0.0))
                    else:
                        coefficients = p.pauli_target[np.searchsorted(p.pauli_ids, p.coord_pauli)]
                        sig[row, p.ctx_ids] = np.sqrt(np.add.reduceat(coefficients ** 2, p.ctx_ptr[:-1]))
                wanted = allocate(sig, epsilon)
            else:
                wanted = allocate_topup(self._plan_sigmas(model, designs, active), epsilon, spent.astype(float))
            target = np.maximum(spent, np.ceil(wanted - 1e-9).astype(np.int64))
            target = np.maximum(target, self._minimum)
            added = target - spent
            grew = np.flatnonzero(added > 0)
            if grew.size:
                first = (added[grew] + 1) // 2
                fold0 = rng.multinomial(first, self.distributions[grew])
                fold1 = rng.multinomial(added[grew] - first, self.distributions[grew])
                model.add(grew, (fold0, fold1))
            spent = target

            if self.config.level != "II-0":
                total = spent.sum()
                if fitted_at is None or total >= self.config.refit_growth * fitted_at:
                    designs = self._refit(model, set(active))
                    fitted_at = total
            estimates = self._estimates(model, designs)
            covariance = self._covariance_matrix(model, designs, active)
            diagonal = np.diag(covariance)
            signs = np.sign(estimates[active])
            pairwise = diagonal[:, None] + diagonal[None, :] - 2.0 * np.outer(signs, signs) * covariance
            magnitudes = np.abs(estimates[active])
            lead = magnitudes[None, :] - magnitudes[:, None]
            dominated = (lead > z * np.sqrt(np.maximum(pairwise, 0.0))).any(axis=1)
            if trace is not None:
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
                    record["pair_radius"] = float(z * np.sqrt(max(pairwise[b, a], 0.0)))
                    record["runner_up_contexts"] = self._variance_by_context(model, designs, active[b])[:4]
                trace.append(record)
            active = [i for k, i in enumerate(active) if not dominated[k]]
            radius *= self.config.shrink

        selected = max(active, key=lambda i: abs(estimates[i]))
        return OnlineOutcome(selected, selected == leader, float(spent.sum()), rounds)
