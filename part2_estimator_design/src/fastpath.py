"""Faster, equivalent statistics for :class:`learning.LearnedM3` (opt-in: ``fastpath.install()``).

The profile of one H2O selection (II-A, 140 arms, 2,366 contexts) shows that almost all
of its time is Python overhead: the estimates, the planning sigmas and the covariance
of the estimates loop over every (arm, context, fold) and call a small NumPy routine for
each, and every refit builds the fragment problems of all 140 arms, active or not.

This module computes exactly the same quantities context by context instead:

* a *layout* of the current designs (built once per design, i.e. once per refit) holds,
  for every fold and context, the Paulis it carries and a dense coefficient matrix
  ``X`` (Paulis x arms);
* estimates: ``g += 0.5 X^T mu`` per context; planning sigmas: ``diag(X^T Sigma X)``
  per context; covariance of the estimates: ``X^T Sigma X / m`` per context;
* refits build fragment problems only for the active arms; the others keep their
  home (II-0) coefficients, which is what the original code gives them.

Every rule of the original is kept, including the small-sample bound ``Sigma <= k I``
with ``k`` the number of Paulis that the original code would have passed (the arm's own
Paulis in the planning sigmas, the union over the active arms in the covariance).
Results agree with the original to floating-point round-off; ``tests`` compare
selections, shots and rounds trial by trial.
"""
from __future__ import annotations

import numpy as np

import learning
from design import FragmentProblem

_ORIGINAL = {}


class Layout:
    """Per fold and context: (Pauli ids, arm ids, X[pauli, arm]) of a design list."""

    def __init__(self, designs) -> None:
        self.n_arms = len(designs[0])
        self.folds = []
        for f in (0, 1):
            ctx_all, pauli_all, x_all, arm_all = [], [], [], []
            for arm, (ctx, pauli, x) in enumerate(designs[f]):
                if len(ctx) == 0:
                    continue
                ctx_all.append(np.asarray(ctx, dtype=np.int64))
                pauli_all.append(np.asarray(pauli, dtype=np.int64))
                x_all.append(np.asarray(x, dtype=float))
                arm_all.append(np.full(len(ctx), arm, dtype=np.int64))
            groups = {}
            if ctx_all:
                ctx = np.concatenate(ctx_all)
                pauli = np.concatenate(pauli_all)
                x = np.concatenate(x_all)
                arm = np.concatenate(arm_all)
                order = np.lexsort((arm, pauli, ctx))
                ctx, pauli, x, arm = ctx[order], pauli[order], x[order], arm[order]
                starts = np.flatnonzero(np.diff(ctx)) + 1
                for lo, hi in zip(np.concatenate([[0], starts]), np.concatenate([starts, [ctx.size]])):
                    p, a, v = pauli[lo:hi], arm[lo:hi], x[lo:hi]
                    paulis, prow = np.unique(p, return_inverse=True)
                    arms, acol = np.unique(a, return_inverse=True)
                    X = np.zeros((paulis.size, arms.size))
                    X[prow, acol] = v
                    groups[int(ctx[lo])] = (paulis, arms, X)
            self.folds.append(groups)


def _layout(self, designs) -> Layout:
    key = (id(designs[0]), id(designs[1]), len(designs[0]))
    cache = self.__dict__.setdefault("_fast_layouts", {})
    hit = cache.get(key)
    if hit is not None and hit[0] is designs[0] and hit[1] is designs[1]:
        return hit[2]
    layout = Layout(designs)
    if len(cache) >= 2:  # a layout of H2O takes a few hundred MB: keep the current design and one more
        cache.pop(next(iter(cache)))
    cache[key] = (designs[0], designs[1], layout)
    return layout


def _estimates(self, model, designs) -> np.ndarray:
    layout = _layout(self, designs)
    g = np.zeros(layout.n_arms)
    for f in (0, 1):
        for alpha, (paulis, arms, X) in layout.folds[f].items():
            mu = model.means(alpha, paulis, 1 - f)
            np.add.at(g, arms, 0.5 * (X.T @ mu))
    return g


def _plan_sigmas(self, model, designs, active) -> np.ndarray:
    layout = _layout(self, designs)
    active = list(active)
    row_of = np.full(layout.n_arms, -1, dtype=np.int64)
    row_of[np.asarray(active, dtype=np.int64)] = np.arange(len(active))
    second = np.zeros((len(active), self.n_contexts))
    shots = [model.shots(0), model.shots(1)]
    threshold = self.config.radius_min_shots
    for f in (0, 1):
        held = shots[1 - f]
        for alpha, (paulis, arms, X) in layout.folds[f].items():
            rows = row_of[arms]
            keep = rows >= 0
            if not keep.any():
                continue
            Xa = X[:, keep]
            if held[alpha] < threshold:
                # the bound k I with k = the arm's own number of Paulis in this context
                k = (Xa != 0).sum(axis=0)
                q = k * (Xa ** 2).sum(axis=0)
            else:
                sigma = model.covariance(alpha, paulis, 1 - f)
                q = np.einsum("pa,pq,qa->a", Xa, sigma, Xa)
            second[rows[keep], alpha] += 0.5 * np.maximum(q, 0.0)
    return np.sqrt(second)


def _covariance_matrix(self, model, designs, active, exact: bool = False) -> np.ndarray:
    layout = _layout(self, designs)
    active = list(active)
    row_of = np.full(layout.n_arms, -1, dtype=np.int64)
    row_of[np.asarray(active, dtype=np.int64)] = np.arange(len(active))
    result = np.zeros((len(active), len(active)))
    shots = [model.shots(0), model.shots(1)]
    oracle_radii = exact or self.config.radii == "oracle"
    for f in (0, 1):
        for alpha, (paulis, arms, X) in layout.folds[f].items():
            m = shots[1 - f][alpha]
            if m <= 0:
                continue
            rows = row_of[arms]
            keep = rows >= 0
            if not keep.any():
                continue
            Xa = X[:, keep]
            used = (Xa != 0).any(axis=1)  # the union over the active arms, as the original builds it
            Xa, sub = Xa[used], paulis[used]
            if oracle_radii:
                sigma = self.oracle.covariance(alpha, sub)
            else:
                sigma = self._held_out(model, alpha, sub, 1 - f, shots[1 - f])
            K = Xa.T @ sigma @ Xa / m
            idx = rows[keep]
            result[np.ix_(idx, idx)] += 0.25 * K
    return result


def _home_designs(self):
    homes = self.__dict__.get("_fast_homes")
    if homes is None:
        zero = lambda a, p: np.zeros((len(p), len(p)))  # noqa: E731  (blocks unused for the home design)
        homes = []
        for i, (ctx, pauli, targets) in enumerate(self.split_coords):
            p = FragmentProblem(i, ctx, pauli, targets, self.library.home, zero)
            x = self._home_x(p)
            keep = x != 0.0
            homes.append((p.coord_ctx[keep], p.coord_pauli[keep], x[keep]))
        self._fast_homes = homes
    return homes


_UNIONS: dict = {}


def _context_unions(split_coords) -> dict[int, np.ndarray]:
    """Per context, the sorted Paulis that any arm's II-A coordinates place in it."""
    key = id(split_coords)
    hit = _UNIONS.get(key)
    if hit is not None and hit[0] is split_coords:
        return hit[1]
    ctx = np.concatenate([np.asarray(c, dtype=np.int64) for c, _, _ in split_coords])
    pauli = np.concatenate([np.asarray(p, dtype=np.int64) for _, p, _ in split_coords])
    order = np.lexsort((pauli, ctx))
    ctx, pauli = ctx[order], pauli[order]
    starts = np.flatnonzero(np.diff(ctx)) + 1
    unions = {}
    for lo, hi in zip(np.concatenate([[0], starts]), np.concatenate([starts, [ctx.size]])):
        unions[int(ctx[lo])] = np.unique(pauli[lo:hi])
    _UNIONS.clear()
    _UNIONS[key] = (split_coords, unions)
    return unions


class _RefitCovariance:
    """``model.covariance`` for one refit, computed once per (fold, context) on the context's union of Paulis.

    Every entry of the shrinkage covariance is a function of its own pair of Paulis only, so the
    sub-block of the union's matrix is bit-for-bit the matrix the original call returns."""

    def __init__(self, model, unions) -> None:
        self.model = model
        self.unions = unions
        self.full: dict[tuple[int, int], np.ndarray] = {}

    def __call__(self, alpha: int, paulis: np.ndarray, fold: int) -> np.ndarray:
        key = (fold, alpha)
        full = self.full.get(key)
        union = self.unions[alpha]
        if full is None:
            full = self.model.covariance(alpha, union, fold)
            self.full[key] = full
        idx = np.searchsorted(union, paulis)
        return full[np.ix_(idx, idx)]


def _refit(self, model, active) -> list[list]:
    cov = _RefitCovariance(model, _context_unions(self.split_coords))
    shots = [model.shots(0).astype(float), model.shots(1).astype(float)]
    supported = np.minimum(shots[0], shots[1]) >= self.config.min_fold_shots
    learnable = [np.where(supported, s, 0.0) for s in shots]
    homes = _home_designs(self)
    designs = [list(homes), list(homes)]
    self.guard_kept = 0
    home_map = self.library.home
    for i in sorted(active):
        ctx, pauli, targets = self.split_coords[i]
        # only the arms that may learn need fragment problems with blocks
        if not supported[np.unique(ctx)].any():
            continue
        folds = [FragmentProblem(i, ctx, pauli, targets, home_map, lambda a, p, f=f: cov(a, p, f))
                 for f in (0, 1)]
        for f in (0, 1):
            p, check = folds[f], folds[1 - f]
            home = self._home_x(p)
            x = home
            p.x = home.copy()
            p.optimise(learnable[f])
            p._structures.clear()
            candidate = p.x
            if not self.config.guard or (
                check.variance(shots[1 - f], candidate) < check.variance(shots[1 - f], home)
            ):
                x = candidate
                self.guard_kept += 1
            keep = x != 0.0
            designs[f][i] = (p.coord_ctx[keep], p.coord_pauli[keep], x[keep])
            self._check_row(designs[f][i], targets, f"arm {i}, fold {f}")
    return designs


def install() -> None:
    """Replace the slow methods of :class:`learning.LearnedM3` (idempotent)."""
    if _ORIGINAL:
        return
    for name, fn in (("_estimates", _estimates), ("_plan_sigmas", _plan_sigmas),
                     ("_covariance_matrix", _covariance_matrix), ("_refit", _refit)):
        _ORIGINAL[name] = getattr(learning.LearnedM3, name)
        setattr(learning.LearnedM3, name, fn)


def uninstall() -> None:
    for name, fn in _ORIGINAL.items():
        setattr(learning.LearnedM3, name, fn)
    _ORIGINAL.clear()
