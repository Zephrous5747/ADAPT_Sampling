"""Fragment designs over a fixed context library: levels II-0, II-A and II-B.

For a fixed library of measurement contexts, every linear unbiased estimator of
the gradients has the form

    g_i_hat = sum_alpha x_{i,alpha}^T mu_hat_alpha,

where ``mu_hat_alpha`` holds the sample means of the library Paulis measured by
context ``alpha`` (the whole measured group, see :mod:`clifford`) and the
coefficients satisfy, Pauli by Pauli,

    sum_{alpha measuring l} x_{i,alpha,l} = A_{i,l}         (exact unbiasedness).

The fragment ``F_{i,alpha} = sum_l x_{i,alpha,l} P_l`` is a universal measurable
operator of the Part II work order, ``B`` is the matrix selecting generator ``i``'s
fragments, and the constraint above is exactly ``A = BC``.  The levels differ only
in which coordinates ``(alpha, l)`` may be non-zero:

``II-0``
    ``l`` in ``supp(C_i)`` and ``alpha`` its Part I parent context: nothing is free,
    and the estimator is Part I's.
``II-A``
    ``l`` in ``supp(C_i)`` and any ``alpha`` that measures it: coefficient splitting.
``II-B``
    II-A plus auxiliary ``l`` with ``A_{i,l} = 0`` measured by at least two contexts
    that already carry generator ``i``: zero-sum ("ghost") control variates.  The
    auxiliary pool is the rest of the library -- Pauli products that appear in
    *other* gradient commutators, rule (i) of the work order -- ranked by a
    single-context control-variate score when a cap is imposed.

With the shots ``n`` fixed, ``Var(g_i_hat) = sum_alpha x^T Sigma_alpha x / n_alpha``
is a convex quadratic in generator ``i``'s coefficients alone, so each generator
is re-optimised exactly by a small linear solve over the free directions (moving
coefficient mass between copies of one Pauli): :meth:`FragmentProblem.optimise`.
That is the right tool when the shots are given, for instance to post-process
data already taken.

It is the wrong tool for the *joint* design.  Alternating it with the allocation
is iterative coefficient splitting (ICS); it is monotone, but it is iteratively
reweighted least squares on a sum of norms and stalls where the optimum empties a
context: once a context's shots reach zero, no coefficient can move back into it.
On H4 side 1.0 HF it stops 9% above the optimum of the single-arm problem.
:meth:`DesignSet.solve` therefore solves the joint programme through its saddle
form.  Eliminating the shots analytically,

    min_{x,n} sum n  s.t.  sum_alpha q_{i,alpha}(x)/n_alpha <= eps**2
      =  min_x max_{lambda >= 0}  sum_alpha h_alpha(Q_alpha) - eps**2 sum_i lambda_i,

with ``q_{i,alpha} = x^T Sigma_alpha x``, ``Q_alpha = sum_i lambda_i q_{i,alpha}``,
``n_alpha = max(c_alpha, sqrt(Q_alpha))`` for committed shots ``c`` and
``h(Q) = n + Q/n``.  The inner minimisation over the coefficients of all active
arms at once is smooth after replacing ``Q`` by ``Q + delta**2`` and is solved by
L-BFGS; the outer multipliers take Part I's multiplicative dual-ascent step
``lambda_i <- lambda_i (Var_i / eps**2)**(1/2)``.  The cost reported for a design is
always recomputed by the exact allocation of that design, so it is feasible
whether or not the iteration has converged; convergence affects only how close
to optimal it is, and it is checked against independent solvers in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from allocation import allocate, reconstruction_variances

LEVELS = ("II-0", "II-A", "II-B")
DENSE_LIMIT = 2500
TARGET_TOLERANCE = 1e-9
DELTA_START, DELTA_SHRINK, DELTA_FLOOR = 0.3, 0.3, 1e-7


class FragmentProblem:
    """The design problem of one gradient operator over a fixed context library."""

    def __init__(
        self,
        generator: int,
        coord_ctx: np.ndarray,
        coord_pauli: np.ndarray,
        targets: dict[int, float],
        home: np.ndarray,
        covariance: Callable[[int, np.ndarray], np.ndarray],
    ) -> None:
        order = np.lexsort((coord_pauli, coord_ctx))
        self.generator = int(generator)
        self.coord_ctx = np.asarray(coord_ctx, dtype=np.int64)[order]
        self.coord_pauli = np.asarray(coord_pauli, dtype=np.int64)[order]
        self.ctx_ids, starts = np.unique(self.coord_ctx, return_index=True)
        self.ctx_ptr = np.append(starts, self.coord_ctx.size)
        self.coord_block = np.repeat(np.arange(self.ctx_ids.size), np.diff(self.ctx_ptr))
        self.update_blocks(covariance)

        self.pauli_ids, inverse = np.unique(self.coord_pauli, return_inverse=True)
        by_pauli = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=self.pauli_ids.size)
        self.pauli_coords = np.split(by_pauli, np.cumsum(counts)[:-1])
        self.pauli_target = np.array([targets.get(int(p), 0.0) for p in self.pauli_ids])
        self.reference = np.empty(self.pauli_ids.size, dtype=np.int64)
        for p, (pauli, coords) in enumerate(zip(self.pauli_ids, self.pauli_coords)):
            at_home = coords[self.coord_ctx[coords] == home[pauli]] if home[pauli] >= 0 else []
            self.reference[p] = at_home[0] if len(at_home) else coords[0]
        self.x = np.zeros(self.coord_ctx.size)
        self.x[self.reference] = self.pauli_target
        self._structures: dict[bytes, dict | None] = {}

    def update_blocks(self, covariance: Callable[[int, np.ndarray], np.ndarray]) -> None:
        """Replace the covariance model (Step 4 refits it from data)."""
        self.blocks = [
            covariance(int(alpha), self.coord_pauli[self.ctx_ptr[k] : self.ctx_ptr[k + 1]])
            for k, alpha in enumerate(self.ctx_ids)
        ]
        self._block_matrix = (
            sp.block_diag([sp.coo_matrix(self.blocks[0])] + self.blocks[1:], format="csr")
            if self.blocks
            else None
        )
        self._structures = {}

    # --- evaluation -----------------------------------------------------------

    @property
    def n_coordinates(self) -> int:
        return int(self.coord_ctx.size)

    @property
    def n_splittable(self) -> int:
        return int(sum(len(c) > 1 for c in self.pauli_coords))

    def context_second_moments(self, x: np.ndarray | None = None) -> np.ndarray:
        """``x_alpha^T Sigma_alpha x_alpha`` for every context in :attr:`ctx_ids`."""
        x = self.x if x is None else x
        if self._block_matrix is None:
            return np.zeros(0)
        products = x * (self._block_matrix @ x)
        return np.add.reduceat(products, self.ctx_ptr[:-1]) if products.size else products

    def variance(self, shots: np.ndarray, x: np.ndarray | None = None) -> float:
        second = np.maximum(self.context_second_moments(x), 0.0)
        return float(reconstruction_variances(second[None, :], shots[self.ctx_ids])[0])

    def constraint_residual(self) -> float:
        """Largest violation of ``sum over copies = A_il``, Pauli by Pauli."""
        totals = np.array([self.x[c].sum() for c in self.pauli_coords])
        return float(np.abs(totals - self.pauli_target).max()) if totals.size else 0.0

    # --- optimisation -----------------------------------------------------------

    def _structure(self, usable: np.ndarray) -> dict | None:
        """Free directions and the Hessian's sparsity for one usable-context pattern."""
        key = usable.tobytes()
        if key in self._structures:
            return self._structures[key]
        usable_coord = usable[self.coord_block]
        var_coord, var_ref = [], []
        for p, coords in enumerate(self.pauli_coords):
            free = coords[usable_coord[coords]]
            if free.size < 2:
                continue
            ref = self.reference[p] if usable_coord[self.reference[p]] else free[0]
            for c in free:
                if c != ref:
                    var_coord.append(c)
                    var_ref.append(ref)
        if not var_coord:
            self._structures[key] = None
            return None
        n_vars = len(var_coord)
        var_coord = np.asarray(var_coord)
        var_ref = np.asarray(var_ref)
        columns = np.arange(n_vars)
        N = sp.csr_matrix(
            (
                np.concatenate([np.ones(n_vars), -np.ones(n_vars)]),
                (np.concatenate([var_coord, var_ref]), np.concatenate([columns, columns])),
            ),
            shape=(self.n_coordinates, n_vars),
        )
        # Hessian = sum_k (1/n_k) N_k^T B_k N_k, assembled from per-block triplets.
        entry_coord = np.concatenate([var_coord, var_ref])
        entry_var = np.concatenate([columns, columns])
        entry_sign = np.concatenate([np.ones(n_vars), -np.ones(n_vars)])
        entry_block = self.coord_block[entry_coord]
        order = np.argsort(entry_block, kind="stable")
        entry_coord, entry_var, entry_sign, entry_block = (
            entry_coord[order], entry_var[order], entry_sign[order], entry_block[order]
        )
        bounds = np.searchsorted(entry_block, np.arange(self.ctx_ids.size + 1))
        rows, cols, vals, labels = [], [], [], []
        for k in range(self.ctx_ids.size):
            lo, hi = bounds[k], bounds[k + 1]
            if lo == hi:
                continue
            local = entry_coord[lo:hi] - self.ctx_ptr[k]
            signs = entry_sign[lo:hi]
            block = self.blocks[k][np.ix_(local, local)] * np.outer(signs, signs)
            variables = entry_var[lo:hi]
            rows.append(np.repeat(variables, variables.size))
            cols.append(np.tile(variables, variables.size))
            vals.append(block.ravel())
            labels.append(np.full(block.size, k))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        pair = rows * n_vars + cols
        unique_pairs, slot = np.unique(pair, return_inverse=True)
        structure = {
            "N": N,
            "n_vars": n_vars,
            "vals": np.concatenate(vals),
            "labels": np.concatenate(labels),
            "slot": slot,
            "rows": unique_pairs // n_vars,
            "cols": unique_pairs % n_vars,
            "n_pairs": unique_pairs.size,
        }
        self._structures[key] = structure
        return structure

    def optimise(self, shots: np.ndarray) -> float:
        """Minimise this generator's variance at fixed ``shots``; return it."""
        block_shots = shots[self.ctx_ids]
        usable = block_shots > 0
        structure = self._structure(usable)
        if structure is None:
            return self.variance(shots)
        inverse = np.zeros(self.ctx_ids.size)
        inverse[usable] = 1.0 / block_shots[usable]
        data = np.bincount(
            structure["slot"],
            weights=structure["vals"] * inverse[structure["labels"]],
            minlength=structure["n_pairs"],
        )
        n_vars = structure["n_vars"]
        hessian = sp.csr_matrix((data, (structure["rows"], structure["cols"])), shape=(n_vars, n_vars))
        weighted = (self._block_matrix @ self.x) * inverse[self.coord_block]
        gradient = structure["N"].T @ weighted
        step = _solve_psd(hessian, -gradient)
        candidate = self.x + structure["N"] @ step
        before = self.variance(shots)
        after = self.variance(shots, candidate)
        if after <= before:
            self.x = candidate
            return after
        return before


def _solve_psd(matrix: sp.csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve a positive semidefinite system with a vanishing ridge."""
    n = matrix.shape[0]
    diagonal = matrix.diagonal()
    scale = float(diagonal.mean()) if n else 0.0
    if scale <= 0:
        return np.zeros(n)
    ridge = 1e-12 * scale
    if n <= DENSE_LIMIT:
        dense = matrix.toarray()
        dense[np.diag_indices(n)] += ridge
        try:
            return scipy.linalg.solve(dense, rhs, assume_a="pos")
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            return np.linalg.lstsq(dense, rhs, rcond=None)[0]
    # Sparse direct factorisation.  Conjugate gradients was used here first and
    # needed ~36,000 iterations per solve on these nearly singular systems: 95% of
    # a LiH Step 4 trial (1,205 of 1,267 s) went into it.
    regularised = (matrix + ridge * sp.identity(n, format="csr")).tocsc()
    try:
        return spla.spsolve(regularised, rhs, permc_spec="MMD_AT_PLUS_A")
    except RuntimeError:  # pragma: no cover - singular despite the ridge
        precondition = sp.diags(1.0 / np.maximum(diagonal + ridge, 1e-300))
        solution, _ = spla.cg(regularised, rhs, rtol=1e-10, maxiter=20 * n, M=precondition)
        return solution


# --- building the problems of one level ------------------------------------------


def build_fragment_problems(
    problem,
    library,
    moments,
    level: str,
    *,
    aux_cap: int | None = None,
) -> list[FragmentProblem]:
    """One :class:`FragmentProblem` per generator at the given level."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    index = {label: position for position, label in enumerate(library.labels)}
    owners = library.contexts_of() if level != "II-0" else None
    problems = []
    for i, terms in enumerate(problem.commutator_terms):
        support = np.array([index[label] for label in terms], dtype=np.int64)
        targets = {index[label]: value for label, value in terms.items()}
        if level == "II-0":
            coord_ctx = library.home[support]
            coord_pauli = support
        else:
            coord_ctx = np.concatenate([owners[l] for l in support])
            coord_pauli = np.repeat(support, [owners[l].size for l in support])
            if level == "II-B":
                aux_ctx, aux_pauli = _auxiliary_coordinates(
                    library, moments, support, targets, np.unique(coord_ctx), aux_cap
                )
                coord_ctx = np.concatenate([coord_ctx, aux_ctx])
                coord_pauli = np.concatenate([coord_pauli, aux_pauli])
        problems.append(
            FragmentProblem(i, coord_ctx, coord_pauli, targets, library.home, moments.covariance)
        )
    return problems


def _auxiliary_coordinates(
    library, moments, support, targets, carrying, cap
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-sum auxiliary coordinates for one generator (rule (i) of the work order).

    A candidate must be measured by at least two of the contexts that already
    carry the generator, otherwise its zero-sum constraint forces it to zero.
    When ``cap`` is set, candidates are ranked by the variance a single-context
    control variate could remove from the generator's Part I fragments,
    ``sum_alpha Cov(F_alpha, P)**2 / Var(P)``.
    """
    carrying = np.asarray(carrying, dtype=np.int64)
    members = [library.contexts[a].members for a in carrying]
    counts = np.bincount(np.concatenate(members), minlength=library.n_library)
    counts[support] = 0
    candidates = np.flatnonzero(counts >= 2)
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if cap is not None and candidates.size > cap:
        score = np.zeros(library.n_library)
        is_candidate = np.zeros(library.n_library, dtype=bool)
        is_candidate[candidates] = True
        home_of = library.home[support]
        for alpha in np.unique(home_of):
            fragment = support[home_of == alpha]
            coefficients = np.array([targets[int(l)] for l in fragment])
            context = library.contexts[int(alpha)]
            local = context.members[is_candidate[context.members]]
            if local.size == 0:
                continue
            both = np.concatenate([fragment, local])
            covariance = moments.covariance(int(alpha), both)
            cross = coefficients @ covariance[: fragment.size, fragment.size :]
            variance = np.diag(covariance)[fragment.size :]
            useful = variance > 1e-14
            score[local[useful]] += cross[useful] ** 2 / variance[useful]
        ranked = candidates[np.argsort(-score[candidates], kind="stable")]
        candidates = np.sort(ranked[:cap])
    owners = library.contexts_of()
    carrying_set = np.zeros(library.n_contexts, dtype=bool)
    carrying_set[carrying] = True
    aux_ctx, aux_pauli = [], []
    for l in candidates:
        contexts = owners[l][carrying_set[owners[l]]]
        if contexts.size >= 2:
            aux_ctx.append(contexts)
            aux_pauli.append(np.full(contexts.size, l))
    if not aux_ctx:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(aux_ctx), np.concatenate(aux_pauli)


# --- the joint design ---------------------------------------------------------------


@dataclass
class JointSolution:
    shots: np.ndarray
    total: float
    iterations: int
    history: list[float] = field(default_factory=list)


class DesignSet:
    """The fragment problems of every generator, optimised jointly with the shots."""

    def __init__(self, problems: Sequence[FragmentProblem], n_contexts: int) -> None:
        self.problems = list(problems)
        self.n_contexts = int(n_contexts)

    def snapshot(self) -> list[np.ndarray]:
        return [p.x.copy() for p in self.problems]

    def restore(self, snapshot: list[np.ndarray]) -> None:
        for p, x in zip(self.problems, snapshot):
            p.x = x.copy()

    def sigmas(self, arms: Sequence[int]) -> np.ndarray:
        """``sigma[i, alpha] = sqrt(x^T Sigma x)`` for the given arms, dense."""
        result = np.zeros((len(arms), self.n_contexts))
        for row, i in enumerate(arms):
            p = self.problems[i]
            result[row, p.ctx_ids] = np.sqrt(np.maximum(p.context_second_moments(), 0.0))
        return result

    def reset(self) -> None:
        for p in self.problems:
            p.x = np.zeros(p.n_coordinates)
            p.x[p.reference] = p.pauli_target

    def solve_ics(
        self,
        arms: Sequence[int],
        epsilon: float = 1.0,
        lower: np.ndarray | None = None,
        *,
        max_iter: int = 60,
        rtol: float = 1e-6,
    ) -> JointSolution:
        """Alternate exact per-generator designs with the allocation (ICS).

        Kept as a baseline: it is monotone but stalls (see the module docstring).
        """
        arms = list(arms)
        shots = allocate(self.sigmas(arms), epsilon, lower)
        history = [float(shots.sum())]
        iterations = 0
        for iterations in range(1, max_iter + 1):
            for i in arms:
                self.problems[i].optimise(shots)
            shots = allocate(self.sigmas(arms), epsilon, lower)
            history.append(float(shots.sum()))
            if history[-2] - history[-1] <= rtol * history[-2]:
                break
        return JointSolution(shots, float(shots.sum()), iterations, history)

    def solve(
        self,
        arms: Sequence[int],
        epsilon: float = 1.0,
        lower: np.ndarray | None = None,
        *,
        max_outer: int = 80,
        inner_iterations: int = 300,
        dual_steps: int = 20,
        patience: int = 8,
        rtol: float = 1e-7,
    ) -> JointSolution:
        """Jointly optimal designs and shots for ``arms`` (saddle form, see above).

        The smoothing starts at ``delta = 0.3`` of the typical context norm and
        shrinks by 0.3 per outer iteration to ``1e-7`` of it.  Starting sharp
        instead leaves L-BFGS on an effectively non-smooth objective and it stalls
        (3% above the optimum on H4 side 1.0 HF); the continuation reaches the
        optimum of an independent smoothed solve to six digits.  Each outer
        iteration takes ``dual_steps`` multiplier steps at the new design: one
        step per iteration converges the multipliers of a many-arm problem only
        slowly (84 iterations and 167 s on all 26 H4 arms), twenty reach the same
        cost to 0.1% in about 4 s.  After the schedule ends the iteration continues
        until the exact cost has not improved by ``rtol`` for ``patience`` outer
        iterations.

        The designs of the arms are left at the best point found, measured by the
        exact allocation cost; the returned shots are that allocation.
        """
        arms = list(arms)
        stack = _Stack(self.problems, arms, self.n_contexts)
        committed = np.zeros(self.n_contexts) if lower is None else np.asarray(lower, float)
        target = epsilon ** 2

        def exact(y: np.ndarray) -> tuple[float, np.ndarray]:
            stack.set_designs(y)
            shots = allocate(self.sigmas(arms), epsilon, lower)
            return float(shots.sum()), shots

        y = np.zeros(stack.n_vars)
        best_total, best_shots = exact(y)
        best_y = y.copy()
        history = [best_total]
        if stack.n_vars == 0:
            return JointSolution(best_shots, best_total, 0, history)

        q = stack.fragment_moments(y)
        multipliers = stack.initial_multipliers(q, target)
        for _ in range(60):  # settle the multipliers on the starting design
            multipliers = stack.dual_step(q, multipliers, committed, target)

        stall = 0
        outer = 0
        for outer in range(1, max_outer + 1):
            relative_delta = max(DELTA_FLOOR, DELTA_START * DELTA_SHRINK ** (outer - 1))
            y = stack.inner(y, multipliers, committed, relative_delta, inner_iterations)
            q = stack.fragment_moments(y)
            for _ in range(dual_steps):
                multipliers = stack.dual_step(q, multipliers, committed, target)
            total, shots = exact(y)
            history.append(total)
            if total < best_total * (1 - rtol):
                best_total, best_shots, best_y = total, shots, y.copy()
                stall = 0
            elif relative_delta <= DELTA_FLOOR:
                stall += 1
                if stall >= patience:
                    break
        stack.set_designs(best_y)
        return JointSolution(best_shots, best_total, outer, history)

    def max_constraint_residual(self) -> float:
        return max((p.constraint_residual() for p in self.problems), default=0.0)

    def reconstruction(self, n_library: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``B`` (generators x fragments) and ``C`` (fragments x library Paulis).

        Fragment ``q`` is one generator's part in one context, a universal
        measurable operator ``M_q = sum_l C_ql P_l``; ``B`` selects each
        generator's fragments with unit weight, so ``BC`` must equal ``A``.
        """
        rows, cols, vals = [], [], []
        b_rows, b_cols = [], []
        q = 0
        for p in self.problems:
            fragment_of = q + p.coord_block
            rows.append(fragment_of)
            cols.append(p.coord_pauli)
            vals.append(p.x)
            b_rows.append(np.full(p.ctx_ids.size, p.generator))
            b_cols.append(q + np.arange(p.ctx_ids.size))
            q += p.ctx_ids.size
        C = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(q, n_library),
        )
        B = sp.csr_matrix(
            (np.ones(q), (np.concatenate(b_rows), np.concatenate(b_cols))),
            shape=(len(self.problems), q),
        )
        return B, C


class _Stack:
    """The free coordinates of several arms stacked into one vector ``y``.

    ``x = x_start + N y`` for every arm at once, with block-diagonal ``N`` and
    covariance ``B`` over the concatenated coordinates, so one objective
    evaluation is a handful of sparse products regardless of the number of arms.
    """

    def __init__(self, problems: Sequence[FragmentProblem], arms: Sequence[int], n_contexts: int):
        self.problems = [problems[i] for i in arms]
        self.n_contexts = n_contexts
        n_rows, n_cols, offsets_x, offsets_y = 0, 0, [], []
        rows, cols, vals = [], [], []
        frag_starts, frag_ctx, frag_arm = [], [], []
        coord_arm, coord_ctx = [], []
        for a, p in enumerate(self.problems):
            structure = p._structure(np.ones(p.ctx_ids.size, dtype=bool))
            offsets_x.append(n_rows)
            offsets_y.append(n_cols)
            if structure is not None:
                N = structure["N"].tocoo()
                rows.append(N.row + n_rows)
                cols.append(N.col + n_cols)
                vals.append(N.data)
                n_cols += structure["n_vars"]
            frag_starts.append(p.ctx_ptr[:-1] + n_rows)
            frag_ctx.append(p.ctx_ids)
            frag_arm.append(np.full(p.ctx_ids.size, a))
            coord_arm.append(np.full(p.n_coordinates, a))
            coord_ctx.append(p.ctx_ids[p.coord_block])
            n_rows += p.n_coordinates
        offsets_x.append(n_rows)
        offsets_y.append(n_cols)
        self.offsets_x = np.array(offsets_x)
        self.offsets_y = np.array(offsets_y)
        self.n_vars = n_cols
        self.N = sp.csr_matrix(
            (np.concatenate(vals) if vals else np.zeros(0),
             (np.concatenate(rows) if rows else np.zeros(0, int),
              np.concatenate(cols) if cols else np.zeros(0, int))),
            shape=(n_rows, n_cols),
        )
        self.NT = self.N.T.tocsr()
        self.B = sp.block_diag([p._block_matrix for p in self.problems], format="csr")
        self.x_start = np.concatenate([p.x for p in self.problems])
        self.frag_starts = np.concatenate(frag_starts)
        self.frag_ctx = np.concatenate(frag_ctx)
        self.frag_arm = np.concatenate(frag_arm)
        self.coord_arm = np.concatenate(coord_arm)
        self.coord_ctx = np.concatenate(coord_ctx)

    def designs(self, y: np.ndarray) -> np.ndarray:
        return self.x_start + self.N @ y

    def set_designs(self, y: np.ndarray) -> None:
        x = self.designs(y)
        for a, p in enumerate(self.problems):
            p.x = x[self.offsets_x[a] : self.offsets_x[a + 1]].copy()

    def fragment_moments(self, y: np.ndarray) -> np.ndarray:
        """``q[f] = x^T Sigma x`` for every (arm, context) fragment."""
        x = self.designs(y)
        return np.maximum(np.add.reduceat(x * (self.B @ x), self.frag_starts), 0.0)

    def _context_load(self, q: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.frag_ctx, weights=multipliers[self.frag_arm] * q, minlength=self.n_contexts
        )

    def initial_multipliers(self, q: np.ndarray, target: float) -> np.ndarray:
        per_arm = np.bincount(self.frag_arm, weights=np.sqrt(q), minlength=len(self.problems))
        return np.maximum(per_arm ** 2, 1e-300) / target ** 2 / len(self.problems)

    def dual_step(
        self, q: np.ndarray, multipliers: np.ndarray, committed: np.ndarray, target: float
    ) -> np.ndarray:
        shots = np.maximum(committed, np.sqrt(self._context_load(q, multipliers)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(shots[self.frag_ctx] > 0, q / shots[self.frag_ctx], 0.0)
        variances = np.bincount(self.frag_arm, weights=ratio, minlength=len(self.problems))
        factor = np.where(variances > 0, variances / target, 1.0)
        return np.clip(multipliers * np.sqrt(factor), 1e-300, 1e300)

    def inner(
        self,
        y: np.ndarray,
        multipliers: np.ndarray,
        committed: np.ndarray,
        relative_delta: float,
        max_iterations: int,
    ) -> np.ndarray:
        """Minimise ``sum_alpha h(Q_alpha + delta**2)`` over the stacked coefficients."""
        import scipy.optimize

        load = self._context_load(self.fragment_moments(y), multipliers)
        scale_q = float(load[load > 0].mean()) if (load > 0).any() else 1.0
        delta_sq = (relative_delta ** 2) * scale_q
        weights = multipliers[self.coord_arm]
        reference = None

        def objective(v: np.ndarray):
            nonlocal reference
            x = self.designs(v)
            Bx = self.B @ x
            q = np.maximum(np.add.reduceat(x * Bx, self.frag_starts), 0.0)
            Q = self._context_load(q, multipliers) + delta_sq
            root = np.sqrt(Q)
            n = np.maximum(committed, root)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = np.where(root >= committed, 2.0 * root, committed + Q / np.where(committed > 0, committed, 1.0))
            value = float(h.sum())
            grad_x = 2.0 * weights * Bx / n[self.coord_ctx]
            gradient = self.NT @ grad_x
            if reference is None:
                reference = max(value, 1e-300)
            return value / reference, gradient / reference

        result = scipy.optimize.minimize(
            objective,
            y,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "maxcor": 10, "ftol": 1e-15, "gtol": 1e-12},
        )
        return result.x


def reconstruction_residual(design: DesignSet, A: sp.csr_matrix) -> float:
    """``max |A - BC|``; a design that fails ``TARGET_TOLERANCE * max|A|`` is invalid."""
    B, C = design.reconstruction(A.shape[1])
    difference = (A - B @ C).tocsr()
    return float(abs(difference).max()) if difference.nnz else 0.0


def assert_exact(design: DesignSet, A: sp.csr_matrix) -> float:
    residual = reconstruction_residual(design, A)
    scale = float(abs(A).max())
    if residual > TARGET_TOLERANCE * scale:
        raise AssertionError(f"A = BC violated: residual {residual:.3e} (scale {scale:.3e})")
    return residual
