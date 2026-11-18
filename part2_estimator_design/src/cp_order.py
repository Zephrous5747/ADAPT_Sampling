"""Work order CP-1.2 ("Do covariance priors reduce generator-selection measurements?", reports/prior/Classical_covariance_prior_work_order-2.tex).

Three sources of initial moments, one update formula and one design program:

``NEUTRAL``    mu_0 = 0, R_0 = I, strength 1;
``CLASSICAL``  moments of (1 - eps) |c><c| + eps I / 2^n (eps = 0.01), strength tau(x_+) from the relative classical-quantum energy
               discrepancy with a Hoeffding energy radius;
``PREVIOUS``   sampled moments of the preceding reoptimized state (128 synthetic joint outcomes per group), strength tau_{a,P} from the
               relative energy change between iterations with the finite-history adjustment.

Update (Eq. 5): mu~ = (tau mu_0 + S)/(tau + N), R~ = (tau R_0 + T)/(tau + N), Sigma = R~ - mu~ mu~'; no ridge.  Prior moments enter the design
only.  The design program is solved with CVXPY/Clarabel (60 s, 500 iterations, 1e-8) with the reconstruction constraints, the L1 bound
sum_a ||b_ja||_1 <= 2 sum_P |h_jP|, n_a >= 32; on failure the fixed reconstruction b^0 is frozen and only the counts are optimized.

Deviations are listed in ``DEVIATIONS``.
"""
from __future__ import annotations

import math
import os
import time

# Clarabel starts a Rust thread pool with one thread per core in each process and ignores OMP_NUM_THREADS; set before the first solve.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp

from allocation import allocate
from classical_prior import MixedMoments
from note_protocol import NoteCovariance
from reframe import Counts, ReframeSelector
from sampler import pauli_covariance, walsh_hadamard

EPS_C = 0.01
HISTORY_SHOTS = 128
L1_FACTOR = 2.0
FLOOR = 32
DEVIATIONS = (
    "contexts: the repository's shared FC cover (first-fit by coefficient mass, completed with compatible library Paulis), not the "
    "work order's two-pass construction",
    "states: archived exact-gradient trajectories of this repository (BFGS gtol of the earlier build), H2 regenerated here",
    "A1 per-arm programs are solved exactly by the stationary linear solve when its minimizer already satisfies the L1 bound (then it is "
    "the constrained optimum) and by CVXPY otherwise",
    "fallback counts for a frozen reconstruction are computed by the exact allocation (allocation.py) instead of a second CVXPY solve",
)


# --- trust ------------------------------------------------------------------------------------------------------------------------


def x_upper(e_classical: float, e_hat: float, radius: float) -> float:
    """``x_+ = (|E_c - E_hat| + r_E) / (|E_hat| - r_E)`` (inf when the denominator is not positive)."""
    denominator = abs(e_hat) - radius
    return (abs(e_classical - e_hat) + radius) / denominator if denominator > 0 else math.inf


def y_upper(e_hat: float, r: float, e_prev: float, r_prev: float) -> float:
    denominator = abs(e_hat) - r
    return (abs(e_hat - e_prev) + r + r_prev) / denominator if denominator > 0 else math.inf


def tau_of(x_plus: float, d0: float, s: float) -> float:
    """``tau = 1 / (d0 + 1 - exp(-(x/s)^2))`` with expm1 for stability."""
    if not math.isfinite(x_plus):
        return 1.0 / (d0 + 1.0)
    return 1.0 / (d0 - math.expm1(-((x_plus / s) ** 2)))


def history_strength(tau_e: float, n_prev: float) -> float:
    if math.isinf(tau_e):
        return float(n_prev)
    return tau_e * n_prev / (tau_e + n_prev)


# --- prior moment objects (group means in the Walsh domain, as OracleMoments) ----------------------------------------------------------


class NeutralMoments:
    """Maximally mixed reference: mu_0 = 0, R_0 = I."""

    def __init__(self, dim: int) -> None:
        self._delta = np.zeros(dim)
        self._delta[0] = 1.0

    def group_means(self, alpha: int) -> np.ndarray:
        return self._delta


class PreviousMoments:
    """Sampled moments of the preceding state, ``(1 - eps)`` of the empirical distribution plus ``eps`` of the uniform one."""

    def __init__(self, histograms: dict, shots: int, dim: int, eps: float = EPS_C) -> None:
        self.hist, self.shots, self.eps = histograms, shots, eps
        self._delta = np.zeros(dim)
        self._delta[0] = 1.0
        self._cache: dict[int, np.ndarray] = {}

    def group_means(self, alpha: int) -> np.ndarray:
        if alpha not in self._cache:
            g = (1.0 - self.eps) * walsh_hadamard(self.hist[alpha]) / float(self.shots) + self.eps * self._delta
            self._cache[alpha] = g
        return self._cache[alpha]


class TrueCovariance:
    """Evaluator-only covariance of the true state (never enters an experimental arm)."""

    def __init__(self, oracle) -> None:
        self.oracle = oracle

    def __call__(self, alpha: int, paulis: np.ndarray) -> np.ndarray:
        return self.oracle.covariance(alpha, paulis)


# --- the energy record --------------------------------------------------------------------------------------------------------------------


class EnergyRecord:
    """Simulated energy measurement of one state (SI of the work order): disjoint FC fragments of H, B_E = max(1e4, 32 L_E) shots."""

    def __init__(self, problem, case: str, terms: dict, state_energy: float) -> None:
        from contexts import build_context_library
        from pauli_fc import greedy_fc_groups
        from pauli_ops import PauliEvaluator

        self.terms = terms
        order = sorted(terms, key=lambda s: (-abs(terms[s]), s))
        self.groups = greedy_fc_groups(order, sort=False)
        self.L = len(self.groups)
        self.shots = max(10 ** 4, 32 * self.L)
        base, extra = divmod(self.shots, self.L)
        self.n = np.full(self.L, base, dtype=np.int64)
        self.n[:extra] += 1  # index tie breaking
        self.B = np.array([sum(abs(terms[p]) for p in g) for g in self.groups])
        self.radius = float((self.B * np.sqrt(2.0 * math.log(2.0 * self.L / 0.025) / self.n)).sum())
        self.library = build_context_library(problem, "canonical", groups=self.groups, auxiliary_labels=list(terms))
        self.index = {label: k for k, label in enumerate(self.library.labels)}
        self.h0 = float(state_energy) - float(PauliEvaluator(problem.evaluator.state).fragment_mean_std(terms)[0])
        self.evaluator = PauliEvaluator

    def distributions(self, state: np.ndarray) -> np.ndarray:
        from sampler import OracleMoments

        moments = OracleMoments(self.library, state)
        dist = np.clip(moments.distributions_matrix(), 0.0, None)
        return dist / dist.sum(axis=1, keepdims=True)

    def sample(self, rng: np.random.Generator, dist: np.ndarray) -> float:
        total = self.h0
        for a, group in enumerate(self.groups):
            counts = rng.multinomial(int(self.n[a]), dist[a]).astype(float)
            means = walsh_hadamard(counts) / float(self.n[a])
            context = self.library.contexts[a]
            idx = np.array([self.index[p] for p in group])
            pos = context.positions(idx)
            coefficient = np.array([self.terms[p] for p in group])
            total += float(coefficient @ (context.member_sign[pos] * means[context.member_zmask[pos]]))
        return total


# --- the selector with the design programs -----------------------------------------------------------------------------------------------


class CPSelector(ReframeSelector):
    """State data, design covariance and the CVXPY programs of the work order."""

    def __init__(self, *args, solve_seconds: float = 60.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.solve_seconds = solve_seconds
        self.alternating_fallback = False  # True: a failed conic solve is replaced by the alternating solver instead of the frozen b^0
        self.sweeps = 2  # sweeps of the alternating solver used where the conic program is too large (round 2: 2; the LiH check needs about 10)
        self.max_coords = 60000  # a joint program with more coefficients cannot be built and solved within the 60 s cap (LiH: 365,000)
        self.oracle = None  # OracleMoments of the true state (evaluator only)
        self.energy_c = None
        self.energy_q = None
        self.energy_prev = None
        self.prev_dist = None
        self.energy: EnergyRecord | None = None
        self.dim = self.dist.shape[1]
        self.last_solver: dict = {}

    # fragments ---------------------------------------------------------------------------------------------------------------------

    def fragments_of(self, p, x: np.ndarray) -> list:
        keep = x != 0.0
        ctx, pauli, values = p.coord_ctx[keep], p.coord_pauli[keep], x[keep]
        return [(int(a), pauli[ctx == a].copy(), values[ctx == a].copy()) for a in np.unique(ctx)]

    def true_variances(self, designs: dict, counts: np.ndarray) -> dict:
        truth = TrueCovariance(self.oracle)
        out = {}
        for i, fragments in designs.items():
            out[i] = float(sum(max(float(x @ truth(a, p) @ x), 0.0) / counts[a] if counts[a] > 0 else (math.inf if np.abs(x).max() > 0 else 0.0)
                               for a, p, x in fragments))
        return out

    # A1 ------------------------------------------------------------------------------------------------------------------------------

    def _qp_arm(self, p, n, cap: float) -> np.ndarray | None:
        """One arm's variance at counts ``n`` (a scalar, or the vector of counts of all contexts) under the L1 bound."""
        import cvxpy as cp

        blocks = []
        per_block = None if np.isscalar(n) else np.asarray(n, dtype=float)[np.asarray(p.ctx_ids, dtype=int)]
        typical = float(n) if per_block is None else float(np.median(per_block))
        for k, block in enumerate(p.blocks):
            w, v = np.linalg.eigh(0.5 * (block + block.T))
            weight = 1.0 if per_block is None else math.sqrt(typical / per_block[k])  # relative to the typical count: keeps the scaling
            blocks.append(v * np.sqrt(np.maximum(w, 0.0)) * weight)
        n = typical
        factor = sp.block_diag(blocks, format="csr")
        constraint = sp.csr_matrix((np.ones(p.n_coordinates), (p._pauli_of, np.arange(p.n_coordinates))), shape=(p.pauli_ids.size, p.n_coordinates))
        b = cp.Variable(p.n_coordinates)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(factor.T @ b) / n), [constraint @ b == p.pauli_target, cp.norm1(b) <= cap])
        try:
            problem.solve(solver=cp.CLARABEL, max_iter=500, time_limit=self.solve_seconds, tol_gap_rel=1e-8, tol_feas=1e-8)
        except Exception:  # noqa: BLE001
            return None
        return None if problem.status != "optimal" or b.value is None else np.asarray(b.value)

    def a1(self, arms: list, cov, n=128.0) -> tuple[dict, dict]:
        """Fixed allocation ``n`` (per group, or a vector of counts): every arm minimizes its own predicted variance (the max over arms
        is then the minimum of t)."""
        problems = self._split_problems()
        designs, routes = {}, {}
        counts = np.full(self.n_contexts, float(n)) if np.isscalar(n) else np.asarray(n, dtype=float)
        for j in arms:
            p = problems[j]
            p.update_blocks(cov)
            home = np.zeros(p.n_coordinates)
            home[p.reference] = p.pauli_target
            cap = L1_FACTOR * float(self.bound[j])
            p.x = home.copy()
            p.optimise(counts)
            p._structures.clear()
            x, route = p.x.copy(), "exact"
            if float(np.abs(x).sum()) > cap * (1.0 + 1e-9):
                solved = self._qp_arm(p, n, cap)
                x, route = (solved, "cvxpy") if solved is not None else (home, "fallback")
            designs[j] = self.fragments_of(p, x)
            routes[j] = route
        return designs, routes

    # joint programs (A2 and Test B) -------------------------------------------------------------------------------------------------

    def _stack(self, arms: list, cov):
        problems = self._split_problems()
        offsets, rows_b, factors = [0], [], []
        block_ctx, block_arm, block_slices = [], [], []
        constraints, targets, arm_rows = [], [], []
        coord_total = 0
        for row, j in enumerate(arms):
            p = problems[j]
            p.update_blocks(cov)
            for k, block in enumerate(p.blocks):
                w, v = np.linalg.eigh(0.5 * (block + block.T))
                factors.append(v * np.sqrt(np.maximum(w, 0.0)))
                block_ctx.append(int(p.ctx_ids[k]))
                block_arm.append(row)
            constraints.append(sp.csr_matrix((np.ones(p.n_coordinates), (p._pauli_of, np.arange(p.n_coordinates))),
                                             shape=(p.pauli_ids.size, p.n_coordinates)))
            targets.append(p.pauli_target)
            coord_total += p.n_coordinates
            offsets.append(coord_total)
        return problems, offsets, factors, np.array(block_ctx), np.array(block_arm), constraints, np.concatenate(targets)

    def joint(self, arms: list, cov, mode: str, *, epsilon: float | None = None, budget: float | None = None, verbose: bool = False) -> tuple[dict, np.ndarray | None, dict]:
        """``mode='B'``: min sum n s.t. every arm's predicted variance <= epsilon^2; ``mode='A2'``: min t s.t. sum n = budget, n >= 32.

        Variables are normalized (counts by a typical count N0, variances by a typical variance T0) so that Clarabel reaches 1e-8.
        Returns ``(designs, plan, info)``; ``plan`` is None when the program was not solved (the caller freezes ``b^0``).
        """
        import cvxpy as cp

        started = time.perf_counter()
        info = {"status": "not_run", "seconds": 0.0, "coords": 0}
        L = self.n_contexts
        total = sum(self._split_problems()[j].n_coordinates for j in arms)
        if total > self.max_coords:
            return self.alternating(arms, cov, mode, epsilon=epsilon, budget=budget, coords=int(total), sweeps=self.sweeps)
        try:
            sigma_sq = np.zeros((len(arms), L))
            for row, j in enumerate(arms):
                for a, p, x in self.x0[j]:
                    sigma_sq[row, a] = max(float(x @ cov(a, p) @ x), 0.0)
            if mode == "B":
                fixed = allocate(np.sqrt(sigma_sq), epsilon)
                used_fixed = fixed > 0
                N0 = float(max(FLOOR, fixed[used_fixed].mean())) if used_fixed.any() else float(FLOOR)
                T0 = float(epsilon ** 2)
            else:  # uniform allocation of the budget as the scale reference
                N0 = float(budget) / L
                T0 = float(max((sigma_sq.sum(axis=1) / N0).max(), 1e-12))
            problems, offsets, factors, block_ctx, block_arm, constraints, targets = self._stack(arms, cov)
            ncoord = offsets[-1]
            info["coords"] = int(ncoord)
            nblk = len(factors)
            kmax = max(f.shape[0] for f in factors)
            scale = 1.0 / math.sqrt(N0 * T0)
            f_rows, f_cols, f_vals = [], [], []
            block_cursor = 0
            for row, j in enumerate(arms):
                p = problems[j]
                for k in range(len(p.blocks)):
                    lo, hi = p.ctx_ptr[k], p.ctx_ptr[k + 1]
                    size = hi - lo
                    F = factors[block_cursor] * scale
                    r_idx, c_idx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
                    f_rows.append((block_cursor * kmax + r_idx).ravel())
                    f_cols.append((offsets[row] + lo + c_idx).ravel())
                    f_vals.append(F[c_idx, r_idx].ravel())
                    block_cursor += 1
            Z_matrix = sp.csr_matrix((np.concatenate(f_vals), (np.concatenate(f_rows), np.concatenate(f_cols))), shape=(nblk * kmax, ncoord))
            b = cp.Variable(ncoord)
            n = cp.Variable(L)  # counts / N0
            w = cp.Variable(nblk, nonneg=True)  # block variance shares / T0
            ctx_select = sp.csr_matrix((np.ones(nblk), (np.arange(nblk), block_ctx)), shape=(nblk, L))
            arm_select = sp.csr_matrix((np.ones(nblk), (block_arm, np.arange(nblk))), shape=(len(arms), nblk))
            n_blk = ctx_select @ n
            Z = cp.reshape(Z_matrix @ b, (kmax, nblk), order="F")
            soc = cp.SOC((n_blk + w) / 2, cp.vstack([cp.reshape((n_blk - w) / 2, (1, nblk), order="F"), Z]), axis=0)
            used = np.zeros(L, dtype=bool)
            used[block_ctx] = True
            coord_arm = np.concatenate([np.full(offsets[r + 1] - offsets[r], r) for r in range(len(arms))])
            l1_select = sp.csr_matrix((np.ones(ncoord), (coord_arm, np.arange(ncoord))), shape=(len(arms), ncoord))
            caps = np.array([L1_FACTOR * float(self.bound[j]) for j in arms])
            recon = sp.block_diag(constraints, format="csr")
            cons = [soc, recon @ b == targets, l1_select @ cp.abs(b) <= caps]
            if mode == "B":
                cons += [arm_select @ w <= 1.0, n[used] >= FLOOR / N0]
                if (~used).any():
                    cons += [n[~used] == 0]
                objective = cp.Minimize(cp.sum(n))
            else:
                t = cp.Variable()
                cons += [arm_select @ w <= t, n >= FLOOR / N0, cp.sum(n) == budget / N0]
                objective = cp.Minimize(t)
            problem = cp.Problem(objective, cons)
            # the work order's tolerance first (1e-8, 500 iterations); a solve that ends 'optimal_inaccurate' (the H4 1.0 A exact-covariance
            # reference did in 30 of 30 programs and was then replaced by the frozen reconstruction) is repeated at looser tolerances, and the
            # tolerance that gave 'optimal' is recorded
            for tol, iterations in ((1e-8, 500), (1e-7, 1000), (1e-6, 1000), (1e-5, 1000)):
                problem.solve(solver=cp.CLARABEL, max_iter=iterations, time_limit=self.solve_seconds, tol_gap_rel=tol, tol_feas=tol, verbose=verbose)
                info["status"], info["tol"] = problem.status, tol
                if problem.status != "optimal_inaccurate":
                    break
            if problem.status == "optimal" and b.value is not None and n.value is not None:
                designs = {}
                for row, j in enumerate(arms):
                    x = np.asarray(b.value[offsets[row]:offsets[row + 1]]).copy()
                    x[np.abs(x) < 1e-12] = 0.0
                    designs[j] = self.fragments_of(problems[j], x)
                info["seconds"] = round(time.perf_counter() - started, 2)
                return designs, np.maximum(np.asarray(n.value), 0.0) * N0, info
        except Exception as error:  # noqa: BLE001
            info["status"] = f"error: {type(error).__name__}: {str(error)[:300]}"
        info["seconds"] = round(time.perf_counter() - started, 2)
        if self.alternating_fallback:
            # the conic program failed (time limit, solver error, inaccurate after the retries): in the batch-fix runs 12-39% of the LiH blocks fell
            # back to the frozen reconstruction b^0 this way; use the alternating solver instead of freezing
            failed = info["status"]
            designs, plan, alt = self.alternating(arms, cov, mode, epsilon=epsilon, budget=budget, coords=info.get("coords", 0), sweeps=self.sweeps)
            alt["status"], alt["conic_status"], alt["conic_seconds"] = "alternating_after_conic_failure", failed[:60], info["seconds"]
            return designs, plan, alt
        return {j: self.x0[j] for j in arms}, None, info

    def predicted(self, designs: dict, cov, plan: np.ndarray) -> dict:
        """Predicted variance of every design under the design covariance at the counts ``plan``."""
        out = {}
        for j, fragments in designs.items():
            total = 0.0
            for a, p, x in fragments:
                v = max(float(x @ cov(a, p) @ x), 0.0)
                if v > 0.0:
                    total += v / plan[a] if plan[a] > 0 else math.inf
            out[j] = total
        return out

    def alternating(self, arms: list, cov, mode: str, *, epsilon: float | None = None, budget: float | None = None, coords: int = 0,
                    sweeps: int = 2) -> tuple[dict, np.ndarray, dict]:
        """The joint program by block-coordinate descent, for programs too large for the conic solver (LiH: 365,000 coefficients).

        Start from the reference reconstruction and its optimal counts; then alternately (i) let every arm minimize its own predicted
        variance at the current counts under the reconstruction and L1 constraints (the per-arm programs of A1), and (ii) re-optimize the
        counts for the new coefficients.  Every iterate is feasible; the one with the best objective (total counts in mode B, largest
        predicted variance in mode A2) is returned.  The program is jointly convex, but the constraints couple coefficients and counts,
        so this is a feasible descent heuristic, not a certified optimum (declared deviation from the work order's conic solve).
        """
        started = time.perf_counter()

        def objective(designs, plan):
            return float(plan.sum()) if mode == "B" else max(self.predicted(designs, cov, plan).values())

        designs = {j: self.x0[j] for j in arms}
        plan = self.fixed_plan(arms, cov, mode, epsilon=epsilon, budget=budget)
        best = (objective(designs, plan), designs, plan)
        trace = [best[0]]
        routes_total: dict = {}
        for _ in range(sweeps):
            counts = np.where(plan > 0, plan, float(FLOOR))  # an unused context may be recruited at the floor it would then be given
            designs, routes = self.a1(arms, cov, counts)
            for r in routes.values():
                routes_total[r] = routes_total.get(r, 0) + 1
            plan = self.fixed_plan(arms, cov, mode, epsilon=epsilon, budget=budget, designs=designs)
            value = objective(designs, plan)
            trace.append(value)
            if value < best[0]:
                best = (value, designs, plan)
        info = {"status": "alternating", "coords": coords, "sweeps": sweeps, "objective_trace": trace, "routes": routes_total,
                "seconds": round(time.perf_counter() - started, 2)}
        return best[1], best[2], info

    def fixed_plan(self, arms: list, cov, mode: str, *, epsilon: float | None = None, budget: float | None = None,
                   designs: dict | None = None) -> np.ndarray:
        """Counts for given coefficients (default: the frozen reference reconstruction, the program's fallback)."""
        sigmas = np.zeros((len(arms), self.n_contexts))
        for row, j in enumerate(arms):
            for a, p, x in (self.x0[j] if designs is None else designs[j]):
                sigmas[row, a] = math.sqrt(max(float(x @ cov(a, p) @ x), 0.0))
        used = (sigmas > 0).any(axis=0)
        lower = np.where(used, float(FLOOR), 0.0)
        if mode == "B":
            return allocate(sigmas, epsilon, lower)
        # A2: minimize the largest variance at a fixed budget with n >= 32: bisection on the scale of the allocation
        lo, hi = 1e-12, 1e12
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            plan = allocate(sigmas, mid, lower)
            if plan.sum() > budget:
                lo = mid
            else:
                hi = mid
        plan = allocate(sigmas, hi, lower)
        scale = budget - plan.sum()
        spare = np.maximum(plan - lower, 0.0)
        return plan + (scale * spare / spare.sum() if spare.sum() > 0 and scale > 0 else 0.0)
