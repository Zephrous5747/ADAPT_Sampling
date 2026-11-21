"""Generator selection exactly as specified by the reframed protocol GS-1.0.

Source: ``reports/reframed/generator_selection_paper`` (manuscript Secs. "Optimize shots for the next decision" and
"Confidence under adaptation", SI Secs. S3-S6).  Methods:

``FC_FULL``    fixed shared reconstruction (every Pauli read from its parent context), every non-constant candidate is
               requested in every round, no elimination;
``FC_SE``      the same measurement library and reconstruction with successive elimination;
``FC_SE_CS``   the same library and elimination with coefficient splitting and the growing classical covariance prior;
``FC_SE_CS_DATA``  *diagnostic, not in the protocol*: as ``FC_SE_CS`` but the design covariance is the measured one (plus the
               same regularizer), so that the effect of the prior is separated from the effect of splitting;
``FC_SE_ORACLE``, ``FC_SE_CS_ORACLE``  *bounds, not methods* (JCTC round 5): ``FC_SE`` and ``FC_SE_CS_DATA`` with the design covariance
               replaced by the exact covariance of the true state (plus the same regularizer) in every block, from the first one.  Same
               intervals, certificate, floors and checkpoints, so that the difference to the data-only arms is the cost of learning the
               covariance from outcomes.  The exact moments never enter an interval.

What is implemented, and where the code deviates from the text (``DEVIATIONS``):

* design covariance  ``(32 Sigma^c + N Sigma_hat)/(32 + N) + 1e-3 I`` (complete method) or ``Sigma_hat + 1e-3 I`` /
  ``I`` before data (baselines); the covariance of the true state never enters a design;
* the fresh-block program ``min sum n  s.t.  sum_a x'Sigma x / n <= eps_r^2``; positive counts are rounded up with floor 32;
  the rounded optimized plan is compared with the rounded fixed-reconstruction plan and the cheaper is used (ties: fixed);
* blocks with base counts ``q`` (halved until the total is below 4096 or every context is at its floor) and checkpoints at
  ``q 2^(l-1)`` shots;
* block intervals by the bounded-variance (Maurer-Pontil) construction with Bernstein radii, the fixed-reconstruction
  historical bound, intersection of all valid intervals, empty intersection = ``interval_conflict``;
* elimination, the relative selection certificate, the 30-round schedule and the shot cap.
"""
from __future__ import annotations

import math
import time

import numpy as np

from allocation import allocate
from design import DesignSet, FragmentProblem
from sampler import pauli_covariance, walsh_hadamard

NU = 32.0
LAMBDA = 1e-3
FLOOR = 32
FIRST_CHECK = 4096
EPS_BASE = 1e-3
T_MULT = 8.0
MAX_ROUNDS = 30
CONFLICT_TOLERANCE = 1e-12
TIE_TOLERANCE = 1e-10
METHODS = ("FC_FULL", "FC_SE", "FC_SE_CS", "FC_SE_CS_DATA", "FC_SE_ORACLE", "FC_SE_CS_ORACLE")
ORACLE_METHODS = ("FC_SE_ORACLE", "FC_SE_CS_ORACLE")  # JCTC round 5: designs with the exact covariance of the true state (a bound, not a method)

DEVIATIONS = (
    "joint design program solved by the saddle-form solver of design.py (validated against independent solvers in "
    "tests/test_design.py) instead of CVXPY/CLARABEL, which are not installed; the plan is checked against the "
    "constraints to 1e-7 after solving",
    "FC_FULL at rho = 0: the literal certificate L_k >= max_S U_i can never hold (the maximum contains U_k >= L_k), so the "
    "certificate is 'every other candidate is strictly dominated', i.e. a sole virtual survivor",
    "contexts are the older first-fit weight-ordered FC library of this repository (strategy mass, grouping weight)",
)


def epsilon_round(r: int) -> float:
    return (5.0 - 0.4 * r) * EPS_BASE if r <= 10 else EPS_BASE * 2.0 ** (-(r - 10))


def rounded_total(plan: np.ndarray, floor: int = FLOOR) -> int:
    positive = plan > 1e-9
    return int(np.maximum(floor, np.ceil(plan[positive] - 1e-9)).sum())


class Counts:
    """Outcome histograms per context (sparse) with cached empirical group means."""

    def __init__(self, n_contexts: int) -> None:
        self.hist: dict[int, np.ndarray] = {}
        self.n = np.zeros(n_contexts, dtype=np.int64)
        self._means: dict[int, np.ndarray] = {}

    def add(self, contexts: np.ndarray, counts: np.ndarray) -> None:
        for a, c in zip(contexts.tolist(), counts):
            if a in self.hist:
                self.hist[a] += c
            else:
                self.hist[a] = c.astype(np.float64)
            self.n[a] += int(c.sum())
            self._means.pop(a, None)

    def group_means(self, a: int) -> np.ndarray:
        if a not in self._means:
            self._means[a] = walsh_hadamard(self.hist[a]) / float(self.n[a])
        return self._means[a]


class DesignCovariance:
    """Sigma-tilde of one context for the design and the plan (never the covariance of the true state)."""

    def __init__(self, library, counts: Counts, prior, nu: float = NU, lam: float = LAMBDA, exact: bool = False) -> None:
        self.library = library
        self.counts = counts
        self.prior = prior  # OracleMoments of the classical state, or None: measured covariance only
        self.nu = nu
        self.lam = lam
        self.exact = exact  # ``prior`` holds the moments of the true state and is used alone (the oracle bound)

    def __call__(self, alpha: int, paulis: np.ndarray) -> np.ndarray:
        k = len(paulis)
        if self.exact:
            sigma = self.prior.covariance(alpha, paulis)
            return 0.5 * (sigma + sigma.T) + self.lam * np.eye(k)
        n = int(self.counts.n[alpha])
        context = self.library.contexts[alpha]
        if self.prior is None and n < 2:
            return np.eye(k)
        if n >= 2:
            positions = context.positions(paulis)
            zmask, sign = context.member_zmask[positions], context.member_sign[positions]
            empirical = pauli_covariance(self.counts.group_means(alpha), zmask, sign) * n / (n - 1.0)
        if self.prior is None:
            sigma = empirical
        elif n < 2:
            sigma = self.prior.covariance(alpha, paulis)
        else:
            sigma = (self.nu * self.prior.covariance(alpha, paulis) + n * empirical) / (self.nu + n)
        sigma = 0.5 * (sigma + sigma.T)
        return sigma + self.lam * np.eye(k)


def fragment_moments(counts: Counts, library, by_context: dict) -> dict:
    """``(arm, context) -> (n, mean, s2, l1)`` of the fragments ``x'p`` from outcome histograms.

    ``s2`` is the unbiased sample variance of the fragment (``x' S x`` with the unbiased covariance S), ``l1 = ||x||_1``.
    """
    out = {}
    for a, fragments in by_context.items():
        n = int(counts.n[a])
        if n <= 0:
            continue
        union = np.unique(np.concatenate([p for _, p, _ in fragments]))
        context = library.contexts[a]
        positions = context.positions(union)
        zmask, sign = context.member_zmask[positions], context.member_sign[positions]
        group = counts.group_means(a)
        means = sign * group[zmask]
        covariance = pauli_covariance(group, zmask, sign) * n / (n - 1.0) if n >= 2 else None
        for arm, p, x in fragments:
            index = np.searchsorted(union, p)
            s2 = float(x @ covariance[np.ix_(index, index)] @ x) if covariance is not None else math.nan
            out[(arm, a)] = (n, float(x @ means[index]), max(s2, 0.0) if covariance is not None else math.nan,
                             float(np.abs(x).sum()))
    return out


def bernstein_interval(n_list, mean_list, s2_list, l1_list, eta: float, h: int, a0: float, residual: float):
    """Fresh-block interval of one gradient (SI S5): fragments with ``n >= 32`` shots each."""
    total = a0
    variance = 0.0
    scale = 0.0
    log_h = math.log(2.0 * h / eta)
    for n, m, s2, l1 in zip(n_list, mean_list, s2_list, l1_list):
        bound = 2.0 * l1
        spread = math.sqrt(s2) + bound * math.sqrt(2.0 * log_h / (n - 1.0))
        variance += min(bound * bound / 4.0, spread * spread) / n
        scale = max(scale, bound / n)
        total += m
    t = math.log(4.0 / eta)
    radius = math.sqrt(2.0 * variance * t) + (2.0 / 3.0) * scale * t + residual
    return total - radius, total + radius


def history_fragment(n: int, m: float, s2: float, l1: float, zeta_base: float) -> tuple[float, float]:
    """Historical fragment interval for the fixed reference reconstruction (SI S5), intersected with the range."""
    if n < 2:
        return -l1, l1
    zeta = zeta_base / (n * (n - 1.0))
    bound = 2.0 * l1
    spread = math.sqrt(s2) + bound * math.sqrt(2.0 * math.log(2.0 / zeta) / (n - 1.0))
    v = min(bound * bound / 4.0, spread * spread)
    log_term = math.log(4.0 / zeta)
    radius = math.sqrt(2.0 * v * log_term / n) + 2.0 * bound * log_term / (3.0 * n)
    return max(m - radius, -l1), min(m + radius, l1)


def magnitude_interval(lo: float, hi: float) -> tuple[float, float]:
    upper = max(abs(lo), abs(hi))
    lower = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
    return lower, upper


class ReframeSelector:
    """One fixed state, one context library, many trials (see the module docstring)."""

    def __init__(self, problem, library, distributions: np.ndarray, base_problems, split_coords, prior, *,
                 cap: float = 1e10, delta_share: float = 0.5) -> None:
        self.problem = problem
        self.library = library
        self.n_contexts = library.n_contexts
        self.dist = distributions
        self.prior = prior
        self.truth_moments = None  # OracleMoments of the true state, set by the runner for the *_ORACLE methods only
        self.floor = FLOOR  # shots per active context below which a plan is rounded up; run(floor=...) sets it for the trial (JCTC: floor sensitivity of the prior)
        self.cap = float(cap)
        self.truth = np.asarray(problem.gradients, dtype=float)
        self.split_coords = split_coords
        n_arms = len(split_coords)
        self.n_arms = n_arms
        # fixed reference reconstruction: every Pauli read from its parent context
        self.x0: list[list] = []
        self.targets: list[dict] = []
        self.bound = np.zeros(n_arms)
        for i, p in enumerate(base_problems):
            x = p.x.copy()
            fragments = []
            for k, a in enumerate(p.ctx_ids):
                lo, hi = p.ctx_ptr[k], p.ctx_ptr[k + 1]
                keep = x[lo:hi] != 0.0
                if keep.any():
                    fragments.append((int(a), p.coord_pauli[lo:hi][keep].copy(), x[lo:hi][keep].copy()))
            self.x0.append(fragments)
            self.targets.append(split_coords[i][2])
            self.bound[i] = float(sum(abs(v) for v in split_coords[i][2].values()))
        self.candidates = [i for i in range(n_arms) if self.targets[i] and self.bound[i] > 0.0]
        self.x0_l1_total = np.array([sum(float(np.abs(x).sum()) for _, _, x in f) for f in self.x0])
        self._problems = None
        self.trace: list | None = None  # debugging: one entry per checkpoint
        self.n_library_contexts = library.n_contexts

    # --- the splitting problems (built once, covariance replaced at every design) -------------------------------

    def _split_problems(self) -> list:
        if self._problems is None:
            identity = lambda a, p: np.eye(len(p))  # noqa: E731
            self._problems = [FragmentProblem(i, c, p, t, self.library.home, identity)
                              for i, (c, p, t) in enumerate(self.split_coords)]
        return self._problems

    # --- a design for the next block ------------------------------------------------------------------------------

    def _fixed_designs(self, requested):
        return {i: self.x0[i] for i in requested}

    def _plan(self, method: str, requested: list[int], epsilon: float, covariance: DesignCovariance, stats: dict):
        """Designs (arm -> fragments), unrounded plan, rounded cost and the choice made."""
        clock = time.perf_counter()
        sigmas = np.zeros((len(requested), self.n_contexts))
        for row, i in enumerate(requested):
            for a, p, x in self.x0[i]:
                sigmas[row, a] = math.sqrt(max(float(x @ covariance(a, p) @ x), 0.0))
        fixed_plan = allocate(sigmas, epsilon)
        fixed_cost = rounded_total(fixed_plan, self.floor)
        stats["plan_seconds"] += time.perf_counter() - clock
        designs, plan, cost, choice = self._fixed_designs(requested), fixed_plan, fixed_cost, "fixed"
        if method in ("FC_SE_CS", "FC_SE_CS_DATA", "FC_SE_CS_ORACLE"):
            clock = time.perf_counter()
            problems = self._split_problems()
            for i in requested:
                p = problems[i]
                p.x = np.zeros(p.n_coordinates)
                p.x[p.reference] = p.pauli_target
                p.update_blocks(covariance)
            solution = DesignSet(problems, self.n_contexts).solve(requested, epsilon)
            optimized_cost = rounded_total(solution.shots, self.floor)
            candidate = {}
            valid = True
            for i in requested:
                p = problems[i]
                keep = p.x != 0.0
                ctx, pauli, x = p.coord_ctx[keep], p.coord_pauli[keep], p.x[keep]
                if p.constraint_residual() > 1e-9 * max(1.0, float(np.abs(p.pauli_target).max(initial=1.0))):
                    valid = False
                    break
                fragments = []
                for a in np.unique(ctx):
                    chosen = ctx == a
                    fragments.append((int(a), pauli[chosen].copy(), x[chosen].copy()))
                candidate[i] = fragments
            stats["solve_seconds"] += time.perf_counter() - clock
            if valid and any(solution.shots[a] <= 1e-9 for fragments in candidate.values() for a, _, _ in fragments):
                valid = False  # a context carries weight but the plan gives it no shots
            if valid and optimized_cost < fixed_cost:
                designs, plan, cost, choice = candidate, solution.shots, optimized_cost, "optimized"
            elif not valid:
                choice = "fixed_invalid"
        return designs, plan, cost, choice

    # --- intervals ----------------------------------------------------------------------------------------------------

    def _by_context(self, designs: dict) -> dict:
        grouped: dict[int, list] = {}
        for i, fragments in designs.items():
            for a, p, x in fragments:
                grouped.setdefault(a, []).append((i, p, x))
        return grouped

    def _residual(self, i: int, fragments) -> float:
        total: dict[int, float] = {}
        for _, p, x in fragments:
            for l, v in zip(p.tolist(), x.tolist()):
                total[l] = total.get(l, 0.0) + v
        target = self.targets[i]
        keys = set(total) | set(target)
        return float(sum(abs(target.get(l, 0.0) - total.get(l, 0.0)) for l in keys))

    # --- the trial --------------------------------------------------------------------------------------------------------

    def run(self, rng: np.random.Generator, method: str, rho: float, delta: float, schedule: str = "literal", *,
            restart_ratio: float = 1.0, batch_fix: bool = False, full_dominance: bool = False, growth: float = 2.0,
            leader_out: bool = False, floor: int = FLOOR) -> dict:
        """``leader_out`` (task T25): the relative certificate compares the leader's lower bound with the largest upper bound of the OTHER
        survivors, L_k >= (1 - rho) max_{i != k} U_i, instead of the maximum over all survivors including k (GS-1.0 as run so far).  Both
        certify the rho-good event; the default form also needs the leader's own half-width to shrink to rho / (2 - rho) of |g_k|.
        ``full_dominance`` (task T24): FC_FULL accepts a sole virtual survivor (every other candidate strictly dominated) also at rho > 0,
        then tries the relative test; without it (the T23 runs) the full-pool arm at rho > 0 can only stop by the relative test, which makes it
        wait for the leader's own interval to shrink to relative width rho.  ``growth``: factor between consecutive checkpoints of a block
        (2 = the doubling schedule of SI S4; a smaller factor tests the checkpoint-granularity question of task T24).
        ``restart_ratio`` (task T22/T23): 1.0 = a new block after every elimination (GS-1.0 as first written, the default); 0 = never
        (the block and its doubling continue; its design was fixed before its data, so the block intervals stay valid); 0.5 = only when at most
        half of the block's requested candidates survive.  ``batch_fix``: first-checkpoint cap max(4096, 64 L_active) instead of 4096 (keeps
        the planned allocation when more than 128 contexts are active)."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        if floor < 2:
            raise ValueError("the floor must be at least 2 outcomes per active context (a sample variance needs two)")
        self.floor = int(floor)
        started = time.perf_counter()
        full = method == "FC_FULL"
        sparse_prior = self.prior if method in ("FC_SE_CS",) else None
        counts = Counts(self.n_contexts)  # all blocks (design covariance, historical bound)
        if method in ORACLE_METHODS:
            if self.truth_moments is None:
                raise ValueError(f"{method} needs the moments of the true state (selector.truth_moments)")
            covariance = DesignCovariance(self.library, counts, self.truth_moments, exact=True)
        else:
            covariance = DesignCovariance(self.library, counts, sparse_prior)
        M = len(self.candidates)
        A = self.n_contexts
        truth_abs = np.abs(self.truth)
        top = float(truth_abs[self.candidates].max()) if M else 0.0
        maximizers = {i for i in self.candidates if truth_abs[i] >= top - TIE_TOLERANCE}
        lo = {i: -float(self.bound[i]) for i in self.candidates}
        hi = {i: float(self.bound[i]) for i in self.candidates}
        survivors = list(self.candidates)
        record = {"schedule": schedule, "status": None, "selected_index": None, "block_shots": [], "design_choices": [], "blocks": [],
                  "block_failures": 0, "history_failures": 0, "maximizer_eliminated": False, "checkpoints": 0,
                  "fallback_designs": 0, "rounds_reached": 1, "restart_ratio": restart_ratio, "batch_fix": batch_fix,
                  "full_dominance": full_dominance, "growth": growth, "leader_out": leader_out, "floor": self.floor}
        stats = {"plan_seconds": 0.0, "solve_seconds": 0.0, "sample_seconds": 0.0, "interval_seconds": 0.0}
        if not self.candidates:
            record.update(status="stationary", selected_index=None)
            return self._finish(record, started, rho, maximizers, stats, lo, hi, survivors)
        history_by_context: dict | None = None
        shots_total = 0
        r = 1
        block = 0

        def halfwidth(i):
            return 0.5 * (hi[i] - lo[i])

        def certificate() -> int | None:
            if len(survivors) == 1:
                return survivors[0]
            mags = {i: magnitude_interval(lo[i], hi[i]) for i in survivors}
            best = max(survivors, key=lambda i: (mags[i][0], -i))
            if full and (rho <= 0.0 or full_dominance):
                biggest_lower = mags[best][0]
                virtual = [i for i in survivors if mags[i][1] >= biggest_lower]
                if len(virtual) == 1:
                    return virtual[0]
                if rho <= 0.0:
                    return None
            ceiling = max(mags[i][1] for i in survivors if not (leader_out and i == best))  # len(survivors) > 1 here
            return best if mags[best][0] >= (1.0 - rho) * ceiling and (rho > 0.0) else None

        def requested_at(round_number: int) -> list[int]:
            threshold = T_MULT * epsilon_round(round_number)
            if full:  # every non-constant candidate constrains every round, until all have reached the round target
                return list(self.candidates) if any(halfwidth(i) > threshold for i in self.candidates) else []
            return [i for i in survivors if halfwidth(i) > threshold]

        decided = certificate()
        while decided is None:
            requested = requested_at(r)
            while not requested and r < MAX_ROUNDS:
                r += 1
                requested = requested_at(r)
            if not requested:
                record["status"] = "unresolved"
                break
            record["rounds_reached"] = r
            epsilon = epsilon_round(r)
            threshold = T_MULT * epsilon
            designs, plan, cost, choice = self._plan(method, requested, epsilon, covariance, stats)
            record["design_choices"].append(choice)
            record["fallback_designs"] += int(choice.startswith("fixed_"))
            block += 1
            active = np.flatnonzero(plan > 1e-9)
            q = np.zeros(self.n_contexts, dtype=np.int64)
            k = 0
            if schedule == "literal":
                # SI S4 as written: halve the plan, rounding with floor 32, until the total is below 4096 or every context
                # is at its floor.  With more than 128 active contexts the floors alone exceed 4096, so every context ends
                # at the floor and the first checkpoint is uniform.
                while True:
                    q[:] = 0
                    q[active] = np.maximum(self.floor, np.ceil(plan[active] / 2.0 ** k - 1e-9)).astype(np.int64)
                    if q.sum() < (max(FIRST_CHECK, 64 * int(active.size)) if batch_fix else FIRST_CHECK) or (q[active] == self.floor).all():
                        break
                    k += 1
            else:
                # deviation ("proportional"): stop halving when the *unfloored* counts total less than 4096, then apply the
                # floor, so that the checkpoints keep the proportions of the plan
                while np.maximum(1.0, np.ceil(plan[active] / 2.0 ** k - 1e-9)).sum() >= FIRST_CHECK and plan[active].max() / 2.0 ** k > 1.0:
                    k += 1
                q[:] = 0
                q[active] = np.maximum(self.floor, np.ceil(plan[active] / 2.0 ** k - 1e-9)).astype(np.int64)
            residuals = {i: self._residual(i, designs[i]) for i in requested}
            by_context = self._by_context(designs)
            blk = Counts(self.n_contexts)
            block_total = 0
            record["blocks"].append({"round": r, "requested": len(requested), "survivors": len(survivors),
                                     "contexts": int(active.size), "plan_total": cost, "choice": choice,
                                     "base_total": int(q.sum()), "halvings": k})
            ell = 0
            stop_block = False
            while not stop_block:
                ell += 1
                if growth == 2.0:
                    add = q if ell == 1 else q * (2 ** (ell - 2))
                else:  # cumulative counts q growth^(ell-1), rounded up per context
                    cumulative = np.ceil(q * growth ** (ell - 1) - 1e-9).astype(np.int64)
                    add = cumulative - (np.ceil(q * growth ** (ell - 2) - 1e-9).astype(np.int64) if ell > 1 else 0)
                increment = int(add[active].sum())
                if shots_total + increment > self.cap:
                    record["status"] = "unresolved"
                    record["cap_hit"] = True
                    break
                clock = time.perf_counter()
                # very large counts: draw in one multinomial per context
                draws = rng.multinomial(add[active], self.dist[active])
                counts.add(active, draws)
                blk.add(active, draws)
                shots_total += increment
                block_total += increment
                stats["sample_seconds"] += time.perf_counter() - clock
                record["checkpoints"] += 1
                clock = time.perf_counter()
                eta = (delta * 0.5) / (M * block * (block + 1) * ell * (ell + 1))
                moments = fragment_moments(blk, self.library, by_context)
                for i in requested:
                    items = [(a, moments[(i, a)]) for a, _, _ in designs[i]]
                    interval = bernstein_interval([m[1][0] for m in items], [m[1][1] for m in items],
                                                  [m[1][2] for m in items], [m[1][3] for m in items], eta,
                                                  len(items), 0.0, residuals[i])
                    if not (interval[0] - CONFLICT_TOLERANCE <= self.truth[i] <= interval[1] + CONFLICT_TOLERANCE):
                        record["block_failures"] += 1
                    lo[i], hi[i] = max(lo[i], interval[0]), min(hi[i], interval[1])
                # historical bound for the fixed reference reconstruction, every surviving candidate
                if history_by_context is None or record.get("history_set") != len(survivors):
                    history_by_context = self._by_context({i: self.x0[i] for i in survivors})
                    record["history_set"] = len(survivors)
                hist_moments = fragment_moments(counts, self.library, history_by_context)
                zeta_base = (delta * 0.5) / (M * A)
                accum_lo = {i: 0.0 for i in survivors}
                accum_hi = {i: 0.0 for i in survivors}
                measured_l1 = {i: 0.0 for i in survivors}
                for (i, a), (n, m, s2, l1) in hist_moments.items():
                    flo, fhi = history_fragment(n, m, s2, l1, zeta_base)
                    accum_lo[i] += flo
                    accum_hi[i] += fhi
                    measured_l1[i] += l1
                conflict = False
                for i in survivors:
                    unmeasured = float(self.x0_l1_total[i]) - measured_l1[i]
                    hlo, hhi = accum_lo[i] - unmeasured, accum_hi[i] + unmeasured
                    if not (hlo - CONFLICT_TOLERANCE <= self.truth[i] <= hhi + CONFLICT_TOLERANCE):
                        record["history_failures"] += 1
                    lo[i], hi[i] = max(lo[i], hlo), min(hi[i], hhi)
                    if lo[i] > hi[i] + CONFLICT_TOLERANCE:
                        conflict = True
                        break
                stats["interval_seconds"] += time.perf_counter() - clock
                if self.trace is not None:
                    widest = max(requested, key=halfwidth)
                    self.trace.append({"block": block, "ell": ell, "round": r, "requested": list(requested)[:6], "n_requested": len(requested),
                                       "widest": int(widest), "halfwidth": halfwidth(widest), "threshold": threshold,
                                       "block_shots": block_total, "h": len(designs[widest]),
                                       "l1": float(sum(np.abs(x).sum() for _, _, x in designs[widest]))})
                if conflict:
                    record["status"] = "interval_conflict"
                    stop_block = True
                    break
                for i in survivors:  # numerical safety: nothing below the tolerance is a conflict
                    if lo[i] > hi[i]:
                        mid = 0.5 * (lo[i] + hi[i])
                        lo[i] = hi[i] = mid
                # elimination
                if not full:
                    best_lower = max(magnitude_interval(lo[i], hi[i])[0] for i in survivors)
                    kept = [i for i in survivors if magnitude_interval(lo[i], hi[i])[1] >= best_lower]
                    if len(kept) < len(survivors):
                        gone = set(survivors) - set(kept)
                        if maximizers & gone:
                            record["maximizer_eliminated"] = True
                        survivors = kept
                        history_by_context = None
                decided = certificate()
                if decided is not None:
                    stop_block = True
                    break
                live = [i for i in requested if i in set(survivors)] if not full else requested
                done = all(halfwidth(i) <= threshold for i in live)
                if done:
                    r += 1
                    stop_block = True
                elif len(live) < len(requested) and len(live) <= restart_ratio * len(requested):
                    stop_block = True  # elimination changed the requested set: redesign in the same round (policy: restart_ratio)
            record["block_shots"].append(block_total)
            if record["status"] in ("unresolved", "interval_conflict"):
                break
            if decided is None and r > MAX_ROUNDS:
                record["status"] = "unresolved"
                break
        if record["status"] is None and decided is not None:
            record["status"] = "certified"
            record["selected_index"] = int(decided)
        elif record["status"] in ("unresolved", "interval_conflict"):
            midpoints = {i: abs(0.5 * (lo[i] + hi[i])) for i in survivors}
            record["selected_index"] = int(max(midpoints, key=midpoints.get)) if midpoints else None
        record["selection_shots"] = int(shots_total)
        record["peak_context_count"] = int(max((b["contexts"] for b in record["blocks"]), default=0))
        return self._finish(record, started, rho, maximizers, stats, lo, hi, survivors)

    def _finish(self, record, started, rho, maximizers, stats, lo, hi, survivors) -> dict:
        record.setdefault("selection_shots", int(sum(record["block_shots"])))
        record.pop("history_set", None)
        chosen = record["selected_index"]
        record["correct"] = bool(chosen is not None and chosen in maximizers)
        truth_abs = np.abs(self.truth)
        top = float(truth_abs[self.candidates].max()) if self.candidates else 0.0
        # rho-good: |g_chosen| >= (1 - rho) max |g| (equals ``correct`` at rho = 0 up to ties); added with T23
        record["rho_good"] = bool(chosen is not None and truth_abs[chosen] >= (1.0 - rho) * top - TIE_TOLERANCE)
        record["tied_maximizers"] = int(len(maximizers))
        record["n_survivors"] = int(len(survivors))
        record["wall_seconds"] = round(time.perf_counter() - started, 3)
        record.update({k: round(v, 3) for k, v in stats.items()})
        record["n_blocks"] = len(record["blocks"])
        return record
