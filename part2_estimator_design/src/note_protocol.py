"""The protocol of ``reports/prior/Classical_priors_for_measurement_design.tex`` (the 19 October 2026 method note).

Differences from the reframed protocol (``reframe.py``), all taken from the note:

* design covariance: the raw-moment update ``Sigma_design = R~ - mu~ mu~'`` with prior strength ``tau`` (equivalent group outcomes) and
  full-support mixing ``eps_c`` (Eqs. 18-23); the prior never enters a gradient estimate;
* every batch uses the *full fixed cover* (all contexts), each with at least ``n_min,r`` shots, and a retained valid splitting for every
  original candidate; the objective is the equal-weight sum of the predicted variances of the *active* candidates at a fixed batch budget
  ``B`` (Eq. 27), solved by alternating exact per-generator splitting at fixed counts with the floored square-root allocation (Eq. 31);
* optional coefficient bound ``sum_a ||b_{j,a}||_1 <= gamma sum_P |h_jP|`` (Eq. 28): the alternating solver has no such constraint, so a
  design that violates it is replaced by the largest convex combination with the fixed reconstruction that satisfies it;
* batches are fresh and prospective; the estimate of a batch is formed from its own outcomes only;
* confidence: ``hoeffding`` is Appendix A as written (data-independent radius from the coefficient ranges and ``n_min,r``, weights ``u_r``
  proportional to the batch budgets); ``bernstein`` replaces it by the bounded-variance block intervals of the reframed protocol, as the
  note allows ("more efficient valid bounds can replace it").

Arms (the note's comparison list, Sec. "Parameters and what must be tested"):
``NOTE_I`` fixed reconstruction, measured covariance (the existing FC+BAI design); ``NOTE_II`` classical-initialized allocation only;
``NOTE_III`` classical-initialized splitting and allocation (static prior); ``NOTE_IV`` the same with quantum covariance updates;
``NOTE_IV_DATA`` (diagnostic, not in the note) splitting and allocation with measured covariances only.
"""
from __future__ import annotations

import math
import time

import numpy as np

from classical_prior import MixedMoments
from reframe import (CONFLICT_TOLERANCE, TIE_TOLERANCE, Counts, ReframeSelector, bernstein_interval, fragment_moments,
                     history_fragment, magnitude_interval)
from sampler import pauli_covariance

NOTE_ARMS = {
    "NOTE_I": dict(split=False, prior=False),
    "NOTE_II": dict(split=False, prior=True, tau=100.0),
    "NOTE_III": dict(split=True, prior=True, tau=math.inf),
    "NOTE_IV": dict(split=True, prior=True, tau=100.0),
    "NOTE_IV_DATA": dict(split=True, prior=False),
}
# Confidence in the classical prior as a function of the energy gap between the classical state and the sampled quantum energy:
#   tau = tau0 / (1 + |dE| / G0)              ("abs":  absolute gap, Hartree)
#   tau = tau0 / (1 + (|dE| / |E_q|) / R0)    ("rel":  gap relative to the sampled quantum energy)
# with dE = E_c - E_hat, E_c = <c|H|c> computed classically and E_hat = E_q + sigma N(0,1) the energy "measured" for the state
# (sigma = 1e-3 Ha: the standard error of a converged VQE energy evaluation used throughout this repository; 1e-4 Ha for ten times more shots).
G0 = 1e-3
R0 = 1e-5
TAU0 = 100.0
for _name, _form, _sigma in (("GABS3", "abs", 1e-3), ("GABS4", "abs", 1e-4), ("GREL3", "rel", 1e-3), ("GREL4", "rel", 1e-4)):
    NOTE_ARMS[f"NOTE_IV_{_name}"] = dict(split=True, prior=True, tau=TAU0, trust=(_form, _sigma))
# degraded classical state (K = max(1, K_m // 2) determinants): a prior that is in fact worse, so the gap varies
NOTE_ARMS["NOTE_IV_HALF"] = dict(split=True, prior=True, tau=TAU0, keep="half")
for _name, _form, _sigma in (("GABS3", "abs", 1e-3), ("GABS4", "abs", 1e-4), ("GREL3", "rel", 1e-3), ("GREL4", "rel", 1e-4)):
    NOTE_ARMS[f"NOTE_IV_HALF_{_name}"] = dict(split=True, prior=True, tau=TAU0, keep="half", trust=(_form, _sigma))
CONFIDENCE = ("hoeffding", "bernstein")


def trusted_tau(tau0: float, form: str, gap: float, energy: float) -> float:
    if form == "abs":
        return tau0 / (1.0 + abs(gap) / G0)
    return tau0 / (1.0 + (abs(gap) / abs(energy)) / R0)
NMIN0 = 32
BETA = 2.0  # batch budget B_r = BETA * L * n_min,r
EPS_C = 0.1
GAMMA = 2.0
ALTERNATIONS = 6


def water_fill(weights: np.ndarray, budget: float, floor: float) -> np.ndarray:
    """``n_a = max(floor, t sqrt(W_a))`` with ``sum n = budget`` (Eq. 31); uniform when every weight is zero."""
    root = np.sqrt(np.maximum(weights, 0.0))
    count = root.size
    if budget <= floor * count + 1e-9:
        return np.full(count, budget / count)
    if root.max() <= 0.0:
        return np.full(count, budget / count)
    lo, hi = 0.0, budget / root[root > 0].min()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.maximum(floor, mid * root).sum() < budget:
            lo = mid
        else:
            hi = mid
    return np.maximum(floor, hi * root)


def integer_counts(plan: np.ndarray, budget: int, floor: int) -> np.ndarray:
    counts = np.maximum(floor, np.floor(plan)).astype(np.int64)
    shortfall = int(budget - counts.sum())
    if shortfall > 0:
        order = np.argsort(-(plan - np.floor(plan)))
        counts[order[:shortfall]] += 1
    elif shortfall < 0:  # only through the floors; keep the larger total
        pass
    return counts


class NoteCovariance:
    """Design covariance of one context (Eqs. 19-23): prior moments updated by the counts, or the measured covariance."""

    def __init__(self, library, counts: Counts, prior, tau: float) -> None:
        self.library, self.counts, self.prior, self.tau = library, counts, prior, tau

    def __call__(self, alpha: int, paulis: np.ndarray) -> np.ndarray:
        k = len(paulis)
        n = int(self.counts.n[alpha])
        context = self.library.contexts[alpha]
        if self.prior is None:
            if n < 2:
                return np.eye(k)
            positions = context.positions(paulis)
            emp = pauli_covariance(self.counts.group_means(alpha), context.member_zmask[positions],
                                   context.member_sign[positions]) * n / (n - 1.0)
            return 0.5 * (emp + emp.T) + 1e-3 * np.eye(k)
        positions = context.positions(paulis)
        zmask, sign = context.member_zmask[positions], context.member_sign[positions]
        g0 = self.prior.group_means(alpha)
        if math.isinf(self.tau) or n == 0:
            g = g0
        else:
            g = (self.tau * g0 + n * self.counts.group_means(alpha)) / (self.tau + n)
        cov = pauli_covariance(g, zmask, sign)
        return 0.5 * (cov + cov.T)


class NoteSelector(ReframeSelector):
    """The note's protocol on the data structures of :class:`reframe.ReframeSelector`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mixed: dict = {}
        self.priors = {"full": self.prior}  # classical states: "full" (K = m + 1) and optionally "half"
        self.energy_c: dict[str, float] = {}  # <c|H|c> of each of them
        self.energy_q: float | None = None

    def _prior(self, eps: float, keep: str = "full"):
        if (keep, eps) not in self._mixed:
            self._mixed[(keep, eps)] = MixedMoments(self.priors[keep], eps)
        return self._mixed[(keep, eps)]

    def _fragments(self, p, x: np.ndarray) -> list:
        keep = x != 0.0
        ctx, pauli, values = p.coord_ctx[keep], p.coord_pauli[keep], x[keep]
        return [(int(a), pauli[ctx == a].copy(), values[ctx == a].copy()) for a in np.unique(ctx)]

    def _weights(self, designs: dict, active: list, cov) -> np.ndarray:
        omega = 1.0 / len(active)
        W = np.zeros(self.n_contexts)
        for i in active:
            for a, p, x in designs[i]:
                W[a] += omega * max(float(x @ cov(a, p) @ x), 0.0)
        return W

    def run_note(self, rng: np.random.Generator, arm: str, confidence: str, rho: float, delta: float, *,
                 gamma: float = GAMMA, eps_c: float = EPS_C) -> dict:
        cfg = NOTE_ARMS[arm]
        if confidence not in CONFIDENCE:
            raise ValueError(confidence)
        started = time.perf_counter()
        split = cfg["split"]
        keep = cfg.get("keep", "full")
        prior = self._prior(eps_c, keep) if cfg["prior"] else None
        tau = cfg.get("tau", 0.0)
        extra: dict = {}
        if cfg["prior"] and self.energy_c.get(keep) is not None:
            extra["gap_true"] = self.energy_c[keep] - self.energy_q
        if "trust" in cfg:
            form, sigma = cfg["trust"]
            sampled = self.energy_q + sigma * rng.normal()  # the energy measured for this state
            gap = self.energy_c[keep] - sampled
            tau = trusted_tau(tau, form, gap, sampled)
            extra.update(gap_hat=gap, tau_used=tau)
        counts = Counts(self.n_contexts)
        cov = NoteCovariance(self.library, counts, prior, tau)
        M = len(self.candidates)
        L = self.n_contexts
        truth_abs = np.abs(self.truth)
        top = float(truth_abs[self.candidates].max()) if M else 0.0
        maximizers = {i for i in self.candidates if truth_abs[i] >= top - TIE_TOLERANCE}
        lo = {i: -float(self.bound[i]) for i in self.candidates}
        hi = {i: float(self.bound[i]) for i in self.candidates}
        survivors = list(self.candidates)
        record = {"schedule": "note", **extra, "status": None, "selected_index": None, "block_shots": [], "design_choices": [], "blocks": [],
                  "block_failures": 0, "history_failures": 0, "maximizer_eliminated": False, "checkpoints": 0,
                  "fallback_designs": 0, "rounds_reached": 0, "capped_arms": 0}
        stats = {"plan_seconds": 0.0, "solve_seconds": 0.0, "sample_seconds": 0.0, "interval_seconds": 0.0}
        if not self.candidates:
            record.update(status="stationary")
            return self._finish(record, started, rho, maximizers, stats, lo, hi, survivors)
        designs = {i: self.x0[i] for i in self.candidates}
        warm: dict[int, np.ndarray] = {}
        bound_C = {i: (gamma if split else 1.0) * float(self.bound[i]) for i in self.candidates}
        numer = {i: 0.0 for i in self.candidates}
        q_sum = {i: 0.0 for i in self.candidates}
        s_sum = 0.0
        shots_total = 0
        history_by_context = None
        history_size = -1

        def certificate():
            if len(survivors) == 1:
                return survivors[0]
            mags = {i: magnitude_interval(lo[i], hi[i]) for i in survivors}
            best = max(survivors, key=lambda i: (mags[i][0], -i))
            ceiling = max(mags[i][1] for i in survivors)
            return best if rho > 0.0 and mags[best][0] >= (1.0 - rho) * ceiling else None

        decided = certificate()
        r = 0
        while decided is None:
            r += 1
            nmin = NMIN0 * 2 ** (r - 1)
            budget = int(round(BETA * L * nmin))
            if shots_total + budget > self.cap:
                record["status"] = "unresolved"
                record["cap_hit"] = True
                break
            record["rounds_reached"] = r
            clock = time.perf_counter()
            active = list(survivors)
            iterations = 0
            capped = 0
            if split:
                problems = self._split_problems()
                for i in active:
                    p = problems[i]
                    p.update_blocks(cov)
                    p.x = warm[i].copy() if i in warm else np.zeros(p.n_coordinates)
                    if i not in warm:
                        p.x[p.reference] = p.pauli_target
                omega = 1.0 / len(active)
                previous = None
                for iterations in range(1, ALTERNATIONS + 1):
                    W = np.zeros(L)
                    for i in active:
                        p = problems[i]
                        W[p.ctx_ids] += omega * np.maximum(p.context_second_moments(), 0.0)
                    n = water_fill(W, budget, nmin)
                    objective = float((W / n).sum())
                    if previous is not None and previous - objective <= 1e-3 * previous:
                        break
                    previous = objective
                    for i in active:
                        problems[i].optimise(n)
                        problems[i]._structures.clear()  # used once: keeps the memory at one generator's worth
                for i in active:  # coefficient bound (Eq. 28): largest convex combination with the fixed reconstruction
                    p = problems[i]
                    home = np.zeros(p.n_coordinates)
                    home[p.reference] = p.pauli_target
                    limit = bound_C[i] * (1.0 + 1e-12)
                    if float(np.abs(p.x).sum()) > limit:
                        capped += 1
                        lo_t, hi_t = 0.0, 1.0
                        for _ in range(50):
                            mid = 0.5 * (lo_t + hi_t)
                            if float(np.abs(home + mid * (p.x - home)).sum()) <= limit:
                                lo_t = mid
                            else:
                                hi_t = mid
                        p.x = home + lo_t * (p.x - home)
                    warm[i] = p.x.copy()
                    designs[i] = self._fragments(p, p.x)
            W = self._weights(designs, active, cov)
            plan = water_fill(W, budget, nmin)
            counts_int = integer_counts(plan, budget, nmin)
            stats["plan_seconds"] += time.perf_counter() - clock
            record["capped_arms"] += capped
            record["design_choices"].append("split" if split else "fixed")
            by_context = self._by_context(designs)
            residuals = {i: self._residual(i, designs[i]) for i in self.candidates}
            clock = time.perf_counter()
            draws = rng.multinomial(counts_int, self.dist)
            blk = Counts(self.n_contexts)
            everything = np.arange(self.n_contexts)
            blk.add(everything, draws)
            counts.add(everything, draws)
            shots_total += int(counts_int.sum())
            record["block_shots"].append(int(counts_int.sum()))
            record["blocks"].append({"round": r, "requested": len(active), "survivors": len(survivors), "contexts": L, "plan_total": budget,
                                     "choice": "split" if split else "fixed", "base_total": budget, "n_min": nmin,
                                     "alternations": iterations, "capped": capped})
            record["checkpoints"] += 1
            stats["sample_seconds"] += time.perf_counter() - clock
            clock = time.perf_counter()
            moments = fragment_moments(blk, self.library, by_context)
            conflict = False
            if confidence == "hoeffding":
                weight = float(budget)  # u_r proportional to the batch budgets
                s_sum += weight
                log_term = math.log(2.0 * M * r * (r + 1) / delta)
                for i in self.candidates:
                    mean = sum(moments[(i, a)][1] for a, _, _ in designs[i])
                    numer[i] += weight * mean
                    q_sum[i] += bound_C[i] ** 2 * weight * weight / nmin
                    d = math.sqrt(2.0 * q_sum[i] * log_term) / s_sum
                    centre = numer[i] / s_sum
                    interval = (centre - d, centre + d)
                    if not (interval[0] - CONFLICT_TOLERANCE <= self.truth[i] <= interval[1] + CONFLICT_TOLERANCE):
                        record["block_failures"] += 1
                    lo[i], hi[i] = max(lo[i], interval[0]), min(hi[i], interval[1])
            else:
                eta = (delta * 0.5) / (M * r * (r + 1) * 2)
                for i in self.candidates:
                    items = [(a, moments[(i, a)]) for a, _, _ in designs[i]]
                    interval = bernstein_interval([m[1][0] for m in items], [m[1][1] for m in items], [m[1][2] for m in items],
                                                  [m[1][3] for m in items], eta, len(items), 0.0, residuals[i])
                    if not (interval[0] - CONFLICT_TOLERANCE <= self.truth[i] <= interval[1] + CONFLICT_TOLERANCE):
                        record["block_failures"] += 1
                    lo[i], hi[i] = max(lo[i], interval[0]), min(hi[i], interval[1])
                # fixed-reconstruction historical bound (all earlier batches), as in the reframed protocol
                if history_by_context is None or history_size != len(survivors):
                    history_by_context = self._by_context({i: self.x0[i] for i in survivors})
                    history_size = len(survivors)
                hist = fragment_moments(counts, self.library, history_by_context)
                zeta_base = (delta * 0.5) / (M * L)
                acc_lo = {i: 0.0 for i in survivors}
                acc_hi = {i: 0.0 for i in survivors}
                measured = {i: 0.0 for i in survivors}
                for (i, a), (n_, m_, s2_, l1_) in hist.items():
                    flo, fhi = history_fragment(n_, m_, s2_, l1_, zeta_base)
                    acc_lo[i] += flo
                    acc_hi[i] += fhi
                    measured[i] += l1_
                for i in survivors:
                    unmeasured = float(self.x0_l1_total[i]) - measured[i]
                    hlo, hhi = acc_lo[i] - unmeasured, acc_hi[i] + unmeasured
                    if not (hlo - CONFLICT_TOLERANCE <= self.truth[i] <= hhi + CONFLICT_TOLERANCE):
                        record["history_failures"] += 1
                    lo[i], hi[i] = max(lo[i], hlo), min(hi[i], hhi)
            for i in survivors:
                if lo[i] > hi[i] + CONFLICT_TOLERANCE:
                    conflict = True
                elif lo[i] > hi[i]:
                    lo[i] = hi[i] = 0.5 * (lo[i] + hi[i])
            stats["interval_seconds"] += time.perf_counter() - clock
            if conflict:
                record["status"] = "interval_conflict"
                break
            best_lower = max(magnitude_interval(lo[i], hi[i])[0] for i in survivors)
            kept = [i for i in survivors if magnitude_interval(lo[i], hi[i])[1] >= best_lower]
            if len(kept) < len(survivors):
                if maximizers & (set(survivors) - set(kept)):
                    record["maximizer_eliminated"] = True
                survivors = kept
                history_by_context = None
            decided = certificate()
        if record["status"] is None and decided is not None:
            record["status"] = "certified"
            record["selected_index"] = int(decided)
        elif record["status"] in ("unresolved", "interval_conflict"):
            midpoints = {i: abs(0.5 * (lo[i] + hi[i])) for i in survivors}
            record["selected_index"] = int(max(midpoints, key=midpoints.get)) if midpoints else None
        record["selection_shots"] = int(shots_total)
        record["peak_context_count"] = int(L)
        return self._finish(record, started, rho, maximizers, stats, lo, hi, survivors)
