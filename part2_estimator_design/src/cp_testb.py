"""Test B of work order CP-1.2: strict best-arm identification with the common conservative confidence (Appendix A) and the common
design program, for one source of initial moments.  The procedure is the one of ``reframe.ReframeSelector.run`` with the work order's
choices: the certificate ``L_k >= max_{j != k} U_j`` (or a sole survivor), a new block after *any* elimination, the program with the L1
bound solved by CVXPY (frozen reconstruction on failure), a cap of 1e7 shots and 100 fresh blocks per trial.
"""
from __future__ import annotations

import math
import time

import numpy as np

from cp_order import FLOOR, CPSelector
from note_protocol import NoteCovariance
from reframe import (CONFLICT_TOLERANCE, FIRST_CHECK, MAX_ROUNDS, T_MULT, TIE_TOLERANCE, Counts, bernstein_interval, epsilon_round,
                     fragment_moments, history_fragment, magnitude_interval)

MAX_BLOCKS = 100


def run_cp(sel: CPSelector, rng: np.random.Generator, prior, tau, *, delta: float = 0.05, cap: float = 1e7, batch_fix: bool = False,
           rho: float = 0.0, design: str = "program", eliminate: bool = True, restart_ratio: float = 1.0) -> dict:
    """``rho`` = 0: certify the largest |gradient| (strict identification, the work order).  ``rho`` > 0: stop when the leader is certified
    rho-good against every survivor, i.e. its lower magnitude bound is at least (1 - rho) times the upper magnitude bound of every other
    survivor (the interval form of Eq. (rhostop) of the main paper: no pairwise covariance term, so slightly more conservative).  Elimination
    stays exact.

    Gate G2 (2026-11-12): the hierarchy of the reframed paper under one confidence construction.  ``design="program"``, ``eliminate=True``: the
    proposed method (data-only when the prior is the neutral one).  ``design="fixed"``, ``eliminate=True``: FC_SE, the frozen reference
    reconstruction ``b^0`` with counts from the measured-covariance allocation (the program's own fallback, used in every block).
    ``design="fixed"``, ``eliminate=False``: FC_FULL, the same estimators for every candidate in every block, no candidate is removed and the
    certificate is taken over all of them.  Block intervals, the history bound, the schedule, floors and caps are those of the other arms."""
    started = time.perf_counter()
    L = sel.n_contexts
    counts = Counts(L)
    cov = NoteCovariance(sel.library, counts, prior, tau)
    M = len(sel.candidates)
    truth_abs = np.abs(sel.truth)
    top = float(truth_abs[sel.candidates].max()) if M else 0.0
    maximizers = {i for i in sel.candidates if truth_abs[i] >= top - TIE_TOLERANCE}
    lo = {i: -float(sel.bound[i]) for i in sel.candidates}
    hi = {i: float(sel.bound[i]) for i in sel.candidates}
    survivors = list(sel.candidates)
    record = {"status": None, "selected_index": None, "block_shots": [], "solver": {}, "block_failures": 0, "history_failures": 0,
              "maximizer_eliminated": False, "checkpoints": 0, "rounds_reached": 1, "design_choices": []}
    seconds = {"program": 0.0, "sample": 0.0, "interval": 0.0}
    shots_total = 0
    blocks = 0
    r = 1
    history = None
    history_size = -1

    def halfwidth(i):
        return 0.5 * (hi[i] - lo[i])

    def kept_requested(current):
        """Survivors that the current block was designed for."""
        planned = set(requested)
        return [i for i in current if i in planned]

    def certificate():
        if len(survivors) == 1:
            return survivors[0]
        mags = {i: magnitude_interval(lo[i], hi[i]) for i in survivors}
        best = max(survivors, key=lambda i: (mags[i][0], -i))
        others = max(mags[j][1] for j in survivors if j != best)
        return best if mags[best][0] >= (1.0 - rho) * others else None

    decided = certificate() if M else None
    if not M:
        record["status"] = "stationary"
    while M and decided is None:
        requested = [i for i in survivors if halfwidth(i) > T_MULT * epsilon_round(r)]
        while not requested and r < MAX_ROUNDS:
            r += 1
            requested = [i for i in survivors if halfwidth(i) > T_MULT * epsilon_round(r)]
        if not requested:
            record["status"] = "unresolved"
            break
        if not eliminate:
            requested = list(survivors)  # FC_FULL: every candidate is requested in every block, including the clearly inferior ones
        if blocks >= MAX_BLOCKS:
            record["status"] = "unresolved"
            record["cap_hit"] = "blocks"
            break
        record["rounds_reached"] = r
        epsilon = epsilon_round(r)
        threshold = T_MULT * epsilon
        clock = time.perf_counter()
        if design == "fixed":
            designs, plan, info = {i: sel.x0[i] for i in requested}, sel.fixed_plan(requested, cov, "B", epsilon=epsilon), {"status": "fixed"}
        else:
            designs, plan, info = sel.joint(requested, cov, "B", epsilon=epsilon)
        status = info["status"]
        record["solver"][status if not status.startswith("error") else "error"] = record["solver"].get(status if not status.startswith("error") else "error", 0) + 1
        if plan is None:  # freeze b^0 and optimize the counts
            designs = {i: sel.x0[i] for i in requested}
            plan = sel.fixed_plan(requested, cov, "B", epsilon=epsilon)
            record["design_choices"].append("frozen")
        else:
            record["design_choices"].append("program")
        seconds["program"] += time.perf_counter() - clock
        blocks += 1
        # task T16 (what does the prior change?): the median L1 norm of the designs of this block against that of the frozen reference
        # reconstruction, the number of requested candidates and active contexts, so that the range term of the radius can be separated
        # from the covariance term afterwards
        l1_design = [float(sum(np.abs(x).sum() for _, _, x in designs[i])) for i in requested]
        record.setdefault("block_l1", []).append([float(np.median(l1_design)), float(np.median([sel.x0_l1_total[i] for i in requested])),
                                                  len(requested), int((plan > 1e-9).sum())])
        active = np.flatnonzero(plan > 1e-9)
        # The work order's first-checkpoint rule (at most 4096 shots) is below the sum of the floors (32 per context) as soon as more than 128
        # contexts are active (LiH: 933 x 32 = 29,856), so every context ends at the floor and the allocation is uniform.  ``batch_fix`` uses
        # the cap max(4096, 64 L_active) of the extended LiH calculation (protocol CP-1.2-Bfix), which keeps the allocation.
        batch_cap = max(FIRST_CHECK, 64 * len(active)) if batch_fix else FIRST_CHECK
        q = np.zeros(L, dtype=np.int64)
        k = 0
        while True:
            q[:] = 0
            q[active] = np.maximum(FLOOR, np.ceil(plan[active] / 2.0 ** k - 1e-9)).astype(np.int64)
            if q.sum() <= batch_cap or (q[active] == FLOOR).all():
                break
            k += 1
        residuals = {i: sel._residual(i, designs[i]) for i in requested}
        by_context = sel._by_context(designs)
        blk = Counts(L)
        block_total = 0
        ell = 0
        stop_block = False
        while not stop_block:
            ell += 1
            add = q if ell == 1 else q * (2 ** (ell - 2))
            increment = int(add[active].sum())
            if shots_total + increment > cap:
                record["status"] = "unresolved"
                record["cap_hit"] = "shots"
                break
            clock = time.perf_counter()
            draws = rng.multinomial(add[active], sel.dist[active])
            counts.add(active, draws)
            blk.add(active, draws)
            shots_total += increment
            block_total += increment
            seconds["sample"] += time.perf_counter() - clock
            record["checkpoints"] += 1
            clock = time.perf_counter()
            eta = (delta * 0.5) / (M * blocks * (blocks + 1) * ell * (ell + 1))
            moments = fragment_moments(blk, sel.library, by_context)
            for i in requested:
                items = [(a, moments[(i, a)]) for a, _, _ in designs[i]]
                interval = bernstein_interval([m[1][0] for m in items], [m[1][1] for m in items], [m[1][2] for m in items],
                                              [m[1][3] for m in items], eta, len(items), 0.0, residuals[i])
                if not (interval[0] - CONFLICT_TOLERANCE <= sel.truth[i] <= interval[1] + CONFLICT_TOLERANCE):
                    record["block_failures"] += 1
                lo[i], hi[i] = max(lo[i], interval[0]), min(hi[i], interval[1])
            if history is None or history_size != len(survivors):
                history = sel._by_context({i: sel.x0[i] for i in survivors})
                history_size = len(survivors)
            hist = fragment_moments(counts, sel.library, history)
            zeta_base = (delta * 0.5) / (M * L)
            acc_lo = {i: 0.0 for i in survivors}
            acc_hi = {i: 0.0 for i in survivors}
            measured = {i: 0.0 for i in survivors}
            for (i, a), (n_, m_, s2_, l1_) in hist.items():
                flo, fhi = history_fragment(n_, m_, s2_, l1_, zeta_base)
                acc_lo[i] += flo
                acc_hi[i] += fhi
                measured[i] += l1_
            conflict = False
            for i in survivors:
                unmeasured = float(sel.x0_l1_total[i]) - measured[i]
                hlo, hhi = acc_lo[i] - unmeasured, acc_hi[i] + unmeasured
                if not (hlo - CONFLICT_TOLERANCE <= sel.truth[i] <= hhi + CONFLICT_TOLERANCE):
                    record["history_failures"] += 1
                lo[i], hi[i] = max(lo[i], hlo), min(hi[i], hhi)
                if lo[i] > hi[i] + CONFLICT_TOLERANCE:
                    conflict = True
                elif lo[i] > hi[i]:
                    lo[i] = hi[i] = 0.5 * (lo[i] + hi[i])
            seconds["interval"] += time.perf_counter() - clock
            if conflict:
                record["status"] = "interval_conflict"
                break
            best_lower = max(magnitude_interval(lo[i], hi[i])[0] for i in survivors)
            kept = [i for i in survivors if magnitude_interval(lo[i], hi[i])[1] >= best_lower] if eliminate else list(survivors)
            eliminated = len(kept) < len(survivors)
            if eliminated:
                if maximizers & (set(survivors) - set(kept)):
                    record["maximizer_eliminated"] = True
                survivors = kept
                history = None
            decided = certificate()
            if decided is not None:
                stop_block = True
            elif eliminated and len(kept_requested(survivors)) <= restart_ratio * len(requested):
                # new block at the same round with the surviving set.  ``restart_ratio`` = 1: after every elimination (the protocol as specified);
                # 0: never (the block and its doubling continue; the block's design was fixed before its data, so the intervals stay valid);
                # 0.5: only when at most half of the block's requested candidates survive
                stop_block = True
            elif all(halfwidth(i) <= threshold for i in survivors):
                r += 1
                stop_block = True
        record["block_shots"].append(block_total)
        if record["status"] in ("unresolved", "interval_conflict"):
            break
    if record["status"] is None:
        record["status"] = "certified" if decided is not None else "unresolved"
        if decided is not None:
            record["selected_index"] = int(decided)
    elif record["status"] in ("unresolved", "interval_conflict") and survivors:
        mids = {i: abs(0.5 * (lo[i] + hi[i])) for i in survivors}
        record["selected_index"] = int(max(mids, key=mids.get))
    record["selection_shots"] = int(shots_total)
    sel_i = record["selected_index"]
    record["exact_best"] = bool(record["status"] == "certified" and sel_i in maximizers)
    # ``correct``: the generator returned is rho-good, |g_sel| >= (1 - rho) max |g| (at rho = 0 this is the exact best, up to the tie tolerance)
    record["correct"] = bool(record["status"] == "certified" and sel_i is not None and truth_abs[sel_i] >= (1.0 - rho) * top - TIE_TOLERANCE)
    record["rho"] = float(rho)
    record["tied_maximizers"] = len(maximizers)
    record["n_blocks"] = blocks
    record["n_survivors"] = len(survivors)
    record["wall_seconds"] = round(time.perf_counter() - started, 3)
    record.update({f"{k}_seconds": round(v, 3) for k, v in seconds.items()})
    return record
