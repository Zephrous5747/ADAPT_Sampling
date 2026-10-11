#!/usr/bin/env python3
"""Reproduce Huang and Izmaylov's published savings (arXiv:2509.14917v1, Table I, UCCSD row) with their own rule.

Their setup (read from the paper, Fig. 2 and its text): STO-3G, linear molecules with 1 A between adjacent
atoms (H4, LiH, BeH2), Hartree-Fock reference, UCCSD pool, exact state-vector VQE re-optimisation (L-BFGS-B)
after every added generator, each commutator split into qubit-wise commuting fragments by sorted insertion,
fragment variances taken exactly from the state and a Gaussian measurement model.

* naive baseline: every fragment of every pool gradient is measured to the target precision
  ``eps = 1e-3`` (``M_n = Var_n / eps^2`` measurements per fragment, their Eq. 8), the largest ``|g_hat|`` is chosen;
* successive elimination (SE): ``L = 10`` rounds; in round ``r`` the active gradients' fragments are measured
  to ``eps'_r = (5 - 2r/5) eps`` (so ``eps'_L = eps``), the confidence radius is ``R_r = 8 eps'_r``, and a
  generator goes when ``|g_i| + R_r < max_j |g_j| - R_r`` (their Eq. 3);
* the table's number is the reduction ``1 - N_SE / N_naive`` of the *total* measurements for generator
  selection along the trajectory to chemical accuracy (1.59e-3 Ha), SE and naive each on their own trajectory.

What the paper does not say, and what is therefore a choice here (both are run): whether measurements of
earlier rounds are reused (``--reuse``, the default: a fragment's samples accumulate) or every round measures
afresh (``--no-reuse``); and whether the cost counts ``ceil(Var/eps^2)`` per fragment (used) or the real value.

Published (UCCSD): H4 93.0%, LiH 80.8%, BeH2 90.0%.

    python scripts/reproduce_huang_izmaylov.py --cases H4_chain1p0_HF LiH_R1p0_HF BeH2_R1p0_HF --seeds 20
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from contexts import qwc_groups  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from trajectory import AdaptSystem, problem_at  # noqa: E402

PUBLISHED = {"H4_chain1p0_HF": 93.0, "LiH_R1p0_HF": 80.8, "BeH2_R1p0_HF": 90.0}
CHEMICAL_ACCURACY = 1.59e-3  # their value
EPSILON = 1.0e-3
ROUNDS = 10
RADIUS_FACTOR = 8.0
MAX_ITERATIONS = 40


def precision_factor(r: int) -> float:
    """``c_r`` with ``eps'_r = c_r eps`` for the UCCSD pool: 5 - 2r/5 (equal to 1 in the last round)."""
    return 5.0 - 2.0 * r / 5.0


class Fragments:
    """QWC fragments of every pool gradient (sorted insertion by coefficient) and their statistics on a state."""

    def __init__(self, system: AdaptSystem) -> None:
        problem = system.problem
        self.groups = [[{label: terms[label] for label in group} for group in qwc_groups(list(terms), weights=terms)]
                       for terms in problem.commutator_terms]
        self.system = system
        self._stats: dict[tuple, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    def stats(self, key: tuple, psi: np.ndarray, gradients: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray]:
        """Means and standard deviations of the fragments of one arm on the state ``key`` (cached)."""
        cache = self._stats.setdefault(key, {})
        if "problem" not in cache:
            full = self.system.full_state(psi)
            cache["problem"] = problem_at(self.system.problem, full, f"{self.system.base_case}_hi{len(key)}",
                                          energy=self.system.energy(psi), sector_gradients=gradients)
        if arm not in cache:
            ev = cache["problem"].evaluator
            pairs = [ev.fragment_mean_std(group) for group in self.groups[arm]]
            cache[arm] = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))
        return cache[arm]


def naive_selection(frag: Fragments, key, psi, gradients, rng) -> tuple[int, float]:
    """Every fragment of every gradient to precision eps; returns the choice and the measurements spent."""
    total = 0.0
    estimates = np.zeros(len(frag.groups))
    for arm in range(len(frag.groups)):
        mean, std = frag.stats(key, psi, gradients, arm)
        m = np.ceil(std ** 2 / EPSILON ** 2)
        total += float(m.sum())
        noise = np.where(m > 0, std / np.sqrt(np.maximum(m, 1.0)) * rng.standard_normal(mean.size), 0.0)
        estimates[arm] = float((mean + noise).sum())
    return int(np.argmax(np.abs(estimates))), total


def se_selection(frag: Fragments, key, psi, gradients, rng, reuse: bool) -> tuple[int, float, int]:
    """Successive elimination with the published schedule; returns the choice, the measurements and the rounds used."""
    n = len(frag.groups)
    active = list(range(n))
    have = {arm: np.zeros(frag.stats(key, psi, gradients, arm)[0].size) for arm in active}
    sums = {arm: np.zeros_like(have[arm]) for arm in active}
    estimates = np.zeros(n)
    total = 0.0
    rounds = 0
    for r in range(1, ROUNDS + 1):
        rounds = r
        eps_r = precision_factor(r) * EPSILON
        for arm in active:
            mean, std = frag.stats(key, psi, gradients, arm)
            target = np.ceil(std ** 2 / eps_r ** 2)
            if reuse:
                new = np.maximum(target - have[arm], 0.0)
                total += float(new.sum())
                add = mean * new + std * np.sqrt(new) * rng.standard_normal(mean.size)
                sums[arm] += add
                have[arm] += new
                count = np.maximum(have[arm], 1.0)
                estimates[arm] = float((np.where(have[arm] > 0, sums[arm] / count, mean)).sum())
            else:
                total += float(target.sum())
                noise = np.where(target > 0, std / np.sqrt(np.maximum(target, 1.0)) * rng.standard_normal(mean.size), 0.0)
                estimates[arm] = float((mean + noise).sum())
        radius = RADIUS_FACTOR * eps_r
        best = max(abs(estimates[i]) for i in active)
        active = [i for i in active if not abs(estimates[i]) + radius < best - radius]
        if len(active) <= 1:
            break
    return max(active, key=lambda i: abs(estimates[i])), total, rounds


def trajectory(system: AdaptSystem, frag: Fragments, rule: str, rng, reuse: bool, states: dict) -> dict:
    ops: list[int] = []
    params = np.zeros(0)
    spent = 0.0
    errors = []
    for k in range(MAX_ITERATIONS + 1):
        key = tuple(ops)
        if key not in states:
            psi = system.state(ops, params)
            states[key] = (psi, system.gradients(psi), system.energy(psi) - system.fci_energy)
        psi, gradients, error = states[key]
        errors.append(error)
        if error < CHEMICAL_ACCURACY or k == MAX_ITERATIONS:
            break
        if rule == "naive":
            chosen, cost = naive_selection(frag, key, psi, gradients, rng)
        else:
            chosen, cost, _ = se_selection(frag, key, psi, gradients, rng, reuse)
        spent += cost
        ops.append(int(chosen))
        params, _ = system.optimise(ops, np.append(params, 0.0))
        if tuple(ops) not in states:
            pass
    return {"iterations": len(errors) - 1, "measurements": spent, "reached": errors[-1] < CHEMICAL_ACCURACY, "final_error": errors[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(PUBLISHED))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--no-reuse", action="store_true", help="every SE round measures afresh (no accumulation of samples)")
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a"))
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        started = time.perf_counter()
        system = AdaptSystem.build(case)
        frag = Fragments(system)
        sizes = [len(g) for g in frag.groups]
        print(f"{case}: {len(frag.groups)} UCCSD generators, {sum(sizes)} QWC fragments ({np.mean(sizes):.1f} per generator), "
              f"sector {system.metadata['sector_dimension']} (setup {time.perf_counter() - started:.0f}s)", flush=True)
        states: dict = {}
        results = {"naive": [], "SE": []}
        for seed in range(args.seeds):
            for rule in ("naive", "SE"):
                rng = np.random.default_rng([seed, 0 if rule == "naive" else 1])
                results[rule].append(trajectory(system, frag, rule, rng, not args.no_reuse, states))
        naive = np.array([r["measurements"] for r in results["naive"]])
        se = np.array([r["measurements"] for r in results["SE"]])
        reduction = 100.0 * (1.0 - se / naive.mean())
        row = {"case": case, "reuse": not args.no_reuse, "seeds": args.seeds, "published_reduction_pct": PUBLISHED.get(case, float("nan")),
               "reduction_pct_mean": float(1e2 * (1.0 - se.mean() / naive.mean())), "reduction_pct_sd": float(reduction.std(ddof=1)),
               "naive_measurements_mean": float(naive.mean()), "se_measurements_mean": float(se.mean()),
               "naive_iterations": float(np.mean([r["iterations"] for r in results["naive"]])),
               "se_iterations": float(np.mean([r["iterations"] for r in results["SE"]])),
               "naive_reached": float(np.mean([r["reached"] for r in results["naive"]])),
               "se_reached": float(np.mean([r["reached"] for r in results["SE"]])),
               "exact_adapt_iterations": None}
        rows.append(row)
        print(f"{case:16s} reduction {row['reduction_pct_mean']:5.1f}% (sd {row['reduction_pct_sd']:.1f}), published {row['published_reduction_pct']:.1f}%; "
              f"naive {row['naive_measurements_mean']:.3e} in {row['naive_iterations']:.1f} steps, SE {row['se_measurements_mean']:.3e} in {row['se_iterations']:.1f} steps "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "noreuse" if args.no_reuse else "reuse"
    write_csv(args.out / f"huang_izmaylov_reproduction_{tag}.csv", list(rows[0]), rows)
    write_json(args.out / f"huang_izmaylov_reproduction_{tag}_meta.json", {"epsilon": EPSILON, "rounds": ROUNDS, "radius_factor": RADIUS_FACTOR,
               "chemical_accuracy": CHEMICAL_ACCURACY, "published": PUBLISHED, "run": run_record()})


if __name__ == "__main__":
    main()
