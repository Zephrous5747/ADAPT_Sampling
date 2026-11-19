#!/usr/bin/env python3
"""Check the measurement circuits and context moments of the production libraries against an independent evaluator.

For every context of the shared (SG) library of a stored problem, the sign-aware group means that the Clifford circuit of the
context yields (``OracleMoments.means``: apply the circuit to the state, read the outcome distribution, take the Walsh-Hadamard
transform) are compared with the expectation value of the same Pauli product computed directly on the state vector
(``problem.evaluator.expectation``).  A wrong circuit, sign or ordering shows up as a deviation of order one.

    python scripts/check_context_moments.py LiH_R1p6_ADAPT3 [--stride 1]

Prints the number of contexts and products checked and the largest absolute deviation; exits with an error above ``--tol``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import part1_bridge  # noqa: E402,F401  (puts Part I's src on the path and registers the extra and window cases)
from contexts import build_context_library  # noqa: E402
from sampler import OracleMoments  # noqa: E402


def check(case_id: str, stride: int = 1, random_state: int | None = None) -> tuple[int, int, int, float]:
    import numpy as np
    from pauli_ops import PauliEvaluator

    problem = part1_bridge.load_problem(case_id)
    library = build_context_library(problem, "canonical")
    evaluator = problem.evaluator
    if random_state is not None:  # a generic complex state on all 2**n basis states instead of the stored one
        rng = np.random.default_rng(random_state)
        vector = rng.normal(size=evaluator.dim) + 1j * rng.normal(size=evaluator.dim)
        evaluator = PauliEvaluator(vector / np.linalg.norm(vector))
    moments = OracleMoments(library, evaluator.state)
    worst, n_contexts, n_products = 0.0, 0, 0
    for context in library.contexts[::stride]:
        means = moments.means(context.index, context.members)
        for member, mean in zip(context.members, means):
            worst = max(worst, abs(float(mean) - float(evaluator.expectation(library.labels[member]))))
        n_contexts += 1
        n_products += len(context.members)
    return library.n_contexts, n_contexts, n_products, worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--stride", type=int, default=1, help="check every stride-th context (1: all)")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--random-state", type=int, default=None, help="seed of a random complex state used instead of the stored one")
    args = parser.parse_args()
    failed = False
    for case in args.cases:
        started = time.perf_counter()
        total, checked, products, worst = check(case, args.stride, args.random_state)
        print(f"{case}: {checked} of {total} contexts, {products} products, max |deviation| {worst:.2e} ({time.perf_counter() - started:.0f} s)", flush=True)
        failed |= worst > args.tol
    sys.exit(1 if failed else 0)
