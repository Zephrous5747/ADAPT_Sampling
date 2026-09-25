#!/usr/bin/env python3
"""Step 4: coefficient splitting with covariances learned from the shots.

Truth is the CISD state; the classical prior is the Hartree-Fock determinant of
the same problem.  Every configuration uses the pairwise rule, top-up allocation,
gamma = 0.9 and shot-level sampling, and is compared with II-0 run the same way.

Configurations (see ``learning.py``):

* ``II-0 / estimated radii``: the non-oracle baseline;
* ``II-0 / oracle radii``: the Step 1 baseline;
* ``II-A / oracle covariance``: the online ceiling (oracle radii);
* ``II-A / HF prior only`` (nu = inf), with estimated and with oracle radii (the
  guard cannot act here: both folds see the same prior);
* ``II-A / data only`` (nu = 0), with and without the guard;
* ``II-A / HF prior shrunk to data`` (nu = 100, 1000).

Example::

    python scripts/step4_learned_designs.py --cases H4_square_eq_side1p0_CISD --trials 200
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
from contexts import build_context_library  # noqa: E402
from design import build_fragment_problems  # noqa: E402
from learning import LearnedM3, LearningConfig  # noqa: E402
from online import summarise  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import load_problem  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from states import hartree_fock_state  # noqa: E402  (Part I)

INF = math.inf
CONFIGS = {
    "II-0 estimated": LearningConfig(level="II-0", prior="none", nu=0.0),
    "II-0 oracle": LearningConfig(level="II-0", radii="oracle"),
    "II-A oracle": LearningConfig(prior="oracle", nu=INF, radii="oracle"),
    "II-A HF prior": LearningConfig(prior="hf", nu=INF),
    "II-A HF prior, oracle radii": LearningConfig(prior="hf", nu=INF, radii="oracle"),
    "II-A data": LearningConfig(prior="none", nu=0.0),
    "II-A data, no guard": LearningConfig(prior="none", nu=0.0, guard=False),
    "II-A HF nu=100": LearningConfig(prior="hf", nu=100.0),
    "II-A HF nu=1000": LearningConfig(prior="hf", nu=1000.0),
}


def setup(case: str, strategy: str):
    problem = load_problem(case)
    library = build_context_library(problem, strategy)
    oracle = OracleMoments(library, problem.evaluator.state)
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    base = build_fragment_problems(problem, library, oracle, "II-0")
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    return problem, library, oracle, prior, base, coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, args.strategy)
        rows = []
        for name in args.configs:
            config = CONFIGS[name]
            model = LearnedM3(problem, library, oracle, prior, base, coords, config)
            rng = np.random.default_rng(args.seed)
            started = time.perf_counter()
            outcomes, kept = [], []
            for _ in range(args.trials):
                outcomes.append(model.run(rng))
                kept.append(model.guard_kept)
            result = summarise(outcomes)
            result["seconds"] = round(time.perf_counter() - started, 1)
            result["mean_designs_kept_last_refit"] = float(np.mean(kept))
            rows.append({"case_id": case, "config": name, "label": config.label, **result})
            print(f"{case:30s} {name:26s} shots {result['shots_mean']:12,.0f} +- {result['shots_sem']:9,.0f}  "
                  f"median {result['shots_median']:12,.0f}  P90 {result['shots_p90']:12,.0f}  "
                  f"correct {result['correct_rate']:.3f} [{result['correct_low']:.3f},{result['correct_high']:.3f}]  "
                  f"({result['seconds']:.0f}s)", flush=True)
        write_csv(args.out / case / f"{case}_step4_learned_designs.csv", list(rows[0]), rows)
        write_json(args.out / case / f"{case}_step4_meta.json",
                   {"trials": args.trials, "seed": args.seed, "strategy": args.strategy,
                    "configs": {k: CONFIGS[k].label for k in args.configs}, "run": run_record()})


if __name__ == "__main__":
    main()
