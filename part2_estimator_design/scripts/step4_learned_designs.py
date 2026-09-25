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
from parallel import parse_shard, run_trials, trial_path, write_trials  # noqa: E402
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
    "II-A flat nu=100": LearningConfig(prior="flat", nu=100.0),
}
# On an HF truth state the HF prior *is* the truth, so it is no test of a prior;
# the flat prior (unit variances, no correlation) takes its place there.
HF_TRUTH_CONFIGS = ["II-0 estimated", "II-0 oracle", "II-A oracle", "II-A data",
                    "II-A data, no guard", "II-A flat nu=100"]


def setup(case: str, strategy: str):
    problem = load_problem(case)
    library = build_context_library(problem, strategy)
    oracle = OracleMoments(library, problem.evaluator.state)
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    base = build_fragment_problems(problem, library, oracle, "II-0")
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    return problem, library, oracle, prior, base, coords


def summary_row(case: str, name: str, outcomes, kept, seconds=None) -> dict:
    result = summarise(outcomes)
    if seconds is not None:
        result["seconds"] = round(seconds, 1)
    result["mean_designs_kept_last_refit"] = float(np.mean(kept))
    return {"case_id": case, "config": name, "label": CONFIGS[name].label, **result}


def print_row(row: dict) -> None:
    print(f"{row['case_id']:30s} {row['config']:26s} shots {row['shots_mean']:12,.0f} +- {row['shots_sem']:9,.0f}  "
          f"median {row['shots_median']:12,.0f}  P90 {row['shots_p90']:12,.0f}  "
          f"correct {row['correct_rate']:.3f} [{row['correct_low']:.3f},{row['correct_high']:.3f}]"
          + (f"  ({row['seconds']:.0f}s)" if "seconds" in row else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD"])
    parser.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS),
                        help="default: all for CISD truth, HF_TRUTH_CONFIGS for HF truth")
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="worker processes (use OMP_NUM_THREADS=1)")
    parser.add_argument("--shard", default="1/1", help="K/N: run every N-th trial starting at K")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    shard = parse_shard(args.shard)

    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, args.strategy)
        names = args.configs or (HF_TRUTH_CONFIGS if problem.state_name == "HF" else
                                 [n for n in CONFIGS if n not in ("II-A flat nu=100",)])
        rows = []
        for name in names:
            model = LearnedM3(problem, library, oracle, prior, base, coords, CONFIGS[name])

            def runner(rng, model=model):
                outcome = model.run(rng)
                return outcome, {"designs_kept_last_refit": model.guard_kept}

            started = time.perf_counter()
            results = run_trials(runner, args.trials, args.seed, workers=args.workers, shard=shard)
            seconds = time.perf_counter() - started
            write_trials(trial_path(args.out, case, "step4", name, shard), name, args.trials, args.seed,
                         [(i, outcome, extra) for i, (outcome, extra) in results])
            row = summary_row(case, name, [o for _, (o, _) in results],
                              [e["designs_kept_last_refit"] for _, (_, e) in results], seconds)
            row["shard"] = args.shard
            rows.append(row)
            print_row(row)
        if shard == (1, 1):
            write_csv(args.out / case / f"{case}_step4_learned_designs.csv", list(rows[0]), rows)
        else:
            print(f"{case}: shard {args.shard} written; run scripts/merge_trials.py when all shards are in")
        write_json(args.out / case / f"{case}_step4_meta.json",
                   {"trials": args.trials, "seed": args.seed, "strategy": args.strategy,
                    "seeding": "trial i uses SeedSequence(seed).spawn(trials)[i]",
                    "configs": {k: CONFIGS[k].label for k in names}, "run": run_record()})


if __name__ == "__main__":
    main()
