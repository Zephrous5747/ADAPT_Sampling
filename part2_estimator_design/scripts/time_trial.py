#!/usr/bin/env python3
"""Time one Step 4 trial per configuration and report peak memory (job sizing).

Example::

    python scripts/time_trial.py --case H2O_eq_CISD --configs "II-0 estimated" "II-A data"
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from step4_learned_designs import CONFIGS, setup  # noqa: E402
from learning import LearnedM3  # noqa: E402


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--configs", nargs="+", default=["II-0 estimated", "II-A data"])
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    problem, library, oracle, prior, base, coords = setup(args.case, "mass")
    print(f"{args.case}: setup {time.perf_counter() - started:.0f}s, peak {peak_gib():.2f} GiB", flush=True)
    for name in args.configs:
        model = LearnedM3(problem, library, oracle, prior, base, coords, CONFIGS[name])
        started = time.perf_counter()
        outcome = model.run(np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0]))
        print(f"{args.case}: {name}: {time.perf_counter() - started:.0f}s per trial, rounds {outcome.rounds}, "
              f"shots {outcome.shots:,.0f}, correct {outcome.correct}, peak {peak_gib():.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
