#!/usr/bin/env python3
"""Phase 3: complete, fully logged, non-oracle generator-selection decisions.

For each case this writes the static record once (``runs/<case>/phase3/static``:
``A``, generator labels, the Pauli library, every context with its Clifford
circuit, and the Hamiltonian when the case carries it) and then runs a few trials
of each configuration with a :class:`runlog.RunLog` attached.  Every trial leaves
six files under ``runs/<case>/phase3/``:

``*_rounds.jsonl``     one line per round: radius, active set, shots added per context
                       and fold, cumulative shots, estimates, standard deviations,
                       absolute intervals, resolved signs, leader, eliminations with the
                       arm and rule that made them, designed contrasts, contexts
                       measured, two-qubit gate counts and classical time;
``*_refits.json``      every redesign: epoch, shots, learnable contexts, designs kept by
                       the guard, arms with split coefficients, prior weight;
``*_designs.npz``      the fold designs of every refit as sparse ``C`` rows
                       (``row``, ``ctx``, ``pauli``, ``x``);
``*_data.npz``         outcome histograms per context and fold (the sufficient
                       statistics) and the covariance matrices of the ten most used
                       contexts (fold models and prior);
``*_summary.json``     the selection, context-shots and timings (no oracle values);
``*_validation.json``  the exact gradients and whether any interval missed them.

Example::

    python scripts/phase3_selection_record.py --cases LiH_R3p0_ADAPT3 --trials 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from learning import LearnedM3  # noqa: E402
from outputs import run_record, write_json  # noqa: E402
from parallel import slug  # noqa: E402
from runlog import RunLog, write_static  # noqa: E402
from step4_learned_designs import CONFIGS, setup  # noqa: E402

DEFAULT_CONFIGS = ["II-0 safe, bound start", "II-A data, contrast, bound start"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["LiH_R3p0_ADAPT3"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true", help="skip trials whose record is already saved")
    args = parser.parse_args()

    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, args.strategy)
        out = args.out / case / "phase3"
        static = write_static(out / "static", problem, library,
                              hamiltonian_terms=problem.metadata.get("hamiltonian_terms"))
        seeds = np.random.SeedSequence(args.seed).spawn(args.trials)
        runs = []
        for name in args.configs:
            learner = LearnedM3(problem, library, oracle, prior, base, coords, CONFIGS[name])
            for trial in range(args.trials):
                if args.resume and (out / f"{slug(name)}_trial{trial}_summary.json").exists():
                    print(f"{case:30s} {name:34s} trial {trial}: already recorded; skipped", flush=True)
                    continue
                log = RunLog()
                outcome = learner.run(np.random.default_rng(seeds[trial]), log=log)
                stem = f"{slug(name)}_trial{trial}"
                log.save(out, stem)
                runs.append({"config": name, "trial": trial, "stem": stem, "selected": outcome.selected,
                             "correct": outcome.correct, "shots": outcome.shots, "rounds": outcome.rounds,
                             "refits": outcome.extra["refits"],
                             "design_seconds": outcome.extra["design_seconds"]})
                print(f"{case:30s} {name:34s} trial {trial}: shots {outcome.shots:12,.0f}  "
                      f"rounds {outcome.rounds:3d}  refits {outcome.extra['refits']:2d}  "
                      f"correct {outcome.correct}  design {outcome.extra['design_seconds']:.1f}s", flush=True)
        write_json(out / "index.json", {"case": case, "static": static, "runs": runs,
                                        "configs": {k: CONFIGS[k].label for k in args.configs},
                                        "seed": args.seed, "run": run_record()})


if __name__ == "__main__":
    main()
