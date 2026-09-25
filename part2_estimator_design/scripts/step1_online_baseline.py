#!/usr/bin/env python3
"""Step 1: the II-0 baseline as a shot-level online experiment.

Three things per case.

1. **Noiseless gate.**  With every estimate replaced by the exact gradient, the
   online loop is deterministic and must reproduce Part I's own trial loop run
   with its noise draws zeroed (``finite_shot._sequential_trial``), up to Part II's
   integer shot counts.
2. **Noisy gate.**  With shots sampled as measurement outcomes, the realised cost
   and correct-selection rate must agree within Monte Carlo error with Part I's
   Gaussian surrogate at the same settings (correlated noise, coarse schedule
   ``gamma = 0.5``, marginal and pairwise rules), rerun here on the same problems.
3. **The Part II baseline (II-0).**  Pairwise elimination on actual precision, top-up
   allocation and a fine schedule, with shots sampled as outcomes.  This is what
   every later level is measured against.

Example::

    python scripts/step1_online_baseline.py --cases H4_square_eq_side1p0_HF --trials 250
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
import finite_shot  # noqa: E402  (Part I)
from contexts import build_context_library  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from online import OnlineConfig, OnlineM3, summarise  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from parallel import parse_shard, run_trials, trial_path, write_trials  # noqa: E402
from part1_bridge import DEFAULT_DELTA, load_problem, z_from_delta  # noqa: E402
from sampler import OracleMoments  # noqa: E402


class _ZeroNoise:
    """Stands in for a NumPy generator in Part I's trial loop: every draw is zero."""

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.zeros(size if size is not None else np.shape(scale))

    def standard_normal(self, size=None):
        return np.zeros(size)


def part1_noiseless(problem, shrink: float, rule: str, covariances) -> float:
    sigmas = problem.parent_fragment_sigmas()
    covariances = covariances if rule == "pairwise" else None
    z = z_from_delta(DEFAULT_DELTA, problem.n_generators)
    outcome = finite_shot._sequential_trial(
        problem, None, sigmas, z, shrink, 0.0, _ZeroNoise(), True,
        finite_shot._AllocationCache(sigmas), False, None, covariances,
    )
    return outcome.shots


def run_config(model: OnlineM3, label: str, case: str, args, shard=(1, 1)) -> dict:
    """Run one configuration's trials (parallel, per-trial seeds) and keep every trial."""
    started = time.perf_counter()
    results = run_trials(model.run, args.trials, args.seed, workers=args.workers, shard=shard)
    write_trials(trial_path(args.out, case, "step1", label, shard), label, args.trials, args.seed,
                 [(i, outcome, {}) for i, outcome in results])
    result = summarise([outcome for _, outcome in results])
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=[
        "H4_square_eq_side1p0_HF", "H4_square_eq_side1p0_CISD",
        "H4_square_stretch_side2p0_HF", "H4_square_stretch_side2p0_CISD", "LiH_R3p0_HF"])
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--skip-part1", action="store_true",
                        help="skip rerunning Part I's Gaussian surrogate")
    parser.add_argument("--gates-only", action="store_true",
                        help="run only the (deterministic) noiseless gate")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (use OMP_NUM_THREADS=1)")
    parser.add_argument("--shard", default="1/1",
                        help="K/N: run every N-th trial of the baseline configurations only; "
                             "merge with scripts/merge_trials.py --step step1")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    shard = parse_shard(args.shard)

    for case in args.cases:
        problem = load_problem(case)
        library = build_context_library(problem, "canonical")
        moments = OracleMoments(library, problem.evaluator.state)
        design = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
        rows = []

        def model(**kwargs) -> OnlineM3:
            return OnlineM3(problem, library, moments, design, OnlineConfig(**kwargs))

        # Part I's context covariances refuse problems with too many generator
        # pairs (LiH, H2O), and its simulator then silently falls back to the
        # marginal rule and independent noise.  Compare only what it really runs.
        part1_covariances = finite_shot.context_covariances(problem, problem.parent_fc_groups())
        part1_rules = ("marginal", "pairwise") if part1_covariances is not None else ("marginal",)

        if shard != (1, 1):  # sharded: the baseline trials only; gates need the whole set
            for gamma in args.gammas:
                for accounting in ("maxima", "topup"):
                    label = f"baseline pairwise {accounting} gamma={gamma:g}"
                    result = run_config(model(rule="pairwise", accounting=accounting, shrink=gamma),
                                        label, case, args, shard)
                    print(f"{case:32s} {label} shard {args.shard}: {result['shots_mean']:14,.0f} "
                          f"({result['seconds']:.0f}s)", flush=True)
            continue

        # 1. noiseless gate
        for rule in part1_rules:
            reference = part1_noiseless(problem, 0.5, rule, part1_covariances)
            real = model(rule=rule, sampling="none", integer_shots=False).run(np.random.default_rng(0)).shots
            ours = model(rule=rule, sampling="none").run(np.random.default_rng(0)).shots
            ratio = real / reference
            rows.append({"case_id": case, "check": "noiseless", "rule": rule, "accounting": "maxima",
                         "gamma": 0.5, "part1_shots": reference, "part2_shots_real": real,
                         "part2_shots": ours, "ratio": ratio, "integer_ratio": ours / reference,
                         "passed": abs(ratio - 1) < 1e-6})
            print(f"{case:32s} noiseless {rule:9s} Part I {reference:14,.0f}  Part II real-valued "
                  f"{real:14,.0f} (ratio {ratio:.7f})  integer {ours:14,.0f} (ratio {ours / reference:.5f})",
                  flush=True)
        gate_rows, rows = rows, []
        write_csv(args.out / case / f"{case}_step1_noiseless_gate.csv",
                  list(dict.fromkeys(k for r in gate_rows for k in r)), gate_rows)
        if args.gates_only:
            continue

        # 2. noisy gate against Part I's Gaussian surrogate (correlated where Part I can)
        for rule in ("marginal", "pairwise"):
            ours = run_config(model(rule=rule, shrink=0.5), f"noisy {rule} maxima gamma=0.5", case, args)
            row = {"case_id": case, "check": "noisy", "rule": rule, "accounting": "maxima", "gamma": 0.5,
                   **{f"part2_{k}": v for k, v in ours.items()}}
            if not args.skip_part1 and rule in part1_rules:
                started = time.perf_counter()
                noise = "correlated" if part1_covariances is not None else "independent"
                reference = finite_shot.simulate(
                    problem, "m3", planning_bound=1.0, n_trials=args.trials, seed=0,
                    noise_model=noise, rule=rule, shrink=0.5,
                )
                difference = ours["shots_mean"] - reference.shots_mean
                spread = math.hypot(ours["shots_sem"], reference.shots_sem)
                row.update({"part1_noise_model": noise, "part1_shots_mean": reference.shots_mean,
                            "part1_shots_sem": reference.shots_sem, "part1_correct_rate": reference.correct_rate,
                            "difference_in_sem": difference / spread if spread else float("nan"),
                            "part1_seconds": round(time.perf_counter() - started, 1)})
                row["passed"] = abs(row["difference_in_sem"]) < 3.0
                print(f"{case:32s} noisy     {rule:9s} Part I {reference.shots_mean:14,.0f} +- {reference.shots_sem:10,.0f} "
                      f"({noise}, correct {reference.correct_rate:.3f})  Part II {ours['shots_mean']:14,.0f} +- "
                      f"{ours['shots_sem']:10,.0f} (correct {ours['correct_rate']:.3f})  "
                      f"diff {row['difference_in_sem']:+.2f} sem", flush=True)
            rows.append(row)

        # 3. the Part II baseline
        for gamma in args.gammas:
            for accounting in ("maxima", "topup"):
                result = run_config(model(rule="pairwise", accounting=accounting, shrink=gamma),
                                    f"baseline pairwise {accounting} gamma={gamma:g}", case, args)
                rows.append({"case_id": case, "check": "baseline", "rule": "pairwise",
                             "accounting": accounting, "gamma": gamma,
                             **{f"part2_{k}": v for k, v in result.items()}})
                print(f"{case:32s} baseline  pairwise {accounting:6s} gamma {gamma:<4g} shots "
                      f"{result['shots_mean']:14,.0f} +- {result['shots_sem']:10,.0f}  median "
                      f"{result['shots_median']:14,.0f}  P90 {result['shots_p90']:14,.0f}  correct "
                      f"{result['correct_rate']:.3f} [{result['correct_low']:.3f}, {result['correct_high']:.3f}]  "
                      f"({result['seconds']:.0f}s)", flush=True)

        columns = list(dict.fromkeys(k for r in rows for k in r))
        write_csv(args.out / case / f"{case}_step1_online_baseline.csv", columns, rows)
        write_json(args.out / case / f"{case}_step1_meta.json",
                   {"trials": args.trials, "seed": args.seed,
                    "seeding": "trial i uses SeedSequence(seed).spawn(trials)[i]", "run": run_record()})


if __name__ == "__main__":
    main()
