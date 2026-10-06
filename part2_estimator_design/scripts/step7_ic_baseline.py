#!/usr/bin/env python3
"""Step 7: an informationally complete (IC) baseline in the spirit of AIM-ADAPT-VQE.

One IC shot (a product of single-qubit tetrahedral POVMs on all qubits) informs every
gradient at once, so there is nothing to allocate: the cost of a selection is the number of
IC shots until the elimination rule certifies the decision.  Configurations:

* ``IC oracle``: radii from the exact covariance of the estimators (favourable to IC);
* ``IC estimated``: radii from the sample covariance of the same shots;
* ``... + free energy data``: the shots of an IC energy estimate to ``--energy-error`` (the
  data AIM-ADAPT already holds from the VQE step) are not charged.

The POVM is the *unoptimised* SIC product POVM; the paper's adaptive optimisation of the POVM
parameters (which lowers the energy variance) is not modelled, so these rows are an upper
bound on what the optimised IC scheme needs for the gradients.  IC shots and context-shots are
different resources (an IC shot needs ancilla-assisted or randomised measurements): compare
statistical efficiency, not circuit cost.  Exact simulation needs ``K 4^n`` memory (H4: tiny,
LiH: 6 GB, H2O: out of reach).

    python scripts/step7_ic_baseline.py --cases H4_square_eq_side1p0_CISD --trials 200 --rho 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from ic import ICConfig, ICMeasurement, ICSelection  # noqa: E402
from merge_trials import merge_case  # noqa: E402
from online import summarise  # noqa: E402
from outputs import run_record, write_json  # noqa: E402
from parallel import Checkpoint, completed_trials, parse_shard, run_trials, trial_path, write_trials  # noqa: E402
from part1_bridge import load_problem  # noqa: E402
from reuse import ENERGY_ERROR, hamiltonian_terms  # noqa: E402

CONFIGS = {
    "IC oracle": dict(radii="oracle", free=False),
    "IC estimated": dict(radii="estimated", free=False),
    "IC oracle + free energy data": dict(radii="oracle", free=True),
    "IC estimated + free energy data": dict(radii="estimated", free=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--rule", default="safe", choices=["marginal", "pairwise", "safe"])
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--anytime", action="store_true",
                        help="spend delta_r = 6 delta / (pi^2 r^2) per round (valid over all looks); IC estimators are "
                             "heavy-tailed and the per-round normal intervals miss often")
    parser.add_argument("--energy-error", type=float, default=ENERGY_ERROR)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", default="1/1")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    shard = parse_shard(args.shard)

    for case in args.cases:
        problem = load_problem(case)
        started = time.perf_counter()
        terms = None
        try:
            terms = hamiltonian_terms(problem, case)
        except ValueError as error:
            print(f"{case}: no free-energy-data rows: {error}", flush=True)
        measurement = ICMeasurement(problem, hamiltonian_terms=terms)
        credit = measurement.energy_shots(args.energy_error) if terms is not None else 0.0
        print(f"{case}: IC data built in {time.perf_counter() - started:.0f}s; per-shot variance of the leading gradient "
              f"{measurement.covariance[problem.abs_gradients.argmax()].max():.3g}; energy at {args.energy_error:g} Ha "
              f"takes {credit:,.0f} IC shots", flush=True)
        names = []
        for name in args.configs:
            spec = CONFIGS[name]
            if spec["free"] and terms is None:
                continue
            config = ICConfig(rule=args.rule, rho=args.rho, radii=spec["radii"],
                              credit=credit if spec["free"] else 0.0, anytime=args.anytime)
            label = (f"{name}, rule={args.rule}" + (", anytime" if args.anytime else "")
                     + (f", rho={args.rho:g}" if args.rho > 0 else ""))
            path = trial_path(args.out, case, "step7", label, shard)
            if args.resume and completed_trials(path, args.trials, shard) is not None:
                print(f"{case} {label}: already complete; skipped", flush=True)
                continue
            model = ICSelection(measurement, config)
            checkpoint = Checkpoint(path)
            if not args.resume:
                checkpoint.clear()
            done = checkpoint.load()

            def runner(rng, model=model):
                outcome = model.run(rng)
                return outcome, dict(outcome.extra)

            clock = time.perf_counter()
            results = run_trials(runner, args.trials, args.seed, workers=args.workers, shard=shard,
                                 skip=done, on_result=checkpoint.add)
            results = sorted(list(done.items()) + results, key=lambda item: item[0])
            write_trials(path, label, args.trials, args.seed, [(i, o, e) for i, (o, e) in results])
            checkpoint.clear()
            summary = summarise([o for _, (o, _) in results])
            print(f"{case:30s} {label:52s} IC shots {summary['shots_mean']:14,.0f} +- {summary['shots_sem']:11,.0f}  "
                  f"median {summary['shots_median']:13,.0f}  correct {summary['correct_rate']:.3f}  "
                  f"({time.perf_counter() - clock:.0f}s)", flush=True)
            names.append(label)
        if shard == (1, 1):
            merge_case(args.out, case, "step7")
        meta_path = args.out / case / f"{case}_step7_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update({"trials": args.trials, "seed": args.seed, "rule": args.rule, "rho": args.rho,
                     "energy_error": args.energy_error, "energy_ic_shots": credit,
                     "povm": "product of tetrahedral (SIC) single-qubit POVMs, unoptimised",
                     "unit": "IC shots (not context-shots)", "run": run_record()})
        meta.setdefault("labels", [])
        meta["labels"] = sorted(set(meta["labels"]) | set(names))
        write_json(meta_path, meta)


if __name__ == "__main__":
    main()
