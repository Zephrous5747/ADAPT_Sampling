#!/usr/bin/env python3
"""Step 5: the external baselines run the way Part II's own methods run.

Fixed state, sampled measurement outcomes, variances estimated from the shots (no
exact variance in any row except ``M1 static``, which is handed the exact gap and the
exact variances as in Part I).  Configurations:

* ``M1 static``: one allocation for the whole pool at radius ``gap / 2``, one draw;
* ``M1 seq``: shared groups, uniform precision for every gradient, shrinking radius,
  elimination switched off (``LearnedM3(elimination="off")``): the sequential,
  non-oracle form of all-gradient estimation, with II-0's rule and estimated radii;
* ``M2 ...``: independent best-arm identification (:class:`baselines.IndependentBAI`):
  ``marginal`` is Successive Elimination on ``|g|`` as in Huang and Izmaylov's
  formulation, ``pairwise``/``safe`` are the rules of II-0 on independent estimates;
* ``Ikh ...``: the shot-reuse baseline of Ikhtiarudin et al. (:mod:`reuse`): all-gradient
  estimation (no elimination) in which every Pauli measured by the energy contexts of the
  last VQE evaluation is read from them, those shots being free; ``FC`` or ``QWC`` groups.
  ``... no reuse`` rows are the same grouping without the free data;
* ``II-0 + reuse`` and ``II-A data + reuse``: Part II's methods given the same free data
  (on the same library, so II-A may split coefficients between energy and parent contexts).

The rows named ``... bound start`` use the a-priori starting radius, i.e. no exact
quantity enters the run.  II-0 and II-A rows to compare with are written by
``step4_learned_designs.py`` (``II-0 estimated``, ``II-A data``, ``II-0 safe, bound
start``, ``II-A data, safe``) into the same trial directories.

Example::

    OMP_NUM_THREADS=1 python scripts/step5_external_baselines.py --cases H4_square_eq_side1p0_CISD \\
        --trials 200 --workers 12
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from baselines import IndependentBAI, IndependentConfig, IndependentContexts, StaticM1  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from learning import LearnedM3, LearningConfig  # noqa: E402
from merge_trials import merge_case  # noqa: E402
from online import summarise  # noqa: E402
from outputs import run_record, write_json  # noqa: E402
from parallel import (  # noqa: E402
    Checkpoint,
    completed_trials,
    parse_shard,
    run_trials,
    trial_path,
    write_trials,
)
from reuse import ReuseLibrary, ReuseSpec, hamiltonian_terms, reuse_fragment_problems  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from states import hartree_fock_state  # noqa: E402  (Part I)
from step4_learned_designs import setup  # noqa: E402


Reuse = ReuseSpec


M1_SEQ = LearningConfig(level="II-0", prior="none", nu=0.0, elimination="off")
II_0 = LearningConfig(level="II-0", prior="none", nu=0.0)
II_A = LearningConfig(prior="none", nu=0.0)
# The sign-aware family: the pairwise rule's plug-in signs can eliminate the best arm while a gradient is near zero (Sec. Q3), and the
# free data of the reuse rows made that happen in 7.5% of LiH trials; these rows use the rule that is safe on the good event.
II_0_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", radius_min_shots=50)
II_A_SAFE = LearningConfig(prior="none", nu=0.0, rule="safe", radius_min_shots=50)
M1_SEQ_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", radius_min_shots=50, elimination="off")

CONFIGS = {
    # Part II's own methods, so that a run on another pool (``<case>@<pool>``) is self-contained.
    "II-0 estimated": II_0,
    "II-A data": II_A,
    "M1 static": "static",
    "M1 seq": M1_SEQ,
    "M1 seq, safe, bound start": LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", start="bound",
                                                radius_min_shots=50, elimination="off"),
    "M2 marginal": IndependentConfig(rule="marginal", start="oracle"),
    "M2 pairwise": IndependentConfig(rule="pairwise", start="oracle"),
    "M2 safe": IndependentConfig(rule="safe", start="oracle"),
    "M2 safe, bound start": IndependentConfig(rule="safe", start="bound"),
    "M2 marginal, bound start": IndependentConfig(rule="marginal", start="bound"),
    "Ikh reuse, FC": Reuse(M1_SEQ, "fc"),
    "Ikh reuse, QWC": Reuse(M1_SEQ, "qwc"),
    "Ikh no reuse, QWC": Reuse(M1_SEQ, "qwc", free_data=False),
    "II-0 + reuse, FC": Reuse(II_0, "fc"),
    "II-A data + reuse, FC": Reuse(II_A, "fc"),
    # sign-aware family (oracle start)
    "II-A data, safe": II_A_SAFE,
    "M1 seq, safe": M1_SEQ_SAFE,
    "Ikh reuse, FC, safe": Reuse(M1_SEQ_SAFE, "fc"),
    "II-0 safe + reuse, FC": Reuse(II_0_SAFE, "fc"),
    "II-A data, safe + reuse, FC": Reuse(II_A_SAFE, "fc"),
    # II-A starting from the home design, free data available as extra coordinates (the splitting decides)
    "II-A data + reuse (home), FC": Reuse(II_A, "fc", True, "home"),
    "II-A data, safe + reuse (home), FC": Reuse(II_A_SAFE, "fc", True, "home"),
    "II-A data, safe, FC library, no data": Reuse(II_A_SAFE, "fc", False, "home"),
}
DEFAULT = [n for n, c in CONFIGS.items() if not isinstance(c, Reuse)
           and n not in ("II-0 estimated", "II-A data", "II-A data, safe", "M1 seq, safe")]
POOL_CONFIGS = ["II-0 estimated", "II-A data", "M1 static", "M1 seq", "M2 pairwise", "M2 marginal"]
REUSE_NAMES = [n for n, c in CONFIGS.items() if isinstance(c, Reuse)]


def reuse_setup(problem, case: str, grouping: str, strategy: str, energy_error: float):
    """Library with the energy contexts, the free data, and the designs on it."""
    terms = hamiltonian_terms(problem, case)
    rl = ReuseLibrary(problem, terms, strategy, grouping)
    library = rl.library
    oracle = OracleMoments(library, problem.evaluator.state)
    credit = rl.credit(oracle, energy_error)
    base = reuse_fragment_problems(problem, rl, oracle, credit)
    base_home = build_fragment_problems(problem, library, oracle, "II-0")  # plain home assignment
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    return {"library": library, "oracle": oracle, "prior": prior, "base": base, "base_home": base_home, "coords": coords,
            "credit": credit, "overlap": rl.overlap(problem), "energy_groups": len(rl.energy)}


def with_rho(config, rho: float):
    """The configuration stopping at a ``rho``-good selection (``rho = 0``: exact identification)."""
    if rho <= 0 or config == "static":
        return config
    if isinstance(config, Reuse):
        return dataclasses.replace(config, learning=dataclasses.replace(config.learning, rho=rho))
    return dataclasses.replace(config, rho=rho)


def build_model(name, problem, case, strategy, shared, cache, energy_error, rho=0.0):
    config = with_rho(CONFIGS[name], rho)
    library, oracle, prior, base, coords = shared
    if config == "static":
        design = DesignSet(build_fragment_problems(problem, library, oracle, "II-0"), library.n_contexts)
        radius = rho * float(problem.abs_gradients.max()) / 2.0 if rho > 0 else problem.top_gap() / 2.0
        return StaticM1(problem, library, oracle, design, radius=radius)
    if isinstance(config, LearningConfig):
        return LearnedM3(problem, library, oracle, prior, base, coords, config)
    if isinstance(config, IndependentConfig):
        if "independent" not in cache:
            started = time.perf_counter()
            cache["independent"] = IndependentContexts(problem, progress=True)
            cache["independent"].set_state(problem.evaluator.state)
            print(f"{case}: {cache['independent'].n_contexts} independent groups built in "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
        return IndependentBAI(problem, cache["independent"], config)
    key = ("reuse", config.grouping)
    if key not in cache:
        started = time.perf_counter()
        cache[key] = reuse_setup(problem, case, config.grouping, strategy, energy_error)
        info = cache[key]
        print(f"{case}: {config.grouping} reuse library: {info['library'].n_contexts} contexts, "
              f"{info['energy_groups']} energy groups, credit {info['credit'].sum():,} shots, overlap "
              f"{info['overlap']['covered_fraction']:.2f} of the support / "
              f"{info['overlap']['covered_mass_fraction']:.2f} of the coefficient mass "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    info = cache[key]
    credit = info["credit"] if config.free_data else None
    # "no reuse" is plain grouping (home assignment); "home" lets the learned splitting use the free data on its own
    base = info["base"] if (config.free_data and config.assign == "fixed") else info["base_home"]
    return LearnedM3(problem, info["library"], info["oracle"], info["prior"], base, info["coords"],
                     config.learning, credit=credit)


def label_of(name: str) -> str:
    config = CONFIGS[name]
    if config == "static":
        return "M1 static (gap/2, exact variances)"
    if isinstance(config, Reuse):
        return (f"{config.learning.label}/{config.grouping}/{'reuse' if config.free_data else 'no reuse'}"
                + (f"/assign={config.assign}" if config.assign != "fixed" else ""))
    return config.label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT, choices=list(CONFIGS))
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--rho", type=float, default=0.0,
                        help="stop at a rho-good selection (|g| >= (1 - rho) max|g|); 0: exact identification. "
                             "Needed on pools with exactly tied leaders; M1 static then uses radius rho max|g| / 2")
    parser.add_argument("--energy-error", type=float, default=1e-3,
                        help="standard error (Ha) of the last energy evaluation whose data the reuse rows hold for free")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="worker processes (use OMP_NUM_THREADS=1)")
    parser.add_argument("--shard", default="1/1", help="K/N: run every N-th trial starting at K")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    shard = parse_shard(args.shard)

    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, args.strategy)
        shared = (library, oracle, prior, base, coords)
        cache: dict = {}
        done_names = []
        for name in args.configs:
            if CONFIGS[name] == "static" and args.rho <= 0 and problem.top_gap() < 1e-9:
                print(f"{case}: {name}: tied leaders, exact identification has no finite cost; skipped", flush=True)
                continue
            label = name if args.rho <= 0 else f"{name}, rho={args.rho:g}"
            path = trial_path(args.out, case, "step5", label, shard)
            finished = completed_trials(path, args.trials, shard) if args.resume else None
            if finished is not None:
                print(f"{case:30s} {name:28s} already complete in {path.name}; skipped", flush=True)
                continue
            try:
                model = build_model(name, problem, case, args.strategy, shared, cache, args.energy_error, args.rho)
            except ValueError as error:  # e.g. a cached problem whose orbitals do not match a rebuilt H
                print(f"{case}: {name}: SKIPPED: {error}", flush=True)
                continue
            checkpoint = Checkpoint(path)
            if not args.resume:
                checkpoint.clear()
            done = checkpoint.load()
            if done:
                print(f"{case:30s} {name:28s} resuming: {len(done)} trials", flush=True)

            def runner(rng, model=model):
                outcome = model.run(rng)
                return outcome, dict(outcome.extra)

            started = time.perf_counter()
            results = run_trials(runner, args.trials, args.seed, workers=args.workers, shard=shard,
                                 skip=done, on_result=checkpoint.add)
            seconds = time.perf_counter() - started
            results = sorted(list(done.items()) + results, key=lambda item: item[0])
            write_trials(path, label, args.trials, args.seed, [(i, o, e) for i, (o, e) in results])
            checkpoint.clear()
            summary = summarise([o for _, (o, _) in results])
            print(f"{case:30s} {name:28s} shots {summary['shots_mean']:14,.0f} +- {summary['shots_sem']:11,.0f}  "
                  f"median {summary['shots_median']:13,.0f}  correct {summary['correct_rate']:.3f}  "
                  f"({seconds:.0f}s)", flush=True)
            done_names.append(label)
        if shard == (1, 1):
            merge_case(args.out, case, "step5")
        else:
            print(f"{case}: shard {args.shard} written; run scripts/merge_trials.py --step step5 when all shards are in")
        meta_path = args.out / case / f"{case}_step5_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("configs", {}).update({k: label_of(k.split(", rho=")[0]) for k in done_names})
        for key, info in cache.items():
            if isinstance(key, tuple):
                meta.setdefault("reuse_libraries", {})[key[1]] = {
                    "contexts": info["library"].n_contexts, "energy_groups": info["energy_groups"],
                    "credit_shots": int(info["credit"].sum()), "overlap": info["overlap"]}
        meta.update({"trials": args.trials, "seed": args.seed, "strategy": args.strategy, "energy_error": args.energy_error, "rho": args.rho,
                     "seeding": "trial i uses SeedSequence(seed).spawn(trials)[i]", "run": run_record()})
        write_json(meta_path, meta)


if __name__ == "__main__":
    main()
