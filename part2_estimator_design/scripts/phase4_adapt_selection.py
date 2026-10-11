#!/usr/bin/env python3
"""Phase 4: ADAPT-VQE trajectories whose generator selections are measured.

At every ADAPT step the generator is chosen by a Part II method from sampled
measurement outcomes on the current state (:class:`learning.LearnedM3`), and the
ansatz grows with that choice; the parameters are then re-optimised exactly.
Only the gradient-selection shots are counted: energy estimation during the VQE
optimisation is outside the scope of Part II and is done in exact arithmetic.

Every method is fully non-oracle inside a selection (estimated radii, ``bound``
start) and stops a selection as soon as its leader is certified ``rho``-good,
since exact ties between spin-partner generators are common along a trajectory
and exact best-arm identification cannot resolve them.  The driver's own stopping
rule is not measured: a trajectory ends when the energy is within the target of
FCI, or when every exact gradient is below 1e-6 (ADAPT has converged; stretched
H4 stalls this way at 3.2 mHa), or after ``len(exact) + 10`` steps.

Outputs under ``runs/<case>/phase4/``: one row per trajectory, one row per
selection, and a summary per method with the exact-ADAPT reference.

Long trajectories (H2O with the reuse library: up to about 30 h each) run in stages that each
fit a 24 h job.  With ``--resume`` every finished selection is checkpointed (the chosen
generator, its diagnostics and the random-number state), and a restarted job replays the recorded
choices (the exact VQE re-optimisation is deterministic) before it selects live.  With
``--max-selections N`` a job stops each trajectory after N live selections and writes no tables;
the next job, started with the same command, continues, and the job without the limit finishes.
``--max-hours H`` does the same by the clock: a trajectory starts no selection that, at twice the duration of
its previous one, would end after H hours of the job (selections of the contrast design take hours, and a
growing number of survivors makes later ones slower); the stage that finishes every trajectory writes the tables.

Example::

    OMP_NUM_THREADS=1 python scripts/phase4_adapt_selection.py --cases H4_square_eq_side1p0_HF \\
        --trajectories 50 --workers 12
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
from baselines import (FixedBudgetM1, FixedBudgetPilotM1, FixedBudgetSpec, IndependentBAI, IndependentConfig,  # noqa: E402
                       IndependentContexts, StaticM1)
from contexts import build_context_library  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from learning import LearnedM3, LearningConfig  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from parallel import run_trials, slug  # noqa: E402
from reuse import ENERGY_ERROR, ReuseLibrary, ReuseSpec, hamiltonian_terms, reuse_fragment_problems  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, problem_at, run_adapt  # noqa: E402

II_0_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", start="bound", radius_min_shots=50)
II_A_SAFE = LearningConfig(prior="none", nu=0.0, rule="safe", start="bound", radius_min_shots=50)
M1_SEQ_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", start="bound",
                             radius_min_shots=50, elimination="off")
METHODS = {
    "II-0 safe": II_0_SAFE,
    "II-A data, safe": II_A_SAFE,
    "II-A data, contrast": LearningConfig(prior="none", nu=0.0, rule="safe", objective="contrast",
                                          start="bound", radius_min_shots=50),
    # External baselines (Paper A, SOTA comparison): run with ``--methods``.
    "M1 seq, safe": M1_SEQ_SAFE,
    "M1 static": "static",  # radius rho max|g| / 2, exact gradient scale and variances (an oracle bound)
    "M2 safe": IndependentConfig(rule="safe", start="bound"),
    "M2 marginal": IndependentConfig(rule="marginal", start="bound"),
    "Ikh reuse, FC": ReuseSpec(M1_SEQ_SAFE, "fc"),
    "II-0 safe + reuse, FC": ReuseSpec(II_0_SAFE, "fc"),
    "II-A data, safe + reuse, FC": ReuseSpec(II_A_SAFE, "fc"),
}
# The practical default of ADAPT-VQE (a baseline the paper owes): a fixed budget of shots per selection, no
# certificate.  One entry per budget (eighth-decades, 10 ... 1e9; the half-decades of the first runs are among them) and
# allocation, named "Fixed <shots>, <allocation>"; allocation "pilot" estimates the variances from a fifth of the budget.
FIXED_BUDGETS = sorted({int(round(10 ** (e / 8))) for e in range(8, 73)})
for _shots in FIXED_BUDGETS:
    for _allocation in ("designed", "uniform", "pilot"):
        METHODS[f"Fixed {_shots}, {_allocation}"] = FixedBudgetSpec(_shots, _allocation)
DEFAULT_METHODS = ["II-0 safe", "II-A data, safe", "II-A data, contrast"]
# A selection may take this many times as long as the one before it (the contrast design's cost grows with the survivors).
SELECTION_GROWTH = 2.0


def with_rho(spec, rho: float):
    if spec == "static" or isinstance(spec, FixedBudgetSpec):
        return spec
    if isinstance(spec, ReuseSpec):
        return dataclasses.replace(spec, learning=dataclasses.replace(spec.learning, rho=rho))
    return dataclasses.replace(spec, rho=rho)


def label_of(spec) -> str:
    if spec == "static":
        return "M1 static (rho max|g|/2, exact variances)"
    if isinstance(spec, FixedBudgetSpec):
        return spec.label
    if isinstance(spec, ReuseSpec):
        return f"{spec.learning.label}/{spec.grouping}/{'reuse' if spec.free_data else 'no reuse'}"
    return spec.label


class Phase4:
    def __init__(self, case: str, strategy: str, tolerance: float, rho: float = 0.1,
                 energy_error: float = ENERGY_ERROR) -> None:
        self.case = case
        self.tolerance = tolerance
        self.rho = rho
        self.energy_error = energy_error
        self.strategy = strategy
        self._independent = None
        self._reuse: dict = {}
        self.system = AdaptSystem.build(case)
        problem = self.system.problem
        self.library = build_context_library(problem, strategy)
        # The HF determinant supplies the (unused by estimated radii) blocks the
        # coordinate builders need; coordinates themselves do not depend on the state.
        self.prior = OracleMoments(self.library, self.system.full_state(self.system.hf))
        self.base = build_fragment_problems(problem, self.library, self.prior, "II-0")
        split = build_fragment_problems(problem, self.library, self.prior, "II-A")
        self.coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target)))
                       for p in split]
        self.exact = run_adapt(self.system, tolerance=tolerance)
        self.max_iterations = len(self.exact) - 1 + 10

    def _independent_contexts(self) -> IndependentContexts:
        if self._independent is None:
            self._independent = IndependentContexts(self.system.problem)
        return self._independent

    def _reuse_library(self, grouping: str) -> dict:
        if grouping not in self._reuse:
            problem = self.system.problem
            rl = ReuseLibrary(problem, hamiltonian_terms(problem, self.case), self.strategy, grouping)
            hf = OracleMoments(rl.library, self.system.full_state(self.system.hf))
            split = build_fragment_problems(problem, rl.library, hf, "II-A")
            coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target)))
                      for p in split]
            self._reuse[grouping] = {"rl": rl, "prior": hf, "coords": coords}
        return self._reuse[grouping]

    def _select_with(self, spec, problem, full, rng):
        """One generator selection by ``spec`` on the current state."""
        if isinstance(spec, LearningConfig):
            learner = LearnedM3(problem, self.library, OracleMoments(self.library, full), self.prior,
                                self.base, self.coords, spec)
            return learner.run(rng)
        if spec == "static":
            moments = OracleMoments(self.library, full)
            design = DesignSet(build_fragment_problems(problem, self.library, moments, "II-0"),
                               self.library.n_contexts)
            radius = self.rho * float(problem.abs_gradients.max()) / 2.0
            return StaticM1(problem, self.library, moments, design, radius=radius).run(rng)
        if isinstance(spec, FixedBudgetSpec):
            moments = OracleMoments(self.library, full)
            design = DesignSet(build_fragment_problems(problem, self.library, moments, "II-0"),
                               self.library.n_contexts)
            cls = FixedBudgetPilotM1 if spec.allocation == "pilot" else FixedBudgetM1
            return cls(problem, self.library, moments, design, spec).run(rng)
        if isinstance(spec, IndependentConfig):
            contexts = self._independent_contexts()
            contexts.set_state(full)
            return IndependentBAI(problem, contexts, spec).run(rng)
        if isinstance(spec, ReuseSpec):
            info = self._reuse_library(spec.grouping)
            rl = info["rl"]
            moments = OracleMoments(rl.library, full)
            credit = rl.credit(moments, self.energy_error)
            base = reuse_fragment_problems(problem, rl, moments, credit)
            learner = LearnedM3(problem, rl.library, moments, info["prior"], base, info["coords"],
                                spec.learning, credit=credit if spec.free_data else None)
            return learner.run(rng)
        raise TypeError(f"unknown method {spec!r}")

    def trajectory(self, rng: np.random.Generator, config, ckpt_stem: Path | None = None,
                   budget: int | None = None, deadline: float | None = None) -> dict:
        """One trajectory.  ``budget``: live selections this job may run; ``deadline``: wall-clock time (``time.time()``)
        by which the job must be done, so no selection starts that would, at ``SELECTION_GROWTH`` times the length
        of the previous one, end after it."""
        selections = []
        # a trajectory is identified by its seed (the random stream it starts with), so a restarted job finds it again
        ckpt = Path(f"{ckpt_stem}.traj_{_rng_key(rng)}.partial.jsonl") if ckpt_stem is not None else None
        replay = _load_selections(ckpt) if ckpt is not None else []
        live = 0
        last_seconds = 0.0

        def select(k, psi, exact_gradients):
            nonlocal live, last_seconds
            if k < len(replay):  # recorded in an earlier job: same choice, no measurement
                selections.append(replay[k]["info"])
                last_seconds = float(replay[k]["info"].get("seconds", last_seconds))
                return replay[k]["selected"], replay[k]["info"]
            if replay and k == len(replay):
                rng.bit_generator.state = replay[-1]["rng"]
            if budget is not None and live >= budget:
                raise _Paused
            if deadline is not None and time.time() + SELECTION_GROWTH * last_seconds > deadline:
                raise _Paused
            full = self.system.full_state(psi)
            problem = problem_at(self.system.problem, full, f"{self.case}_step{k}",
                                 energy=self.system.energy(psi), sector_gradients=exact_gradients)
            started = time.perf_counter()
            outcome = self._select_with(config, problem, full, rng)
            absg = np.abs(exact_gradients)
            info = {"iteration": k, "selected": outcome.selected,
                    "selected_label": self.system.labels[outcome.selected],
                    "exact_best_arm": int(np.argmax(absg)),
                    # a tie partner of the best arm counts as best: shortfall zero
                    "exact_best": bool(outcome.extra["shortfall"] < 1e-9),
                    "shortfall": outcome.extra["shortfall"], "shots": outcome.shots,
                    "rounds": outcome.rounds, "stopped_rho": outcome.extra["stopped_rho"],
                    "max_abs_gradient": float(absg.max()),
                    "seconds": round(time.perf_counter() - started, 2)}
            selections.append(info)
            live += 1
            last_seconds = info["seconds"]
            if ckpt is not None:
                _append_selection(ckpt, {"k": k, "selected": int(outcome.selected), "info": info,
                                         "rng": rng.bit_generator.state})
            return outcome.selected, info

        try:
            steps = run_adapt(self.system, select, tolerance=self.tolerance, max_iterations=self.max_iterations)
        except _Paused:
            return {"paused": True, "selections_done": len(selections)}
        shots = np.cumsum([s["shots"] for s in selections]) if selections else np.zeros(1)
        for s, total in zip(selections, shots):
            s["cumulative_shots"] = float(total)
        if ckpt is not None:
            ckpt.unlink(missing_ok=True)
        return {"iterations": len(steps) - 1, "final_error": steps[-1].error,
                "reached": bool(steps[-1].error < self.tolerance), "total_shots": float(shots[-1]),
                "exact_best_selections": sum(s["exact_best"] for s in selections),
                "selections": selections,
                "energy_errors": [s.error for s in steps]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_HF"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS))
    parser.add_argument("--energy-error", type=float, default=ENERGY_ERROR,
                        help="standard error (Ha) of the last energy evaluation whose data reuse rows hold for free")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--trajectories", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true", help="skip trajectories already checkpointed")
    parser.add_argument("--max-selections", type=int, default=None,
                        help="stop every trajectory after this many live selections of this job and write no tables; "
                             "run the same command again (with --resume) to continue, the last stage without the limit")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop every trajectory before the selection that would end after this many hours of this job "
                             "(at twice the length of its previous one) and write no tables; run the same command in the next "
                             "job, the stage in which every trajectory finishes writes the tables")
    parser.add_argument("--tag", default="",
                        help="write <case>_phase4_<tag>_*.csv instead of <case>_phase4_*.csv, so that a run of other "
                             "methods never overwrites an earlier run's tables")
    args = parser.parse_args()
    deadline = None if args.max_hours is None else time.time() + args.max_hours * 3600.0

    for case in args.cases:
        started = time.perf_counter()
        study = Phase4(case, args.strategy, args.tolerance, args.rho, args.energy_error)
        exact_iterations = len(study.exact) - 1
        print(f"{case}: exact ADAPT {exact_iterations} steps to dE {study.exact[-1].error:.2e} "
              f"(setup {time.perf_counter() - started:.0f}s)", flush=True)
        out = args.out / case / "phase4"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{case}_phase4" + (f"_{args.tag}" if args.tag else "")
        per_trajectory, per_selection, summary, partials = [], [], [], []
        incomplete = False
        for name in args.methods:
            config = with_rho(METHODS[name], args.rho)
            clock = time.perf_counter()
            partial = out / f"{case}_phase4_{slug(name)}.partial.jsonl"
            done = _load_partial(partial) if args.resume else {}
            ckpt_stem = out / f"{case}_phase4_{slug(name)}" if args.resume else None
            if not args.resume:
                partial.unlink(missing_ok=True)
                for stale in out.glob(f"{case}_phase4_{slug(name)}.traj_*.partial.jsonl"):
                    stale.unlink()
            if done:
                print(f"{case}: {name}: resuming with {len(done)} finished trajectories", flush=True)
            fresh = run_trials(lambda rng, c=config, st=ckpt_stem: study.trajectory(rng, c, st, args.max_selections, deadline),
                               args.trajectories, args.seed, workers=args.workers, skip=done,
                               on_result=lambda index, result, p=partial: (
                                   None if result.get("paused") else _append_partial(p, index, result)))
            paused = [index for index, result in fresh if result.get("paused")]
            if paused:
                incomplete = True
                print(f"{case}: {name}: {len(paused)} trajectories paused ({', '.join(f'{k} selections done' for k in sorted({r['selections_done'] for _, r in fresh if r.get('paused')}))}); "
                      "run the next stage", flush=True)
                continue
            results = sorted(list(done.items()) + fresh, key=lambda item: item[0])
            partials.append(partial)
            seconds = time.perf_counter() - clock
            totals = np.array([r["total_shots"] for _, r in results])
            iterations = np.array([r["iterations"] for _, r in results])
            reached = np.array([r["reached"] for _, r in results])
            for index, r in results:
                per_trajectory.append({"method": name, "trajectory": index,
                                       **{k: v for k, v in r.items() if k not in ("selections", "energy_errors")}})
                per_selection.extend({"method": name, "trajectory": index, **s} for s in r["selections"])
            selections = [s for _, r in results for s in r["selections"]]
            row = {
                "case_id": case, "method": name, "label": label_of(config), "trajectories": len(results),
                "exact_adapt_iterations": exact_iterations,
                "exact_adapt_final_error": study.exact[-1].error,
                "reached_rate": float(reached.mean()),
                "iterations_mean": float(iterations.mean()), "iterations_max": int(iterations.max()),
                "final_error_median": float(np.median([r["final_error"] for _, r in results])),
                "final_error_p90": float(np.percentile([r["final_error"] for _, r in results], 90)),
                "shots_median": float(np.median(totals)), "shots_mean": float(totals.mean()),
                "shots_p90": float(np.percentile(totals, 90)),
                "selections": len(selections),
                "exact_best_rate": float(np.mean([s["exact_best"] for s in selections])),
                "rho_good_rate": float(np.mean([s["shortfall"] <= args.rho + 1e-12 for s in selections])),
                "mean_shortfall": float(np.mean([s["shortfall"] for s in selections])),
                "seconds": round(seconds, 1),
            }
            summary.append(row)
            print(f"{case:28s} {name:22s} reached {row['reached_rate']:.2f}  iterations "
                  f"{row['iterations_mean']:.1f} (exact {exact_iterations})  shots median "
                  f"{row['shots_median']:12,.0f}  P90 {row['shots_p90']:12,.0f}  exact-best "
                  f"{row['exact_best_rate']:.2f}  rho-good {row['rho_good_rate']:.3f}  ({seconds:.0f}s)",
                  flush=True)
        if incomplete:
            print(f"{case}: not all trajectories are finished; tables are written by the last stage", flush=True)
            continue
        write_csv(out / f"{stem}_trajectories.csv", list(per_trajectory[0]), per_trajectory)
        write_csv(out / f"{stem}_selections.csv",
                  list(dict.fromkeys(k for r in per_selection for k in r)), per_selection)
        write_csv(out / f"{stem}_summary.csv", list(summary[0]), summary)
        write_json(out / f"{stem}_meta.json", {
            "rho": args.rho, "tolerance": args.tolerance, "trajectories": args.trajectories,
            "seed": args.seed, "strategy": args.strategy,
            "methods": {k: label_of(with_rho(METHODS[k], args.rho)) for k in args.methods},
            "energy_error": args.energy_error,
            "exact_adapt": [{"iteration": s.iteration, "energy_error": s.error,
                             "added": study.system.labels[s.selected] if s.selected is not None else None}
                            for s in study.exact],
            "counted": "gradient-selection context-shots only; VQE energies are exact",
            "run": run_record()})
        for partial in partials:
            partial.unlink(missing_ok=True)


class _Paused(Exception):
    """The selection budget of this job is used up; the trajectory continues in the next stage."""


def _rng_key(rng: np.random.Generator) -> str:
    return format(int(rng.bit_generator.state["state"]["state"]) % (1 << 48), "012x")


def _append_selection(path: Path, record: dict) -> None:
    import os

    with path.open("a") as handle:
        handle.write(json.dumps(record, default=_plain) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_selections(path: Path) -> list[dict]:
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # cut off by the time limit
                break
    return records


def _append_partial(path: Path, index: int, result: dict) -> None:
    """Checkpoint one finished trajectory (JSON line) so an interrupted job can resume."""
    import json
    import os

    with path.open("a") as handle:
        handle.write(json.dumps({"index": index, "result": result}, default=_plain) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_partial(path: Path) -> dict[int, dict]:
    import json

    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(record["index"])] = record["result"]
    return done


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    main()
