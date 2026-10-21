#!/usr/bin/env python3
"""Feature study: what the confidence level of the selection does to a complete ADAPT run.

For every confidence level 1 - delta (for example 99, 97, 95, 90, 80 and 68 %) and every
method, measured ADAPT-VQE trajectories are run exactly as in Phase 4 (non-oracle
selection from sampled outcomes, a priori first radius, sign-aware rule, rho-good stop),
and for every trajectory we record

* the selection cost (context-shots) of every step,
* the number of ADAPT iterations (parameters in the final ansatz),
* the BFGS optimiser evaluations after every step (the "optimization steps"),
* the modeled measurement cost of that re-optimisation, C_opt (Sec. II K of the draft:
  every evaluation = one energy + an eta-term parameter-shift gradient, energies to
  1 mHa on sorted-insertion fully commuting groups of H, eta = 2),
* the energy error after every step, and whether each choice was the exact best,
  rho-good, and how much energy it lowered compared with the exact best choice.

The same seed is used for every delta (common random numbers), so differences between
levels are not differences between random streams.

Output: ``runs/features/<case>/ci_<method-slug>_d<delta>.jsonl`` (one line per trajectory;
the run resumes from these files).

    OMP_NUM_THREADS=1 python scripts/features_ci_sweep.py --case H4_square_eq_side1p0_HF \\
        --methods "II-A data, safe" --deltas 0.01 0.03 0.05 0.1 0.2 0.32 --trajectories 48 --workers 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from parallel import run_trials, slug  # noqa: E402
from pauli_ops import PauliEvaluator  # noqa: E402  (Part I)
from reuse import hamiltonian_terms  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, problem_at, run_adapt  # noqa: E402

from energy_trajectory_policies import noise_stream, set_noisy_optimizer, set_optimizer_tolerance  # noqa: E402
from paper_a_cost_decomposition import sorted_insertion_groups  # noqa: E402
from phase4_adapt_selection import METHODS, Phase4  # noqa: E402

ENERGY_ERROR = 1e-3  # Ha, standard error of one energy estimate in the C_opt model
NOISE: dict = {}     # filled by --gtol / --noise-eps: {"gtol": ..., "eps": ...}; the c_opt of a noisy run is c_opt * (ENERGY_ERROR / eps)^2


def configure(name: str, delta: float, rho: float):
    spec = METHODS[name]
    fields = {f.name for f in dataclasses.fields(spec)}
    if "delta" not in fields:  # an uncertified fixed budget: no confidence level
        return spec
    return dataclasses.replace(spec, delta=delta, rho=rho)


class Study:
    def __init__(self, case: str, rho: float, tolerance: float) -> None:
        self.case = case
        self.rho = rho
        self.phase4 = Phase4(case, "mass", tolerance, rho)
        self.system = self.phase4.system
        terms = hamiltonian_terms(self.system.problem, case)
        self.terms = terms
        self.groups = sorted_insertion_groups(terms)
        self.exact = self.phase4.exact
        self.max_iterations = self.phase4.max_iterations
        self.tolerance = tolerance

    def energy_shots(self, psi) -> float:
        """Context-shots of one energy estimate to 1 mHa at this state (optimal allocation)."""
        evaluator = PauliEvaluator(self.system.full_state(psi))
        total = sum(evaluator.fragment_std({p: self.terms[p] for p in g}) for g in self.groups)
        return total ** 2 / ENERGY_ERROR ** 2

    def trajectory(self, rng: np.random.Generator, config, ckpt_dir: Path | None = None,
                   deadline: float | None = None) -> dict:
        """One run.  With ``ckpt_dir`` every selection is checkpointed (choice, diagnostics, random state); a
        restarted job replays the recorded choices (the optimisation is deterministic) and continues live.
        With ``deadline`` (time.time()) no selection starts that would, at twice the previous one, end after it."""
        records = []
        ckpt = None
        replay = []
        if ckpt_dir is not None:
            key = format(int(rng.bit_generator.state["state"]["state"]) % (1 << 48), "012x")
            ckpt = ckpt_dir / f"traj_{key}.partial.jsonl"
            if ckpt.exists():
                for line in ckpt.read_text().splitlines():
                    try:
                        replay.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        last = [0.0]
        self.system.noise_rng = noise_stream(rng) if NOISE.get("eps") is not None else None

        def select(k, psi, exact_gradients):
            if k < len(replay):
                records.append(replay[k]["info"])
                last[0] = float(replay[k]["info"].get("seconds", 0.0))
                if k == len(replay) - 1:
                    rng.bit_generator.state = replay[k]["rng"]
                return replay[k]["info"]["selected"], replay[k]["info"]
            if deadline is not None and time.time() + 2.0 * last[0] > deadline:
                raise Paused
            full = self.system.full_state(psi)
            problem = problem_at(self.system.problem, full, f"{self.case}_step{k}",
                                 energy=self.system.energy(psi), sector_gradients=exact_gradients)
            started = time.perf_counter()
            outcome = self.phase4._select_with(config, problem, full, rng)
            absg = np.abs(exact_gradients)
            order = np.argsort(-absg)
            rank = int(np.sum(absg > absg[outcome.selected] + 1e-12)) + 1
            info = {"k": k, "selected": int(outcome.selected),
                    "label": self.system.labels[outcome.selected],
                    "best": int(order[0]), "rank": rank,
                    "g_sel": float(absg[outcome.selected]), "g_max": float(absg.max()),
                    "shortfall": float(outcome.extra["shortfall"]),
                    "exact_best": bool(outcome.extra["shortfall"] < 1e-9),
                    "shots": float(outcome.shots), "rounds": int(outcome.rounds),
                    "seconds": round(time.perf_counter() - started, 2)}
            records.append(info)
            last[0] = info["seconds"]
            if ckpt is not None:
                with ckpt.open("a") as handle:
                    handle.write(json.dumps({"k": k, "info": info, "rng": rng.bit_generator.state}, default=_plain) + "\n")
            return outcome.selected, info

        try:
            steps = run_adapt(self.system, select, tolerance=self.tolerance, max_iterations=self.max_iterations)
        except Paused:
            return {"paused": True}
        # (audit change) the checkpoint is removed only after the summary below succeeded: a failure there used to
        # throw away a run of many hours
        evaluations, n_params, c_opt, errors = [], [], [], [steps[0].error]
        for step in steps[:-1]:
            nxt = steps[step.iteration + 1]
            psi = self.system.state(nxt.ops, nxt.params)
            m_e = self.energy_shots(psi)
            n = len(nxt.ops)
            evaluations.append(int(step.evaluations))
            n_params.append(n)
            c_opt.append(float(step.evaluations * (1 + 2 * n) * m_e))
            errors.append(nxt.error)
        result = {"iterations": len(steps) - 1, "final_error": steps[-1].error,
                  "reached": bool(steps[-1].error < self.tolerance),
                  "selection_shots": float(sum(r["shots"] for r in records)),
                  "optimizer_evaluations": int(sum(evaluations)),
                  "c_opt_eta2": float(sum(c_opt)),
                  "exact_best": int(sum(r["exact_best"] for r in records)),
                  "rho_good": int(sum(r["shortfall"] <= self.rho + 1e-12 for r in records)),
                  "selections": records, "evaluations": evaluations, "n_params": n_params,
                  "c_opt_steps": c_opt, "energy_errors": errors, "noise_eps": NOISE.get("eps"), "gtol": NOISE.get("gtol")}
        if ckpt is not None:
            ckpt.unlink(missing_ok=True)
        return result

    def exact_reference(self) -> dict:
        evaluations, c_opt = [], []
        for step in self.exact[:-1]:
            nxt = self.exact[step.iteration + 1]
            psi = self.system.state(nxt.ops, nxt.params)
            evaluations.append(int(step.evaluations))
            c_opt.append(float(step.evaluations * (1 + 2 * len(nxt.ops)) * self.energy_shots(psi)))
        return {"iterations": len(self.exact) - 1, "final_error": self.exact[-1].error,
                "optimizer_evaluations": int(sum(evaluations)), "c_opt_eta2": float(sum(c_opt)),
                "evaluations": evaluations, "energy_errors": [s.error for s in self.exact],
                "added": [self.system.labels[s.selected] for s in self.exact[:-1]]}


class Paused(Exception):
    """The job's time is up; the run continues from its checkpoint in the next job."""


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--methods", nargs="+", default=["II-A data, safe"])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.01, 0.03, 0.05])
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--trajectories", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/features"))
    parser.add_argument("--fast", action="store_true", help="use src/fastpath.py (same results, faster)")
    parser.add_argument("--checkpoint", action="store_true",
                        help="checkpoint every selection, so that a killed or timed-out job resumes")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop starting selections that would end after this many hours; rerun to continue")
    parser.add_argument("--gtol", type=float, default=None,
                        help="BFGS gradient tolerance of every re-optimisation, the exact reference included (default 1e-9 of AdaptSystem.optimise)")
    parser.add_argument("--noise-eps", type=float, default=None,
                        help="BFGS on shot-noise energies (standard deviation in Ha) and gradients, see energy_trajectory_policies.set_noisy_optimizer "
                        "(needs --gtol); the reference trajectory stays exact")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help="run only these trajectory indices (to spread one level over several jobs)")
    args = parser.parse_args()
    if args.noise_eps is not None:
        if args.gtol is None:
            parser.error("--noise-eps needs --gtol")
        set_noisy_optimizer(args.noise_eps, args.gtol)
        NOISE.update(gtol=args.gtol, eps=args.noise_eps)
    elif args.gtol is not None:
        set_optimizer_tolerance(args.gtol)
        NOISE.update(gtol=args.gtol, eps=None)
    if args.fast:
        import fastpath

        fastpath.install()
    deadline = None if args.max_hours is None else time.time() + 3600.0 * args.max_hours

    started = time.perf_counter()
    study = Study(args.case, args.rho, args.tolerance)
    out = args.out / args.case
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / "exact_reference.json"
    if not ref_path.exists():
        ref_path.write_text(json.dumps(study.exact_reference(), indent=1))
    print(f"{args.case}: exact ADAPT {len(study.exact) - 1} steps (setup {time.perf_counter() - started:.0f}s)",
          flush=True)
    for name in args.methods:
        for delta in args.deltas:
            config = configure(name, delta, args.rho)
            path = out / f"ci_{slug(name)}_d{delta:g}.jsonl"
            if args.only is not None:  # one file per index, so that parallel jobs never write the same file
                path = out / f"ci_{slug(name)}_d{delta:g}_i{'-'.join(str(i) for i in sorted(args.only))}.jsonl"
            done = {}
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                        done[int(rec["index"])] = rec
                    except json.JSONDecodeError:
                        pass
            clock = time.perf_counter()

            def save(index, result, p=path):
                if result.get("paused"):
                    return
                with p.open("a") as handle:
                    handle.write(json.dumps({"index": index, "method": name, "delta": delta, **result}) + "\n")

            ckpt_dir = None
            if args.checkpoint:
                ckpt_dir = out / f"ckpt_{slug(name)}_d{delta:g}"
                ckpt_dir.mkdir(exist_ok=True)
            skip = set(done)
            if args.only is not None:
                skip |= set(range(args.trajectories)) - set(args.only)
            run_trials(lambda rng, c=config, d=ckpt_dir: study.trajectory(rng, c, d, deadline), args.trajectories,
                       args.seed, workers=args.workers, skip=skip, on_result=save)
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            shots = np.array([r["selection_shots"] for r in rows])
            its = np.array([r["iterations"] for r in rows])
            evs = np.array([r["optimizer_evaluations"] for r in rows])
            print(f"{args.case} {name:20s} conf {100 * (1 - delta):5.1f}%  n={len(rows):3d}  "
                  f"shots median {np.median(shots):11.3e}  iterations {its.mean():5.2f}  "
                  f"opt evals {evs.mean():7.1f}  reached {np.mean([r['reached'] for r in rows]):.2f}  "
                  f"({time.perf_counter() - clock:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
