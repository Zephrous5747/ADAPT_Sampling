#!/usr/bin/env python3
"""ADAPT-VQE trajectories whose generator selections follow the reframed protocol GS-1.0 (``src/reframe.py``).

The trajectory stage of ``reframe/protocol.json`` ("trajectories": 7 systems, 30 trials, settings strict / rho 0.05 / 0.1 / 0.2 / risk 0.2,
at most 60 generators, trajectory cap 1e12 shots, selection cap 1e10).  At every ADAPT step the generator is chosen by a GS-1.0 method
(FC_FULL, FC_SE, FC_SE_CS) from sampled measurement outcomes of the current state; the ansatz grows with that choice and the parameters are
re-optimised exactly (analytic gradient BFGS, protocol ``gtol`` 1e-5).  Only gradient-selection shots are counted.  A trajectory ends when
the energy is within chemical accuracy of FCI, when every exact gradient is below 1e-6 (stalled), after ``max_generators`` generators, or
when the total shot budget is used up (``budget``).

Selector details are those of ``reframe_run.py`` (policy flags ``--restart-ratio`` and ``--batch-fix`` as in task T22/T23).  The classical
covariance prior of FC_SE_CS is rebuilt from the current ansatz at every step (sparse state with K = m + 1 determinants, nu = 32).

Records: one JSON line per finished trajectory in ``<out>/<system>__<method>__<setting>.jsonl``.  With ``--resume`` every finished selection is
checkpointed (generator, diagnostics and the random-number state) in ``<out>/ckpt/``; a requeued job replays the recorded choices (the exact
re-optimisation is deterministic) before it selects live.

    OMP_NUM_THREADS=1 python scripts/reframe_traj_run.py --systems BeH2_r133 --methods FC_SE_CS --settings rho010 --trials 2 --workers 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
from classical_prior import sparse_state  # noqa: E402
from contexts import build_context_library  # noqa: E402
from design import build_fragment_problems  # noqa: E402
from reframe import METHODS, ReframeSelector  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, problem_at, run_adapt  # noqa: E402

PROTOCOL = ROOT / "reframe" / "protocol.json"
SETTINGS = {"strict": (0.0, 0.05), "rho005": (0.05, 0.05), "rho010": (0.1, 0.05), "rho020": (0.2, 0.05), "risk020": (0.0, 0.2)}
BASE_CASE = {"H4_a1": "H4_square_eq_side1p0_HF", "H4_a2": "H4_square_stretch_side2p0_HF", "LiH_r16": "LiH_R1p6_HF", "LiH_r30": "LiH_R3p0_HF",
             "BeH2_r133": "BeH2_HF", "H2O_eq": "H2O_eq_HF", "H2O_str": "H2O_stretch_HF",
             # 16-qubit cc-pVDZ active-space water (protocol "active_space_horizon": 40 generators, caps 1e11 per selection and 1e13 per trajectory)
             "H2O_cas_eq": "H2O_dz8o_eq_HF", "H2O_cas_str": "H2O_dz8o_str_HF"}


class _Stop(Exception):
    """The trajectory's total shot budget is used up."""


class Study:
    """The state-independent parts of one geometry: system, context library and the fragment problems."""

    def __init__(self, system_id: str, selection_cap: float, trajectory_cap: float, max_generators: int, restart_ratio: float,
                 batch_fix: bool, gtol: float, full_dominance: bool = False, leader_out: bool = False) -> None:
        self.system_id = system_id
        self.case = BASE_CASE[system_id]
        self.selection_cap = selection_cap
        self.trajectory_cap = trajectory_cap
        self.max_generators = max_generators
        self.restart_ratio = restart_ratio
        self.batch_fix = batch_fix
        self.full_dominance = full_dominance
        self.leader_out = leader_out  # T25: relative certificate against the other survivors only
        self.system = AdaptSystem.build(self.case)
        self.system.gtol = gtol
        problem = self.system.problem
        self.library = build_context_library(problem, "mass")
        hf_prior = OracleMoments(self.library, self.system.full_state(self.system.hf))
        self.base = build_fragment_problems(problem, self.library, hf_prior, "II-0")
        split = build_fragment_problems(problem, self.library, hf_prior, "II-A")
        self.coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
        self.exact = run_adapt(self.system, tolerance=CHEMICAL_ACCURACY)
        # gtol applies to the measured trajectories; the exact reference is the protocol's too
        self.exact_iterations = len(self.exact) - 1

    def selector(self, psi: np.ndarray, exact_gradients: np.ndarray, k: int, remaining: float, method: str = "FC_SE_CS") -> ReframeSelector:
        full = self.system.full_state(psi)
        problem = problem_at(self.system.problem, full, f"{self.case}_step{k}", energy=self.system.energy(psi), sector_gradients=exact_gradients)
        dist = np.clip(OracleMoments(self.library, full).distributions_matrix().copy(), 0.0, None)
        dist /= dist.sum(axis=1, keepdims=True)
        if method == "FC_SE_CS":  # the other arms read no classical prior
            ops, params = self.system.current_ansatz
            prior = OracleMoments(self.library, self.system.full_state(sparse_state(self.system.generators, ops, params, self.system.hf)))
            for alpha in range(self.library.n_contexts):
                prior.group_means(alpha)
        else:
            prior = None
        return ReframeSelector(problem, self.library, dist, self.base, self.coords, prior, cap=min(self.selection_cap, remaining))

    def trajectory(self, task: dict, ckpt: Path | None) -> dict:
        rng = np.random.default_rng(task["seed"])
        replay = _load(ckpt) if ckpt is not None else []
        selections: list[dict] = []
        total = {"shots": 0.0}
        method, rho, delta = task["method"], task["rho"], task["delta"]

        def select(k, psi, exact_gradients):
            if k < len(replay):
                info = replay[k]["info"]
                selections.append(info)
                total["shots"] += info["shots"]
                return replay[k]["selected"], info
            if replay and k == len(replay):
                rng.bit_generator.state = replay[-1]["rng"]
            remaining = self.trajectory_cap - total["shots"]
            if remaining <= 4096:
                raise _Stop
            started = time.perf_counter()
            selector = self.selector(psi, exact_gradients, k, remaining, method)
            result = selector.run(rng, method, rho, delta, "literal", restart_ratio=self.restart_ratio, batch_fix=self.batch_fix,
                                  full_dominance=self.full_dominance, leader_out=self.leader_out)
            chosen = result["selected_index"]
            fallback = chosen is None
            if fallback:  # no candidate left to select (stationary): take the exact best so that the trajectory is defined
                chosen = int(np.argmax(np.abs(exact_gradients)))
            absg = np.abs(exact_gradients)
            info = {"iteration": k, "selected": int(chosen), "selected_label": self.system.labels[int(chosen)], "status": result["status"],
                    "shots": int(result["selection_shots"]), "exact_best": bool(result["correct"]), "rho_good": bool(result["rho_good"]),
                    "shortfall": float(1.0 - absg[int(chosen)] / absg.max()) if absg.max() > 0 else 0.0, "n_blocks": result["n_blocks"],
                    "rounds": result["rounds_reached"], "block_failures": result["block_failures"], "history_failures": result["history_failures"],
                    "cap_hit": bool(result.get("cap_hit", False)), "fallback": fallback, "max_abs_gradient": float(absg.max()),
                    "seconds": round(time.perf_counter() - started, 2)}
            selections.append(info)
            total["shots"] += info["shots"]
            if ckpt is not None:
                _append(ckpt, {"k": k, "selected": int(chosen), "info": info, "rng": rng.bit_generator.state})
            return int(chosen), info

        ended = "reached"
        try:
            steps = run_adapt(self.system, select, tolerance=CHEMICAL_ACCURACY, max_iterations=self.max_generators)
            last = steps[-1]
            if last.error >= CHEMICAL_ACCURACY:
                ended = "max_generators" if len(steps) - 1 >= self.max_generators else "stalled"
            errors = [s.error for s in steps]
        except _Stop:
            ended = "budget"
            errors = []
        cumulative = np.cumsum([s["shots"] for s in selections]) if selections else np.zeros(1)
        for s, c in zip(selections, cumulative):
            s["cumulative_shots"] = float(c)
        if ckpt is not None and ended != "budget":
            ckpt.unlink(missing_ok=True)
        return {"ended": ended, "reached": ended == "reached", "iterations": len(selections),
                "final_error": errors[-1] if errors else None, "total_shots": float(total["shots"]),
                "exact_adapt_iterations": self.exact_iterations, "selections": selections, "energy_errors": errors,
                "exact_energy_errors": [float(s.error) for s in self.exact]}


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load(path: Path) -> list[dict]:
    records = []
    if path is not None and path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # cut off by the time limit
                break
    return records


_STUDY: Study | None = None
_OUT: Path | None = None
_RESUME = False


def _one(task: dict) -> dict:
    ckpt = _OUT / "ckpt" / f"{task['task_id']}.jsonl" if _RESUME else None
    started = time.perf_counter()
    result = _STUDY.trajectory(task, ckpt)
    return {**{k: task[k] for k in ("task_id", "system_id", "method", "setting_id", "rho", "delta", "trial", "seed")},
            "restart_ratio": _STUDY.restart_ratio, "batch_fix": _STUDY.batch_fix, "full_dominance": _STUDY.full_dominance, "leader_out": _STUDY.leader_out, "protocol_version": _CONTEXT["version"],
            "wall_seconds": round(time.perf_counter() - started, 1), **result}


_CONTEXT: dict = {}


def tasks_for(protocol: dict, system_id: str, method: str, setting_id: str, trials: int, start: int, full_dominance: bool = False,
              leader_out: bool = False) -> list[dict]:
    rho, delta = SETTINGS[setting_id]
    tasks = []
    for trial in range(start, start + trials):
        key = "|".join(map(str, [protocol["protocol_version"], "trajectory", system_id, "UCCSD", method, setting_id, trial]))
        digest = hashlib.sha256(key.encode()).digest()  # the seed does not depend on the rule (paired streams)
        if full_dominance:
            key += "|dominance1"
        if leader_out:  # T25: same seed, a different record id
            key += "|leaderout1"
        tasks.append(dict(task_id=hashlib.sha256(key.encode()).hexdigest()[:20], system_id=system_id, method=method, setting_id=setting_id,
                          rho=rho, delta=delta, trial=trial, seed=int.from_bytes(digest[:8], "big")))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", required=True, choices=list(BASE_CASE))
    parser.add_argument("--methods", nargs="+", default=["FC_FULL", "FC_SE", "FC_SE_CS"], choices=list(METHODS))
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--trials", type=int, default=None, help="default: the protocol's (30)")
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-generators", type=int, default=None, help="default: the protocol's (60)")
    parser.add_argument("--selection-cap", type=float, default=None, help="default: the protocol's (1e10)")
    parser.add_argument("--trajectory-cap", type=float, default=None, help="default: the protocol's (1e12)")
    parser.add_argument("--restart-ratio", type=float, default=0.0, help="0: never a new block after an elimination (policy of T22/T23); 1: GS-1.0 as first written")
    parser.add_argument("--no-batch-fix", action="store_true", help="the literal 4096 first-checkpoint cap instead of max(4096, 64 L_active)")
    parser.add_argument("--full-dominance", action="store_true",
                        help="T24: FC_FULL accepts a sole virtual survivor (all other candidates strictly dominated) also at rho > 0")
    parser.add_argument("--leader-out", action="store_true",
                        help="T25: the relative certificate takes the maximum upper bound over the other survivors only (leader excluded)")
    parser.add_argument("--gtol", type=float, default=None, help="BFGS gtol of the re-optimisation (default: the protocol's 1e-5)")
    parser.add_argument("--out", type=Path, default=Path("runs/reframe_traj"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    global _STUDY, _OUT, _RESUME
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol["trajectories"]
    trials = args.trials or spec["trials"]
    gtol = args.gtol or protocol["optimizer"]["gtol"]

    def limits(system_id: str) -> tuple[int, float, float]:
        """Horizon and caps by system: the active-space systems use the protocol's active-space horizon and caps."""
        if system_id.startswith("H2O_cas"):
            return (args.max_generators or protocol["active_space_horizon"]["max_generators"],
                    args.selection_cap or protocol["caps"]["selection_active_space"],
                    args.trajectory_cap or protocol["caps"]["trajectory_active_space"])
        return (args.max_generators or spec["max_generators"], args.selection_cap or protocol["caps"]["selection_sto3g"],
                args.trajectory_cap or protocol["caps"]["trajectory_sto3g"])
    args.out.mkdir(parents=True, exist_ok=True)
    _OUT, _RESUME = args.out, args.resume
    _CONTEXT["version"] = protocol["protocol_version"]
    for system_id in args.systems:
        started = time.perf_counter()
        max_generators, selection_cap, trajectory_cap = limits(system_id)
        _STUDY = Study(system_id, selection_cap, trajectory_cap, max_generators, args.restart_ratio, not args.no_batch_fix, gtol, args.full_dominance,
                        args.leader_out)
        print(f"{system_id}: {_STUDY.library.n_contexts} contexts, {len(_STUDY.coords)} generators, exact ADAPT {_STUDY.exact_iterations} steps "
              f"to dE {_STUDY.exact[-1].error:.2e} ({time.perf_counter() - started:.0f}s)", flush=True)
        for method in args.methods:
            for setting_id in args.settings:
                path = args.out / f"{system_id}__{method}__{setting_id}.jsonl"
                done = set()
                if path.exists():
                    done = {json.loads(line)["task_id"] for line in path.read_text().splitlines() if line.strip()}
                tasks = [t for t in tasks_for(protocol, system_id, method, setting_id, trials, args.trial_start, args.full_dominance, args.leader_out) if t["task_id"] not in done]
                if not tasks:
                    print(f"{system_id} {method} {setting_id}: complete", flush=True)
                    continue
                began = time.perf_counter()
                records = []
                with open(path, "a") as handle:
                    if args.workers <= 1:
                        results = map(_one, tasks)
                        for record in results:
                            handle.write(json.dumps(record) + "\n"); handle.flush(); records.append(record)
                    else:
                        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as pool:
                            for future in as_completed([pool.submit(_one, t) for t in tasks]):
                                record = future.result()
                                handle.write(json.dumps(record) + "\n"); handle.flush(); records.append(record)
                shots = np.array([r["total_shots"] for r in records])
                print(f"{system_id:10s} {method:10s} {setting_id:8s} n={len(records):3d} reached {np.mean([r['reached'] for r in records]):.2f} "
                      f"iterations {np.mean([r['iterations'] for r in records]):.1f} (exact {_STUDY.exact_iterations}) shots mean {shots.mean():12,.0f} "
                      f"({time.perf_counter() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
