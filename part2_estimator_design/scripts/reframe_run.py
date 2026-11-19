#!/usr/bin/env python3
"""Run the fixed-state selection trials of the reframed protocol GS-1.0 (``src/reframe.py``).

Tasks, seeds and record ids follow ``reports/reframed/generator_selection_paper/make_plan.py`` (copy of the protocol in
``reframe/protocol.json``).  One JSON line per trial is appended to ``<out>/<state>__<method>.jsonl``; finished record ids are
skipped on ``--resume``.  The record carries every field of ``validate_records.py`` plus diagnostics.

    OMP_NUM_THREADS=1 python scripts/reframe_run.py --states LiH_r16_ADAPT2 --methods FC_SE FC_SE_CS --trials 5 --workers 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
from reframe import METHODS, ReframeSelector  # noqa: E402

PROTOCOL = ROOT / "reframe" / "protocol.json"
SYSTEM_TO_CASE = {
    "H4_a1": "H4_square_eq_side1p0", "H4_a2": "H4_square_stretch_side2p0", "LiH_r16": "LiH_R1p6", "LiH_r30": "LiH_R3p0",
    "BeH2_r133": "BeH2", "H2O_eq": "H2O_eq", "H2O_str": "H2O_stretch", "H2O_cas_eq": "H2O_dz8o_eq", "H2O_cas_str": "H2O_dz8o_str",
}
DIAGNOSTIC = "FC_SE_CS_DATA"


def case_of(state_id: str, protocol: dict) -> str:
    state = next(s for s in protocol["fixed_states"] if s["id"] == state_id)
    prefix = SYSTEM_TO_CASE[state["system"]]
    kind = state["kind"]
    if kind == "HF":
        return f"{prefix}_HF"
    if kind == "CISD":
        return f"{prefix}_CISD"
    return f"{prefix}_ADAPT{state['m']}"


def stage_of(state_id: str, protocol: dict) -> str:
    return "fixed_" + next(s for s in protocol["fixed_states"] if s["id"] == state_id)["group"]


SETTINGS = {"strict": 0.0, "rho005": 0.05, "rho010": 0.1, "rho020": 0.2, "rho010risk020": 0.1,
            "cal020": 0.0, "cal050": 0.0, "cal090": 0.0}  # setting id -> rho
DELTAS = {"rho010risk020": 0.2,  # setting id -> delta where it differs from the default 0.05 (T26: rho 0.1 together with delta 0.2)
          "cal020": 0.2, "cal050": 0.5, "cal090": 0.9}  # coverage calibration (JCTC round 4): exact identification at inflated delta, to see the observed error rate


def task_list(protocol: dict, digest: str, state_id: str, method: str, trials: int, start: int, schedule: str = "literal",
              setting_id: str = "strict", restart_ratio: float = 1.0, batch_fix: bool = False, full_dominance: bool = False,
              growth: float = 2.0, grouping: str = "fc", leader_out: bool = False) -> list[dict]:
    stage = stage_of(state_id, protocol)
    system = next(s for s in protocol["fixed_states"] if s["id"] == state_id)["system"]
    setting = {"id": setting_id, "rho": SETTINGS[setting_id], "delta": DELTAS.get(setting_id, 0.05)}
    tasks = []
    for trial in range(start, start + trials):
        key = "|".join(map(str, [protocol["protocol_version"], stage, state_id, "UCCSD", method, setting["id"], trial]))
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")  # same seed for both schedules: paired streams
        if schedule != "literal":
            key += "|" + schedule
        if restart_ratio != 1.0 or batch_fix:  # T23 policy: the same seed (paired streams), a different record id
            key += f"|restart{restart_ratio:g}|batch{int(batch_fix)}"
        if full_dominance:  # T24: same seed, a different record id
            key += "|dominance1"
        if growth != 2.0:
            key += f"|growth{growth:g}"
        if grouping != "fc":
            key += f"|{grouping}"
        if leader_out:  # T25: same seed, a different record id
            key += "|leaderout1"
        tasks.append(dict(restart_ratio=restart_ratio, batch_fix=batch_fix, full_dominance=full_dominance, growth=growth, leader_out=leader_out, task_id=hashlib.sha256(key.encode()).hexdigest()[:20], schedule=schedule, stage=stage, system_id=system,
                          state_id=state_id, method=method, setting_id=setting["id"], rho=setting["rho"], delta=setting["delta"],
                          trial=trial, seed=seed))
    return tasks


def sha(obj) -> str:
    data = obj if isinstance(obj, bytes) else json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


GROUPINGS = ("fc", "qwc", "pivot", "pivot_merged", "individual")


class _ArmFragments:
    """The fixed reading of one gradient from its own contexts: the fields of a fragment problem that ``ReframeSelector`` uses."""

    def __init__(self, x, ctx_ids, ctx_ptr, coord_pauli) -> None:
        self.x, self.ctx_ids, self.ctx_ptr, self.coord_pauli = x, ctx_ids, ctx_ptr, coord_pauli


def individual_setup(problem):
    """Individual grouping (IG): every gradient has its own fully commuting groups and its own contexts, nothing is shared between
    gradients (the grouping of a method that estimates each gradient separately).  Context ``a`` measures group ``a`` of the concatenation of
    the arms' groups; the reading of gradient i uses only its own contexts."""
    from contexts import build_context_library
    from step5_external_baselines import moments_for

    arms = problem.individual_fc_groups()
    groups = [list(g) for arm in arms for g in arm]
    library = build_context_library(problem, "canonical", groups=groups)
    oracle = moments_for(library, problem.evaluator.state)
    index = {label: k for k, label in enumerate(library.labels)}
    base, coords, context = [], [], 0
    for terms, arm_groups in zip(problem.commutator_terms, arms):
        ctx_ids, pointer, pauli_ids, x = [], [0], [], []
        for group in arm_groups:
            ids = np.array([index[p] for p in group], dtype=np.int64)
            coefficient = np.array([float(np.real(terms[p])) for p in group])
            order = np.argsort(ids)
            ctx_ids.append(context)
            context += 1
            pauli_ids.extend(ids[order].tolist())
            x.extend(coefficient[order].tolist())
            pointer.append(len(x))
        base.append(_ArmFragments(np.array(x), np.array(ctx_ids, dtype=np.int64), np.array(pointer, dtype=np.int64),
                                  np.array(pauli_ids, dtype=np.int64)))
        coords.append((None, None, {index[p]: float(np.real(c)) for p, c in terms.items()}))
    return {"library": library, "oracle": oracle, "base": base, "coords": coords}


def build(case: str, cap: float, selector_class=ReframeSelector, grouping: str = "fc", need_prior: bool = True):
    """``grouping``: the measurement contexts.  ``fc`` (default): the shared fully commuting library of GS-1.0.  ``qwc``: qubit-wise commuting
    contexts (blocks of one qubit, no entangling gates).  ``pivot`` / ``pivot_merged``: the pivot contexts of Anastasiou et al. (every product read
    from all pivots that produce it, or from one pivot chosen by greedy set cover); SG_SE reads the gradients as the published scheme does and
    SG_SE_CS may split over every context that measures a product.  Task T7."""
    from classical_prior import sparse_state_for_case
    from reuse import hamiltonian_terms
    from sampler import OracleMoments

    if grouping == "individual":
        from step4_learned_designs import load_problem

        problem = load_problem(case)
        shared = individual_setup(problem)
        library, oracle, base, coords = shared["library"], shared["oracle"], shared["base"], shared["coords"]
        dist = np.clip(oracle.distributions_matrix().copy(), 0.0, None)
        dist /= dist.sum(axis=1, keepdims=True)
        selector = selector_class(problem, library, dist, base, coords, None, cap=cap)  # SG_SE reads no prior
        terms = hamiltonian_terms(problem, case)
        selector.hamiltonian_terms = terms
        selector.state_energy = float(problem.state_energy)
        hashes = {
            "state_sha256": sha(np.ascontiguousarray(problem.evaluator.state).tobytes()),
            "hamiltonian_sha256": sha(sorted(terms.items())),
            "pool_sha256": sha([sorted(t.items()) for t in problem.commutator_terms]),
            "contexts_sha256": sha([library.labels, [c.members.tolist() for c in library.contexts]]),
        }
        return selector, hashes, {"prior": "none (individual grouping, SE arm only)"}
    if grouping == "fc":
        from step4_learned_designs import setup

        problem, library, oracle, hf_prior, base, coords = setup(case, "mass", "weight")
    else:
        from states import hartree_fock_state
        from step4_learned_designs import load_problem
        from step5_external_baselines import block_setup, pivot_setup

        problem = load_problem(case)
        if grouping == "qwc":
            shared = block_setup(problem, 1, splitting=True)
        else:
            shared = pivot_setup(problem, case, "auto", "split" if grouping == "pivot" else "merged", splitting=True)
        library, oracle, base, coords = shared["library"], shared["oracle"], shared["base"], shared["coords"]
        hf_prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    dist = oracle.distributions_matrix().copy()
    dist = np.clip(dist, 0.0, None)
    dist /= dist.sum(axis=1, keepdims=True)
    if not need_prior:  # T24: only FC_SE_CS reads the classical prior; the other arms skip building it (time and memory)
        prior, info = None, {"prior": "none (not needed by the requested methods)"}
    elif case.endswith("_CISD"):  # no ansatz replay: HF covariance initialization with the same weight (stress test)
        prior, info = hf_prior, {"prior": "HF covariance (CISD stress test)"}
    else:
        state, info = sparse_state_for_case(case)
        prior = OracleMoments(library, state)
        for alpha in range(library.n_contexts):
            prior.group_means(alpha)  # primed here so that forked workers do not recompute them
        info = {k: v for k, v in info.items() if not hasattr(v, "shape")}
    if need_prior:
        for alpha in range(library.n_contexts):
            hf_prior.group_means(alpha)
    selector = selector_class(problem, library, dist, base, coords, prior, cap=cap)
    terms = hamiltonian_terms(problem, case)
    selector.hamiltonian_terms = terms
    selector.state_energy = float(problem.state_energy)
    hashes = {
        "state_sha256": sha(np.ascontiguousarray(problem.evaluator.state).tobytes()),
        "hamiltonian_sha256": sha(sorted(terms.items())),
        "pool_sha256": sha([sorted(t.items()) for t in problem.commutator_terms]),
        "contexts_sha256": sha([library.labels, [c.members.tolist() for c in library.contexts]]),
    }
    return selector, hashes, info


_SELECTOR = None
_CONTEXT = {}


def _one(task: dict) -> dict:
    rng = np.random.default_rng(task["seed"])
    started = time.perf_counter()
    result = _SELECTOR.run(rng, task["method"], task["rho"], task["delta"], task.get("schedule", "literal"),
                           restart_ratio=task.get("restart_ratio", 1.0), batch_fix=task.get("batch_fix", False),
                           full_dominance=task.get("full_dominance", False), growth=task.get("growth", 2.0),
                           leader_out=task.get("leader_out", False))
    record = dict(
        record_id=task["task_id"], protocol_version=_CONTEXT["protocol"]["protocol_version"], protocol_sha256=_CONTEXT["digest"],
        code_commit=_CONTEXT["commit"], environment_file="part2_estimator_design/requirements.txt",
        system_id=task["system_id"], state_id=task["state_id"], method=task["method"], pool_id="UCCSD",
        rho=task["rho"], delta=task["delta"], seed=task["seed"], stage=task["stage"], trial=task["trial"],
        schedule=task.get("schedule", "literal"), status=result["status"], selection_shots=int(result["selection_shots"]), block_shots=[int(b) for b in result["block_shots"]],
        selected_index=result["selected_index"], used_exact_moments_for_design=False, evidence_directory=_CONTEXT["out"],
        wall_seconds=round(time.perf_counter() - started, 3), peak_memory_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        **_CONTEXT["hashes"],
    )
    record.update(setting_id=task["setting_id"], restart_ratio=task.get("restart_ratio", 1.0), batch_fix=task.get("batch_fix", False),
                  full_dominance=task.get("full_dominance", False), growth=task.get("growth", 2.0), leader_out=task.get("leader_out", False),
                  grouping=_CONTEXT.get("grouping", "fc"), n_contexts=int(_SELECTOR.n_contexts))
    for key in ("correct", "rho_good", "tied_maximizers", "rounds_reached", "n_blocks", "checkpoints", "block_failures", "history_failures",
                "maximizer_eliminated", "fallback_designs", "n_survivors", "peak_context_count", "plan_seconds", "solve_seconds",
                "sample_seconds", "interval_seconds", "cap_hit"):
        if key in result:
            record[key] = result[key]
    record["design_choices"] = {c: result["design_choices"].count(c) for c in set(result["design_choices"])}
    record["blocks"] = result["blocks"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", required=True, help="protocol state ids, e.g. LiH_r16_ADAPT2")
    parser.add_argument("--methods", nargs="+", default=["FC_FULL", "FC_SE", "FC_SE_CS"], choices=list(METHODS) + ["IND_SE"])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cap", type=float, default=None, help="shot cap per selection (default: the protocol's)")
    parser.add_argument("--out", type=Path, default=Path("runs/reframe"))
    parser.add_argument("--schedule", choices=("literal", "proportional"), default="literal",
                        help="literal: SI S4 as written; proportional: deviation that keeps the plan's proportions in the first checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setting", choices=tuple(SETTINGS), default="strict", help="protocol setting: strict (rho 0) or a rho-good stop")
    parser.add_argument("--restart-ratio", type=float, default=1.0,
                        help="1: new block after every elimination (GS-1.0 as first written); 0: never; 0.5: when at most half of the requested survive")
    parser.add_argument("--batch-fix", action="store_true", help="first-checkpoint cap max(4096, 64 L_active) instead of 4096")
    parser.add_argument("--full-dominance", action="store_true",
                        help="T24: FC_FULL accepts a sole virtual survivor (all other candidates strictly dominated) also at rho > 0")
    parser.add_argument("--growth", type=float, default=2.0, help="T24: factor between consecutive checkpoints of a block (2 = the doubling schedule)")
    parser.add_argument("--leader-out", action="store_true",
                        help="T25: the relative certificate takes the maximum upper bound over the other survivors only (leader excluded)")
    parser.add_argument("--suffix", default="", help="appended to the record file names")
    parser.add_argument("--grouping", choices=GROUPINGS, default="fc",
                        help="measurement contexts: fc (GS-1.0 shared FC library), qwc, pivot (Anastasiou et al.), pivot_merged (task T7); use --suffix and --out to keep the files apart")
    args = parser.parse_args()
    _CONTEXT["grouping"] = args.grouping
    global _SELECTOR
    protocol = json.loads(PROTOCOL.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
        commit += "+uncommitted" if dirty else ""
    except OSError:
        commit = "unknown"
    args.out.mkdir(parents=True, exist_ok=True)
    for state_id in args.states:
        case = case_of(state_id, protocol)
        stage = stage_of(state_id, protocol)
        cap = args.cap if args.cap is not None else protocol["caps"]["selection_active_space" if stage == "fixed_active_space" else "selection_sto3g"]
        started = time.perf_counter()
        selector, hashes, info = build(case, cap, grouping=args.grouping, need_prior="FC_SE_CS" in args.methods)
        print(f"{state_id} ({case}): {selector.n_arms} candidates ({len(selector.candidates)} non-constant), {selector.n_contexts} contexts, "
              f"prior {info.get('K', info.get('prior'))} ({time.perf_counter() - started:.0f}s)", flush=True)
        _SELECTOR = selector
        _CONTEXT.update(protocol=protocol, digest=digest, commit=commit, out=str(args.out), hashes=hashes)
        for method in args.methods:
            if method == "IND_SE":
                print("IND_SE: supplementary independent-gradient comparator is not part of this runner", flush=True)
                continue
            path = args.out / f"{state_id}__{method}{'' if args.schedule == 'literal' else '__' + args.schedule}{args.suffix}.jsonl"
            done = set()
            if args.resume and path.exists():
                done = {json.loads(line)["record_id"] for line in path.read_text().splitlines() if line.strip()}
            tasks = [t for t in task_list(protocol, digest, state_id, method, args.trials, args.trial_start, args.schedule,
                                                 args.setting, args.restart_ratio, args.batch_fix, args.full_dominance, args.growth,
                                                 args.grouping, args.leader_out) if t["task_id"] not in done]
            if not tasks:
                print(f"{state_id} {method}: complete", flush=True)
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
            shots = np.array([r["selection_shots"] for r in records], dtype=float)
            status = {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})}
            print(f"{state_id:20s} {method:14s} n={len(records):3d} shots mean {shots.mean():12,.0f} median {np.median(shots):12,.0f} "
                  f"correct {np.mean([r['correct'] for r in records]):.3f} status {status} ({time.perf_counter() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
