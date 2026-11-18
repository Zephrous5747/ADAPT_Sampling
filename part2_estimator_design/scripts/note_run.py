#!/usr/bin/env python3
"""Run the fixed-state selection trials of the method note ``reports/prior/Classical_priors_for_measurement_design.tex``
(``src/note_protocol.py``): arms NOTE_I ... NOTE_IV (+ the diagnostic NOTE_IV_DATA) with the note's Hoeffding confidence (Appendix A) or the
bounded-variance replacement.

One JSON line per trial in ``<out>/<state>__<arm>__<confidence>.jsonl``; the fields are those of ``reframe_run.py``.

    OMP_NUM_THREADS=1 python scripts/note_run.py --states LiH_r16_ADAPT2 --arms NOTE_I NOTE_IV --confidence bernstein --trials 5 --workers 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run as rr  # noqa: E402
from note_protocol import CONFIDENCE, NOTE_ARMS, NoteSelector  # noqa: E402

VERSION = "NOTE-1.0"
_SELECTOR = None
_CONTEXT: dict = {}


def tasks_for(protocol: dict, state_id: str, arm: str, confidence: str, trials: int, start: int) -> list[dict]:
    stage = rr.stage_of(state_id, protocol)
    system = next(s for s in protocol["fixed_states"] if s["id"] == state_id)["system"]
    out = []
    for trial in range(start, start + trials):
        key = "|".join(map(str, [VERSION, stage, state_id, "UCCSD", arm, confidence, "strict", trial]))
        out.append(dict(task_id=hashlib.sha256(key.encode()).hexdigest()[:20], stage=stage, system_id=system, state_id=state_id, arm=arm,
                        confidence=confidence, rho=0.0, delta=0.05, trial=trial,
                        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")))
    return out


def attach_energies(selector, case: str, info: dict) -> None:
    """Classical energies of the prior state(s) and the exact energy of the state (the 'sampled' energy adds shot noise per trial)."""
    from classical_prior import sparse_state_for_case
    from pauli_ops import PauliEvaluator
    from sampler import OracleMoments

    terms = selector.hamiltonian_terms
    energy = lambda state: float(PauliEvaluator(state).fragment_mean_std(terms)[0])  # noqa: E731
    reference = energy(selector.problem.evaluator.state)
    selector.energy_q = selector.state_energy
    selector.energy_c["full"] = selector.energy_q + energy(selector.prior.state) - reference
    if not case.endswith("_CISD") and not case.endswith("_HF") and int(info.get("K", 1)) > 1:
        half = max(1, int(info["K"]) // 2)
        state, half_info = sparse_state_for_case(case, keep=half)
        moments = OracleMoments(selector.library, state)
        for alpha in range(selector.library.n_contexts):
            moments.group_means(alpha)
        selector.priors["half"] = moments
        selector.energy_c["half"] = selector.energy_q + energy(state) - reference
        print(f"{case}: degraded prior K = {half}, overlap {half_info['overlap']:.4f}; "
              f"gap full {1e3 * (selector.energy_c['full'] - selector.energy_q):+.3f} mHa, half {1e3 * (selector.energy_c['half'] - selector.energy_q):+.3f} mHa", flush=True)
    else:
        print(f"{case}: gap full {1e3 * (selector.energy_c['full'] - selector.energy_q):+.3f} mHa", flush=True)


def _one(task: dict) -> dict:
    started = time.perf_counter()
    result = _SELECTOR.run_note(np.random.default_rng(task["seed"]), task["arm"], task["confidence"], task["rho"], task["delta"])
    record = dict(record_id=task["task_id"], protocol_version=VERSION, code_commit=_CONTEXT["commit"], system_id=task["system_id"],
                  state_id=task["state_id"], method=f"{task['arm']}_{task['confidence']}", arm=task["arm"], confidence=task["confidence"],
                  pool_id="UCCSD", rho=task["rho"], delta=task["delta"], seed=task["seed"], stage=task["stage"], trial=task["trial"],
                  status=result["status"], selection_shots=int(result["selection_shots"]), block_shots=[int(b) for b in result["block_shots"]],
                  selected_index=result["selected_index"], used_exact_moments_for_design=False,
                  wall_seconds=round(time.perf_counter() - started, 3),
                  peak_memory_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, **_CONTEXT["hashes"])
    for key in ("correct", "tied_maximizers", "rounds_reached", "n_blocks", "checkpoints", "block_failures", "history_failures",
                "maximizer_eliminated", "n_survivors", "gap_true", "gap_hat", "tau_used", "plan_seconds", "solve_seconds", "sample_seconds", "interval_seconds", "cap_hit",
                "capped_arms"):
        if key in result:
            record[key] = result[key]
    record["blocks"] = result["blocks"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(NOTE_ARMS), choices=list(NOTE_ARMS))
    parser.add_argument("--confidence", nargs="+", default=list(CONFIDENCE), choices=list(CONFIDENCE))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/note"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    global _SELECTOR
    protocol = json.loads(rr.PROTOCOL.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for state_id in args.states:
        case = rr.case_of(state_id, protocol)
        cap = args.cap if args.cap is not None else protocol["caps"]["selection_sto3g"]
        started = time.perf_counter()
        selector, hashes, info = rr.build(case, cap, NoteSelector)
        attach_energies(selector, case, info)
        print(f"{state_id} ({case}): {len(selector.candidates)} candidates, {selector.n_contexts} contexts ({time.perf_counter() - started:.0f}s)", flush=True)
        _SELECTOR = selector
        _CONTEXT.update(commit="note", hashes=hashes)
        for confidence in args.confidence:
            for arm in args.arms:
                path = args.out / f"{state_id}__{arm}__{confidence}.jsonl"
                done = set()
                if args.resume and path.exists():
                    done = {json.loads(line)["record_id"] for line in path.read_text().splitlines() if line.strip()}
                todo = [t for t in tasks_for(protocol, state_id, arm, confidence, args.trials, args.trial_start) if t["task_id"] not in done]
                if not todo:
                    print(f"{state_id} {arm} {confidence}: complete", flush=True)
                    continue
                began = time.perf_counter()
                records = []
                with open(path, "a") as handle:
                    if args.workers <= 1:
                        for record in map(_one, todo):
                            handle.write(json.dumps(record) + "\n"); handle.flush(); records.append(record)
                    else:
                        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as pool:
                            for future in as_completed([pool.submit(_one, t) for t in todo]):
                                record = future.result()
                                handle.write(json.dumps(record) + "\n"); handle.flush(); records.append(record)
                shots = np.array([r["selection_shots"] for r in records], dtype=float)
                status = {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})}
                print(f"{state_id:20s} {arm:13s} {confidence:9s} n={len(records):3d} shots mean {shots.mean():12,.0f} median {np.median(shots):12,.0f} "
                      f"correct {np.mean([r['correct'] for r in records]):.3f} status {status} ({time.perf_counter() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
