#!/usr/bin/env python3
"""Replay recorded measured ADAPT runs to obtain their optimisation cost (no new shots).

A measured ADAPT run (Phase 4) records the generator chosen at every step.  The
parameter re-optimisation is exact and deterministic, so re-running the ansatz with the
recorded choices reproduces the run exactly; here we also record what Phase 4 did not:
the BFGS evaluations of every step and the modeled optimisation measurement cost C_opt
(the same model as ``features_ci_sweep.py``).  Each replayed run is checked against the
recorded one (same labels, same number of steps, same final energy to 1e-7 Ha).

Output: ``runs/features/<case>/ci_<method>_d0.05.jsonl`` in the format of
``features_ci_sweep.py`` (``"source": "replay"``), plus ``exact_reference.json``.

    OMP_NUM_THREADS=1 python scripts/features_replay.py --case H2O_eq_HF
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from parallel import slug  # noqa: E402
from pauli_ops import PauliEvaluator  # noqa: E402
from reuse import hamiltonian_terms  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, run_adapt  # noqa: E402

from paper_a_cost_decomposition import sorted_insertion_groups  # noqa: E402

ENERGY_ERROR = 1e-3
RHO = 0.1


class Replayer:
    def __init__(self, case: str) -> None:
        self.case = case
        self.system = AdaptSystem.build(case)
        self.terms = hamiltonian_terms(self.system.problem, case)
        self.groups = sorted_insertion_groups(self.terms)
        self._states: dict[tuple, tuple] = {(): (np.zeros(0), 0)}  # ops -> (params, evaluations to reach it)
        self._energy_shots: dict[tuple, float] = {}

    def node(self, ops: tuple) -> tuple[np.ndarray, int]:
        if ops not in self._states:
            parent, _ = self.node(ops[:-1])
            params, _ = self.system.optimise(list(ops), np.append(parent, 0.0))
            self._states[ops] = (params, int(self.system.last_evaluations))
        return self._states[ops]

    def energy_shots(self, ops: tuple) -> float:
        if ops not in self._energy_shots:
            params, _ = self.node(ops)
            evaluator = PauliEvaluator(self.system.full_state(self.system.state(list(ops), params)))
            total = sum(evaluator.fragment_std({p: self.terms[p] for p in g}) for g in self.groups)
            self._energy_shots[ops] = total ** 2 / ENERGY_ERROR ** 2
        return self._energy_shots[ops]

    def replay(self, selections: list[dict]) -> dict:
        ops: tuple = ()
        evaluations, n_params, c_opt, errors = [], [], [], []
        params, _ = self.node(ops)
        errors.append(self.system.energy(self.system.state([], params)) - self.system.fci_energy)
        for s in selections:
            arm = int(s["selected"])
            if self.system.labels[arm] != s["selected_label"]:
                raise ValueError(f"label mismatch at {s}: {self.system.labels[arm]}")
            ops = ops + (arm,)
            params, ev = self.node(ops)
            evaluations.append(ev)
            n_params.append(len(ops))
            c_opt.append(ev * (1 + 2 * len(ops)) * self.energy_shots(ops))
            errors.append(self.system.energy(self.system.state(list(ops), params)) - self.system.fci_energy)
        return {"evaluations": evaluations, "n_params": n_params, "c_opt_steps": c_opt, "energy_errors": errors}

    def exact_reference(self) -> dict:
        exact = run_adapt(self.system, tolerance=CHEMICAL_ACCURACY)
        sel = [{"selected": s.selected, "selected_label": self.system.labels[s.selected]} for s in exact[:-1]]
        r = self.replay(sel)
        return {"iterations": len(exact) - 1, "final_error": exact[-1].error,
                "optimizer_evaluations": int(sum(r["evaluations"])), "c_opt_eta2": float(sum(r["c_opt_steps"])),
                "evaluations": r["evaluations"], "energy_errors": [s.error for s in exact],
                "added": [self.system.labels[s.selected] for s in exact[:-1]]}


def recorded_runs(case: str, root: Path, patterns: list[str]):
    """{method: {trajectory key: (trajectory row, [selection rows])}} from the Phase 4 files."""
    runs: dict[str, dict] = defaultdict(dict)
    folder = root / case / "phase4"
    for path in sorted(folder.glob(f"{case}_phase4_*_trajectories.csv")):
        stem = path.name[: -len("_trajectories.csv")]
        tag = stem[len(f"{case}_phase4_"):]
        if not any(p in tag for p in patterns):
            continue
        sel_path = folder / f"{stem}_selections.csv"
        if not sel_path.exists():
            continue
        meta_path = folder / f"{stem}_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if abs(float(meta.get("rho", RHO)) - RHO) > 1e-12:  # the rho-sensitivity runs: other tolerance
                continue
        selections = defaultdict(list)
        for row in csv.DictReader(sel_path.open()):
            selections[(row["method"], int(row["trajectory"]))].append(row)
        for row in csv.DictReader(path.open()):
            key = (row["method"], int(row["trajectory"]))
            sel = sorted(selections[key], key=lambda r: int(r["iteration"]))
            runs[row["method"]][f"{tag}:{row['trajectory']}"] = (row, sel)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--patterns", nargs="+",
                        default=["II_A_data__safe", "II_0_safe", "M2_safe", "M1_seq__safe", "fixed_"],
                        help="substrings of the Phase 4 file tags to replay")
    parser.add_argument("--exclude", nargs="+", default=["reuse", "uniform"])
    parser.add_argument("--methods", nargs="*", default=None, help="replay only these method names")
    parser.add_argument("--max-runs", type=int, default=None, help="replay only the first N runs of a method")
    parser.add_argument("--skip-done", action="store_true", help="skip methods whose output file exists")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/features"))
    args = parser.parse_args()
    clock = time.perf_counter()
    rep = Replayer(args.case)
    out = args.out / args.case
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / "exact_reference.json"
    if not ref_path.exists():
        ref_path.write_text(json.dumps(rep.exact_reference(), indent=1))
    print(f"{args.case}: setup {time.perf_counter() - clock:.0f}s", flush=True)
    runs = recorded_runs(args.case, args.runs, args.patterns)
    worst = 0.0
    for method, trajectories in sorted(runs.items(), key=lambda kv: (kv[0].startswith("Fixed"), kv[0])):
        if any(x in method for x in args.exclude):
            continue
        if args.methods is not None and method not in args.methods:
            continue
        path = out / f"ci_{slug(method)}_d0.05.jsonl"
        if args.skip_done and path.exists():
            continue
        lines = []
        clock = time.perf_counter()
        items = sorted(trajectories.items())[: args.max_runs] if args.max_runs else sorted(trajectories.items())
        for index, (key, (row, sel)) in enumerate(items):
            r = rep.replay([{"selected": s["selected"], "selected_label": s["selected_label"]} for s in sel])
            final = r["energy_errors"][-1]
            worst = max(worst, abs(final - float(row["final_error"])))
            if len(sel) != int(row["iterations"]):
                raise ValueError(f"{method} {key}: replay {final:.9f} ({len(sel)} steps) against recorded "
                                 f"{float(row['final_error']):.9f} ({row['iterations']} steps)")
            records = [{"k": int(s["iteration"]), "selected": int(s["selected"]), "label": s["selected_label"],
                        "rank": None, "shortfall": float(s["shortfall"]),
                        "exact_best": s["exact_best"] == "True", "shots": float(s["shots"]),
                        "rounds": int(s["rounds"]) if s["rounds"] else 0} for s in sel]
            if abs(final - float(row["final_error"])) > 1e-5:
                print(f"  warning: {key}: replayed final error differs from the record by "
                      f"{1e3 * (final - float(row['final_error'])):.4f} mHa", flush=True)
            lines.append({"index": index, "key": key, "method": method, "delta": 0.05, "source": "replay",
                          "recorded_final_error": float(row["final_error"]),
                          "iterations": len(sel), "final_error": final,
                          "reached": final < CHEMICAL_ACCURACY,
                          "selection_shots": float(sum(x["shots"] for x in records)),
                          "optimizer_evaluations": int(sum(r["evaluations"])), "c_opt_eta2": float(sum(r["c_opt_steps"])),
                          "exact_best": int(sum(x["exact_best"] for x in records)),
                          "rho_good": int(sum(x["shortfall"] <= RHO + 1e-12 for x in records)),
                          "selections": records, **r})
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        print(f"{args.case} {method:30s} {len(lines):3d} runs replayed ({time.perf_counter() - clock:.0f}s, "
              f"{len(rep._states)} distinct states)", flush=True)
    print(f"largest final-energy difference replay vs record: {worst:.2e} Ha")


if __name__ == "__main__":
    main()
