#!/usr/bin/env python3
"""Re-evaluate the optimisation cost of measured ADAPT runs with another BFGS tolerance (no new shots).

The sweeps of the feature study count the evaluations of ``AdaptSystem.optimise`` (BFGS, gradient tolerance 1e-9).  That tolerance is
far below what a sampled gradient resolves, and it makes the count strongly path dependent (LiH 3.0: 54 to 297 evaluations for orders of the same
generators that end at the same energy; with 1e-5 every order needs 39).  The selections of a run do not depend on it, so the runs can be
replayed: the recorded choices are re-optimised with the new tolerance and ``evaluations``, ``n_params``, ``c_opt_steps``, ``c_opt_eta2``,
``optimizer_evaluations`` and ``energy_errors`` are replaced (all other fields, in particular the selection shots, are kept).

    python scripts/features_regtol.py --case H4_square_eq_side1p0_HF --gtol 1e-5 --src runs/features_ext --out runs/features_ext_gtol1e-5

Writes ``<out>/<case>/ci_*.jsonl`` (same layout and names as the source) and ``exact_reference.json`` recomputed with the same tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from energy_trajectory_policies import exact_reference, set_optimizer_tolerance  # noqa: E402
from features_replay import Replayer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gtol", type=float, required=True)
    parser.add_argument("--src", type=Path, default=Path("runs/features"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pattern", default="ci_*.jsonl")
    args = parser.parse_args()
    out_root = args.out or Path(f"{args.src}_gtol{args.gtol:g}")
    set_optimizer_tolerance(args.gtol)
    clock = time.perf_counter()
    rep = Replayer(args.case)
    out = out_root / args.case
    out.mkdir(parents=True, exist_ok=True)
    ref = exact_reference(rep.system, 1.6e-3)  # same stall rules as the policy runs (see energy_trajectory_policies.set_optimizer_tolerance)
    r0 = rep.replay([{"selected": a, "selected_label": lab} for a, lab in zip(ref["arms"], ref["added"])])
    (out / "exact_reference.json").write_text(json.dumps({
        "iterations": ref["iterations"], "final_error": ref["final_error"], "optimizer_evaluations": int(sum(r0["evaluations"])),
        "c_opt_eta2": float(sum(r0["c_opt_steps"])), "evaluations": r0["evaluations"], "energy_errors": ref["energy_errors"],
        "added": ref["added"]}, indent=1))
    print(f"{args.case}: gtol {args.gtol:g}, setup {time.perf_counter() - clock:.0f}s", flush=True)
    for path in sorted((args.src / args.case).glob(args.pattern)):
        lines = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sel = [{"selected": s["selected"], "selected_label": s.get("label", s.get("selected_label"))} for s in r["selections"]]
            new = rep.replay(sel)
            r.update({"evaluations": new["evaluations"], "n_params": new["n_params"], "c_opt_steps": new["c_opt_steps"],
                      "energy_errors": new["energy_errors"], "final_error": new["energy_errors"][-1],
                      "optimizer_evaluations": int(sum(new["evaluations"])), "c_opt_eta2": float(sum(new["c_opt_steps"])),
                      "gtol": args.gtol})
            lines.append(json.dumps(r))
        (out / path.name).write_text("\n".join(lines) + "\n")
        print(f"  {path.name}: {len(lines)} runs ({time.perf_counter() - clock:.0f}s, {len(rep._states)} distinct states)", flush=True)


if __name__ == "__main__":
    main()
