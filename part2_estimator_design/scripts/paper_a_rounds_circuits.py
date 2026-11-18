#!/usr/bin/env python3
"""Paper A, critique round (Q7): shots are not circuits.  Rounds per selection and distinct measurement circuits (contexts) used per selection, for II-A and for the best
baseline of Table ``tab:minshots``, on its 21 states (n_min = 400, rho = 0), from the stored trials (``rounds``, ``contexts_used``, ``shots``).

Writes ``runs/paper_a/rounds_circuits.csv`` and prints the ranges quoted in the text.

    python scripts/paper_a_rounds_circuits.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_a_minshots_headline as mh  # noqa: E402
from parallel import read_trials  # noqa: E402

N_MIN = 400


def stats(rows) -> dict:
    return {"rounds": float(np.median([float(r["rounds"]) for r in rows])), "contexts": float(np.median([float(r["contexts_used"]) for r in rows])),
            "shots": float(np.median([float(r["shots"]) for r in rows]))}


def main() -> None:
    headline = {r["case"]: r for r in csv.DictReader((ROOT / "runs" / "paper_a" / f"minshots_headline_n{N_MIN}.csv").open())}
    lookup = {name: (step, f"{lab}, min shots={N_MIN}") for name, step, lab in mh.BASELINES}
    out = []
    for case, _ in mh.STATES:
        a_label = next(l for l in mh.ii_a_labels(N_MIN) if ("step5", l) in mh.load(case))
        a = stats(read_trials(ROOT / "runs", case, "step5")[a_label])
        step, label = lookup[headline[case]["best_baseline"]]
        b = stats(read_trials(ROOT / "runs", case, step)[label])
        out.append({"case": case, "baseline": headline[case]["best_baseline"], **{f"iia_{k}": v for k, v in a.items()}, **{f"base_{k}": v for k, v in b.items()}})
    with (ROOT / "runs" / "paper_a" / "rounds_circuits.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for key in ("iia_rounds", "iia_contexts", "base_rounds", "base_contexts"):
        v = [r[key] for r in out]
        print(f"{key}: median over states {np.median(v):.0f}, range {min(v):.0f} to {max(v):.0f}")
    per = [r["iia_shots"] / r["iia_contexts"] for r in out]
    print(f"II-A shots per circuit used: median {np.median(per):.0f}, range {min(per):.0f} to {max(per):.0f}")


if __name__ == "__main__":
    main()
