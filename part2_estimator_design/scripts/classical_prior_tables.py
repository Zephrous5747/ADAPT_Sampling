#!/usr/bin/env python3
"""Classical priors for the design: the comparison of the note's arms on every saved ADAPT state and on measured trajectories.

Reads the trial files written by ``step5_external_baselines.py`` (the ``CPD`` / ``CPB`` rows and the data-only reference rows) and
the phase-4 summaries (tag ``cp``) and writes

* ``runs/classical_prior/fixed_states.csv``: one row per (family, state, arm): mean / median shots, share of correct selections, ratio of
  mean shots to the data-only II-A of the same family with a 95% percentile interval (``paper_a_bootstrap.ratio_interval``);
* ``runs/classical_prior/trajectories.csv``: one row per (system, method) with the energy-trajectory measures;
* ``reports/classical_prior_comparison_tables.md``: the tables of the comparison (generated; the commentary is hand-written).

Arms of the note (Sec. 7.1): (i) the existing estimators, II-0 and II-A learned from the data; (ii) a classical covariance for the allocation
only; (iii) classical-initialised splitting and allocation; (iv) the same with quantum covariance updates.  The families differ in what the
prior may touch: CPD, the design only (radii and planning from the data, minimum 400 shots; reference rows at 400), CPB, the design and the
radii and planning (minimum 50 shots; reference rows at 50).  See ``step5_external_baselines.py``.

    python scripts/classical_prior_tables.py [--runs runs] [--out-md ../reports/classical_prior_comparison_tables.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_a_bootstrap import ratio_interval  # noqa: E402
from parallel import read_trials  # noqa: E402

SEED = 20261021
STATES = [
    ("H4_square_eq_side1p0_ADAPT3", "H4 1.0 A, ADAPT 3"), ("H4_square_eq_side1p0_ADAPT5", "H4 1.0 A, ADAPT 5"),
    ("H4_square_stretch_side2p0_ADAPT2", "H4 2.0 A, ADAPT 2"), ("H4_square_stretch_side2p0_ADAPT5", "H4 2.0 A, ADAPT 5"),
    ("LiH_R3p0_HF", "LiH 3.0 A, HF"), ("LiH_R3p0_ADAPT3", "LiH 3.0 A, ADAPT 3"), ("LiH_R3p0_ADAPT5", "LiH 3.0 A, ADAPT 5"),
    ("LiH_R1p6_HF", "LiH 1.6 A, HF"), ("LiH_R1p6_ADAPT2", "LiH 1.6 A, ADAPT 2"), ("LiH_R1p6_ADAPT3", "LiH 1.6 A, ADAPT 3"),
    ("LiH_R1p6_ADAPT4", "LiH 1.6 A, ADAPT 4"),
    ("BeH2_ADAPT2", "BeH2, ADAPT 2"), ("BeH2_ADAPT4", "BeH2, ADAPT 4"), ("BeH2_ADAPT7", "BeH2, ADAPT 7"), ("BeH2_ADAPT9", "BeH2, ADAPT 9"),
    ("H2O_eq_ADAPT11", "H2O eq, ADAPT 11"), ("H2O_stretch_ADAPT8", "H2O str, ADAPT 8"),
]
# arm label in the step5 trial files -> (note's arm, short description)
FAMILIES = {
    "CPD": {
        "title": "CPD: the prior steers the design only (radii and planning from the data), minimum 400 shots",
        "reference": ("II-A data, safe, bound start, min shots=400, diagnose", "(i) II-A, data only"),
        "arms": [
            ("II-0, safe, bound start, min shots=400, diagnose", "(i) II-0, data only"),
            ("CPD split, sparse static", "(iii) split+alloc, sparse prior, static"),
            ("CPD split, sparse tau=100", "(iv) as (iii), updated by data, tau=100"),
            ("CPD split, sparse tau=1000", "(iv) tau=1000"),
            ("CPD split, sparse tau=100, eps=0.01", "(iv) tau=100, eps=0.01"),
            ("CPD split, HF tau=100", "HF prior, tau=100"),
            ("CPD split, true state static", "true state, static (ceiling)"),
            ("CPD split, true state tau=100", "true state, tau=100 (ceiling)"),
        ],
    },
    "CPB": {
        "title": "CPB: the prior also enters the radii and the planning (design covariance), minimum 50 shots",
        "reference": ("II-A data, safe, bound start, min shots=50, diagnose", "(i) II-A, data only"),
        "arms": [
            ("II-0, safe, bound start, min shots=50, diagnose", "(i) II-0, data only"),
            ("CPB alloc, sparse tau=100", "(ii) alloc only, sparse, tau=100"),
            ("CPB split, sparse tau=100", "(iv) split+alloc, sparse, tau=100"),
            ("CPB split, sparse tau=1000", "(iv) tau=1000"),
            ("CPB split, HF nu=100 shrink", "HF prior, draft's shrinkage"),
            ("CPB split, HF tau=100", "HF prior, tau=100"),
            ("CPB alloc, true state tau=100", "true state, alloc only (ceiling)"),
            ("CPB split, true state tau=100", "true state, split+alloc (ceiling)"),
        ],
    },
}
VALID = 0.95  # an arm counts as sound only if at least this share of its selections is correct
TRAJECTORY_CASES = [("H4_square_eq_side1p0_HF", "H4 1.0 A"), ("H4_square_stretch_side2p0_HF", "H4 2.0 A"),
                    ("LiH_R3p0_HF", "LiH 3.0 A"), ("LiH_R1p6_HF", "LiH 1.6 A")]
TRAJECTORY_METHODS = {
    "n400": [("II-0 safe, min shots=400", "II-0 data"), ("II-A data, safe, min shots=400", "II-A data (reference)"),
             ("CPD split, sparse tau=100", "CPD split, sparse")],
    "n50": [("II-0 safe", "II-0 data"), ("II-A data, safe", "II-A data (reference)"),
            ("CPB alloc, sparse tau=100", "CPB alloc, sparse"), ("CPB split, sparse tau=100", "CPB split, sparse")],
}


def load(runs: Path, case: str) -> dict[str, tuple[np.ndarray, float]]:
    out = {}
    try:
        labels = read_trials(runs, case, "step5")
    except (SystemExit, FileNotFoundError):
        return out
    for label, rows in labels.items():
        if len(rows) >= int(rows[0]["n_trials"]):
            out[label] = (np.array([float(r["shots"]) for r in rows]), float(np.mean([int(r["correct"]) for r in rows])))
    return out


def fmt_shots(x: float) -> str:
    return f"{x:.3g}"


def cell(ratio: float, lo: float, hi: float, correct: float) -> str:
    flag = "" if correct >= VALID else f" !{100 * correct:.0f}%"
    star = "*" if hi < 1.0 else ("^" if lo > 1.0 else "")
    return f"{ratio:.2f}{star}{flag}"


def fixed_state_tables(runs: Path, rng: np.random.Generator):
    rows, md = [], []
    for family, spec in FAMILIES.items():
        ref_label, ref_name = spec["reference"]
        arm_list = [(ref_label, ref_name)] + spec["arms"]
        header = ["state"] + [name for _, name in spec["arms"]]
        table = []
        ratios = {label: [] for label, _ in spec["arms"]}
        correct_all = {label: [] for label, _ in spec["arms"]}
        for case, name in STATES:
            trials = load(runs, case)
            if ref_label not in trials:
                continue
            ref, ref_correct = trials[ref_label]
            line = [name, f"{ref.mean():.3g}" + ("" if ref_correct >= VALID else f" !{100 * ref_correct:.0f}%")]
            for label, arm in spec["arms"]:
                if label not in trials:
                    line.append("-")
                    continue
                shots, correct = trials[label]
                r, lo, hi = ratio_interval(shots, ref, "mean", rng)
                line.append(cell(r, lo, hi, correct))
                rows.append({"family": family, "case": case, "state": name, "arm": arm, "label": label, "trials": shots.size,
                             "shots_mean": shots.mean(), "shots_median": float(np.median(shots)), "correct": correct,
                             "reference_shots_mean": ref.mean(), "ratio_to_ref": r, "ratio_low": lo, "ratio_high": hi})
                ratios[label].append(r)
                correct_all[label].append(correct)
            table.append(line)
        if not table:
            continue
        md.append(f"### {spec['title']}\n")
        md.append(f"Ratio of mean context-shots to the reference ({ref_name}: first column, mean shots); `*` = the 95% interval lies below 1, "
                  f"`^` above 1, `!p%` = fewer than {100 * VALID:.0f}% correct selections.\n")
        md.append("| " + " | ".join(["state", f"ref: {ref_name}"] + [n for _, n in spec["arms"]]) + " |")
        md.append("|" + "---|" * (len(spec["arms"]) + 2))
        for line in table:
            md.append("| " + " | ".join(line) + " |")
        summary = ["geometric mean over states", ""]
        worst = ["worst state / lowest correct share", ""]
        for label, _ in spec["arms"]:
            v = np.array(ratios[label])
            summary.append(f"{np.exp(np.mean(np.log(v))):.2f} (n={v.size})" if v.size else "-")
            worst.append(f"{v.max():.2f} / {100 * min(correct_all[label]):.0f}%" if v.size else "-")
        md.append("| " + " | ".join(summary) + " |")
        md.append("| " + " | ".join(worst) + " |")
        md.append("")
    return rows, md


def classical_state_table(runs: Path):
    md = ["### The classical state (sparse approximation, K = m + 1 determinants)\n",
          "| state | m | K | determinants of the state | sector dimension | overlap with the state | state weight outside the K determinants |",
          "|---|---|---|---|---|---|---|"]
    n = 0
    for case, name in STATES:
        path = runs / case / f"{case}_step5_meta.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text()).get("classical_prior")
        if not info:
            continue
        n += 1
        md.append(f"| {name} | {info['m']} | {info['K']} | {info['true_determinants']} | {info['sector_dimension']} | "
                  f"{info['overlap']:.4f} | {info['true_weight_outside']:.2e} |")
    return md if n else []


def trajectory_tables(runs: Path, rng: np.random.Generator):
    rows, md = [], []
    for group, methods in TRAJECTORY_METHODS.items():
        ref_method = methods[1][0]
        md.append(f"### Measured ADAPT trajectories, {'minimum 400 shots (CPD)' if group == 'n400' else 'minimum 50 shots (CPB)'}\n")
        md.append("Selection shots over the whole trajectory (ratio of means to II-A data, 95% interval from the trajectories), "
                  "steps to chemical accuracy, share of trajectories reaching it, share of selections that are rho-good (rho = 0.1).\n")
        md.append("| system | method | trajectories | reached | steps (mean) | final error median (Ha) | shots mean | ratio to II-A | rho-good |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for case, system in TRAJECTORY_CASES:
            path = runs / case / "phase4" / f"{case}_phase4_cp_trajectories.csv"
            spath = runs / case / "phase4" / f"{case}_phase4_cp_summary.csv"
            sel_path = runs / case / "phase4" / f"{case}_phase4_cp_selections.csv"
            if not path.exists():
                continue
            with path.open() as h:
                trajectories = list(csv.DictReader(h))
            with sel_path.open() as h:
                selections = list(csv.DictReader(h))
            by = {}
            for t in trajectories:
                by.setdefault(t["method"], []).append(t)
            ref = np.array([float(t["total_shots"]) for t in by.get(ref_method, [])])
            for method, name in methods:
                if method not in by:
                    continue
                t = by[method]
                shots = np.array([float(x["total_shots"]) for x in t])
                reached = float(np.mean([x["reached"] in ("True", "1", "true") for x in t]))
                steps = float(np.mean([float(x["iterations"]) for x in t]))
                final = float(np.median([float(x["final_error"]) for x in t]))
                sel = [s for s in selections if s["method"] == method]
                good = float(np.mean([float(s["shortfall"]) <= 0.1 + 1e-12 for s in sel])) if sel else float("nan")
                r, lo, hi = ratio_interval(shots, ref, "mean", rng) if ref.size else (float("nan"),) * 3
                rows.append({"group": group, "case": case, "system": system, "method": method, "trajectories": len(t),
                             "reached": reached, "iterations_mean": steps, "final_error_median": final,
                             "shots_mean": shots.mean(), "shots_median": float(np.median(shots)), "ratio_to_ref": r,
                             "ratio_low": lo, "ratio_high": hi, "rho_good": good})
                md.append(f"| {system} | {name} | {len(t)} | {reached:.2f} | {steps:.2f} | {final:.1e} | {shots.mean():.3g} | "
                          f"{r:.2f} [{lo:.2f}, {hi:.2f}] | {good:.3f} |")
        md.append("")
    return rows, md


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "runs" / "classical_prior")
    parser.add_argument("--out-md", type=Path, default=ROOT.parent / "reports" / "classical_prior_comparison_tables.md")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    fixed_rows, fixed_md = fixed_state_tables(args.runs, rng)
    traj_rows, traj_md = trajectory_tables(args.runs, rng)
    state_md = classical_state_table(args.runs)
    write_csv(args.out_csv / "fixed_states.csv", fixed_rows)
    write_csv(args.out_csv / "trajectories.csv", traj_rows)
    body = ["# Classical priors for measurement design: tables of the comparison (generated by scripts/classical_prior_tables.py)\n"]
    body += ["## Fixed-state selection\n"] + fixed_md + state_md + [""] + (["## Measured ADAPT trajectories\n"] + traj_md if traj_md else [])
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"{len(fixed_rows)} fixed-state rows, {len(traj_rows)} trajectory rows -> {args.out_md}")


if __name__ == "__main__":
    main()
