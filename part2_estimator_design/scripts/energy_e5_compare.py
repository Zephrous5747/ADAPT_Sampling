#!/usr/bin/env python3
"""E5: does repairing the small-sample intervals (minimum shots per fold 200 / 400 instead of 50) change the trajectories?

Phase 4 trajectories of II-A (a priori start, sign-aware rule, rho = 0.1) on one case at the default minimum of 50 shots per fold
(pooled main and extra runs) and at 200 and 400 (tags ``minshots200`` / ``minshots400``, ``cluster/jobs/make_e5_jobs.sh``).  Per set:
trajectories, steps (mean, share above exact ADAPT's), the share of selections that were rho-good / the exact best, the number of
non-rho-good selections, the selection shots (median, mean with a bootstrap interval) and the shots of the expensive late steps.

    python scripts/energy_e5_compare.py [--case LiH_R3p0_HF]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

RHO = 0.1


def rows_of(case: str, tags: list[str], method: str) -> tuple[list[dict], list[dict], int]:
    trajectories, selections, exact_steps = [], [], None
    for tag in tags:
        stem = f"runs/{case}/phase4/{case}_phase4" + (f"_{tag}" if tag else "")
        for tpath in sorted(glob.glob(stem + "*_trajectories.csv")):
            name = Path(tpath).name[len(f"{case}_phase4"):-len("_trajectories.csv")]
            if tag == "" and ("sens_rho" in name or "reuse" in name or "fixed" in name.lower() or "minshots" in name or "extra" in name.lower()
                              and False):
                continue
            if tag and not name.lstrip("_").startswith(tag):
                continue
            meta = Path(tpath[: -len("_trajectories.csv")] + "_meta.json")
            if meta.exists():
                m = json.loads(meta.read_text())
                if abs(float(m.get("rho", RHO)) - RHO) > 1e-12:
                    continue
                if isinstance(m.get("exact_adapt"), list):
                    exact_steps = len(m["exact_adapt"]) - 1
            for r in csv.DictReader(open(tpath)):
                if r["method"] == method:
                    r["file"] = Path(tpath).name
                    trajectories.append(r)
            spath = tpath.replace("_trajectories.csv", "_selections.csv")
            if Path(spath).exists():
                for r in csv.DictReader(open(spath)):
                    if r["method"] == method:
                        r["file"] = Path(tpath).name
                        selections.append(r)
    return trajectories, selections, exact_steps or 0


def boot_ci(x: np.ndarray, rng, n: int = 10000) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="LiH_R3p0_HF")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    sets = [("min shots 50 (default)", [""], "II-A data, safe"),
            ("min shots 200", ["minshots200"], "II-A data, safe, min shots=200"),
            ("min shots 400", ["minshots400"], "II-A data, safe, min shots=400")]
    print(f"{args.case}")
    print(f"{'set':24s} {'n':>4s} {'steps':>13s} {'>exact':>7s} {'rho-good':>9s} {'exact-best':>10s} {'bad/traj':>9s} "
          f"{'shots median':>13s} {'shots mean [95% CI]':>34s} {'late-step shots (3,4,5) median':>34s}")
    for label, tags, method in sets:
        traj, sel, exact_steps = rows_of(args.case, tags, method)
        if not traj:
            print(f"{label:24s} no data")
            continue
        steps = np.array([float(r["iterations"]) for r in traj])
        shots = np.array([float(r["total_shots"]) for r in traj])
        sh = np.array([float(r["shortfall"]) for r in sel])
        by_iter = {}
        for r in sel:
            by_iter.setdefault(int(r["iteration"]), []).append(float(r["shots"]))
        late = ", ".join(f"{np.median(by_iter[k]):.2e}" for k in (3, 4, 5) if k in by_iter)
        lo, hi = boot_ci(shots, rng)
        print(f"{label:24s} {len(traj):4d} {steps.mean():6.3f}+-{steps.std(ddof=1) / np.sqrt(len(steps)):5.3f} {100 * np.mean(steps > exact_steps):6.1f}% "
              f"{np.mean(sh <= RHO + 1e-12):9.4f} {np.mean(sh < 1e-9):10.3f} {np.sum(sh > RHO + 1e-12) / len(traj):9.3f} "
              f"{np.median(shots):13.3e} {shots.mean():12.3e} [{lo:.3e}, {hi:.3e}] {late:>34s}")


if __name__ == "__main__":
    main()
