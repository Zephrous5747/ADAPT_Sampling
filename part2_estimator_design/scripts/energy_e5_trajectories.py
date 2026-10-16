"""E5 first look: trajectory outcomes of the certified II-A / II-0 runs at the default 50-shot minimum (Phase 4, rho = 0.1).
Per case: trajectories, share that reached chemical accuracy, steps (mean) vs exact ADAPT, share of rho-good / exact-best selections,
and how many selections per trajectory were not rho-good."""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path("runs")
cases = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]
methods = ["II-A data, safe", "II-0 safe", "M2 safe", "M1 seq, safe"]
print(f"{'case':30s} {'method':16s} {'n':>4s} {'reach':>6s} {'steps':>12s} {'exactN':>6s} {'rho-good sel':>12s} {'exact-best sel':>14s} {'traj w/ >=1 non-rho-good':>25s} {'final dE med mHa':>16s}")
for case in cases:
    exact_iter = None
    for method in methods:
        trajs, sels = [], []
        for path in sorted(glob.glob(str(root / case / "phase4" / f"{case}_phase4*_trajectories.csv"))):
            stem = path[: -len("_trajectories.csv")]
            tag = Path(stem).name[len(case) + len("_phase4"):].lstrip("_")
            if "sens_rho" in tag or "reuse" in tag or "fixed" in tag.lower():
                continue
            meta = Path(stem + "_meta.json")
            if meta.exists():
                m = json.loads(meta.read_text())
                if abs(float(m.get("rho", 0.1)) - 0.1) > 1e-12:
                    continue
                exact_iter = len(m["exact_adapt"]) - 1 if isinstance(m.get("exact_adapt"), list) else exact_iter
            rows = [r for r in csv.DictReader(open(path)) if r["method"] == method]
            trajs += rows
            sp = stem + "_selections.csv"
            if Path(sp).exists():
                sels += [r for r in csv.DictReader(open(sp)) if r["method"] == method]
        if not trajs:
            continue
        reach = np.mean([r.get("reached", "") == "True" or float(r["final_error"]) < 1.6e-3 for r in trajs])
        steps = np.array([float(r["iterations"]) for r in trajs])
        fin = np.array([float(r["final_error"]) for r in trajs])
        sh = np.array([float(r["shortfall"]) for r in sels]) if sels else np.array([])
        rg = np.mean(sh <= 0.1 + 1e-12) if sh.size else float("nan")
        eb = np.mean(sh < 1e-9) if sh.size else float("nan")
        by = defaultdict(int)
        for r in sels:
            if float(r["shortfall"]) > 0.1 + 1e-12:
                by[(r["trajectory"], tag)] += 1
        ntr = len(trajs)
        bad = len({(r["trajectory"]) for r in sels if float(r["shortfall"]) > 0.1 + 1e-12})
        print(f"{case:30s} {method:16s} {ntr:4d} {reach:6.2f} {steps.mean():6.2f}+-{steps.std():4.2f} {str(exact_iter):>6s} {rg:12.3f} {eb:14.3f} {bad:25d} {1e3*np.median(fin):16.3f}")
