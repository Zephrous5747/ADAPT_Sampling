#!/usr/bin/env python3
"""Second critique list, jobs q1 (grouping), q2 (trajectories at 400 shots) and q9 (other pools at 400 shots): the stored results.

* **Q2**: ``runs/<case>/phase4/<case>_phase4_n400_summary.csv`` (tag ``n400``, seed 7, ``min shots=400``) next to the default-minimum summary
  ``<case>_phase4_summary.csv`` of the same cases: iterations and final energy against exact ADAPT, selection shots (mean), rho-good share.
* **Q1**: ``runs_group_mass/<case>/trials`` (first fit replaced by sorted insertion by coefficient mass) against ``runs/<case>/trials`` (first fit by weight), every method at
  400 shots, rho = 0: ratios of means with 95% bootstrap intervals, cheapest baseline over II-A under each grouping.
* **Q9**: ``runs/<state>@<pool>/trials`` at 400 shots and rho = 0.1 (qubit and qeb pools): best baseline over II-A and the share inside the rho-set.

Writes ``runs/paper_a/q1_grouping.csv``, ``q2_trajectories400.csv``, ``q9_pools400.csv`` and prints them.

    python scripts/paper_a_q129_summary.py
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paper_a_bootstrap import ratio_interval

ROOT = Path(__file__).resolve().parents[1]
RUNS, GROUP = ROOT / "runs", ROOT / "runs_group_mass"
OUT = RUNS / "paper_a"
STEMS = {"II-A": "II-A_data_safe_bound_start_min_shots=400_diagnose", "II-0": "II-0_safe_bound_start_min_shots=400_diagnose", "M1 seq": "M1_seq_safe_bound_start_min_shots=400",
         "M2": "M2_safe_bound_start_min_shots=400", "M2 marginal": "M2_marginal_bound_start_min_shots=400", "Ikh": "Ikh_reuse_FC_safe_bound_start_min_shots=400"}
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def trials(root: Path, case: str, stem: str, rho: str = "") -> pd.DataFrame | None:
    f = glob.glob(str(root / case / "trials" / f"step5__{stem}{rho}__shard*.csv"))
    if not f:
        return None
    d = pd.read_csv(f[0])
    return d if len(d) == int(d.n_trials.iloc[0]) else None  # complete sets only


def q1(rng) -> pd.DataFrame:
    rows = []
    for case in sorted(os.listdir(GROUP)):
        mass, base = {}, {}
        for m, stem in STEMS.items():
            a, b = trials(GROUP, case, stem), trials(RUNS, case, stem)
            if a is not None and b is not None:
                mass[m], base[m] = a, b
        if "II-A" not in mass or len(mass) < 2:
            continue
        row = {"case": case, "trials": len(mass["II-A"])}
        for tag, d in (("weight", base), ("mass", mass)):
            others = {m: v for m, v in d.items() if m not in ("II-A",) and v.correct.mean() >= 0.95}
            row[f"IIA_{tag}"] = d["II-A"].shots.mean()
            if others:
                best = min(others, key=lambda m: others[m].shots.mean())
                r = ratio_interval(others[best].shots.to_numpy(float), d["II-A"].shots.to_numpy(float), "mean", rng)
                row.update({f"best_{tag}": best, f"best_over_IIA_{tag}": r[0], f"best_lo_{tag}": r[1], f"best_hi_{tag}": r[2]})
            if "II-0" in d:
                row[f"II0_over_IIA_{tag}"] = d["II-0"].shots.mean() / d["II-A"].shots.mean()
        for m in mass:
            row[f"{m}_mass_over_weight"] = mass[m].shots.mean() / base[m].shots.mean()
        row["correct_IIA_mass"] = mass["II-A"].correct.mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "q1_grouping.csv", index=False)
    return t


def q2() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*" / "phase4" / "*_phase4_n400_summary.csv"))):
        n4 = pd.read_csv(f)
        d = os.path.join(os.path.dirname(f), os.path.basename(f).replace("_n400", ""))
        n5 = pd.read_csv(d) if os.path.exists(d) else None
        for r in n4.itertuples():
            row = {"case": r.case_id, "method": r.method, "trajectories": r.trajectories, "exact_iterations": r.exact_adapt_iterations, "iterations_400": r.iterations_mean,
                   "reached_400": r.reached_rate, "final_error_median_mHa_400": 1e3 * r.final_error_median, "shots_mean_400": r.shots_mean, "rho_good_400": r.rho_good_rate, "exact_best_400": r.exact_best_rate}
            if n5 is not None:
                base = n5[n5.method == r.method.replace(", min shots=400", "")]
                if len(base):
                    b = base.iloc[0]
                    row.update(iterations_50=b.iterations_mean, shots_mean_50=b.shots_mean, ratio_400_over_50=r.shots_mean / b.shots_mean, rho_good_50=b.rho_good_rate)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "q2_trajectories400.csv", index=False)
    return t


LINE = re.compile(r"^(\S+)\s+(.+?) reached ([\d.]+)\s+iterations ([\d.]+) \(exact (\d+)\)\s+shots median\s+([\d,]+)\s+P90\s+([\d,]+)\s+exact-best ([\d.]+)\s+rho-good ([\d.]+)")


def q2_methods() -> pd.DataFrame:
    """One-line summaries of the line logs (``cluster/logs_fir``, git-ignored): the four methods share the tag ``n400``, so the phase-4 files keep only the last to finish."""
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "cluster" / "logs_fir" / "q2_traj400_*_line*.log"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = LINE.match(line.strip())
            if m:
                case, method = m.group(1), m.group(2).strip()
                rows[(case, method)] = {"case": case, "method": method, "reached": float(m.group(3)), "iterations": float(m.group(4)), "exact_iterations": int(m.group(5)),
                                        "shots_median": float(m.group(6).replace(",", "")), "shots_p90": float(m.group(7).replace(",", "")), "exact_best": float(m.group(8)),
                                        "rho_good": float(m.group(9))}
    t = pd.DataFrame(list(rows.values()))
    if len(t):
        ref = t[t.method.str.startswith("II-A")].set_index("case").shots_median
        t["median_over_IIA"] = [r.shots_median / ref.get(r.case, np.nan) for r in t.itertuples()]
    t.to_csv(OUT / "q2_methods400.csv", index=False)
    return t


def q9(rng) -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(str(RUNS / "*@*"))):
        case = os.path.basename(d)
        if "ceo" in case or "gsd" in case or "uccsd" in case or "ADAPT" in case:
            continue
        data = {m: trials(RUNS, case, stem, "_rho=0.1") for m, stem in STEMS.items()}
        data = {m: v for m, v in data.items() if v is not None}
        if "II-A" not in data:
            continue
        a = data["II-A"]
        within = {m: float((v.shortfall <= 0.1 + 1e-12).mean()) for m, v in data.items()}
        valid = {m: v for m, v in data.items() if m != "II-A" and within[m] >= 0.95}
        row = {"case": case, "trials": len(a), "IIA_mean": a.shots.mean(), "IIA_within_rho": within["II-A"]}
        for m in ("II-0", "M1 seq", "M2", "M2 marginal", "Ikh"):
            if m in data:
                row[f"{m}_over_IIA"] = data[m].shots.mean() / a.shots.mean()
        if valid:
            best = min(valid, key=lambda m: valid[m].shots.mean())
            r = ratio_interval(valid[best].shots.to_numpy(float), a.shots.to_numpy(float), "mean", rng)
            row.update(best=best, best_over_IIA=r[0], best_lo=r[1], best_hi=r[2])
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "q9_pools400.csv", index=False)
    return t


def main() -> None:
    rng = np.random.default_rng(0)
    for name, fn in (("Q2 trajectories at 400", lambda: q2()), ("Q2 all methods, medians from the logs", q2_methods), ("Q1 grouping by coefficient mass", lambda: q1(rng)), ("Q9 other pools at 400, rho = 0.1", lambda: q9(rng))):
        print(f"\n=== {name}")
        print(fn().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
