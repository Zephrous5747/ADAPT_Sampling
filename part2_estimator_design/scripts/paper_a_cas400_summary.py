#!/usr/bin/env python3
"""CAS(8e,8o) fixed states (16-qubit cc-pVDZ window of H2O) at the minimum of 400 shots against the default of 50.

Reads the trial files ``runs/H2O_dz8o_{eq,str}_{CISD,ADAPT10}/trials/step5__<method>[_min_shots=400...]_rho=0.1*.csv`` (16 trials per method and minimum:
II-A, II-0, M1 seq, M2) and writes ``runs/paper_a/cas400_summary.csv`` and ``cas400_facts.json``:

* per state, method and n_min: share of exact-best selections, share within rho = 0.1, mean and median shots, rounds with a missed interval, classical hours per trial;
* ratios of means with 95% bootstrap intervals (``paper_a_bootstrap.ratio_interval``): II-A at 400 over II-A at 50; every other method over II-A at the same n_min; the
  cheapest other method (by mean shots) over II-A, at the same n_min and at 50 shots against II-A at 400;
* for the ADAPT10 states the share of the selection in the shots of its iteration, selection / (selection + C_opt of the next step) with C_opt from
  ``runs/window_share_gtol1e-5.csv`` and ``..._gtol1e-9.csv`` (``window_selection_share.py``), at 50 and 400 shots.

    python scripts/paper_a_cas400_summary.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_a_bootstrap import ratio_interval

RUNS = Path(__file__).resolve().parents[1] / "runs"
STATES = [("H2O_dz8o_eq_CISD", "eq CISD"), ("H2O_dz8o_eq_ADAPT10", "eq ADAPT10"), ("H2O_dz8o_str_CISD", "str CISD"), ("H2O_dz8o_str_ADAPT10", "str ADAPT10")]
METHODS = {"II-A": "II-A_data_safe_bound_start", "II-0": "II-0_safe_bound_start", "M1 seq": "M1_seq_safe_bound_start", "M2": "M2_safe_bound_start"}
RHO = 0.1


def trials(case: str, method: str, nmin: int) -> pd.DataFrame | None:
    stem = METHODS[method]
    pattern = f"step5__{stem}_rho=0.1*.csv" if nmin == 50 else f"step5__{stem}_min_shots=400*_rho=0.1*.csv"
    files = glob.glob(str(RUNS / case / "trials" / pattern))
    return pd.read_csv(files[0]) if files else None


def main() -> None:
    rng = np.random.default_rng(0)
    rows, facts = [], {}
    for case, label in STATES:
        data = {(m, n): trials(case, m, n) for m in METHODS for n in (50, 400)}
        for (m, n), d in data.items():
            if d is None:
                continue
            rows.append({"state": label, "method": m, "n_min": n, "trials": len(d), "exact_best": float(d.correct.mean()), "within_rho": float((d.shortfall <= RHO + 1e-12).mean()),
                         "shots_mean": float(d.shots.mean()), "shots_median": float(d.shots.median()), "missed_interval_rounds_per_trial": float(d.miscovered_rounds.mean()),
                         "classical_hours_per_trial": float(d.filter(regex="_seconds$").sum(axis=1).mean() / 3600)})
        f = {}
        a50, a400 = data[("II-A", 50)], data[("II-A", 400)]
        f["IIA_400_over_50"] = ratio_interval(a400.shots.to_numpy(float), a50.shots.to_numpy(float), "mean", rng)
        for n in (50, 400):
            others = {m: data[(m, n)] for m in ("II-0", "M1 seq", "M2") if data[(m, n)] is not None}
            base = data[("II-A", n)]
            for m, d in others.items():
                f[f"{m}_over_IIA_at_{n}"] = ratio_interval(d.shots.to_numpy(float), base.shots.to_numpy(float), "mean", rng)
            if others:
                best = min(others, key=lambda m: others[m].shots.mean())
                f[f"cheapest_other_at_{n}"] = best
                f[f"cheapest_other_over_IIA_at_{n}"] = f[f"{best}_over_IIA_at_{n}"]
        best50 = f.get("cheapest_other_at_50")
        if best50:
            f["cheapest_other_at_50_over_IIA_at_400"] = ratio_interval(data[(best50, 50)].shots.to_numpy(float), a400.shots.to_numpy(float), "mean", rng)
        facts[label] = {k: ([round(x, 3) for x in v] if isinstance(v, tuple) else v) for k, v in f.items()}

    table = pd.DataFrame(rows)
    out = RUNS / "paper_a"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "cas400_summary.csv", index=False)

    shares = {}
    for tag in ("1e-5", "1e-9"):
        w = pd.read_csv(RUNS / f"window_share_gtol{tag}.csv")
        for geom, label in (("eq", "eq ADAPT10"), ("str", "str ADAPT10")):
            c_opt = float(w[(w.geometry == geom) & (w.k == 10)].c_opt.iloc[0])
            for n in (50, 400):
                med = float(table[(table.state == label) & (table.method == "II-A") & (table.n_min == n)].shots_median.iloc[0])
                shares[f"{label} gtol {tag} n_min {n}"] = round(med / (med + c_opt), 4)
    facts["selection_share_of_next_iteration"] = shares
    (out / "cas400_facts.json").write_text(json.dumps(facts, indent=1))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print(table.round(3).to_string(index=False))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
