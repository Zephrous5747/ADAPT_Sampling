#!/usr/bin/env python3
"""Does a lower confidence level of the selection cost the optimisation anything?  Non-inferiority against the highest level.

Reads the measured ADAPT runs of the confidence sweep (``runs/features/<case>/ci_*.jsonl`` and, when present, the larger sets
``runs/features_ext/<case>/``; a run is identified by (method, delta, index) and the larger set wins) and compares, per case, every level
with the highest one on

* the optimiser evaluations of a run, and the weighted count  sum_k ev_k (1 + 2 n_k)  (the cost model without the energy-shot factor);
* the modelled optimisation cost C_opt and the selection shots;
* the number of steps and the share of runs that reach the target.

The ratio of means level / highest level has a bootstrap interval (10,000 resamples of the runs of both levels); a level is
*non-inferior* when the upper end of the 95 % interval is below ``--margin`` (default 1.15).  The same ratios are given against the
tie-averaged exact-ADAPT baseline (``rho_random:0`` of ``energy_trajectory_policies.py``, when ``runs/energy_policies`` has it), which
replaces the evaluation count of the single exact path as the reference (that count depends on how ties are broken).

    python scripts/energy_conf_noninferiority.py [--runs runs/features runs/features_ext] [--margin 1.15]

Writes ``runs/features/report/conf_noninferiority.csv``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {"H4_square_eq_side1p0_HF": "H4 1.0", "H4_square_stretch_side2p0_HF": "H4 2.0", "LiH_R3p0_HF": "LiH 3.0",
         "H2O_eq_HF": "H2O eq", "H2O_stretch_HF": "H2O str"}
SLACK = 1e-4


def load_case(case: str, roots: list[Path]) -> tuple[pd.DataFrame, dict | None]:
    rows, ref = {}, None
    for root in roots:  # later roots win
        d = root / case
        if not d.exists():
            continue
        if (d / "exact_reference.json").exists() and ref is None:
            ref = json.loads((d / "exact_reference.json").read_text())
        for path in sorted(d.glob("ci_*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r["method"].startswith("II-A"):
                    continue
                weighted = float(sum(e * (1 + 2 * n) for e, n in zip(r["evaluations"], r["n_params"])))
                rows[(r["method"], round(r["delta"], 6), int(r["index"]), r.get("source", "measured"))] = {
                    "delta": round(r["delta"], 6), "index": int(r["index"]), "source": r.get("source", "measured"),
                    "steps": r["iterations"], "final_error": r["final_error"], "evals": r["optimizer_evaluations"],
                    "weighted": weighted, "c_opt": r["c_opt_eta2"], "selection": r["selection_shots"],
                    "non_best": len(r["selections"]) - r["exact_best"], "not_rho_good": len(r["selections"]) - r["rho_good"]}
    return pd.DataFrame(list(rows.values())), ref


def boot_ratio(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float, float]:
    """mean(a) / mean(b) with a percentile bootstrap interval."""
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    ratios = a[ia].mean(axis=1) / b[ib].mean(axis=1)
    return float(a.mean() / b.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[Path("runs/features"), Path("runs/features_ext")])
    parser.add_argument("--policies", type=Path, default=Path("runs/energy_policies"))
    parser.add_argument("--margin", type=float, default=1.15)
    parser.add_argument("--cases", nargs="*", default=list(LABEL))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for case in args.cases:
        df, ref = load_case(case, args.runs)
        if df.empty:
            continue
        target = 1.6e-3 if ref is None or ref["final_error"] < 1.6e-3 else ref["final_error"] + SLACK
        base_evals = base_weighted = np.nan
        tie = args.policies / case / "policy_rho_random_0.jsonl"
        if tie.exists():
            tr = [json.loads(x) for x in tie.read_text().splitlines() if x.strip()]
            base_evals = float(np.mean([r["evals_total"] for r in tr]))
            base_weighted = float(np.mean([r["evals_weighted"] for r in tr]))
        levels = sorted(df.delta.unique())
        top = df[df.delta == levels[0]]  # smallest delta = highest confidence
        for delta in levels:
            g = df[df.delta == delta]
            row = {"case": case, "system": LABEL.get(case, case), "confidence_pct": 100 * (1 - delta), "n": len(g),
                   "reached_pct": 100 * float((g.final_error <= target).mean()), "steps_mean": g.steps.mean(),
                   "selection_median": g.selection.median(), "evals_mean": g.evals.mean(),
                   "evals_se": g.evals.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                   "not_rho_good_per_run": g.not_rho_good.mean(), "non_best_per_run": g.non_best.mean(),
                   "evals_vs_tie_avg": g.evals.mean() / base_evals, "weighted_vs_tie_avg": g.weighted.mean() / base_weighted}
            if len(g) > 1 and len(top) > 1:
                for q in ("evals", "weighted", "c_opt"):
                    r, lo, hi = boot_ratio(g[q].to_numpy(float), top[q].to_numpy(float), rng)
                    row[f"{q}_ratio_top"], row[f"{q}_lo"], row[f"{q}_hi"] = r, lo, hi
                row["non_inferior_evals"] = bool(row["evals_hi"] <= args.margin)
                row["non_inferior_c_opt"] = bool(row["c_opt_hi"] <= args.margin)
            out.append(row)
    t = pd.DataFrame(out)
    if t.empty:
        print("no sweep data")
        return
    rep = args.runs[0] / "report"
    rep.mkdir(parents=True, exist_ok=True)
    t.to_csv(rep / "conf_noninferiority.csv", index=False)
    cols = ["system", "confidence_pct", "n", "reached_pct", "steps_mean", "selection_median", "evals_mean", "evals_ratio_top",
            "evals_lo", "evals_hi", "non_inferior_evals", "c_opt_ratio_top", "c_opt_lo", "c_opt_hi", "non_inferior_c_opt",
            "evals_vs_tie_avg", "not_rho_good_per_run"]
    print(t[[c for c in cols if c in t]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
