#!/usr/bin/env python3
"""Is her number a plausible draw from our larger run set?  (statistical companion of ``features_compare_tzs.py``)

Seeds do not carry over between builds of the Hamiltonian (the SCF orbitals of degenerate subspaces differ, which changes
which generators tie and the optimiser's path), so a re-run cannot be expected to give her numbers digit for digit.  What can be
tested is whether her statistic, computed on her ``n`` runs, lies in the distribution of the same statistic over random subsets of
``n`` runs of our larger sets (``runs/features_ext``: 48 trajectories per level for H4, 24 for LiH).  For every cell the table gives our
value on all runs, the 2.5 / 97.5 percentiles of the subset distribution, and the percentile at which her value falls.

    python scripts/features_compare_stats.py [--runs runs/features_ext] [--draws 20000]

Quantities: median selection shots, mean optimiser evaluations (absolute), exact-best share of the selections, non-rho-good selections
per run.  Her numbers are those printed in ``reports/TZS_three experiments.tex`` (Table 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Table 2 of the report: system -> level (% confidence) -> (runs, median shots, mean steps, mean evals, exact-best %, rho-good %)
HERS = {
    "H4_square_eq_side1p0_HF": {99: (12, 1.5e6, 10.0, 353, 91, 100), 97: (12, 1.2e6, 10.0, 287, 90, 100), 95: (12, 1.2e6, 10.0, 285, 86, 100),
                                90: (12, 8.2e5, 10.0, 275, 88, 100), 80: (12, 7.6e5, 10.0, 294, 88, 99), 68: (12, 5.7e5, 10.0, 271, 91, 100)},
    "H4_square_stretch_side2p0_HF": {99: (12, 1.4e7, 10.0, 200, 94, 100), 97: (12, 1.2e7, 10.0, 202, 98, 100), 95: (12, 9.5e6, 10.0, 217, 95, 100),
                                     90: (12, 6.4e6, 10.0, 205, 93, 100), 80: (12, 5.7e6, 10.0, 241, 88, 100), 68: (12, 4.8e6, 10.0, 230, 92, 100)},
    "LiH_R3p0_HF": {99: (2, 8.6e7, 6.0, 96, 92, 100), 95: (2, 4.2e7, 6.0, 96, 92, 100), 68: (3, 3.7e7, 6.0, 97, 89, 100)},
}


def load(root: Path, case: str) -> dict[float, list[dict]]:
    out: dict[float, dict[int, dict]] = {}
    for path in sorted((root / case).glob("ci_II-A_data_safe_d*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(round(100 * (1 - r["delta"])), {})[int(r["index"])] = r
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def stats(runs: list[dict]) -> dict:
    sel = sum(len(r["selections"]) for r in runs)
    return {"shots": float(np.median([r["selection_shots"] for r in runs])), "evals": float(np.mean([r["optimizer_evaluations"] for r in runs])),
            "exact_best": 100 * (1 - sum(len(r["selections"]) - r["exact_best"] for r in runs) / sel),
            "steps": float(np.mean([r["iterations"] for r in runs]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/features_ext"))
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    print(f"{'system':32s} {'conf':>4s} {'n_her':>5s} {'quantity':>11s} {'hers':>10s} {'ours(all)':>10s} {'2.5%':>10s} {'97.5%':>10s} {'her pctile':>10s}")
    inside = total = 0
    records = []
    for case, levels in HERS.items():
        data = load(args.runs, case)
        for conf, (n, shots, steps, evals, best, _) in levels.items():
            runs = data.get(conf)
            if not runs or len(runs) <= n:
                continue
            allv = stats(runs)
            draws = {k: [] for k in ("shots", "evals", "exact_best")}
            for _ in range(args.draws if n >= 5 else 2000):
                sub = [runs[i] for i in rng.choice(len(runs), n, replace=False)]
                s = stats(sub)
                for k in draws:
                    draws[k].append(s[k])
            for key, hers in (("shots", shots), ("evals", evals), ("exact_best", best)):
                d = np.array(draws[key])
                pct = 100 * (np.mean(d < hers) + 0.5 * np.mean(d == hers))
                lo, hi = np.percentile(d, [2.5, 97.5])
                total += 1
                inside += (lo <= hers <= hi)
                records.append({"case": case, "confidence_pct": conf, "n_her": n, "quantity": key, "hers": hers, "ours_all": allv[key],
                                "lo": float(lo), "hi": float(hi), "her_percentile": float(pct), "inside": bool(lo <= hers <= hi)})
                print(f"{case[:32]:32s} {conf:4d} {n:5d} {key:>11s} {hers:10.4g} {allv[key]:10.4g} {lo:10.4g} {hi:10.4g} {pct:9.1f}%")
    print(f"\nher value inside the 95% range of n-run subsets of our runs: {inside}/{total}")
    import pandas as pd

    df = pd.DataFrame(records)
    for q, g in df.groupby("quantity"):
        print(f"  {q:>11s}: {int(g.inside.sum())}/{len(g)} inside; median of (hers / ours) = {float((g.hers / g.ours_all).median()):.3f}")
    out = args.runs.parent / "features" / "report"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "compare_stats.csv", index=False)


if __name__ == "__main__":
    main()
