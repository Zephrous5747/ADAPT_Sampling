#!/usr/bin/env python3
"""Paper A, Discussion: shots against measurement-circuit depth.

Collects, for each state, the methods of the study on every family of measurement contexts:

* ``FC`` fully commuting contexts of Part I (the main study; mean 7 to 22 two-qubit gates per circuit);
* ``blocks=4``, ``blocks=2``: contexts that commute block by block, so every circuit is a product of
  Cliffords on 4 or 2 qubits (gate count capped by construction);
* ``QWC``: blocks of one qubit, product measurements without entangling gates;
* ``pivot``: the pivot contexts of Anastasiou et al., as published (every pivot measures a product it
  produces) and merged (each product read from one pivot context).

For every family and method it reports the mean cost, the share of trials that selected the exact best
generator and the mean two-qubit gates per shot (``cz_per_shot_mean``, weighted by the shots each circuit
received), and writes ``runs/paper_a/depth_frontier.csv`` (long form).  Costs are means over trials of
context-shots; the per-state tables are printed.

    python scripts/paper_a_depth_tables.py --cases H4_square_eq_side1p0_CISD LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METHODS = ("M1 static", "M1 seq", "II-0", "II-A")
SAFE = ("M1 seq, safe", "II-0, safe", "II-A, safe")

BLOCK_NAME = {"M1 static": "M1 static", "M1 seq": "M1 seq", "II-0": "II-0", "II-A": "II-A data",
              "M1 seq, safe": "M1 seq, safe", "II-0, safe": "II-0, safe", "II-A, safe": "II-A data, safe"}

# (family label, {method: step5/step4 configuration name})
FAMILIES = [
    ("FC", {"M1 static": "M1 static", "M1 seq": "M1 seq", "II-0": "II-0 estimated", "II-A": "II-A data",
            "M1 seq, safe": "M1 seq, safe", "II-A, safe": "II-A data, safe"}),
    ("blocks=4", {m: f"{BLOCK_NAME[m]}, blocks=4" for m in METHODS + SAFE}),
    ("blocks=2", {m: f"{BLOCK_NAME[m]}, blocks=2" for m in METHODS + SAFE}),
    ("QWC", {m: f"{BLOCK_NAME[m]}, blocks=1" for m in METHODS + SAFE}),
    ("pivot (published)", {"M1 static": "Pivot M1 static", "M1 seq": "Pivot M1 seq", "II-0": "Pivot II-0", "II-A": "Pivot II-A",
                           "M1 seq, safe": "Pivot M1 seq, safe", "II-0, safe": "Pivot II-0, safe",
                           "II-A, safe": "Pivot II-A, safe"}),
    ("pivot (merged)", {"M1 static": "Pivot merged M1 static", "M1 seq": "Pivot merged M1 seq", "II-0": "Pivot merged II-0",
                        "II-A": "Pivot merged II-A", "II-0, safe": "Pivot merged II-0, safe",
                        "II-A, safe": "Pivot merged II-A, safe"}),
]
INDEPENDENT = [("M2 FC", "M2 pairwise"), ("M2 FC, safe", "M2 safe"), ("M2 QWC", "M2 QWC pairwise"),
               ("M2 QWC, safe", "M2 QWC safe"), ("M2 QWC, Successive Elimination", "M2 QWC marginal"),
               ("M2 FC, Successive Elimination", "M2 marginal")]


def load(case: str, root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for stem in ("step4_learned_designs", "step5_external_baselines"):
        path = root / case / f"{case}_{stem}.csv"
        if path.exists():
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    rows[row["config"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a/depth_frontier.csv"))
    args = parser.parse_args()

    out_rows = []
    for case in args.cases:
        rows = load(case, args.root)
        print(f"\n{case}  (mean context-shots; correct share; mean two-qubit gates per shot)")
        for family, mapping in FAMILIES:
            line = []
            for method in METHODS + SAFE:
                name = mapping.get(method, "")
                row = rows.get(name)
                if row is None:
                    continue
                cz = float(row["cz_per_shot_mean_mean"]) if row.get("cz_per_shot_mean_mean") not in (None, "") else float("nan")
                out_rows.append({"case": case, "family": family, "method": method, "config": name,
                                 "trials": row["n_trials"], "shots_mean": float(row["shots_mean"]),
                                 "shots_sem": float(row["shots_sem"]), "shots_median": float(row["shots_median"]),
                                 "correct_rate": float(row["correct_rate"]), "cz_per_shot": cz})
                line.append(f"{method} {float(row['shots_mean']):,.0f} ({float(row['correct_rate']):.2f}, cz {cz:.1f})")
            if line:
                print(f"  {family:18s} " + "; ".join(line))
        for label, name in INDEPENDENT:
            row = rows.get(name)
            if row is None:
                continue
            cz = float(row["cz_per_shot_mean_mean"]) if row.get("cz_per_shot_mean_mean") not in (None, "") else float("nan")
            out_rows.append({"case": case, "family": "independent arms", "method": label, "config": name,
                             "trials": row["n_trials"], "shots_mean": float(row["shots_mean"]),
                             "shots_sem": float(row["shots_sem"]), "shots_median": float(row["shots_median"]),
                             "correct_rate": float(row["correct_rate"]), "cz_per_shot": cz})
            print(f"  {label:18s} {float(row['shots_mean']):,.0f} ({float(row['correct_rate']):.2f}, cz {cz:.1f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
