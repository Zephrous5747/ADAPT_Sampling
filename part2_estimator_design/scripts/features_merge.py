#!/usr/bin/env python3
"""Put the pieces written by parallel cluster jobs into the layout that ``features_report.py`` reads.

The notebook's scripts write one table per process; when the cases run as separate jobs their files would overwrite
each other, so the cluster jobs write ``runs/features/s1/<case>/energy_lowering_*.csv`` (S1) and
``runs/features/gradient_correlation_pairs_<case>.csv`` (Table 1, H4 and LiH).  This script concatenates them into

* ``runs/features/energy_lowering_generators.csv`` and ``energy_lowering_states.csv``
* ``runs/features/gradient_correlation_pairs.csv``

(the H2O pieces ``gradient_correlation_pairs_h2o_*.csv`` are read by the report as they are).

    python scripts/features_merge.py [--runs runs/features]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CASE_ORDER = ["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF", "H2O_eq_HF", "H2O_stretch_HF"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/features"))
    args = parser.parse_args()
    runs = args.runs

    for name in ("energy_lowering_generators.csv", "energy_lowering_states.csv"):
        parts = [runs / "s1" / case / name for case in CASE_ORDER if (runs / "s1" / case / name).exists()]
        if parts:
            pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(runs / name, index=False)
            print(f"{name}: {len(parts)} cases")
    parts = [runs / f"gradient_correlation_pairs_{case}.csv" for case in CASE_ORDER[:3]
             if (runs / f"gradient_correlation_pairs_{case}.csv").exists()]
    if parts:
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
            runs / "gradient_correlation_pairs.csv", index=False)
        print(f"gradient_correlation_pairs.csv: {len(parts)} cases")
    print("H2O Table 1 pieces:", len(list(runs.glob("gradient_correlation_pairs_h2o*.csv"))))


if __name__ == "__main__":
    main()
