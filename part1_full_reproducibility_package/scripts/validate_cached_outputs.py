#!/usr/bin/env python3
"""Validate cached outputs against selected numbers in the Part I report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

EXPECTED = {
    ("LiH, R=3.0 A", "HF"): {"M1_FCUG": 2088380, "M2_BAIFCIG": 7368553, "M3_BAIFCUG": 1267508},
    ("H4 square, side=1.0 A", "HF"): {"M1_FCUG": 182282619, "M2_BAIFCIG": 451756, "M3_BAIFCUG": 45857},
    ("H4 square, side=1.0 A", "CISD"): {"M1_FCUG": 6066911, "M2_BAIFCIG": 53265, "M3_BAIFCUG": 9582},
    ("H4 square, side=2.0 A", "HF"): {"M1_FCUG": 70335943, "M2_BAIFCIG": 112099, "M3_BAIFCUG": 9936},
    ("H4 square, side=2.0 A", "CISD"): {"M1_FCUG": 7295352, "M2_BAIFCIG": 19188, "M3_BAIFCUG": 3441},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="reproduced_tables")
    args = ap.parse_args()
    path = Path(args.tables_dir) / "table_completed_three_methods.csv"
    df = pd.read_csv(path)
    failures = []
    for _, r in df.iterrows():
        key = (r["case"], r["state"])
        if key not in EXPECTED:
            continue
        for col, exp in EXPECTED[key].items():
            got = int(round(float(r[col])))
            if got != exp:
                failures.append((key, col, exp, got))
    if failures:
        print("VALIDATION FAILED")
        for key, col, exp, got in failures:
            print(f"  {key} {col}: expected {exp}, got {got}")
        sys.exit(1)
    print("VALIDATION PASSED: cached tables match expected Part I report values for completed M1/M2/M3 rows.")

if __name__ == "__main__":
    main()
