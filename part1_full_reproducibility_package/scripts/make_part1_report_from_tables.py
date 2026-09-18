#!/usr/bin/env python3
"""Generate a minimal markdown report from reproduced Part I tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

TABLES = [
    ("Observable structure", "table_observable_structure.csv"),
    ("Gradient difficulty and M2 diagnostic", "table_gradient_difficulty_m2.csv"),
    ("Completed M1/M2/M3 rows", "table_completed_three_methods.csv"),
    ("H2O status", "table_h2o_status.csv"),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="reproduced_tables")
    ap.add_argument("--out", default="reproduced_tables/part1_tables_report.md")
    args = ap.parse_args()
    td = Path(args.tables_dir)
    lines = ["# Part I reproduced tables", "", "Generated from cached CSV/JSON outputs."]
    for title, fname in TABLES:
        df = pd.read_csv(td / fname)
        lines.extend(["", f"## {title}", "", df.to_markdown(index=False)])
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
