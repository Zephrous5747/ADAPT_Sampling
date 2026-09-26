#!/usr/bin/env python3
"""Collect the per-case outputs of Steps 1-4 into combined tables.

Each step writes ``runs/<case>/<case>_<step>.csv``; runs over subsets of cases are
therefore never lost when a later run covers different cases.  This script
concatenates them into ``runs/<step>.csv`` for every step in ``TABLES``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outputs import write_csv  # noqa: E402
from part1_bridge import CASES  # noqa: E402

TABLES = {
    "step1_online_baseline": "{case}_step1_online_baseline.csv",
    "step2_overlap_census": "{case}_overlap_census.csv",
    "step3_oracle_ceiling": "{case}_step3_oracle_ceiling.csv",
    "step4_learned_designs": "{case}_step4_learned_designs.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    for table, pattern in TABLES.items():
        rows = []
        for case in CASES:
            path = args.runs / case / pattern.format(case=case)
            if path.exists():
                with path.open() as handle:
                    rows.extend({"case_id": case, **row} for row in csv.DictReader(handle))
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            write_csv(args.runs / f"{table}.csv", columns, rows)
            print(f"{table}: {len(rows)} rows from {len({r['case_id'] for r in rows})} cases")


if __name__ == "__main__":
    main()
