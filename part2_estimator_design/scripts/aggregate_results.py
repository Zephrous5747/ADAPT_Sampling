#!/usr/bin/env python3
"""Collect the per-case outputs of Steps 1-4 into combined tables.

Each step writes ``runs/<case>/<case>_<step>.csv``; runs over subsets of cases are
therefore never lost when a later run covers different cases.  This script
concatenates them into ``runs/<step>.csv`` for every step in ``TABLES``.  Cases are
the Part I cases followed by any other case directory (ADAPT trajectory states).
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
    "step5_external_baselines": "{case}_step5_external_baselines.csv",
    "step7_ic_baseline": "{case}_step7_ic_baseline.csv",
    "phase2_adapt_trajectories": "{case}_adapt_trajectory.csv",
    "phase4_adapt_selection": "phase4/{case}_phase4_summary.csv",
}
# Tagged Phase 4 runs (other methods, ``--tag``) are collected into one more table.
PHASE4_TAGGED = "phase4/{case}_phase4_*_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    extra = sorted(p.name for p in args.runs.iterdir() if p.is_dir() and p.name not in CASES
                   and not p.name.startswith("step"))
    for table, pattern in TABLES.items():
        rows = []
        for case in list(CASES) + extra:
            path = args.runs / case / pattern.format(case=case)
            if path.exists():
                with path.open() as handle:
                    rows.extend({**row, "case_id": row.get("case_id") or case} for row in csv.DictReader(handle))
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            write_csv(args.runs / f"{table}.csv", columns, rows)
            print(f"{table}: {len(rows)} rows from {len({r['case_id'] for r in rows})} cases")
    rows = []
    for case in list(CASES) + extra:
        for path in sorted((args.runs / case / "phase4").glob(f"{case}_phase4_*_summary.csv")) if (args.runs / case / "phase4").exists() else []:
            with path.open() as handle:
                rows.extend({**row, "case_id": row.get("case_id") or case, "tag": path.stem.split("_phase4_")[-1][: -len("_summary")]}
                            for row in csv.DictReader(handle))
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(args.runs / "phase4_adapt_selection_tagged.csv", columns, rows)
        print(f"phase4_adapt_selection_tagged: {len(rows)} rows")


if __name__ == "__main__":
    main()
