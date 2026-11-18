#!/usr/bin/env python3
"""Add the summary rows of another step-5 summary CSV to the local one, by config label, without touching existing rows.

A job on another cluster writes ``runs/<case>/<case>_step5_external_baselines.csv`` with the rows of its own configs only; copying it over the local file
would drop the rows of the other configs.  Rows whose ``config`` is already present are kept as they are (a warning if the numbers differ).

    python scripts/step5_merge_summary.py --other <dir with <case>/<case>_step5_external_baselines.csv> --cases H2O_dz8o_str_CISD ...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    for case in args.cases:
        name = f"{case}_step5_external_baselines.csv"
        mine, theirs = args.runs / case / name, args.other / case / name
        with mine.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        with theirs.open(newline="") as handle:
            other = list(csv.DictReader(handle))
        have = {r["config"]: r for r in rows}
        added = 0
        for r in other:
            if r["config"] in have:
                if abs(float(r["shots_mean"]) - float(have[r["config"]]["shots_mean"])) > 1e-6 * abs(float(r["shots_mean"])):
                    print(f"  warning: {case}: '{r['config']}' present with other numbers; kept the local row")
                continue
            rows.append({k: r.get(k, "") for k in fields})
            added += 1
        with mine.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{case}: {added} row(s) added, {len(rows)} in all")


if __name__ == "__main__":
    main()
