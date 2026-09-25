#!/usr/bin/env python3
"""Merge sharded per-trial files into the step summaries.

Each sharded run writes ``runs/<case>/trials/<step>__<label>__shard<K>of<N>.csv``.
This script collects every shard of every label, checks that each trial index
``0 .. n_trials-1`` appears exactly once, and writes the same summary files an
unsharded run would:

* ``step4`` -> ``runs/<case>/<case>_step4_learned_designs.csv``
* ``step1`` -> ``runs/<case>/<case>_step1_baseline_merged.csv``

Example::

    python scripts/merge_trials.py --step step4 --cases LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from online import OnlineOutcome, summarise  # noqa: E402
from outputs import write_csv  # noqa: E402

OUTPUT = {"step4": "{case}_step4_learned_designs.csv", "step1": "{case}_step1_baseline_merged.csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=list(OUTPUT), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    for case in args.cases:
        by_label: dict[str, list[dict]] = defaultdict(list)
        for path in sorted((args.runs / case / "trials").glob(f"{args.step}__*.csv")):
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    by_label[row["label"]].append(row)
        summaries = []
        for label, rows in by_label.items():
            n_trials = int(rows[0]["n_trials"])
            indices = sorted(int(r["trial"]) for r in rows)
            if indices != list(range(n_trials)):
                missing = sorted(set(range(n_trials)) - set(indices))
                duplicated = len(indices) - len(set(indices))
                raise SystemExit(f"{case} / {label}: {len(missing)} trials missing, {duplicated} duplicated")
            outcomes = [OnlineOutcome(int(r["selected"]), bool(int(r["correct"])), float(r["shots"]),
                                      int(r["rounds"])) for r in rows]
            summary = {"case_id": case, "config": label, **summarise(outcomes)}
            if "designs_kept_last_refit" in rows[0]:
                summary["mean_designs_kept_last_refit"] = sum(
                    float(r["designs_kept_last_refit"]) for r in rows) / len(rows)
            summaries.append(summary)
            print(f"{case:30s} {label:28s} trials {n_trials}  shots {summary['shots_mean']:14,.0f} "
                  f"+- {summary['shots_sem']:10,.0f}  correct {summary['correct_rate']:.3f}")
        if summaries:
            columns = list(dict.fromkeys(k for s in summaries for k in s))
            write_csv(args.runs / case / OUTPUT[args.step].format(case=case), columns, summaries)


if __name__ == "__main__":
    main()
