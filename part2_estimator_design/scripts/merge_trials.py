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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from online import summarise  # noqa: E402
from outputs import write_csv  # noqa: E402
from parallel import check_complete, outcomes_from_rows, read_trials  # noqa: E402

OUTPUT = {"step4": "{case}_step4_learned_designs.csv", "step1": "{case}_step1_baseline_merged.csv",
          "step5": "{case}_step5_external_baselines.csv", "step7": "{case}_step7_ic_baseline.csv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=list(OUTPUT), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="skip (with a warning) labels whose trials are not all present yet, e.g. while "
                             "cluster shards are still running; the default refuses, which is the safe choice for final tables")
    args = parser.parse_args()

    for case in args.cases:
        summaries = merge_case(args.runs, case, args.step, allow_incomplete=args.allow_incomplete)
        for summary in summaries:
            print(f"{case:30s} {summary['config']:28s} trials {summary['n_trials']}  "
                  f"shots {summary['shots_mean']:14,.0f} +- {summary['shots_sem']:10,.0f}  "
                  f"correct {summary['correct_rate']:.3f}")


def merge_case(runs: Path, case: str, step: str, allow_incomplete: bool = False) -> list[dict]:
    """Summaries of every label with a complete set of trials; writes the step's CSV."""
    summaries = []
    for label, rows in read_trials(runs, case, step).items():
        if allow_incomplete:
            try:
                check_complete(case, label, rows)
            except SystemExit as incomplete:
                print(f"skipped (incomplete): {incomplete}")
                continue
        else:
            check_complete(case, label, rows)
        summary = {"case_id": case, "config": label, **summarise(outcomes_from_rows(rows))}
        if "designs_kept_last_refit" in rows[0]:
            summary["mean_designs_kept_last_refit"] = sum(
                float(r["designs_kept_last_refit"]) for r in rows) / len(rows)
        summaries.append(summary)
    if summaries:
        columns = list(dict.fromkeys(k for s in summaries for k in s))
        write_csv(Path(runs) / case / OUTPUT[step].format(case=case), columns, summaries)
    return summaries

if __name__ == "__main__":
    main()
