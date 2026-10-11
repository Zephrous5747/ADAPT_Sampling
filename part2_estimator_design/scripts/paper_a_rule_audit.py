#!/usr/bin/env python3
"""Paper A (referee point 4): which stored fixed-state rows use the pairwise rule, and where a sign-aware row is missing.

For every state with a pairwise II-0 / II-A row (``II-0 estimated``, ``II-A data``: oracle starting radius, the rule
the manuscript's headline percentages came from) the script reports

* the share of trials that selected the exact best generator in the pairwise rows (below 0.95: the row is not valid);
* whether the sign-aware counterparts exist: ``II-0 safe`` (step 4), ``II-A data, safe`` (step 5), and the rows with
  the a priori starting radius, ``II-0 safe, bound start`` and ``II-A data, safe, bound start``;
* the saving of II-A over II-0 in the pairwise, the sign-aware and the a priori setting (mean cost ratio).

The output ``runs/paper_a/rule_audit.csv`` lists the gaps, so that only they are rerun.

    python scripts/paper_a_rule_audit.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outputs import write_csv  # noqa: E402

PAIRS = {  # setting -> (II-0 row, II-A row)  as (step, label)
    "pairwise, oracle start": (("step4", "II-0 estimated"), ("step4", "II-A data")),
    "sign-aware, oracle start": (("step4", "II-0 safe"), ("step5", "II-A data, safe")),
    "sign-aware, a priori start": (("step4", "II-0 safe, bound start"), ("step5", "II-A data, safe, bound start")),
}


def rows_of(path: Path) -> dict[str, dict]:
    return {r["config"]: r for r in csv.DictReader(path.open())} if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    out = []
    for case_dir in sorted(p for p in args.runs.iterdir() if p.is_dir() and "@" not in p.name):
        case = case_dir.name
        tables = {"step4": rows_of(case_dir / f"{case}_step4_learned_designs.csv"),
                  "step5": rows_of(case_dir / f"{case}_step5_external_baselines.csv")}
        pw = [tables[s].get(l) or tables["step5"].get(l) or tables["step4"].get(l) for s, l in PAIRS["pairwise, oracle start"]]
        if not any(pw):
            continue
        record = {"case": case}
        for setting, (a, b) in PAIRS.items():
            ra = tables[a[0]].get(a[1]) or tables["step5"].get(a[1])
            rb = tables[b[0]].get(b[1]) or tables["step4"].get(b[1])
            tag = setting.split(",")[0] + ("_oracle" if "oracle" in setting else "_apriori")
            record[f"{tag}_ii0"] = float(ra["shots_mean"]) if ra else float("nan")
            record[f"{tag}_iia"] = float(rb["shots_mean"]) if rb else float("nan")
            record[f"{tag}_correct_iia"] = float(rb["correct_rate"]) if rb else float("nan")
            record[f"{tag}_saving"] = (1.0 - float(rb["shots_mean"]) / float(ra["shots_mean"])) if (ra and rb) else float("nan")
        record["gap"] = ", ".join(s for s in ("sign-aware, oracle start", "sign-aware, a priori start")
                                  if any(r is None for r in [tables[a[0]].get(a[1]) or tables["step5"].get(a[1]) for a in (PAIRS[s][0],)]
                                         + [tables[b[0]].get(b[1]) or tables["step4"].get(b[1]) for b in (PAIRS[s][1],)]))
        out.append(record)
    write_csv(args.runs / "paper_a" / "rule_audit.csv", list(out[0]), out)
    print(f"{'state':34s} {'pairwise: correct, saving':>28s} {'sign-aware oracle: correct, saving':>36s} {'a priori: correct, saving':>28s}   missing")
    for r in out:
        def cell(tag):
            c, s = r[f"{tag}_correct_iia"], r[f"{tag}_saving"]
            return "          --" if s != s else f"{100 * c:5.0f}%  {100 * s:5.0f}%"
        print(f"{r['case']:34s} {cell('pairwise_oracle'):>28s} {cell('sign-aware_oracle'):>36s} {cell('sign-aware_apriori'):>28s}   {r['gap']}")


if __name__ == "__main__":
    main()
