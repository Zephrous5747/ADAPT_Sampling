#!/usr/bin/env python3
"""Step 6: does the measurement structure of Paper A survive on other operator pools?

For every ``<case>@<pool>`` this records the pool and measurement structure (pool size K,
parent support ``|B_0|``, parent FC contexts, the number of contexts of the per-gradient
groupings, how many gradients each Pauli appears in on average) and Part I's planning
bounds for M1, M2 and M3 (exact gradients and variances, eliminating on each arm's actual
precision).  The planning ratios say whether sharing and elimination still compound when
the generators lose their Jordan-Wigner strings and the pool shrinks or grows; the
shot-level comparison on the same problems is ``step5_external_baselines.py`` with the
``<case>@<pool>`` case ids.

Output: ``runs/step6_pool_census.csv`` and ``runs/step6_pool_census_meta.json``.

    python scripts/step6_pool_census.py --cases H4_square_eq_side1p0_CISD LiH_R3p0_HF --pools uccsd qubit qeb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
import m1_fcug  # noqa: E402  (Part I)
import m2_baifcig  # noqa: E402
import m3_baifcug  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import load_problem  # noqa: E402
from pools import POOLS  # noqa: E402


def census(case: str, pool: str) -> dict:
    started = time.perf_counter()
    problem = load_problem(case if pool == "uccsd" and "@" not in case else f"{case}@{pool}")
    support = problem.universal_support
    uses = sum(len(t) for t in problem.commutator_terms) / max(len(support), 1)
    groups = problem.parent_fc_groups()
    individual = problem.individual_fc_groups()
    gap = problem.top_gap()
    row = {
        "case": case, "pool": pool, "K": problem.n_generators, "qubits": problem.n_qubits,
        "support_B0": len(support), "parent_contexts": len(groups),
        "individual_contexts": int(sum(len(g) for g in individual)),
        "uses_per_pauli": round(uses, 3),
        "mean_terms_per_gradient": round(sum(len(t) for t in problem.commutator_terms) / problem.n_generators, 1),
        "top_gradient": float(problem.abs_gradients.max()), "gap": gap,
        "nonzero_gradients": problem.n_nonzero_gradients,
    }
    if gap > 1e-9:
        m1 = m1_fcug.run(problem).total_shots
        m2 = m2_baifcig.run(problem).total_shots
        m3 = m3_baifcug.run_actual_radii(problem)["total_shots"]
        row.update({"M1": m1, "M2": m2, "M3": m3, "M1/M2": m1 / m2, "M2/M3": m2 / m3, "M1/M3": m1 / m3})
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "H4_square_eq_side1p0_HF",
                                                       "LiH_R3p0_HF"])
    parser.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        for pool in args.pools:
            row = census(case, pool)
            rows.append(row)
            print(f"{case:28s} {pool:6s} K={row['K']:4d} |B0|={row['support_B0']:6d} N={row['parent_contexts']:5d} "
                  f"uses={row['uses_per_pauli']:5.2f}"
                  + (f"  M1/M2/M3 = {row['M1']:.3g} / {row['M2']:.3g} / {row['M3']:.3g}"
                     f"  (M1/M3 {row['M1/M3']:.2f}, M2/M3 {row['M2/M3']:.2f})" if "M1" in row else "  (tied leaders)")
                  + f"  [{row['seconds']:.0f}s]", flush=True)
    # merge with earlier invocations: one row per (case, pool), newest wins
    path = args.out / "step6_pool_census.csv"
    merged: dict[tuple, dict] = {}
    if path.exists():
        import csv

        with path.open() as handle:
            for old in csv.DictReader(handle):
                merged[(old["case"], old["pool"])] = old
    for row in rows:
        merged[(row["case"], row["pool"])] = row
    all_rows = list(merged.values())
    columns = list(dict.fromkeys(k for r in all_rows for k in r))
    write_csv(path, columns, all_rows)
    write_json(args.out / "step6_pool_census_meta.json",
               {"bounds": "Part I planning bounds: M1 identification radius gap/2, M2 exact schedule, M3 on actual precision",
                "run": run_record()})


if __name__ == "__main__":
    main()
