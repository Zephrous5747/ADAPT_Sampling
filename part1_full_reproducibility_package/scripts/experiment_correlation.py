"""Table: M3 under correct joint noise, with the marginal and pairwise rules.

The pairwise rule compares each pair at the confidence radius of the combination
the comparison actually depends on.  Because elimination compares |g_i| against
|g_j|, that combination is g_i - s_ij g_j with s_ij the relative sign of the two
estimates, not the signed difference; the two differ whenever the leading pair
has opposite signs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cases import CASES, get_case          # noqa: E402
from finite_shot import simulate            # noqa: E402
from problem_cache import load_or_build     # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=".cache")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    problem = load_or_build(get_case(args.case), cache_dir=args.cache)
    summary_path = Path(args.out) / args.case / f"{args.case}_M3_BAIFCUG_summary.json"
    bound = json.loads(summary_path.read_text())["total_context_shots"]

    rows = []
    for noise, rule in (("independent", "marginal"),
                        ("correlated", "marginal"),
                        ("correlated", "pairwise")):
        result = simulate(
            problem, "m3", planning_bound=bound, n_trials=args.trials,
            seed=args.seed, noise_model=noise, rule=rule,
        )
        row = result.as_row() | {"noise_model": noise, "rule": rule}
        rows.append(row)
        print(f"{noise:12s} {rule:9s} shots={row['shots_mean']:14,.0f}"
              f" +-{row['shots_sem']:10,.0f}  correct={row['correct_rate']:6.1%}"
              f" [{row['correct_low']:.3f},{row['correct_high']:.3f}]")

    out = Path(args.out) / args.case / f"{args.case}_correlation.json"
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
