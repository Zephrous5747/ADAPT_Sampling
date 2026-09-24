#!/usr/bin/env python3
"""Step 3: oracle ceiling of coefficient splitting (II-A) and zero-sum ghosts (II-B).

Every number is an oracle planning bound in Part I's sense: exact gradients set
the elimination trajectory and exact covariances drive the designs.  They are
therefore *ceilings* on what a level could deliver; Step 4 has to show how much
survives when the covariances are learned.  For each case, completion rule and
level the script reports

* M1 at the identification radius ``gap/2`` (one joint design for the pool);
* M3 on Part I's exact-limit trajectory with one static design, under Part I's
  per-breakpoint "maxima" accounting and under top-up accounting;
* M3 eliminating on actual precision (Part I's spillover column) with that design;
* optionally, M3 with the design re-optimised for the survivors at every
  breakpoint (oracle II-D), under both accountings.

II-0 is Part I's estimator on Part I's contexts and reproduces Part I's numbers.
A level passes the gate if it removes at least 10% of the II-0 cost on the
primary metric, M3 eliminating on actual precision (the planning analogue of the
Part II baseline).  Every design is checked for ``A = BC`` and saved as the pair
``(B, C)``.

Example::

    python scripts/step3_oracle_ceiling.py --cases H4_square_eq_side1p0_HF --redesign
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401  (puts Part I's src on the path)
import m1_fcug  # noqa: E402  (Part I)
import m3_baifcug  # noqa: E402
from contexts import STRATEGIES, build_context_library  # noqa: E402
from design import LEVELS, DesignSet, assert_exact, build_fragment_problems  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import (  # noqa: E402
    CASES,
    DEFAULT_DELTA,
    epsilon_from_radius,
    gradient_matrix,
    load_problem,
    part1_summary,
    z_from_delta,
)
from planning import actual_precision, m3_common_radius  # noqa: E402
from sampler import OracleMoments  # noqa: E402

GATE = 0.10
PRIMARY = "m3_actual_static"


def reference_row(problem, z) -> dict:
    """Part I's own numbers for this problem, recomputed with Part I's modules."""
    summary = part1_summary(problem.case_id)
    return {
        "m1": float(m1_fcug.run(problem).shots_per_context.sum()),
        "m2": float(summary["M2_BAIFCIG"]),
        "m3_bound": float(m3_baifcug.run(problem).shots_per_context.sum()),
        "m3_actual": float(m3_baifcug.run_actual_radii(problem)["total_shots"]),
    }


def evaluate_level(problem, library, moments, level, z, *, aux_cap, redesign, out_dir, stem, log):
    A = gradient_matrix(problem, library.n_library)
    n = problem.n_generators
    arms = list(range(n))
    started = time.perf_counter()
    problems = build_fragment_problems(problem, library, moments, level, aux_cap=aux_cap)
    design = DesignSet(problems, library.n_contexts)
    built = time.perf_counter() - started

    started = time.perf_counter()
    static = design.solve(arms, 1.0)
    static_seconds = time.perf_counter() - started
    residual = assert_exact(design, A)
    B, C = design.reconstruction(library.n_library)
    sp.save_npz(out_dir / f"{stem}_B.npz", B)
    sp.save_npz(out_dir / f"{stem}_C.npz", C)
    sigmas = design.sigmas(arms)
    snapshot = design.snapshot()

    epsilon_m1 = epsilon_from_radius(problem.top_gap() / 2.0, z)
    row = {
        "level": level,
        "coordinates": int(sum(p.n_coordinates for p in problems)),
        "splittable_paulis": int(sum(p.n_splittable for p in problems)),
        "auxiliary_coordinates": int(
            sum(int((p.pauli_target[np.searchsorted(p.pauli_ids, p.coord_pauli)] == 0).sum()) for p in problems)
        ),
        "build_seconds": round(built, 1),
        "static_solve_seconds": round(static_seconds, 1),
        "static_outer_iterations": static.iterations,
        "abs_residual_A_minus_BC": residual,
        "m1": static.total / epsilon_m1 ** 2,
    }
    common = m3_common_radius(design, problem.abs_gradients, z, static_solution=True)
    row["m3_bound_static_maxima"] = common["total"]
    topup = m3_common_radius(design, problem.abs_gradients, z, accounting="topup", static_solution=True)
    row["m3_bound_static_topup"] = topup["total"]
    actual = actual_precision(problem.abs_gradients, z, sigmas=sigmas)
    row["m3_actual_static"] = actual["total"]
    ranking = problem.ranking()
    row["winner_recon_variance"] = float((sigmas[ranking[0]] ** 2).sum())
    binding = int(np.argmax((sigmas ** 2).sum(axis=1)))
    row["binding_generator"] = problem.labels[binding]
    row["binding_recon_variance"] = float((sigmas[binding] ** 2).sum())
    log(
        f"  {level}: M1 {row['m1']:,.0f}  M3 bound {row['m3_bound_static_maxima']:,.0f} "
        f"(top-up {row['m3_bound_static_topup']:,.0f})  M3 actual {row['m3_actual_static']:,.0f}  "
        f"[{static_seconds:.0f}s, {static.iterations} outer]"
    )

    rounds = []
    if redesign and level != "II-0":
        for accounting in ("maxima", "topup"):
            design.restore(snapshot)
            started = time.perf_counter()
            result = m3_common_radius(
                design, problem.abs_gradients, z, redesign=True, accounting=accounting
            )
            assert_exact(design, A)
            row[f"m3_bound_redesign_{accounting}"] = result["total"]
            row[f"redesign_{accounting}_seconds"] = round(time.perf_counter() - started, 1)
            for record in result["history"]:
                rounds.append({"accounting": accounting, **record})
            log(f"  {level}: M3 redesign ({accounting}) {result['total']:,.0f} [{row[f'redesign_{accounting}_seconds']:.0f}s]")
        write_csv(out_dir / f"{stem}_redesign_rounds.csv", list(rounds[0]), rounds)
        design.restore(snapshot)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--strategies", nargs="+", default=["mass", "canonical"], choices=STRATEGIES)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS)
    parser.add_argument("--aux-cap", type=int, default=200)
    parser.add_argument("--redesign", action="store_true", help="also run oracle II-D per breakpoint")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    started_all = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - started_all:7.0f}s] {message}", flush=True)

    combined = []
    for case in args.cases:
        problem = load_problem(case)
        z = z_from_delta(DEFAULT_DELTA, problem.n_generators)
        reference = reference_row(problem, z)
        log(
            f"{case}: Part I M1 {reference['m1']:,.0f}  M3 bound {reference['m3_bound']:,.0f}  "
            f"M3 actual {reference['m3_actual']:,.0f}  M2 {reference['m2']:,.0f}"
        )
        out_dir = args.out / case
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        level_zero = None
        for strategy in args.strategies:
            library = build_context_library(problem, strategy)
            moments = OracleMoments(library, problem.evaluator.state)
            for level in args.levels:
                if level == "II-0" and level_zero is not None:
                    row = dict(level_zero)  # II-0 does not depend on the completion
                else:
                    row = evaluate_level(
                        problem, library, moments, level, z,
                        aux_cap=args.aux_cap, redesign=args.redesign, out_dir=out_dir,
                        stem=f"{case}_step3_{strategy}_{level}", log=log,
                    )
                    if level == "II-0":
                        level_zero = row
                row = {"case_id": case, "strategy": strategy, **row}
                rows.append(row)
        base = level_zero or {}
        for row in rows:
            for key in ("m1", "m3_bound_static_maxima", "m3_bound_static_topup", "m3_actual_static",
                        "m3_bound_redesign_maxima", "m3_bound_redesign_topup"):
                ref_key = {"m1": "m1", "m3_actual_static": "m3_actual"}.get(key, "m3_bound")
                if key in row and ref_key in reference:
                    row[f"gain_{key}"] = 1.0 - row[key] / reference[ref_key]
            if PRIMARY in row:
                row["gate_pass"] = bool(row[f"gain_{PRIMARY}"] >= GATE)
            row["part1_m1"] = reference["m1"]
            row["part1_m2"] = reference["m2"]
            row["part1_m3_bound"] = reference["m3_bound"]
            row["part1_m3_actual"] = reference["m3_actual"]
        if base:
            assert abs(base["m1"] - reference["m1"]) <= 1e-6 * reference["m1"] + 1
            assert abs(base["m3_actual_static"] - reference["m3_actual"]) <= 1e-6 * reference["m3_actual"] + 1
        columns = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(out_dir / f"{case}_step3_oracle_ceiling.csv", columns, rows)
        write_json(out_dir / f"{case}_step3_meta.json", {"reference": reference, "gate": GATE,
                                                          "primary_metric": PRIMARY, "run": run_record()})
        combined.extend(rows)
    columns = list(dict.fromkeys(key for row in combined for key in row))
    name = "step3_oracle_ceiling.csv" if len(args.cases) == len(CASES) else f"step3_oracle_ceiling_{args.cases[0]}{'_etc' if len(args.cases) > 1 else ''}.csv"
    write_csv(args.out / name, columns, combined)


if __name__ == "__main__":
    main()
