#!/usr/bin/env python3
"""Paper A: how the structure and the design cost grow with the molecule (referee point on scale), no shot simulation.

For each case this builds the objects every method of the paper needs and times them:

* the pool (generators ``K``), the Hamiltonian's Pauli terms, the universal support (the distinct Pauli products of
  all commutators ``[H, G_i]``), the parent fully commuting contexts of Part I and the context library of the
  main study (strategy ``mass``), with the two-qubit gates of its circuits;
* the II-0 design (every product at its home context) and the II-A coordinates (every context that measures a
  product is a coordinate: the design problem's size), with the time to build them;
* one *refit* of the II-A designs: the per-generator convex problem solved with exact covariances on the
  reference state and ``--shots`` shots in every context, for ``--arms`` randomly chosen generators; the time
  per generator and the number of free variables are measured, the time of a full refit of all ``K``
  generators is the extrapolation (the generators are solved independently).

Written to ``runs/paper_a/scaling.csv`` (one row per case; ``nan`` where a stage was not run).  A case that
is not built yet is built with Part I's pipeline (and cached); its build time is recorded.

    python scripts/paper_a_scaling.py --cases H4_square_eq_side1p0_CISD LiH_R3p0_HF BeH2_HF
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402
from allocation import allocate  # noqa: E402
from contexts import build_context_library  # noqa: E402
from design import DesignSet, FragmentProblem, build_fragment_problems  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from states import hartree_fock_state  # noqa: E402  (Part I)

DEFAULT_CASES = ["H4_square_eq_side1p0_CISD", "LiH_R3p0_HF", "LiH_R1p6_HF", "H6_chain_HF", "BeH2_HF", "H2O_eq_HF"]


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def measure(case: str, strategy: str, shots: float, arms: int, seed: int, structure_only: bool = False) -> dict:
    row: dict = {"case": case}
    problem, row["build_problem_s"] = timed(lambda: part1_bridge.load_problem(case))
    row.update(qubits=problem.n_qubits, generators=problem.n_generators,
               hamiltonian_terms=int(problem.metadata.get("hamiltonian_pauli_terms", np.nan)),
               universal_support=len(problem.universal_support),
               commutator_terms_mean=float(np.mean([len(t) for t in problem.commutator_terms])))
    parent, row["parent_groups_s"] = timed(problem.parent_fc_groups)
    row["parent_contexts"] = len(parent)
    library, row["library_s"] = timed(lambda: build_context_library(problem, strategy))
    cz = library.two_qubit_counts()
    uses = library.contexts_of()
    row.update(library_contexts=library.n_contexts, members_mean=float(np.mean([len(c.members) for c in library.contexts])),
               two_qubit_mean=float(cz.mean()), two_qubit_max=int(cz.max()),
               contexts_per_product_mean=float(np.mean([len(u) for u in uses])))
    state = hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"])
    oracle = OracleMoments(library, state)
    base, row["design_ii0_s"] = timed(lambda: build_fragment_problems(problem, library, oracle, "II-0"))
    split, row["design_iia_coordinates_s"] = timed(lambda: build_fragment_problems(problem, library, oracle, "II-A"))
    row["coordinates_total"] = int(sum(p.coord_ctx.size for p in split))
    row["coordinates_per_generator_mean"] = row["coordinates_total"] / len(split)
    row["coordinates_per_generator_max"] = int(max(p.coord_ctx.size for p in split))
    if structure_only:  # the shot simulator stores a 2^n outcome distribution per context: not available at this size
        row.update(simulation_skipped=True, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        return row
    design = DesignSet(base, library.n_contexts)
    sigmas, row["sigmas_s"] = timed(lambda: design.sigmas(list(range(problem.n_generators))))
    _, row["allocation_s"] = timed(lambda: allocate(sigmas, 0.5 * float(problem.abs_gradients.max())))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(split), size=min(arms, len(split)), replace=False)
    held = np.full(library.n_contexts, float(shots))
    times, free = [], []
    for i in chosen:
        p = split[int(i)]
        fragment = FragmentProblem(p.generator, p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target)),
                                   library.home, lambda a, pauli, o=oracle: o.covariance(a, pauli))
        fragment.x = np.zeros(fragment.n_coordinates)
        fragment.x[fragment.reference] = fragment.pauli_target
        _, seconds = timed(lambda: fragment.optimise(held))
        times.append(seconds)
        free.append(fragment.n_coordinates - fragment.pauli_ids.size)
    row.update(refit_arms_sampled=len(chosen), refit_seconds_per_generator_median=float(np.median(times)),
               refit_seconds_per_generator_max=float(np.max(times)), free_variables_median=float(np.median(free)),
               refit_seconds_extrapolated=float(np.mean(times) * problem.n_generators),
               peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--shots", type=float, default=1e4, help="shots held in every context at the refit")
    parser.add_argument("--arms", type=int, default=20, help="generators whose refit is timed")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/paper_a"))
    parser.add_argument("--structure-only", action="store_true",
                        help="stop after the context library and the II-0 / II-A coordinates (no sigmas, no refit): for cases "
                             "above about 20 qubits, where the simulator's 2^n outcome distributions do not fit")
    args = parser.parse_args()
    out = args.out / "scaling.csv"
    existing = {}
    if out.exists():
        import csv
        existing = {r["case"]: r for r in csv.DictReader(out.open())}
    rows = []
    for case in args.cases:
        try:
            row = measure(case, args.strategy, args.shots, args.arms, args.seed, args.structure_only)
        except MemoryError as error:  # pragma: no cover
            print(f"{case}: out of memory ({error})", flush=True)
            continue
        rows.append(row)
        print(f"{case:28s} {row['qubits']} qubits  K={row['generators']:4d}  support {row['universal_support']:8,d}  "
              f"contexts {row['library_contexts']:6,d} (parent {row['parent_contexts']:5,d})  CZ {row['two_qubit_mean']:5.1f}  "
              f"coords/gen {row['coordinates_per_generator_mean']:9,.0f}  refit/gen {row.get('refit_seconds_per_generator_median', float('nan')):7.2f}s "
              f"(all {row.get('refit_seconds_extrapolated', float('nan')) / 60:6.1f} min)  II-A coords {row['design_iia_coordinates_s']:6.1f}s  "
              f"build {row['build_problem_s']:6.0f}s", flush=True)
        merged = {**existing, **{r["case"]: r for r in rows}}
        ordered = list(merged.values())
        columns = list(dict.fromkeys(k for r in ordered for k in r))
        write_csv(out, columns, ordered)
    write_json(args.out / "scaling_meta.json", {"strategy": args.strategy, "shots": args.shots, "arms": args.arms, "run": run_record()})


if __name__ == "__main__":
    main()
