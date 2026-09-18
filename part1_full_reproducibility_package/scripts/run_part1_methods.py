#!/usr/bin/env python3
"""Run the Part I measurement methods M1: FC-UG, M2: BAI-FC-IG and M3: BAI-FC-UG.

The pipeline is: PySCF RHF -> Jordan-Wigner Hamiltonian -> validation gate ->
fixed state (HF or CISD) -> UCCSD generator pool -> gradient commutators ->
fully commuting groupings -> oracle fragment variances -> per-method shot
proxies.  Every method writes its own files, named with its method code, and
saves the FC groups and fragment standard deviations behind each number.

Examples::

    python scripts/run_part1_methods.py --case H4_square_eq_side1p0_HF --out runs
    python scripts/run_part1_methods.py --all-small --out runs
    python scripts/run_part1_methods.py --case LiH_R3p0_HF --methods m1 m3 --out runs
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline_noshare
import m1_fcug
import m2_baifcig
import m3_baifcug
from cases import CASES, SMALL_CASES, get_case
from problem_cache import load_or_build
from io_utils import write_csv, write_json
from shot_models import DEFAULT_DELTA

METHODS = ("noshare", "m1", "m2", "m3")


def run_case(
    case_id: str,
    outdir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    delta: float = DEFAULT_DELTA,
    validate: bool = True,
    cache_dir: Path | None = None,
    save_commutators: bool = False,
) -> dict:
    """Run the selected methods for one case and write all outputs."""
    spec = get_case(case_id)
    outdir = Path(outdir) / case_id
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    def log(message: str) -> None:
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        print(
            f"[{case_id}] {time.perf_counter() - started:7.1f}s  peak {peak_gib:5.2f} GiB  {message}",
            flush=True,
        )

    problem = load_or_build(spec, cache_dir, validate=validate)
    log(
        f"gradients ready: {len(problem.universal_support):,} universal Pauli terms, "
        f"{problem.n_nonzero_gradients} non-zero gradients"
    )
    groups = problem.parent_fc_groups()
    log(f"parent FC grouping: {len(groups):,} contexts")
    order = problem.ranking()

    summary = {
        "case_id": case_id,
        "state": problem.state_name,
        "n_qubits": problem.n_qubits,
        "state_energy_hartree": problem.state_energy,
        "universal_commutator_pauli_terms": len(problem.universal_support),
        "parent_fc_contexts": len(groups),
        "nonzero_gradients": problem.n_nonzero_gradients,
        "nonzero_gradients_by_threshold": problem.nonzero_gradient_scan(),
        "top_generator": problem.labels[order[0]],
        "top_abs_gradient": float(problem.abs_gradients[order[0]]),
        "second_abs_gradient": float(problem.abs_gradients[order[1]]),
        "top_gap_abs_gradient": problem.top_gap(),
        "delta_family_wise": delta,
        **problem.metadata,
    }

    if save_commutators:
        _write_commutators(outdir / f"{case_id}_commutators.jsonl", problem)

    write_csv(
        outdir / f"{case_id}_gradients.csv",
        ["generator_index", "label", "kind", "gradient", "abs_gradient", "pauli_terms"],
        (
            {
                "generator_index": i,
                "label": problem.labels[i],
                "kind": problem.kinds[i],
                "gradient": float(problem.gradients[i]),
                "abs_gradient": float(problem.abs_gradients[i]),
                "pauli_terms": len(problem.commutator_terms[i]),
            }
            for i in order
        ),
    )

    if "noshare" in methods:
        log("running the no-sharing baseline")
        result = baseline_noshare.run(problem, delta=delta)
        result.save(outdir, problem.labels, problem.abs_gradients)
        summary["NOSHARE"] = result.total_shots

    if "m1" in methods:
        log("running M1")
        result = m1_fcug.run(problem, target="identification", delta=delta)
        result.save(outdir, problem.labels)
        summary["M1_FCUG"] = result.total_shots
        summary["M1_target"] = result.target
        summary["M1_confidence_radius"] = result.radius
        # The harder legacy task, reported alongside rather than as the headline.
        summary["M1_FCUG_full_vector"] = m1_fcug.run(
            problem, target="full_vector", delta=delta
        ).total_shots
    if "m2" in methods:
        log("running M2")
        result = m2_baifcig.run(problem, delta=delta)
        result.save(outdir, problem.labels, problem.abs_gradients)
        summary["M2_BAIFCIG"] = result.total_shots
        summary["M2_selected"] = [problem.labels[i] for i in result.remaining]
    if "m3" in methods:
        log("running M3")
        result = m3_baifcug.run(problem, delta=delta)
        result.save(outdir, problem.labels)
        summary["M3_BAIFCUG"] = result.total_shots
        summary["M3_selected"] = [problem.labels[i] for i in result.remaining]

    if "M1_FCUG" in summary and "M2_BAIFCIG" in summary:
        summary["M1_over_M2"] = summary["M1_FCUG"] / summary["M2_BAIFCIG"]
    if "M2_BAIFCIG" in summary and "M3_BAIFCUG" in summary:
        summary["M2_over_M3"] = summary["M2_BAIFCIG"] / summary["M3_BAIFCUG"]
    # Attribution against the fourth cell of the two-by-two: sharing alone,
    # elimination alone, and the two together.
    for code, name in (("M1_FCUG", "sharing_only"), ("M2_BAIFCIG", "elimination_only"),
                       ("M3_BAIFCUG", "both")):
        if "NOSHARE" in summary and code in summary:
            summary[f"NOSHARE_over_{name}"] = summary["NOSHARE"] / summary[code]
    summary["runtime_seconds"] = time.perf_counter() - started

    write_json(outdir / f"{case_id}_summary.json", _merged(outdir, case_id, summary))
    return summary


def _write_commutators(path: Path, problem) -> None:
    """One JSON object per generator: its full Pauli expansion of [H, G_i].

    These are the raw objects every later number derives from, so a reader can
    rebuild the groupings and the variances without rerunning the chemistry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, terms in enumerate(problem.commutator_terms):
            handle.write(
                json.dumps(
                    {
                        "generator_index": index,
                        "label": problem.labels[index],
                        "kind": problem.kinds[index],
                        "gradient": float(problem.gradients[index]),
                        "terms": {p: c for p, c in sorted(terms.items())},
                    }
                )
                + "\n"
            )


def _merged(outdir: Path, case_id: str, summary: dict) -> dict:
    """Fold this run into any summary already written for the same case.

    Methods can be run as separate invocations -- useful for the larger cases,
    where one process per method keeps each run short -- so a later run must not
    discard the columns an earlier one produced.
    """
    path = outdir / f"{case_id}_summary.json"
    if not path.exists():
        return summary
    previous = json.loads(path.read_text())
    previous.update(summary)
    for numerator, denominator, name in (
        ("M1_FCUG", "M2_BAIFCIG", "M1_over_M2"),
        ("M2_BAIFCIG", "M3_BAIFCUG", "M2_over_M3"),
    ):
        if numerator in previous and denominator in previous:
            previous[name] = previous[numerator] / previous[denominator]
    return previous


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", choices=sorted(CASES))
    group.add_argument("--all-small", action="store_true", help="every case except H2O")
    group.add_argument("--all", action="store_true", help="every case, H2O included")
    parser.add_argument("--out", default="runs", help="output directory")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the RHF/FCI energy gate (not recommended)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log per-stage progress"
    )
    parser.add_argument(
        "--save-commutators", action="store_true",
        help="write the raw Pauli expansion of every commutator as JSONL")
    parser.add_argument(
        "--cache",
        default=None,
        help="directory for cached gradient problems, so methods can be run "
        "as separate processes without rebuilding the chemistry each time",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.all:
        selected = sorted(CASES)
    elif args.all_small:
        selected = SMALL_CASES
    else:
        selected = [args.case]

    outdir = Path(args.out)
    summaries = []
    for case_id in selected:
        print(f"[{case_id}] running {', '.join(args.methods)} ...", flush=True)
        summary = run_case(
            case_id,
            outdir,
            methods=tuple(args.methods),
            delta=args.delta,
            validate=not args.skip_validation,
            cache_dir=Path(args.cache) if args.cache else None,
            save_commutators=args.save_commutators,
        )
        summaries.append(summary)
        print(
            f"[{case_id}] "
            + "  ".join(
                f"{key}={summary[key]:,}"
                for key in ("NOSHARE", "M1_FCUG", "M2_BAIFCIG", "M3_BAIFCUG")
                if key in summary
            ),
            flush=True,
        )

    fields = sorted({key for summary in summaries for key in summary})
    write_csv(
        outdir / "part1_methods_summary.csv",
        fields,
        [{key: summary.get(key) for key in fields} for summary in summaries],
    )
    print(f"wrote {outdir / 'part1_methods_summary.csv'}")


if __name__ == "__main__":
    main()
