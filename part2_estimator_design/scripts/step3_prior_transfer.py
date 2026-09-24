#!/usr/bin/env python3
"""Step 3 addendum: does a design optimised on a classical prior survive the true state?

On a determinant every Pauli outcome is either deterministic or perfectly
(anti)correlated with others of the same ``X`` part, so an oracle design on an HF
state can cancel variance exactly, and the Step 3 ceilings on HF rows may owe much
to that.  Here the design is optimised on the HF covariances (a classically
available prior) and then *evaluated* on the CISD state of the same geometry,
whose covariances it has never seen.  HF and CISD share the Pauli supports and
hence the context library, so the same coefficients apply unchanged.

Rows per geometry: II-0 on CISD (Part I), II-A optimised on CISD (oracle ceiling),
II-A optimised on HF and evaluated on CISD (prior transfer).  This is a planning-
level preview of Step 4's "HF prior, CISD truth" run, not a substitute for it.

Example::

    python scripts/step3_prior_transfer.py --pairs H4_square_eq_side1p0 H4_square_stretch_side2p0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from contexts import build_context_library  # noqa: E402
from design import DesignSet, assert_exact, build_fragment_problems  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import DEFAULT_DELTA, epsilon_from_radius, gradient_matrix, load_problem, z_from_delta  # noqa: E402
from planning import actual_precision, m3_common_radius  # noqa: E402
from sampler import OracleMoments  # noqa: E402


def costs(problem, design, z) -> dict:
    arms = list(range(problem.n_generators))
    from allocation import allocate

    sigmas = design.sigmas(arms)
    epsilon = epsilon_from_radius(problem.top_gap() / 2.0, z)
    return {
        "m1": float(allocate(sigmas, epsilon).sum()),
        "m3_bound_static_maxima": m3_common_radius(design, problem.abs_gradients, z, static_solution=True)["total"],
        "m3_actual_static": actual_precision(problem.abs_gradients, z, sigmas=sigmas)["total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=["H4_square_eq_side1p0", "H4_square_stretch_side2p0"])
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--level", default="II-A")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    from states import hartree_fock_state  # Part I

    rows = []
    for stem in args.pairs:
        cisd = load_problem(f"{stem}_CISD")
        z = z_from_delta(DEFAULT_DELTA, cisd.n_generators)
        library = build_context_library(cisd, args.strategy)
        A = gradient_matrix(cisd, library.n_library)
        # The prior is the determinant in the CISD problem's own orbital basis.  A
        # separately built HF problem may carry different orbital phases (it does at
        # side 2.0), which flips signs in A; the determinant itself does not depend on
        # them, so taking it here keeps prior and truth in one convention.
        hf_state = hartree_fock_state(cisd.n_qubits, cisd.metadata["n_electrons"])
        truth = OracleMoments(library, cisd.evaluator.state)
        prior = OracleMoments(library, hf_state)

        base = DesignSet(build_fragment_problems(cisd, library, truth, "II-0"), library.n_contexts)
        oracle = DesignSet(build_fragment_problems(cisd, library, truth, args.level), library.n_contexts)
        oracle.solve(range(cisd.n_generators), 1.0)
        assert_exact(oracle, A)
        on_prior = DesignSet(build_fragment_problems(cisd, library, prior, args.level), library.n_contexts)
        on_prior.solve(range(cisd.n_generators), 1.0)
        transferred = DesignSet(build_fragment_problems(cisd, library, truth, args.level), library.n_contexts)
        for target, source in zip(transferred.problems, on_prior.problems):
            assert np.array_equal(target.coord_ctx, source.coord_ctx)
            assert np.array_equal(target.coord_pauli, source.coord_pauli)
            target.x = source.x.copy()
        assert_exact(transferred, A)

        reference = costs(cisd, base, z)
        for label, design in (("II-0 (Part I)", base), (f"{args.level} oracle on CISD", oracle),
                              (f"{args.level} from HF prior, on CISD", transferred)):
            row = {"geometry": stem, "design": label, **costs(cisd, design, z)}
            for key in ("m1", "m3_bound_static_maxima", "m3_actual_static"):
                row[f"gain_{key}"] = 1.0 - row[key] / reference[key]
            rows.append(row)
            print(f"{stem:28s} {label:34s} M1 {row['m1']:12,.0f} ({row['gain_m1']:+.1%})  "
                  f"M3 bound {row['m3_bound_static_maxima']:10,.0f} ({row['gain_m3_bound_static_maxima']:+.1%})  "
                  f"M3 actual {row['m3_actual_static']:10,.0f} ({row['gain_m3_actual_static']:+.1%})", flush=True)
    write_csv(args.out / "step3_prior_transfer.csv", list(rows[0]), rows)
    write_json(args.out / "step3_prior_transfer_meta.json", {"strategy": args.strategy, "level": args.level,
                                                              "run": run_record()})


if __name__ == "__main__":
    main()
