#!/usr/bin/env python3
"""Deprecated entry point kept for the command lines in README.md and MANIFEST.txt.

The original version of this script carried its own copy of the chemistry
pipeline, and that copy was wrong in two ways: the two-electron integrals were
inserted into the OpenFermion tensor in chemist rather than physicist ordering
with the wrong spin pattern and no one-half prefactor, and the computational
basis was indexed little-endian while OpenFermion is big-endian.  Together they
produced a Hartree-Fock energy of -5.21 Ha for H4 where RHF gives -1.76 Ha, so
every gradient and shot count derived from it was meaningless.

The pipeline now lives in ``src/`` behind a validation gate that refuses to
continue unless the qubit Hamiltonian reproduces the PySCF RHF and FCI energies.
This wrapper forwards to :mod:`run_part1_methods`, which also computes M1, M2
and M3 -- something this script never did.

Prefer calling the new driver directly::

    python scripts/run_part1_methods.py --case LiH_R3p0_HF --out runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_part1_methods
from cases import CASES

DEPRECATION_NOTICE = (
    "run_part1_from_scratch.py is deprecated; forwarding to run_part1_methods.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--all-small", action="store_true")
    parser.add_argument("--out", default="from_scratch_runs")
    parser.add_argument(
        "--skip-h2o-m3",
        action="store_true",
        help="retained for compatibility; H2O is excluded by --all-small anyway",
    )
    args = parser.parse_args()
    if not args.case and not args.all_small:
        raise SystemExit("Specify --case CASE_ID or --all-small")

    print(DEPRECATION_NOTICE, file=sys.stderr)
    forwarded = ["--out", args.out]
    forwarded += ["--all-small"] if args.all_small else ["--case", args.case]
    run_part1_methods.main(forwarded)


if __name__ == "__main__":
    main()
