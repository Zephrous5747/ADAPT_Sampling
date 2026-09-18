"""The Part I benchmark case registry.

Geometries match ``configs/part1_cases.json``.  Case identifiers are the keys
used on the command line and in every output file name.
"""
from __future__ import annotations

from chemistry import CaseSpec, water_geometry

_H4_SQUARE = lambda side: [  # noqa: E731 - a table of geometries reads better inline
    ("H", (0.0, 0.0, 0.0)),
    ("H", (side, 0.0, 0.0)),
    ("H", (0.0, side, 0.0)),
    ("H", (side, side, 0.0)),
]

CASES: dict[str, CaseSpec] = {
    spec.case_id: spec
    for spec in (
        CaseSpec("LiH_R3p0_HF", [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 3.0))], "HF"),
        CaseSpec("H4_square_eq_side1p0_HF", _H4_SQUARE(1.0), "HF"),
        CaseSpec("H4_square_eq_side1p0_CISD", _H4_SQUARE(1.0), "CISD"),
        CaseSpec("H4_square_stretch_side2p0_HF", _H4_SQUARE(2.0), "HF"),
        CaseSpec("H4_square_stretch_side2p0_CISD", _H4_SQUARE(2.0), "CISD"),
        CaseSpec("H2O_eq_HF", water_geometry(0.9572), "HF"),
        CaseSpec("H2O_eq_CISD", water_geometry(0.9572), "CISD"),
        CaseSpec("H2O_stretch_HF", water_geometry(1.75), "HF"),
        CaseSpec("H2O_stretch_CISD", water_geometry(1.75), "CISD"),
    )
}

SMALL_CASES = [name for name in CASES if not name.startswith("H2O")]


def get_case(case_id: str) -> CaseSpec:
    try:
        return CASES[case_id]
    except KeyError:
        raise SystemExit(
            f"unknown case {case_id!r}; choose one of: " + ", ".join(sorted(CASES))
        ) from None
