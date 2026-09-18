"""Small helpers for writing auditable method outputs.

The Part I report requires that the fully commuting groups and the fragment
variances behind every table number are saved alongside it, so each method
module writes a JSON summary plus one CSV per underlying object.  File names
carry the fixed method codes ``M1_FCUG``, ``M2_BAIFCIG`` and ``M3_BAIFCUG``.
"""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Iterable, Mapping, Sequence

GROUPING_CONVENTIONS = {
    "fc_grouping": "deterministic greedy first-fit fully commuting grouping",
    "pauli_ordering": "sort by decreasing Pauli weight, then lexicographic",
    "tie_breaking": "first group that accepts the Pauli string",
    "random_seed": None,
    "note": "the grouping is deterministic, so no seed is used",
}


def environment_record() -> dict:
    """Package versions and interpreter, recorded with every run."""
    versions = {}
    for name in ("numpy", "scipy", "pandas", "pyscf", "openfermion"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # pragma: no cover - optional at report time
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def write_json(path: Path, payload: Mapping) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=_default))
    return path


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _default(value):
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)!r}")
