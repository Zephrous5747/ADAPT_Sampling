"""On-disk cache for built gradient problems.

Building the gradient problem for a 14-qubit case costs about a minute: the
Hamiltonian, the validation gate, 140 commutators and the parent fully commuting
grouping.  Running M1, M2 and M3 as three separate processes would pay that
three times, so the built problem is cached and reloaded instead.

The cache key includes the case identifier and the pipeline conventions that
would change the result, so a cache entry is never reused across a change in
those conventions.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

from chemistry import CaseSpec
from gradients import GradientProblem, build_gradient_problem

LOGGER = logging.getLogger(__name__)

CACHE_VERSION = "1"


def cache_key(spec: CaseSpec) -> str:
    """Stable identifier for one case under the current conventions."""
    material = "|".join(
        [
            CACHE_VERSION,
            spec.case_id,
            spec.state.upper(),
            spec.basis,
            str(spec.charge),
            str(spec.spin),
            spec.geometry,
        ]
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:12]
    return f"{spec.case_id}_{digest}.pickle"


def load_or_build(
    spec: CaseSpec, cache_dir: Path | None, *, validate: bool = True
) -> GradientProblem:
    """Return the gradient problem for ``spec``, using ``cache_dir`` if given."""
    if cache_dir is None:
        return build_gradient_problem(spec, validate=validate)

    cache_dir = Path(cache_dir)
    path = cache_dir / cache_key(spec)
    if path.exists():
        LOGGER.info("loading cached gradient problem from %s", path)
        with path.open("rb") as handle:
            return pickle.load(handle)

    problem = build_gradient_problem(spec, validate=validate)
    problem.parent_fc_groups()  # cache the grouping in the pickle as well
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    LOGGER.info("wrote gradient problem cache to %s", path)
    return problem
