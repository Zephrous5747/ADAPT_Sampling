"""Pivot-based grouping of the gradient observables (Anastasiou et al., arXiv:2306.03227).

Part I's M1 measures the gradients in fully commuting contexts found by first-fit insertion
over the union of all Pauli products of all commutators.  Anastasiou, Mayhall, Barnes and
Economou partition the same observables by a different rule, and the published version of
the shot-reuse paper (Ikhtiarudin et al., Phys. Scr. 101, 255103, 2026) reports that this
*pivot-based grouping* alone gives the dominant reduction and is the practical baseline.

The rule.  Write ``[H, G_i] = sum_P h_P sum_S g_{i,S} [P, S]`` with ``P`` a Pauli term of
``H`` (the *pivot*) and ``S`` a Pauli string of the generator.  For a pivot ``P`` and Pauli
strings ``S_j`` of the pool that commute with each other,

    [[P, S_j], [P, S_k]] = 4 [S_k, S_j] = 0,

so the commutators ``[P, S_j]`` of one pivot with a commuting set of pool strings commute and
can be measured together.  The pool strings are partitioned once into commuting *classes*;
the contexts are the pairs ``(P, class)``, each measuring ``{P S : S in class, {P, S} = 0}``.
All members of a context carry the coefficient ``2 |h_P| |g_S|``, which is what makes their
allocation nearly optimal without variances in the original.  For the qubit pool the classes
are the ``2N`` anchored sets (every string with its single ``Y`` at qubit ``a``; every string
with its single ``X`` at ``a``), which also gives diagonalising circuits of at most ``N - 3``
CNOT gates in their construction.  Qubit-excitation and CEO pools are linear combinations of
qubit-pool strings, so the same classes apply; for the UCCSD pool (Jordan-Wigner ``Z``
strings) the classes are found by first-fit insertion.

What is modelled here.  Every Pauli of ``[H, G_i]`` is read from the context of the pivot
it came from, with the coefficient it had there (:func:`pivot_fragment_problems`), so a
Pauli that arises from several pivots is measured, and its variance paid, in each of them
as in the original.  The allocation is Part I's optimal one with exact (static) or estimated
(sequential) variances rather than their bound ``Var <= 1``, so this baseline is at least
as strong as the original allocation.  Circuits are synthesised generically, so their CNOT
counts upper-bound the construction of the original.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import part1_bridge  # noqa: F401  (puts Part I's src on the path)
from contexts import build_context_library
from design import FragmentProblem
from pauli_fc import greedy_fc_groups, openfermion_qubitop_to_dict  # Part I
from symplectic import anticommutes, labels_from_xz, masks_from_xz, popcount, xz_from_labels

CLASSES = ("auto", "anchor", "firstfit")
TOLERANCE = 1e-8


def generator_pauli_terms(problem) -> list[dict[str, complex]]:
    """Jordan-Wigner Pauli expansion of every generator of the problem, in the problem's order.

    The generators are rebuilt from the pool named in the problem (``metadata["pool"]``, the
    UCCSD pool of Part I otherwise) and matched to ``problem.labels``, because pool problems
    drop generators that commute with ``H``.
    """
    from pools import POOL_BUILDERS

    pool = problem.metadata.get("pool", "uccsd")
    built = POOL_BUILDERS[pool](problem.n_qubits, problem.metadata["n_electrons"])
    by_label = {g.label: g for g in built}
    missing = [label for label in problem.labels if label not in by_label]
    if missing:
        raise ValueError(f"{problem.case_id}: generators {missing[:3]} are not in the {pool} pool")
    return [openfermion_qubitop_to_dict(by_label[label].qubit_operator, problem.n_qubits)
            for label in problem.labels]


def anchor_classes(strings: list[str]) -> dict[str, int] | None:
    """The ``2N`` anchored classes of the qubit pool, or ``None`` if a string does not fit.

    A string with one ``Y`` (and ``X`` elsewhere) goes to class ``("Y", position of Y)``; a
    string with three ``Y`` and one ``X`` to ``("X", position of X)``.  Strings of one class
    commute qubit by qubit.  Only ``I``, ``X`` and ``Y`` letters, and an odd ``Y`` count of
    one or three, are accepted.
    """
    keys: dict[str, tuple[str, int]] = {}
    for s in strings:
        if set(s) - {"I", "X", "Y"}:
            return None
        y = s.count("Y")
        if y == 1:
            keys[s] = ("Y", s.index("Y"))
        elif y == 3 and s.count("X") == 1:
            keys[s] = ("X", s.index("X"))
        else:
            return None
    ids = {key: k for k, key in enumerate(sorted(set(keys.values())))}
    return {s: ids[keys[s]] for s in strings}


def firstfit_classes(strings: list[str]) -> dict[str, int]:
    """Mutually commuting classes of the pool strings by first-fit insertion (Part I's routine)."""
    return {s: k for k, group in enumerate(greedy_fc_groups(strings)) for s in group}


@dataclass
class PivotStructure:
    """The contexts of the pivot partition and what each gradient reads from them."""

    groups: list[list[str]]  # Pauli products measured by each context
    keys: list[tuple[str, int]]  # (pivot, class) of each context
    contributions: list[dict[tuple[int, str], float]]  # per arm: (context, Pauli) -> coefficient
    classes: str  # "anchor" or "firstfit"
    n_classes: int
    n_pivots: int
    n_pool_strings: int
    extra_labels: list[str] = field(default_factory=list)  # measured products no gradient needs

    @property
    def n_contexts(self) -> int:
        return len(self.groups)


def _hermitian_phase_exponent(x1, z1, x2, z2):
    """Exponent ``e`` with ``P1 P2 = i**e * (Hermitian string of x1 ^ x2, z1 ^ z2)``."""
    x1 = np.asarray(x1, dtype=np.int64)
    z1 = np.asarray(z1, dtype=np.int64)
    x2 = np.asarray(x2, dtype=np.int64)
    z2 = np.asarray(z2, dtype=np.int64)
    k1 = (x1 & z1).sum(axis=-1)
    k2 = (x2 & z2).sum(axis=-1)
    x3, z3 = x1 ^ x2, z1 ^ z2
    k3 = (x3 & z3).sum(axis=-1)
    return (k1 + k2 + 2 * (z1 * x2).sum(axis=-1) - k3) % 4


def build_pivot_structure(problem, terms: dict[str, float], classes: str = "auto") -> PivotStructure:
    """Pivot contexts of the problem's pool, and every gradient's reading of them.

    ``terms`` are the Pauli terms of ``H`` (without the identity).  The decomposition is
    checked against the problem's own commutator expansions, so a cached problem built on
    other orbitals, or a pool that does not match, fails here rather than silently.
    """
    if classes not in CLASSES:
        raise ValueError(f"classes must be one of {CLASSES}")
    n = problem.n_qubits
    arm_terms = generator_pauli_terms(problem)
    strings = sorted({s for t in arm_terms for s in t})
    kind = classes
    class_of = None
    if classes in ("auto", "anchor"):
        class_of = anchor_classes(strings)
        kind = "anchor"
        if class_of is None and classes == "anchor":
            raise ValueError("the pool strings do not fit the anchored classes of the qubit pool")
    if class_of is None:
        class_of, kind = firstfit_classes(strings), "firstfit"
    n_classes = len(set(class_of.values()))

    pivots = sorted(terms, key=lambda p: (-abs(terms[p]), p))
    h = np.array([terms[p] for p in pivots])
    px, pz = xz_from_labels(pivots)
    pxm, pzm = masks_from_xz(px, pz)

    sx, sz = xz_from_labels(strings)
    sxm, szm = masks_from_xz(sx, sz)
    s_index = {s: k for k, s in enumerate(strings)}
    s_class = np.array([class_of[s] for s in strings])
    # anticommutation of every pivot with every pool string: (pivots, strings)
    anti = anticommutes(pxm[:, None], pzm[:, None], sxm[None, :], szm[None, :])

    # contexts: one per (pivot, class) with at least one anticommuting string
    context_index: dict[tuple[int, int], int] = {}
    keys: list[tuple[str, int]] = []
    groups: list[list[str]] = []
    for p in range(len(pivots)):
        for k in np.unique(s_class[anti[p]]):
            members = np.flatnonzero(anti[p] & (s_class == k))
            qx = px[p][None, :] ^ sx[members]
            qz = pz[p][None, :] ^ sz[members]
            context_index[(p, int(k))] = len(groups)
            keys.append((pivots[p], int(k)))
            groups.append(labels_from_xz(qx, qz))

    contributions: list[dict[tuple[int, str], float]] = []
    for i, gen in enumerate(arm_terms):
        labels = list(gen)
        g = np.array([gen[s] for s in labels])
        cols = np.array([s_index[s] for s in labels])
        rows, cs = np.nonzero(anti[:, cols])
        qx = px[rows] ^ sx[cols[cs]]
        qz = pz[rows] ^ sz[cols[cs]]
        exponent = _hermitian_phase_exponent(px[rows], pz[rows], sx[cols[cs]], sz[cols[cs]])
        value = 2.0 * h[rows] * g[cs] * (1j) ** exponent
        if value.size and np.abs(value.imag).max() > 1e-9:
            raise ValueError(f"{problem.case_id}: generator {i} gives a non-Hermitian commutator")
        q_labels = labels_from_xz(qx, qz)
        accumulated: dict[tuple[int, str], float] = {}
        for r, c, q, v in zip(rows, cs, q_labels, value.real):
            key = (context_index[(int(r), int(s_class[cols[c]]))], q)
            accumulated[key] = accumulated.get(key, 0.0) + float(v)
        contributions.append({k: v for k, v in accumulated.items() if abs(v) > 1e-14})

    worst = 0.0
    for i, (parts, target) in enumerate(zip(contributions, problem.commutator_terms)):
        total: dict[str, float] = {}
        for (_, q), v in parts.items():
            total[q] = total.get(q, 0.0) + v
        for q in set(total) | set(target):
            worst = max(worst, abs(total.get(q, 0.0) - target.get(q, 0.0)))
    if worst > TOLERANCE:
        raise ValueError(f"{problem.case_id}: the pivot decomposition disagrees with the stored commutators by "
                         f"{worst:.2e}: the cached problem and the rebuilt H or pool do not match")

    needed = set(problem.universal_support)
    extra = sorted({q for group in groups for q in group} - needed)
    return PivotStructure(groups, keys, contributions, kind, n_classes, len(pivots), len(strings), extra)


def pivot_library(problem, terms: dict[str, float], classes: str = "auto", strategy: str = "canonical"):
    """The context library of the pivot partition and its structure."""
    structure = build_pivot_structure(problem, terms, classes)
    library = build_context_library(problem, strategy, groups=structure.groups,
                                    auxiliary_labels=structure.extra_labels)
    return library, structure


def pivot_fragment_problems(problem, structure: PivotStructure, library, moments) -> list[FragmentProblem]:
    """Every gradient read from the pivot contexts it came from, coefficient by coefficient."""
    index = {label: i for i, label in enumerate(library.labels)}
    problems = []
    for i, (parts, terms) in enumerate(zip(structure.contributions, problem.commutator_terms)):
        coord_ctx = np.array([c for c, _ in parts], dtype=np.int64)
        coord_pauli = np.array([index[q] for _, q in parts], dtype=np.int64)
        values = np.array(list(parts.values()))
        targets = {index[label]: v for label, v in terms.items()}
        p = FragmentProblem(i, coord_ctx, coord_pauli, targets, library.home, moments.covariance)
        codes = p.coord_ctx * library.n_library + p.coord_pauli  # sorted by construction
        order = np.searchsorted(codes, coord_ctx * library.n_library + coord_pauli)
        x = np.zeros(p.n_coordinates)
        x[order] = values
        p.x = x
        if p.constraint_residual() > TOLERANCE:  # pragma: no cover  (checked at build time)
            raise AssertionError(f"arm {i}: pivot coefficients do not reproduce the gradient")
        problems.append(p)
    return problems


def merged_assignment(structure: PivotStructure) -> dict[str, int]:
    """One pivot context for every Pauli product, chosen by greedy set cover.

    In the original every pivot that produces a product measures it, so a product that arises
    from several pivots is estimated several times and its variance is paid each time.  Reading
    each product from a single context removes that duplication, which the optimal splitting of
    II-A could do by itself, and makes the pivot contexts a library in the sense of Part I.  The
    contexts are picked in decreasing number of still unassigned products they can supply, the
    classical greedy cover, so that a shot informs as many products as possible; ties go to the
    lower context index.  This is stronger than the published scheme, not a reproduction of it.
    """
    import heapq

    candidates: dict[str, set[int]] = {}
    for parts in structure.contributions:
        for (ctx, q), _ in parts.items():
            candidates.setdefault(q, set()).add(ctx)
    supply: dict[int, set[str]] = {}
    for q, contexts in candidates.items():
        for ctx in contexts:
            supply.setdefault(ctx, set()).add(q)
    heap = [(-len(qs), ctx) for ctx, qs in supply.items()]
    heapq.heapify(heap)
    unassigned = set(candidates)
    assigned: dict[str, int] = {}
    while unassigned:
        negative, ctx = heapq.heappop(heap)
        fresh = supply[ctx] & unassigned
        if len(fresh) != -negative:  # stale entry: re-queue with its current supply
            if fresh:
                heapq.heappush(heap, (-len(fresh), ctx))
            continue
        for q in fresh:
            assigned[q] = ctx
        unassigned -= fresh
    return assigned


def apply_merged_home(library, structure: PivotStructure) -> None:
    """Make the merged assignment the library's home context of every product (II-A starts from it)."""
    index = {label: i for i, label in enumerate(library.labels)}
    for q, ctx in merged_assignment(structure).items():
        library.home[index[q]] = ctx


def merged_fragment_problems(problem, structure: PivotStructure, library, moments) -> list[FragmentProblem]:
    """II-0 coefficients with every product read from its one assigned pivot context."""
    assigned = merged_assignment(structure)
    index = {label: i for i, label in enumerate(library.labels)}
    problems = []
    for i, terms in enumerate(problem.commutator_terms):
        support = np.array([index[label] for label in terms], dtype=np.int64)
        targets = {index[label]: value for label, value in terms.items()}
        chosen = np.array([assigned[label] for label in terms], dtype=np.int64)
        problems.append(FragmentProblem(i, chosen, support, targets, library.home, moments.covariance))
    return problems


def describe(structure: PivotStructure, library=None) -> dict:
    """Counts for the report: contexts, classes, members, CNOT/CZ counts of the circuits."""
    sizes = np.array([len(g) for g in structure.groups])
    out = {"classes": structure.classes, "n_classes": structure.n_classes, "pivots": structure.n_pivots,
           "pool_strings": structure.n_pool_strings, "contexts": structure.n_contexts,
           "members_mean": float(sizes.mean()), "members_max": int(sizes.max()),
           "extra_products": len(structure.extra_labels),
           "reads_per_gradient_mean": float(np.mean([len(c) for c in structure.contributions]))}
    if library is not None:
        cz = library.two_qubit_counts()
        out.update({"two_qubit_mean": float(cz.mean()), "two_qubit_max": int(cz.max()),
                    "bound_n_minus_3": library.n_qubits - 3})
    return out
