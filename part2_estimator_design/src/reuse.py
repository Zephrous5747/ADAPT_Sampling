"""Reuse of the last energy evaluation's measurement data in the generator selection.

Ikhtiarudin et al. (arXiv:2507.16879) save the Pauli outcomes of the final VQE energy
evaluation of an ADAPT iteration and reuse them, in the next iteration's gradient
estimation, for every Pauli string of ``[H, A_k]`` that shares a measurement basis with
a Hamiltonian clique; shots are allocated by variance (VMSA/VPSR) over qubit-wise
commuting (QWC) cliques.  Their reported saving from reuse is about 6 percentage
points of the naive count on top of grouping (38.6% -> 32.3% of the naive shots), and
they report shot budgets, not a statistical decision.

This module puts that mechanism into the Part II framework so that it can be compared
on the selection decision itself:

* the **energy contexts** are the groups of ``H`` (fully commuting by sorted insertion,
  the grouping of the ``C_opt`` model of Paper A, or QWC) appended to the library as
  contexts that already hold shots;
* the **credit** is the optimal allocation of one energy estimate to standard error
  ``eps`` (``n_alpha = sigma_alpha sum(sigma) / eps^2``), drawn once from the state's
  outcome distributions.  The same shots are already paid for by the energy
  evaluation, so they are not charged to the selection;
* **assignment** (:func:`reuse_fragment_problems`): a gradient Pauli that an energy
  context measures is read from that context (the best funded one), otherwise from
  its parent context.  This is Ikhtiarudin's fixed rule; the covariance-optimal
  split between energy and parent contexts is level II-A on the same library
  ("ours + reuse").

The allocation of new shots is the optimal one of Part I rather than their
variance-proportional rule, so this baseline is at least as strong as their allocation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import part1_bridge  # noqa: F401  (puts Part I's src on the path)
from contexts import build_context_library, contiguous_blocks, qwc_groups
from design import FragmentProblem
from learning import LearningConfig
from pauli_fc import greedy_fc_groups  # Part I
from pauli_ops import PauliEvaluator  # Part I

ENERGY_ERROR = 1e-3  # Hartree: one standard error per energy estimate, as in the C_opt model
ENERGY_TOLERANCE = 1e-7


@dataclass(frozen=True)
class ReuseSpec:
    """A learned configuration run on the energy-augmented library."""

    learning: LearningConfig
    grouping: str = "fc"  # "fc" or "qwc"
    free_data: bool = True  # False: same grouping, plain home assignment, no data held in advance
    assign: str = "fixed"  # "fixed": every Pauli an energy context measures is read from it (Ikhtiarudin);
    #                        "home": start from the home design and let II-A split onto the energy contexts


def hamiltonian_terms(problem, case_id: str | None = None) -> dict[str, float]:
    """Pauli terms of H without the identity, checked against the problem's own state.

    Trajectory problems carry the terms of the Hamiltonian they were built from; for a
    fixed-state case the Hamiltonian is rebuilt, and the energy of the problem's state
    must agree with it (an SCF that picks other orbitals within a degenerate shell
    would expand H in other Pauli products and silently break the overlap).
    """
    terms = problem.metadata.get("hamiltonian_terms")
    rebuilt = terms is None
    if rebuilt:
        from chemistry import build_qubit_hamiltonian  # Part I

        spec = part1_bridge.get_case(case_id or problem.case_id)
        ham = build_qubit_hamiltonian(spec)
        terms = part1_bridge_terms(ham)
    constant = float(np.real(terms.get("I" * problem.n_qubits, 0.0)))
    real = {p: float(np.real(c)) for p, c in terms.items() if set(p) != {"I"}}
    energy = constant + PauliEvaluator(problem.evaluator.state).fragment_mean_std(real)[0]
    if abs(energy - problem.state_energy) > ENERGY_TOLERANCE:
        raise ValueError(f"{problem.case_id}: Hamiltonian terms give E = {energy:.9f}, the state's energy is "
                         f"{problem.state_energy:.9f}: they do not describe the same orbitals")
    if rebuilt:
        _check_commutators(problem, ham)
    return real


def _check_commutators(problem, hamiltonian, arms: int | None = None) -> None:
    """The rebuilt H must reproduce the stored commutator expansions of a few generators.

    The energy check cannot see a rotation within a degenerate orbital shell for a
    determinant state (the HF energy is invariant), but the Pauli expansions of
    ``[H, G_i]`` change, and the overlap of H's support with the gradient support is
    exactly what the reuse baseline measures.
    """
    from gradients import commutator  # Part I
    from pool import uccsd_pool  # Part I

    pool = uccsd_pool(hamiltonian.n_qubits, hamiltonian.n_electrons)
    picks = range(len(pool)) if arms is None else sorted({0, len(pool) // 2, len(pool) - 1})[:arms]
    for i in picks:
        fresh = part1_bridge_terms_of(commutator(hamiltonian.operator, pool[i].qubit_operator),
                                      hamiltonian.n_qubits)
        stored = problem.commutator_terms[i]
        keys = set(fresh) | set(stored)
        worst = max(abs(float(np.real(fresh.get(k, 0.0))) - stored.get(k, 0.0)) for k in keys)
        if worst > 1e-8:
            raise ValueError(f"{problem.case_id}: the rebuilt Hamiltonian disagrees with the stored commutator "
                             f"of generator {i} by {worst:.2e}: the cached problem used other orbitals")


def part1_bridge_terms_of(operator, n_qubits: int) -> dict:
    from pauli_fc import openfermion_qubitop_to_dict  # Part I

    return openfermion_qubitop_to_dict(operator, n_qubits)


def part1_bridge_terms(hamiltonian) -> dict:
    from pauli_fc import openfermion_qubitop_to_dict  # Part I

    return openfermion_qubitop_to_dict(hamiltonian.operator, hamiltonian.n_qubits)


def energy_groups(terms: dict[str, float], grouping: str) -> list[list[str]]:
    """Groups of H by sorted insertion (decreasing ``|h_p|``): ``fc`` or ``qwc``."""
    if grouping == "fc":
        order = sorted(terms, key=lambda p: (-abs(terms[p]), p))
        return greedy_fc_groups(order, sort=False)
    if grouping == "qwc":
        return qwc_groups(list(terms), weights=terms)
    raise ValueError("grouping must be 'fc' or 'qwc'")


def parent_groups(problem, grouping: str) -> list[list[str]]:
    """Parent contexts of the gradient support: Part I's FC grouping, or QWC."""
    if grouping == "fc":
        return [list(g) for g in problem.parent_fc_groups()]
    if grouping == "qwc":
        return qwc_groups(problem.universal_support)
    raise ValueError("grouping must be 'fc' or 'qwc'")


class ReuseLibrary:
    """The parent contexts of the gradient support plus the energy contexts of H."""

    def __init__(self, problem, terms: dict[str, float], strategy: str = "mass", grouping: str = "fc") -> None:
        self.grouping = grouping
        self.terms = terms
        parents = parent_groups(problem, grouping)
        self.energy = energy_groups(terms, grouping)
        self.n_parent = len(parents)
        # QWC cliques are measured in a product basis: no entangling gates, and the Paulis a context measures
        # are those of its local basis.  (An earlier version completed them like fully commuting groups,
        # which gave circuits with entangling gates and a larger measured group than a product measurement.)
        blocks = contiguous_blocks(problem.n_qubits, 1) if grouping == "qwc" else None
        self.library = build_context_library(problem, "canonical" if blocks else strategy, groups=parents,
                                             extra_groups=self.energy, blocks=blocks)
        self.energy_context = np.arange(self.n_parent, self.n_parent + len(self.energy))

    def credit(self, moments, epsilon: float = ENERGY_ERROR) -> np.ndarray:
        """Shots per context from one energy estimate to standard error ``epsilon``.

        ``moments`` supplies the exact fragment variances on the state (planning of the
        energy evaluation, which the energy measurement itself would estimate).
        """
        index = {label: i for i, label in enumerate(self.library.labels)}
        sigma = np.zeros(len(self.energy))
        for k, group in enumerate(self.energy):
            ids = np.array([index[p] for p in group])
            x = np.array([self.terms[p] for p in group])
            sigma[k] = math.sqrt(max(float(x @ moments.covariance(int(self.n_parent + k), ids) @ x), 0.0))
        shots = np.zeros(self.library.n_contexts, dtype=np.int64)
        shots[self.energy_context] = np.ceil(sigma * sigma.sum() / epsilon ** 2 - 1e-9).astype(np.int64)
        return shots

    def overlap(self, problem) -> dict:
        """How much of the gradient support the energy contexts measure."""
        owners = self.library.contexts_of()
        index = {label: i for i, label in enumerate(self.library.labels)}
        support = np.array([index[p] for p in problem.universal_support])
        covered = np.array([bool((owners[l] >= self.n_parent).any()) for l in support])
        mass = np.zeros(self.library.n_library)
        for terms in problem.commutator_terms:
            for label, value in terms.items():
                mass[index[label]] += abs(value)
        total = float(mass[support].sum())
        return {"support": int(support.size), "covered": int(covered.sum()),
                "covered_fraction": float(covered.mean()),
                "covered_mass_fraction": float(mass[support][covered].sum() / total) if total else 0.0,
                "in_h_support": int(sum(p in self.terms for p in problem.universal_support))}


def reuse_fragment_problems(problem, reuse: ReuseLibrary, moments, credit: np.ndarray) -> list[FragmentProblem]:
    """II-0 coefficients with every Pauli read from the best-funded energy context that measures it."""
    library = reuse.library
    owners = library.contexts_of()
    index = {label: i for i, label in enumerate(library.labels)}
    problems = []
    for i, terms in enumerate(problem.commutator_terms):
        support = np.array([index[label] for label in terms], dtype=np.int64)
        targets = {index[label]: value for label, value in terms.items()}
        chosen = np.empty(support.size, dtype=np.int64)
        for k, l in enumerate(support):
            energy = owners[l][owners[l] >= reuse.n_parent]
            chosen[k] = energy[int(np.argmax(credit[energy]))] if energy.size else library.home[l]
        problems.append(FragmentProblem(i, chosen, support, targets, library.home, moments.covariance))
    return problems
