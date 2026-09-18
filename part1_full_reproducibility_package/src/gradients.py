"""The Part I gradient-measurement problem for one fixed state.

For a Hamiltonian ``H``, a fixed state ``|psi>`` and a generator pool
``{G_i}``, this module builds the gradient operators

    C_i = [H, G_i],   g_i = <psi| C_i |psi>,

expands every ``C_i`` in the shared Pauli-coordinate system, and exposes the
fully commuting fragment standard deviations that the M1, M2 and M3 shot models
consume.  ``C_i`` is Hermitian because ``H`` is Hermitian and ``G_i`` is
anti-Hermitian, so all Pauli coefficients are real.

A fragment is the part of one gradient operator that lives inside one fully
commuting measurement context:

    F_{i,alpha} = sum_{l in M_alpha and supp(C_i)} A_{i,l} R_l.

Only non-empty fragments are evaluated, which is what keeps the large universal
supports tractable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
PROGRESS_INTERVAL = 20

from pauli_fc import greedy_fc_groups, openfermion_qubitop_to_dict
from pauli_ops import PauliEvaluator

COEFFICIENT_TOLERANCE = 1e-12

# Gradients that vanish by symmetry come out at the 1e-8 numerical noise floor of
# the SCF and integral pipeline, while the smallest physically non-zero gradient
# seen in the Part I cases is ~1e-3.  Any threshold between these is equivalent;
# 1e-6 sits in the middle of that plateau.  Counts are reported at several
# thresholds so the choice stays visible rather than buried.
NONZERO_GRADIENT_THRESHOLD = 1e-6
NONZERO_GRADIENT_THRESHOLD_SCAN = (1e-4, 1e-6, 1e-8, 1e-9)


def commutator(left, right):
    """Return the compressed commutator ``[left, right]`` of two operators."""
    result = left * right - right * left
    result.compress(abs_tol=COEFFICIENT_TOLERANCE)
    return result


@dataclass
class GradientProblem:
    """Gradient operators, values and fragment variances for one fixed state."""

    case_id: str
    state_name: str
    n_qubits: int
    state_energy: float
    labels: list[str]
    kinds: list[str]
    gradients: np.ndarray
    commutator_terms: list[dict[str, float]]
    evaluator: PauliEvaluator
    metadata: dict = field(default_factory=dict)
    fc_ordering: str = "weight"
    fc_seed: int = 0
    _parent_groups: list[list[str]] | None = field(default=None, repr=False)

    def set_fc_routine(self, ordering: str, seed: int = 0) -> None:
        """Switch the fully commuting grouping routine and drop cached groupings.

        Used by the routine-independence study: the physics is untouched, only the
        insertion order of the first-fit grouping changes, so everything derived
        from the grouping has to be rebuilt.
        """
        self.fc_ordering = ordering
        self.fc_seed = seed
        self._parent_groups = None
        self._parent_sigmas = None

    @property
    def n_generators(self) -> int:
        return len(self.labels)

    @property
    def abs_gradients(self) -> np.ndarray:
        return np.abs(self.gradients)

    @property
    def universal_support(self) -> list[str]:
        """The parent Pauli support ``B_0 = union_i supp(C_i)``, sorted."""
        support: set[str] = set()
        for terms in self.commutator_terms:
            support.update(terms)
        return sorted(support)

    def count_nonzero_gradients(self, threshold: float = NONZERO_GRADIENT_THRESHOLD) -> int:
        """Number of gradients above ``threshold`` in absolute value."""
        return int((self.abs_gradients > threshold).sum())

    @property
    def n_nonzero_gradients(self) -> int:
        return self.count_nonzero_gradients()

    def nonzero_gradient_scan(self) -> dict[str, int]:
        """Non-zero counts across thresholds, so the cut-off is auditable."""
        return {
            f"{threshold:g}": self.count_nonzero_gradients(threshold)
            for threshold in NONZERO_GRADIENT_THRESHOLD_SCAN
        }

    def ranking(self) -> list[int]:
        """Generator indices ordered by decreasing absolute gradient."""
        return sorted(
            range(self.n_generators),
            key=lambda i: (-self.abs_gradients[i], i),
        )

    def top_gap(self) -> float:
        """Absolute-gradient gap between the best and second-best generator."""
        order = self.ranking()
        if len(order) < 2:
            return float(self.abs_gradients[order[0]])
        return float(self.abs_gradients[order[0]] - self.abs_gradients[order[1]])

    def parent_fc_groups(self) -> list[list[str]]:
        """Deterministic greedy fully commuting grouping of ``B_0`` (cached)."""
        if self._parent_groups is None:
            self._parent_groups = greedy_fc_groups(
                self.universal_support, ordering=self._ordering, seed=self.fc_seed
            )
        return self._parent_groups

    @property
    def _ordering(self) -> str:
        return getattr(self, "fc_ordering", "weight")

    def parent_fragment_sigmas(self) -> np.ndarray:
        """Fragment standard deviations against the parent contexts (cached).

        M1 and M3 share the same parent grouping, and M1 is evaluated at more than
        one target radius, so this is computed once per problem.  Loaded from a
        pickle written before this method existed, the attribute is simply absent
        and is rebuilt on first use.
        """
        cached = getattr(self, "_parent_sigmas", None)
        if cached is None:
            cached = self.fragment_sigmas(self.parent_fc_groups())
            self._parent_sigmas = cached
        return cached

    def individual_fc_groups(self) -> list[list[list[str]]]:
        """Deterministic greedy fully commuting grouping of each ``supp(C_i)``."""
        groups = []
        for index, terms in enumerate(self.commutator_terms):
            groups.append(
                greedy_fc_groups(
                    terms.keys(), ordering=self._ordering, seed=getattr(self, "fc_seed", 0)
                )
            )
            self._log_progress("individual FC grouping", index)
        return groups

    def fragment_sigmas(self, groups: Sequence[Sequence[str]]) -> np.ndarray:
        """Fragment standard deviations ``sigma[i, alpha]`` for shared contexts.

        Returns a dense ``(n_generators, len(groups))`` array.  Entries for
        empty fragments are exactly zero and are never evaluated.
        """
        group_of_pauli = {
            pauli: index for index, group in enumerate(groups) for pauli in group
        }
        sigmas = np.zeros((self.n_generators, len(groups)))
        for i, terms in enumerate(self.commutator_terms):
            for group_index, fragment in self._bucket(terms, group_of_pauli).items():
                sigmas[i, group_index] = self.evaluator.fragment_std(fragment)
            self._log_progress("shared-context fragment sigmas", i)
        return sigmas

    def individual_fragment_sigmas(
        self, groups_per_generator: Sequence[Sequence[Sequence[str]]]
    ) -> list[np.ndarray]:
        """Fragment standard deviations for per-generator groupings."""
        result = []
        for index, (terms, groups) in enumerate(
            zip(self.commutator_terms, groups_per_generator)
        ):
            result.append(
                np.array(
                    [
                        self.evaluator.fragment_std({p: terms[p] for p in group})
                        for group in groups
                    ]
                )
            )
            self._log_progress("individual fragment sigmas", index)
        return result

    def _log_progress(self, stage: str, index: int) -> None:
        if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == self.n_generators:
            LOGGER.info("%s: %d/%d generators", stage, index + 1, self.n_generators)

    @staticmethod
    def _bucket(
        terms: Mapping[str, float], group_of_pauli: Mapping[str, int]
    ) -> dict[int, dict[str, float]]:
        """Split one gradient's terms into its non-empty per-context fragments."""
        buckets: dict[int, dict[str, float]] = {}
        for pauli, coefficient in terms.items():
            group_index = group_of_pauli.get(pauli)
            if group_index is None:
                continue
            buckets.setdefault(group_index, {})[pauli] = coefficient
        return buckets


def build_gradient_problem(spec, *, validate: bool = True) -> GradientProblem:
    """Run the full chemistry-to-gradients pipeline for one case.

    With ``validate`` set, the qubit Hamiltonian must reproduce the RHF and FCI
    energies before any gradient is computed.
    """
    from chemistry import build_qubit_hamiltonian, validate_hamiltonian
    from pool import uccsd_pool
    from states import prepare_state

    hamiltonian = build_qubit_hamiltonian(spec)
    validation = (
        validate_hamiltonian(spec, hamiltonian) if validate else {"validated": False}
    )
    state_energy, state = prepare_state(hamiltonian, spec.state)
    generators = uccsd_pool(hamiltonian.n_qubits, hamiltonian.n_electrons)
    evaluator = PauliEvaluator(state)

    labels, kinds, gradient_values, commutator_terms = [], [], [], []
    for generator in generators:
        operator = commutator(hamiltonian.operator, generator.qubit_operator)
        terms = _real_terms(
            openfermion_qubitop_to_dict(operator, hamiltonian.n_qubits)
        )
        mean, _ = evaluator.fragment_mean_std(terms)
        labels.append(generator.label)
        kinds.append(generator.kind)
        gradient_values.append(mean)
        commutator_terms.append(terms)

    return GradientProblem(
        case_id=spec.case_id,
        state_name=spec.state.upper(),
        n_qubits=hamiltonian.n_qubits,
        state_energy=state_energy,
        labels=labels,
        kinds=kinds,
        gradients=np.array(gradient_values),
        commutator_terms=commutator_terms,
        evaluator=evaluator,
        metadata={
            "n_electrons": hamiltonian.n_electrons,
            "n_spatial_orbitals": hamiltonian.n_spatial_orbitals,
            "pool_size": len(generators),
            "hamiltonian_pauli_terms": hamiltonian.n_pauli_terms,
            "rhf_energy_hartree": hamiltonian.rhf_energy,
            "validation": validation,
            **hamiltonian.metadata,
        },
    )


def _real_terms(terms: Mapping[str, complex]) -> dict[str, float]:
    """Drop the (numerically zero) imaginary part of a Hermitian expansion."""
    largest_imaginary = max((abs(c.imag) for c in terms.values()), default=0.0)
    if largest_imaginary > 1e-9:
        raise ValueError(
            f"commutator is not Hermitian: largest imaginary coefficient "
            f"{largest_imaginary:.3e}"
        )
    return {pauli: float(c.real) for pauli, c in terms.items()}
