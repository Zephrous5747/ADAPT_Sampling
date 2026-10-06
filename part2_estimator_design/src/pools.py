"""Operator pools other than the spin-orbital UCCSD pool of Part I.

Paper A is built on the fermionic UCCSD pool (K = 26, 92, 140), whose gradients share
about four generators per Pauli product.  A reviewer will ask whether the measurement
gains survive on the pools that the ADAPT literature has moved to, whose generators
have no Jordan-Wigner strings and far fewer Pauli terms.  A problem for another pool is
requested as ``<case>@<pool>`` (e.g. ``LiH_R3p0_HF@qubit``) and built here on the same
Hamiltonian and state with the same pipeline (``gradients.GradientProblem``), so every
Part II script and test runs on it unchanged.

``qubit``
    Qubit-ADAPT (Tang et al., PRX Quantum 2, 020310): every Pauli string of the
    Jordan-Wigner image of a UCCSD generator, with its ``Z`` operators removed, as
    its own generator ``i P`` (odd number of ``Y``); duplicates across generators are
    kept once.  Hardware-efficient single-string generators.
``qeb``
    Qubit-excitation-based pool (Yordanov et al., Commun. Phys. 4, 228): the same index
    sets as UCCSD with the excitation written in qubit operators,
    ``E = s+_a s-_i`` and ``E = s+_a s+_b s-_j s-_i``, ``G = E - E^dagger``: no ``Z``
    strings, so each generator has two (single) or eight (double) Pauli terms.

``gsd_qeb``, ``gsd_qubit``, ``ceo``
    The pools of the CEO-ADAPT paper (Ramoa et al., npj Quantum Inf. 11, 86, arXiv:2407.08696) on
    *generalised* index sets (every spin-conserving single and double among all spin
    orbitals, not only occupied -> virtual): all qubit excitations (QEs); the Pauli strings
    of those QEs; and the one-variational-parameter coupled exchange operators (OVP-CEOs).
    On every set of four spin orbitals two alpha and two beta there are two QEs
    ``E1 = T_{a1 b1 -> a2 b2}``, ``E2 = T_{a2 b1 -> a1 b2}`` and two OVP-CEOs ``E1 +- E2``;
    on four orbitals of one spin there are three QEs and six OVP-CEOs (sums and
    differences of pairs); singles are QEs.  (The CEO algorithm's final multi-parameter
    step is an ansatz-update rule; the *selection* problem, which is what is measured here,
    is the gradients of the OVP-CEO pool.)

Not implemented (needs the definition from the paper): the symmetry-adapted minimal
complete pool of Shkolnikov et al.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np

import part1_bridge  # noqa: F401  (puts Part I's src on the path)
from gradients import GradientProblem, _real_terms, commutator  # Part I
from pauli_fc import openfermion_qubitop_to_dict  # Part I
from pauli_ops import PauliEvaluator  # Part I
from pool import Generator, uccsd_pool  # Part I

LOGGER = logging.getLogger(__name__)
POOLS = ("uccsd", "qubit", "qeb", "gsd_qeb", "gsd_qubit", "ceo")
POOL_CACHE_VERSION = "1"


def split_pool_case(case_id: str) -> tuple[str, str] | None:
    """``("LiH_R3p0_HF", "qubit")`` for ``"LiH_R3p0_HF@qubit"``, else ``None``."""
    if "@" not in case_id:
        return None
    base, pool = case_id.split("@", 1)
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; choose one of {POOLS}")
    return base, pool


def _qubit_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    from openfermion import QubitOperator

    strings: dict[str, None] = {}
    for generator in uccsd_pool(n_qubits, n_electrons):
        terms = openfermion_qubitop_to_dict(generator.qubit_operator, n_qubits)
        for label in terms:
            stripped = "".join("I" if c == "Z" else c for c in label)
            if stripped.count("Y") % 2 == 1:  # real generator i P
                strings[stripped] = None
    pool = []
    for label in sorted(strings):
        op = QubitOperator(" ".join(f"{c}{q}" for q, c in enumerate(label) if c != "I"), 1j)
        pool.append(Generator(len(pool), f"Q {label}", "Q", op))
    return pool


def _qeb_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    from openfermion import QubitOperator, hermitian_conjugated

    def raise_(q):  # |0> -> |1>, the JW creation operator without its Z string
        return 0.5 * QubitOperator(f"X{q}") - 0.5j * QubitOperator(f"Y{q}")

    def lower(q):
        return 0.5 * QubitOperator(f"X{q}") + 0.5j * QubitOperator(f"Y{q}")

    def anti_hermitian(excitation):
        return excitation - hermitian_conjugated(excitation)

    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    pool: list[Generator] = []
    for i in occupied:
        for a in virtual:
            if i % 2 == a % 2:
                pool.append(Generator(len(pool), f"S {i} -> {a}", "S", anti_hermitian(raise_(a) * lower(i))))
    for i, j in combinations(occupied, 2):
        occupied_alpha = (i % 2 == 0) + (j % 2 == 0)
        for a, b in combinations(virtual, 2):
            if occupied_alpha == (a % 2 == 0) + (b % 2 == 0):
                excitation = raise_(a) * raise_(b) * lower(j) * lower(i)
                pool.append(Generator(len(pool), f"D {i},{j} -> {a},{b}", "D", anti_hermitian(excitation)))
    return pool


def _qe_toolkit():
    from openfermion import QubitOperator, hermitian_conjugated

    def raise_(q):  # |0> -> |1>, the JW creation operator without its Z string
        return 0.5 * QubitOperator(f"X{q}") - 0.5j * QubitOperator(f"Y{q}")

    def lower(q):
        return 0.5 * QubitOperator(f"X{q}") + 0.5j * QubitOperator(f"Y{q}")

    def anti_hermitian(excitation):
        return excitation - hermitian_conjugated(excitation)

    return raise_, lower, anti_hermitian


def _generalised_qes(n_qubits: int):
    """Every unique spin-conserving QE on all spin orbitals, grouped as the CEO paper does.

    Returns ``(singles, opposite, same)``: lists of ``(label, operator)`` for singles, of
    ``(set label, [E1, E2])`` for four orbitals two alpha and two beta, and of ``(set label,
    [E1, E2, E3])`` for four orbitals of one spin.  Even spin orbitals are alpha, odd beta.
    """
    raise_, lower, anti_hermitian = _qe_toolkit()
    alpha = list(range(0, n_qubits, 2))
    beta = list(range(1, n_qubits, 2))
    singles, opposite, same = [], [], []
    for orbitals in (alpha, beta):
        for p, q in combinations(orbitals, 2):
            singles.append((f"S {p} -> {q}", anti_hermitian(raise_(q) * lower(p))))
    for a1, a2 in combinations(alpha, 2):
        for b1, b2 in combinations(beta, 2):
            e1 = anti_hermitian(raise_(a2) * raise_(b2) * lower(b1) * lower(a1))
            e2 = anti_hermitian(raise_(a1) * raise_(b2) * lower(b1) * lower(a2))
            opposite.append((f"{a1},{a2}|{b1},{b2}", [(f"D {a1},{b1} -> {a2},{b2}", e1),
                                                       (f"D {a2},{b1} -> {a1},{b2}", e2)]))
    for orbitals in (alpha, beta):
        for o1, o2, o3, o4 in combinations(orbitals, 4):
            parts = []
            for (s1, s2), (t1, t2) in (((o1, o2), (o3, o4)), ((o1, o3), (o2, o4)), ((o1, o4), (o2, o3))):
                parts.append((f"D {s1},{s2} -> {t1},{t2}",
                              anti_hermitian(raise_(t1) * raise_(t2) * lower(s2) * lower(s1))))
            same.append((f"{o1},{o2},{o3},{o4}", parts))
    return singles, opposite, same


def _gsd_qeb_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    singles, opposite, same = _generalised_qes(n_qubits)
    pool: list[Generator] = []
    for label, op in singles:
        pool.append(Generator(len(pool), label, "S", op))
    for _, parts in opposite + same:
        for label, op in parts:
            pool.append(Generator(len(pool), label, "D", op))
    return pool


def _ceo_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    """OVP-CEOs: QE singles, and sums/differences of the QEs on each set of four spin orbitals."""
    singles, opposite, same = _generalised_qes(n_qubits)
    pool: list[Generator] = []
    for label, op in singles:
        pool.append(Generator(len(pool), label, "S", op))
    for name, parts in opposite + same:
        for (la, a), (lb, b) in combinations(parts, 2):
            pool.append(Generator(len(pool), f"C+ {la} + {lb}", "C", a + b))
            pool.append(Generator(len(pool), f"C- {la} - {lb}", "C", a - b))
    return pool


def _gsd_qubit_pool(n_qubits: int, n_electrons: int) -> list[Generator]:
    from openfermion import QubitOperator

    singles, opposite, same = _generalised_qes(n_qubits)
    operators = [op for _, op in singles] + [op for _, parts in opposite + same for _, op in parts]
    strings: dict[str, None] = {}
    for op in operators:
        for label in openfermion_qubitop_to_dict(op, n_qubits):
            if label.count("Y") % 2 == 1:
                strings[label] = None
    pool = []
    for label in sorted(strings):
        term = QubitOperator(" ".join(f"{c}{q}" for q, c in enumerate(label) if c != "I"), 1j)
        pool.append(Generator(len(pool), f"Q {label}", "Q", term))
    return pool


POOL_BUILDERS = {"uccsd": uccsd_pool, "qubit": _qubit_pool, "qeb": _qeb_pool,
                 "gsd_qeb": _gsd_qeb_pool, "gsd_qubit": _gsd_qubit_pool, "ceo": _ceo_pool}


def build_pool_problem(case_id: str, pool_name: str, *, validate: bool = True) -> GradientProblem:
    """The gradient problem of a Part I case with another generator pool (self-consistent build)."""
    from chemistry import build_qubit_hamiltonian, validate_hamiltonian
    from states import prepare_state

    spec = part1_bridge.get_case(case_id)
    hamiltonian = build_qubit_hamiltonian(spec)
    validation = validate_hamiltonian(spec, hamiltonian) if validate else {"validated": False}
    state_energy, state = prepare_state(hamiltonian, spec.state)
    generators = POOL_BUILDERS[pool_name](hamiltonian.n_qubits, hamiltonian.n_electrons)
    evaluator = PauliEvaluator(state)
    labels, kinds, gradients, terms_list = [], [], [], []
    for generator in generators:
        operator = commutator(hamiltonian.operator, generator.qubit_operator)
        terms = _real_terms(openfermion_qubitop_to_dict(operator, hamiltonian.n_qubits))
        if not terms:  # a generator that commutes with H everywhere carries no gradient observable
            continue
        labels.append(generator.label)
        kinds.append(generator.kind)
        gradients.append(evaluator.fragment_mean_std(terms)[0])
        terms_list.append(terms)
    problem = GradientProblem(
        case_id=f"{case_id}@{pool_name}",
        state_name=spec.state.upper(),
        n_qubits=hamiltonian.n_qubits,
        state_energy=state_energy,
        labels=labels,
        kinds=kinds,
        gradients=np.array(gradients),
        commutator_terms=terms_list,
        evaluator=evaluator,
        metadata={
            "n_electrons": hamiltonian.n_electrons,
            "n_spatial_orbitals": hamiltonian.n_spatial_orbitals,
            "pool": pool_name,
            "pool_size": len(labels),
            "pool_generators_built": len(generators),
            "hamiltonian_pauli_terms": hamiltonian.n_pauli_terms,
            "rhf_energy_hartree": hamiltonian.rhf_energy,
            "validation": validation,
            "hamiltonian_terms": openfermion_qubitop_to_dict(hamiltonian.operator, hamiltonian.n_qubits),
            **hamiltonian.metadata,
        },
    )
    problem.parent_fc_groups()
    return problem


def load_pool_problem(case_id: str, cache_dir: Path) -> GradientProblem:
    """Build or reload ``<case>@<pool>`` (cached next to the other problems, never tracked by git)."""
    import pickle

    base, pool = split_pool_case(case_id)
    path = Path(cache_dir) / f"{base}@{pool}_v{POOL_CACHE_VERSION}.pickle"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    problem = build_pool_problem(base, pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return problem
