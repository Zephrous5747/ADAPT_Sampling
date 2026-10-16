"""States along an ADAPT-VQE trajectory: Phase 2 snapshots and the Phase 4 driver.

Part I measured gradients on two fixed states per geometry, the HF determinant and
the CISD ground state as a proxy for "a correlated state later in ADAPT".  This
module produces the real thing: the ADAPT-VQE ansatz

    |psi_k> = exp(theta_k G_{i_k}) ... exp(theta_1 G_{i_1}) |HF>,

grown one generator at a time from the same UCCSD pool, with all parameters
re-optimised after each addition.  The generator at step ``k`` is the one of
largest ``|g_i| = |<psi_{k-1}| [H, G_i] |psi_{k-1}>|``: exactly the selection
problem Parts I and II measure.

Everything is exact linear algebra in the ``(N_alpha, N_beta)`` sector (36, 225
and 441 determinants for H4, LiH and H2O), which the pool conserves.  In the sector
``g_i = 2 Re <H psi | G_i psi>`` because ``G_i`` is anti-Hermitian.

Saved states become ordinary benchmark cases: ``<geometry>_ADAPT<k>`` is the state
after ``k`` generators of the trajectory started from ``<geometry>_HF``, and every
Part II script runs on it unchanged.  Its gradient problem is built by Part I's
code from the *same* Hamiltonian object the trajectory was run with, and saved
next to the states.  Reusing the cached HF problem instead would be wrong: a fresh
SCF may return orbitals of opposite sign (observed on H4), which flips the signs of
some commutator coefficients relative to the saved states.  The two problems agree
in every ``|g_i|`` at HF, and all Part II statistics are invariant to such signs.
:func:`problem_at` rebuilds the gradients from the commutator expansions and checks
them against the sector computation.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

import part1_bridge  # noqa: F401  (Part I's src on the path)
from part1_bridge import DEFAULT_CACHE, get_case

CHEMICAL_ACCURACY = 1.6e-3  # hartree
TRAJECTORY_CASE = re.compile(r"^(?P<geometry>.+)_ADAPT(?P<k>\d+)$")
GRADIENT_AGREEMENT = 1e-8


def trajectory_path(base_case: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / "trajectory" / f"{base_case}_adapt.npz"


def split_case(case_id: str) -> tuple[str, int] | None:
    """``("H4_..._HF", k)`` for a trajectory case id, ``None`` otherwise."""
    match = TRAJECTORY_CASE.match(case_id)
    if match is None:
        return None
    return f"{match['geometry']}_HF", int(match["k"])


@dataclass
class AdaptSystem:
    """Hamiltonian and pool of one geometry, restricted to the reference sector."""

    base_case: str
    n_qubits: int
    n_electrons: int
    indices: np.ndarray  # sector basis indices, OpenFermion ordering
    hamiltonian: sp.csr_matrix
    generators: list[sp.csr_matrix]
    labels: list[str]
    hf: np.ndarray
    fci_energy: float
    metadata: dict = field(default_factory=dict)

    problem: object = None  # Part I GradientProblem on the same Hamiltonian, HF state

    @classmethod
    def build(cls, base_case: str) -> "AdaptSystem":
        from chemistry import build_qubit_hamiltonian, validate_hamiltonian  # Part I
        from pool import uccsd_pool
        from states import basis_index, sector_block, sector_indices

        spec = get_case(base_case)
        if spec.state.upper() != "HF":
            raise ValueError("a trajectory starts from an HF case")
        ham = build_qubit_hamiltonian(spec)
        validation = validate_hamiltonian(spec, ham)
        indices = np.asarray(sector_indices(ham.n_qubits, ham.n_alpha, ham.n_beta), dtype=np.int64)
        block = sector_block(ham.operator, ham.n_qubits, indices).real.tocsr()
        pool = uccsd_pool(ham.n_qubits, ham.n_electrons)
        generators = [sector_block(g.qubit_operator, ham.n_qubits, indices).tocsr() for g in pool]
        for g in generators:
            if abs(g.imag).max() > 1e-12 if g.nnz else False:
                raise ValueError("pool generators are expected to be real in the computational basis")
        generators = [g.real.tocsr() for g in generators]
        hf = np.zeros(indices.size)
        hf[np.searchsorted(indices, basis_index(range(ham.n_electrons), ham.n_qubits))] = 1.0
        dense = block.toarray()
        fci = float(np.linalg.eigvalsh(dense)[0])
        system = cls(base_case, ham.n_qubits, ham.n_electrons, indices, block, generators,
                     [g.label for g in pool], hf, fci,
                     {"rhf_energy": ham.rhf_energy, "sector_dimension": int(indices.size)})
        system.problem = _gradient_problem(spec, ham, pool, validation)
        return system

    # --- states ------------------------------------------------------------------

    def state(self, ops: list[int], params: np.ndarray) -> np.ndarray:
        psi = self.hf.copy()
        for op, theta in zip(ops, params):
            psi = expm_multiply(theta * self.generators[op], psi)
        return psi

    def energy(self, psi: np.ndarray) -> float:
        return float(psi @ (self.hamiltonian @ psi))

    def gradients(self, psi: np.ndarray) -> np.ndarray:
        """``g_i = <psi|[H, G_i]|psi> = 2 <H psi | G_i psi>`` for every pool generator."""
        h_psi = self.hamiltonian @ psi
        return np.array([2.0 * float(h_psi @ (g @ psi)) for g in self.generators])

    def full_state(self, psi: np.ndarray) -> np.ndarray:
        full = np.zeros(2 ** self.n_qubits, dtype=complex)
        full[self.indices] = psi
        return full

    # --- VQE -----------------------------------------------------------------------

    def _energy_and_gradient(self, params: np.ndarray, ops: list[int]) -> tuple[float, np.ndarray]:
        """Energy and its exact parameter gradient by one forward and one reverse sweep."""
        psi = self.state(ops, params)
        energy = self.energy(psi)
        lam = self.hamiltonian @ psi
        grad = np.zeros(len(ops))
        for k in range(len(ops) - 1, -1, -1):
            g = self.generators[ops[k]]
            grad[k] = 2.0 * float(lam @ (g @ psi))
            psi = expm_multiply(-params[k] * g, psi)
            lam = expm_multiply(-params[k] * g, lam)
        return energy, grad

    def optimise(self, ops: list[int], start: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(self._energy_and_gradient, start, args=(ops,), jac=True,
                          method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
        self.last_evaluations = int(result.nfev)  # energy-and-gradient evaluations
        return np.asarray(result.x), float(result.fun)


@dataclass
class AdaptStep:
    iteration: int  # number of generators in the ansatz
    ops: list[int]
    params: np.ndarray
    energy: float
    error: float
    gradients: np.ndarray  # of every pool generator at this state (exact)
    selected: int | None = None  # the generator added next
    selection: dict = field(default_factory=dict)  # what the selector reports
    evaluations: int = 0  # optimiser evaluations after adding ``selected``


def run_adapt(
    system: AdaptSystem,
    select: Callable[[int, np.ndarray, np.ndarray], tuple[int, dict]] | None = None,
    *,
    tolerance: float = CHEMICAL_ACCURACY,
    max_iterations: int = 40,
    gradient_tolerance: float = 1e-6,
) -> list[AdaptStep]:
    """Grow the ansatz until the energy is within ``tolerance`` of FCI.

    The run also stops when every ``|g_i|`` is below ``gradient_tolerance`` (the
    Part I non-zero threshold), the usual ADAPT convergence test.

    ``select(k, psi, exact_gradients)`` returns the generator to add at step ``k``
    and a dict of diagnostics; the default is exact ADAPT (largest ``|g_i|``, ties
    to the lowest index).  A measured selector may ignore ``exact_gradients``;
    they are passed only so the caller can score the choice.
    """
    ops: list[int] = []
    params = np.zeros(0)
    steps: list[AdaptStep] = []
    for k in range(max_iterations + 1):
        psi = system.state(ops, params)
        energy = system.energy(psi)
        g = system.gradients(psi)
        step = AdaptStep(k, list(ops), params.copy(), energy, energy - system.fci_energy, g)
        steps.append(step)
        if step.error < tolerance or k == max_iterations:
            break
        if np.abs(g).max() < gradient_tolerance:
            # No generator can lower the energy to first order: ADAPT has stalled
            # (on stretched H4 at 3.2 mHa, every gradient is zero by symmetry).
            break
        if select is None:
            chosen, info = int(np.argmax(np.abs(g))), {}
        else:
            chosen, info = select(k, psi, g)
        step.selected, step.selection = int(chosen), info
        ops.append(int(chosen))
        params, _ = system.optimise(ops, np.append(params, 0.0))
        step.evaluations = getattr(system, "last_evaluations", 0)
    return steps


def _gradient_problem(spec, ham, pool, validation):
    """Part I's :func:`gradients.build_gradient_problem` on a given Hamiltonian."""
    from gradients import GradientProblem, _real_terms, commutator  # Part I
    from pauli_fc import openfermion_qubitop_to_dict
    from pauli_ops import PauliEvaluator
    from states import expectation_without_matrix, hartree_fock_state

    state = hartree_fock_state(ham.n_qubits, ham.n_electrons)
    evaluator = PauliEvaluator(state)
    terms = [
        _real_terms(openfermion_qubitop_to_dict(commutator(ham.operator, g.qubit_operator), ham.n_qubits))
        for g in pool
    ]
    problem = GradientProblem(
        case_id=spec.case_id,
        state_name="HF",
        n_qubits=ham.n_qubits,
        state_energy=expectation_without_matrix(ham, state),
        labels=[g.label for g in pool],
        kinds=[g.kind for g in pool],
        gradients=np.array([evaluator.fragment_mean_std(t)[0] for t in terms]),
        commutator_terms=terms,
        evaluator=evaluator,
        metadata={
            "n_electrons": ham.n_electrons,
            "n_spatial_orbitals": ham.n_spatial_orbitals,
            "pool_size": len(pool),
            "hamiltonian_pauli_terms": ham.n_pauli_terms,
            "rhf_energy_hartree": ham.rhf_energy,
            "validation": validation,
            "hamiltonian_terms": openfermion_qubitop_to_dict(ham.operator, ham.n_qubits),
            **ham.metadata,
        },
    )
    problem.parent_fc_groups()
    return problem


def problem_cache_path(base_case: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / "trajectory" / f"{base_case}_problem.pickle"


def save_trajectory(system: AdaptSystem, steps: list[AdaptStep], path: Path) -> Path:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with problem_cache_path(system.base_case, path.parents[1]).open("wb") as handle:
        pickle.dump(system.problem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    n = len(steps)
    width = max(len(s.ops) for s in steps)
    ops = np.full((n, width), -1, dtype=np.int64)
    params = np.zeros((n, width))
    for row, s in enumerate(steps):
        ops[row, : len(s.ops)] = s.ops
        params[row, : len(s.ops)] = s.params
    np.savez_compressed(
        path,
        base_case=system.base_case,
        indices=system.indices,
        states=np.stack([system.state(s.ops, s.params) for s in steps]),
        ops=ops,
        params=params,
        energies=np.array([s.energy for s in steps]),
        fci_energy=system.fci_energy,
        gradients=np.stack([s.gradients for s in steps]),
        labels=np.array(system.labels),
    )
    return path


def load_trajectory(base_case: str, cache_dir: Path = DEFAULT_CACHE) -> dict:
    path = trajectory_path(base_case, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/phase2_adapt_states.py --cases {base_case} first"
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def problem_at(base_problem, psi_full: np.ndarray, case_id: str, *, energy: float,
               sector_gradients: np.ndarray | None = None, extra: dict | None = None):
    """The base case's gradient problem on another state of the same Hamiltonian.

    Commutator expansions, pool, labels and parent grouping are shared with
    ``base_problem``; only the state, its gradients and the cached fragment
    standard deviations change.  If the sector gradients are given, the two
    computations must agree to ``GRADIENT_AGREEMENT``.
    """
    from pauli_ops import PauliEvaluator  # Part I

    problem = copy.copy(base_problem)
    problem.evaluator = PauliEvaluator(np.asarray(psi_full, dtype=complex))
    problem.gradients = np.array(
        [problem.evaluator.fragment_mean_std(terms)[0] for terms in base_problem.commutator_terms]
    )
    if sector_gradients is not None:
        worst = float(np.abs(problem.gradients - sector_gradients).max())
        if worst > GRADIENT_AGREEMENT:
            raise ValueError(f"{case_id}: commutator and sector gradients differ by {worst:.2e}")
    problem.case_id = case_id
    problem.state_name = case_id.rsplit("_", 1)[-1]
    problem.state_energy = float(energy)
    problem._parent_sigmas = None
    problem.metadata = {**base_problem.metadata, "trajectory": extra or {}}
    return problem


def load_trajectory_problem(case_id: str, cache_dir: Path = DEFAULT_CACHE):
    """Build the gradient problem of a saved trajectory state, e.g. ``LiH_R3p0_ADAPT4``."""
    import pickle

    base_case, k = split_case(case_id)
    data = load_trajectory(base_case, cache_dir)
    if k >= data["states"].shape[0]:
        raise ValueError(f"{case_id}: the saved trajectory has only {data['states'].shape[0] - 1} steps")
    with problem_cache_path(base_case, cache_dir).open("rb") as handle:
        base = pickle.load(handle)
    psi = np.zeros(2 ** base.n_qubits, dtype=complex)
    psi[data["indices"]] = data["states"][k]
    ops = [int(o) for o in data["ops"][k] if o >= 0]
    return problem_at(base, psi, case_id, energy=float(data["energies"][k]),
                      sector_gradients=data["gradients"][k],
                      extra={"base_case": base_case, "iteration": k,
                             "ops": [base.labels[o] for o in ops],
                             "energy_error": float(data["energies"][k] - data["fci_energy"])})
