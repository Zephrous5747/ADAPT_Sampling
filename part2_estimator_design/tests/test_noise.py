"""Noisy measurement circuits: depolarising error after every CZ and readout error, against a density-matrix simulation."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from clifford import synthesise_measurement_circuit
from noise import NoisyMoments, damping, noisy_distribution
from sampler import OracleMoments
from symplectic import masks_from_xz, pack, xz_from_labels

N = 3
PAULIS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.array([[1, 0], [0, -1]])}


def kron(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def single(op, q, n=N):
    return kron([op if k == q else PAULIS["I"] for k in range(n)])


def two_qubit_unitary(gate, n=N):
    kind = gate[0]
    if kind == "H":
        return single(np.array([[1, 1], [1, -1]]) / np.sqrt(2), gate[1], n)
    if kind == "S":
        return single(np.diag([1, 1j]), gate[1], n)
    dim = 2 ** n
    cz = np.eye(dim, dtype=complex)
    for b in range(dim):
        bits = [(b >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[gate[1]] and bits[gate[2]]:
            cz[b, b] = -1
    return cz


def depolarise(rho, a, b, p, n=N):
    out = (1 - p) * rho
    for pa, pb in itertools.product("IXYZ", repeat=2):
        if pa == pb == "I":
            continue
        ops = [PAULIS["I"]] * n
        ops = [PAULIS[pa] if q == a else PAULIS[pb] if q == b else ops[q] for q in range(n)]
        e = kron(ops)
        out = out + p / 15.0 * (e @ rho @ e.conj().T)
    return out


def simulate(circuit, state, p2, pr, n=N):
    rho = np.outer(state, state.conj())
    for gate in circuit.gates:
        u = two_qubit_unitary(gate, n)
        rho = u @ rho @ u.conj().T
        if gate[0] == "CZ":
            rho = depolarise(rho, gate[1], gate[2], p2, n)
    p = np.real(np.diag(rho)).copy()
    for q in range(n):  # readout flips
        p = p.reshape((2,) * n)
        flipped = np.flip(p, axis=q)
        p = ((1 - pr) * p + pr * flipped).reshape(-1)
    return p


@pytest.fixture(scope="module")
def circuit():
    labels = ["XXX", "ZZI", "IZZ"]
    packed = pack(*masks_from_xz(*xz_from_labels(labels)), N)
    return synthesise_measurement_circuit([int(v) for v in packed], N)


def state(seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    return v / np.linalg.norm(v)


def test_the_circuit_has_entangling_gates(circuit):
    assert circuit.two_qubit_count >= 2


@pytest.mark.parametrize("p2, pr", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.03), (0.04, 0.02)])
def test_damped_distribution_matches_a_density_matrix_simulation(circuit, p2, pr):
    psi = state()
    ideal = circuit.outcome_distribution(psi)
    formula = noisy_distribution(ideal, damping(circuit, p2, pr))
    assert np.allclose(formula, simulate(circuit, psi, p2, pr), atol=1e-12)


def test_no_entangling_gates_means_only_readout_damping():
    labels = ["XXI", "IIZ", "IXI"]
    packed = pack(*masks_from_xz(*xz_from_labels(labels)), N)
    qwc = synthesise_measurement_circuit([int(v) for v in packed], N)
    assert qwc.two_qubit_count == 0
    d = damping(qwc, 0.2, 0.0)
    assert np.allclose(d, 1.0)
    d = damping(qwc, 0.2, 0.1)
    weights = np.array([bin(z).count("1") for z in range(2 ** N)])
    assert np.allclose(d, 0.8 ** weights)


def test_noisy_moments_replace_the_distributions_only(h4_cisd_problem):
    from contexts import build_context_library, contiguous_blocks, block_commuting_groups

    problem = h4_cisd_problem
    blocks = contiguous_blocks(problem.n_qubits, 2)
    library = build_context_library(problem, "canonical",
                                    groups=block_commuting_groups(problem.universal_support, blocks), blocks=blocks)
    ideal = OracleMoments(library, problem.evaluator.state)
    clean = NoisyMoments(library, problem.evaluator.state, 0.0, 0.0)
    noisy = NoisyMoments(library, problem.evaluator.state, 0.01, 0.005)
    for alpha in (0, 5, 17):
        assert np.allclose(clean.distribution(alpha), ideal.distribution(alpha), atol=1e-12)
        d = noisy.distribution(alpha)
        assert d.min() >= 0 and abs(d.sum() - 1) < 1e-12
        # noise pulls every non-trivial group mean towards zero
        a, b = ideal.group_means(alpha), noisy.group_means(alpha)
        assert (np.abs(b) <= np.abs(a) + 1e-12).all()
