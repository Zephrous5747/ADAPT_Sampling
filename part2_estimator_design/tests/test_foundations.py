"""Step 0 foundations: Pauli algebra, measurement circuits, moments and sampling.

Every check that matters is made against an independent route: dense matrices
built gate by gate, Part I's state-vector Pauli evaluator, and Part I's context
covariances.  Nothing here trusts the code under test to check itself.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from clifford import (
    MeasurementCircuit,
    complete_generators,
    independent_subset,
    synthesise_measurement_circuit,
)
from contexts import build_context_library, span_multiplicity
from sampler import (
    OracleMoments,
    empirical_covariance,
    pauli_covariance,
    sample_counts,
    walsh_hadamard,
)
from symplectic import (
    GF2Basis,
    anticommutes,
    dense_pauli,
    gf2_nullspace,
    hermitian_phase,
    labels_from_xz,
    masks_from_xz,
    pack,
    pauli_product,
    unpack,
    xz_from_labels,
    xz_from_masks,
)

# --- independent dense references ---------------------------------------------

H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
S1 = np.diag([1.0, 1j])


def _single(gate: np.ndarray, q: int, n: int) -> np.ndarray:
    matrix = np.array([[1.0 + 0j]])
    for position in range(n):
        matrix = np.kron(matrix, gate if position == q else np.eye(2))
    return matrix


def _cz(a: int, b: int, n: int) -> np.ndarray:
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for index in range(dim):
        if (index >> (n - 1 - a)) & 1 and (index >> (n - 1 - b)) & 1:
            diag[index] = -1
    return np.diag(diag)


def dense_circuit(circuit: MeasurementCircuit) -> np.ndarray:
    n = circuit.n_qubits
    unitary = np.eye(2 ** n, dtype=complex)
    for gate in circuit.gates:
        if gate[0] == "H":
            unitary = _single(H1, gate[1], n) @ unitary
        elif gate[0] == "S":
            unitary = _single(S1, gate[1], n) @ unitary
        else:
            unitary = _cz(gate[1], gate[2], n) @ unitary
    return unitary


def random_maximal_generators(n: int, rng: np.random.Generator) -> list[int]:
    candidates = rng.integers(1, 1 << (2 * n), size=64 * n, dtype=np.int64)
    generators, _ = complete_generators([], n, candidates)
    return generators


# --- Pauli algebra -------------------------------------------------------------


def test_label_round_trip_and_masks():
    labels = ["XYZI", "IIZZ", "YYYY", "ZIXI"]
    x, z = xz_from_labels(labels)
    assert labels_from_xz(x, z) == labels
    xm, zm = masks_from_xz(x, z)
    assert xm[0] == 0b1100 and zm[0] == 0b0110  # the Part I convention
    xb, zb = xz_from_masks(xm, zm, 4)
    assert np.array_equal(xb, x) and np.array_equal(zb, z)


def test_hermitian_phase_gives_the_pauli_string():
    x, z = xz_from_labels(["XYZI"])
    expected = np.kron(np.kron(np.kron([[0, 1], [1, 0]], [[0, -1j], [1j, 0]]), np.diag([1, -1])), np.eye(2))
    assert np.allclose(dense_pauli(x[0], z[0], hermitian_phase(x, z)[0]), expected)


def test_product_and_commutation_match_dense():
    rng = np.random.default_rng(3)
    n = 3
    for _ in range(40):
        x1, z1, x2, z2 = (rng.integers(0, 2, n).astype(bool) for _ in range(4))
        k1, k2 = rng.integers(0, 4, 2)
        x, z, k = pauli_product(x1, z1, k1, x2, z2, k2)
        assert np.allclose(dense_pauli(x, z, k), dense_pauli(x1, z1, k1) @ dense_pauli(x2, z2, k2))
        a = dense_pauli(x1, z1)
        b = dense_pauli(x2, z2)
        xm1, zm1 = masks_from_xz(x1, z1)
        xm2, zm2 = masks_from_xz(x2, z2)
        assert bool(anticommutes(xm1, zm1, xm2, zm2)[0]) == (not np.allclose(a @ b, b @ a))


def test_gf2_basis_and_nullspace():
    rng = np.random.default_rng(5)
    for _ in range(20):
        rows = [int(v) for v in rng.integers(0, 1 << 10, size=6)]
        basis = GF2Basis()
        for r in rows:
            basis.add(r)
        reduced = basis.reduce_many(np.array(rows, dtype=np.int64))
        assert (reduced == 0).all()
        null = gf2_nullspace(rows, 10)
        assert len(null) == 10 - basis.rank
        for v in null:
            assert all(bin(v & r).count("1") % 2 == 0 for r in rows)


# --- measurement circuits --------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(n, s) for n in (2, 3, 4, 5) for s in range(4)])
def test_circuit_diagonalises_the_whole_group(n, seed):
    rng = np.random.default_rng(100 * n + seed)
    generators = random_maximal_generators(n, rng)
    assert len(generators) == n
    circuit = synthesise_measurement_circuit(generators, n)
    unitary = dense_circuit(circuit)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(2 ** n))

    gx, gz = unpack(np.array(generators, dtype=np.int64), n)
    x, z = xz_from_masks(gx, gz, n)
    # every one of the 2**n group elements, as an unsigned Hermitian string
    for subset in itertools.product((0, 1), repeat=n):
        ex = np.zeros(n, dtype=bool)
        ez = np.zeros(n, dtype=bool)
        for bit, xg, zg in zip(subset, x, z):
            if bit:
                ex ^= xg
                ez ^= zg
        zmask, sign = circuit.diagonal_images(ex[None, :], ez[None, :])
        zx, zz = xz_from_masks(np.zeros(1, dtype=np.int64), zmask, n)
        image = unitary @ dense_pauli(ex, ez, hermitian_phase(ex, ez)) @ unitary.conj().T
        assert np.allclose(image, sign[0] * dense_pauli(zx[0], zz[0], 0))


def test_conjugation_phases_match_dense_for_arbitrary_paulis():
    rng = np.random.default_rng(11)
    n = 4
    circuit = synthesise_measurement_circuit(random_maximal_generators(n, rng), n)
    unitary = dense_circuit(circuit)
    x = rng.integers(0, 2, (30, n)).astype(bool)
    z = rng.integers(0, 2, (30, n)).astype(bool)
    k = rng.integers(0, 4, 30)
    xo, zo, ko = circuit.conjugate(x, z, k)
    for i in range(30):
        expected = unitary @ dense_pauli(x[i], z[i], k[i]) @ unitary.conj().T
        assert np.allclose(dense_pauli(xo[i], zo[i], ko[i]), expected)


def test_state_vector_application_matches_dense():
    rng = np.random.default_rng(13)
    n = 5
    circuit = synthesise_measurement_circuit(random_maximal_generators(n, rng), n)
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    state /= np.linalg.norm(state)
    assert np.allclose(circuit.apply(state), dense_circuit(circuit) @ state)


def test_completion_keeps_given_generators_and_is_maximal():
    n = 4
    x, z = xz_from_labels(["XXII", "ZZII"])
    given = [int(v) for v in pack(*masks_from_xz(x, z), n)]
    generators, _ = complete_generators(given, n)
    assert generators[:2] == given and len(generators) == n
    assert len(independent_subset(generators)) == n


# --- moments ---------------------------------------------------------------------


def test_walsh_hadamard_matches_definition():
    rng = np.random.default_rng(17)
    n = 4
    p = rng.random(2 ** n)
    transformed = walsh_hadamard(p)
    for zmask in range(2 ** n):
        expected = sum(p[b] * (-1) ** bin(zmask & b).count("1") for b in range(2 ** n))
        assert transformed[zmask] == pytest.approx(expected)


@pytest.fixture(scope="module")
def h4_library(h4_cisd_problem):
    return build_context_library(h4_cisd_problem, "canonical")


def test_every_context_measures_its_parent_members(h4_library, h4_cisd_problem):
    assert h4_library.n_contexts == len(h4_cisd_problem.parent_fc_groups())
    multiplicity = h4_library.multiplicity()
    assert (multiplicity[: h4_library.n_required] >= 1).all()
    for context in h4_library.contexts:
        assert np.isin(context.home_members, context.members).all()
        assert (h4_library.home[context.home_members] == context.index).all()


def test_group_means_match_part1_evaluator(h4_library, h4_cisd_problem):
    evaluator = h4_cisd_problem.evaluator
    moments = OracleMoments(h4_library, evaluator.state)
    for context in h4_library.contexts[::4]:
        means = moments.means(context.index, context.members)
        for member, mean in zip(context.members, means):
            assert mean == pytest.approx(evaluator.expectation(h4_library.labels[member]), abs=1e-12)


def test_pauli_covariance_matches_dense_products(h4_library, h4_cisd_problem):
    evaluator = h4_cisd_problem.evaluator
    moments = OracleMoments(h4_library, evaluator.state)
    context = h4_library.contexts[0]
    members = context.members[:12]
    covariance = moments.covariance(context.index, members)
    vectors = [evaluator.apply(h4_library.labels[m]) for m in members]
    means = [np.vdot(evaluator.state, v).real for v in vectors]
    for a, b in itertools.product(range(len(members)), repeat=2):
        expected = np.vdot(vectors[a], vectors[b]).real - means[a] * means[b]
        assert covariance[a, b] == pytest.approx(expected, abs=1e-12)


def test_fragment_covariances_match_part1(h4_library, h4_cisd_problem):
    """Generator-level covariance per context against Part I's Gram construction."""
    import finite_shot

    groups = h4_cisd_problem.parent_fc_groups()
    reference = finite_shot.context_covariances(h4_cisd_problem, groups)
    moments = OracleMoments(h4_library, h4_cisd_problem.evaluator.state)
    index = {label: i for i, label in enumerate(h4_library.labels)}
    for context in h4_library.contexts:
        members = context.home_members
        fragments = np.zeros((members.size, h4_cisd_problem.n_generators))
        position = {m: k for k, m in enumerate(members)}
        for i, terms in enumerate(h4_cisd_problem.commutator_terms):
            for label, coefficient in terms.items():
                ell = index[label]
                if ell in position:
                    fragments[position[ell], i] = coefficient
        covariance = fragments.T @ moments.covariance(context.index, members) @ fragments
        assert np.allclose(covariance, reference[context.index], atol=1e-12)


def test_multinomial_moments_match_the_oracle(h4_library, h4_cisd_problem):
    moments = OracleMoments(h4_library, h4_cisd_problem.evaluator.state)
    context = h4_library.contexts[1]
    members = context.members[:6]
    rng = np.random.default_rng(23)
    counts = sample_counts(rng, 400_000, moments.distribution(context.index))
    zmask = context.member_zmask[context.positions(members)]
    sign = context.member_sign[context.positions(members)]
    sampled = empirical_covariance(counts, zmask, sign)
    exact = moments.covariance(context.index, members)
    assert np.allclose(sampled, exact, atol=5e-3)
    assert np.allclose(pauli_covariance(walsh_hadamard(moments.distribution(context.index)), zmask, sign), exact)


def test_span_overlap_never_exceeds_completed_overlap(h4_library, h4_cisd_problem):
    span = span_multiplicity(h4_cisd_problem)
    completed = h4_library.multiplicity()[: h4_library.n_required]
    assert (span >= 1).all() and (completed >= span).all()
