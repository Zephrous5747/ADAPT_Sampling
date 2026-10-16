"""Block-wise commuting contexts: depth capped by construction, from QWC to full commutation."""
from __future__ import annotations

import numpy as np
import pytest

from contexts import (
    block_commuting_groups,
    block_generators,
    build_context_library,
    contiguous_blocks,
    qwc_groups,
)
from pauli_fc import greedy_fc_groups  # Part I
from sampler import OracleMoments
from symplectic import anticommutes, masks_from_xz, xz_from_labels


def test_contiguous_blocks():
    assert contiguous_blocks(8, 2) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert contiguous_blocks(5, 2) == [[0, 1], [2, 3], [4]]


def test_single_qubit_blocks_are_qubit_wise_commutation(h4_cisd_problem):
    labels = h4_cisd_problem.universal_support
    assert block_commuting_groups(labels, contiguous_blocks(8, 1)) == qwc_groups(labels)


def test_one_block_of_every_qubit_is_full_commutation(h4_cisd_problem):
    labels = h4_cisd_problem.universal_support
    assert block_commuting_groups(labels, contiguous_blocks(8, 8)) == greedy_fc_groups(labels)


def test_blockwise_groups_commute_on_every_block_and_need_fewer_groups_as_blocks_grow(h4_cisd_problem):
    labels = h4_cisd_problem.universal_support
    counts = []
    for size in (1, 2, 4, 8):
        blocks = contiguous_blocks(8, size)
        groups = block_commuting_groups(labels, blocks)
        assert sorted(p for g in groups for p in g) == sorted(labels)
        for group in groups:
            x, z = masks_from_xz(*xz_from_labels(group))
            for block in blocks:
                mask = int(sum(1 << (7 - q) for q in block))
                for i in range(len(group)):
                    form = (x[i] & z & mask) ^ (z[i] & x & mask)
                    assert not (np.bitwise_count(form) & 1).any()
        counts.append(len(groups))
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


@pytest.mark.parametrize("size", [1, 2, 4])
def test_block_contexts_are_block_local_and_cap_the_two_qubit_gates(h4_cisd_problem, size):
    problem = h4_cisd_problem
    blocks = contiguous_blocks(problem.n_qubits, size)
    groups = block_commuting_groups(problem.universal_support, blocks)
    library = build_context_library(problem, "canonical", groups=groups, blocks=blocks)
    cap = len(blocks) * size * (size - 1) // 2
    assert library.two_qubit_counts().max() <= cap
    if size == 1:
        assert (library.two_qubit_counts() == 0).all()
    for context in library.contexts:
        assert np.isin(context.home_members, context.members).all()
        for gate in context.circuit.gates:
            if gate[0] == "CZ":
                assert any(gate[1] in b and gate[2] in b for b in blocks)  # never couples two blocks


def test_block_generators_measure_every_member_and_reject_wrong_use(h4_cisd_problem):
    problem = h4_cisd_problem
    blocks = contiguous_blocks(problem.n_qubits, 2)
    group = block_commuting_groups(problem.universal_support, blocks)[0]
    generators, rank = block_generators(group, blocks, problem.n_qubits)
    assert len(generators) == problem.n_qubits and rank <= problem.n_qubits
    with pytest.raises(ValueError):
        build_context_library(problem, "mass", groups=[group], blocks=blocks)


def test_block_library_reproduces_the_gradients(h4_cisd_problem):
    from design import build_fragment_problems

    problem = h4_cisd_problem
    blocks = contiguous_blocks(problem.n_qubits, 2)
    library = build_context_library(problem, "canonical",
                                    groups=block_commuting_groups(problem.universal_support, blocks), blocks=blocks)
    oracle = OracleMoments(library, problem.evaluator.state)
    for p in build_fragment_problems(problem, library, oracle, "II-0"):
        total = 0.0
        for alpha in np.unique(p.coord_ctx):
            chosen = p.coord_ctx == alpha
            total += p.x[chosen] @ oracle.means(int(alpha), p.coord_pauli[chosen])
        assert abs(total - problem.gradients[p.generator]) < 1e-9
