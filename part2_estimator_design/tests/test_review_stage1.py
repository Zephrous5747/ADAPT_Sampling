"""Tests added after the Stage 1 review of the paper (2026-11-17, REMAINING_TASKS 0a-26):

* sampled outcome histograms converge to the exact context moments (H4);
* on a production LiH library the exact context means agree with an independent state-vector evaluator for random (context, member) pairs;
* a fixed seed reproduces stored records bit for bit (the records were produced on the cluster, Python 3.11.4 / NumPy 2.4.2; the tests run
  under the pinned Python 3.12.3 / NumPy 2.5.2).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402
from contexts import build_context_library  # noqa: E402
from sampler import OracleMoments, empirical_group_means, sample_counts  # noqa: E402


def test_sampled_means_converge_to_the_exact_moments(h4_cisd_problem):
    library = build_context_library(h4_cisd_problem, "canonical")
    oracle = OracleMoments(library, h4_cisd_problem.evaluator.state)
    shots = 4_000_000
    rng = np.random.default_rng(20261117)
    for alpha in (0, library.n_contexts // 2, library.n_contexts - 1):
        counts = sample_counts(rng, shots, oracle.distribution(alpha))
        assert counts.sum() == shots
        error = np.abs(empirical_group_means(counts) - oracle.group_means(alpha))
        # every group element is a +-1 variable, so its sample mean has a standard error of at most 1 / sqrt(shots)
        assert error.max() < 5.0 / np.sqrt(shots)


@pytest.fixture(scope="module")
def lih_selector():
    selector, _, _ = reframe_run.build("LiH_R1p6_ADAPT2", 1e10, need_prior=False)
    return selector


def test_lih_context_means_match_the_independent_evaluator(lih_selector):
    """Circuit synthesis, signs and bit masks of the production LiH library, against the state-vector Pauli evaluator."""
    library = lih_selector.library
    oracle = OracleMoments(library, lih_selector.problem.evaluator.state)
    rng = np.random.default_rng(7)
    for alpha in rng.choice(library.n_contexts, size=12, replace=False):
        context = library.contexts[int(alpha)]
        for member in rng.choice(context.members, size=4, replace=False):
            mean = float(oracle.means(int(alpha), np.array([member]))[0])
            assert mean == pytest.approx(lih_selector.problem.evaluator.expectation(library.labels[int(member)]), abs=1e-10)


STORED = [  # (method, setting, rho, trial) -> (selection_shots, selected_index) of the stored cluster records, state H4_a1_ADAPT3
    ("FC_SE", "rho010", 0.1, 0, 726016, 19),
    ("FC_SE", "rho010", 0.1, 1, 363008, 19),
    ("FC_SE_CS_DATA", "strict", 0.0, 0, 415616, 19),
    ("FC_SE_CS_DATA", "strict", 0.0, 2, 831232, 19),
]


def test_fixed_seed_reproduces_stored_cluster_records():
    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    selector, _, _ = reframe_run.build(reframe_run.case_of("H4_a1_ADAPT3", protocol), 1e10, need_prior=False)
    for method, setting, rho, trial, shots, index in STORED:
        task = reframe_run.task_list(protocol, "x", "H4_a1_ADAPT3", method, 1, trial, setting_id=setting, restart_ratio=0.0, batch_fix=True)[0]
        result = selector.run(np.random.default_rng(task["seed"]), method, rho, 0.05, "literal", restart_ratio=0.0, batch_fix=True)
        assert (result["selection_shots"], result["selected_index"]) == (shots, index)


@pytest.mark.parametrize("method", ["FC_SE", "FC_SE_CS_DATA"])
def test_exact_tie_runs_to_the_cap_at_rho_zero_and_is_certified_with_a_tolerance(method):
    """H4 at 1.0 A, ADAPT 2: the two largest gradients are exactly equal (0.26146134, next 0.21233172).  Exact identification cannot separate them,
    so rho = 0 runs to the cap (status unresolved) and its fallback is a member of the tied pair; rho = 0.1 certifies a member of the pair with far
    fewer shots (both certificates)."""
    cap = 2e8
    built, _, _ = reframe_run.build("H4_square_eq_side1p0_ADAPT2", cap, need_prior=False)
    g = np.abs(np.asarray(built.problem.abs_gradients))
    tied = set(np.flatnonzero(np.abs(g - g.max()) < 1e-9).tolist())
    assert len(tied) == 2
    kw = dict(restart_ratio=0.0, batch_fix=True)
    strict = built.run(np.random.default_rng(5), method, 0.0, 0.05, **kw)
    assert strict["status"] == "unresolved" and strict["cap_hit"] and strict["selection_shots"] >= 0.5 * cap
    assert strict["selected_index"] in tied
    for leader_out in (False, True):
        r = built.run(np.random.default_rng(5), method, 0.1, 0.05, leader_out=leader_out, **kw)
        assert r["status"] == "certified" and r["rho_good"] and r["selected_index"] in tied
        assert r["selection_shots"] < 0.1 * strict["selection_shots"]


def test_cisd_energy_equals_pyscf_cisd_on_h4():
    """Our CISD state (lowest eigenstate of the qubit Hamiltonian within the determinants at most doubly excited from HF, same spin numbers) against PySCF's RCISD
    on the same square H4 (1.0 A, STO-3G): the total energies agree to 1e-10."""
    pytest.importorskip("pyscf")
    from pyscf import ci, gto, scf

    problem = part1_bridge.load_problem("H4_square_eq_side1p0_CISD")
    atoms = [("H", (0, 0, 0)), ("H", (1.0, 0, 0)), ("H", (0, 1.0, 0)), ("H", (1.0, 1.0, 0))]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    solver = ci.CISD(mf)
    solver.conv_tol = 1e-12
    solver.kernel()
    assert problem.state_energy == pytest.approx(mf.e_tot + solver.e_corr, abs=1e-10)
