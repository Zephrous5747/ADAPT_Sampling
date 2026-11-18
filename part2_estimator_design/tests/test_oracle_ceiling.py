"""The oracle-ceiling script (``scripts/paper_a_oracle_ceiling.py``) on the two-qubit toy of the grouping notes, and its contrast rows."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_a_oracle_ceiling as oc  # noqa: E402
from contexts import build_context_library  # noqa: E402
from pauli_fc import greedy_fc_groups  # noqa: E402

# The toy of the grouping notes: g1 leads, g2 is the near-tied competitor (eps 0.05), g3 is far (eps 0.2).  Unit variances, uncorrelated Paulis.
TERMS = [
    {"XI": 0.5, "IZ": 1.0, "XX": 0.5},
    {"ZI": 0.2, "IZ": 0.8, "XX": 0.1},
    {"XI": -1.0, "YY": -0.5, "ZZ": -0.1},
]
EPS = {1: 0.05, 2: 0.2}


class UnitMoments:
    """Every Pauli has variance one and no two Paulis are correlated (the toy model)."""

    def covariance(self, alpha, paulis):
        return np.eye(len(paulis))


def toy_problem():
    support = sorted({p for t in TERMS for p in t})
    return SimpleNamespace(
        commutator_terms=TERMS, universal_support=support, n_qubits=2, n_generators=len(TERMS),
        parent_fc_groups=lambda: greedy_fc_groups(support, sort=True),
    )


def toy_rows(arms):
    rows = []
    for i in arms:
        d = {p: TERMS[0].get(p, 0.0) - TERMS[i].get(p, 0.0) for p in set(TERMS[0]) | set(TERMS[i])}
        rows.append({p: v / EPS[i] for p, v in d.items() if v != 0.0})
    return rows


def shots(groups, arms):
    problem = toy_problem()
    library = build_context_library(problem, "mass", groups=groups)
    return oc.optimal_shots(toy_rows(arms), library, UnitMoments(), "II-A")[0]


def groupings():
    support = toy_problem().universal_support
    mass = {p: sum(abs(t.get(p, 0.0)) for t in TERMS) for p in support}
    weight_g = greedy_fc_groups(support, sort=True)
    mass_g = greedy_fc_groups(sorted(mass, key=lambda p: (-mass[p], p)), sort=False)
    contrast_g, key = oc.contrast_key_groups(toy_rows([1, 2]), {})
    return weight_g, mass_g, contrast_g, key


def test_contrast_key_beats_weight_and_mass_and_matches_the_family_optimum():
    weight_g, mass_g, contrast_g, key = groupings()
    assert weight_g == [["XX", "YY", "ZZ"], ["IZ", "XI"], ["ZI"]]
    assert mass_g == [["IZ", "XI"], ["XX", "YY", "ZZ"], ["ZI"]]  # the same partition as by weight
    assert contrast_g == [["XI", "XX"], ["IZ", "ZI", "ZZ"], ["YY"]]  # XI and XX, the critical pair, share a context
    n = {name: shots(g, [1, 2]) for name, g in
         [("weight", weight_g), ("mass", mass_g), ("contrast", contrast_g), ("family", weight_g + mass_g + contrast_g)]}
    assert n["mass"] == pytest.approx(n["weight"], rel=1e-2)
    assert n["contrast"] < 0.8 * n["weight"]
    assert n["family"] == pytest.approx(n["contrast"], rel=1e-2)


def test_single_row_lower_bound_by_hand():
    """One contrast, no splitting: n = (sum over contexts of sqrt(sum of d^2 / eps^2 of its Paulis))^2, a bound the solver respects."""
    row = toy_rows([1])[0]
    w = {p: v * v for p, v in row.items()}  # XI 100, XX 64, IZ 16, ZI 16
    _, _, contrast_g, _ = groupings()
    by_hand = sum(np.sqrt(sum(w.get(p, 0.0) for p in g)) for g in contrast_g) ** 2
    assert by_hand == pytest.approx((np.sqrt(164.0) + np.sqrt(32.0)) ** 2)
    assert shots(contrast_g, [1]) <= by_hand * (1 + 1e-3)


def test_the_penalty_of_the_wrong_grouping_persists_when_the_near_tie_dominates():
    """The far arm is cheap in either grouping, so the late stage (g1 against g2 only) carries the same penalty as the early one."""
    weight_g, _, contrast_g, _ = groupings()
    early = shots(weight_g, [1, 2]) / shots(contrast_g, [1, 2])
    late = shots(weight_g, [1]) / shots(contrast_g, [1])
    assert early > 1.25 and late > 1.25
    assert late == pytest.approx(early, rel=0.1)


def test_the_contrast_rows_follow_the_stopping_test():
    g = np.array([0.5, -0.2, 0.1])
    terms = [{"ZZ": 1.0}, {"XX": 1.0, "ZZ": 0.5}, {"YY": 2.0}]
    z = 2.0
    rows, meta = oc.contrast_rows(terms, g, rho=0.1, z=z)
    assert meta[0] == (0, 0.5) and len(rows) == 3
    # leader row: A_l scaled so that its variance must be at most one, eps = |g_l| / z
    assert rows[0] == {"ZZ": pytest.approx(z / 0.5)}
    # competitor 1 (t = sign(g_1) = -1): value |g_l| - 0.9 |g_1| = 0.32; row = A_l + 0.9 A_1
    value = 0.5 - 0.9 * 0.2
    assert meta[1][1] == pytest.approx(value)
    assert rows[1]["XX"] == pytest.approx(0.9 * z / value) and rows[1]["ZZ"] == pytest.approx((1.0 + 0.9 * 0.5) * z / value)
    # at rho = 0 the margin is the plain gap and the coefficient of the competitor is one
    rows0, meta0 = oc.contrast_rows(terms, g, rho=0.0, z=z)
    assert meta0[1][1] == pytest.approx(0.3) and rows0[1]["XX"] == pytest.approx(z / 0.3)
    # a competitor inside the tolerance with no positive margin is skipped
    rows_t, meta_t = oc.contrast_rows(terms, np.array([0.5, 0.49, 0.1]), rho=0.0, z=z)
    assert len(rows_t) == 3  # 0.5 - 0.49 > 0 still certifiable
    rows_x, _ = oc.contrast_rows(terms, np.array([0.5, 0.5, 0.1]), rho=0.0, z=z)
    assert len(rows_x) == 2  # an exact tie has no margin to certify


def test_state_vector_moments_match_the_oracle_moments_of_the_library():
    from sampler import OracleMoments

    problem = toy_problem()
    library = build_context_library(problem, "mass")
    rng = np.random.default_rng(3)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    mean, cov = oc.moment_matrices(library, state)
    oracle = OracleMoments(library, state)
    for ctx in library.contexts:
        members = ctx.members
        assert np.allclose(mean[members], oracle.means(ctx.index, members), atol=1e-12)
        assert np.allclose(cov[np.ix_(members, members)], oracle.covariance(ctx.index, members), atol=1e-12)


def test_variance_aware_insertion_merges_only_where_the_covariance_helps():
    """Bell state: ZI and IZ always agree.  Row ZI - IZ cancels in one context (merge); ZI + IZ gains nothing (no merge)."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    problem = SimpleNamespace(commutator_terms=[{"ZI": 1.0, "IZ": 1.0}], universal_support=["IZ", "ZI"], n_qubits=2,
                              n_generators=1, parent_fc_groups=lambda: [["IZ", "ZI"]])
    library = build_context_library(problem, "mass")
    _, cov = oc.moment_matrices(library, bell)
    assert cov[library.labels.index("ZI"), library.labels.index("IZ")] == pytest.approx(1.0)
    assert oc.variance_aware_groups(library, [{"ZI": 1.0, "IZ": -1.0}], cov) == [["IZ", "ZI"]]
    assert sorted(map(sorted, oc.variance_aware_groups(library, [{"ZI": 1.0, "IZ": 1.0}], cov))) == [["IZ"], ["ZI"]]


def test_variance_aware_groups_form_a_commuting_partition():
    from itertools import combinations

    problem = toy_problem()
    library = build_context_library(problem, "mass")
    rng = np.random.default_rng(5)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    _, cov = oc.moment_matrices(library, state / np.linalg.norm(state))
    groups = oc.variance_aware_groups(library, toy_rows([1, 2]), cov)
    assert sorted(p for g in groups for p in g) == sorted(problem.universal_support)
    assert all(oc.commuting(library, library.n_required)[library.labels.index(a), library.labels.index(b)]
               for g in groups for a, b in combinations(g, 2))


def test_apply_vaware_hf_gives_a_commuting_partition_of_the_support():
    from itertools import combinations

    import part1_bridge as bridge

    try:
        problem = bridge.load_problem("H4_square_eq_side1p0_CISD")
    except Exception as error:  # the cached problem is not part of every checkout
        pytest.skip(f"problem not available: {error}")
    n = oc.apply_vaware_hf(problem, 0.0)
    groups = problem.parent_fc_groups()
    flat = [p for g in groups for p in g]
    assert n == len(groups) and sorted(flat) == sorted(problem.universal_support) and len(flat) == len(set(flat))

    def commute(a, b):
        return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2 == 0

    assert all(commute(a, b) for g in groups for a, b in combinations(g, 2))
    assert problem.fc_ordering == "vaware_hf"
