#!/usr/bin/env python3
"""Oracle ceiling of the selection cost at a fixed state, with and without the rho tolerance.

With the exact covariances of the state, the exact gradients (so the leader and every gap are known) and the confidence factor
``z`` of the run, the stopping test of the method asks for one contrast per competitor ``i`` (``rules.rho_good_stop``):

    |g_l| - (1 - rho) |g_i|  >  z sd( s_l ghat_l - (1 - rho) t_i ghat_i ),      t_i = sign(g_i),

and for the sign of the leader, ``|g_l| > z sd(ghat_l)``.  A row ``d_i = s_l A_l - (1 - rho) t_i A_i`` scaled by the required
precision ``eps_i = (|g_l| - (1 - rho)|g_i|) / z`` is then a linear estimator whose variance must be at most one.  For a fixed list of
contexts, choosing the shots ``n_a`` and the splitting weights together is a convex programme
(``min sum n_a  s.t.  sum_a x_ia^T Sigma_a x_ia / n_a <= 1``; :class:`design.DesignSet`, saddle form), so its optimum is a lower bound
on the shots of *any* procedure that certifies these contrasts with the same ``z``.  The grouping enters only through the list of
contexts, and the script compares

``II-0``      each Pauli measured in its home context only (no splitting) in the default grouping,
``weight``    the default grouping (first fit by decreasing Pauli weight) with coefficient splitting,
``mass``      sorted insertion by the pooled coefficient mass ``sum_i |A_il|``,
``contrast``  sorted insertion by the sensitivity ``max_r d_rl^2 Var(P_l)`` to the scaled contrasts (needs the oracle),
``family``    the union of the contexts of the three groupings: a richer family, so the convex optimum can only fall.

``--regroup K`` adds the question of redoing the grouping late: the stage-1 design certifies every competitor except the ``K`` closest
ones and its shots are kept (they are valid samples of every Pauli of their context); stage 2 then adds the contrasts of the ``K``
closest competitors, either over the same contexts or over those plus the contexts of a grouping keyed to the stage-2 rows.  Only
the shots added in stage 2 are compared.

Read the output with the toy example of the notes (two near-tied gradients, a far third): the cost differences come from how many
contexts the critical Paulis are spread over (``spread``) and from how concentrated the optimal shots are (``n_eff``).

Run in the WSL venv (numpy-2 pickles), from ``part2_estimator_design``::

    python scripts/paper_a_oracle_ceiling.py --cases H4_square_eq_side1p0_ADAPT3 --rho 0 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401  (puts Part I on the path)
from contexts import build_context_library  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from part1_bridge import DEFAULT_DELTA, apply_grouping, confidence_z, load_problem  # noqa: E402
from pauli_fc import greedy_fc_groups  # noqa: E402  (Part I)
from sampler import OracleMoments  # noqa: E402

DESIGNS = ("II-0", "weight", "mass", "contrast", "family")
EXTRA = ("vaware", "vaware_hf", "weight+vaware")  # covariance-aware insertion, see variance_aware_groups


# --- the contrasts -------------------------------------------------------------------


def contrast_rows(terms, gradients, rho: float, z: float, both_signs: bool = False):
    """Scaled rows ``d / eps`` (variance at most one) and their metadata, one per certificate of the stopping test.

    ``terms[i]`` is ``{Pauli: A_il}``; returns ``(rows, meta)`` with ``meta[r] = (arm, value)``; the leader's own row has
    ``arm = leader`` and ``value = |g_l|``.  Competitors inside the tolerance with no positive margin are skipped.
    """
    g = np.asarray(gradients, dtype=float)
    lead = int(np.argmax(np.abs(g)))
    s = float(np.sign(g[lead]))
    c = 1.0 - rho
    rows = [{p: v * s * z / abs(g[lead]) for p, v in terms[lead].items()}]
    meta = [(lead, abs(g[lead]))]
    for i in range(len(terms)):
        if i == lead:
            continue
        ts = (1.0, -1.0) if both_signs or g[i] == 0.0 else (float(np.sign(g[i])),)
        for t in ts:
            value = s * g[lead] - c * t * g[i]
            if value <= 0.0:
                continue
            row = {p: s * v for p, v in terms[lead].items()}
            for p, v in terms[i].items():
                row[p] = row.get(p, 0.0) - c * t * v
            row = {p: v * z / value for p, v in row.items() if v != 0.0}
            if row:
                rows.append(row)
                meta.append((i, value))
    return rows, meta


# --- the grouping families -----------------------------------------------------------


def pauli_variances(library, oracle) -> dict:
    """``Var(P) = 1 - <P>^2`` of every library Pauli, read from the context that holds it."""
    out = {}
    for ctx in library.contexts:
        means = oracle.means(ctx.index, ctx.home_members)
        for label_index, m in zip(ctx.home_members, means):
            out[library.labels[int(label_index)]] = float(1.0 - m * m)
    return out


def contrast_key_groups(rows, variances):
    """Sorted insertion by the sensitivity ``max_r d_rl^2 Var(P_l)`` to the scaled rows (the oracle key)."""
    key: dict[str, float] = {}
    for row in rows:
        for p, v in row.items():
            key[p] = max(key.get(p, 0.0), v * v * variances.get(p, 1.0))
    order = sorted(key, key=lambda p: (-key[p], p))
    return greedy_fc_groups(order, sort=False), key


def moment_matrices(library, state):
    """``<P>`` and the covariance ``<PQ> - <P><Q>`` of every pair of library Paulis in ``state`` (only commuting pairs are meaningful).

    ``<PQ> = <P psi | Q psi>`` for Hermitian ``P``; the Paulis act on the state vector with the library's big-endian masks.
    """
    n, dim = library.n_qubits, 1 << library.n_qubits
    psi = np.asarray(state, dtype=complex)
    idx = np.arange(dim, dtype=np.int64)
    images = np.empty((library.n_library, dim), dtype=complex)
    for k in range(library.n_library):
        x, z = int(library.xmask[k]), int(library.zmask[k])
        sign = 1 - 2 * (np.bitwise_count(idx & z).astype(np.int64) & 1)
        images[k, idx ^ x] = (1j) ** bin(x & z).count("1") * sign * psi
    mean = np.real(images @ psi.conj())
    cov = np.real(images.conj() @ images.T) - np.outer(mean, mean)
    return mean, cov


def commuting(library, count: int):
    """Boolean ``(count, count)`` matrix: Paulis ``i`` and ``j`` of the library commute."""
    x, z = library.xmask[:count].astype(np.int64), library.zmask[:count].astype(np.int64)
    return (np.bitwise_count((x[:, None] & z[None, :]) ^ (z[:, None] & x[None, :])).astype(np.int64) & 1) == 0


def variance_aware_groups(library, rows, cov):
    """Sorted insertion that sees the covariances: each Pauli goes where it raises the estimated cost least.

    The Paulis of the library (the required ones) are inserted by decreasing sensitivity ``max_r d_rl^2 Var(P_l)`` to the scaled rows.
    A group ``g`` costs ``sqrt(V_g)`` with ``V_g = sum_r Var_g(d_r restricted to g)`` (the variance of every row's part measured in
    that context, cross-covariances included); the cost of the partition is ``(sum_g sqrt(V_g))^2``, the one-constraint optimum with
    equal multipliers.  A Pauli joins the commuting group with the smallest increase of ``sqrt(V_g)`` or opens a group of its own when
    that is cheaper; Paulis that no row involves go to the first commuting group.
    """
    count = library.n_required
    labels = library.labels[:count]
    index = {p: k for k, p in enumerate(labels)}
    D = np.zeros((len(rows), count))
    for r, row in enumerate(rows):
        for p, v in row.items():
            D[r, index[p]] = v
    C = cov[:count, :count]
    var = np.diag(C)
    key = (D ** 2).max(axis=0) * var
    order = sorted(range(count), key=lambda k: (-key[k], labels[k]))
    comm = commuting(library, count)
    members: list[list[int]] = []
    V: list[float] = []
    U: list[np.ndarray] = []  # U[g][r, p] = sum_{q in g} D[r, q] C[q, p]
    for k in order:
        d = D[:, k]
        alone = float((d ** 2).sum() * C[k, k])
        if not np.any(d):
            target = next((g for g, m in enumerate(members) if comm[k, m].all()), None)
            if target is None:
                members.append([k]); V.append(0.0); U.append(np.zeros_like(D))
            else:
                members[target].append(k)
            continue
        best, best_delta, best_V = None, np.sqrt(max(alone, 0.0)), alone
        for g, m in enumerate(members):
            if not comm[k, m].all():
                continue
            merged = V[g] + alone + 2.0 * float(d @ U[g][:, k])
            delta = np.sqrt(max(merged, 0.0)) - np.sqrt(max(V[g], 0.0))
            if delta < best_delta - 1e-12:
                best, best_delta, best_V = g, delta, merged
        if best is None:
            members.append([k]); V.append(alone); U.append(np.outer(d, C[k]))
        else:
            members[best].append(k); V[best] = best_V; U[best] += np.outer(d, C[k])
    return [[labels[k] for k in m] for m in members]


def apply_vaware_hf(problem, rho: float = 0.0, strategy: str = "mass", delta: float = DEFAULT_DELTA) -> int:
    """Replace the shared first-fit grouping of a loaded problem by the covariance-aware insertion with Hartree-Fock covariances.

    The contrast rows come from the exact gradients of the problem's state (the oracle's knowledge of the leader and the gaps) at the
    tolerance ``rho``; the covariances are those of the Hartree-Fock determinant, so the only oracle input is the rows.  The learned
    methods then run on this partition as they do on any other (``step5_external_baselines.py --grouping vaware_hf``).  The gradients
    of M2 keep their own default groups.  Returns the number of groups.
    """
    from states import hartree_fock_state  # Part I

    z = confidence_z(delta, problem.n_generators)
    rows, _ = contrast_rows(problem.commutator_terms, problem.gradients, rho, z)
    base = build_context_library(problem, strategy)
    _, cov = moment_matrices(base, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    groups = variance_aware_groups(base, rows, cov)
    problem._parent_groups = groups
    problem._parent_sigmas = None
    problem.fc_ordering = "vaware_hf"
    return len(groups)


def spread(groups, key, share: float = 0.8) -> int:
    """Number of groups that hold the Paulis carrying ``share`` of the total sensitivity (fewer is better)."""
    total = sum(key.values())
    where = {p: k for k, g in enumerate(groups) for p in g}
    acc, used = 0.0, set()
    for p in sorted(key, key=lambda q: -key[q]):
        if acc >= share * total:
            break
        acc += key[p]
        used.add(where[p])
    return len(used)


def complete_groups(problem, groups, rows):
    """Add singleton groups for Paulis of the rows that a partial grouping does not hold (stage-2 groupings)."""
    held = {p for g in groups for p in g}
    return [list(g) for g in groups] + [[p] for p in sorted({p for r in rows for p in r} - held)]


# --- the convex programme -------------------------------------------------------------


def optimal_shots(rows, library, oracle, level: str, lower=None):
    """Optimal total shots and per-context shots of the joint design of ``rows`` (every variance at most one)."""
    stub = SimpleNamespace(commutator_terms=rows)
    problems = build_fragment_problems(stub, library, oracle, level)
    solution = DesignSet(problems, library.n_contexts).solve(list(range(len(rows))), 1.0, lower)
    return float(solution.total), np.asarray(solution.shots, dtype=float)


def n_eff(shots) -> float:
    shots = np.asarray(shots, dtype=float)
    return float(shots.sum() ** 2 / np.maximum((shots ** 2).sum(), 1e-300))


def extra_groups(problem, rows, name: str, base_library, weight_groups):
    """Groupings by the covariance-aware insertion: ``vaware`` with the covariances of the state (the oracle), ``vaware_hf`` with those
    of the Hartree-Fock determinant (a classical prior, no oracle), ``weight+vaware`` the default partition and the ``vaware`` one."""
    if name == "vaware_hf":
        from states import hartree_fock_state  # Part I

        state = hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"])
    else:
        state = problem.evaluator.state
    _, cov = moment_matrices(base_library, state)
    groups = variance_aware_groups(base_library, rows, cov)
    return weight_groups + groups if name == "weight+vaware" else groups


def build_libraries(problem, rows, strategy: str, groupings=DESIGNS):
    """The libraries of the compared designs (and the Pauli key used for ``spread``)."""
    base = build_context_library(problem, strategy)
    oracle = OracleMoments(base, problem.evaluator.state)
    variances = pauli_variances(base, oracle)
    weight_groups = [list(g) for g in problem.parent_fc_groups()]
    apply_grouping(problem, "mass")
    mass_groups = [list(g) for g in problem.parent_fc_groups()]
    contrast_groups, key = contrast_key_groups(rows, variances)
    families = {"weight": weight_groups, "mass": mass_groups, "contrast": contrast_groups,
                "family": weight_groups + mass_groups + contrast_groups}
    libraries = {name: build_context_library(problem, strategy, groups=groups) for name, groups in families.items()}
    return libraries, families, key, weight_groups


def compare_designs(problem, rho: float, strategy: str, delta: float = DEFAULT_DELTA, both_signs: bool = False,
                    designs=DESIGNS) -> dict:
    z = confidence_z(delta, problem.n_generators)
    rows, meta = contrast_rows(problem.commutator_terms, problem.gradients, rho, z, both_signs)
    libraries, families, key, weight_groups = build_libraries(problem, rows, strategy)
    out = {"rho": rho, "z": z, "n_rows": len(rows), "leader": meta[0][0], "designs": {}}
    for name in designs:
        if name in EXTRA:
            groups = extra_groups(problem, rows, name, libraries["weight"], weight_groups)
            families[name] = groups
            libraries[name] = build_context_library(problem, strategy, groups=groups)
        library = libraries["weight" if name == "II-0" else name]
        oracle = OracleMoments(library, problem.evaluator.state)
        t0 = time.time()
        total, shots = optimal_shots(rows, library, oracle, "II-0" if name == "II-0" else "II-A")
        out["designs"][name] = {
            "shots": total, "contexts": library.n_contexts, "n_eff": n_eff(shots),
            "spread": spread(families.get(name, weight_groups), key) if name not in ("family", "weight+vaware") else None,
            "seconds": round(time.time() - t0, 1),
        }
    return out


def regroup_study(problem, rho: float, strategy: str, k: int, delta: float = DEFAULT_DELTA) -> dict:
    """Shots added in stage 2 (the ``k`` closest competitors) with the stage-1 shots kept: same contexts versus regrouped."""
    z = confidence_z(delta, problem.n_generators)
    rows, meta = contrast_rows(problem.commutator_terms, problem.gradients, rho, z)
    competitors = sorted({m[0] for m in meta[1:]}, key=lambda i: min(v for a, v in meta if a == i))
    late = set(competitors[:k])
    early_rows = [r for r, (a, _) in zip(rows, meta) if a not in late]
    late_rows = [r for r, (a, _) in zip(rows, meta) if a in late]

    base = build_context_library(problem, strategy)
    oracle = OracleMoments(base, problem.evaluator.state)
    variances = pauli_variances(base, oracle)
    weight_groups = [list(g) for g in problem.parent_fc_groups()]
    new_groups, _ = contrast_key_groups(late_rows, variances)
    lib_old = build_context_library(problem, strategy, groups=weight_groups)
    lib_new = build_context_library(problem, strategy, groups=weight_groups + new_groups)
    o_old, o_new = OracleMoments(lib_old, problem.evaluator.state), OracleMoments(lib_new, problem.evaluator.state)

    stage1, committed = optimal_shots(early_rows, lib_old, o_old, "II-A")
    c_old = committed
    c_new = np.concatenate([committed, np.zeros(lib_new.n_contexts - lib_old.n_contexts)])
    total_same, _ = optimal_shots(rows, lib_old, o_old, "II-A", c_old)
    total_regroup, _ = optimal_shots(rows, lib_new, o_new, "II-A", c_new)
    scratch, _ = optimal_shots(rows, lib_new, o_new, "II-A")
    return {
        "rho": rho, "late_arms": sorted(late), "stage1_shots": stage1,
        "stage2_added_same_contexts": total_same - stage1, "stage2_added_regrouped": total_regroup - stage1,
        "all_at_once_regrouped_family": scratch, "extra_contexts": lib_new.n_contexts - lib_old.n_contexts,
    }


# --- command line ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_ADAPT3"])
    parser.add_argument("--rho", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--strategy", default="mass", help="completion of the contexts to maximal groups (as step 4)")
    parser.add_argument("--both-signs", action="store_true", help="keep the non-binding sign t of unresolved competitors too")
    parser.add_argument("--regroup", type=int, default=0, metavar="K", help="also run the late re-grouping study with K late arms")
    parser.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=DESIGNS + EXTRA,
                        help="designs to solve (split a long state over several jobs; the outputs are merged afterwards)")
    parser.add_argument("--skip-compare", action="store_true", help="only the --regroup study")
    parser.add_argument("--resume", action="store_true", help="do nothing if --out already exists")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.resume and args.out is not None and args.out.exists():
        print(f"skip: {args.out} exists")
        return

    results = []
    for case in args.cases:
        for rho in args.rho:
            result = {"rho": rho, "designs": {}}
            if not args.skip_compare:
                result = compare_designs(load_problem(case), rho, args.strategy, both_signs=args.both_signs,
                                         designs=args.designs)
            result["case_id"] = case
            d = result["designs"]
            best = min((v["shots"] for v in d.values()), default=1.0)
            print(f"\n{case}  rho={rho:g}  rows={result.get('n_rows')}  z={result.get('z')}")
            print(f"  {'design':13s} {'shots':>14s} {'x ceiling':>9s} {'contexts':>8s} {'n_eff':>7s} {'spread':>6s}")
            for name in d:
                v = d[name]
                print(f"  {name:13s} {v['shots']:14,.0f} {v['shots'] / best:9.2f} {v['contexts']:8d} {v['n_eff']:7.1f} "
                      f"{'' if v['spread'] is None else v['spread']:>6}")
            if args.regroup:
                problem = load_problem(case)
                result["regroup"] = regroup_study(problem, rho, args.strategy, args.regroup)
                r = result["regroup"]
                print(f"  regroup K={args.regroup}: stage 1 {r['stage1_shots']:,.0f}; added in stage 2: same contexts "
                      f"{r['stage2_added_same_contexts']:,.0f}, regrouped (+{r['extra_contexts']} contexts) "
                      f"{r['stage2_added_regrouped']:,.0f}; all at once {r['all_at_once_regrouped_family']:,.0f}")
            results.append(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
