"""Classical priors for measurement design (note of 2026-10-19, "Classical priors for measurement design in ADAPT-VQE").

A compact classical approximation of the current ADAPT state supplies the covariances that
the design (coefficient splitting, shot allocation) needs before and while quantum outcomes
arrive.  No classical *expectation value* is ever used as a gradient observation.

Sparse state (Secs. 2.1-2.2 of the note)
    ``|c_m>`` is rebuilt from HF with all current amplitudes,
    ``|c_{m,k}> = T_K( exp(t_k G_k) |c_{m,k-1}> )``, and ``K = m + 1`` for ``m`` generators.
    ``T_K`` keeps the ``K`` largest ``|v_D|^2`` (ties: lowest determinant index), zeroes the rest and
    normalises.  This retains the largest weights of every intermediate sparse expansion; it does not
    find the exact top-``K`` determinants of the full state.  An excitation-minus-de-excitation
    generator satisfies ``G^3 = -G``, so ``exp(tG) = I + sin(t) G + (1 - cos t) G^2`` (checked, with
    a matrix-exponential fallback).

Full support (Eqs. 18-19)
    The design uses ``rho_0 = (1 - eps) |c><c| + eps I / 2^n``.  For distinct non-identity Paulis of a
    group, ``mu_0 = (1 - eps) mu_c`` and ``R_0 = (1 - eps) R_c + eps I``.  In terms of group means
    this is ``(1 - eps) <z>_c + eps * [z = 0]`` (the maximally mixed state has uniform outcomes), so
    :class:`MixedMoments` needs no new circuit evaluation.  The mixed state is analytic only; it is
    never prepared on the device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

import part1_bridge  # noqa: F401  (Part I's src on the path)
from part1_bridge import DEFAULT_CACHE
from sampler import OracleMoments
from trajectory import load_trajectory, split_case

WEIGHT_DIGITS = 12  # weights are rounded before ranking, so that ties are decided by the determinant index


def truncate(v: np.ndarray, keep: int) -> np.ndarray:
    """``T_K``: keep the ``keep`` largest ``|v_D|^2`` (ties to the lowest index), zero the rest, normalise."""
    v = np.asarray(v)
    if keep >= int(np.count_nonzero(v)):
        out = v.copy()
    else:
        weight = np.round(np.abs(v) ** 2, WEIGHT_DIGITS)
        order = np.lexsort((np.arange(v.size), -weight))[:keep]
        out = np.zeros_like(v)
        out[order] = v[order]
    norm = np.linalg.norm(out)
    if norm == 0.0:
        raise ValueError("truncation removed the whole state")
    return out / norm


def apply_rotation(generator: sp.spmatrix, t: float, v: np.ndarray, check: bool = True) -> np.ndarray:
    """``exp(t G) v`` through ``I + sin(t) G + (1 - cos t) G^2`` when ``G^3 = -G``, else a matrix exponential."""
    gv = generator @ v
    g2v = generator @ gv
    if check:
        g3v = generator @ g2v
        if np.abs(g3v + gv).max(initial=0.0) > 1e-9 * max(np.abs(gv).max(initial=0.0), 1.0):
            return expm_multiply(t * generator, v)
    return v + np.sin(t) * gv + (1.0 - np.cos(t)) * g2v


def sparse_state(generators, ops, params, hf: np.ndarray, keep: int | None = None) -> np.ndarray:
    """The coherent sparse approximation of ``exp(t_m G_m) ... exp(t_1 G_1) |HF>`` (sector basis).

    ``keep`` defaults to ``m + 1`` (the note's growth prescription).  Phases are retained.
    """
    ops = [int(o) for o in ops]
    keep = len(ops) + 1 if keep is None else int(keep)
    v = np.asarray(hf, dtype=float).astype(complex)
    for op, t in zip(ops, params):
        v = truncate(apply_rotation(generators[op], float(t), v), keep)
    return v


def exact_state(generators, ops, params, hf: np.ndarray) -> np.ndarray:
    v = np.asarray(hf, dtype=float).astype(complex)
    for op, t in zip(ops, params):
        v = expm_multiply(float(t) * generators[int(op)], v)
    return v


def sector_generators(base_case: str):
    """``(indices, generators, hf, labels)`` of a geometry in its reference sector, without the gradient problem."""
    from chemistry import build_qubit_hamiltonian  # Part I
    from pool import uccsd_pool
    from states import basis_index, sector_block, sector_indices

    from part1_bridge import get_case

    spec = get_case(base_case)
    ham = build_qubit_hamiltonian(spec)
    indices = np.asarray(sector_indices(ham.n_qubits, ham.n_alpha, ham.n_beta), dtype=np.int64)
    pool = uccsd_pool(ham.n_qubits, ham.n_electrons)
    generators = [sector_block(g.qubit_operator, ham.n_qubits, indices).tocsr().real.tocsr() for g in pool]
    hf = np.zeros(indices.size)
    hf[np.searchsorted(indices, basis_index(range(ham.n_electrons), ham.n_qubits))] = 1.0
    return indices, generators, hf, [g.label for g in pool], ham.n_qubits


def prior_cache_path(case_id: str, keep: int | None, cache_dir: Path = DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / "classical_prior" / f"{case_id}_K{'m1' if keep is None else keep}.npz"


def sparse_state_for_case(case_id: str, keep: int | None = None, cache_dir: Path = DEFAULT_CACHE):
    """``(full state vector, info)`` of the sparse prior of a saved trajectory state ``<geometry>_ADAPT<k>``.

    ``info`` records ``m``, ``K``, the overlap with the true state, the weight of the true state
    outside the retained determinants and how many determinants the true state has.  The result is
    cached (it needs the Hamiltonian and the pool, about a minute for H2O).
    """
    split = split_case(case_id)
    if split is None and case_id.endswith("_HF"):  # m = 0: K = 1, the determinant itself, which is the state
        from states import hartree_fock_state  # Part I

        problem = part1_bridge.load_problem(case_id)
        full = np.asarray(hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]), dtype=complex)
        return full, {"m": 0, "K": 1, "sector_dimension": -1, "true_determinants": 1, "overlap": 1.0, "true_weight_outside": 0.0}
    if split is None:
        raise ValueError(f"{case_id}: the sparse prior needs the ansatz of a saved ADAPT state (<geometry>_ADAPT<k>)")
    base_case, k = split
    path = prior_cache_path(case_id, keep, cache_dir)
    if path.exists():
        with np.load(path) as data:
            return data["full"], {key: data[key].item() for key in data.files if key != "full"}
    data = load_trajectory(base_case, cache_dir)
    row_ops = [int(o) for o in data["ops"][k] if o >= 0]
    row_params = np.asarray(data["params"][k])[: len(row_ops)]
    indices, generators, hf, _, n_qubits = sector_generators(base_case)
    if not np.array_equal(indices, data["indices"]):
        raise ValueError(f"{case_id}: sector basis differs from the saved trajectory")
    true = np.asarray(data["states"][k], dtype=complex)
    rebuilt = exact_state(generators, row_ops, row_params, hf)
    if np.abs(rebuilt - true).max() > 1e-6:
        raise ValueError(f"{case_id}: the pool does not reproduce the saved state (max error {np.abs(rebuilt - true).max():.2e})")
    c = sparse_state(generators, row_ops, row_params, hf, keep)
    retained = np.abs(c) > 0
    info = {"m": len(row_ops), "K": int(retained.sum()), "sector_dimension": int(indices.size),
            "true_determinants": int((np.abs(true) > 1e-12).sum()),
            "overlap": float(abs(np.vdot(c, true)) ** 2),
            "true_weight_outside": float(np.sum(np.abs(true[~retained]) ** 2))}
    full = np.zeros(2 ** n_qubits, dtype=complex)
    full[indices] = c
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp.npz")
    np.savez(scratch, full=full, **info)
    scratch.replace(path)
    return full, info


class MixedMoments(OracleMoments):
    """Outcome distributions and group means of ``(1 - eps) |c><c| + eps I / 2^n`` (Eq. 18), from those of ``|c>``."""

    def __init__(self, base: OracleMoments, eps: float) -> None:
        if not 0.0 <= eps < 1.0:
            raise ValueError("eps must lie in [0, 1)")
        self.base = base
        self.eps = float(eps)
        self.library = base.library
        self.state = base.state
        self._distributions: dict[int, np.ndarray] = {}
        self._means: dict[int, np.ndarray] = {}

    def distribution(self, alpha: int) -> np.ndarray:
        if alpha not in self._distributions:
            p = self.base.distribution(alpha)
            self._distributions[alpha] = (1.0 - self.eps) * p + self.eps / p.size
        return self._distributions[alpha]

    def group_means(self, alpha: int) -> np.ndarray:
        if alpha not in self._means:
            g = (1.0 - self.eps) * self.base.group_means(alpha)
            g[0] += self.eps  # the uniform distribution has mean 1 on the identity and 0 elsewhere
            self._means[alpha] = g
        return self._means[alpha]
