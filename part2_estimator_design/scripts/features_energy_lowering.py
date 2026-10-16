#!/usr/bin/env python3
"""Feature study: how well the gradient |g_i| predicts the energy a generator actually gains.

For every state of the benchmark (the HF and CISD states of each geometry and every
state of the exact ADAPT trajectory) and every generator of the UCCSD pool we compute,
in exact arithmetic,

* ``g_i = <psi|[H, G_i]|psi>``,
* ``dE1_i``: the energy lowering of a one-parameter move ``min_theta E(e^{theta G_i} psi) - E(psi)``,
  exact because a fermionic excitation generator satisfies ``G^3 = -G``, so
  ``e^{theta G} = 1 + sin(theta) G + (1 - cos(theta)) G^2``;
* ``dEfull_i`` (trajectory states, optional): the energy after appending ``G_i`` to the
  current ansatz and re-optimising every parameter, minus the current energy.

Per state we report the rank correlation of |g| with the lowering, whether the largest
|g| is also the largest lowering, and the share of the best possible lowering that the
largest-|g| choice and the worst rho-good choice (|g_i| >= (1 - rho) max|g|) obtain.

Output: ``runs/features/energy_lowering_generators.csv`` (one row per state and generator)
and ``runs/features/energy_lowering_states.csv`` (one row per state).

    python scripts/features_energy_lowering.py --full-reopt H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from part1_bridge import get_case  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, run_adapt  # noqa: E402

GEOMETRIES = {
    "H4_square_eq_side1p0_HF": ("H4 1.0", "H4_square_eq_side1p0_CISD"),
    "H4_square_stretch_side2p0_HF": ("H4 2.0", "H4_square_stretch_side2p0_CISD"),
    "LiH_R3p0_HF": ("LiH 3.0", None),
    "H2O_eq_HF": ("H2O eq", "H2O_eq_CISD"),
    "H2O_stretch_HF": ("H2O str", "H2O_stretch_CISD"),
}


def light_system(base_case: str) -> AdaptSystem:
    """``AdaptSystem.build`` without the Pauli gradient problem (not needed here; costly for H2O)."""
    from chemistry import build_qubit_hamiltonian, validate_hamiltonian  # Part I
    from pool import uccsd_pool
    from states import basis_index, sector_block, sector_indices

    spec = get_case(base_case)
    ham = build_qubit_hamiltonian(spec)
    validate_hamiltonian(spec, ham)
    indices = np.asarray(sector_indices(ham.n_qubits, ham.n_alpha, ham.n_beta), dtype=np.int64)
    block = sector_block(ham.operator, ham.n_qubits, indices).real.tocsr()
    pool = uccsd_pool(ham.n_qubits, ham.n_electrons)
    generators = [sector_block(g.qubit_operator, ham.n_qubits, indices).real.tocsr() for g in pool]
    hf = np.zeros(indices.size)
    hf[np.searchsorted(indices, basis_index(range(ham.n_electrons), ham.n_qubits))] = 1.0
    fci = float(np.linalg.eigvalsh(block.toarray())[0])
    system = AdaptSystem(base_case, ham.n_qubits, ham.n_electrons, indices, block, generators,
                         [g.label for g in pool], hf, fci, {"rhf_energy": ham.rhf_energy})
    system._ham = ham
    return system


def cisd_vector(system: AdaptSystem, cisd_case: str) -> np.ndarray:
    from states import prepare_state  # Part I

    _, full = prepare_state(system._ham, "CISD")
    vec = np.real(full[system.indices])
    return vec / np.linalg.norm(vec)


def one_parameter_lowering(system: AdaptSystem, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``min_theta E(theta) - E(0)`` and its argmin for every generator (``G^3 = -G``)."""
    e0 = system.energy(psi)
    thetas = np.linspace(-np.pi, np.pi, 4001)
    s, c1 = np.sin(thetas), 1.0 - np.cos(thetas)
    lowering, angles = np.zeros(len(system.generators)), np.zeros(len(system.generators))
    for i, g in enumerate(system.generators):
        v1 = g @ psi
        v2 = g @ v1
        if np.linalg.norm(v1) < 1e-14:
            continue
        check = g @ v2 + v1
        if np.linalg.norm(check) > 1e-10:
            raise ValueError(f"generator {i} does not satisfy G^3 = -G on this state")
        V = np.stack([psi, v1, v2], axis=1)
        M = V.T @ (system.hamiltonian @ V)
        coeff = np.stack([np.ones_like(s), s, c1])  # 3 x n
        energies = np.einsum("an,ab,bn->n", coeff, M, coeff)
        norms = np.einsum("an,ab,bn->n", coeff, V.T @ V, coeff)
        energies = energies / norms
        j = int(np.argmin(energies))
        # refine around the grid minimum
        from scipy.optimize import minimize_scalar

        def f(t):
            cc = np.array([1.0, np.sin(t), 1.0 - np.cos(t)])
            return float(cc @ M @ cc) / float(cc @ (V.T @ V) @ cc)

        lo, hi = thetas[max(j - 1, 0)], thetas[min(j + 1, len(thetas) - 1)]
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        best = min(res.fun, energies[j])
        lowering[i] = min(best - e0, 0.0)
        angles[i] = res.x if res.fun <= energies[j] else thetas[j]
    return lowering, angles


def full_reopt_lowering(system: AdaptSystem, ops: list[int], params: np.ndarray, e0: float,
                        candidates: np.ndarray) -> np.ndarray:
    out = np.full(len(system.generators), np.nan)
    for i in candidates:
        _, energy = system.optimise(list(ops) + [int(i)], np.append(params, 0.0))
        out[i] = min(energy - e0, 0.0)
    return out


def summarise(label: str, kind: str, k: int | None, g: np.ndarray, de1: np.ndarray, defull: np.ndarray,
              rho: float, energy_error: float) -> dict:
    absg = np.abs(g)
    gmax = absg.max()
    row = {"system": label, "state": kind, "k": "" if k is None else k, "energy_error_mHa": 1e3 * energy_error,
           "max_abs_g": gmax}
    for name, de in (("1p", de1), ("full", defull)):
        if np.all(np.isnan(de)):
            for key in ("spearman", "best_g_is_best_dE", "share_best_g", "share_worst_rho_good", "best_dE_mHa",
                        "n_rho_good", "best_dE_in_rho_good"):
                row[f"{key}_{name}"] = ""
            continue
        mask = absg > 1e-8
        gain = -de  # positive lowering
        best_gain = np.nanmax(gain)
        top_g = np.flatnonzero(absg >= gmax - 1e-10)  # the largest |g| (ties included)
        rho_good = np.flatnonzero(absg >= (1 - rho) * gmax - 1e-12)
        row[f"spearman_{name}"] = float(spearmanr(absg[mask], gain[mask]).statistic) if mask.sum() > 2 else ""
        best_e = np.flatnonzero(gain >= best_gain - 1e-9)
        row[f"best_g_is_best_dE_{name}"] = bool(np.intersect1d(top_g, best_e).size > 0)
        row[f"share_best_g_{name}"] = float(np.nanmax(gain[top_g]) / best_gain) if best_gain > 0 else ""
        row[f"share_worst_rho_good_{name}"] = float(np.nanmin(gain[rho_good]) / best_gain) if best_gain > 0 else ""
        row[f"best_dE_mHa_{name}"] = 1e3 * best_gain
        row[f"n_rho_good_{name}"] = int(rho_good.size)
        row[f"best_dE_in_rho_good_{name}"] = bool(np.intersect1d(rho_good, best_e).size > 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(GEOMETRIES))
    parser.add_argument("--full-reopt", nargs="*", default=[],
                        help="geometries whose trajectory states also get the full re-optimisation lowering")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("runs/features"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gen_path, state_path = args.out / "energy_lowering_generators.csv", args.out / "energy_lowering_states.csv"
    gen_rows, state_rows = [], []
    for case in args.cases:
        label, cisd_case = GEOMETRIES[case]
        clock = time.perf_counter()
        system = light_system(case)
        states = [("HF", None, system.hf, [], np.zeros(0))]
        if cisd_case is not None:
            states.append(("CISD", None, cisd_vector(system, cisd_case), None, None))
        exact = run_adapt(system, tolerance=CHEMICAL_ACCURACY)
        for step in exact[1:-1] if len(exact) > 2 else []:
            psi = system.state(step.ops, step.params)
            states.append(("ADAPT", step.iteration, psi, step.ops, step.params))
        for kind, k, psi, ops, params in states:
            g = system.gradients(psi)
            e0 = system.energy(psi)
            de1, theta = one_parameter_lowering(system, psi)
            defull = np.full_like(de1, np.nan)
            if case in args.full_reopt and ops is not None:
                if len(ops) == 0:
                    defull = de1.copy()
                else:
                    defull = full_reopt_lowering(system, ops, params, e0, np.flatnonzero(np.abs(g) > 1e-8))
            for i in range(len(g)):
                gen_rows.append({"system": label, "state": kind, "k": "" if k is None else k, "generator": i,
                                 "label": system.labels[i], "abs_g": abs(g[i]), "dE1_mHa": 1e3 * de1[i],
                                 "theta1": theta[i], "dEfull_mHa": "" if np.isnan(defull[i]) else 1e3 * defull[i]})
            state_rows.append(summarise(label, kind, k, g, de1, defull, args.rho, e0 - system.fci_energy))
        print(f"{label}: {len(states)} states, {len(system.generators)} generators "
              f"({time.perf_counter() - clock:.0f}s)", flush=True)
        _write(gen_path, gen_rows)
        _write(state_path, state_rows)


def _write(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
