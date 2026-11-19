"""Symmetry check of the pool and of exact-gradient ADAPT paths (JCTC round 5, review item on N, S_z and S^2).

For each requested Hartree-Fock case it (i) builds the full-space operator of every pool generator and checks that it commutes with the number
operator N and with S_z and that the sector block used by the trajectory code equals the full operator restricted to the sector (no leakage out
of the sector), and (ii) runs exact-gradient ADAPT with the production code and reports the largest deviation of <N> and <S_z> from the
reference values and the range of <S^2> along the path.  The pool is spin-orbital, so S^2 is not conserved by construction; the values show
how far the states leave the singlet.

    python scripts/check_symmetries.py H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import part1_bridge  # noqa: F401,E402
from openfermion import get_sparse_operator, number_operator, s_squared_operator, sz_operator  # noqa: E402
from pool import uccsd_pool  # noqa: E402
from trajectory import AdaptSystem, run_adapt  # noqa: E402


def check(case: str, max_iterations: int) -> None:
    started = time.time()
    system = AdaptSystem.build(case)
    n_qubits, n_spatial = system.n_qubits, system.n_qubits // 2
    nop = get_sparse_operator(number_operator(n_qubits), n_qubits)
    szop = get_sparse_operator(sz_operator(n_spatial), n_qubits)
    s2op = get_sparse_operator(s_squared_operator(n_spatial), n_qubits)
    pool = uccsd_pool(n_qubits, system.n_electrons)
    from chemistry import build_qubit_hamiltonian  # Part I
    from part1_bridge import get_case
    from scipy.sparse.linalg import expm_multiply

    hfull = get_sparse_operator(build_qubit_hamiltonian(get_case(case)).operator, n_qubits).tocsr()
    fulls = []
    worst_n = worst_sz = worst_leak = 0.0
    for g, block in zip(pool, system.generators):
        full = get_sparse_operator(g.qubit_operator, n_qubits).tocsr()
        fulls.append(full)
        worst_n = max(worst_n, abs(full @ nop - nop @ full).max())
        worst_sz = max(worst_sz, abs(full @ szop - szop @ full).max())
        restricted = full[system.indices][:, system.indices]
        worst_leak = max(worst_leak, abs(restricted - block).max())
        outside = np.setdiff1d(np.arange(2 ** n_qubits), system.indices)
        worst_leak = max(worst_leak, abs(full[outside][:, system.indices]).max() if outside.size else 0.0)
    print(f"{case}: {len(pool)} generators; max |[G,N]| {worst_n:.1e}, max |[G,S_z]| {worst_sz:.1e}, max leakage out of the sector {worst_leak:.1e}")
    system.gtol = 1e-9
    steps = run_adapt(system, max_iterations=max_iterations)
    n_ref = system.n_electrons
    dn = dsz = 0.0
    s2 = []
    d_energy = d_gradient = 0.0
    for step in steps:
        psi = system.state(step.ops, step.params)
        full = system.full_state(psi)
        # independent recomputation in the full 2^n space: propagate with the full-space generators, then energy and every pool gradient
        # g_i = <psi|(H G_i - G_i H)|psi> from the full-space matrices (no sector restriction, no use of the production gradient formula)
        ref = system.full_state(system.hf).astype(complex)
        for op, theta in zip(step.ops, step.params):
            ref = expm_multiply(theta * fulls[op], ref)
        d_energy = max(d_energy, abs(float(np.real(ref.conj() @ (hfull @ ref))) - step.energy))
        h_ref = hfull @ ref
        grads = np.array([float(np.real(ref.conj() @ (hfull @ (g @ ref)) - ref.conj() @ (g @ h_ref))) for g in fulls])
        d_gradient = max(d_gradient, float(np.abs(grads - step.gradients).max()))
        dn = max(dn, abs(float(np.real(full.conj() @ (nop @ full))) - n_ref))
        dsz = max(dsz, abs(float(np.real(full.conj() @ (szop @ full)))))
        s2.append(float(np.real(full.conj() @ (s2op @ full))))
    print(f"  exact ADAPT, {len(steps)} states (final error {steps[-1].error * 1e3:.3f} mE_h): max |<N>-N| {dn:.1e}, max |<S_z>| {dsz:.1e}, "
          f"<S^2> from {min(s2):.1e} to {max(s2):.1e} (final {s2[-1]:.1e})")
    print(f"  full-space recomputation: max |E - E_sector| {d_energy:.1e} E_h, max |g_i - g_i,sector| over all generators and states {d_gradient:.1e}  [{time.time() - started:.0f} s]", flush=True)


if __name__ == "__main__":
    cases = sys.argv[1:] or ["H4_square_eq_side1p0_HF"]
    for c in cases:
        check(c, 40)
