"""Larger-basis benchmark cases: H2O in cc-pVDZ restricted to a window of retained orbitals.

A referee asked for a case nearer to chemistry's standard benchmarks than STO-3G.  The colleagues'
note on larger-basis benchmarks (notebook ``larger_basis_full_report.ipynb``, modules ``windows.py`` and
``windows2.py``) defines, for H2O and N2, a *window* of retained orbitals: the core stays frozen, every
other occupied orbital is kept, and a fixed number of the lowest virtuals of each irreducible
representation is kept; at a later geometry each irrep is filled back to the same count with the
virtuals that overlap most with the previous window.  The Hamiltonian is the CASCI effective
Hamiltonian of the window (frozen-core one-body operator plus the window's two-electron integrals), so
the window FCI is the CASCI energy and the HF determinant of the window has exactly the RHF energy of
the full molecule.  The orbital-selection logic below is theirs, rewritten without the CASCI/CASSCF
model-space part (not needed for the measurement problem); the geometry (angle 104.45 deg) and the
windows are theirs, so the energies can be compared with their archive.

Cases (``<system>_dz<window>_<geometry>_<state>``):

* ``H2O_dz8o_{eq,str}_{HF,CISD}``: 8 orbitals (4 occupied + 2 a1 + 2 in-plane b virtuals), 8 electrons,
  16 qubits; R(OH) = 0.9584 and 1.75 A.
* ``H2O_dz11o_{eq,str}_{HF,CISD}``: 11 orbitals, 22 qubits (structure studies only: the shot simulator
  stores an outcome distribution of size 2^n per context).
* ``H2O_631g_{eq,str}_{HF,CISD}``: 6-31G, every non-core orbital (12), 24 qubits (structure only).

The registry is Part I's ``cases.CASES``; :func:`install` registers the cases and makes Part I's
``chemistry.build_qubit_hamiltonian`` / ``reference_fci_energy`` understand them (Part I's own files
are untouched).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from chemistry import CaseSpec

GEOMETRIES = {"eq": 0.9584, "str": 1.75}
ANGLE = 104.45
# virtual orbitals kept per irrep at the first geometry ("Bin" = in-plane b of H2O, "Bout" = out-of-plane b)
VIRT = {"8o": {"A1": 2, "Bin": 2}, "11o": {"A1": 3, "Bin": 3, "Bout": 1}, "14o": {"A1": 4, "Bin": 3, "Bout": 2, "A2": 1}}
NCORE = 1
BASIS = {"dz": "cc-pvdz", "631g": "6-31g"}


def water(r: float, angle: float = ANGLE) -> str:
    t = np.deg2rad(angle / 2)
    return f"O 0 0 0; H 0 {r * np.sin(t):.8f} {r * np.cos(t):.8f}; H 0 {-r * np.sin(t):.8f} {r * np.cos(t):.8f}"


@dataclass(frozen=True)
class WindowSpec(CaseSpec):
    """A case whose Hamiltonian is a window of orbitals of H2O (see the module docstring)."""

    window_key: str = "dz8o"       # <basis tag><window>, e.g. dz8o, dz11o, 631g (all non-core orbitals)
    geometry_tag: str = "eq"

    @property
    def geometry(self) -> str:  # part of the cache key: the window and the geometry must enter it
        return f"window {self.window_key} {self.geometry_tag} R={GEOMETRIES[self.geometry_tag]} angle={ANGLE}"


def _irreps(mol, mf):
    from pyscf import scf

    names = dict(zip(mol.irrep_id, mol.irrep_name))
    return [names[s] for s in scf.hf_symm.get_orbsym(mol, mf.mo_coeff)]


def _inplane_b(mol) -> str:
    """The B irrep that contains hydrogen s AOs (in-plane O-H bonding / antibonding)."""
    hs = [i for i, lab in enumerate(mol.ao_labels()) if lab.split()[1] == "H" and lab.split()[2].endswith("s")]
    for name, so in zip(mol.irrep_name, mol.symm_orb):
        if name.startswith("B") and np.abs(so[hs]).max() > 1e-8:
            return name
    raise RuntimeError("no in-plane B irrep")


def _pick(mol, mf, counts_total, ref=None, ref_sel=None):
    """Core + all occupied + virtuals per irrep (lowest, or by overlap with a reference window)."""
    from pyscf import gto

    irr = _irreps(mol, mf)
    nocc = mol.nelectron // 2
    occ = list(range(NCORE, nocc))
    left = dict(counts_total)
    for i in occ:
        left[irr[i]] = left.get(irr[i], 0) - 1
    if ref is not None:
        rmol, rmf = ref
        overlap = gto.intor_cross("int1e_ovlp", rmol, mol)
        score = ((rmf.mo_coeff[:, ref_sel].T @ overlap @ mf.mo_coeff) ** 2).sum(0)
    vir = []
    for name, k in left.items():
        assert k >= 0, (name, k)
        cand = [i for i in range(nocc, len(irr)) if irr[i] == name]
        if ref is not None:
            cand = sorted(cand, key=lambda i: -score[i])
        vir += cand[:k]
    return list(range(NCORE)), occ, sorted(vir), irr


def _rhf(mol, guess):
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    mf.init_guess = guess
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    return mf


@lru_cache(maxsize=None)
def window_problem(window_key: str, geometry_tag: str) -> dict:
    """The window Hamiltonian (``h1``, ``eri``, ``ecore``), the HF energy and the window FCI energy."""
    from pyscf import ao2mo, fci, gto, mcscf

    basis = BASIS["dz" if window_key.startswith("dz") else "631g"]
    window = window_key[2:] if window_key.startswith("dz") else "all"
    guess = "minao" if basis == "cc-pvdz" else "huckel"
    prev = None
    for tag in ("eq", "str"):
        r = GEOMETRIES[tag]
        mol = gto.M(atom=water(r), basis=basis, symmetry=True, unit="Angstrom", verbose=0)
        mf = _rhf(mol, guess)
        irr = _irreps(mol, mf)
        nocc = mol.nelectron // 2
        if window == "all":
            core, occ, vir = list(range(NCORE)), list(range(NCORE, nocc)), list(range(nocc, len(irr)))
        elif prev is None:
            b_in = _inplane_b(mol)
            b_out = "B1" if b_in == "B2" else "B2"
            counts = {}
            for i in range(NCORE, nocc):
                counts[irr[i]] = counts.get(irr[i], 0) + 1
            for name, k in VIRT[window].items():
                name = {"Bin": b_in, "Bout": b_out}.get(name, name)
                counts[name] = counts.get(name, 0) + k
            core, occ, vir, irr = _pick(mol, mf, counts)
        else:
            core, occ, vir, irr = _pick(mol, mf, counts, prev[:2], prev[2])
        win = occ + vir
        if tag == geometry_tag:
            norb, nelec = len(win), mol.nelectron - 2 * len(core)
            mc = mcscf.CASCI(mf, norb, nelec)
            order = list(core) + list(win) + [i for i in range(mf.mo_coeff.shape[1]) if i not in core and i not in win]
            mo = mf.mo_coeff[:, order]
            h1, ecore = mc.get_h1eff(mo)
            eri = ao2mo.restore(1, mc.get_h2eff(mo), norb)
            e_fci = float(fci.direct_spin1.kernel(h1, eri, norb, nelec, ecore=ecore, conv_tol=1e-12, max_cycle=300)[0])
            return dict(h1=h1, eri=eri, ecore=float(ecore), norb=norb, nelec=nelec, e_hf=float(mf.e_tot), e_fci=e_fci,
                        irreps=[irr[i] for i in win], basis=basis, nuclear=float(mol.energy_nuc()))
        prev = (mol, mf, win)
        if window != "all":
            counts = {}
            for i in win:
                counts[irr[i]] = counts.get(irr[i], 0) + 1
    raise KeyError(geometry_tag)


def build_window_hamiltonian(spec: WindowSpec):
    """Part I's ``QubitHamiltonian`` of a window (same Jordan-Wigner construction as ``chemistry``)."""
    from chemistry import QubitHamiltonian, _spin_orbital_tensors
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner

    w = window_problem(spec.window_key, spec.geometry_tag)
    one_body, two_body = _spin_orbital_tensors(w["h1"], w["eri"])
    operator = jordan_wigner(get_fermion_operator(InteractionOperator(w["ecore"], one_body, two_body)))
    operator.compress(abs_tol=1e-12)
    na = w["nelec"] // 2
    return QubitHamiltonian(
        operator=operator, n_qubits=2 * w["norb"], n_electrons=w["nelec"], n_alpha=na, n_beta=w["nelec"] - na,
        n_spatial_orbitals=w["norb"], rhf_energy=w["e_hf"], nuclear_repulsion=w["nuclear"],
        metadata={"basis": w["basis"], "mapping": "Jordan-Wigner", "frozen_core": True, "window": spec.window_key,
                  "irreps": w["irreps"]},
    )


def _specs():
    atoms = {tag: [("O", (0.0, 0.0, 0.0))] for tag in GEOMETRIES}  # geometry is rebuilt from the tag; atoms kept for CaseSpec
    out = []
    for window_key, label in (("dz8o", "H2O_dz8o"), ("dz11o", "H2O_dz11o"), ("631g", "H2O_631g")):
        for tag in GEOMETRIES:
            for state in ("HF", "CISD"):
                out.append(WindowSpec(f"{label}_{tag}_{state}", atoms[tag], state, BASIS["dz" if window_key.startswith("dz") else "631g"],
                                      window_key=window_key, geometry_tag=tag))
    return tuple(out)


WINDOW_CASES = _specs()


def install(cases: dict) -> None:
    """Register the window cases and teach Part I's chemistry module to build them (idempotent)."""
    import chemistry

    for spec in WINDOW_CASES:
        cases.setdefault(spec.case_id, spec)
    if getattr(chemistry, "_window_patched", False):
        return
    original_build, original_fci = chemistry.build_qubit_hamiltonian, chemistry.reference_fci_energy

    def build(spec):
        return build_window_hamiltonian(spec) if isinstance(spec, WindowSpec) else original_build(spec)

    def fci_energy(spec):
        return window_problem(spec.window_key, spec.geometry_tag)["e_fci"] if isinstance(spec, WindowSpec) else original_fci(spec)

    chemistry.build_qubit_hamiltonian = build
    chemistry.reference_fci_energy = fci_energy
    chemistry._window_patched = True


# ---------------------------------------------------------------------------------------------------------------------------
# Parallel problem build.  Part I's ``build_gradient_problem`` expands the commutators [H, G_i] one generator after the other
# (15 min at 16 qubits, hours at 22-24 qubits); the K expansions are independent, so they are distributed over processes here.
# The result is the same ``GradientProblem`` (same fields, same order) written under Part I's cache key.

_BUILD: dict = {}


def _expand_one(index: int):
    from gradients import _real_terms, commutator
    from pauli_fc import openfermion_qubitop_to_dict

    hamiltonian, generators, evaluator = _BUILD["ham"], _BUILD["gens"], _BUILD["evaluator"]
    operator = commutator(hamiltonian.operator, generators[index].qubit_operator)
    terms = _real_terms(openfermion_qubitop_to_dict(operator, hamiltonian.n_qubits))
    mean, _ = evaluator.fragment_mean_std(terms)
    return terms, mean


def build_problem_parallel(case_id: str, cache_dir, workers: int, validate: bool = True, log=print):
    """Build and cache the gradient problem of a case with ``workers`` processes; returns the problem."""
    import multiprocessing as mp
    import pickle
    import time
    from pathlib import Path

    from chemistry import build_qubit_hamiltonian, validate_hamiltonian
    from cases import get_case
    from gradients import GradientProblem
    from pauli_ops import PauliEvaluator
    from pool import uccsd_pool
    from problem_cache import cache_key
    from states import prepare_state

    spec = get_case(case_id)
    path = Path(cache_dir) / cache_key(spec)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    t0 = time.time()
    hamiltonian = build_qubit_hamiltonian(spec)
    # The sector diagonalisation of Part I's gate needs memory that grows with 2^n: above 18 qubits only the Hartree-Fock energy of the
    # qubit operator is checked against the RHF energy (the window construction is the one validated at 16 qubits, tests/test_cases_window.py).
    validation = (validate_hamiltonian(spec, hamiltonian, check_fci=hamiltonian.n_qubits <= 18) if validate else {"validated": False})
    state_energy, state = prepare_state(hamiltonian, spec.state)
    generators = uccsd_pool(hamiltonian.n_qubits, hamiltonian.n_electrons)
    evaluator = PauliEvaluator(state)
    log(f"{case_id}: Hamiltonian {hamiltonian.n_pauli_terms} terms, {len(generators)} generators, validated in {time.time() - t0:.0f} s")
    _BUILD.update(ham=hamiltonian, gens=generators, evaluator=evaluator)
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_expand_one, range(len(generators)), chunksize=1)
    log(f"{case_id}: commutators expanded in {time.time() - t0:.0f} s")
    problem = GradientProblem(
        case_id=spec.case_id, state_name=spec.state.upper(), n_qubits=hamiltonian.n_qubits, state_energy=state_energy,
        labels=[g.label for g in generators], kinds=[g.kind for g in generators],
        gradients=np.array([mean for _, mean in results]), commutator_terms=[terms for terms, _ in results], evaluator=evaluator,
        metadata={"n_electrons": hamiltonian.n_electrons, "n_spatial_orbitals": hamiltonian.n_spatial_orbitals,
                  "pool_size": len(generators), "hamiltonian_pauli_terms": hamiltonian.n_pauli_terms,
                  "rhf_energy_hartree": hamiltonian.rhf_energy, "validation": validation, **hamiltonian.metadata},
    )
    problem.parent_fc_groups()
    log(f"{case_id}: parent grouping done at {time.time() - t0:.0f} s ({len(problem.parent_fc_groups())} contexts)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(problem, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return problem
