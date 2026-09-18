# Part I reproducibility package

This package contains scripts, configuration, cached outputs, and report files for the Part I fixed-state ADAPT-VQE gradient-measurement diagnostics.

## What this package reproduces exactly

The command

```bash
./run_cached_reproduction.sh
```

rebuilds the Part I report tables from the cached CSV/JSON outputs included in `data/cached/` and validates the completed M1/M2/M3 rows against the numbers in the report.

This is the exact reproducibility path for the data currently shown in the Part I report.

## What this package can regenerate from scratch

```bash
python -m pip install -r requirements.txt
python scripts/run_part1_methods.py --case H4_square_eq_side1p0_HF --out runs
python scripts/run_part1_methods.py --all-small --out runs
```

This regenerates everything from the geometry: molecular integrals, the
Jordan-Wigner Hamiltonian, the standard spin-orbital UCCSD generator pool,
commutator Pauli expansions, gradient values, deterministic fully commuting
groupings, oracle fragment variances, and the three method shot proxies
`M1_FCUG`, `M2_BAIFCIG` and `M3_BAIFCUG`.

### Validation gate

Every run first checks that the qubit Hamiltonian reproduces the PySCF RHF
energy on the Hartree-Fock determinant and the PySCF FCI energy as its lowest
eigenvalue in the target particle-number sector, both to 1e-8.  A run that fails
this gate aborts rather than producing numbers.

This gate exists because the previous version of the from-scratch script was
wrong in two independent ways, and silently: the two-electron integrals were
placed into the OpenFermion tensor in chemist rather than physicist ordering,
with the spin pattern `(a, b, a, b)` instead of `(a, b, b, a)` and no one-half
prefactor; and the computational basis was indexed little-endian while
OpenFermion is big-endian.  For H4 it produced a Hartree-Fock energy of
-5.21 Ha where RHF gives -1.76 Ha.  `scripts/run_part1_from_scratch.py` is now a
thin deprecated wrapper around the corrected pipeline.

### Method modules

Each method is one module with one `run(problem)` entry point, and each saves
the fully commuting groups and fragment standard deviations behind every number
it reports.

| Module | Method | Statistical target | Measurement contexts |
|---|---|---|---|
| `src/m1_fcug.py` | `M1_FCUG` | full gradient vector to a common confidence radius | one global FC grouping of `B_0 = union_i supp(C_i)` |
| `src/m2_baifcig.py` | `M2_BAIFCIG` | winner identification | an independent FC grouping of each `supp(C_i)` |
| `src/m3_baifcug.py` | `M3_BAIFCUG` | winner identification | M1's fixed parent contexts, restricted to the active support |

Supporting modules: `chemistry.py` (RHF and the qubit Hamiltonian, plus the
validation gate), `states.py` (HF, CISD and sector states in OpenFermion's qubit
ordering), `pool.py` (the UCCSD generator pool), `gradients.py` (commutators,
gradients and fragment standard deviations), `pauli_fc.py` (deterministic
first-fit FC grouping on symplectic bit masks), `pauli_ops.py` (Pauli action on
state vectors), `shot_models.py` (the shot proxy and the convex context-shot
allocation), `bai.py` (the shared successive-elimination schedule),
`cases.py` (the case registry) and `io_utils.py` (output files and conventions).

### Shot model

For one gradient split into independently measured FC fragments with oracle
standard deviations `sigma_a`, the optimal allocation needs
`(sum_a sigma_a / epsilon)**2` shots for a one-standard-error target `epsilon`.
For shared contexts, `shot_models.allocate_context_shots` solves the convex
programme

    minimise sum_a n_a   subject to   sum_a sigma_{i,a}**2 / n_a <= epsilon**2 for every active i

by a damped multiplicative fixed-point iteration on the Lagrange multipliers,
and then rescales the result so that the tightest constraint holds exactly.  The
returned allocation is therefore always feasible, and for a single gradient it
reduces to the closed form above.  Confidence radii use a normal model with a
Bonferroni family-wise `delta = 0.05` over the pool.

### Tests

```bash
python -m pytest tests -q
```

The suite pins the conventions that were previously wrong: the Hartree-Fock
index is checked to be big-endian, the Hamiltonian is checked against PySCF RHF
and FCI, and the top gradient is checked against a finite-difference energy
derivative `dE/dtheta`.  It also checks that the context-shot allocation
reproduces the closed form for a single gradient, that it is always feasible,
and that all three methods select the true winner.

## Directory layout

```text
configs/part1_cases.json                         Case definitions and global settings.
data/cached/lih_sto3g_stretched_pilot/           Cached LiH outputs.
data/cached/adapt_gradient_oracle_h4_h2o_pilot/  Cached H4/H2O outputs.
report/part1_fixed_gradient_measurement_report.tex
report/part1_fixed_gradient_measurement_report.pdf
scripts/rebuild_part1_tables_from_cached.py      Exact cached-table reproduction.
scripts/validate_cached_outputs.py               Validates report table numbers.
scripts/make_part1_report_from_tables.py         Builds a markdown table report.
scripts/run_part1_from_scratch.py                From-geometry regeneration workflow.
src/pauli_fc.py                                  Deterministic Pauli/FC grouping utilities.
src/shot_models.py                               Documented shot-proxy formulas.
reproduced_tables/                               Outputs produced by run_cached_reproduction.sh.
```

## Reproduction commands

### Cached exact reproduction

```bash
python -m pip install numpy pandas matplotlib
./run_cached_reproduction.sh
```

Expected output:

```text
VALIDATION PASSED: cached tables match expected Part I report values for completed M1/M2/M3 rows.
```

### From-scratch smoke test

```bash
python -m pip install -r requirements.txt
python scripts/run_part1_from_scratch.py --case LiH_R3p0_HF --out from_scratch_runs
```

### From-scratch full small set

```bash
python -m pip install -r requirements.txt
python scripts/run_part1_from_scratch.py --all-small --out from_scratch_runs
```

H2O can be slow because it has 14 qubits, 140 UCCSD generators, and large commutator supports.

## Fixed method names

- `M1_FCUG`: Fully Commuting Universal Groups.
- `M2_BAIFCIG`: BAI with Fully Commuting Individual Gradients.
- `M3_BAIFCUG`: BAI with Fully Commuting Universal Groups.

Do not use `naive/no-sharing` as a method name. It was a primitive diagnostic only.

## Required production upgrade

Before this becomes a complete production reproducibility package, every run should additionally save:

1. raw commutator Pauli expansions for every generator;
2. individual-gradient FC groups and fragment variances for M2;
3. global universal FC groups and variance/covariance records for M1/M3;
4. BAI elimination histories with confidence radii and active sets;
5. environment lockfile and code commit hash.

The current package already includes the scripts and filenames for these outputs, but the historical cached exploratory data do not contain all of them.
