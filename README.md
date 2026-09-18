# ADAPT_Sampling

Measurement-cost analysis for the generator-selection step of gradient-based
ADAPT-VQE.

At each ADAPT iteration the next generator is chosen by estimating

    g_i = <psi| [H, G_i] |psi>

for every candidate in the pool and taking the largest in magnitude. On hardware
those gradients have to be measured, and measurement is the dominant cost. This
project asks how much of that cost can be removed by two ideas: changing the
statistical target from estimating every gradient to identifying the winner, and
measuring gradients in shared fully commuting contexts so that one circuit
execution informs many candidates at once.

## The methods

The two ideas are independent, so they span a two-by-two:

|                    | per-gradient grouping | shared global grouping |
| ------------------ | --------------------- | ---------------------- |
| **no elimination** | baseline (no sharing) | **M1: FC-UG**          |
| **elimination**    | **M2: BAI-FC-IG**     | **M3: BAI-FC-UG**      |

M3 is the intended method: one global grouping of the union of all gradient
commutator supports, combined with best-arm-identification elimination of
candidates that can no longer be the largest.

## Layout

```
reports/                            Part I, II and III write-ups (LaTeX)
part1_full_reproducibility_package/ Part I implementation
  src/                              chemistry, groupings, shot models, the methods
  scripts/                          drivers, experiments, verification
  tests/                            unit tests
  runs/                             per-case results behind every table
  data/cached/                      historical pilot data from an earlier revision
```

## Reproducing Part I

```bash
python -m pip install -r part1_full_reproducibility_package/requirements.txt
cd part1_full_reproducibility_package

python scripts/run_part1_methods.py --all --out runs
python scripts/run_finite_shot.py --case H4_square_eq_side1p0_HF --trials 200 --out runs
python -m pytest tests -q
python scripts/verify_part1.py --runs runs
```

Every run passes a validation gate first: the Jordan-Wigner Hamiltonian must
reproduce the PySCF RHF energy on the Hartree-Fock determinant and the PySCF FCI
energy as its lowest eigenvalue in the target particle-number sector, both to
1e-8 Ha. A case that fails aborts rather than producing numbers. `verify_part1.py`
then checks the published results rather than the implementation: that every
reported total reconstructs from the file that explains it, that the allocations
meet their confidence targets, that M3 never exceeds M1, and that a geometric
radius schedule converges onto the schedule-free limit used in the tables.

Further experiments:

```bash
python scripts/experiment_grouping_routines.py --case LiH_R3p0_HF   # routine independence
python scripts/experiment_staged_regrouping.py --case H4_square_eq_side1p0_HF
```

## What the Part I results currently say

- **M3 is cheaper than M1 in every case**, by factors of 1.1 to 192. The ordering
  holds by construction and is robust to the grouping routine.
- **Elimination is the mechanism that matters.** Against the no-sharing baseline,
  elimination alone is worth 5 to 253 while shared grouping alone is worth 2.2 to
  3.4 on H4 and only 1.1 to 1.2 on H2O. The interaction term is below one in every
  case: the two mechanisms substitute for each other rather than compounding.
- **M2 versus M3 on H2O is unresolved.** The ratio ranges over 0.74 to 1.30
  depending on an arbitrary grouping convention, so no verdict is claimed there.
- All results are single-ADAPT-step diagnostics. Nothing here yet speaks to the
  cost of a complete ADAPT trajectory, which Part III names as the primary metric.

See `part1_full_reproducibility_package/report/` for the full write-up, including
the finite-shot study and the open items.

## Conventions

Fully commuting groupings are deterministic first-fit over a chosen insertion
order, and the order is a parameter rather than a convention, because group counts
are sensitive to it. Every run records its grouping conventions, interpreter and
package versions alongside its results.

Large derived artifacts under `runs/` (per-fragment standard deviations, per-arm
groupings, raw commutator expansions) are not tracked; they are regenerated
exactly by the commands above.
