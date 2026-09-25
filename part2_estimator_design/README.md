# Part II: universal measurable estimators — Steps 0–3

This package implements Steps 0–4 of the implementation plan for
`reports/part2_rewritten_universal_estimator_design.tex`. The write-up for
colleagues is `reports/part2_rewritten.tex` (and its PDF).

| Step | What | Where |
|---|---|---|
| 0 | Symplectic Pauli algebra, measurement-circuit synthesis, group moments, shot-level sampler, and their gates | `src/symplectic.py`, `src/clifford.py`, `src/sampler.py`, `scripts/step0_validate.py` |
| 1 | II-0 baseline as a shot-level online experiment (pairwise rule, top-up allocation, fine schedule) | `src/online.py`, `scripts/step1_online_baseline.py` |
| 2 | Overlap census: which Paulis are measured by more than one context | `src/contexts.py`, `scripts/step2_overlap_census.py` |
| 3 | Oracle ceiling of coefficient splitting (II-A), zero-sum ghosts (II-B) and per-breakpoint redesign (oracle II-D) | `src/design.py`, `src/allocation.py`, `src/planning.py`, `scripts/step3_oracle_ceiling.py`, `scripts/step3_prior_transfer.py` |
| 4 | II-A with covariances learned from the shots (shrinkage prior, cross-fitting, guard, estimated radii) | `src/learning.py`, `scripts/step4_learned_designs.py` |

Results and their reading are in `RESULTS.md`.

## Relation to Part I

Nothing chemical is re-derived. `src/part1_bridge.py` puts
`part1_full_reproducibility_package/src` on the path and builds each problem with
the current Part I pipeline (validation gate and SCF-instability fix included)
into this package's own `.cache/`. `scripts/build_problems.py` checks every build
against the gradients Part I committed: `|g_i|` agree to better than 3e-10 on all
nine cases. Three H4 rows differ in the *sign* of some gradients. That is an
orbital-phase convention, to which every Part I statistic is invariant.

Level II-0 is Part I's estimator on Part I's contexts, and it reproduces Part I:

- M1 exactly;
- M3's planning bound to Part I's own recomputation;
- M3 on actual precision exactly;
- the noiseless trial loop to 3e-5.

## Three ideas the implementation rests on

1. **A context measures a whole maximal group.** Rotating by a Clifford and
   reading out all qubits yields the eigenvalues of `2**n` commuting Paulis, not
   only the strings the context was built from. Which group that is depends on
   how the context's generators are completed to `n`. Part I's partition
   therefore has plenty of overlap once the measured groups are used
   (`contexts.py`). One Walsh–Hadamard transform of a context's outcome
   distribution gives every group mean. The covariances follow from
   `Cov(P_l, P_m) = s_l s_m (E[z_l ^ z_m] - E[z_l] E[z_m])`. Exact distributions
   give the oracle moments and sampled histograms give the empirical ones, from
   the same code (`sampler.py`).

2. **Coefficients are post-processing.** For a fixed context library, every
   linear unbiased estimator is `g_i = sum_alpha x_{i,alpha} . mu_alpha`, subject
   to `sum_alpha x_{i,alpha,l} = A_{il}`. That constraint is `A = BC`, with the
   fragments as the universal operators. The levels differ only in which
   coordinates may be non-zero (`design.py`).

3. **The joint design problem is solved in saddle form.** Alternating exact
   per-generator designs with the allocation (ICS) is monotone but stalls: on
   H4 side 1.0 HF it ends 9% above the single-arm optimum. The reason is that a
   context whose shots reach zero can never be refilled. `DesignSet.solve`
   eliminates the shots analytically and works on the result:
   - an inner smoothed L-BFGS over all active arms' coefficients, with a
     continuation in the smoothing;
   - an outer multiplicative dual ascent, Part I's own allocation rule.

   The reported cost is always the exact allocation of the returned design, so it
   is feasible even if the solver is not converged. It matches an independent
   single-arm solve to six digits and SLSQP on small joint instances.

## Running

The code needs NumPy ≥ 2, SciPy, and PySCF + OpenFermion for building problems.
It was run in WSL Ubuntu with Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
PySCF 2.14.0 and OpenFermion 1.8.1.

```bash
cd part2_estimator_design
python scripts/build_problems.py                      # all nine cases, checked against Part I
python -m pytest tests -q                             # 50 tests
python scripts/step0_validate.py --low-shot 0.001     # sampler gate
python scripts/step1_online_baseline.py --trials 250  # II-0 baseline, H4 + LiH
python scripts/step2_overlap_census.py                # overlap census, all nine cases
python scripts/step3_oracle_ceiling.py --cases H4_square_eq_side1p0_HF --redesign
python scripts/step4_learned_designs.py --trials 200  # learned II-A, H4 CISD rows
```

Every script writes CSV/JSON under `runs/<case>/`, with interpreter, package
versions and git commit in a `*_meta.json`. Step 3 saves the static design of
every level as the sparse pair `B`, `C` (`*_B.npz`, `*_C.npz`) and hard-fails if
`max |A - BC|` exceeds `1e-9 max |A|`.

## What is oracle, and what is not

- **Steps 2 and 3 are oracle throughout.** Exact covariances drive the designs,
  and exact gradients set the elimination trajectory. Their numbers are ceilings,
  not predictions.
- **Step 1 samples measurement outcomes.** Every elimination is decided from
  sampled data. The confidence radii still use oracle variances, as in Part I;
  learning them is Step 4.
- **The starting radius and the floor are oracle quantities.** They are taken
  from `max_i |g_i|`, as in Part I.
