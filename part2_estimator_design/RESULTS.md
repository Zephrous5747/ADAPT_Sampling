# Part II, Steps 0–3: results

All numbers are context-shots at family-wise δ = 0.05, with Bonferroni *z* over
the full pool as in Part I. The per-case CSVs are under `runs/<case>/`, and
`scripts/aggregate_results.py` rebuilds the combined tables. "Mass" and
"canonical" are completion rules for the measured groups (`src/contexts.py`).

## Summary

1. **Gates.** All three kinds of gate pass:
   - *Step 0.* The shot-level sampler is unbiased. Its variances and correlations
     match the analytic covariance, and its distribution is indistinguishable from
     Part I's Gaussian surrogate on all five small cases. This holds down to 1/1000
     of the M1 allocation (86 shots in total).
   - *Step 1, noiseless.* The online loop reproduces Part I's noiseless trial
     exactly (ratio 1.0000000 on nine case/rule pairs).
   - *Step 1, sampled.* With shots sampled as outcomes, it agrees with Part I's
     surrogate within 1.4 standard errors on every row.
2. **The II-0 baseline is already much cheaper than Part I's M3 as run.** Pairwise
   elimination, a γ = 0.9 schedule and top-up allocation cost 65–76% less than
   Part I's realised M3 (marginal rule, γ = 0.5), on shot-level sampling, with no
   selection errors in 1,100 trials.
3. **Overlap exists and is not the bottleneck (Step 2).** Of the Paulis in 𝓑₀,
   58–75% are measured by two or more contexts, carrying 57–89% of the coefficient
   mass. Most of it comes free from the parent groups' own spans; completion adds
   5–10 points. II-A is a go.
4. **The oracle ceiling is large (Step 3).** With exact covariances, coefficient
   splitting (II-A) cuts M3 eliminating on actual precision by:
   - 38–54% on the four H₄ rows;
   - 70–76% on LiH.

   Every level passes the 10% gate on every case computed.
5. **The ceiling is fragile to the covariance model.** A II-A design fitted to the
   Hartree–Fock covariances and applied to the CISD state is *66–69% more
   expensive than II-0* on both H₄ geometries; the oracle design saves 43–54%
   there. The HF prior is not a safe initializer on its own. Step 4's shrinkage
   toward measured covariances is the whole question.
6. **Ghosts (II-B) add little; redesign (oracle II-D) adds a lot where depth dominates.**
   - *II-B* adds 2–10 points over II-A on H₄, none on LiH, and is not monotone
     under the canonical completion.
   - *Per-breakpoint redesign* is compared here on the common-radius bound with
     top-up accounting. On H₄ side 1.0 HF it takes II-A from −38% to −57%, and on
     side 2.0 HF from −42% to −59%. Those are the two depth-dominated rows in
     Part I's breadth/depth table. On the CISD rows it adds 1–7 points.
   - Under Part I's per-breakpoint "maxima" accounting, redesign can be *worse*
     than a static design (H₄ side 1.0 CISD: 31,447 against 24,652). Redesign
     requires top-up accounting.
7. **Top-up accounting alone saves 0–8% at II-0** on the common-radius bound
   (8% on LiH and H₄ side 1.0 CISD). This resolves Part I's open item
   "allocate incrementally" without changing any estimator.

## Step 0: shot-level sampler (`runs/step0_sampler_check.csv`)

2,000 replicates on H₄ and 400 on LiH, at Part I's M1 allocation:

| Case | Shots | max bias (σ units) | max var. error | corr. error | KS p (winner / binding) |
|---|---|---|---|---|---|
| H₄ 1.0 HF | 2,244,893 | 2.56 | 0.066 | 0.060 | 0.79 / 0.21 |
| H₄ 1.0 CISD | 59,671 | 1.87 | 0.078 | 0.075 | 0.86 / 0.86 |
| H₄ 2.0 HF | 598,129 | 1.97 | 0.049 | 0.074 | 0.00 / 0.37 |
| H₄ 2.0 CISD | 58,094 | 1.82 | 0.109 | 0.065 | 0.20 / 0.35 |
| LiH | 3,564,398 | 3.53 | 0.140 | 0.190 | 0.32 / 0.52 |

The tolerances are 5/√R on correlations and 5√(2/R) on variances. The
H₄ 2.0 HF winner's small KS p-value (still above the 10⁻³ threshold) reflects the
lattice structure of estimates on a determinant, not bias or wrong variance. The
same checks pass at 1/1000 of the allocation (`--low-shot 0.001`).

## Step 1: II-0 as a shot-level online experiment

Noiseless gate (`*_step1_noiseless_gate.csv`): with real-valued shots the loop
equals Part I's `_sequential_trial` with zeroed noise to machine precision. With
integer shots it differs by at most 0.08% on H₄ and 1.8% on LiH. There, rounding
shots up makes one arm fall below its elimination threshold a round early. The
`noiseless` rows inside `*_step1_online_baseline.csv` come from the first run,
which compared integer shots only. `*_step1_noiseless_gate.csv` supersedes them.

The baseline (`*_step1_online_baseline.csv`) uses pairwise rule and
γ = 0.9, with 250 trials on H₄ and 100 on LiH; mean ± s.e.m. The two middle
columns are pairwise with maxima accounting. The Part I-style column is marginal,
maxima, γ = 0.5, shot-level sampled.

| Case | Part I-style M3 | γ = 0.5 | γ = 0.9 | **II-0 (γ = 0.9, top-up)** | median | P90 | correct |
|---|---|---|---|---|---|---|---|
| H₄ 1.0 HF | 1,377,896 | 670,291 | 410,172 | **337,162 ± 11,144** | 300,180 | 570,780 | 250/250 |
| H₄ 1.0 CISD | 147,834 | 92,132 | 40,994 | **40,876 ± 920** | 39,275 | 61,958 | 250/250 |
| H₄ 2.0 HF | 528,619 | 515,224† | 1,130,462† | **164,348 ± 6,118** | 152,312 | 297,895 | 250/250 |
| H₄ 2.0 CISD | 97,121 | 52,096 | 24,076 | **25,000 ± 662** | 23,030 | 38,735 | 250/250 |
| LiH | 872,391 | 570,427 | 312,640 | **303,322 ± 5,460** | 304,076 | 370,885 | 100/100 |

† H₄ 2.0 HF has an exactly degenerate runner-up (Part I, Sec. "Why the construction
matters"). Under maxima accounting a few trials never separate the tie and run to
the radius floor. Those two columns are therefore dominated by a heavy tail
(medians 514,906 and 160,433) and include one wrong selection each. Top-up
accounting removes the tail.

## Step 2: overlap census (`*_overlap_census.csv`)

| System | span: mult. / splittable / mass | mass completion: mult. / splittable / mass | winner / binding mass | CZ per context |
|---|---|---|---|---|
| H₄ (both geometries) | 1.71 / 52% / 49–53% | 1.95 / 60–62% / 62–65% | 61–71% / 61–87% | 8.3 |
| LiH | 2.57 / 67% / 75% | 2.95 / 76% / 84% | 88% / 89% | 23.3 |
| H₂O (both geometries) | 2.72 / 67% / 82–83% | 3.10 / 74% / 88% | 87–88% / 99% | 32.2 |

Here "mult." is mean multiplicity, "splittable" the share of 𝓑₀ Paulis with
multiplicity ≥ 2, and "mass" the share of Σ|A| they carry. The binding
generator (the one that sets M1's cost) is 89–99% splittable on LiH and H₂O. The
random completion lands between canonical and mass. Completion choice moves
these numbers by less than it moves the design costs of Step 3.

## Step 3: oracle ceiling (`*_step3_oracle_ceiling.csv`)

Gains are relative to Part I's II-0 on the same metric. "M3 actual" is M3
eliminating on actual precision, the primary metric. "Redesign" is oracle II-D
with top-up accounting, compared with II-0 under top-up.

| Case | Level | M1 | M3 bound | **M3 actual** | M3 redesign |
|---|---|---|---|---|---|
| H₄ 1.0 HF | II-A mass | −68% | −37% | **−38%** | −57% |
| | II-B mass | −70% | −39% | **−40%** | −58% |
| H₄ 1.0 CISD | II-A mass | −57% | −54% | **−54%** | −53% |
| | II-B mass | −60% | −57% | **−56%** | −56% |
| H₄ 2.0 HF | II-A mass | −60% | −42% | **−40%** | −59% |
| | II-B mass | −73% | −49% | **−50%** | −60% |
| H₄ 2.0 CISD | II-A mass | −52% | −44% | **−43%** | −50% |
| | II-B mass | −60% | −51% | **−50%** | −57% |
| LiH | II-A mass | −84% | −81% | **−70%** | not run |
| | II-A canonical | −80% | −77% | **−76%** | not run |
| | II-B mass | −84% | −82% | **−70%** | not run |
| | II-B canonical | −80% | −78% | **−64%** | not run |
| H₂O eq HF | II-A mass | −84% | −81% | **−80%** | not run |
| H₂O eq CISD | II-A mass | −79% | −78% | **−75%** | not run |
| H₂O str HF | II-A mass | −84% | −71% | **−64%** | not run |
| H₂O str CISD | II-A mass | −73% | −69% | **−68%** | not run |

With the canonical completion, H₄ gains on M3 actual are 0–9 points smaller than
with mass; on LiH, canonical is 5 points better (`*_step3_oracle_ceiling.csv`). Every design satisfies `A = BC` to within
floating-point error, and the static designs are saved as `B`/`C` pairs.

### Prior transfer (`runs/step3_prior_transfer.csv`)

II-A mass designs, all evaluated on the CISD state:

| Geometry | Design | M1 | M3 bound | M3 actual |
|---|---|---|---|---|
| H₄ side 1.0 | oracle (fitted on CISD) | −57% | −54% | −54% |
| | fitted on HF, applied to CISD | **+56%** | **+66%** | **+69%** |
| H₄ side 2.0 | oracle (fitted on CISD) | −52% | −44% | −43% |
| | fitted on HF, applied to CISD | **+43%** | **+66%** | **+66%** |

On a determinant, outcomes are either deterministic or perfectly
(anti)correlated. A design fitted to that structure concentrates coefficients
where the cancellations are exact, and they do not survive correlation. This is
the work order's failure mode (1), bad covariance prior, observed at planning
level before any measurement noise.

## Step 4: designs learned from the shots (`*_step4_learned_designs.csv`)

Setup: truth is the CISD state; the prior is the HF determinant. II-A uses the mass
completion, the pairwise rule, γ = 0.9, top-up allocation and shot-level sampling.
Each row has 200 trials at seed 2.

| Design | radii | H₄ 1.0 CISD: mean (change), correct | H₄ 2.0 CISD: mean (change), correct |
|---|---|---|---|
| II-0 | oracle | 41,215, 200/200 | 24,354, 200/200 |
| II-A oracle covariance | oracle | 23,450 (−43%), 200 | 12,125 (−50%), 200 |
| II-0 | estimated | 41,940, 200 | 24,591, 200 |
| II-A HF prior only | estimated | 439,749 (+949%), 170 | 197,780 (+704%), 138 |
| II-A HF, ν = 1000 | estimated | 28,360 (−32%), 199 | 23,233 (−6%), 193 |
| II-A HF, ν = 100 | estimated | 23,850 (−43%), 200 | 14,297 (−42%), 199 |
| **II-A data only (ν = 0)** | estimated | **24,688 (−41%), 200** | **16,045 (−35%), 200** |
| II-A data only, no guard | estimated | 25,253 (−40%), 200 | 32,265 (+31%), 197 |

What the table shows:

- **Learned from data, fully non-oracle, II-A keeps most of the ceiling.** It costs
  35–41% less than II-0 with no wrong selections; the oracle ceiling is 43–50%.
- **The HF prior alone fails.** Its design and its near-zero variances make the runs
  overconfident (69–85% correct). With oracle radii, the allocation still follows
  the prior's near-zero variances, and runs stall until the radius floor
  (10⁸–10⁹ shots).
- **Shrinkage helps only with little prior weight.**
- **The guard is needed.** It removes the tail on side 2.0.

## Reading and caveats

- **Steps 2–3 are oracle.** They bound what the levels can do; they do not
  predict it. The prior-transfer result shows the bound can turn into a loss.
- **The static design optimises the full-pool (M1-shape) objective.** The
  downstream M3 costs are therefore not guaranteed monotone in the level. That is
  why II-B canonical on LiH is worse than II-A on M3 actual while better on M1.
- **The joint solver is not exact.** It matches independent solvers to 10⁻⁴ on
  single-arm problems and on small joint problems. On the large problems it
  returns a feasible design whose exact allocation cost is reported. Its
  optimality there is not certified.
- **H₂O timing.** The H₂O static solves take hours each (81,566 library Paulis,
  2,356–2,366 contexts). Per-breakpoint redesign was run on H₄ only.
- **Step 1 radii use oracle variances**, as Part I's finite-shot study does.

## Implications for Step 4

- Start from II-A with top-up accounting and an empirical-covariance design
  refitted by cross-fitting.
- Test the prior alone, the empirical covariance alone, and their shrinkage.
  CISD truth with an HF prior is the case to beat, since the prior alone loses.
- Keep a guard that falls back to the II-0 coefficients whenever the design's
  predicted variance, under the current covariance estimate, is not lower.
- Leave II-B until II-A survives Step 4.
- Redesign per breakpoint is worth testing on the depth-dominated rows.
