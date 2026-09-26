# Part II, Steps 0–4: results

All numbers are context-shots at family-wise δ = 0.05, with Bonferroni *z* over
the full pool as in Part I. The per-case CSVs are under `runs/<case>/`, and
`scripts/aggregate_results.py` rebuilds the combined tables. "Mass" and
"canonical" are completion rules for the measured groups (`src/contexts.py`).
Steps 1 and 4 use per-trial random streams: trial *i* uses
`SeedSequence(seed).spawn(n)[i]`, so results do not depend on `--workers` or
`--shard`. The full write-up is `reports/part2_rewritten.tex`.

## Summary

1. **Gates pass.**
   - The shot-level sampler matches the analytic covariance and Part I's
     Gaussian surrogate, down to 86 shots in total.
   - The online loop reproduces Part I's noiseless trial exactly.
   - With shots sampled as outcomes, it agrees with Part I's surrogate within
     1.5 s.e.m.
2. **The II-0 baseline (Step 1)** uses the pairwise rule, γ = 0.9 and top-up
   allocation. Its median cost is 62–83% below Part I's M3 as run. It made 5
   wrong selections in 1,100 trials, all on the tied H₄ HF rows.
3. **Overlap is abundant (Step 2).** 58–75% of 𝓑₀ is measured by two or more
   contexts, carrying 57–89% of the coefficient mass. II-A is a go.
4. **The oracle ceiling of II-A is large (Step 3).** It cuts M3 on actual precision
   by 38–54% on H₄, 70–76% on LiH and 64–80% on H₂O. II-B adds little.
   Per-breakpoint redesign helps depth-dominated rows, but only with top-up
   accounting.
5. **The HF covariance is harmful as a prior.** A design fitted to it and applied
   to the CISD state costs 66–69% more than II-0.
6. **Learned from the shots, II-A keeps 75–84% of its ceiling, fully non-oracle
   (Step 4).** It costs 36–43% less than II-0 on H₄ CISD ×2 and LiH, with 500/500
   correct selections and no cost tail. It needs three safeguards:
   - cross-fitting;
   - radii and allocation computed on the held-out fold;
   - at least 50 shots per fold before a context is learned.

## Step 0: shot-level sampler (`runs/step0_sampler_check.csv`, `runs/step0_lih/`)

The sampler was run at Part I's M1 allocation, with 2,000 replicates on H₄ and 400
on LiH.

| Case | Shots | max bias (σ) | max var. error | corr. error | KS p (winner / binding) |
|---|---|---|---|---|---|
| H₄ 1.0 HF | 2,244,893 | 2.56 | 0.066 | 0.060 | 0.79 / 0.21 |
| H₄ 1.0 CISD | 59,671 | 1.87 | 0.078 | 0.075 | 0.86 / 0.86 |
| H₄ 2.0 HF | 598,129 | 1.97 | 0.049 | 0.074 | 0.003 / 0.37 |
| H₄ 2.0 CISD | 58,094 | 1.82 | 0.109 | 0.065 | 0.20 / 0.35 |
| LiH | 3,564,398 | 3.53 | 0.140 | 0.190 | 0.32 / 0.52 |

The tolerances are 5√(2/R) on variances and 5/√R on correlations. The same checks
pass at 1/1000 of the allocation.

## Step 1: II-0 as a shot-level experiment (`*_step1_online_baseline.csv`)

The noiseless gate (`*_step1_noiseless_gate.csv`) passes with ratio 1.0000000 on
all nine case/rule pairs (real-valued shots). The table below uses the pairwise
rule, 250 trials on H₄ and 100 on LiH.

| Case | Part I M3 as run: mean / median | II-0 (γ = 0.9, top-up): mean ± s.e.m. | median | P90 | wrong |
|---|---|---|---|---|---|
| H₄ 1.0 HF | 1,315,987 / 921,024 | 385,766 ± 53,637 | 285,250 | 568,386 | 2 |
| H₄ 1.0 CISD | 150,437 / 163,910 | 41,515 ± 921 | 39,754 | 61,289 | 0 |
| H₄ 2.0 HF † | 543,563 / 627,922 | 1.62e6 ± 0.83e6 | 160,459 | 295,264 | 3 |
| H₄ 2.0 CISD | 102,348 / 128,284 | 23,827 ± 613 | 22,136 | 37,671 | 0 |
| LiH | 856,976 / 784,940 | 307,170 ± 6,238 | 299,479 | 395,948 | 0 |

† H₄ 2.0 HF has an exactly degenerate runner-up. A few trials never separate the
tie and run to the radius floor, so use the median. An earlier single-stream run
suggested top-up accounting removes this tail; the per-trial reruns do not
confirm it.

## Step 2: overlap census (`*_overlap_census.csv`)

| System | span: mult. / splittable / mass | mass completion: mult. / splittable / mass | winner / binding mass | CZ per context |
|---|---|---|---|---|
| H₄ | 1.71 / 52% / 49–53% | 1.95 / 60–62% / 62–65% | 61–71% / 61–87% | 8.3 |
| LiH | 2.57 / 67% / 75% | 2.95 / 76% / 84% | 88% / 89% | 23.3 |
| H₂O | 2.72 / 67% / 82–83% | 3.10 / 74% / 88% | 87–88% / 99% | 32.2 |

## Step 3: oracle ceiling (`*_step3_oracle_ceiling.csv`)

Changes are relative to Part I's II-0 on the same metric. "Redesign" is oracle
II-D with top-up accounting.

| Case | Level | M1 | M3 bound | **M3 actual** | M3 redesign |
|---|---|---|---|---|---|
| H₄ 1.0 HF | II-A / II-B mass | −68% / −70% | −37% / −39% | **−38% / −40%** | −57% / −58% |
| H₄ 1.0 CISD | II-A / II-B mass | −57% / −60% | −54% / −57% | **−54% / −56%** | −53% / −56% |
| H₄ 2.0 HF | II-A / II-B mass | −60% / −73% | −42% / −49% | **−40% / −50%** | −59% / −60% |
| H₄ 2.0 CISD | II-A / II-B mass | −52% / −60% | −44% / −51% | **−43% / −50%** | −50% / −57% |
| LiH | II-A mass / canonical | −84% / −80% | −81% / −77% | **−70% / −76%** | not run |
| H₂O eq HF / CISD | II-A mass | −84% / −79% | −81% / −78% | **−80% / −75%** | not run |
| H₂O str HF / CISD | II-A mass | −84% / −73% | −71% / −69% | **−64% / −68%** | not run |

**Prior transfer** (`runs/step3_prior_transfer.csv`). II-A designs were evaluated
on CISD. Fitted on CISD itself (oracle), they save 54% (side 1.0) and 43%
(side 2.0) on M3 actual. Fitted on the HF determinant, they cost 69% and 66%
*more* than II-0.

## Step 4: designs learned from the shots (`*_step4_learned_designs.csv`)

II-A with the mass completion, the pairwise rule, γ = 0.9, top-up allocation and
shot-level sampling. Trials: 200 on H₄, 100 on LiH. Rows marked "estimated" are
fully non-oracle. On LiH the truth is itself the HF state, so a flat prior
replaces the HF prior.

| Design | radii | H₄ 1.0 CISD | H₄ 2.0 CISD | LiH |
|---|---|---|---|---|
| II-0 | oracle | 42,313 | 25,563 | 305,794 |
| II-A, oracle covariance | oracle | 24,220 (−43%) | 11,516 (−55%) | 146,188 (−52%) |
| II-0 | estimated | 42,314 | 24,123 | 303,118 |
| II-A, HF prior only | estimated | 81,047 (+92%), 197 ok | 143,851 (+496%), 171 ok | — |
| II-A, HF ν = 1000 | estimated | 32,099 (−24%) | 31,696 (+31%), 195 ok | — |
| II-A, HF ν = 100 | estimated | 28,307 (−33%) | 15,293 (−37%) | — |
| II-A, flat ν = 100 | estimated | — | — | 193,042 (−36%) |
| **II-A, data only** | estimated | **27,129 (−36%)** | **13,727 (−43%)** | **184,511 (−39%)** |
| II-A, data only, no guard | estimated | 26,749 (−37%) | 13,764 (−43%) | 184,511 (−39%) |

All rows are 100% correct unless marked.

### Three failures of the first implementation

All three are fixed, and each is covered by a test or a diagnostic.

1. **Learning from too little data.** The design was fitted straight after the
   pilot round, on covariances from one or two samples per fold. On LiH only 54%
   of trials selected the winner, and 46 of 100 eliminated almost every arm in
   round 2. Fix: a context is learned only with ≥ 50 shots per fold.
2. **Radii from a covariance that includes the fitting fold.** Fix: held-out
   radii. A test checks estimated variance against the actual replicate spread.
3. **Planning and radii that disagreed.** A context whose 5-shot fold showed zero
   variance was never topped up, while the radii kept the other fold's variance
   there. On H₄ side 2.0 two trials ran to the radius floor at 7.8e8 shots each
   (median 1.3e4). Fix: plan with exactly the radii's variance.

## Reading and caveats

- Steps 2–3 are oracle ceilings, not predictions.
- The joint solver matches independent solvers to 10⁻⁴ on small problems. On the
  large problems it returns a feasible design whose exact allocation cost is
  reported, but its optimality is not certified.
- Step 4 refits per generator, not jointly, so its costs are upper bounds.
- The 50-shot threshold is untuned.
- Step 4 has not been run on H₂O yet (cluster). One H₂O eq CISD trial takes 329 s
  and 3.6 GB for II-0, and 685 s and 7.1 GB for II-A (data only). That puts a
  full case (5 configurations × 100 trials) at about 60–80 CPU-hours, and memory
  is the constraint.
