# Part II, Steps 0–4 and Phases 2–4: results

All numbers are context-shots at family-wise δ = 0.05, with Bonferroni *z* over
the full pool as in Part I. The per-case CSVs are under `runs/<case>/`, and
`scripts/aggregate_results.py` rebuilds the combined tables. "Mass" and
"canonical" are completion rules for the measured groups (`src/contexts.py`).
Steps 1 and 4 and Phases 2–4 use per-trial random streams: trial *i* uses
`SeedSequence(seed).spawn(n)[i]`, so results do not depend on `--workers` or
`--shard`. The full write-up is `reports/part2_rewritten.tex`. Runs marked
"Trillium" were done on the cluster and synchronised with
`cluster/sync_from_trillium.sh`.

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
3. **Overlap is abundant (Step 2).** With the mass completion, 60–76% of 𝓑₀ is
   measured by two or more contexts, carrying 62–89% of the coefficient mass.
   II-A is a go.
4. **The oracle ceiling of II-A is large (Step 3).** It cuts M3 on actual precision
   by 38–54% on H₄, 73–76% on LiH and 70–80% on H₂O. II-B adds 2–10 points on H₄
   and nothing resolvable on LiH and H₂O. Per-breakpoint redesign adds 15–19
   points on depth-dominated rows (H₄ HF, H₂O stretched HF) and 1–3 points
   elsewhere on LiH and H₂O, but only with top-up accounting.
5. **The HF covariance is harmful as a prior.** A design fitted to it and applied
   to the CISD state costs 66–69% more than II-0.
6. **Learned from the shots, II-A keeps 73–75% of its ceiling on H₄ and LiH, and
   all of it on H₂O, fully non-oracle (Step 4).** It costs 35–44% less than II-0 on
   H₄ CISD ×2 and LiH, and 54–57% less on H₂O CISD ×2, with 700/700 correct
   selections and no cost tail. On ADAPT trajectory states it costs 30–58% less. It needs three safeguards:
   - cross-fitting;
   - radii and allocation computed on the held-out fold;
   - at least 50 shots per fold before a context is learned.
7. **Part I's plug-in sign rule fails on correlated ADAPT states.** On H₄ 2.0
   ADAPT2 it made 5–24% wrong selections in every configuration except II-A with
   oracle covariances. The sign-aware rule is 100% correct there. It costs +6 to
   +38% on H₄ CISD and +80% at II-0 on LiH.
8. **Direct contrast designs (II-E) pay on hard, small-gap states** (H₄ ADAPT5:
   −32% against arm-wise II-A, −52% against II-0). They are 11–39% dearer where
   the gap is large.
9. **The non-oracle bound start is cheap on most states but not yet safe on the
   hardest:** −35% to +11%, but 92% correct on H₂O eq CISD with the contrast design.
10. **Coverage.** Per-round Bonferroni misses in 4–29% of trials. The anytime
    bound removes every miss at 1.8–2.6× the cost where the variance estimates are
    sound, but not on LiH ADAPT5 with the bound start (78%).
11. **ADAPT trajectories have exact ties at every geometry; ρ-good stopping is
    required.** With measured, fully non-oracle selections, all 288 H₄
    trajectories and 88 of 90 LiH trajectories reproduce exact ADAPT; the other
    two take one extra step and reach the same energy. II-A cuts the cumulative
    selection cost by 41–44% (H₄) and 42% (LiH) against II-0, and II-E by up to
    60% (LiH). Tied steps dominate that cost.

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
| LiH | II-A / II-B mass | −84% / −84% | −81% / −82% | **−73% / −71%** | −83% / −83% |
| LiH | II-A canonical | −80% | −77% | **−76%** | not run |
| H₂O eq HF | II-A / II-B mass | −84% / −84% | −81% / −81% | **−80% / −83%** | −83% / −83% |
| H₂O eq CISD | II-A / II-B mass | −79% / −80% | −79% / −79% | **−76% / −78%** | −80% / −81% |
| H₂O str HF | II-A / II-B mass | −84% / −84% | −73% / −72% | **−70% / −68%** | −88% / −88% |
| H₂O str CISD | II-A / II-B mass | −73% / −74% | −70% / −70% | **−79% / −69%** | −75% / −75% |

The mass rows of LiH and H₂O are the Oct 2 runs on Trillium (job 2485445, 3.6 h;
`runs_table5/`, merged in `runs/step3_oracle_ceiling_table5.csv`). The redesign
column is relative to II-0 on the M3 bound with top-up accounting.

- **II-B adds nothing resolvable on LiH and H₂O.** It changes the design
  objective (M3 bound, top-up) by −1.9% to +2.4% against II-A, and M3 actual by
  between 3 points better and 9 points worse.
- **M3 actual is not what the solver minimises, and it varies between solves.**
  The earlier solve of II-A on these five states reached the M3 bound within 2
  points of the new one, but M3 actual within 11 points: −70% / −80% / −75% /
  −64% / −68% then, against −73% / −80% / −76% / −70% / −79% now. Differences of
  that size between II-A and II-B are not significant.
- **Redesign pays where the cost comes late.** It adds 15–16 points on H₂O
  stretched HF, where about 63% of the static design's cost falls after the first
  elimination, and 1–3 points on LiH and the other H₂O states (11–25% after the
  first elimination).

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
| II-0 | oracle | 43,864 | 26,751 | 307,885 |
| II-A, oracle covariance | oracle | 24,317 (−45%) | 11,643 (−56%) | 146,538 (−52%) |
| II-0 | estimated | 41,463 | 24,699 | 304,990 |
| II-A, HF prior only | estimated | 81,047 (+95%), 197 ok | 143,851 (+482%), 171 ok | — |
| II-A, HF ν = 1000 | estimated | 32,099 (−23%) | 31,696 (+28%), 195 ok | — |
| II-A, HF ν = 100 | estimated | 28,307 (−32%) | 15,293 (−38%) | — |
| II-A, flat ν = 100 | estimated | — | — | 193,042 (−37%) |
| **II-A, data only** | estimated | **27,129 (−35%)** | **13,727 (−44%)** | **184,511 (−40%)** |
| II-A, data only, no guard | estimated | 26,749 (−35%) | 13,764 (−44%) | 184,511 (−40%) |

All rows are 100% correct unless marked.

*Correction (Sept 29).* The three baseline rows (II-0 with oracle and with
estimated radii, II-A with oracle covariance) were first run before the three
fixes below and were not rerun with the II-A rows. They have now been rerun with
the fixed code; the numbers above are the reruns. The data-only saving moves from
36/43/39% to 35/44/40%, and the share of the oracle ceiling captured from
75–84% to 73–75%.

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

## Step 4+: sign-aware rule, contrast objective, non-oracle start

All rows use estimated radii and selected the winner in every trial (200 on H₄,
100 on LiH). Changes are against II-0 with the pairwise rule and estimated radii.
"Bound": the run starts at the a-priori bound max_i Σ_l |A_il| instead of
max_i |g_i|, so no exact quantity enters. "Anytime": round *r* uses
δ_r = 6δ/(π² r²). Every new configuration also bounds the covariance of a context
with fewer than 50 shots per fold by k·I (see "Small-sample radii" below).

| Configuration | H₄ 1.0 CISD | H₄ 2.0 CISD | LiH |
|---|---|---|---|
| II-0, pairwise (reference) | 41,463 | 24,699 | 304,990 |
| II-0, safe | 47,037 (+13%) | 26,126 (+6%) | 549,094 (+80%) |
| II-0, safe, bound start | 42,411 (+2%) | 24,815 (+0%) | 568,140 (+86%) |
| II-0, contrast allocation | 61,080 (+47%) | 35,031 (+42%) | — |
| II-A data, pairwise | 27,129 (−35%) | 13,727 (−44%) | 184,511 (−40%) |
| II-A data, safe | 32,814 (−21%) | 18,962 (−23%) | — |
| II-A data, contrast | 45,569 (+10%) | 21,114 (−15%) | 238,484 (−22%) |
| II-A data, contrast, bound start | 40,618 (−2%) | 17,926 (−27%) | 227,290 (−25%) † |
| II-A data, contrast, bound start, anytime | 74,255 (+79%) | 38,420 (+56%) | — |
| II-A oracle covariance, pairwise | 24,317 | 11,643 | 146,538 |
| II-A oracle covariance, contrast | 37,027 | 15,297 | 146,205 |

† Run on Trillium.

- **The bound start is cheap but not always safe:** −35% to +11% at matched rule
  and objective on most states. On H₂O eq CISD (|g₁| = 0.007, gap 0.002) it selects
  correctly in 99% (II-0) and 92% (II-E) of trials, with intervals missing in 76% and
  99% of trials. The coarse early rounds decide on contexts whose few tens of shots
  have not yet shown their rare outcomes, so the sample variance is zero; the k·I
  bound only covers contexts below 50 shots per fold.
- **The safe rule costs +6 to +13% at II-0 and +21 to +38% at II-A on H₄, and
  +80% at II-0 on LiH.** The safe rows also carry the small-sample radius bound,
  so on LiH the two effects are not yet separated.
- **The contrast objective does not pay on these fixed states.** It is 11–39%
  dearer than arm-wise II-A with the same rule on H₄, and +52% even with oracle
  covariances on H₄ 1.0. It plans the leader at ε plus a 2ε contrast for every
  survivor, which is dearer than resolving each arm to ε when the gap is large.
  On LiH the two objectives tie at the oracle level.

### Interval coverage

Share of trials in which some interval used in some round (marginal for every
active arm, and every designed contrast) missed the exact value. The nominal
family-wise level per round is δ = 0.05.

| Configuration | H₄ 1.0 CISD | H₄ 2.0 CISD | LiH |
|---|---|---|---|
| II-0, pairwise, oracle radii | 0.22 | 0.25 | 0.14 |
| II-0, pairwise, estimated radii | 0.23 | 0.29 | 0.22 |
| II-A, oracle covariance | 0.12 | 0.19 | 0.08 |
| II-0, safe | 0.16 | 0.04 | — |
| II-A data, safe | 0.09 | 0.04 | — |
| II-A data, contrast, bound start | 0.12 | 0.14 | — |
| II-A data, contrast, bound start, anytime | 0.00 | 0.00 | — |

Part I's convention (Bonferroni over the pool, a fresh look every round) misses
in 4–29% of trials. Selections stay correct because misses rarely involve the
decisive pair. The anytime bound removes every miss at 1.8–2.6× the cost.

### Small-sample radii

With the bound start, decisions can come when a context holds 2–5 shots per fold,
where a sample covariance can be exactly zero. On the tied state H₄ 1.0 ADAPT4,
II-0 with the bound start then stopped at 9,141 shots on an arm with 6% of the
best gradient. The fix (`radius_min_shots=50`) replaces such a held-out
covariance by k·I over its k Paulis, a valid upper bound because every Pauli has
variance at most one. Radii and planning both use it, so the planner buys these
contexts out of the regime. With it, the same trial selects a best arm.

## Phase 2: ADAPT trajectory states (`*_adapt_trajectory.csv`, `.cache/trajectory/`)

Exact ADAPT-VQE from each HF case (largest |g_i|, all parameters re-optimised),
in the (N_α, N_β) sector, stopping at 1.6 mHa or when every |g_i| < 1e-6. Every
state is a case `<geometry>_ADAPT<k>`. Its gradient problem is built from the
same Hamiltonian object as the trajectory, because a fresh SCF can flip orbital
signs relative to the cached HF problem.

| Geometry | steps | final error | exact ties at step | gap < 5% at step |
|---|---|---|---|---|
| H₄ side 1.0 | 10 | 1.09 mHa | 2, 4, 8 | 1, 2, 4, 6, 8 |
| H₄ side 2.0 | 10 | 3.22 mHa (stalls) | 1, 4, 8 | 1, 4, 6, 8 |
| LiH R = 3.0 | 6 | 0.68 mHa | 1, 4 | 1, 2, 4 |
| H₂O eq | 17 | 1.54 mHa | 2, 4, 8 | 2, 3, 4, 6, 8, 9, 12, 14 |
| H₂O stretched | 18 | 1.43 mHa | 2, 4 | 2, 4, 5, 7, 12, 13, 15, 16 |

Stretched H₄ stalls with every gradient zero by symmetry, a property of the
spin-orbital pool.

### Snapshots (100 trials each, estimated radii unless marked)

Means; [x] = % correct where not 100. † tail-dominated (medians in the report).

| Configuration | H₄ 1.0 ADAPT3 (gap 33%) | H₄ 1.0 ADAPT5 (15%) | H₄ 2.0 ADAPT2 (11%) | H₄ 2.0 ADAPT5 (56%) |
|---|---|---|---|---|
| II-0, pairwise | 26,504 | 411,200 | 789,821 † [95] | 58,402 |
| II-0, pairwise, oracle radii | 26,455 | 2.92e8 † [99] | 6.70e7 † [92] | 53,636 |
| II-A oracle, pairwise | 14,240 | 239,815 | 16,622 | 26,895 |
| II-A data, pairwise | 18,161 | 288,503 [99] | 246,207 † [95] | 29,749 |
| II-A HF prior only | 16,954 | 455,298 [99] | 39,181 † [76] | 69,934 |
| II-A HF ν = 100 | 17,751 | 246,442 | 80,193 † [85] | 32,520 |
| II-0, safe | 29,883 | 406,671 | 28,031 | 55,818 |
| II-0, safe, bound start | 28,233 | 399,827 | 30,726 | 53,882 |
| II-A data, contrast | 27,787 | **197,352** | 23,357 | 34,881 |
| II-A data, contrast, bound start | 24,513 | 218,787 [99] | 24,023 | 30,269 |
| … + anytime | 51,251 | 568,295 | 62,195 | 62,772 |
| II-A oracle, contrast | 17,895 | 193,558 | 21,049 | 28,543 |

- **The plug-in sign rule fails on correlated trajectory states.** On H₄ 2.0
  ADAPT2 every pairwise configuration except II-A with oracle covariances made
  wrong selections (5–24%), with heavy tails. With the safe rule every
  configuration was 100% correct, at 21–31k shots (62k with the anytime bound).
  The learned pairwise configurations miss coverage in 63–78% of trials there;
  the learned safe ones in 1–2%.
- **On hard, small-gap states the contrast objective is the best non-oracle
  method.** On H₄ 1.0 ADAPT5 it costs 197,352: 32% below II-A data, 52% below
  II-0, level with the oracle contrast design.
- **Splitting still pays:** with the safe rule and the bound start, II-A contrast
  costs 13–45% less than II-0.
- **The HF prior is unreliable along the trajectory.** It is the cheapest
  non-oracle row on ADAPT3 and 76% correct on ADAPT2.

**Tied states, ρ = 0.1.** On H₄ 1.0 ADAPT4, exact BAI ran to 1.66e9 shots in a
single trial. With ρ = 0.1, II-0 safe with the bound start stops at 493,275 shots
and II-A contrast at 236,115. On H₄ 2.0 ADAPT1 the figures are 24,633 and 17,361.
All selections were ρ-good: 36–52% picked the lower-indexed generator of the
tie, 44–54% its tie partner, and 4–10% a third generator within 3–5% of the
best.

### H₂O CISD states (Trillium, 100 trials each, two shards)

Mean context-shots (change against II-0, pairwise); [x] = % correct where not 100.

| Configuration | H₂O eq CISD | H₂O stretched CISD |
|---|---|---|
| II-0, pairwise | 2.05e10 | 8.63e7 |
| II-0, pairwise, oracle radii | 2.07e10 (+1%) | 9.19e7 (+6%) |
| II-A oracle covariance | 9.90e9 (−52%) | 3.99e7 (−54%) |
| II-A data | 8.80e9 (−57%) | 3.95e7 (−54%) |
| II-A HF ν = 100 | 8.78e9 (−57%) | 3.92e7 (−55%) |
| II-0, safe | 2.05e10 (+0%) | 9.98e7 (+16%) |
| II-0, safe, bound start | 2.02e10 (−2%) [99] | 1.02e8 (+18%) |
| II-A data, contrast | 8.65e9 (−58%) | 3.92e7 (−55%) |
| II-A data, contrast, bound start | 2.31e9 (−89%) [92] | 2.55e7 (−70%) |

- Learned II-A costs 57% and 54% less than II-0, as much as with exact
  covariances, with every selection correct.
- The contrast design is within 2% of II-A.
- The sign-aware rule costs 0% and +16%.

### LiH and H₂O snapshots (Trillium, 100 trials each)

| Configuration | LiH ADAPT3 (gap 75%) | LiH ADAPT5 (15%) | H₂O eq ADAPT11 (11%) | H₂O str ADAPT8 (15%) |
|---|---|---|---|---|
| II-0, pairwise | 709,261 | 6.60e7 | 1.93e8 | 1.84e8 |
| II-0, pairwise, oracle radii | 707,341 (−0%) | 6.60e7 (+0%) | — | — |
| II-A oracle covariance | 375,474 (−47%) | 3.61e7 (−45%) | 7.43e7 (−62%) | 9.03e7 (−51%) |
| II-A data | 430,494 (−39%) | 4.34e7 (−34%) | 8.17e7 (−58%) | 8.57e7 (−53%) |
| II-A HF prior only | 437,864 (−38%) | 2.36e7 (−64%) [29] | — | — |
| II-A HF ν = 100 | 430,152 (−39%) | 4.14e7 (−37%) | 7.56e7 (−61%) | 8.61e7 (−53%) |
| II-0, safe | 1.14e6 (+61%) | 6.92e7 (+5%) | 2.01e8 (+4%) | 1.81e8 (−2%) |
| II-0, safe, bound start | 1.21e6 (+71%) | 6.64e7 (+1%) | 1.92e8 (−1%) | 1.85e8 (+1%) |
| II-A data, contrast | 600,144 (−15%) | 2.77e7 (−58%) | 8.24e7 (−57%) | 9.10e7 (−50%) |
| II-A data, contrast, bound start | 530,004 (−25%) | 1.16e8 (+76%) [99] | 6.66e7 (−66%) | 7.88e7 (−57%) |
| … + anytime | 910,289 (+28%) | 1.93e8 (+192%) | — | — |
| II-A oracle, contrast | 370,494 (−48%) | 2.91e7 (−56%) | 7.81e7 (−60%) | 9.44e7 (−49%) |

- Learned II-A pays on every state: −34% to −39% on LiH and −53% to −58% on H₂O.
- The contrast design is the best non-oracle method on LiH ADAPT5 (−36% against
  II-A), within +1 to +6% of II-A on H₂O, and 39% dearer on the wide-gap LiH ADAPT3.
- The HF prior alone selects correctly in only 29% of trials on LiH ADAPT5.
- On LiH ADAPT5 the bound start costs a 1% error rate and a heavy tail (median
  5.4e7); the anytime bound does not remove its misses (78% of trials).
- **Tied LiH ADAPT4, ρ = 0.1:** II-0 4.06e7 and II-E 1.55e7 (−62%). The ρ-good
  rates are 100% and 97%.

## Phase 3: logged decisions (`runs/<case>/phase3/`)

`scripts/phase3_selection_record.py` writes the work order's outputs (i)–(xi)
for complete non-oracle decisions:
- (i)–(iii): `static/` holds A, the generators, the Hamiltonian, the library and
  the contexts with their circuits.
- Per round: `*_rounds.jsonl`.
- (iv): `*_refits.json` and the fold designs as sparse C rows in
  `*_designs.npz`.
- (vi)–(vii): outcome histograms per context and fold, and covariance matrices,
  in `*_data.npz`.
- (x): `*_summary.json` gives context-shots, two-qubit gate counts and classical
  time.
- (xi): oracle values go only in `*_validation.json`.

The A = BC check (v) runs at every refit of every run and fails hard. Six
logged decisions on LiH ADAPT3 (three per configuration, non-oracle start) were all
correct, with no interval missing in any round.
- II-0 (safe rule): 1.09–1.27e6 shots in 47–49 rounds.
- II-E: 4.8–6.8e5 shots in 52–53 rounds, with 13 refits and 6–10 min of design
  time per decision.

## Phase 4: measured ADAPT trajectories (`runs/<case>/phase4/`)

Each selection is made from sampled outcomes, fully non-oracle (estimated radii,
bound start, safe rule, small-sample radii), and stops at a ρ-good leader with
ρ = 0.1. Parameters are re-optimised exactly; only selection shots are counted.
There were 48 trajectories per method on H₄ and 30 on LiH.

| Geometry | Method | Median | P90 | Best | ρ-good |
|---|---|---|---|---|---|
| H₄ 1.0 (10 steps, 1.09 mHa) | II-0, safe | 1.88e6 | 2.34e6 | 93% | 100% |
| | II-A data, safe | 1.11e6 (−41%) | 1.44e6 | 92% | 100% |
| | II-A data, contrast | 1.09e6 (−42%) | 1.61e6 | 90% | 100% |
| H₄ 2.0 (10 steps, 3.22 mHa) | II-0, safe | 1.59e7 | 2.29e7 | 94% | 100% |
| | II-A data, safe | 9.39e6 (−41%) | 1.56e7 | 93% | 100% |
| | II-A data, contrast | 8.85e6 (−44%) | 1.56e7 | 94% | 100% |
| LiH (6 steps, 0.68 mHa) | II-0, safe | 8.57e7 | 1.16e8 | 94% | 100% |
| | II-A data, safe | 4.94e7 (−42%) | 7.66e7 | 94% | 99.4% |
| | II-A data, contrast | 3.41e7 (−60%) | 5.52e7 | 96% | 99.4% |

- All 288 H₄ trajectories match exact ADAPT: same length, same final energy. The
  mean shortfall of the selected |g| is 0.13–0.20%.
- On LiH, one of 181 selections per learned method was not ρ-good (shortfall
  13–15%). That trajectory took one extra step and reached the same energy.
- II-A lowers the cumulative selection cost by 41–44% (H₄) and 42% (LiH), and II-E
  by 42–60%.
- Tied and small-gap steps dominate the cost:
  - On H₄ 2.0, the tied step 8 (max|g| = 0.0094) takes 7–14e6 of the total.
  - On H₄ 1.0, the tied steps 4 and 8 take about half of it.

## Paper A diagnostics (`runs/paper_a/`)

`scripts/paper_a_run_diagnostics.py` and `scripts/paper_a_cost_decomposition.py`
produce the shot-reuse, run-evolution and cost-decomposition numbers used in the
Paper A draft (`reports/AFI_ paper_A_plan_projects_1_2.tex`).

**Shot reuse.** Surviving candidates informed per context-shot, accumulated over
a selection; 3 trials each, estimated radii. Per-gradient measurement (M2) has
reuse 1.

| Case | II-0 | II-A, learned |
|---|---|---|
| H₄ 1.0 CISD | 8.9 of 26 | 8.5 of 26 |
| H₄ 2.0 CISD | 8.7 of 26 | 8.4 of 26 |
| LiH HF | 38.6 of 92 | 41.2 of 92 |

**Run evolution.** Both runs below use the same random stream.
- H₄ 1.0 CISD:
  - After the first refit, II-A eliminates 12 candidates in round 2 (II-0: 2).
  - |𝓑_r| falls from 1,276 to 396 products.
  - The leader's variance falls to 0.65 of II-0 at the same shots.
  - The learned covariances are within 12–14% of the exact ones.
  - Cost: 27,874 shots against 41,454.
- LiH:
  - The pilot round carries two thirds of the shots.
  - The first refit already eliminates 47 of 92 candidates (II-0: 19).
  - Cost: 167,789 shots against 247,560.

**Selection versus optimisation.** The optimisation cost model counts the exact
optimiser's evaluations × (1 + r n_k) energies, each at 1 mHa standard error over
sorted-insertion FC groups of H. Selection costs are M1 at its best (planning
bound, radius ρ max|g|/2) and the measured Phase 4 medians.

| Geometry | C_sel M1 | C_sel II-0 | C_sel II-A | C_opt r=2 | C_opt r=4 | S_total (M1 → best), r=2 / r=4 |
|---|---|---|---|---|---|---|
| H₄ 1.0 | 4.1e7 | 1.9e6 | 1.1e6 | 1.6–2.1e9 | 3.1–4.1e9 | 1.02–1.03 / 1.01 |
| H₄ 2.0 | 2.8e8 | 1.6e7 | 9.4e6 | 2.4–2.8e9 | 4.5–5.4e9 | 1.10–1.11 / 1.05–1.06 |
| LiH | 5.6e9 | 8.6e7 | 4.9e7 (II-E 3.4e7) | 6.7e8 | 1.3e9 | 8.9–9.0 / 5.3 |

C_opt ranges cover two independent Hamiltonian builds: degenerate orbitals make
the SCF solution, and with it the Pauli expansion of H, vary between builds.

Static selection is 1–11% of the total on H₄ but 82–89% on LiH. After shared BAI,
selection is below 0.7% on H₄ and 6–11% on LiH, and the learned designs bring LiH
to 3–7%; reoptimisation dominates.
Phase 4 selections under ρ = 0.1:
- 90–94% were the exact best generator (ties included).
- 90–96% were 1 mHa-good.
- All were ρ-good.

## Cluster runs (Trillium)

`cluster/trillium_run.sbatch` runs one whole node per job file in
`cluster/jobs/`.
- **Checkpoints:** every finished trial and trajectory is checkpointed, and
  `--resume` skips finished work.
- **Requeue:** the job requeues itself 15 min before the 24 h limit, at most
  twice. A resumed run reproduces an uninterrupted one exactly (tested).

All jobs are complete and synchronised (`cluster/sync_from_trillium.sh`):
- `lih_a`/`lih_b` took 2.2 and 5.1 h: snapshots, tied state, Phase 4 and Phase 3.
- Eight H₂O node-jobs took 11–17.5 h: Step 4 on both CISD states, plus eq ADAPT11
  and stretched ADAPT8, at 9 configurations × 100 trials in 2 shards each.
- About 110 node-hours in total; none needed a requeue.

Sizing on Trillium:
- One H₂O II-A trial takes about 6 min at a 6.6 GB peak.
- II-0 with the bound start takes about 15 min at 3.6 GB.
- Learned contrast trials take 1.4–5.5 h of design time.

## Reading and caveats

- Steps 2–3 are oracle ceilings, not predictions.
- The joint solver matches independent solvers to 10⁻⁴ on small problems. On the
  large problems it returns a feasible design whose exact allocation cost is
  reported, but its optimality is not certified.
- Step 4 refits per generator, not jointly, so its costs are upper bounds.
- The 50-shot thresholds (learning and small-sample radii) are untuned.
- Per-round Bonferroni intervals do not give run-wide coverage (see "Interval
  coverage"). Only the anytime rows carry a run-wide guarantee.
- Local and Trillium runs use different NumPy/SciPy builds (2.5.2/1.18.1 against
  2.4.2/1.17.1). Per-trial seeding makes the results statistically, not bitwise,
  comparable.
