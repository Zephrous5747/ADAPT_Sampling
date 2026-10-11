# Part II development log

Newest entries last. One entry per code change or run: what changed, why, files, how it was checked,
and (for runs) where it ran. Job output files (`*.log`, `*.out`) in this directory are git-ignored;
this file is tracked.

## 2026-10-05 — SOTA-baseline work started

Trigger: gap analysis of Paper A against `reports/adapt_vqe_state_of_art_2026_with_benchmarks.tex`
(see memory `paper-a-sota-baselines-todo`). Planned items: shot-level M1/M2 baselines, Ikhtiarudin-style
reuse baseline, other pools (qubit-ADAPT first), AIM-style IC baseline or scope note, C_opt sensitivity,
related-work text.

- Baseline check before any change: `pytest tests` -> 67 passed (WSL venv `VENV`,
  numpy 2.5.2, 4 min 21 s).
- No Trillium SSH master connection open, so cluster jobs are prepared but not submitted until the user opens one.

## 2026-10-05 — item 1/2: shot-level external baselines and the reuse baseline

Code (all under `part2_estimator_design/`):
- `src/learning.py`: `LearningConfig.elimination` ("on"/"off"). Off = sequential non-oracle M1: every arm stays in
  the allocation (uniform precision), an `alive` set (arms the rule has not removed; dead arms may not eliminate)
  decides stopping and the returned arm. New `credit` argument of `LearnedM3`: shots already held per context,
  sampled once per trial into the folds, subtracted from the reported cost (zero credit leaves every old random
  stream and result unchanged).
- `src/baselines.py` (new): `StaticM1` (one allocation at a common radius, one multinomial draw; exact variances,
  as Part I's M1), `neyman_topup`, `IndependentContexts` (each arm's own FC groups as circuits + fragment-value
  vectors f(b) = WHT(v); state-independent, `set_state` computes outcome distributions), `IndependentBAI` (M2 with
  sampled outcomes, estimated variances, Popoviciu bound below `min_shots`, rules marginal/pairwise/safe, rho stop).
- `src/contexts.py`: `qwc_groups` (first-fit qubit-wise commuting groups); `build_context_library(groups=..., extra_groups=...)`
  (custom parent grouping; appended contexts with no home Pauli).
- `src/reuse.py` (new): `hamiltonian_terms` (rebuilds H for fixed-state cases and REFUSES unless the energy of the
  problem's state and three stored commutators [H,G_i] are reproduced), `energy_groups` (FC sorted insertion or QWC),
  `ReuseLibrary` (parents + energy contexts, `credit` = optimal allocation of one energy estimate to eps = 1 mHa,
  `overlap`), `reuse_fragment_problems` (fixed assignment: a Pauli measured by an energy context is read from the
  best-funded one).
- `scripts/step5_external_baselines.py` (new): fixed-state driver (checkpoints/shards/resume like step 4), configs
  M1 static / M1 seq / M2 {marginal,pairwise,safe} / Ikh reuse {FC,QWC} / Ikh no reuse QWC / II-0 + reuse / II-A + reuse;
  `--energy-error` sets the energy precision of the free data. `merge_trials.py`, `aggregate_results.py` know step5.
- `cluster/wsl_run.sh` (new): run any command in the WSL venv from Git-bash (`MSYS_NO_PATHCONV=1 wsl -e bash .../wsl_run.sh ...`).
- Tests: `tests/test_baselines.py` (10), `tests/test_reuse.py` (7): all pass.

Findings so far (smoke runs, 8 trials, H4 side 1.0 CISD): M1 static 59,671 shots (Part I planning bound 59,642, so
the sampler/allocator agree); M1 seq 56.8k; II-0 (paper) 41.5k; M2 marginal 166.8k, M2 pairwise 111.0k;
Ikh reuse FC 26.5k; II-0 + reuse FC 23.1k. LiH (2 trials): M1 static 3.56M, M1 seq 0.756M (II-0 paper: 0.305M),
M1 seq safe/bound start 1.59M. NOTE: the covariance-aware sequential M1 is 4.7x cheaper than Paper A's M1 on
LiH -- Paper A's M1 (static, marginal worst-case radius) is a weaker baseline than the sequential one.
Overlap of H's measured groups with the gradient support (FC / QWC): H4 11% / 23% of Paulis (22% / 42% of coefficient
mass); LiH 15% / 32% (34% / 51%); H2O eq 14% / 26% (35% / 56%). Credit at 1 mHa: H4 5.2e5, LiH 6.6e5, H2O 7.3e6 shots.

Caveat found: the cached H4_square_stretch_side2p0_CISD problem is in a different orbital basis from a freshly
built Hamiltonian (E = -1.17 vs -1.87 Ha), so the reuse rows are skipped for that case (guard in `reuse.py`);
all other cached cases pass the check.

## 2026-10-05 — items 1/2 (trajectory driver), 3 (pools), 6 (C_opt sensitivity)

- `scripts/phase4_adapt_selection.py`: new `--methods` (default unchanged: II-0 safe, II-A data safe, II-A data contrast):
  `M1 seq, safe`, `M1 static` (radius rho max|g|/2, exact variances), `M2 safe`, `M2 marginal`, `Ikh reuse, FC`,
  `II-0 safe + reuse, FC`, `II-A data, safe + reuse, FC`; `--energy-error` for the reuse rows. Per-step selection goes
  through `Phase4._select_with`. Smoke run (H4 side 1.0 HF, 4 trajectories, rho 0.1, safe rule, bound start), median
  cumulative selection shots: II-0 safe 2.24M; M1 seq 5.26M; M1 static 41.4M (= Q7's M1 4.1e7); M2 safe 2.72M;
  M2 marginal 3.40M; Ikh reuse FC 4.32M; II-0 + reuse 0.89M; II-A + reuse 1.34M. All 10-step trajectories, all rho-good.
- `src/pools.py` (new) + `part1_bridge.load_problem("<case>@<pool>")`: qubit-ADAPT (`qubit`, Pauli strings of the UCCSD
  JW images with Z removed, one generator each) and qubit-excitation (`qeb`, UCCSD index sets in qubit operators) pools on
  the same Hamiltonian and state; problem cached as `.cache/<case>@<pool>_v1.pickle`; metadata keeps `hamiltonian_terms`.
  `tests/test_pools.py` (7 pass: anti-Hermitian, QEB conserves N, gradients equal dense matrix algebra).
  NOT implemented: symmetry-adapted minimal complete pool (Shkolnikov) and CEO (Ramoa): need the papers' definitions.
- `scripts/step6_pool_census.py` (new): K, |B0|, contexts, uses/Pauli, Part I planning bounds M1/M2/M3 per pool.
  H4 eq CISD: uccsd K=26 |B0|=1276 uses 3.25, M1/M3 1.20, M2/M3 2.07 (= Part I's numbers); qeb K=26 |B0|=2036 uses 2.04,
  M1/M3 1.26, M2/M3 2.15; qubit K=160 |B0|=2064 uses 3.65 but exactly tied leaders (symmetry partners) -> needs rho-good.
- `scripts/step5_external_baselines.py --rho`: rho-good stopping for every config (trial label carries `, rho=...`).
- `scripts/paper_a_copt_sensitivity.py` (new; output `runs/paper_a/copt_sensitivity.csv`): divides the modelled C_opt by f
  (Hessian recycling reports ~10x lower total measurement cost, arXiv:2401.05172). Selection share of the total at f = 10
  (r = 2): H4 1.0 II-0 1%, II-A 1%; H4 2.0 5% / 3%; LiH II-0 56%, II-A 42%, II-E 34%. Breakeven f* where selection =
  half of the cost: H4 1.0 ~850-1450, H4 2.0 ~170-310, LiH 7.8-19.7. So Q7's "optimisation dominates" survives a 10x cheaper
  optimiser on H4 but NOT on LiH.

## 2026-10-05 — items 3 (CEO etc.), 5 (IC baseline), cluster preparation

- `src/pools.py`: added `gsd_qeb`, `gsd_qubit`, `ceo` = the CEO paper's pools (arXiv:2407.08696) on generalised index sets:
  singles = QEs; on each set of 4 spin orbitals (2 alpha + 2 beta): QEs E1 = T_{a1b1->a2b2}, E2 = T_{a2b1->a1b2}, OVP-CEOs E1 +- E2;
  4 same-spin orbitals: 3 QEs, 6 OVP-CEOs (sums/differences of pairs). `tests/test_pools.py` now 12 tests (anti-Hermitian, conserve N and Sz,
  sizes 90/96 on 4 spatial orbitals, CEO = E1 +- E2, CEO doubles are 4-Pauli sums as in the paper's Eqs. 25-26). Census (H4 CISD):
  gsd_qeb K=90 |B0|=3332 uses 5.21, M1/M3 1.29, M2/M3 3.34; ceo K=96 |B0|=3332 uses 3.40 (tied leaders -> rho-good needed);
  LiH: qeb K=92 |B0|=83.5k uses 1.68, M1/M3 9.5, M2/M3 5.6; qubit K=640 |B0|=84.0k uses 2.27 (ties).
  Still NOT implemented: Shkolnikov's symmetry-adapted minimal complete pool.
- `src/ic.py`, `scripts/step7_ic_baseline.py`, `tests/test_ic.py` (6 pass): AIM-ADAPT data model (arXiv:2212.09719): product of single-qubit tetrahedral
  (SIC) POVMs, 4^n outcomes, every gradient and the energy from the same shots; exact outcome distribution, exact estimator covariance, sampled
  multinomial shots; single-shot second moment of a weight-w Pauli is 3^w. UNOPTIMISED POVM (the paper's adaptive POVM optimisation, which lowers the
  energy variance, is not modelled), so IC numbers are an upper bound for the optimised scheme. Memory K 4^n (LiH 6 GB float32, H2O out of reach).
  H4 side 1.0 CISD: leading gradient per-shot variance 66; IC needs ~1.2e6 IC shots (no anytime) / median 2.8e6 with anytime intervals (95.8% correct,
  24 trials) vs 3.5e4-6e4 context-shots for FC methods; the energy to 1 mHa takes 4.87e7 IC shots, so with free energy data the selection costs 0.
  IC estimators are heavy-tailed: per-round normal intervals miss in 7-9 rounds of failing trials and the best arm is eliminated in ~50% of 6 trials
  without the anytime correction -> `--anytime` flag.
- `scripts/phase4_adapt_selection.py --tag`: tagged outputs (`<case>_phase4_<tag>_*.csv`) so new methods never overwrite the old Phase 4 tables
  (nothing of the old tables was touched; the smoke runs went to the scratchpad). `aggregate_results.py` writes `runs/phase4_adapt_selection_tagged.csv`;
  `paper_a_copt_sensitivity.py` reads every tagged trajectory file.
- Cluster: `cluster/jobs/make_ext_jobs.sh` writes `ext_h4.txt`, `ext_lih.txt`, `ext_h2o_<case>_s{1,2}.txt` (8 files); `sync_from_trillium.sh` now also
  merges step5/step7. NOT SUBMITTED: needs the user's MFA SSH master connection (see README, "SOTA-baseline extension").
- Local timing probes (2 trials): LiH step5 per-trial CPU: M1 seq ~19 s, M1 seq safe/bound ~93 s, Ikh reuse FC ~75 s, II-0 + reuse FC ~42 s;
  LiH reuse library build 34 s, M2 contexts 24 s. LiH reuse (FC): Ikh reuse 498k vs M1 seq 756k; II-0 + reuse 202k vs II-0 305k.
- Local H4 step5 (200 trials, eq CISD): M1 static 59,671; M1 seq 55,870; M1 seq safe/bound 53,638; M2 marginal 150,634; M2 pairwise 101,157;
  M2 safe/bound 80,741; Ikh reuse FC 33,988; Ikh reuse QWC 29,765 (II-0 paper: 41,463; II-A data: 27,129).

## 2026-10-05 — bug fix (credit and refits), H4 results, H2O probe

- BUG (mine): `LearnedM3` counted the free credit in the "shots held have grown 1.5x" refit trigger, which froze the II-A design
  after the pilot when the credit (5e5 shots) dwarfed the new shots; "II-A + reuse" on H4 CISD cost 29.8k, worse than II-0 + reuse (24.0k).
  Fixed: `total = max(spent.sum() - credit.sum(), 1)` (new shots only; identical when credit = 0). Regression test
  `tests/test_reuse.py::test_refits_are_counted_on_new_shots_not_on_the_credit`. Old trial file kept in `runs/H4_square_eq_side1p0_CISD/trials_superseded/`.
  Rerun: II-A + reuse FC 14,149 +- 256 (0.34 of II-0). Learning tests (test_learning.py) unchanged and passing. The LiH timing probe's
  II-A + reuse number (240k) predates the fix and is not used.
- Fixed-state H4 results, 200 trials, oracle start, exact identification (mean context-shots): H4 1.0 CISD: M1 static 59,671; M1 seq 55,870;
  M2 SE 150,634; M2 pairwise 101,157; Ikh FC 33,988; Ikh QWC 29,765; Ikh QWC no reuse 72,571; II-0 41,463; II-A 27,129; II-0 + reuse 23,997;
  II-A + reuse 14,149. H4 2.0 CISD: M1 static 58,094; M1 seq 41,979; M2 SE 141k; M2 pairwise 106k; II-0 24,699; II-A 13,727 (reuse rows skipped:
  cached problem in other orbitals).
- Trajectories, H4 side 1.0, 48 each, rho = 0.1, safe rule, bound start, median cumulative selection shots: M1 static 4.137e7 (= Q7's M1),
  M1 seq 5.74e6, M2 safe 2.67e6, M2 marginal 3.18e6, Ikh reuse FC 4.23e6, II-0 1.88e6 (old run), II-A 1.11e6 (old run), II-0 + reuse 1.03e6,
  II-A + reuse 4.67e5. All reached chemical accuracy in 10 steps (as exact ADAPT); rho-good rates 0.998-1.000 (M1 seq: 1 selection of ~480 not rho-good).
- Validation of the proxies against Part I: M1 static 59,671 vs Part I bound 59,642 (H4 1.0 CISD), 3,564,398 vs 3,563,904 (LiH); M1 static trajectory
  median 41,370,161 vs Q7's 41,372,827; M2 pairwise on H2O eq CISD 4.83e10 vs Part I's planning bound 4.65e10; LiH M2 pairwise 1.09e6 vs 1.14e6.
- H2O probe (1 trial, M2 pairwise): 13,235 independent groups built in 200 s, 17 s per trial, peak 4.75 GB. Cheap: M2 is not the H2O bottleneck.
- Run scripts added: `cluster/local_h4_step5.sh`, `local_h4_iia_reuse.sh`, `local_h4_phase4.sh`, `local_pool_census.sh`, `local_pool_runs.sh` (not run: H4 pools go to the
  cluster as `ext_pools_h4.txt`), `local_tests.sh`, `time_*.sh` probes.
- New: `scripts/paper_a_sota_tables.py` (-> `runs/paper_a/sota_fixed_state.csv`, `sota_trajectories.csv`) and `scripts/paper_a_sota_latex.py` (-> `reports/generated/sota_tables.tex`).

## 2026-10-05 — paper text (Paper A) and sensitivity/IC runs

- `reports/AFI_ paper_A_plan_projects_1_2.tex`: new Methods subsection "External baselines at shot level" (M1 static/seq, sampled M2, reuse of the
  energy measurement, IC, pools), new Results subsection Q8 (fixed states, reuse, trajectories, pools, IC) with generated tables (spliced between
  `% BEGIN/END GENERATED sota_tables` by `scripts/paper_a_sota_latex.py --splice`), Q7 paragraph "Sensitivity to the optimisation cost", conclusions
  (done / gained / limiting / follows), related-work sentence on transductive linear BAI + bib entries soare2014, fiez2019 (existence verified by
  search; page numbers not given). Red `\note{}` marks every statement waiting for Trillium runs (LiH, H2O, stretched H4 trajectories, pool shot-level
  columns). Compiles with pdflatex (16 pages, no undefined refs); the tracked PDF was NOT regenerated (build was in the scratchpad).
- Reuse vs energy precision (H4 1.0 CISD, 100 trials; `runs_energy_error/`): eps 0.3 / 1 / 3 / 10 / 30 mHa, free shots 5.8e6 / 5.2e5 / 5.8e4 / 5.2e3 / 580:
  Ikh FC 33.5k / 34.0k / 37.2k / 47.0k / 53.5k; II-0 + reuse 24.2k / 24.0k / 26.4k / 38.1k / 39.1k; II-A + reuse 13.4k / 14.1k / 18.7k / 25.3k / 28.2k
  (no reuse: 55.9k / 41.5k / 27.1k). Saturates below 1 mHa, gone by 30 mHa.
- IC on H4 (200 trials, anytime): 1.0 A CISD median 3.12e6 IC shots (mean 4.00e6, 99.5% correct; rho 0.1: 3.26e6); 2.0 A CISD median 7.15e5 (mean 8.2e5);
  oracle vs estimated radii identical. Without anytime (100 trials, seed 11): 3 wrong, 69% of trials have a missed interval, median 1.34e6.
- Full test suite: 103 passed (67 old + 36 new) in 2.4 min.

## 2026-10-05 (evening) — Trillium submission

- SSH master was open in WSL as root (`/root/.ssh/cm-USER@trillium...`), not in Git-bash: run the sync scripts inside WSL
  (`MSYS_NO_PATHCONV=1 wsl -e bash -c 'cd ".../part2_estimator_design" && bash cluster/sync_to_trillium.sh USER'`).
- Job files regenerated: equilibrium-H4 trajectories dropped from `ext_h4` (done locally; 7 x 27 workers, stretched only); LiH split into
  `ext_lih_a` (fixed + Phase 2 states + IC oracle radii + census; IC with sample-covariance radii dropped: 92^2 x 4^12 flops per round) and
  `ext_lih_b` (7 trajectory methods x 12 workers + 5 pools x 12 workers).
- Submitted (account <allocation>, 24 h, one node each): 2500868 ext_h4, 2500869 ext_pools_h4, 2500870 ext_lih_a, 2500871 ext_lih_b,
  2500872-2500878 and 2500882 = the eight ext_h2o_<case>_s<k> shards. Cluster env: numpy 2.4.2, scipy 1.17.1 (workstation 2.5.2 / 1.18.1).
- `scripts/merge_trials.py --allow-incomplete` (used by `sync_from_trillium.sh`): skips labels whose trials are not all present (mid-run syncs);
  the default still refuses, which is what a final table needs.
- Guard strengthened: `reuse._check_commutators` now compares ALL generators (was three), because a rotation inside a degenerate shell can leave
  the three picked generators unchanged. Strict check passes on the workstation (H4 eq CISD/HF, H4 stretch HF, LiH) and on the cluster (H4 eq CISD,
  LiH, H2O eq CISD, H2O stretch CISD, 14-17 s each); only the cached H4_square_stretch_side2p0_CISD fails (other orbitals). The cluster's LiH energy
  grouping has 39 groups / 5.6e5 free shots (workstation: 38 / 6.6e5): same Hamiltonian terms, different greedy FC order among near-equal |h_p|
  -> not an inconsistency, but cluster LiH reuse numbers use that grouping.

## 2026-10-05 (night) — cluster results in; two flaws in my reuse rows found and fixed

- All 13 jobs completed (22-35 min each; ext_lih_b 2 h 11 min). Results synced; tables rebuilt.
- FLAW 1: "Ikh no reuse, QWC" (the control meant to separate grouping from data) still assigned Paulis to the energy contexts (it only withheld the free
  shots), so it was not plain QWC grouping and the paper sentence "without the data their grouping is 30% dearer than FC" was unsupported.
  Fix: `ReuseSpec.assign` ("fixed" = Ikhtiarudin forced assignment, "home" = plain home design); free_data=False now uses the home design.
  Old trial files moved to `trials_superseded/` (H4 CISD, LiH HF). New rows: "II-A data(, safe) + reuse (home), FC" (II-A starts from the home design and may
  split onto the energy contexts) and "II-A data, safe, FC library, no data" (library with energy contexts, no credit).
- FLAW 2 (found with it): the pairwise II-A + reuse row on LiH HF was only 92.5% correct (15 of 200: the plug-in-sign failure, best arm eliminated at
  rounds 2-3, an exactly-zero-gradient arm returned). Added the sign-aware family (`ext_safe_family`, job 2501236, all 100% correct).
- Observations to be careful with: free data HURT II-A on LiH (sign-aware 325k -> 444k HF, 845k -> 1.03M ADAPT3; helped on ADAPT5 and H4) and the Ikhtiarudin
  baseline is worse than plain M1 seq on H2O eq CISD (1.83e11 vs 9.4e10) and ADAPT11/8; both are consistent with the forced reassignment (free data are tiny
  against the selection cost there) -- being tested with the home-assignment variants.
- Home-start / plain-QWC results: H4 CISD (200 trials): plain QWC no reuse 111,957 (2.0x FC M1 seq 55.9k; with reuse 29.8k); II-A + reuse (home) 14,317 (pairwise),
  21,197 (safe, vs forced assignment 27.7k and plain II-A safe 33.5k); II-A safe with the FC library but no data 33,483 = plain II-A safe, so the extra
  contexts alone cost nothing. LiH HF: plain QWC no reuse mean 1.05e7 (median 2.6e6, 98.5% correct); II-A + reuse (home) 173.5k (100% correct, vs II-A 185k and
  Ikh 410k); safe 323.7k vs 325k (no gain, no harm). ADAPT3: 379k / 764k; ADAPT5: 29.7M. H2O (II-A + reuse home, pairwise): eq CISD 1.094e10 (99% correct; worse
  than plain II-A 8.8e9 and forced assignment 7.16e9), eq ADAPT11 7.91e7, stretch CISD 3.79e7, stretch ADAPT8 8.06e7. Conclusion: forced assignment is better on H2O,
  home start on LiH; "ours + data" in the tables is the best of the three reuse variants per state (the baseline gets the best of FC/QWC likewise).
- Sign-aware family had no M2: added "M2 safe" (oracle start) and queued it for all fixed states plus II-A safe on H2O (jobs 2502002-2502004), because the
  sign-aware ranges (e.g. LiH ADAPT5 "11x") otherwise compare with the weak sequential M1 only.

## 2026-10-05 (late night) — all extension runs complete; paper updated

- Remaining jobs finished: `ext_home_variants` (2501436), `ext_h2o_home_{eq,stretch}` (2501459/60), `ext_m2safe_small` (2502002), `ext_safe_h2o_{eq,stretch}` (2502003/04).
  Sign-aware M2 added ("M2 safe", oracle start) because the sign-aware family had no M2: the LiH ADAPT5 sign-aware margin is 2.39x against M2, not the 11x against M1 seq.
- Final tallies (all from `runs/paper_a/*.csv`): fixed states, pairwise, no free data: II-A 2.06-4.52x cheaper than the strongest of {M1 seq, M2 SE, M2 pairwise}
  (9 states), II-0 1.21-2.73x; sign-aware (9 states): II-A 1.59-4.21x, II-0 1.13-1.83x; trajectories (H4 x2, LiH): II-A 2.41-2.47x, II-0 1.41-1.45x;
  pools at rho = 0.1: H4 12/12 1.70-22.2x (II-0 alone 11/12, loses 0.94 on OVP-CEO CISD), LiH 5/5 2.36-19.5x (II-0 5/5, 1.48-8.19x).
  With free data (ours best-of-variants vs strongest baseline incl. reuse): 1.71-5.17x fixed, 2.88-5.71x trajectories.
  Strongest baseline: M1 seq on H4 and LiH HF/ADAPT3; M2 on H2O, LiH ADAPT5 and all trajectories. Elimination (II-0 vs M1 seq) 1.35-27.8x, sharing (II-0 vs M2) 1.21-4.28x;
  static M1 is 1.07-16.2x M1 seq, never the strongest.
- `scripts/paper_a_q8_text.py` (new) writes the Q8 prose from the data and splices it between `\subsection{Q8` and the scaffold; `paper_a_sota_latex.py` gained the best-baseline
  table (`tab:q8best`), the reuse/energy-precision table, the pool table and sign-aware rows. Paper: abstract, Q3 caveat, Q8, conclusions rewritten; compiles (18 pages,
  no undefined references); scratch build only. Tool note: inline `python - <<EOF` scripts collapse `\` to `\` in this environment: use raw strings or Write a script file.
- LiH IC (oracle radii only): median 2.6e8 IC shots (one trial 9.8e12 dominates the mean), energy to 1 mHa 5.7e8 IC shots (860x the FC energy shots).
- Tests: 103 pass (2.0 min).

## 2026-10-05 (after the commit) — baseline audit against the original papers, literature review, Discussion plan

- No change to the library code or to any result. Committed state is `f2730c3`.
- New: `scripts/price_of_validity.py` (no inputs; two-arm Gaussian idealisation). At delta = 0.05 the per-round Bonferroni radii cost 3.6 / 4.5 / 4.8 times the
  Garivier-Kaufmann lower bound (K = 26 / 92 / 140), the anytime radii 7.3-10.8 times at 10-50 rounds; anytime / per-round = 1.76-2.64, consistent with the measured
  1.8-2.6 of Q3. It is a back-of-the-envelope figure (equal variances, no elimination of the other arms), meant for the Discussion, not for a table.
- New: `reports/paper_a_baseline_gap_and_discussion_plan.tex` (+ pdf in `reports/generated/`): coverage of the 13 items of the SOTA note (3 done, 9 partial, 1 missing),
  audit of our baselines against the papers read in full, 2026 literature table, plan for a Discussion section, order of work.
- Findings from reading the originals (not from our runs):
  * Anastasiou et al. (arXiv:2306.03227v3) is a *pivot* partition (pivot = one Hamiltonian term; commutators with a commuting set of pool operators commute; 2N sets per pivot,
    equal |coefficients| per set, <= N-3 CNOT circuits). Our M1 is first-fit universal FC over the union of all products: a different construction. NOT implemented.
  * Ikhtiarudin et al. published form (Phys. Scr. 101, 255103, 19 Jun 2026; abstract only): pivot-based grouping alone gives the dominant reduction ("practical baseline");
    our reuse baseline models the arXiv v1 (QWC, reuse of the energy data).
  * Huang-Izmaylov (arXiv:2509.14917v1): QWC fragments per arm, exact variances, hand-set radius schedule R_r = 8 eps'_r, no delta, compared with uniform estimation only.
    Our M2 uses FC groups per arm, estimated variances and delta-calibrated radii (cheaper in shots, deeper circuits).
  * Anastasiou's own Table II: pivot grouping vs gradient-by-gradient term partitioning = 0.93 (H2), 1.28 (H4), 3.24 (H6); reduced pools, so not comparable in absolute terms.
- Tool note: the WebFetch page summariser returned wrong content for two papers (it described parameter-shift / hardware-efficient ansatzes for Huang-Izmaylov); the PDFs
  were saved under the session's tool-results folder and read directly. Treat web summaries as leads, not facts.

## 2026-10-05 (evening) — order of work 1-5: pivot grouping, M2-QWC, termination cost, block contexts, score axis

- **Pivot grouping** (`src/pivot.py`, `tests/test_pivot.py`): contexts `(Hamiltonian term P, class of commuting pool strings)`; every gradient is read from the pivot contexts its
  Paulis came from (`pivot_fragment_problems`), or each product from one pivot context by greedy set cover (`merged_assignment`, stronger than the published scheme).  The
  decomposition `[H, G_i] = sum_P h_P sum_S g_S [P, S]` reproduces the stored commutators to 1e-8 on H4, LiH and H2O (hard check at build time).  Anchored 2N classes for the
  qubit-type pools, first-fit classes for UCCSD.  Validation: static cost on the qubit pool is 34.8 in units of (sum|h|)^2/eps^2; Anastasiou's own estimate is 4N = 32.
  H4 1.0 CISD, 200 trials: pivot M1 static 763k, seq 715k; merged static 98k, seq 97k, II-0 68k, II-A 45k; first-fit M1 static 59.7k, seq 55.9k, II-A 27.1k.
  H2O would need 56,242 contexts x 16,384 outcomes: not run.
  `learning.py`: the base design now uses the base problems' own coefficients (`p.x`), not the Pauli targets, so that a product can be read from several contexts
  (identical for II-0 and the reuse designs).  `step5_external_baselines.py`: `PivotSpec`, `BlockSpec`, `M2 QWC ...`, `pivot_setup`, `block_setup`.
- **Block-wise contexts** (`contexts.py`: `contiguous_blocks`, `block_commuting_groups`, `block_generators`, `build_context_library(blocks=...)`, `tests/test_blocks.py`): Paulis commute
  block by block, so each circuit is a tensor product of one Clifford per block (CZ <= size(size-1)/2 per block); size 1 is QWC (no entangling gates), size n is Part I's FC.
  H4 1.0 CISD contexts / mean CZ: QWC 314 / 0, 2-qubit blocks 215 / 1.6, 4-qubit blocks 110 / 4.3, FC 53 / 7.3.
- **M2 with QWC fragments** (`IndependentContexts(grouping="qwc")`, `IndependentConfig.grouping`): Huang and Izmaylov's fragmentation.
- **Bug found and fixed in the QWC reuse baseline**: `ReuseLibrary(..., "qwc")` completed the QWC cliques like FC groups (strategy "mass"): circuits with about 6 two-qubit gates per shot and a
  measured group larger than a product measurement, i.e. more free overlap than QWC measurement provides (H4 1.0 CISD: 23% of the support / 42% of the coefficient mass, now 23% / 25%;
  LiH HF: 32% / 51%, now 13% / 31%).  Now `blocks=contiguous_blocks(n, 1)` with the canonical completion; regression test `test_qwc_contexts_are_product_measurements`.  Old trial files go to
  `trials_superseded/` (`cluster/qwc_fix.sh`, idempotent).  H4 1.0 CISD "Ikh reuse, QWC": 29,765 -> 67,205 shots (the old row favoured the baseline).  The Q8 tables and text that use
  the QWC rows (best baseline with data) must be regenerated once the LiH rows are rerun.
- `scripts/paper_a_termination_cost.py` (cost of certifying `max|g| < tau`, planning bound), `scripts/paper_a_score_axis_cost.py` (energy-score cost `K n_E M_E`).
- Environment: an SSH master started with `ssh -f` through `wsl -e` dies when that call returns; start it in the foreground (`-N`, no `-f`) in a terminal tab.
- **Noisy measurement circuits** (`src/noise.py`, `tests/test_noise.py`): a two-qubit depolarising error `p2` after every CZ and a readout flip `pr`; for Clifford circuits this scales
  the mean of every measured Pauli by a factor from the gates it passes through (`damping`), so `NoisyMoments` replaces each context's outcome distribution by the damped one.
  Verified against a density-matrix simulation (depolarising channel after each CZ plus readout flips) to 1e-12.  `step5 --gate-error P --readout-error P` (label suffix
  `, noise=p2/pr`); correctness is judged against the noiseless gradients.  The intervals know only shot noise, so the bias shows as wrong selections.
- **Radius rule** (`part1_bridge.confidence_z`, `confidence="selection"` in `LearningConfig`, `IndependentConfig`, `OnlineConfig`, `StaticM1`): Part I's `z = z_{delta/2}/sqrt(2)`
  instead of the Bonferroni value (2.2 times smaller at K = 26).  Configs `... , selection z` (`StaticSpec` for M1 static): does any ranking depend on the radius convention?
- Cluster (user keeps a foreground master): jobs 2502773 `depth_lih_pivot`, 2502774 `depth_lih_blocks`, 2502775 `depth_lih_qwc` (corrected QWC reuse rows via `cluster/qwc_fix.sh` + M2 QWC),
  2502781 `depth_h2o_light` (H2O termination bound, energy-score cost, M2 QWC), 2502803 `depth2_lih_selz`, 2502804 `depth2_lih_noise`, 2502805 `depth2_h2o_qwc` (M1 static/seq, II-0 on QWC
  contexts of H2O), 2502807/2502808 `depth2_h2o_traj_eq/stretch` (measured ADAPT trajectories on H2O, four methods, 12 each).  Job files `cluster/jobs/make_depth_jobs*.sh`; helper scripts
  `cluster/submit.sh`, `check_logs.sh`, `watch_jobs.sh`.  LiH pivot probes: library 21,243 contexts (5.6-8.7 CZ mean), 14,700 for the qubit-type pools.
- **Paper** (`reports/AFI_ paper_A_plan_projects_1_2.tex`, 22 pages): Methods gained paragraphs on pivot grouping and M2 with QWC fragments and a subsection on block-wise contexts and
  the noise model; a new Discussion section (sec:discussion; baseline fidelity, depth, noise, radius rule, stopping the run, selection and optimisation, other scores, scale, threats)
  with five generated tables spliced from `runs/` by `scripts/paper_a_discussion_tables.py --splice` (markers `% BEGIN/END GENERATED disc_<name>`); Q8 tables and prose regenerated after
  the QWC fix (`cluster/regen_q8.sh`; overlap statistics from `scripts/paper_a_reuse_overlap.py`; H4 1.0 CISD best-with-data baseline is now reuse FC, 33,988, ratio 2.40, not reuse QWC 29,765);
  bibliography extended (Ikhtiarudin published form, Garivier-Kaufmann, Gonthier, Bansingh, Dalton, Haravu, Larrucea, Utama, Hagelueken, Lee, Patel, Vahedi, Mullinax, Vaquero-Sabater);
  Conclusions edited (circuit caveat, pivot, stop certification).  The H2O QWC rows, the H2O measured trajectories and the LiH pivot sequential rows are still being produced.
- **Failed cluster jobs (2026-10-06 01:40) and what was done.** `depth2_lih_selz` (2502803) and `depth2_lih_noise` (2502804) ended FAILED (exit 1), and lines of `depth_lih_pivot` also crashed:
  every one of them died with `json.decoder.JSONDecodeError` in the *post-processing* of `step5_external_baselines.py` (several processes of one case rewrote `<case>_step5_meta.json` at
  the same time and one read a half-written file), after all trials of the line were written and merged.  All trial files were complete (200/100 rows each, checked on the cluster); nothing
  was lost.  The runner now reads the meta file tolerantly and writes it atomically (already on the cluster), and the two jobs were relaunched as 2503136 (`depth2_lih_selz`) and 2503137
  (`depth2_lih_noise`, now with the p2 = 0.06 level on LiH as well) with `--resume`, which skips the finished configurations (a first attempt, 2503131/2503132, ran an empty
  job-file name because of a shell quoting slip in the submit command and did nothing).  The first H2O QWC job (2502805) ran out of memory (24 workers per line)
  and was replaced by 2502844 (10 workers per line), which completed.  Results added to the paper: LiH pivot sequential rows (published 1.78e7 = 21x first-fit M1 seq, merged 4.44e6, merged II-A 3.95e5)
  and the H2O QWC rows (II-0 on QWC 2.11e10 / 8.80e7 against 2.05e10 / 8.63e7 on FC, M2 QWC 3.84e10 / 1.32e8).
- Tests: 133 pass (11 min) after the pivot, block, M2-QWC, QWC-reuse, noise and confidence changes.
- LiH HF depth frontier (100 trials, cluster job 2502774): II-A pairwise / safe 3.63e5 / 6.08e5 on QWC contexts (0 CZ), 3.64e5 / 4.87e5 on 2-qubit blocks (2.6 CZ), 3.10e5 / 6.02e5 on 4-qubit
  blocks (6.2 CZ), 1.85e5 / 3.25e5 on fully commuting contexts (21 CZ); M2 with QWC fragments 1.14e6 (0 CZ), M2 FC 1.01e6, M1 seq FC 8.33e5.  Radius rule (sign-aware, H4): II-A 33.5k -> 17.8k,
  M1 seq 53.3k -> 24.1k, 100% correct; the pairwise rule with the loose z is 76-79% correct (H4) and 34-38% (LiH).
- Local H4 1.0 CISD results added (200 trials): block frontier M1 static 120k / 99k / 71k (QWC / 2 / 4-qubit blocks) against 59.7k (FC); M1 seq 113k / 86k / 64k against 55.9k;
  II-0 80k / 63k / 49k; II-A 61k / 50k / ... (see `runs/paper_a/depth_frontier.csv` after `scripts/paper_a_depth_tables.py`).

## 2026-10-06 (afternoon): stretched-H2O trajectories landed; sanity checks (`scripts/sanity_checks.py`)
- Cluster job 2502808 (`depth2_h2o_traj_stretch`, II-A line) COMPLETED in 11 h 42 min; queue empty.  `sync_from_trillium.sh` merged it:
  stretched H2O, 12 trajectories each, median selection shots II-A 2.97e9, II-0 7.85e9, M2 safe 8.45e9, M1 seq safe 8.39e10 (II-A 2.8x below M2); every trajectory reached chemical accuracy.
  (Q8, depth and discussion tables regenerated afterwards, see below.)
- New `scripts/sanity_checks.py` (checks C1-C14 in its docstring).  Statistics are recomputed from the raw trial files, so a stale summary cannot hide a corrupted trial file;
  validity of rho=0.1 runs is judged by the 10%-good rate.  Mutation test (7 deliberate corruptions of a copy of LiH ADAPT3: missing/duplicate trials, wrong arm, NaN, all-wrong method,
  CZ in a QWC row, II-A costlier than II-0) -> all detected.
- Result on LiH (HF, ADAPT3, ADAPT5 and the five pool variants; 179 job requests, all with the requested trial counts): 0 FAIL.  One true arm per case across all 103/41/41 configs (91 / 76 / 11).
  KNOWN (documented, unsafe by design): loose selection radius with the pairwise rule (33-51% correct) and with the sign-aware rule on ADAPT5 (48-74%; the paper already says so), II-A HF prior at ADAPT5 (29%).
  WARN: `Pivot merged II-A` has 50 trials by design (cost); `II-A data + reuse, FC` 92.5% correct (n=200, not significantly below 95%; pairwise rule with reuse is outside the guarantee).
  M1 static realised/planning = 1.0001; II-0 oracle realised/planning 0.92 and II-A oracle 1.8 (canonical ceiling; the learned design pays for refits and elimination rounds).
  Depth/noise/pivot rows exist for LiH HF only (ADAPT3/5 have pivot, QWC and reuse rows): by design of the job files.
- Paper regenerated (`cluster/regen_all.sh`, local only: sota tables, Q8 text, depth, discussion tables): the stretched-H2O II-A trajectories fill the Q8 trajectory table and the trajectory-cost table
  (II-A 2.97e9 = 0.38 x II-0; M2 sign-aware 2.85 x II-A).  Text edits, four sentences only: abstract and Conclusions "2.4--2.5" -> "2.4--2.8" times fewer along trajectories, Q8 "three" -> "five measured
  trajectories", and the threats sentence "measured trajectories were run only on H4 and LiH" -> 12 per method on each H2O geometry.  Paper builds, 23 pages (same as before).

## 2026-10-06 (evening): jobs to complete Tables XI/XII on H2O; phase 4 runs in stages
- Blank cells of Table XI (strongest baseline) and XII (fixed states) on H2O: M2 successive elimination, sign-aware M1 seq, the four sign-aware reuse rows (fixed states and the two ADAPT states),
  and the reuse rows of the two H2O trajectory sets.  (H4 2.0 CISD reuse rows stay blocked: the cached problem is in other orbitals; needs a rebuild and a rerun of that column.)
- `scripts/phase4_adapt_selection.py`: selections are checkpointed (chosen generator, diagnostics, RNG state; `<method>.traj_<seed key>.partial.jsonl`) and replayed on restart (the VQE re-optimisation is
  deterministic); `--max-selections N` stops every trajectory after N live selections of the job and writes no tables, the last stage (no limit) finishes.  Staged run == direct run exactly
  (H4, 3 trajectories, 30 selections; `tests/test_phase4_stages.py`).  Reason: measured per-trajectory time of II-A on H2O is 8.4 h (eq) / 11.5 h (stretch) without reuse and reuse costs 2.5-2.6x on LiH
  (II-A 2300 -> 5842 s, II-0 836 -> 2190 s), i.e. about 21 h / 29 h per trajectory, which does not fit one 24 h job and a trajectory cannot be split across jobs otherwise.
- Job files (cluster/jobs/make_h2o_completion_jobs.sh, local only) and `cluster/submit_h2o_completion.sh`, submitted as 2505231-2505240: `h2o_fix_{eq,stretch}` (36 workers per line, ~1 h expected),
  `h2o_trj_{eq,stretch}_cheap` (Ikh reuse + II-0 + reuse, ~4-9 h per trajectory), `h2o_trj_{eq,stretch}_iiA_s{1,2,3}` chained with afterok (6 + 6 + rest selections per trajectory, each stage ~8-14 h at 3x).
- When done: `cluster/sync_from_trillium.sh USER VENV/bin/python`, `cluster/regen_all.sh` (check the H2O trajectory rows of Q8 generators pick up the reuse methods), recompile, commit.

## 2026-10-06 (night): H4 2.0 A CISD rebuilt; reuse rows filled; II-A on QWC contexts: coverage and jobs
- **Why the stretched-H4 CISD reuse rows were blocked.**  The cached problem and a fresh build (2 s) have the same energy, the same |g_i| for all 26 generators and identical Pauli support in every
  commutator; their coefficients differ by exactly +-1 per Pauli, i.e. the cached one was built in another orbital sign gauge, so the rebuilt Hamiltonian failed the reuse guard (`reuse._check_commutators`).
  Fix: old pickle moved to `.cache/superseded/`, the problem rebuilt (validation gate on); the guard now passes for all 26 generators.  NOTE: the cluster still holds the old pickle (any cluster run of this case
  with reuse needs `sync_to_trillium.sh` of the new one; nothing queued uses it).
- **Gauge check** (`cluster/local_h4s_reuse.sh`, 200 trials, `runs_gauge_check/`, not merged): M1 static 58,094 (identical), M1 seq 41,307 vs 41,979 (z -0.3), M2 pairwise 104,292 vs 105,605 (z -1.0),
  II-A safe 19,382 vs 19,157 (z +0.7): the existing H4 2.0 rows stay valid; they were not rerun.
- **New H4 2.0 rows** (200 trials, all 100% correct): pairwise Ikh reuse 33,638, II-0 + reuse 17,129, II-A + reuse 10,872 (home 10,720); sign-aware 36,316 / 20,747 / 18,891 (home 16,925).  Q8 tables regenerated
  (best with data at H4 2.0: reuse FC, II-A + data 3.14x pairwise / 2.15x sign-aware); `paper_a_q8_text.py` now takes the free-data percentages over both H4 geometries (ranges) and the trajectory ranges/count over
  all trajectory systems (a regeneration had reverted my hand edit "2.4 to 2.8 / five").
- `scripts/sanity_checks.py` C10 relaxed to 2% (M2 safe vs pairwise differ in 2 of 200 trials on H4 2.0: independent arms, sign near zero).
- **II-A on QWC contexts: what exists.**  Complete (100-200 trials, pairwise and sign-aware, blocks 1/2/4, M2 QWC, pivot): H4 1.0 and LiH HF only.  Missing before today: H4 2.0 (all depth rows), LiH ADAPT3/5
  (II-A and all block rows), H2O (II-A, sign-aware, blocks 2 and 4, pivot).  Local one-trial probe of II-A on H2O eq QWC: the library (13,609 contexts, 81,566 Paulis, 4.8e6 II-A coordinates) builds in 2-6 min and
  the fragment problems need 4.9 GB, but the sampler tables (n_contexts x 2^14 float64/int64 = 1.8 GB each, several alive at once) exceed 13.5 GB -> cannot run on the 15 GB workstation, runs on Trillium.
- Jobs: local `cluster/local_h4s_depth.sh` (H4 2.0 blocks 1/2/4 x 7 configs, M2 QWC, QWC reuse, pivot; 200 trials); Trillium 2505353 `depth3_lih_qwc_adapt` (LiH ADAPT3/5, blocks=1, 5 lines each, 100 trials),
  2505382/2505383 `depth3_h2o_{eq,stretch}_qwc_iiA` (II-A blocks=1, 24 trials, 8 workers, one node each).

## 2026-10-06 (night, later): why II-A on QWC contexts ran out of memory, and the dual solve
- **Symptom.**  `depth3_h2o_{eq,stretch}_qwc_iiA` (2505382/3, 8 workers each) died after 4 min with OUT_OF_MEMORY; the kernel log shows the eight workers at 67-124 GB each (killed one: 123.8 GB anon RSS,
  parent 7.5 GB), i.e. the whole 755 GB node.  `depth3_lih_qwc_adapt` (2505353) lost its two LiH ADAPT5 II-A lines (7 and 9: "Not enough memory to perform factorization", BrokenProcessPool) while the ADAPT3 ones finished.
- **It was not the parallelism.**  Instrumented single trial (`cluster/profile_structs.py`: RSS every 15 s, size of every Hessian structure): at the FIRST refit of H2O eq on QWC contexts 12,010 of 12,011 contexts are
  usable (LiH: ~7%), arm 0 has 88,228 free variables and 1.9e9 Hessian entries = a 76 GB structure that took 102 s to build, the worker at 94 GB.  There are 140 arms x 2 folds, all kept alive until the refit
  ends, and each needs a sparse LU of a nearly dense system.  Over all arms (every context usable): mean 3.5e8, max 4.4e9 Hessian entries per arm, ~2.4 TB of structures per worker and 3e13-5e14 flops per solve.
  Cause: the primal (null-space) formulation has one variable per extra copy of a Pauli (mean 15.3 copies, up to 7,064 on QWC contexts of H2O); no number of workers per node changes that, a single worker cannot finish.
- **Fix 1** (`learning._refit`, `_design_contrasts`): the structure is built for one refit's shots and used once, so it is dropped after `optimise`.  LiH trial: identical result (421,766 shots), peak 4.8 -> 2.9 GB.
- **Fix 2** (`design.FragmentProblem._optimise_dual`): the same minimiser from the dual, one unknown per Pauli (H2O QWC: mean 1,166, max 1,694 instead of 32,364 / 112,538; dense Cholesky 7e8 flops, 0.02 GB),
  assembled from the per-context covariance blocks (block ridge `DUAL_RIDGE = 1e-5` x block mean diagonal; the sum of the copies of every Pauli is restored exactly on its reference coordinate).
  Validation against the primal on LiH QWC (all 92 arms; usable fractions 3/7/20%; oracle, 60-shot and 2000-shot covariances; `scratchpad/dual_validate.py`): the dual minimum is within 3e-8 of the primal one for
  ridge 1e-6 and 1e-5 (1e-7: errors up to 5e-3, ill-conditioned; 1e-4: 2e-6; 1e-3: 2e-4), constraints exact (1e-16), 3-10x faster even where the primal works.
  Used only when the primal would be large: `DUAL_TRIPLETS = 5e7` Hessian entries or `DUAL_ABOVE = 12,000` free variables (`FragmentProblem.primal_cost`, equal to the real structure size; `solver = "primal" | "dual" | "auto"`).
  It never triggers on any FC or reuse library, even with every context usable (H2O eq: 0/140 arms; LiH: 0/92), so no earlier result and none of the queued trajectory stages is affected.
  Tests: `tests/test_design_dual.py` (5).
- Consequence for the paper: the H2O II-A QWC rows (and the LiH ADAPT5 ones) are computed with the dual solve of the same optimisation problem.  The earlier note "II-A coordinates would not fit" in
  `make_depth_jobs2.sh` was about this and is obsolete.
- **Measured with the fix** (profile job 2505772, one H2O eq trial, 1 worker): 16 min 38 s in total (library 2 min, 14 min for the trial: 9 rounds, 2 refits, first refit 123 s with 240 dual solves in 39 s, second 68 s),
  peak 20.3 GB (MaxRSS 20.4 GB), 1.84e10 shots, correct.  So the memory per worker is ~20 GB, not 100+: batches of 8 workers use 160 GB.
- **Resubmitted in small batches** (`cluster/jobs/make_depth4_jobs.sh`, 8 h limit each): H2O eq and stretch, II-A on QWC, 48 trials per geometry in 3 shard jobs of 16 trials on 8 workers (`--shard K/3`):
  2505835-2505840; LiH ADAPT5 II-A (pairwise and sign-aware), 100 trials, 8 workers, one job each: 2505841/2505842.  The two H2O fixed-state jobs of the completion batch (2505231, 2505236) completed in 2 h 04/2 h 11.
- Local: H4 2.0 A depth rows complete (200 trials, 100% correct; `cluster/local_h4s_depth.sh`): II-A on QWC 51,203, blocks of 2 36,559, of 4 27,755, FC 27,129; II-0 on QWC 70,153; M1 seq on QWC 122,305; M2 QWC 208,164
  (II-A at 0 CZ is 4.1x below M2 on QWC); pivot merged II-A 22,711.  Full test suite: 139 passed.

## 2026-10-06 (night): coverage audit of the draft's blank table cells against the jobs
- Regenerated the draft from the synced data (`cluster/regen_all.sh`) and listed every "--" (`scratchpad/blanks.py`).  Table XII (fixed states) has none left; XI has 2 (H2O trajectory "ours + data").
- Filled by queued jobs (generators are generic over method and case names, nothing to change): XVI H2O II-A on QWC (+ ratio) <- 2505835-40; XI/XIII H2O reuse trajectories <- `h2o_trj_*_cheap` (reuse baseline, II-0 + reuse)
  and the chained `h2o_trj_*_iiA_s1..3` (II-A + reuse; ETA ~1-1.5 days: 3 stages of 7-12 h each at the LiH reuse slowdown of 2.5x).  LiH ADAPT5 II-A QWC <- 2505841/2 (feeds the text only).
- Checked that the solver switch cannot touch the queued stages: no FC or reuse library of H2O eq or stretch ever reaches the dual (0/140 arms; max 8,682 free variables, 2.6e6 entries against 12,000 / 5e7).
- Filled now: Table III reuse of the H4 HF states, 3.0 of 26 (`paper_a_run_diagnostics.py`, 3 trials, seed 11 like the CISD rows; rows appended to `runs/paper_a/reuse.csv`).
- Not covered by any job and cheap: H2O trajectories of M2 with Successive Elimination and of the static M1 (Table XIII, 4 cells): `cluster/jobs/make_h2o_extra_jobs.sh` -> `h2o_extra_traj_{eq,stretch}.txt`, NOT yet submitted (ssh master down).
- Not covered and not worth running: II-E on H2O trajectories (about 5.6x II-A: 40-60 h per trajectory), LiH generalised-qubit pool (K = 2100: 4 cells of Table XIV), II-0/II-A on the as-published pivot contexts ("Pivot II-0" is
  defined but never run, "Pivot II-A" is not defined).  Structural blanks (by design): M2 rows of XVI, M1/M3 and M2/M3 of the qubit-type pools in XIV (tied leaders), the refit columns of VIII, HF prior at the HF state in VI.
- Table VI, LiH HF "II-A, learned; rule: sign-aware" is blank because step 4 has no such row for LiH; the same configuration was run in step 5 (324,758 +- 1,480, 200 trials; H4 step 4 vs step 5 differ by < 1 sigma: 32,814 vs 33,467,
  18,962 vs 19,157), so the cell can be filled from it (+6% against II-0 pairwise) if wanted.

## 2026-10-06 (evening): job check, extra jobs submitted, dual-versus-primal at the trial level
- **Jobs checked** (ssh master back): all eight `depth4_*` jobs COMPLETED, status 0, no OOM: H2O eq/stretch II-A on QWC 36-41 min per 16-trial batch (8 workers), LiH ADAPT5 II-A 52-53 min per config.
  `h2o_trj_*_cheap` completed in 3 h 10 / 3 h 22 and `h2o_trj_eq_iiA_s1` in 3 h 08: the reuse slowdown of the trajectories is ~1.2x (not the 2.5x of LiH), so the II-A chain ends tonight (eq, ~22:30) and
  early tomorrow (stretch, ~03:30), not in a day and a half.  Eq stage 2 and stretch stage 2 are running; stage 3 pending (afterok).
- **Results synced** (`sync_from_trillium.sh`): II-A on QWC, H2O eq 1.205e10 +- 8.9e7 (48 trials), H2O stretch 6.06e7 +- 1.25e6 (48), all correct; LiH ADAPT5 9.62e7 (pairwise) / 1.02e8 (sign-aware), 100 trials, all correct.
  M2 on QWC / II-A on QWC: H4 1.0 3.2, H4 2.0 4.1, LiH HF 3.1, LiH ADAPT3 2.8, LiH ADAPT5 1.1, H2O eq 3.2, H2O stretch 2.2.  II-A on QWC against the best full-depth baseline (M1 seq FC): 1.10x and 1.22x MORE shots on H4,
  2.3x, 1.5x, 5.1x fewer on LiH HF/ADAPT3/ADAPT5, 7.8x and 24.6x fewer on H2O.  LiH ADAPT5 is the exception to the learned-splitting gain on QWC: II-A 9.6e7 vs II-0 9.3e7 (means; medians 6.9e7 vs 8.3e7; the mean is
  carried by three trials with 8 refits and 3.8e8 shots, without the 5 largest 8.4e7 vs 8.7e7), while on FC contexts II-A is 34% below II-0 (4.3e7 vs 6.6e7).  Sanity checks: no FAIL.
- **Submitted** `h2o_extra_traj_eq` / `_stretch` (2506204/5): M2 with Successive Elimination and the static M1 along the H2O trajectories (Table XIII), 12 per method, 12 workers per line.
- **Dual versus primal, end to end** (`cluster/force_solver.py dual|primal`: every arm by one solver, same trial seeds).  LiH HF, II-A on QWC, 6 trials: dual vs the stored primal run: shots equal to 1-2 out of ~4e5,
  identical rounds, refits, selections and correctness (one of the six is one of the 2 wrong trials of the original 100).  Optimiser level on the ADAPT5 state (ridge 1e-5, 92 arms, 3 usable fractions,
  oracle / 60-shot / 2000-shot covariances): max relative difference 1.3e-7 (oracle), 2.6e-7 and 6e-5 for the 60-shot covariances at 20% usable, no arm better than the primal by more than 1e-6.
  LiH ADAPT3 against the stored (cluster) run: not identical trial by trial (shot ratios 0.78-1.39); the stored run was made on the cluster (numpy 2.4.2 / scipy 1.17.1) and a trajectory state, unlike the HF determinant, is not
  reproduced bit for bit across environments, so the same trials were rerun with the primal on this workstation (see below).
- **Extra trajectory jobs finished** (2506204/5, 28 and 34 min): H2O trajectories, 12 per method, all reached: M2 with Successive Elimination 5.25e9 (eq) / 1.73e10 (stretch), the static M1 2.44e11 / 9.55e11
  (sign-aware M2: 2.73e9 / 8.45e9; II-0: 2.46e9 / 7.85e9).  Synced; sanity checks (C13) clean.
- **Dual versus primal on ADAPT3, same machine** (`cluster/force_solver.py`, 6 trials): the primal rerun here reproduces the stored cluster run exactly (shot ratio 1.000 on every trial), so the cluster/workstation environment is
  not the issue; the forced dual differs trial by trial (ratios 0.78-1.39 pairwise, 0.90-1.39 sign-aware; means 1.01 and 1.09).  Cause (`cluster/compare_solvers_refit.py`, both solvers inside the real refits of two trials,
  706 (arm, fold) designs): in-sample the designs agree (median 2e-9 .. 3e-8; max 3e-4), out of sample (variance on the other fold, the guard's check) median -2e-6 .. -4e-5 (dual slightly better), extremes +8e-3 / -6.6e-2,
  |difference| > 1% for 0-5 designs per refit, guard decisions differ in 1 of 706, but the coefficients can differ by more than the home scale: the objective does not determine them (near-null directions of the 50-shot
  covariance blocks) and the dual's ridge (1e-5 of the block mean diagonal) picks the minimum-norm ones, the primal the minimum-norm ones in its own parametrisation.  Every design is feasible (sums exact), so validity does not
  depend on it; the cost of II-A can.  Submitted `dualcheck_lih` (2506560): 100 trials each of LiH HF and ADAPT3, pairwise and sign-aware, II-A on QWC with the dual forced, to compare the averages with the stored primal runs.

## 2026-10-06 (night): the remaining table cells go to Trillium; `--max-hours`; the published pivot contexts get II-0 and II-A
- **Dual versus primal, averages** (`dualcheck_lih`, 2506560, 8 min): LiH, II-A on QWC, 100 trials each, dual forced against the stored primal rows: HF pairwise 3.634e5 +- 9.1e3 (primal) vs 3.634e5 +- 9.1e3 (dual),
  1.000, z +0.01; HF sign-aware 6.075e5 vs 6.082e5, 1.001, z +0.09; ADAPT3 pairwise 8.48e5 vs 8.37e5, 0.987, z -0.45; ADAPT3 sign-aware 1.235e6 vs 1.225e6, 0.991, z -0.51; correct shares identical (0.98/0.98, 1.0/1.0).  The two solvers
  give the same average cost to within 1.3% (noise ~2%); the trial-by-trial differences (flat directions) average out.  The Methods sentence ("same variance ... though not always the same coefficients") stands; the
  100-trial comparison can be cited.
- **`--max-hours H`** (`scripts/phase4_adapt_selection.py`): stages by the clock instead of by selection count.  A trajectory starts no live selection that, at `SELECTION_GROWTH = 2` times the length of its previous one
  (recorded or live), would end after H hours of the job; the stage in which every trajectory finishes writes the tables.  Needed because selections of the contrast design take hours and later ones are slower (LiH: 1.2x
  II-A time at k=0, 3.5-4.5x at k=1-3, 7-12x at k=4-6).  Test: `tests/test_phase4_stages.py::test_time_budget_stops_before_a_selection_that_would_not_fit` (deadline in the past pauses at 0; recorded selections of 1e5 s
  with 1e4 s left pause at 2; with time enough the trajectory equals the direct run); the earlier staged-equals-direct test still passes.
- **New configurations** `Pivot II-A` and `Pivot II-A, safe` (`PivotSpec(II_A[_SAFE])`, published assignment): the learned splitting started from the published pivot design.  `paper_a_depth_tables.py` maps them to the
  "pivot (published)" family.  H4 1.0 CISD smoke test, 4 trials: Pivot II-0 5.7e5 (M1 seq 7.15e5), Pivot II-A 1.05e5 (merged II-A 4.5e4).  Run locally for H4 1.0 (`cluster/local_h4_pivot_published.sh`, 200 trials).
- **Submitted** (cluster scripts stay out of git): `fill_misc` (2507739, 10 h limit): LiH generalised-qubit pool (K = 2100; II-0, II-A, M1 seq on 16 workers each, M1 static + M2 pairwise on 4; 100 trials, rho 0.1) and II-0 / II-A on the
  published pivot contexts of LiH HF (30 workers each, 100 trials).  `h2o_iiE_s1..s7` (2507740-6, chained afterok, 24 h each, `--max-hours 20`): II-E (the contrast design, "II-A data, contrast") along the H2O eq and stretched
  trajectories, 12 per geometry, both geometries on one node (24 workers); the guard `[ -f ..._summary.csv ] ||` makes stages after the finishing one no-ops (without it `--resume` would find no checkpoints and start over).
  Expected: 45-70 h (eq) and 60-90 h (stretch) per trajectory (II-A 8.4 / 11.5 h x 5-8), i.e. 3-6 stages.
- **Finding: the pairwise II-A is not valid on the published pivot contexts.**  H4 1.0 CISD, 200 trials: `Pivot II-0` 5.09e5 +- 1.2e4 (correct 100%; M1 seq 7.15e5, M1 static 7.63e5) but `Pivot II-A` correct in only 132 of 200
  (mean 9.3e6, median 1.05e5; the 68 wrong trials end after 3-4 rounds at ~78k shots with the best arm eliminated, or run away to 7e8 shots).  LiH HF (partial, 74 trials): 27% correct, median 6.5e5, mean 5.0e6.
  Diagnosis (`scratchpad/dbg_pivot_iia*.py`, trial 8): the designs are consistent (with the exact context means every refitted design reproduces the true gradients to 4e-16), but after the first refit the held-out
  variance of the leading arms is 10x too small (arm 19: estimate 0.39 +- 0.03 against truth 0.024).  Cause: in the published scheme every product is measured in several pivot contexts, so the pilot spreads
  ~67k shots over 898 contexts; the pairwise configurations have `radius_min_shots = 0`, i.e. plug-in covariances from folds with a few dozen shots (the documented small-sample failure of `learning.py`: a sample
  covariance can be exactly zero), while the sign-aware family sets `radius_min_shots = 50` and replaces them by the bound k I.  Sign-aware rows on H4 (40-trial probe): II-A 1.10e5 +- 1.2e3, II-0 5.38e5, M1 seq 7.31e5, all
  correct.  So Table XVI's "pivot, as published" row shows the sign-aware II-A (dagger) with the sign-aware M1 seq in the ratio (about 6.7x), and the caption says why (generated: `paper_a_discussion_tables.py`
  swaps the cell when the pairwise row's correct share is below 0.95 and the sign-aware row's is not; the shares enter the caption).  `sanity_checks.py` lists `Pivot II-A` (pairwise) as KNOWN, and checks
  Pivot II-A <= II-0 <= M1 seq (pairwise and merged).  Runs: H4 `cluster/local_h4_pivot_published_safe.sh` (200 trials, local); LiH `fill_pivot_safe` (2507794: M1 seq / II-0 / II-A sign-aware, 100 trials, 30 workers
  each); the pairwise LiH `Pivot II-A` of `fill_misc` stays (it documents the failure).  LiH published `Pivot II-0` (pairwise): 3.28e6 +- 1.0e5, correct 100% (M1 seq 1.78e7, 5.4x).
- First results of `fill_misc` (LiH gsd_qubit, K = 2100, rho 0.1): M1 static 4.96e8 (correct 14%: tied leaders), M2 pairwise 6.95e7 (9%).
- Final numbers of the published pivot contexts.  H4 1.0 CISD, 200 trials (all in `runs/`): M1 seq 7.15e5 (sign-aware 7.37e5), II-0 5.09e5 (5.37e5), II-A pairwise 9.29e6 mean / 1.05e5 median, correct 66%,
  II-A sign-aware 1.098e5 +- 6e2, correct 100% -> ratio sign-aware M1 seq / II-A 6.71.  LiH HF, 100 trials (fill_misc): Pivot II-0 3.28e6 +- 1.0e5 (100% correct), pairwise Pivot II-A 8.83e6 +- 4.3e6, median 6.8e5, correct 24%.
  Local full test suite after the changes: 140 passed (9 min 41 s).  End-to-end check of the stage logic through `main()` (H4 HF, M2 safe, 2 trajectories): `--max-hours 1e-7` pauses at 0 selections and
  prints the stage message, the next run with `--max-hours 1` finishes and writes the tables.

## 2026-10-07 (afternoon): fill jobs finished and synced; dry-run of the tables; II-E projection
- **Finished** (all status 0): II-A + reuse chain (eq s3 4 h 09, stretch s2 5 h 53, s3 7 h 08), `fill_misc` (2 h 22), `fill_pivot_safe` (31 min).  `h2o_iiE_s1` is at 18 h of its 24 h.  Synced (`sync_from_trillium.sh`).
- **LiH generalised-qubit pool (K = 2100, rho 0.1, 100 trials)**: II-0 8.26e6 +- 1.5e5, II-A 1.83e6 +- 1.8e4, M1 seq 8.52e7 +- 2.5e6, M1 static 4.96e8, M2 pairwise 6.95e7 +- 1.1e5; shortfall <= 0.1 in every trial of every
  configuration (the exact best arm is 9-17%: tied leaders).  **Published pivot contexts, LiH HF, 100 trials**: M1 seq 1.78e7 (sign-aware 1.52e7), II-0 3.28e6 (sign-aware 4.01e6), II-A pairwise 8.83e6 (24% correct), II-A sign-aware 9.08e5 +- 7e3
  (100% correct) -> sign-aware ratio 16.8.
- **Sanity checks** (LiH HF, LiH@gsd_qubit, H2O eq/stretch HF, H4 1.0): no FAIL; 41 WARN (the known ones: noise rows, selection-z rows, reuse pairwise 0.925, `Pivot merged II-A` 50 trials), 9 KNOWN (selection-z rows, pairwise `Pivot II-A`).
  All H2O trajectory methods reached the target in every trajectory, rho-good 1.000.
- **Dry run of `regen_all.sh` on a scratch copy of the draft** (the real draft untouched): the only non-structural blanks left are the two II-E cells of Table XIII (H2O eq / str) and Table VI's LiH sign-aware cell (fillable from step 5: 324,758);
  structural blanks: Table VI HF prior at HF, Table VIII refit columns (15), Table XIV tied-leader bounds (12), Table XVI M2 independent-arm rows (32).  Table XI H2O "ours + data": 3.07 (eq), 3.56 (stretch).
- **II-E projection** (checkpoints of stage 1, 12 trajectories per geometry): selection hours by k, eq 0.8 / 2.5 / 2.6 / 2.7 / 2.9 / 2.3 / 4.0 / 2.2, stretch 1.4 / 1.8 / 3.7 / 2.5 / 4.3 / 4.0; E/A ratio ~6 and flat (not the LiH growth);
  per trajectory ~50 h (eq) and ~75 h (stretch), i.e. about 30 h and 57 h still to do after stage 1 -> stages s2..s5 (each 15-20 h of work); the 7 chained stages suffice, finish about Oct 11-12.
- **Tables filled in the manuscripts (2026-10-07, on request): `AFI_ paper_A_plan_projects_1_2.tex` and `reports/draft.tex`.**  Generators now: Table XIII caption adds "A dash means that the configuration was not run" only while a dash exists (the two H2O II-E cells),
  Table XIV's dash is "no planning bound, because the leaders are tied", Table XVI's caption explains the independent-arm rows and carries the dagger note; both files spliced with the same generated blocks.  Hand edits (both files): depth paragraph
  (seven-state QWC comparison 1.1-4.1x against independent BAI with QWC fragments; against the best baseline allowed the full depth 1.10x and 1.22x more on H4, 2.3x, 1.5x, 3.1x, 2.1x fewer on LiH HF / ADAPT3 and H2O, parity (1.02x) on LiH ADAPT5, which is the
  exception: II-A on QWC 9.6e7 vs II-0 9.3e7; published pivot contexts; H2O QWC II-A 1.2e10 / 6.1e7), the limitations sentence, the dual solver (mean costs agree to 1.3% over 100 LiH trials), Table VI LiH sign-aware cell 324,758 (+6%) from step 5, Table VIII caption.
  Draft-only: the Q8 pool paragraph (6 LiH pool problems, 2.4-38.0x; II-0 1.5-8.4x) copied from the regenerated original.  Both compile; PDF copy `reports/generated/AFI_paper_A_with_SOTA_baselines.pdf` refreshed.  Open: Table XIII II-E on H2O.
- 2026-10-08: paper_a_q8_text.py now includes the H2O trajectories in the 'cheaper than independent BAI' and 'cheaper than the reuse baseline' ranges (the original read 4.6 to 9.1, the data give 4.6 to 29); draft.tex Q8 (tables and prose, edited by another session since) re-verified against the stored data: free-data and sign-aware percentages, pool counts, trajectory ratios all match.

## 2026-10-08 (evening): generators reproduce the hand-edited parts of `reports/draft.tex`
- **Why.** The Q8 prose of `draft.tex` was rewritten by hand on 2026-10-07 and the captions and spelling of the generated tables were edited after the last splice, so a regeneration would have reverted both (`reports/consistency_audit.md`, Sec. 8). Now the scripts produce the draft's text.
- **`scripts/paper_a_q8_text.py` rewritten** as templates with `@slot@` entries holding the draft's prose; every number is computed from `runs/paper_a/sota_fixed_state.csv`, `sota_trajectories.csv`, `reuse_overlap.csv`, `cost_summary.csv`, the step-3 planning bounds (`part1_m1`, `part1_m2`, rounded up to whole shots), the step-5 meta (`credit_shots`), the step-7 trial files and `step6_pool_census.csv`.
  Blind spots of audit Sec. 8 closed: static-M1 and reuse-baseline ratios along the trajectories now cover all five systems; fixed-state (1.1-5.8) and trajectory (7.2-16) static/sequential ranges are separate; no "small pool" claim; the states where each baseline is strongest are named from the data (`describe`); free-data percentages are given for the pairwise and the sign-aware rule (ranges over both H4 geometries and the three LiH states); the stall of the stretched H4 pool (3.2 mHa), the number of trajectories and steps, the pool names, the pool maxima and minima, the "twice as expensive" QWC statement and the 30 mHa of the energy-precision sweep come from the data; `\section{Discussion}` is the end marker when the scaffold comment is absent; the file is written with LF (the paper is LF-only; `write_text` gave CRLF on Windows).
  34 claims of the text are checked against the data (`check(...)`: the strongest baseline per state, sharing against elimination state by state, "never raises the cost", "every sign-aware row correct", the pool statements, ...) and a failed check is printed as a warning.
  **Typed, with no stored source (listed in `TYPED`, printed at every run):** per-shot variance of the leading H4 gradient (66), the IC run without the anytime correction (69% of 100 trials with a missed interval, best candidate eliminated in 3, factor 2.3 in the median). **Also typed:** the LiH HF credit of the energy data (5.6e5, `LIH_CREDIT_FALLBACK`): the merged `LiH_R3p0_HF_step5_meta.json` has no `reuse_libraries` (it is in the meta of ADAPT3/ADAPT5 and H4/H2O but not of the HF state; HEAD has the same gap), so the value is the one of the 2026-10 log.
- **`scripts/paper_a_sota_latex.py`:** `POOL_CASES` has the Hartree-Fock state of H4 at 1.0 A (six more rows) and the caption says "CISD and HF states of H$_4$"; the captions of the four Q8 tables carry basis, pool, delta (typed `SETUP`) and the numbers of trials and trajectories (read from the CSVs); `\tabcolsep` is 2.4 pt for `q8fixed` and 2.2 pt for `q8eps` (parameter of `table`); `\mathrm{opt}`; US spelling ("generalized"); the splice writes LF.
- **`scripts/paper_a_discussion_tables.py`:** score-table ratios below 10 print two decimals (H2O eq CISD: 0.58, not 1); "depolarizing"; the splice writes LF.
- **Test (scratch copy, `reports/draft.tex` not touched):** `paper_a_sota_latex.py --splice`, `paper_a_q8_text.py --paper` and `paper_a_discussion_tables.py --splice` on a copy of `draft.tex` give a byte-identical file (empty diff). Mutation test: 13 deliberate changes in the copy (numbers in the Q8 prose, a tabcolsep, the 0.58, the pool caption, "depolarising", "generalised") were all restored by the regeneration. `sanity_checks.py H4_square_eq_side1p0_CISD LiH_R3p0_HF H2O_eq_CISD H2O_stretch_CISD`: exit 0, no FAIL.
- **Note.** `cluster/regen_all.sh` still points at `reports/AFI_ paper_A_plan_projects_1_2.tex`; running it would now rewrite that file's Q8 with the draft's text. After the II-E jobs (h2o_iiE_s1..s7) finish, run the three scripts on `draft.tex` (and `paper_a_sota_tables.py`, `paper_a_depth_tables.py` first); only the two II-E cells of Table XIII change.
- 2026-10-09: Table XVI independent-arm rows now carry the ratio of independent BAI to II-A on the same family of contexts (FC or QWC; generator paper_a_discussion_tables.py; original spliced, only the disc_depth block copied into draft.tex so the draft's other edits stay).  Tables VIII and XIV keep their dashes: VIII has refit diagnostics only in refit rounds, XIV has no planning bound for tied leaders.

## 2026-10-09: referee checklist, Part A (coding and data generation)
Context: a referee-style review of Paper A listed nine concerns and a coding checklist A1-A12.  This entry records the code written and the runs started; results are appended below as they arrive.  Manuscripts are not touched except Table VIII (see first item).

- **Table VIII** (both manuscripts, hand-written table): the empty refit columns of rounds 3, 4, 6, 7, 8 now carry the values of the last refit in parentheses; caption says so.
- **A1, uncertified fixed-budget selection (the practical default).**  `baselines.py`: `FixedBudgetSpec(shots, allocation)` and `FixedBudgetM1` (one draw of `shots` context-shots, argmax of the estimated |g|, no elimination, no stop); allocation `designed` (static-M1 minimax allocation with the exact fragment variances, the best a fixed budget can do) or `uniform` over the contexts that carry a coefficient.  `phase4_adapt_selection.py`: methods `Fixed <shots>, <designed|uniform>` for 100 ... 1e9 shots per selection (half decades), new summary columns `final_error_median`, `final_error_p90`; `paper_a_sota_tables.py` and `paper_a_bootstrap.py` skip files tagged `fixed_*` and `sens_*` (so the certified methods' rows are unchanged).  Aggregator `scripts/paper_a_fixed_budget.py` -> `runs/paper_a/fixed_budget.csv`, `fixed_budget_summary.csv` (cheapest budget with >= 95% of the trajectories at chemical accuracy and <= exact + 1 steps; the premium of certification = certified II-A total / that total) and a LaTeX table.  `tests/test_fixed_budget.py` (budget spent up to rounding, large budget exact, designed beats uniform in worst-case variance).  Runs: H4 both geometries locally (`cluster/local_h4_fixed_budget.sh`, 48 trajectories), LiH 3.0 and 1.6 A and H2O eq/stretch on the cluster (`ref_fixed_lih`, `ref_h2o_*_fixed_budget`).  First result, H4 1.0 HF (exact ADAPT: 10 steps): a fixed budget of 3.2e4 shots per selection reaches chemical accuracy in 10.1 steps in 100% of 48 trajectories at 3.2e5 shots in total, 3.5x below the certified II-A (1.1e6), 5.9x below II-0, 8.4x below M2; at 1e4 per selection the trajectory is 1.5 steps longer (11.5) and still cheaper (1.1e5); below 3e3 trajectories fail.  The budget is an oracle choice (nobody knows it in advance), and the 'designed' allocation uses exact variances.
- **A2, rule audit.**  `scripts/paper_a_rule_audit.py` -> `runs/paper_a/rule_audit.csv`: for every state the pairwise (oracle start), sign-aware (oracle start) and sign-aware (a priori start) II-0 / II-A rows, their correct shares and savings, and what is missing.  Findings: sign-aware oracle-start rows exist on the nine states of Table XII; the H4 snapshot states (eq ADAPT3/5, stretch ADAPT2/5, Table q4more) had none (pairwise only; stretch ADAPT2 95% correct): job `ref_h4_snapshots` (200 trials, II-0 safe, II-A safe, a priori versions, M1 seq and M2 bound start).  Savings II-A over II-0, pairwise -> sign-aware (oracle start): H4 1.0 CISD 35 -> 29%, H4 2.0 CISD 44 -> 27%, LiH HF 40 -> 41%, ADAPT3 39 -> 26%, ADAPT5 34 -> 43%, H2O eq CISD 57 -> 57%, str CISD 54 -> 51%, ADAPT11 58 -> 62%, ADAPT8 53 -> 51%.
- **A3, a priori starting radius for the sign-aware family** on all nine fixed states.  `step5_external_baselines.py` (`REFEREE` configs, not in DEFAULT): `II-A data, safe, bound start`, `II-0, safe, bound start` (same method as step 4's `II-0 safe, bound start`, used for the rho sweeps), reuse rows `Ikh reuse, FC, safe, bound start`, `II-0 safe + reuse, FC, bound start`, `II-A data, safe + reuse, FC, bound start`.  Together with the existing `M1 seq, safe, bound start`, `M2 safe/marginal, bound start` and step 4's `II-0 safe, bound start` no exact quantity enters any of these rows.  Jobs `ref_small_fixed`, `ref_h2o_{eq,stretch}_bound`.
- **A8, anytime intervals on the headline rows**: `II-0, safe, bound start, anytime`, `II-A data, safe, bound start, anytime`, `M1 seq, safe, bound start, anytime`, `M2 safe, bound start, anytime` (union bound over the rounds); II-0/II-A on all nine states (`ref_small_fixed`, `ref_h2o_*_anytime`), the baselines on H4 and LiH.
- **A9, sensitivity**: minimum shots (the 50-shot threshold of `radius_min_shots` and `min_fold_shots`, together) 20, 100, 200; refit factor 1.25, 2, 3 (`LearningConfig.label` now shows `refit_growth` and `min_fold_shots` when they differ from the defaults, so labels of earlier rows are unchanged); rho 0.05, 0.2, 0.3 on fixed states (`--rho`, H4 CISD both, LiH HF) and along trajectories (phase4 `--rho`, tag `sens_rho<r>`, H4 both geometries and LiH).  The cap of 200 auxiliary candidates belongs to II-B only (`design._auxiliary_coordinates`), which the II-A results do not use.
- **A4, bootstrap intervals**: `scripts/paper_a_bootstrap.py` -> `runs/paper_a/bootstrap_fixed_state.csv`, `bootstrap_trajectories.csv` (10,000 resamples of trials / trajectories, numerator and denominator independent; ratio of means and ratio of medians; flags relative half-widths above 15%) and a LaTeX headline table (`--latex`).  First look, II-A / II-0 mean ratios with 95% intervals (oracle start, sign-aware): H4 1.0 0.712 [0.677, 0.746], H4 2.0 0.733 [0.697, 0.770], LiH HF 0.591 [0.578, 0.605], LiH ADAPT3 0.741 [0.724, 0.759], ADAPT5 0.569 [0.480, 0.676], H2O eq CISD 0.429 [0.409, 0.450], str CISD 0.488 [0.468, 0.510], ADAPT11 0.380 [0.355, 0.405], ADAPT8 0.489 [0.468, 0.510].
- **A5, figures**: `scripts/paper_a_figures.py` -> `reports/figures/fig_{pipeline,rounds,trajectory,depth,fixed_budget}.{pdf,png}` (Okabe-Ito palette, one marker per method; data from the stored runs; a figure whose data are missing is skipped).
- **A6, structure scaling**: `scripts/paper_a_scaling.py` -> `runs/paper_a/scaling.csv`: qubits, K, support, contexts, CZ per circuit, II-A coordinates, time of the II-0/II-A design builds, time of one refit solve per generator (exact covariances, 1e4 shots per context; a sample of generators) and the extrapolation to all K.  New cases (`src/cases_extra.py`, registered into Part I's `CASES` at import of `part1_bridge`; Part I's package is untouched): `LiH_R1p6_{HF,CISD}`, `BeH2_{HF,CISD}` (linear, 1.33 A), `H6_chain_{HF,CISD}` (1.0 A), `N2_HF` (1.098 A), and the linear 1 A cases of Huang and Izmaylov `H4_chain1p0_HF`, `LiH_R1p0_HF`, `BeH2_R1p0_HF`.  Problem builds (workstation): LiH 1.6 28 s (K 92, 30,196 Paulis, 980 contexts), BeH2 167 s (K 204, 85,487 Paulis, 2,450 contexts), H6 75 s (K 117, 35,094, 1,054).  N2 (20 qubits) runs on the cluster (`ref_scaling_n2`; fetch `runs/paper_a_n2/` by hand).
- **A7, a more standard case**: LiH at 1.6 A.  HF (gap 72%), exact ADAPT reaches chemical accuracy in 5 steps; fixed states HF, ADAPT2 (39%), ADAPT3 (7.6%), ADAPT4 (2.6%) with the a priori sign-aware family + M1 static + anytime (`ref_lih_r1p6_fixed`), trajectories of our methods, the baselines, reuse rows (`ref_lih_r1p6_traj`) and the fixed-budget selection (`ref_fixed_lih`).  The CISD state of LiH 1.6 is nearly the ground state: max|g| 7.8e-4, top gap 5.3e-5, not run.  BeH2 (14 qubits, 204 generators; exact ADAPT 10 steps): fixed states ADAPT2 (11%), ADAPT4 (8.6%), ADAPT7 (24%), ADAPT9 (51%) with the a priori family (`ref_beh2_fixed_a/b`).
- **A10, Huang and Izmaylov reproduction**: `scripts/reproduce_huang_izmaylov.py`.  Read from their paper (arXiv:2509.14917v1, PDF fetched and read): STO-3G, linear geometries with 1 A between adjacent atoms, HF reference, UCCSD / qubit / qubit-excitation pools, QWC fragments by sorted insertion, exact fragment variances with a Gaussian model, naive baseline eps = 1e-3 (0.5e-3 for the qubit pool) per fragment (M_n = Var_n / eps^2), SE with eps'_r = (5 - 2r/5) eps (UCCSD, QE) or (2 - r/10) eps (qubit), at most 10 rounds, R_r = 8 eps'_r, elimination |g_i| + R_r < max|g| - R_r; Table I = percentage reduction of the total measurements for selection to chemical accuracy (1.59 mHa).  Published UCCSD row: H4 93.0%, LiH 80.8%, BeH2 90.0%.  Our reproduction (UCCSD pool; qubit and QE pools not run: our trajectory driver is restricted to the particle-number sector): H4 chain 93.0% (10 seeds; 92.7% in a 3-seed test).  Sample reuse across rounds is not specified in the paper: both variants are run (`--no-reuse`).  `cluster/local_reproduce_hi.sh`; results in `runs/paper_a/huang_izmaylov_reproduction_*.csv`.
- **A11, sampled optimiser**: `scripts/paper_a_sampled_optimizer.py` (ExcitationSolve-style coordinate updates, five noisy energies per parameter, trigonometric fit of degree two valid for generators with spectrum {0, +-i}); cost = energy evaluations x M_E (1 mHa / eps_E)^2 against the model C_opt (r = 2, 4); output `sampled_optimizer.csv`, `sampled_optimizer_summary.csv`.  H4 1.0, 2 seeds: 3.6e8 shots at eps_E = 1 mHa (658 energies), 4.6e9 at 0.3 mHa; to be compared with the model once the full run (10 seeds, H4 and LiH, workstation log `cluster/logs/sampled_optimizer.log`) is in.
- **Job files** (`cluster/jobs/make_referee_jobs.sh`, git-ignored): the fixed-budget lines first failed because Python on Windows ends printed lines with CR LF (a CR inside the method names); fixed by `fixed_lines` (`tr -d '\r'`), the affected jobs were resubmitted.  Submitted on 2026-10-09 (Trillium): ref_small_fixed 2528529, ref_traj_small 2528530, ref_h2o_{eq,stretch}_{bound,anytime} 2528521/2/4/5, ref_lih_r1p6_fixed 2528527, ref_lih_r1p6_traj 2528528, ref_h4_snapshots 2528573, ref_scaling_n2 2528578, ref_beh2_fixed_a/b 2528583/8, ref_fixed_lih 2528619, ref_h2o_{eq,stretch}_fixed_budget 2528620/1 (the first fixed-budget lines of ref_traj_small and ref_lih_r1p6_traj fail on the CR and are covered by ref_fixed_lih).
- **A11 result (10 seeds, `cluster/logs/sampled_optimizer.log`).** Sampled ExcitationSolve optimisation along the exact ADAPT trajectory, shots against the model C_opt (r = 2 / r = 4): H4 1.0: eps_E = 1 mHa 3.7e8 shots (672 energies; x0.23 / x0.12 of the model), 0.3 mHa 4.6e9 (x2.9 / x1.5), 0.1 mHa 5.0e10 (x31 / x16), chemical accuracy reached in 100% at every eps_E; LiH 3.0: 1 mHa 3.1e8 (x0.46 / x0.24, chemical accuracy in only 50% of the runs), 0.3 mHa 4.2e9 (x6.3 / x3.3, 70%), 0.1 mHa 4.8e10 (x71 / x38, 100%).  So the model's C_opt is optimistic once the optimiser must reliably reach chemical accuracy under sampling noise (the noise floor needs eps_E well below 1 mHa); the Q7 conclusion (optimisation dominates selection) gets stronger, not weaker.  Final energies of the sampled runs are within 0.2 mHa of BFGS's.
- **2026-10-09 (night): four jobs in state FAILED, cause and audit.**  `ref_h2o_eq_fixed_budget` (2528523), `ref_h2o_stretch_fixed_budget` (2528526), `ref_traj_small` (2528530) and `ref_lih_r1p6_traj` (2528528) ended with exit code 1 only because of the carriage-return bug in their fixed-budget lines (`invalid choice: 'Fixed ..., uniform'`); every other line of the four jobs finished (H4 and LiH rho sweeps, LiH 1.6 baselines and reuse trajectories).  The fixed-budget lines had already been rerun in `ref_fixed_lih` (2528619) and the two H2O jobs (2528620/1), all COMPLETED; no relaunch needed.  `cluster/audit_referee.py` (new, git-ignored) checks every requested configuration of every `ref_*` job file against the merged local tables: after the sync, everything is complete except what the six running jobs have not reached yet (BeH2 `II-A + reuse` and `M1 static`; H2O `M1 seq`, `M2`, reuse rows and II-A anytime on H2O eq CISD; all inside their 24 h with `--resume` requeue).  First results (a priori sign-aware family, 95% bootstrap intervals): II-A / II-0 0.54 (LiH HF) ... 0.11 (H2O eq CISD, but II-A only 95% correct there), and two states where II-A is dearer than II-0 with the a priori start: LiH 3.0 ADAPT5 (3.9x, 99% correct) and LiH 1.6 ADAPT4 (1.09x); anytime costs x1.4-2.5 of per-round and removes the missed intervals except on the near-tied states (LiH ADAPT5 52%, LiH 1.6 ADAPT4 98% still missed).

## 2026-10-10: larger-basis cases (H2O cc-pVDZ restricted window, 6-31G)
Context: a referee point on scale; the colleagues' note (`larger_basis_full_report.ipynb`, modules written by its `%%writefile` cells) defines orbital *windows* of H2O cc-pVDZ and N2 (core frozen, all other occupied orbitals kept, a fixed number of the lowest virtuals per irrep, later geometries by irrep-resolved overlap; Hamiltonian = CASCI effective Hamiltonian of the window).  Used here only for the windows (their gradient-cost code is Part I's oracle model, not our shot-level pipeline).
- **Code.**  `src/cases_window.py` (new): `WindowSpec` (a `CaseSpec` subclass), the window construction (their selection logic, rewritten), cases `H2O_{dz8o,dz11o,631g}_{eq,str}_{HF,CISD}` (R(OH) 0.9584 and 1.75 A, angle 104.45 as in the note), and `install()`, called from `part1_bridge`, which makes Part I's `chemistry.build_qubit_hamiltonian` / `reference_fci_energy` understand them (Part I's files untouched; the validation gate runs on them).  Energies against the note's archive: RHF and window FCI agree to 0.05 uHa for the 8-orbital window at both geometries (`tests/test_cases_window.py`).  `build_problem_parallel` (same file) expands the K commutators on many processes and writes the problem under Part I's cache key; result identical to the serial build (gradients to 2e-15, commutator terms and contexts equal; 14 workers: 418 s against 905 s serial on a workstation; the parent grouping, ~110 s, stays serial).  `cluster/build_window_case.py` is the command line.  `scripts/paper_a_scaling.py --structure-only` stops after the context library and the II-0 / II-A coordinates (no sigmas, no refit) for cases the simulator cannot hold (a 2^n outcome distribution per context).
- **Sizes** (HF state; H2O STO-3G in brackets): dz8o: 16 qubits, 8 electrons, K = 360 [140], 1,905 Hamiltonian terms [1,086], 220,355 Pauli products [81,566], 5,408 contexts [2,366], 44 CZ per circuit [32], 13,643 II-A coordinates per generator [7,483]; library 181 s, II-A coordinates 30 s, one refit 0.16 s per generator (about 1 min for all, as for STO-3G); peak 8 GB.  The dz11o (22 qubits, K 1,092, 6,078 terms) and 6-31G (24 qubits, K 1,424, 8,921 terms) problems need 10x and 19x the commutator work of dz8o.  N2 (20 qubits, K 609, from `ref_scaling_n2`): 740,431 products, 14,931 contexts, 72 CZ, 19,458 coordinates per generator, refit 0.54 s per generator (5.5 min for all), serial problem build 30,124 s (8.4 h).
- **Workstation**: dz8o eq HF and eq CISD problems built (CISD: E = -76.05888, max|g| 2.2e-3; the window FCI is -76.05928, so the CISD state is within 0.4 mHa of FCI).
- **Cluster jobs** (job files `cluster/jobs/make_window_jobs.sh` -> `win_*.txt`, `cluster/submit_window.sh`): submitted 2026-10-10: `win_prep` 2532448 (exact ADAPT trajectories of the HF cases, which also build the HF problems; stretched HF and CISD problems; parallel builder), `win_sizing` 2532449 (two trials of each a priori method on eq CISD with `/usr/bin/time -v`: seconds and peak memory per trial), `win_scaling_dz11o` 2532450 and `win_scaling_631g` 2532451 (parallel build + structure-only).  To follow: fixed states (CISD and one ADAPT snapshot per geometry; II-A, II-0, M1 seq, M2, anytime rows) and trajectories (II-0, M1 seq, M2, fixed budget, II-A in 24 h stages) once the sizing is known.
- All referee jobs of 2026-10-09 are COMPLETED and synced (`cluster/audit_referee.py`: 0 requested items missing); only the II-E chain (2507744-6) is still running.
- **2026-10-10: completion check of the A1-A12 runs.**  `sacct`: every `ref_*` job COMPLETED except the four known CR failures (their lines were rerun and are COMPLETED); error scan of all line logs finds only that `invalid choice: 'Fixed ..., uniform\r'` message.  Synced (`sync_from_trillium.sh`, plus `runs/paper_a_n2` by rsync); `cluster/audit_referee.py`: 0 requested items missing.  Local: full test suite 145 passed (`cluster/logs/tests_referee.log`); aggregators rerun (sota tables, bootstrap, fixed budget, rule audit, referee tables, figures) and `sanity_checks.py` on the nine fixed states: 0 FAIL.  Lost with the workstation session and resubmitted: the Huang-Izmaylov no-reuse run (`ref_hi_noreuse` 2532519, outputs `runs/paper_a_hi/{H4,LiH,BeH2}`; with reuse all three done: 93.0 / 89.5 +- 6.5 / 93.1 against published 93.0 / 80.8 / 90.0; the partial no-reuse log gave 91.3 and 82.2 for H4 and LiH).  Bootstrap flags (relative half-width > 15%): trajectory ratios on H4 stretch (48 trajectories) and LiH 3.0 (30), the H2O baseline ratios against II-A and LiH 3.0 ADAPT5 a priori (3.9 [1.3, 8.0], 100 trials); to extend if the manuscript quotes them.  N2 20 qubits: refit 0.54 s per generator (5.5 min for all 609), peak 250 GB, serial problem build 8.4 h.
- **2026-10-10: A1-A11 results written into `reports/draft.tex` (and built to `reports/draft.pdf`, 31 pages, 0 LaTeX errors, 0 undefined references).**  New subsections Q9 (a priori start, sign-aware rule, bootstrap intervals, anytime intervals; Tables apriori, headline CI, anytime), Q10 (fixed-budget selection, certification premium; Table, Fig.), Q11 (thresholds, rho, scale; Tables), Q12 (reproduction of Huang-Izmaylov with samples kept; sampled optimizer against C_opt; Tables), inserted between Q8 and the Discussion between `% BEGIN/END REFEREE RESULTS` markers; the five figures of `paper_a_figures.py` placed (pipeline in Methods, rounds in Q5, trajectory before Q7, depth after the depth table, fixed budget in Q10).  Edited sentences: abstract (two sentences), introduction question list, benchmark-settings paragraph (thresholds now varied), Q8 takeaway (oracle start qualifier), Conclusions (main result, new two-check sentence, scope).  Generated tables read from `reports/generated/tab_*.tex` (scripts: `paper_a_referee_tables.py`, `paper_a_bootstrap.py --latex`, `paper_a_fixed_budget.py --latex`); generators fixed to print `--` instead of `nan`, readable system names, eta in the optimizer header, stretched H4 left out of the rho-trajectory table (stalls by symmetry), no-reuse column of the reproduction table only once that run exists.  Not in the draft: the Huang-Izmaylov no-reuse variant (job 2532519), the 16/22/24-qubit window runs, the headline reframe (Part B).  The pre-patch draft is kept in the session scratchpad (`draft_before_referee.tex`).
- **2026-10-10: `reports/draft.tex` made self-contained (Overleaf).**  Its only dependency is now `figures/*.pdf`: the nine generated tables of Q9-Q12 (`reports/generated/tab_*.tex`) are pasted into the draft between `% BEGIN INLINE <name>` / `% END INLINE <name>` markers and the bibliography is the BibTeX output (`draft.bbl`) between `% BEGIN INLINE BIBLIOGRAPHY` markers (no `\input`, no `.bib`).  `scripts/paper_a_inline.py` refreshes the blocks after a generator has rewritten a table; `--bbl <file>` pastes a new bibliography.  Checked by compiling a directory that holds only `draft.tex` and `figures/` (31 pages, 0 errors, 0 undefined references).  To sync to Overleaf: `reports/draft.tex` and `reports/figures/`.

## 2026-10-10: second round of referee runs (todo list after the first write-up)
Context: after the A1-A11 results were written into the draft, five open items remained: (1) why intervals fail on the hard states, (2) a tighter fixed-budget comparison, (3) more trials where the bootstrap is wide, (4) pending jobs, (5) aggregator fixes.  Job generator `cluster/jobs/make_referee2_jobs.sh` (git-ignored, with `cluster/submit_referee2.sh`), 28 jobs submitted 2026-10-10 ~11:40.
- **Code.**  `LearningConfig.diagnose` (not part of the label) and `LearnedM3._diagnose_miss`: for every round with a missed interval, the standardized error against the *true* sd of the estimator (the exact covariance of the cross-fitted estimator, `_covariance_matrix(exact=True)`), the ratio estimated/true sd, the shots in the thinnest context, zero-width intervals, whether the best arm was missed; trial-file fields `miss_*` (`tests/test_learning.py::test_miss_diagnostics...`).  Step 5 configs `<II-0|II-A>, safe, bound start[, anytime], min shots=<50..1000>, diagnose`.  `FixedBudgetSpec(allocation="pilot")` and `FixedBudgetPilotM1` (a fifth of the budget uniform, fragment sds estimated from it through `FragmentProblem.update_blocks`, minimax allocation of the rest, all shots in the estimate; no exact variance; tests in `tests/test_fixed_budget.py`); phase4 budgets on an eighth-decade grid (10 ... 1e9) for designed, uniform and pilot.  `paper_a_fixed_budget.py`: reach criterion relative to exact ADAPT's final error (+0.1 mHa) when the pool stalls (stretched H4), duplicates resolved by the run with most trajectories, certified runs pooled with the extra trajectories (seed 8, tag `extra1`), new table of the premium against the tolerance rho (`tab_fixed_rho`).  `paper_a_sota_tables.py`, `paper_a_bootstrap.py` pool the `extra` trajectory files.  `paper_a_referee_tables.py`: interval-diagnostics table, correctness column in the threshold table, one merged structure table (`runs/paper_a/scaling_all.csv`: STO-3G cases, N2 and the cc-pVDZ windows), no-reuse column of the reproduction table.  `cases_window.py`: above 18 qubits the Part I gate checks only the HF energy (the sector diagonalisation needed more than 750 GB at 22-24 qubits: jobs 2532450/1 died of out-of-memory).
- **Item 1: why the intervals miss** (LiH 3.0 ADAPT5, LiH 1.6 ADAPT4: 100 trials per row; H2O eq CISD: 40).  II-A at the default 50 shots: missed 52% / 94% of the trials, estimated sd 0.39 / 0.16 times the true sd (median over missed trials), best arm among the missed in 65% / 80%, shots in the thinnest context 50; correct 99% / 100%, mean shots 2.62e8 / 2.27e8.  Minimum 200: missed 6% / 9%, sd ratio 1.00 / 0.98, 4.67e7 / 5.56e7 shots (5.6x / 4.1x cheaper), below II-0 (7.1e7 / 2.3e8); 400: 5% / 3%, 3.87e7 / 5.19e7; 1000: 3% / 4%.  II-0 has sd ratio 1.00 at every minimum and still misses in 41% / 32% / 75% of the trials (LiH5 / LiH1.6 / H2O): the misses of the per-round intervals are the 60-90 looks of one selection.  Anytime intervals: II-0 and II-A with minimum >= 200: no miss in LiH5 and H2O, 1 of 100 on LiH 1.6 ADAPT4 (II-A at 200); at 50 shots II-A anytime still misses in 52% / 98% (sd ratio 0.23 / 0.09).  Price of anytime ~2x (II-A at 200: 2.2x LiH5, 2.3x LiH1.6; 1000 on H2O: 1.95x); II-A anytime 1.5-4.9 times below II-0 anytime.  H2O eq CISD, II-A: 100% correct from 200 shots (40 trials), 4.6e9 against 2.1e10 for II-0; its per-round misses 30-38% have sd ratio 1.0 (repeated looks).  So the three states on which II-A lost to II-0 lose because of the 50-shot threshold, not the design; H4 ADAPT5 was not retested.
- **Item 2: fixed budget.**  Eighth-decade grid and more trajectories (H2O 36, LiH 60, H4 96): cheapest adequate designed budget H4 1.0 1.3e4/selection (total 1.5e5; premium of certified II-A 7.2, pilot 7.9), H4 2.0 1.3e5 (1.5e6; 6.4, pilot 5.3; stalled pool scored against exact ADAPT's final energy), LiH 1.6 3.2e6 (1.6e7; 2.1, pilot 1.6), LiH 3.0 7.5e6 (5.2e7; 1.0/1.0), H2O eq 1.3e8 (2.3e9; 0.5/0.4), H2O str 4.2e8 (7.6e9; 0.4/0.3, pilot budget 5.6e8).  Pilot allocation costs 0.9-1.4x the designed one.  Premium against rho (certified II-A / fixed total): H4 1.0 25, 7.2, 2.6, 1.8 at rho 0.05, 0.1, 0.2, 0.3; H4 2.0 25, 6.4, 2.0, 1.07; LiH 1.6 3.7, 2.1, 1.3, 1.0; LiH 3.0 1.3, 0.95, 0.43, 0.29; H2O eq 0.47, 0.31, 0.26 (0.1, 0.2, 0.3); H2O stretched rho 0.2/0.3 running.  The fixed budget does not certify: its selections are rho-good in 76-95% of the cases at the adequate budgets, against >= 98.7% for the certified II-0/II-A.
- **Item 3: more trials.**  LiH 3.0 ADAPT5, a priori start, 400 trials: II-0/II-A 0.42 [0.29, 0.62] (was 0.25 [0.12, 0.78] with 100).  Extra trajectories (seed 8, tag `extra1`, pooled): H4 48+48, LiH 3.0 +60, LiH 1.6 +30 done; H2O (24 more per geometry, II-A and II-A + reuse in staged jobs) running.
- **Item 4: pending jobs.**  Huang-Izmaylov without reuse (10 seeds): H4 91.8 +- 2.0 (published 93.0), LiH 80.7 +- 7.8 (80.8), BeH2 90.3 +- 4.9 (90.0); with reuse 93.0 / 89.5 / 93.1.  Window runs: exact ADAPT of the 16-qubit cc-pVDZ window is at 5.9 mHa (eq) and 3.7 mHa (stretched) after the 40 steps allowed, gradients > 3e-2: no measured trajectories for it; sizing on the eq CISD state: II-0 with exact identification costs 5.9e12 shots, about 2 h per trial (fixed states of this window need rho > 0).  dz11o problem (22 qubits, K 1092, 1.6e6 products, 24,753 contexts) built in 4.9 h with 160 workers (the commutator expansion took 3.8 h, against 2 min for the 16-qubit window); 631g (24 qubits) still building.
- **Draft.**  Q9-Q12 rewritten with these results, table of interval diagnostics, premium-against-rho table, Huang-Izmaylov with both variants; abstract, conclusions and Q8 takeaway adjusted; `reports/draft.tex` made self-contained (see above).

## 2026-10-11: draft refreshed with the finished jobs; 16-qubit fixed states submitted
- Synced and aggregated: stretched-H2O certified II-0 / II-A at rho 0.2 and 0.3 (II-A 2.4e9 and 2.0e9 against 7.6e9 for the cheapest adequate fixed budget: premium 0.32 and 0.27; II-A/II-0 0.32 and 0.30), `ref2_diag_h2o` rows so far (II-A 200/400/1000 shots and anytime 1000, II-0 50/200/1000 and anytime), the sizing run of the 16-qubit window (eq CISD, exact identification, 2 trials: II-A 1.3e9 shots, 13 h and about 32 GB per trial; II-0 5.9e12; M1 seq 1.3e14; M2 6.1e10 with 0% correct; the leading gradients are nearly tied), H2O extra trajectories finished so far (eq: II-A, II-A + reuse, II-0 + reuse, M2; stretched: M2).
- Submitted (2536096-101, `cluster/jobs/make_window_jobs.sh`, ADAPT_EQ=10 ADAPT_STR=10): fixed states of the 16-qubit window at rho = 0.1, 16 trials: II-A on eq CISD, eq ADAPT10, str CISD, str ADAPT10 (one node each, 16 workers, about 13 h), and II-0 / M1 seq / M2 on both geometries (one node each).
- Still running: `ref2_diag_h2o` (remaining rows), `ref2_more_traj_h2o_{eq,stretch}` chains (24 extra trajectories each), `win_scaling_dz11o` (structure run) and `win_scaling_631g` (problem still building after 13 h), II-E chain s6/s7.
- Draft: premium table complete for H2O; sizing paragraph in Q11 with a \note marking the pending fixed-state results.
