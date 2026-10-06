# Part II development log

Newest entries last. One entry per code change or run: what changed, why, files, how it was checked,
and (for runs) where it ran. Job output files (`*.log`, `*.out`) in this directory are git-ignored;
this file is tracked.

## 2026-10-05 — SOTA-baseline work started

Trigger: gap analysis of Paper A against `reports/adapt_vqe_state_of_art_2026_with_benchmarks.tex`
(see memory `paper-a-sota-baselines-todo`). Planned items: shot-level M1/M2 baselines, Ikhtiarudin-style
reuse baseline, other pools (qubit-ADAPT first), AIM-style IC baseline or scope note, C_opt sensitivity,
related-work text.

- Baseline check before any change: `pytest tests` -> 67 passed (WSL venv `/root/venvs/quasisymmetry`,
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
