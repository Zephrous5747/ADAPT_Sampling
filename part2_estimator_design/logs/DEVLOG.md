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
