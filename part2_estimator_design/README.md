# Part II: universal measurable estimators — Steps 0–4, Phases 2–4

This package implements Steps 0–4 and Phases 2–4 of the implementation plan for
`reports/part2_rewritten_universal_estimator_design.tex`. All planned runs are
complete, including the LiH trajectory-state and H₂O finite-shot runs done on
Trillium, and the round of 19 October 2026 (confidence level at BFGS 1e-5, BFGS on
shot-noise energies, the CAS(8e,8o) active space along its exact path). The results feed
Paper A, which is written for the Journal of Chemical Theory and Computation:
- `reports/draft.tex` is the main text (self-contained: generated tables and the
  bibliography are pasted in; it needs only `figures/`);
- `reports/supplement.tex` is the Supporting Information, generated from the same
  stored runs by `scripts/paper_a_make_si.py` (see "Paper A: tables and figures");
- `reports/energy_trajectory_findings.tex` is the working report of the
  energy-trajectory study (policies, optimizer tolerance, confidence level, noise).

The `reports/` folder is kept locally and is not tracked by git (see `.gitignore`).

| Step | What | Where |
|---|---|---|
| 0 | Symplectic Pauli algebra, measurement-circuit synthesis, group moments, shot-level sampler, and their gates | `src/symplectic.py`, `src/clifford.py`, `src/sampler.py`, `scripts/step0_validate.py` |
| 1 | II-0 baseline as a shot-level online experiment (pairwise rule, top-up allocation, fine schedule) | `src/online.py`, `scripts/step1_online_baseline.py` |
| 2 | Overlap census: which Paulis are measured by more than one context | `src/contexts.py`, `scripts/step2_overlap_census.py` |
| 3 | Oracle ceiling of coefficient splitting (II-A), zero-sum ghosts (II-B) and per-breakpoint redesign (oracle II-D) | `src/design.py`, `src/allocation.py`, `src/planning.py`, `scripts/step3_oracle_ceiling.py`, `scripts/step3_prior_transfer.py` |
| 4 | II-A with covariances learned from the shots (shrinkage prior, cross-fitting, guard, estimated radii) | `src/learning.py`, `scripts/step4_learned_designs.py` |
| 4+ | Sign-aware elimination (absolute intervals before signs are resolved), contrast objective with directly designed contrast estimators (`U_BAI`, II-E), `rho`-good stopping, non-oracle start | `src/rules.py`, `src/learning.py` |
| Phase 2 | Exact ADAPT-VQE trajectories; every intermediate state saved as a case `<geometry>_ADAPT<k>` | `src/trajectory.py`, `scripts/phase2_adapt_states.py` |
| Phase 3 | Complete, fully logged selection decisions: the work order's required outputs (i)–(xi) | `src/runlog.py`, `scripts/phase3_selection_record.py` |
| Phase 4 | ADAPT trajectories whose selections are measured; cumulative selection cost to chemical accuracy | `scripts/phase4_adapt_selection.py` |
| Paper A | Shot reuse and run evolution; selection versus optimisation cost along trajectories; policies with injected selection errors; confidence level; noisy optimizer; tables and figures of the paper | `scripts/paper_a_*.py`, `scripts/energy_*.py`, `scripts/features_*.py` (see below) |

Results and their reading are in `RESULTS.md`. In brief:
- **Learned coefficient splitting (II-A):** costs 35–57% less than the shot-level
  II-0 baseline on H₄, LiH and H₂O, and 30–58% less on ADAPT trajectory states.
  It needs no oracle input, and all selections on the fixed CISD/HF states were
  correct.
- **Measured ADAPT:** reproduces exact ADAPT. Learned designs cut the trajectory's
  selection cost by 41–44% (H₄) and 42–60% (LiH).
- **After compression:** the repeated parameter reoptimisation dominates the total
  measurement cost.
- **Open issue:** the non-oracle starting radius is not yet reliable on the states
  with the smallest gradients (92% correct on H₂O eq CISD).

**Status (Oct 4, 2026).** All Step 4, Phase 2–4 and Paper A runs are complete and
synchronised, including the oracle cells that were missing from Paper A Table V
(`cluster/jobs/table5.txt`, Trillium job 2485445, 3.6 h on one node):
- static II-B on the four H₂O states;
- per-breakpoint redesign (oracle II-D, top-up accounting) of II-A and II-B on
  LiH and H₂O.

The per-case outputs are in `runs_table5/IIA` and `runs_table5/IIB`, and the rows
are merged in `runs/step3_oracle_ceiling_table5.csv`. Findings:
- II-B adds nothing resolvable on LiH and H₂O.
- Redesign adds 15–16 points on H₂O stretched HF and 1–3 points on LiH and the
  other H₂O states.
- M3 actual varies by up to 11 points between independent solves of the same
  design problem (the solver minimises the M3 bound), so the LiH and H₂O rows of
  Table V were all taken from the new runs.

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

## Paper A: tables and figures

Every table and figure of the main text and of the Supporting Information is
written by a script from the stored runs in `runs/`. The
Supporting Information (Sec. S7, table "Scripts and stored runs behind the tables
and figures") lists which script reads which runs. In order:

```bash
cd part2_estimator_design
python scripts/paper_a_results_tables.py       # Tables 1-3 (systems, planning bounds, learning)
python scripts/paper_a_minshots_headline.py    # Table 4 (headline comparison, 21 states)
python scripts/paper_a_trajectory_results.py   # Tables 5, 6 and Fig. 4
python scripts/paper_a_figures.py              # Figs. 1-3 and SI figures
python scripts/paper_a_totalcost_figure.py     # Fig. 5
python scripts/paper_a_si_tables.py            # planning and learning tables of the SI
python scripts/window_selection_share.py --policies runs/energy_policies_nrho_gtol1e-5 --out runs/window_share_gtol1e-5.csv   # cluster job: modeled share on the CAS(8e,8o) active space (also gtol1e-9)
python scripts/paper_a_window_path_table.py    # certified II-A and its share along the active-space path
python scripts/paper_a_eta_shares.py           # share of the selection for 2, 4, 8 energies per parameter, with bootstrap intervals (SI)
python scripts/paper_a_nmin_table.py           # headline ratios at 50 and 400 shots per context and fold, 21 states (SI)
python scripts/paper_a_window_rho_ci.py        # intervals of the tolerance effect on the CAS(8e,8o) energy (SI)
python scripts/paper_a_ladder_table.py         # floor, learned design and oracle ceiling at 50 and 400 shots (SI)
python scripts/paper_a_cas400_summary.py       # CAS(8e,8o) fixed states at 400 and 50 shots (SI, main text); step5_merge_summary.py merges the summary rows
python scripts/paper_a_coverage_summary.py     # coverage test on states with a set gap (coverage_gap_states.py writes the states): SI table and cluster agreement
python scripts/paper_a_critique_tables.py      # tail of the cost of a selection and predictors of the saving (needs the cached problems: WSL or cluster)
python scripts/paper_a_breakeven.py            # break-even device shot rate, classical time against device time
python scripts/paper_a_minshots_headline.py --rho 0.1   # headline at rho = 0.1 (jobs rho01_headline_*)
python scripts/paper_a_ablations_400.py        # cross-fitting, gamma, reuse at 400 shots (jobs crit_*)
python scripts/paper_a_round4_tables.py        # SI tables of these runs and of the Yen et al. reproduction
python scripts/paper_a_q129_summary.py         # grouping by mass (runs_group_mass/), trajectories at 400 shots, other pools at 400 shots
python scripts/paper_a_grouping_tables.py       # covariance-aware grouping (runs_group_vaware_hf/), oracle optimum of the groupings, mass-grouping trajectories (job logs), scale of the first radius
python scripts/paper_a_circuits_q7.py          # circuit configurations loaded per selection (runs labelled ", circuits")
python scripts/reproduce_yen_splitting.py      # Hamiltonian-variance tables of Yen et al. with our splitting code
python scripts/paper_a_baselines_table.py      # every method against II-0 on nine fixed states (SI)
python scripts/paper_a_trajectory_counts.py    # what the shot-limited selections did to the ADAPT trajectories (SI, main text)
python scripts/paper_a_si_numbers.py           # check: every number of the hand-written SI parts occurs in its sources
python scripts/energy_findings_tables.py       # tables of the findings report and the SI
python scripts/paper_a_inline.py               # paste the generated tables into reports/draft.tex
python scripts/paper_a_make_si.py              # build reports/supplement.tex
python scripts/paper_a_si_refs.py --aux <supplement.aux>   # numbers of SI sections cited in the draft
```

Coverage of the selection on states with a set gap (not run yet): `cluster/jobs/coverage_gap.txt` writes the states with
`scripts/coverage_gap_states.py` (a template ADAPT state rotated to a target relative gap) and runs the interval diagnostics of
`step5_external_baselines.py` on them (seed 13); `scripts/paper_a_coverage_table.py` tabulates wrong selections with their exact
upper bound and the share of trials with a missed interval.

The other generators (`paper_a_referee_tables.py`, `paper_a_sota_tables.py`,
`paper_a_sota_latex.py`, `paper_a_bootstrap.py`, `paper_a_fixed_budget.py`,
`paper_a_depth_tables.py`, `paper_a_discussion_tables.py`) are listed in the SI
table and in `scripts/make_archive.py`. `scripts/make_archive.py` packs code, stored
outputs, a manifest and SHA-256 sums for the data availability statement.

Random numbers: trial `i` of a run with seed `s` uses
`SeedSequence(s).spawn(n)[i]`; the seed is stored in each run's meta file.
Two builds were used: the Trillium cluster (Python 3.11.4, NumPy 2.4.2, SciPy
1.17.1) and the WSL workstation (below). Optimizer evaluation counts are compared
only within one build.

## Running

The code needs NumPy ≥ 2, SciPy, and PySCF + OpenFermion for building problems.
It was run in WSL Ubuntu with Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
PySCF 2.14.0 and OpenFermion 1.8.1.

```bash
cd part2_estimator_design
python scripts/build_problems.py                      # all nine cases, checked against Part I
python -m pytest tests -q                             # 159 tests (conftest pins BLAS to one thread, see DEVLOG 2026-10-18)
python scripts/check_context_moments.py LiH_R1p6_ADAPT3 BeH2_ADAPT4 H2O_eq_ADAPT11 H2O_dz8o_eq_ADAPT10   # every context of production libraries against the state-vector evaluator (Linux/WSL)
python scripts/check_symmetries.py H4_square_eq_side1p0_HF LiH_R3p0_HF   # pool commutes with N and S_z; full-space recomputation of energies and all gradients along exact ADAPT; <S^2> (Linux/WSL)
python scripts/step0_validate.py --low-shot 0.001     # sampler gate
python scripts/step1_online_baseline.py --trials 250  # II-0 baseline, H4 + LiH
python scripts/step2_overlap_census.py                # overlap census, all nine cases
python scripts/step3_oracle_ceiling.py --cases H4_square_eq_side1p0_HF --redesign
python scripts/step3_oracle_ceiling.py --cases LiH_R3p0_HF --strategies mass --levels II-0 II-A        --redesign --redesign-accounting topup --out runs_table5/IIA   # top-up only: half the cost
python scripts/step4_learned_designs.py --trials 200  # learned II-A, H4 CISD rows
python scripts/phase2_adapt_states.py                 # ADAPT states of H4 x2 and LiH (add H2O_eq_HF ...)
python scripts/step4_learned_designs.py --cases LiH_R3p0_ADAPT3 --configs "II-A data, contrast"
python scripts/phase3_selection_record.py --cases LiH_R3p0_ADAPT3 --trials 3
python scripts/phase4_adapt_selection.py --cases H4_square_eq_side1p0_HF --trajectories 50 --workers 12
python scripts/paper_a_run_diagnostics.py --cases H4_square_eq_side1p0_CISD LiH_R3p0_HF
python scripts/paper_a_cost_decomposition.py      # H4 x2 and LiH; reads the Phase 4 outputs
python scripts/step5_external_baselines.py --help # published strategies at fixed states (options in the header)
python scripts/energy_trajectory_policies.py --help # injected selection errors, --gtol, --noise-eps
```

`paper_a_cost_decomposition.py` rebuilds the Hamiltonian with a fresh SCF. In
square H₄ and in LiH's π shell the orbitals are degenerate, so independent builds
can expand H in different Pauli products. That changes the energy groups and the
optimiser's evaluation counts, and C_opt moves by up to ±25% between builds. The
write-ups quote the range over two builds.

### Trajectory cases

`scripts/phase2_adapt_states.py` runs exact ADAPT-VQE from an HF case and saves
every state to `.cache/trajectory/`. The state after `k` generators is then the
case `<geometry>_ADAPT<k>` (for example `LiH_R3p0_ADAPT3`) for every script.
Its gradient problem is built by Part I's code from the same Hamiltonian object
the trajectory used, and it is saved with the states. A fresh SCF can return
orbitals of opposite sign to the cached HF problem, which flips some gradient
signs (seen on H4), so the cached problem is not reused.

### Parallel and cluster runs

Steps 1 and 4 run their trials in worker processes. Set `OMP_NUM_THREADS=1` and
pass `--workers N`:

```bash
OMP_NUM_THREADS=1 python scripts/step4_learned_designs.py --cases LiH_R3p0_HF --trials 100 --workers 10
```

Each trial has its own random stream, so the result does not depend on the
number of workers. To spread one run over several machines or cluster jobs, give
each job `--shard K/N`, then merge:

```bash
python scripts/merge_trials.py --step step4 --cases LiH_R3p0_HF
```

The merge checks that every trial is present exactly once. Memory is about
1.5 GB per worker on LiH and about 3.6 GB per process on H₂O.

On Trillium (SciNet) the runs use `cluster/trillium_run.sbatch`: one whole node
per job file in `cluster/jobs/`, whose lines run concurrently with the project
venv (`module load StdEnv/2023 gcc/12.3 python/3.12.4`, then
`pip install numpy scipy pyscf openfermion pytest` from the Alliance wheelhouse,
recorded in `requirements_trillium.txt` on the cluster). The cached problems and
trajectory states in `.cache/` are copied with the code, so no SCF is rerun.
Sizing on Trillium:
- One H₂O II-A trial takes about 6 min and 6.6 GB.
- II-0 with the bound start takes about 15 min and 3.6 GB.
- A learned contrast trial takes 1.4–5.5 h of design time.
- H₂O jobs run 60 workers per node (memory-bound); LiH jobs run about 190.

The full run used two LiH node-jobs (2.2 and 5.1 h) and eight H₂O node-jobs
(11–17.5 h), about 110 node-hours.

Checkpoints and requeueing:
- Every trial and trajectory is checkpointed as it finishes.
- `--resume` skips finished configurations and trials.
- A job requeues itself 15 min before the 24 h limit, at most twice.

```bash
# once, in your own terminal: open an SSH master connection and approve MFA
ssh -fN -o ControlMaster=yes -o ControlPersist=12h -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" USER@trillium.scinet.utoronto.ca
bash cluster/sync_to_trillium.sh USER         # code, job files, .cache
# on Trillium, in /scratch/USER/adapt_sampling/part2_estimator_design:
sbatch --job-name=lih_a cluster/trillium_run.sbatch cluster/jobs/lih_a.txt
bash cluster/jobs/make_h2o_jobs.sh 60         # writes the sharded H2O job files
sbatch --job-name=table5 cluster/trillium_run.sbatch cluster/jobs/table5.txt   # Table V oracle cells
# back on the workstation, when the jobs are done:
bash cluster/sync_from_trillium.sh USER python   # trial files, Phase 3/4, logs; rebuilds all tables
```

`sync_from_trillium.sh` never deletes local files. It rebuilds every Step 4 table
from all trial files, local and cluster (`merge_trials.py`, which checks that each
trial is present exactly once), and then runs `aggregate_results.py`.

The environment is pinned in `requirements.txt`. `Dockerfile` builds it from the
repository root; it has not been tested, since no Docker was available on the
development machine.

Every script writes CSV/JSON under `runs/<case>/`, with interpreter, package
versions and git commit in a `*_meta.json`. Step 4 rebuilds a case's table from
all of its per-trial files, so runs of different configurations accumulate. Step 3 saves the static design of
every level as the sparse pair `B`, `C` (`*_B.npz`, `*_C.npz`) and hard-fails if
`max |A - BC|` exceeds `1e-9 max |A|`.

## What is oracle, and what is not

- **Steps 2 and 3 are oracle throughout.** Exact covariances drive the designs,
  and exact gradients set the elimination trajectory. Their numbers are ceilings,
  not predictions.
- **Step 1 samples measurement outcomes.** Every elimination is decided from
  sampled data. The confidence radii still use oracle variances, as in Part I;
  learning them is Step 4.
- **The starting radius and the floor are oracle quantities** by default. They
  are taken from `max_i |g_i|`, as in Part I. `start="bound"` removes them: the
  run starts at `max_i sum_l |A_il|`, an a-priori bound on every `|g_i|`.
- **Validation only.** Exact gradients enter a learned run only through its
  validation record: whether the selection was the best arm, its shortfall, and
  whether any interval used missed the truth. In Phase 3 these go to a separate
  `*_validation.json`.

## SOTA-baseline extension (Oct 2026)

Paper A's comparisons against the literature were planning bounds or missing. This extension runs the
external baselines the way Part II's own methods run (sampled outcomes, estimated variances) and adds
other operator pools. Development history: `logs/DEVLOG.md`.

| What | Where |
|---|---|
| Static M1 (one allocation at a common radius, one draw), independent-arm M2 (every gradient on its own groups, sampled), sequential M1 (`LearningConfig(elimination="off")`) | `src/baselines.py`, `src/learning.py`, `scripts/step5_external_baselines.py` |
| Shot reuse after Ikhtiarudin et al. (the last energy evaluation's data, held free), FC or QWC groups; "ours + reuse" | `src/reuse.py`, `src/contexts.py` (`qwc_groups`, `extra_groups`), `scripts/step5_external_baselines.py` |
| Qubit-ADAPT, qubit-excitation (QEB), generalised QEB / qubit, CEO (OVP-CEO) pools, as `<case>@<pool>` | `src/pools.py`, `scripts/step6_pool_census.py` |
| Informationally complete (AIM-style, unoptimised SIC product POVM) baseline | `src/ic.py`, `scripts/step7_ic_baseline.py` |
| Trajectories of the new methods (`--methods`, `--tag`) | `scripts/phase4_adapt_selection.py` |
| Hessian-recycling sensitivity of Q7 | `scripts/paper_a_copt_sensitivity.py` |
| The comparison tables from the stored summaries | `scripts/paper_a_sota_tables.py` -> `runs/paper_a/sota_*.csv` |

Things to know before reading the numbers:
- **M1 is two baselines.** Paper A's M1 is the static all-gradient estimate with the exact gap and a
  worst-case radius (`M1 static`). The sequential M1 (`M1 seq`: same groups, uniform precision, shrinking
  radius, covariance-aware stopping, no elimination) is much cheaper (LiH: 0.76M vs 3.56M shots) and is the
  fair non-oracle reference for the benefit of elimination.
- **Reuse holds one energy evaluation's data for free.** Its size is set by the energy precision
  (`--energy-error`, default 1 mHa, the `C_opt` model's). `reuse.hamiltonian_terms` refuses to run unless the
  rebuilt Hamiltonian reproduces the problem's state energy and stored commutators; the cached
  `H4_square_stretch_side2p0_CISD` fails this (other orbitals), so its reuse rows are skipped.
- **Qubit and CEO pools have exactly tied leaders** (symmetry partners): use `--rho 0.1` and read
  `good_10pct_rate`, not `correct_rate`.
- **IC rows are in IC shots**, an unoptimised POVM, with memory `K 4^n` (H2O is out of reach).
- **Run status (Oct 5, 2026): everything in this section has been run**, on the workstation (H4) and on Trillium (LiH, H2O, stretched H4, pools, IC):
  jobs `ext_h4`, `ext_pools_h4`, `ext_lih_a`, `ext_lih_b`, `ext_h2o_*` (8), `ext_safe_family`, `ext_home_variants`, `ext_h2o_home_*`,
  `ext_m2safe_small`, `ext_safe_h2o_*`; 20-130 min each. Tables: `scripts/paper_a_sota_tables.py` -> `scripts/paper_a_sota_latex.py --splice`;
  the prose of the paper's Q8 is generated by `scripts/paper_a_q8_text.py` (every number computed).
- **Two rules.** The pairwise rule's plug-in signs can eliminate the best arm near zero gradients (7.5% of LiH trials with free data); the sign-aware
  rows (`... safe ...`) were correct in every trial. Tables report both.
- **Reuse variants.** `fixed` assignment (Ikhtiarudin: Paulis an energy context measures are read from it) can cost more than no reuse when the free data
  are small against the selection (H2O); `home` start (II-A decides) never raised the cost on H4/LiH but did on H2O eq CISD. "No reuse" means plain home assignment.
- `LearnedM3(credit=...)` counts only new shots when deciding to refit (a first version counted the
  credit and froze the design; the II-A + reuse rows were rerun).

`cluster/` (sbatch file, sync scripts, job files, workstation run scripts, their logs) is git-ignored: it is site-specific and kept
local. The job files are written by `cluster/jobs/make_ext_jobs.sh`, and every command in them is a documented `scripts/` call with
`--workers`, `--shard` and `--resume`, so they can be rebuilt from this section.

Workstation runs (WSL; `cluster/wsl_run.sh` runs any command in the project venv):

```bash
MSYS_NO_PATHCONV=1 wsl -e bash "/mnt/d/.../part2_estimator_design/cluster/wsl_run.sh" python scripts/step5_external_baselines.py \
    --cases H4_square_eq_side1p0_CISD --trials 200 --workers 10
python scripts/step6_pool_census.py --cases H4_square_eq_side1p0_CISD --pools uccsd qubit qeb gsd_qeb gsd_qubit ceo
python scripts/step7_ic_baseline.py --cases H4_square_eq_side1p0_CISD --anytime --rho 0.1 --trials 100
python scripts/paper_a_sota_tables.py
```

Trillium (anything over about an hour). The job files are written by `cluster/jobs/make_ext_jobs.sh`:
`ext_h4.txt` (trajectories, 14 x 13 workers), `ext_pools_h4.txt` (10 x 12), `ext_lih.txt` (about 176 workers),
and eight `ext_h2o_<case>_s<k>.txt` (one node each, 60 workers; 3.6-7 GB per worker). Submission needs your
MFA-approved SSH master connection:

```bash
ssh -fN -o ControlMaster=yes -o ControlPersist=12h -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" USER@trillium.scinet.utoronto.ca
bash cluster/sync_to_trillium.sh USER
# on Trillium, in /scratch/USER/adapt_sampling/part2_estimator_design:
for f in ext_h4 ext_pools_h4 ext_lih; do sbatch --job-name=$f cluster/trillium_run.sbatch cluster/jobs/$f.txt; done
for f in cluster/jobs/ext_h2o_*.txt; do sbatch --job-name=$(basename $f .txt) cluster/trillium_run.sbatch $f; done
# when done, on the workstation:
bash cluster/sync_from_trillium.sh USER python    # also merges step5 / step7
python scripts/paper_a_sota_tables.py && python scripts/paper_a_copt_sensitivity.py
```


## Generator selection under a finite-sample protocol (Nov 2026)

The paper "Spend Measurements on the Decision" (shared gradient sampling, successive elimination and a selection
tolerance in ADAPT-VQE) rests on a finite-sample selection protocol with empirical-Bernstein intervals, a history
bound, fresh blocks at doubling checkpoints and a relative certificate. Development history: `logs/DEVLOG.md`
(entries from 2026-11-12); the scope and the plan of the study are in `reframe/SCOPE.md` and `reframe/protocol.json`.

| What | Where |
|---|---|
| The selector (design, intervals, elimination, certificate, caps) | `src/reframe.py` (`ReframeSelector.run`; `leader_out=True` is the leader-out certificate) |
| Fixed states (UCCSD pool, 19 states plus CISD stress states, shared / qubit-wise / pivot / individual contexts) | `scripts/reframe_run.py` |
| Measured ADAPT trajectories (selection from sampled outcomes, exact reoptimization) | `scripts/reframe_traj_run.py` |
| Injected selection errors in exact-gradient ADAPT | `scripts/energy_trajectory_policies.py`, `scripts/energy_policies_report.py` (run it without `--cases`: it rewrites `report/policies_all.csv`) |
| Tables, ratios, intervals and the facts quoted in the paper | `scripts/reframe_summary.py` (writes `reports/prior/*.csv` and the `generated/*.tex` bodies of the paper folder), `scripts/reframe_paper_results.py` (figures and the injected-error facts), `scripts/reframe_state_gaps.py` (gaps, contexts per product, CZ counts; needs the cached problems), `scripts/reframe_context_sizes.py` |
| Snapshots of the summary tables and facts at the commit of the paper | `reframe/summaries/` |
| Tests | `tests/test_reframe_intervals.py`, `test_reframe_t24.py`, `test_reframe_t25.py`, `test_review_stage1.py`, `test_pivot.py`, `test_design.py`, `test_energy_policies.py` |

Names. The paper writes SG-FULL, SG-SE, SG-SE-CS (the proposed method), SG-SE-CS+prior and IG-SE; the code and the record
files use `FC_FULL`, `FC_SE`, `FC_SE_CS_DATA`, `FC_SE_CS` and `FC_SE` on the individual-grouping library. Settings are
named `strict` (rho = 0), `rho005`, `rho010`, `rho020` and `rho010risk020` (rho = 0.1 with delta = 0.2).

Records (one JSON line per trial or per trajectory; each carries the protocol hash, the seed and the state, Hamiltonian,
pool and context hashes):

| Folder under `runs/` | Content |
|---|---|
| `reframe_v2`, `reframe_v2_cas_a/b/c` | the matrix at rho = 0 and 0.1 (the three CAS folders hold trials 5-29 of one cell, merged by the summary script) |
| `reframe_v4`, `reframe_v4_ig`, `reframe_v4_growth`, `reframe_v7_growth` | full pool with the dominance rule, individual grouping, checkpoint spacing 1.25 (v4: two H4 and two LiH states; v7: two BeH2 and two water states, SG-SE and SG-SE-CS only) |
| `reframe_v2_grp` | qubit-wise and pivot contexts |
| `reframe_traj`, `reframe_traj_lih`, `reframe_traj_fix`, `reframe_traj_cas`, `reframe_traj_cas_fix` | measured trajectories |
| `reframe_v5_lo`, `reframe_traj_lo` | leader-out certificate (fixed states and trajectories) |
| `reframe_v6_delta` | rho = 0.1 together with delta = 0.2 |
| `reframe_v5_lo` | leader-out certificate on the fixed states (rf6: seven small states; rl1, rl5a, rl5b: the other twelve STO-3G and both active-space states) |
| `reframe_v8_growth` | checkpoint spacing 1.25 on the nine STO-3G states not in v4 and v7 |
| `reframe_v9_cal` | exact identification at inflated delta (0.2, 0.5, 0.9; settings `cal020`, `cal050`, `cal090`): observed error rate against the nominal budget |
| `reframe_traj_h4` | measured trajectories on square H4 at 1.0 and 2.0 A |
| `state_gaps_cas` | gaps of the active-space states |
| `energy_policies`, `energy_policies_gtol1e-5_cl` | injected-error runs (exact gradients, exact optimization) at optimizer gradient threshold 1e-9 and 1e-5; `reframe_paper_results.py` reads both `report/policies_all.csv` |

Regenerate the numbers of the paper from the stored records (no cluster access needed; the tables go to
`reports/reframed/generator_selection_paper_filled/generated`, which is created if it is missing):

```bash
cd part2_estimator_design
python scripts/reframe_summary.py            # matrix, cost, growth, trajectories, grouping, leader-out, rho and delta
python scripts/reframe_paper_results.py      # figures and the injected-error facts
```

The ADAPT snapshot states (`<geometry>_ADAPT<k>`) are rebuilt from the saved exact-gradient ADAPT trajectories (`runs/<geometry>_HF/*_adapt_trajectory*`); cached problems live in the git-ignored `.cache/`.

Rerun a small case (a few minutes once the problems are cached; the production runs used whole cluster nodes, 24 h each, checkpointed):

```bash
python scripts/reframe_run.py --states H4_a1_ADAPT3 --methods FC_SE FC_SE_CS_DATA --trials 10 --workers 2 \
    --setting rho010 --restart-ratio 0 --batch-fix --out runs/reframe_local
python -m pytest tests/test_reframe_intervals.py tests/test_reframe_t25.py -q
```

The runs of the cluster were made with Python 3.11.4, NumPy 2.4.2, SciPy 1.17.1, PySCF 2.14.0 and OpenFermion 1.8.1; the tests
run under the versions of `requirements.txt` (a fixed seed reproduces stored cluster records bit for bit under both). The `resource`
module is POSIX-only, so the test suite runs under WSL or Linux.
