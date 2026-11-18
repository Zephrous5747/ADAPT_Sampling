# Reframed protocol GS-1.0: implementation notes, deviations and scope amendments (2026-11-10)

Code: `src/reframe.py` (selector), `scripts/reframe_run.py` (runner; tasks, seeds and record ids follow `make_plan.py`).
Jobs: `cluster/jobs/rf_*.txt`. Records: `runs/reframe/<state>__<method>[__proportional].jsonl` (fields of `validate_records.py` plus diagnostics).

## Deviations from the text (also listed in `reframe.DEVIATIONS`)
1. The joint design program is solved by the saddle-form solver of `design.py`, not CVXPY/CLARABEL (not installed). Constraints are
   checked after solving. No cross-check against CVXPY has been made.
2. FC_FULL at rho = 0: the literal certificate `L_k >= max_S U_i` cannot hold (the max contains U_k >= L_k); a sole virtual survivor is used.
3. Contexts are the older first-fit weight-ordered FC library; states are the archived states of this repository (not regenerated).
4. Added diagnostic method `FC_SE_CS_DATA` (splitting with measured covariance only): not in the protocol.
5. `--schedule proportional` (sensitivity, not the protocol): the first-checkpoint rule halves the plan until the *unfloored* counts total
   < 4096, then applies the floor. The literal rule (default) ends with every context at the floor whenever more than 128 contexts are active
   (128 x 32 = 4096), which makes the first checkpoint uniform and costs about 5x in shots on LiH 1.6 ADAPT 2 (pilot).

## Method-note runs (`src/note_protocol.py`, `scripts/note_run.py`, jobs `cluster/jobs/nt_*.txt`, records `runs/note/`)
The protocol of `reports/prior/Classical_priors_for_measurement_design.tex`: moment update with tau = 100 (NOTE_II, NOTE_IV; NOTE_III static)
and eps_c = 0.1; every batch measures the full cover with n_min,r = 32 * 2^(r-1) and budget B_r = 2 L n_min,r; equal-weight objective over the
active candidates by alternating exact splitting and floored square-root allocation (6 alternations); coefficient bound gamma = 2 enforced by
a convex combination with the fixed reconstruction; confidence = Appendix A Hoeffding as written (`hoeffding`) or the reframed Bernstein
intervals as the note's "more efficient valid bound" (`bernstein`). Arms: NOTE_I (fixed reconstruction, measured covariance), NOTE_II
(prior, allocation only), NOTE_III (prior static, splitting + allocation), NOTE_IV (prior with updates), NOTE_IV_DATA (diagnostic).
Not fixed by the note and chosen here: tau, eps_c, gamma, n_min, the batch schedule and budget, the number of alternations.

## Work order CP-1.2 (`src/cp_order.py`, `cp_states.py`, `cp_testb.py`, `scripts/cp_order_run.py`, jobs `cluster/jobs/cp_cal|cp_testa_*|cp_testb_*.txt`)
Calibration (leave-one-family-out, 3x3 grid, constants 8/32/128, all scores), Test A (A1/A2, N = 0/8/32/128, true-variance evaluator) and
Test B (strict BAI, common confidence, cap 1e7 shots and 100 blocks) on H2 (0.7414 m=0; 1.5 m=1), H4 (1.0 m=1; 2.0 m=3) and LiH (1.6 m=1; 3.0 m=3).
Design program: CVXPY 1.9.3 / Clarabel 0.11.1 (installed in the Fir venv), 60 s, 500 iterations, 1e-8, normalized variables; programs with more than
60,000 coefficients (LiH: 365,000) are not attempted and use the frozen reconstruction (the work order's fallback) in A2 and Test B; A1 is solved
per arm (exact stationary solve when it satisfies the L1 bound, CVXPY otherwise). Outputs: `runs/cp_order/{calibration.json,testa/,testb/}`.
Test A/B jobs depend on the calibration job (`afterok`).

## Scope amendments (the protocol's cost gate)
* 30 trials per state/method instead of 100 for the first stage; extend only if warranted.
* Not run: supported-pool (QEB) test (blocked pending operator equivalence), active-space water, IND_SE, trajectories, R4 diagnostic.
* The splitting arms on LiH/BeH2/H2O are gated on timing probes (`rf_cs_probe`).
