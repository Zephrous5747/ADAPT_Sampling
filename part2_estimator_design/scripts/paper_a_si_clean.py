#!/usr/bin/env python3
"""Paper A, step 3 of the compression plan: bring the Results-derived parts of the SI to the standard of the SI.

``paper_a_make_si.py`` copies the subsections, paragraphs and floats of the full draft (``archive/draft_full_pre_compression.tex``) that the
live draft no longer states.  They arrive with the labels and wording of the old Results (Q1-Q12, long captions, "steps").  ``apply`` turns
them into SI text: descriptive headings, no Q labels, one-line captions with the explanation in a ``\\tabnote``, and no sentence that the
main text now makes.  Every replacement is count-checked, so a change of the source stops the build instead of leaving old wording behind.
"""
from __future__ import annotations

import re
from pathlib import Path

# --- headings -----------------------------------------------------------------------------------------------------------------------

HEADINGS = {
    "Q1: Does sharing help? Alone, only by a factor of 1--3": "Sharing alone",
    "Q2: Does elimination beat static estimation? Usually, but not at wide gaps": "Elimination alone",
    "Q3: Do sharing and elimination compound? Yes, wherever the pool is large": "Sharing and elimination together",
    "Q4: What does estimator design add? Learned splitting is the main gain": "Learned splitting",
    "Q5: Does redesign during elimination help? It narrows the field earlier": "Redesign during elimination",
    "Q8: How does the method compare with published baselines? Fewer shots than the strongest, which depends on the pool": "Published strategies at shot level",
    "Q9: Do the savings survive an a priori start, a run-wide guarantee and uncertainty? Yes, once the minimum number of shots is raised": "A priori start, run-wide intervals and the minimum number of shots",
    "Q6: Does the saving persist along a full ADAPT trajectory? Yes": "Selection along measured trajectories",
    "Q10: What does certification cost? Selection shots on small systems, nothing in total": "Certified selection against a fixed budget",
}

# --- paragraphs that the main text now states, dropped or shortened ------------------------------------------------------------------

PARAGRAPH_REPLACE = {
    # the headline paragraph of the main text; what it does not state stays
    "\\paragraph{One setting without exact input.}": (
        "\\paragraph{Details of the headline setting.} Below the minimum of 400 shots, II-0, II-A, M1 seq and the reuse baseline replace a "
        "held-out covariance by its bound, and M2 replaces a sample standard deviation by the Popoviciu bound; the default of every method "
        "is 50 shots. Table~\\ref{M-tab:minshots} covers the 17 states of Table~\\ref{tab:apriori} and four ADAPT snapshots of H$_4$. The larger "
        "minimum raises the cost of II-0 by up to $2.0$ and, on LiH at 1.6~\\AA\\ after four iterations, repairs M2 as well "
        "($7.3\\times10^{8}$ against $1.7\\times10^{8}$ shots, $97\\%$ against $100\\%$ correct). Under unequal rules, with II-0 and the "
        "baselines at their default of 50 shots, II-A with 400 shots is dearer on seven early states of H$_4$ and LiH, by up to $1.8$ times "
        "against II-0 and $1.6$ times against the reuse baseline. A minimum of 400 shots makes every method safe on every state tested and "
        "keeps II-A the cheapest when all methods use it; its price falls on the early states, where a selection is cheap."),
    "\\paragraph{Takeaway.} The gain comes from learning": "",
    "\\paragraph{Takeaway.} Against the strongest published baseline": "",
    "\\paragraph{In total shots.} The premium above counts": (
        "\\paragraph{In total shots.} Table~\\ref{M-tab:premiumtotal} gives the totals. The adequate fixed budgets lengthen the trajectories "
        "by $4\\%$ (H$_2$O) to $17\\%$ (LiH at 1.6~\\AA), which the premium in selection shots does not see. At the looser tolerances the extra "
        "selection shots of the certified method eat part of the gain from the shorter trajectories, because the selection is a larger share of "
        "the total, on LiH at 1.6~\\AA\\ all of it (at most $+2\\%$, inside the interval). The intervals of the ratios are $\\pm8$--$40\\%$ at "
        "$10^{-9}$, because of the path noise of the tight threshold (Table~\\ref{tab:find_gtol_paths}), and about $\\pm2$--$12\\%$ at $10^{-5}$ "
        "and $10^{-3}$. The uncertified pilot budgets of the "
        "confidence-level runs (12 runs per budget) behave in the same way: those that reach chemical accuracy in every run cost $1.2$ to "
        "$1.6$ times the total of II-A at all three thresholds, and those that miss it in one run of twelve cost $1.4$ to $3.0$ times."),
    "Two things limit what this comparison shows.": (
        "\\paragraph{Selections within the tolerance.} At the adequate budgets, $83$--$84\\%$ of the fixed-budget selections were within "
        "$\\rho=0.1$ of the best generator on H$_4$ at 1.0~\\AA, $90$--$92\\%$ at 2.0~\\AA, $76$--$84\\%$ on LiH at 1.6~\\AA, $87$--$89\\%$ "
        "at 3.0~\\AA\\ and $88$--$95\\%$ on H$_2$O, against at least $98.7\\%$ for the certified II-0 and II-A."),
}

# --- exact replacements -------------------------------------------------------------------------------------------------------------

REPLACE = [
    # sharing alone
    ("Sharing helps, but alone it is a modest saving (Table~\\ref{tab:q1}). Each product of $\\mathcal B_0$ enters about four gradients, and one "
     "global grouping needs 3.5 to 6.1 times fewer distinct measurement settings than grouping every gradient separately. Measured against the "
     "no-sharing baseline, however, the shared grouping lowers the planning cost by only $2.2$--$3.4$ on H$_4$, $1.9$ on LiH and $1.1$--$1.2$ "
     "on H$_2$O, even when M1 is handed the exact gap. The diagonalizing",
     "Table~\\ref{M-tab:systems} of the main text gives the structure behind the planning bounds. Sharing alone lowers the planning cost by "
     "$2.2$--$3.4$ on H$_4$, $1.9$ on LiH and $1.1$--$1.2$ on H$_2$O, and one global grouping needs 3.5 to 6.1 times fewer distinct "
     "measurement settings than grouping every gradient separately. The diagonalizing"),
    ("this overlap is what the designs of Q4 exploit", "this overlap is what the learned designs exploit"),
    # elimination alone
    ("In most systems BAI dominates grouping (Table~\\ref{tab:q23}). Elimination alone lowers the no-sharing cost by a factor of $1.3$ to $253$, "
     "against $1.1$ to $3.4$ for sharing alone, and M2 beats M1 on seven of the nine states, by a factor of up to $222$ on stretched H$_2$O, "
     "where M1 must resolve high-variance generators with vanishing gradients. M2 loses on the two H$_4$ CISD states. There the gap is wide "
     "relative to the spread of the field, so elimination saves only a factor of $1.3$--$2.1$ while sharing saves $2.2$--$3.1$.",
     "Table~\\ref{tab:q23} gives the absolute costs behind Table~\\ref{M-tab:planning}. M2 loses to M1 on the two H$_4$ CISD states because "
     "the gap is wide relative to the spread of the field: elimination saves only a factor of $1.3$--$2.1$ there, while sharing saves "
     "$2.2$--$3.1$."),
    ("This is why the two are combined in Q3.", "This is why the two are combined."),
    # together
    ("M3 is the cheapest of the three on every state when it eliminates on its actual precision (Table~\\ref{tab:q23}): $1.2$ to $265$ times "
     "cheaper than M1 and $1.19$ to $3.42$ times cheaper than M2. On LiH and H$_2$O the two mechanisms compound (interaction $1.05$--$1.79$). "
     "The shared allocation is sized by expensive losers, so it resolves the leader to $0.48$--$0.79$ of its target, and elimination then "
     "fires earlier. On H$_4$ the leader binds and the two mechanisms substitute ($0.55$--$0.95$). The ordering",
     "The ordering"),
    (" These ratios are planning bounds against the static M1 and the independent M2 with exact variances. The comparison that matters, run "
     "from sampled outcomes, is with the sequential M1, which uses the covariance of the shared estimates, and with independent BAI (Q8): "
     "shared elimination with fixed estimators is $1.3$--$28$ times cheaper than the sequential M1 and $1.2$--$4.3$ times cheaper than "
     "independent BAI on the nine fixed states, and the static M1 costs $1.1$--$5.8$ times the sequential one ($7.2$--$16$ along the "
     "trajectories), so the factors above against static M1 overstate the gain over the best published strategy.",
     " Against the sequential M1 and independent BAI, run from sampled outcomes (Sec.~\\ref{sec:si4}), the static M1 costs $1.1$--$5.8$ "
     "times the sequential one on the fixed states and $7.2$--$16$ along the trajectories."),
    # learning
    ("so Q4 measures what estimator design gains when the radii start right; Q9 repeats the comparison without exact input "
     "(Table~\\ref{tab:minshots}).",
     "so this subsection measures what estimator design gains when the radii start right; Sec.~\\ref{sec:q9} repeats the comparison without "
     "exact input (Table~\\ref{M-tab:minshots})."),
    ("$^d$From the 200-trial baseline study of Q8:", "$^d$From the 200-trial baseline study of Sec.~\\ref{sec:q8}:"),
    # baselines
    ("Q1--Q7 compared the method with reference strategies that we defined. Q8 asks whether the saving survives against the strategies of the "
     "published work, so that the answer does not depend on our choice of reference. Every baseline",
     "Every baseline"),
    (" If a published strategy were cheaper on some state, the saving of Q1--Q7 would be an artifact of our references; if the strongest "
     "baseline changed with the state, the value of sharing and of elimination would depend on it. The systems are those of Q1--Q7 and the "
     "stretched and ADAPT states of H$_2$O, which have the largest UCCSD pool, and the pools of the ADAPT literature test whether the saving "
     "depends on the structure of the UCCSD gradients.",
     " The systems are those of the planning and learning studies (Secs.~\\ref{sec:si2} and~\\ref{sec:si3}) and the stretched and ADAPT "
     "states of H$_2$O, which have the largest UCCSD pool; the pools of the ADAPT literature test whether the saving depends on the "
     "structure of the UCCSD gradients."),
    ("against $4.137\\times10^{7}$ and $5.637\\times10^{9}$ in Q7.", "against $4.137\\times10^{7}$ and $5.637\\times10^{9}$ in the study of the total cost (Sec.~\\ref{sec:cost})."),
    (" On the first four states sharing is worth more than elimination (II-0 against independent BAI: $2.4$ to $4.3$ times; against the "
     "sequential M1: $1.3$ to $2.7$ times), and on the other five elimination is worth more ($4.6$ to $27.8$ times against $1.2$ to $1.8$). "
     "This is the balance seen with exact variances in Q2: sharing wins",
     " On the first four states sharing is worth more than elimination and on the other five elimination is worth more. This is the balance "
     "seen with exact variances (Sec.~\\ref{sec:si2}): sharing wins"),
    ("(the plug-in-sign failure of Q3)", "(the plug-in-sign failure of Table~\\ref{tab:sign})"),
    ("fails when they are wrong (Q3).", "fails when they are wrong (Table~\\ref{tab:sign})."),
    ("except one LiH trajectory per learned method (Q6), and all", "except one LiH trajectory per learned method (Sec.~\\ref{sec:si5}), and all"),
    (" This is the side of the balance that the states of small gradient take above, and it is consistent with Q6, where the cost "
     "concentrates at tied and small-gap steps. Independent BAI costs $1.1$ to $1.4$ times II-0, whereas the sequential M1 costs $3.2$, $2.4$ "
     "and $4.1$ times II-0 on H$_4$ (two geometries) and LiH and $10.1$ to $10.3$ times on H$_2$O, and the static M1 $17$ to $68$ times on "
     "H$_4$ and LiH (Q7) and $98$ and $121$ times on H$_2$O.",
     " This is the side of the balance that the states of small gradient take above, and it is consistent with the selection along "
     "trajectories (Sec.~\\ref{sec:si5}), where the cost concentrates at tied and small-gap iterations. The sequential M1 costs $10.1$ to "
     "$10.3$ times II-0 on H$_2$O."),
    # minimum
    ("Q1--Q8 start every selection from the exact largest gradient (to set the first radius) and use intervals that hold for each round, and "
     "the comparisons of Q4 and Q8 use the pairwise elimination rule.",
     "The studies above start every selection from the exact largest gradient (to set the first radius) and use intervals that hold for "
     "each round, and the comparisons of Secs.~\\ref{sec:si3} and~\\ref{sec:si4} use the pairwise elimination rule."),
    (" The last paragraph fixes one setting without exact input for the headline comparison.", ""),
    ("(the nine of Q4 and Q8 and four ADAPT snapshots of H$_4$)", "(the nine of Secs.~\\ref{sec:si3} and~\\ref{sec:si4} and four ADAPT snapshots of H$_4$)"),
    (" The last paragraph repeats every state of the table, and four ADAPT snapshots of H$_4$, with a minimum of 400 shots for every method.",
     " Table~\\ref{M-tab:minshots} repeats every state of the table, and four ADAPT snapshots of H$_4$, with a minimum of 400 shots for every method."),
    ("The per-round intervals of Q1--Q8 hold", "The per-round intervals of the studies above hold"),
    ("so the range of Q8 holds on a second", "so the range of Sec.~\\ref{sec:si4} holds on a second"),
    ("On the late states the minimum matters in the opposite direction (Q9): there", "On the late states the minimum matters in the opposite direction (Sec.~\\ref{sec:q9}): there"),
    # trajectories and certification
    ("Yes. On H$_4$ the learned designs", "On H$_4$ the learned designs"),
    ("this rise is the path noise of the tight threshold (Q12), not a property", "this rise is the path noise of the tight threshold (Table~\\ref{tab:find_gtol_paths}), not a property"),
    ("Q7 used the model of the optimization cost of Sec.~\\ref{sec:resources}. We replaced it", "The main text uses the model of the optimization cost of Sec.~\\ref{M-sec:resources}. We replaced it"),
    ("and the conclusion of Q7, that optimization", "and the conclusion of the main text, that optimization"),
    ("Relative to the cheaper model of the looser thresholds of Q7 the gap is larger still.", "Relative to the cheaper model of the looser thresholds of the main text the gap is larger still."),
    ("at $95\\%$ the 36 trajectories of Q8)", "at $95\\%$ the 36 trajectories of Table~\\ref{tab:q8traj})"),
    # paragraphs of the old Discussion
    ("(over the nine fixed states of Q8)", "(over the nine fixed states of Sec.~\\ref{sec:si4})"),
    ("one look per round (Q3), not a run-wide guarantee", "one look per round (Sec.~\\ref{sec:si2}), not a run-wide guarantee"),
    ("is what we measured in Q3, so", "is what we measured in Sec.~\\ref{sec:si2}, so"),
    ("the last exact input (Q3), is not tested", "the last exact input, is not tested"),
    ("The totals of Q7 and Q10 do not include it", "The totals of the main text do not include it"),
    ("of the total (Q7). Hessian recycling", "of the total (Sec.~\\ref{sec:rq_total}). Hessian recycling"),
    ("reached reliably (Q12).", "reached reliably (Table~\\ref{tab:sampledopt})."),
    ("the adequate uncertified budgets of Q10 add", "the adequate uncertified budgets of Sec.~\\ref{sec:q10} add"),
    ("the safer currency (Q11)", "the safer currency (Sec.~\\ref{sec:rq_traj})"),
    ("the covariance prior of Q4 fails alone", "the covariance prior of Sec.~\\ref{sec:si3} fails alone"),
    # the scaling table is Table 1 of the main text
]

# --- floats -------------------------------------------------------------------------------------------------------------------------

DROP_FLOATS = {"tab:scaling", "tab:q1"}  # the structure of the systems is Table 1 of the main text

# tables that the SI takes from a generator (scripts/paper_a_si_tables.py) instead of the numbers typed into the old draft
GENERATED = {"tab:q23": "si_planning_costs", "tab:q3": "si_surrogate", "tab:sign": "si_sign", "tab:q4oracle": "si_ceilings",
             "tab:q4learned": "si_learned_fixed", "tab:q4more": "si_learned_states", "tab:q5": "si_rounds", "tab:q6": "si_trajectory"}
GENERATED_DIR = Path(__file__).resolve().parents[2] / "reports" / "generated"

CAPTIONS = {
    "tab:q1": ("Structure of the shared measurements.",
               "$N$: fully commuting contexts of the shared grouping; $\\sum_iN_i$: contexts of the per-gradient groupings (M2); uses: average "
               "number of gradients in which a Pauli product of $\\mathcal B_0$ appears; mult.: number of contexts whose measured group "
               "contains a product of $\\mathcal B_0$; split.: share of $\\mathcal B_0$ measured by at least two contexts; CZ: two-qubit "
               "gates per diagonalizing circuit; sharing: planning cost of the no-sharing baseline divided by that of M1."),
    "tab:q23": ("Planning-bound costs in context-shots.",
                "Exact gradients and variances, $\\delta=0.05$, STO-3G, UCCSD pool; H$_4$ is the square molecule of side 1.0 or 2.0~\\AA, H$_2$O is "
                "at equilibrium (eq) or stretched (str). Gap: difference of the two largest $\\lvert g_i\\rvert$. M3 eliminates on each "
                "candidate's actual precision. Interaction: (baseline/M3)/[(baseline/M1)(baseline/M2)], above one when sharing and "
                "elimination compound."),
    "tab:q3": ("Shared elimination with sampled outcomes against the Gaussian surrogate.",
               "STO-3G, UCCSD pool, $\\delta=0.05$. II-0: pairwise rule, $\\gamma=0.9$, top-up allocation, shot level; surrogate: M3 as run in the "
               "Gaussian surrogate (marginal rule, $\\gamma=0.5$). Median context-shots, 250 trials (LiH 100). Wrong: selections that were "
               "not the best generator. Reuse: surviving candidates informed per context-shot (Sec.~\\ref{M-sec:planning}), from logged runs "
               "with estimated radii; M2 has reuse one."),
    "tab:sign": ("Failure of the plug-in signs on a correlated state.",
                 "Stretched H$_4$ (side 2.0~\\AA) after two ADAPT iterations (relative gap 11\\%), 100 trials per row. Missed: share of trials in "
                 "which some interval missed the exact value. Every wrong selection here is also not 1~mHa-good."),
    "tab:q4oracle": ("Planning ceilings of estimator design.",
                     "M3 on actual precision with exact covariances: change relative to II-0 on the same metric. Redesign: oracle II-D, "
                     "reoptimized at every elimination breakpoint, relative to II-0 on the M3 bound with top-up accounting. Mass-ordered "
                     "completion; with the canonical completion LiH reaches $-76\\%$. The solver minimizes the M3 bound, not the "
                     "actual-precision cost: an independent second solve of II-A on LiH and H$_2$O reached the bound within 2 points but the "
                     "actual-precision cost within 11 points ($-70\\%$, $-80\\%$, $-75\\%$, $-64\\%$, $-68\\%$), so differences of that size "
                     "between II-A and II-B are not significant."),
    "tab:q4learned": ("Learned estimators on fixed states, finite shots.",
                      "CISD states of H$_4$ (side 1.0 and 2.0~\\AA) and Hartree--Fock state of LiH; STO-3G, UCCSD pool, $\\delta=0.05$; estimated "
                      "radii unless marked. Mean context-shots (change relative to II-0). Every row selected the winner in every trial except "
                      "where marked (correct trials of 200). Extends Table~\\ref{M-tab:learning}."),
    "tab:q4more": ("Learned estimators on H$_2$O CISD and on trajectory states.",
                   "State after $k$ ADAPT iterations, 100 trials each, estimated radii. Mean context-shots of II-0 and change relative to it; "
                   "in brackets the percentage of correct selections where not 100. II-0 and II-A use the pairwise rule, II-E the sign-aware rule."),
    "tab:q5": ("One selection with fixed and with learned coefficients.",
               "H$_4$ at 1.0~\\AA, CISD, same random stream. $\\lvert\\mathcal A_r\\rvert$, $\\lvert\\mathcal B_r\\rvert$: active candidates and their "
               "union support at the start of round $r$; ctx: contexts receiving shots in the round. At each refit: median relative error of the "
               "learned covariances of the leader's contexts (cov.\\ err.), share of the leader's coefficient mass moved off its parent "
               "contexts (moved), and the leader's variance relative to II-0 at the shots held, with exact covariances (var.\\ ratio). Rounds "
               "without a refit show in parentheses the values of the most recent refit."),
    "tab:q8best": ("Strongest baseline against II-0 and II-A, per state.",
                   "Mean context-shots; ratios of the baseline's cost to ours (above one: our method is cheaper). STO-3G, UCCSD pool, "
                   "$\\delta=0.05$; 100 to 200 trials per state and 12 to 96 trajectories per method. Without free data: the best of M1 static, "
                   "M1 seq and M2 against II-0 and II-A. With the energy data of the last VQE evaluation (1~mHa): the best of all baselines "
                   "including the reuse baseline against the better of II-0 + reuse and II-A + reuse. Sign-aware: rows with the sign-aware rule "
                   "throughout. Rows whose selections were correct in fewer than 95\\% of the trials are excluded."),
    "tab:q8fixed": ("All methods on five fixed states, estimated radii.",
                    "Mean context-shots and, in brackets, the ratio to II-0; starting radius $\\max_i\\lvert g_i\\rvert$ (exact identification). "
                    "STO-3G, UCCSD pool, $\\delta=0.05$; 200 trials on H$_4$, 100 to 200 on LiH, 100 on H$_2$O. M1 static is handed the exact "
                    "gap and variances. The reuse rows hold the shots of one energy evaluation to 1~mHa for free (they are part of "
                    "$C_{\\mathrm{opt}}$) and charge only new shots. Square brackets: percentage of correct selections when below 100\\%."),
    "tab:q8traj": ("Baselines along measured ADAPT trajectories.",
                   "Median cumulative selection context-shots to chemical accuracy and, in brackets, the ratio to II-0; $\\rho=0.1$, a priori "
                   "starting radius, sign-aware rule, STO-3G, UCCSD pool, $\\delta=0.05$; 48 to 96 trajectories per method on H$_4$, 30 to 90 on "
                   "LiH, 12 to 36 on H$_2$O. M1 static uses the exact gradient scale and variances. A dash: configuration not run."),
    "tab:q8pools": ("Operator pools of the ADAPT literature.",
                    "CISD and HF states of H$_4$ at 1.0~\\AA, HF state of LiH; STO-3G, $\\delta=0.05$, 100 trials per row. $K$: generators; "
                    "$\\lvert\\mathcal B_0\\rvert$: parent support; $N$: fully commuting contexts of the shared grouping; uses: gradients per "
                    "Pauli product. Planning bounds (exact gradients and variances) are the ratios M1/M3 and M2/M3 where the leader is unique; "
                    "qubit-type pools have exactly tied leaders (dash). Right: sampled outcomes with estimated variances and a $\\rho=0.1$ stop, "
                    "mean context-shots."),
    "tab:q8eps": ("Reuse of the energy measurement against its precision.",
                  "H$_4$ at 1.0~\\AA, CISD; mean context-shots, 100 trials (200 at 1~mHa). $\\varepsilon_E$: standard error of the last energy "
                  "evaluation; the data held for free scale as $\\varepsilon_E^{-2}$. Last row: the same methods without free data."),
    "tab:reproduction": ("Reproduction of the published reduction in measurements.",
                         "Percentage reduction of the measurements needed for selection to chemical accuracy relative to estimating every fragment "
                         "to a fixed precision: Table~I of Huang and Izmaylov (UCCSD pool) and our implementation of their method (mean $\\pm$ "
                         "s.d. over 10 seeds), with the samples of earlier rounds kept or with fresh samples in every round."),
    "tab:apriori": ("Selection cost with an a priori first radius.",
                    "Mean context-shots to select the best generator over 100 trials, $\\delta=0.05$, $\\rho=0$, sign-aware rule, default "
                    "minimum of 50 shots; STO-3G, UCCSD pool. II-A + reuse: II-A with the energy data of the last VQE evaluation; Ikh: reuse "
                    "baseline of Ikhtiarudin \\textit{et al.}; M2 marg.: independent elimination with Successive Elimination. Brackets: share "
                    "of correct selections when below $100\\%$."),
    "tab:diag": ("Why intervals miss on three late states.",
                 "Sign-aware rule, a priori start, $\\rho=0$; 100 trials (40 for H$_2$O). $n_{\\min}$: minimum number of shots per fold below "
                 "which a held-out covariance is replaced by its bound. Missed: share of trials in which some interval used by the selection "
                 "missed its exact value in some round; first round: median round of the first miss; zero-width: share of the missed trials "
                 "with an estimated standard deviation of zero; sd ratio: median over missed trials of the smallest ratio of the estimated to "
                 "the true standard deviation among the missing intervals. Anytime: intervals that hold over the whole selection."),
    "tab:anytime": ("Price of intervals that hold over the whole selection.",
                    "Mean context-shots; a priori start, sign-aware rule, default minimum of 50 shots, 100 trials per state. Ratio II-A: anytime "
                    "over per-round cost of II-A. Missed: share of trials in which some interval used by the selection missed its exact value, "
                    "for per-round and anytime intervals (Table~\\ref{tab:diag} for larger minima)."),
    "tab:headlineci": ("Ratios of mean cost with bootstrap intervals at the default minimum.",
                       "A priori start, sign-aware rule, default minimum of 50 shots; 95\\% bootstrap intervals. II-0 / II-A and best baseline / "
                       "II-A: ratios of mean context-shots (the baseline of lowest mean cost, whatever its share of correct selections); below "
                       "one, II-A is the more expensive. Correct: share of correct selections by II-A. LiH after five iterations: 400 trials; "
                       "the others 100."),
    "tab:thresholds": ("Sensitivity of II-A to its thresholds.",
                       "A priori start, sign-aware rule, 200 trials. Mean context-shots at the default (50 shots, refit factor 1.5) and the ratio "
                       "to it for minimum shots of 20, 100 and 200 and refit factors of 1.25, 2 and 3; last column: smallest share of correct "
                       "selections over all settings."),
    "tab:q6": ("Measured ADAPT on H$_4$ at 1.0~\\AA.",
               "STO-3G, UCCSD pool, $\\delta=0.05$, 48 trajectories per method, $\\rho=0.1$, empirical mode. Iteration $k$ selects at "
               "$\\lvert\\psi_k\\rangle$; $\\Delta E$: exact energy error after the selected generator is added and the parameters are "
               "reoptimized. Shots: median cumulative selection context-shots."),
    "fig:trajectory": ("Selection cost along measured trajectories.",
                       "Median shots of each selection (top) and cumulative (bottom) for II-0, II-A, independent elimination (M2), the sequential "
                       "estimator (M1 seq) and the reuse baseline, on H$_4$ at 1.0~\\AA, LiH at 3.0~\\AA\\ and H$_2$O at equilibrium."),
    "tab:fixedbudget": ("Cheapest adequate fixed budget against certified II-A.",
                        "Along complete ADAPT trajectories; median total selection shots; $\\rho=0.1$ for the certified method. Designed: "
                        "allocation from exact variances; pilot: from variances estimated on a fifth of the budget. II-A / designed (pilot): "
                        "certified over fixed total, the premium paid for certification when above one."),
    "fig:fixedbudget": ("Fixed-budget selection along ADAPT trajectories.",
                        "Top: share of trajectories reaching chemical accuracy against shots per selection. Bottom: median total shots; dashed "
                        "lines: the certified methods (II-A, II-0 and M2)."),
    "tab:fixedrho": ("Premium of certification at four tolerances.",
                     "Certified II-A and II-0 along the ADAPT trajectories at four tolerances $\\rho$: median total selection shots (mean "
                     "iterations in brackets) against the cheapest adequate fixed budget (designed allocation), which does not depend on $\\rho$."),
    "tab:confidence": ("Confidence level of the selection against the cost of complete runs.",
                       "II-A, sign-aware rule, a priori start, $\\rho=0.1$, measured trajectories. Runs per level: 48 (H$_4$), 24 (LiH) and 12 "
                       "(H$_2$O, at 99, 90 and $68\\%$; at $95\\%$ the 36 trajectories of Table~\\ref{tab:q8traj}). Selection shots: median at "
                       "$99\\%$ and its ratio to the median at $68\\%$. Last columns: modeled optimization cost at $68\\%$ relative to $99\\%$ "
                       "(ratio of means with 95\\% bootstrap interval) at three optimizer thresholds; at $10^{-5}$ and $10^{-3}$ the recorded "
                       "runs are replayed."),
    "tab:sampledopt": ("Sampled optimization along the exact ADAPT trajectory.",
                       "10 seeds: energy evaluations, total shots, ratio to the model $C_{\\mathrm{opt}}$ ($\\eta=2$ / $\\eta=4$) and mean "
                       "final error."),
    "tab:depth": ("Shots against measurement-circuit depth.",
                  "Every method on every family of measurement contexts: mean context-shots over trials, mean two-qubit gates per shot (CZ), "
                  "and the gain of II-A over M1 seq on the same contexts. Blocks of $b$ qubits commute block by block, so each circuit is a "
                  "product of Cliffords on $b$ qubits (at most $b(b-1)/2$ CZ per block); blocks of one qubit are qubit-wise commuting, and the "
                  "pivot contexts are those of Anastasiou \\textit{et al.} The rows of independent arms give the cost of M2 in the column of "
                  "M1 seq, and a dash marks the columns that do not apply to it; $^\\ddagger$its ratio is to II-A on the same family of "
                  "contexts (fully commuting or qubit-wise commuting). $^\\dagger$Sign-aware rule, and the sign-aware M1 seq in the ratio: on "
                  "the published pivot contexts every product is measured in several contexts with few shots each, the plug-in variances of "
                  "such contexts are unreliable, and the pairwise II-A selected the wrong generator in 34\\% of the trials on H$_4$ 1.0 CISD "
                  "and 76\\% of the trials on LiH HF."),
    "tab:noise": ("Noisy measurement circuits.",
                  "Depolarizing error of probability $p_2$ after every CZ gate. Each cell: share of selections that picked the exact best "
                  "generator (of the noiseless gradients) and mean context-shots. The intervals know shot noise only."),
    "tab:radius": ("The radius rule.",
                   "Median context-shots and share of exact selections with the Bonferroni radius of the main text and with the less "
                   "conservative $z=z_{\\delta/2}/\\sqrt2$ calibrated to the selection error (about $2.2$ times smaller at $K=26$)."),
    "tab:termination": ("Certifying the stop.",
                        "Planning bound (exact variances, fully commuting contexts) for the shots that give every gradient the radius "
                        "$\\tau-\\lvert g_i\\rvert$, the measured selection cost of the whole trajectory with II-A, and their ratio."),
    "tab:score": ("Cost of an energy-based score.",
                  "Each candidate needs its own energy landscape (five energies) on its own state, so nothing is shared across candidates; "
                  "energies are measured with the fully commuting groups of $\\hat H$. Columns 4--5: every energy to 1~mHa. Columns 6--8: "
                  "every energy to the precision that resolves the energy lowerings $g^2/2h$ of the two best candidates ($h=1$~Ha assumed). "
                  "Ratios are to the mean cost of II-A."),
    "tab:window": ("Fixed states of the 16-qubit cc-pVDZ window of H$_2$O.",
                   "8 orbitals; a priori start, sign-aware rule, $\\rho=0.1$, $\\delta=0.05$; mean context-shots over the trials. Ratio: mean "
                   "shots over those of II-A. Within $\\rho$: share of trials whose selection is within $\\rho$ of the best generator; exactly: "
                   "share that selected the best."),
}

FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}\n?", re.S)


def _balanced(text: str, start: int) -> int:
    """Index after the brace group that opens at ``text[start - 1]``."""
    depth, j = 1, start
    while depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return j


def _restyle(m: re.Match) -> str:
    block = m.group(0)
    lab = re.search(r"\\label\{([^}]*)\}", block)
    name = lab.group(1) if lab else None
    if name in DROP_FLOATS:
        return ""
    if name not in CAPTIONS:
        return block
    title, note = CAPTIONS[name]
    if name in GENERATED:
        body = (GENERATED_DIR / f"{GENERATED[name]}.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
        block = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", lambda _: f"% BEGIN INLINE {GENERATED[name]}\n{body}% END INLINE {GENERATED[name]}\n", block, count=1, flags=re.S)
    i = block.index("\\caption{") + len("\\caption{")
    block = block[: i] + title + block[_balanced(block, i) - 1:]
    end = block.rindex("\\end{" + m.group(1) + "}")
    return block[:end] + "\\tabnote{" + note + "}\n" + block[end:]


NUMBER = r"(?:\d+|\$[\d.]+\$|one|two|three|four|five|six|seven|eight|nine|ten)"
ITERATION_PHRASES = [
    (re.compile(r"(?<![\w$])(%s)( ADAPT)? steps\b" % NUMBER), r"\1\2 iterations"),
    (re.compile(r"\bADAPT steps\b"), "ADAPT iterations"),
    (re.compile(r"\bnumber of steps\b"), "number of iterations"),
    (re.compile(r"\bextra step\b"), "extra iteration"),
    (re.compile(r"\b(tied|small-gap) steps\b"), r"\1 iterations"),
    (re.compile(r"\btied step (\d)"), r"tied iteration \1"),
    (re.compile(r"\bwithin one step of exact ADAPT"), "within one iteration of exact ADAPT"),
    (re.compile(r"\bthese steps do not terminate"), "these iterations do not terminate"),
]


def iterations(text: str) -> str:
    """ADAPT iterations are called iterations in the main text (and in Table S1); the old Results said steps."""
    body = []
    for line in text.split("\n"):
        if not line.startswith("%"):
            for pattern, new in ITERATION_PHRASES:
                line = pattern.sub(new, line)
        body.append(line)
    return "\n".join(body)


def apply(text: str) -> str:
    for old, new in HEADINGS.items():
        assert text.count("\\subsection{" + old + "}") == 1, f"heading not found: {old[:50]}"
        text = text.replace("\\subsection{" + old + "}", "\\subsection{" + new + "}")
    for start, new in PARAGRAPH_REPLACE.items():
        i = text.index(start) if text.count(start) == 1 else -1
        assert i >= 0, f"paragraph not found once: {start[:50]}"
        j = text.index("\n\n", i)
        text = text[:i] + new + text[j:]
        if not new:
            text = text[:i].rstrip("\n") + "\n" + text[i:].lstrip("\n")
    for old, new in REPLACE:
        assert text.count(old) == 1, (text.count(old), old[:70])
        text = text.replace(old, new)
    text = iterations(text)
    for gone in ("tab:scaling", "tab:q1"):
        assert "\\ref{%s}" % gone in text, gone
        text = text.replace("\\ref{%s}" % gone, "\\ref{M-tab:systems}")  # the structure table is Table 1 of the main text
    text = FLOAT.sub(_restyle, text)
    return text
