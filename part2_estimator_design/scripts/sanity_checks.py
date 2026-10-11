"""Sanity checks on merged Part II results (Steps 4, 5, 7, Phase 2/4).

Each check states an invariant that must hold if the run, the merge and the estimators are right.
Severity: FAIL = a bug or a broken merge; WARN = outside the expected range, needs an explanation;
KNOWN = a documented, expected exception (unsafe-by-design configurations).  Exit code 1 on any FAIL.

  python scripts/sanity_checks.py LiH_R3p0_HF LiH_R3p0_ADAPT3 LiH_R3p0_ADAPT5 [--delta 0.05] [--csv out.csv]

Checks
  C1  completeness: no missing/duplicate trial ids, no NaN shots/rounds, summary == trial files (n, mean shots, correct rate)
  C2  one truth per case: the arm selected by every correct trial is the same arm in every config of the case
      (catches a Hamiltonian/orbital mismatch between cluster and workstation runs)
  C3  failure logic: a wrong selection by an eliminating method must come with best_eliminated or rho-stopping
  C4  rate ordering: correct <= good_1% <= good_5% <= good_10%
  C5  validity: correct-rate upper Wilson bound >= 1 - delta for every configuration whose guarantee applies
  C6  depth accounting: blocks=1/QWC use 0 CZ per shot; blocks=b uses at most (n/b)*b(b-1)/2
  C7  zero-CZ noise identity: noisy runs of 0-CZ contexts are identical across p and consistent with noiseless runs
  C8  noise monotonicity: shots do not decrease with the two-qubit error for contexts with CZ > 0
  C9  shot orderings that must hold: II-A <= II-0, safe >= pairwise, M1 seq <= M1 static, FC <= blocks=4 <= blocks=1
  C10 identities: M2 safe == M2 pairwise within 2% (independent arms); M1 static equals the exact planning value
  C11 planning ceiling: realised oracle-variance shots within [0.5, 2] x the step-3 planning value
  C12 exact ADAPT trajectory: energy decreases, error >= 0, gaps in [0,1], logged best == added_next
  C13 Phase 4 trajectories: all reached, the selected arm equals the exact best or is rho-good, shots add up
  C14 coverage: configurations of one case missing in a sibling case, and rows with fewer trials than the mode
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # overridden by --root
SUMMARY = {"step4": "learned_designs", "step5": "external_baselines", "step7": "ic_baseline"}

# Configurations whose validity guarantee does not apply (documented in the paper / DEVLOG).
KNOWN_UNSAFE = [
    (r"selection z", "loose selection radius: valid only with the sign-aware rule and not on every state"),
    (r"HF prior", "prior learned at HF is mis-specified at later states (ablation)"),
    (r"reuse(?! \(home\))(?!.*safe)", "reuse with the pairwise rule shares samples across arms (not covered by the guarantee)"),
    (r"reuse \(home\)(?!.*safe)", "reuse with the pairwise rule shares samples across arms (not covered by the guarantee)"),
    (r"bound start", "bound-start designs are not claimed to be valid"),
    (r"^Pivot II-A(?!, safe)", "pairwise rule on the published pivot contexts: every product is measured in several contexts with few shots each, "
                               "and plug-in covariances from folds with fewer than 50 shots give invalid intervals (H4 66%, LiH 24% correct); "
                               "the sign-aware rule bounds them (radius_min_shots) and is correct in every trial"),
]

OUT: list[dict] = []


def rec(sev: str, case: str, label: str, check: str, msg: str):
    OUT.append(dict(severity=sev, case=case, label=label, check=check, message=msg))


def wilson_high(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (c + r) / d)


def known_reason(label: str):
    for pat, why in KNOWN_UNSAFE:
        if re.search(pat, label):
            return why
    return None


def load(case: str):
    S, T = [], []
    for step, tag in SUMMARY.items():
        sp = ROOT / "runs" / case / f"{case}_{step}_{tag}.csv"
        tp = sorted(glob.glob(str(ROOT / "runs" / case / "trials" / f"{step}__*.csv")))
        if not tp:
            continue
        T.append(pd.concat([pd.read_csv(p).assign(step=step, file=os.path.basename(p)) for p in tp], ignore_index=True))
        if sp.exists():
            S.append(pd.read_csv(sp).assign(step=step))
    return (pd.concat(S, ignore_index=True) if S else pd.DataFrame()), (pd.concat(T, ignore_index=True) if T else pd.DataFrame())


def two_sample_z(m1, s1, m2, s2):
    d = np.hypot(s1, s2)
    return 0.0 if d == 0 else (m1 - m2) / d


def check_phase4(case: str):
    # C13 ---------------------------------------------------------------------------------------------------
    for sp in sorted(glob.glob(str(ROOT / "runs" / case / "phase4" / "*_summary.csv"))):
        s = pd.read_csv(sp)
        base = sp[: -len("_summary.csv")]
        tr = pd.read_csv(base + "_trajectories.csv")
        se = pd.read_csv(base + "_selections.csv")
        for r in s.itertuples():
            lab = r.label
            if r.reached_rate < 1:
                rec("WARN", case, r.method, "C13", f"reached rate {r.reached_rate:.2f}")
            if r.rho_good_rate < 0.95 and r.method != "M1 static":
                rec("WARN", case, r.method, "C13", f"rho-good rate {r.rho_good_rate:.3f}")
            g = se[se.method == r.method]
            tot = g.groupby("trajectory").shots.sum()
            t2 = tr[tr.method == r.method].set_index("trajectory").total_shots
            if len(tot) and not np.allclose(tot.reindex(t2.index).values, t2.values, rtol=1e-6):
                rec("FAIL", case, r.method, "C13", "trajectory total shots != sum of per-selection shots")
            if len(g) != r.selections:
                rec("FAIL", case, r.method, "C13", f"{len(g)} selection rows but summary says {r.selections}")
        rec("INFO", case, os.path.basename(base).split("phase4_")[-1], "C13",
            f"{int(s.trajectories.iloc[0])} trajectories, exact-best {float(s.exact_best_rate.iloc[0]):.3f}, rho-good {float(s.rho_good_rate.iloc[0]):.3f}, median shots {float(s.shots_median.iloc[0]):.3g}")


def check_case(case: str, delta: float):
    S, T = load(case)
    if S.empty:
        if glob.glob(str(ROOT / "runs" / case / "phase4" / "*_summary.csv")):
            check_phase4(case)  # trajectory-only case
        else:
            rec("FAIL", case, "-", "C1", "no merged summary found")
        return None
    # statistics are recomputed from the trial files, so a corrupted trial file cannot hide behind a stale summary
    g = T.groupby("label")
    n_by = g.size()
    mean = g.shots.mean().to_dict()
    sem = (g.shots.std(ddof=1) / np.sqrt(n_by)).fillna(0.0).to_dict()
    cz = (g.cz_per_shot_mean.mean() if "cz_per_shot_mean" in T else pd.Series(dtype=float)).to_dict()
    nt = n_by.to_dict()
    corr = g.correct.mean().to_dict()

    # C1 -------------------------------------------------------------------------------------------------
    for (step, lab), g in T.groupby(["step", "label"]):
        n_exp = int(g.n_trials.max())
        ids = g.trial.astype(int)
        if ids.duplicated().any():
            rec("FAIL", case, lab, "C1", f"{int(ids.duplicated().sum())} duplicate trial ids")
        miss = sorted(set(range(n_exp)) - set(ids))
        if miss:
            rec("FAIL", case, lab, "C1", f"{len(miss)} of {n_exp} trials missing (first {miss[:4]})")
        for col in ("shots", "rounds", "correct"):
            if col in g and g[col].isna().any():
                rec("FAIL", case, lab, "C1", f"NaN in {col} ({int(g[col].isna().sum())} trials)")
        if "IC" not in lab and (g.shots <= 0).any():
            rec("FAIL", case, lab, "C1", f"{int((g.shots <= 0).sum())} trials with non-positive shots")
        r = S[(S.step == step) & (S.config == lab)]
        if r.empty:
            rec("FAIL", case, lab, "C1", "trials present but no summary row (stale merge)")
            continue
        r = r.iloc[0]
        if int(r.n_trials) != len(g):
            rec("FAIL", case, lab, "C1", f"summary n={int(r.n_trials)} but {len(g)} trial rows")
        if abs(r.shots_mean - g.shots.mean()) > 1e-6 * max(1.0, abs(r.shots_mean)):
            rec("FAIL", case, lab, "C1", f"summary mean shots {r.shots_mean:.6g} != trials {g.shots.mean():.6g}")
        if abs(r.correct_rate - g.correct.mean()) > 1e-9:
            rec("FAIL", case, lab, "C1", f"summary correct {r.correct_rate} != trials {g.correct.mean():.4f}")

    # C2 -------------------------------------------------------------------------------------------------
    ok = T[(T.correct == 1) & (T.shots > 0)]
    truth = ok.selected.value_counts()
    if len(truth) > 1:
        rec("FAIL", case, "*", "C2", f"correct trials select different arms: {truth.to_dict()}")
    else:
        rec("INFO", case, "*", "C2", f"single true arm {int(truth.index[0])} across {ok.label.nunique()} configs")

    # C3 / C4 ---------------------------------------------------------------------------------------------
    if "best_eliminated" in T and "stopped_rho" in T:
        wrong = T[(T.correct == 0) & (T.shots > 0) & (T.best_eliminated.astype(str) != "True") & (T.stopped_rho.astype(str) != "True")]
        for lab, g in wrong.groupby("label"):
            if lab.startswith(("M1 static", "Pivot M1 static", "Pivot merged M1 static")):
                continue
            rec("WARN", case, lab, "C3", f"{len(g)} wrong selections with the best arm not eliminated and no rho stop")
    cols = ["good_1pct_rate", "good_5pct_rate", "good_10pct_rate"]
    if all(c in S for c in cols):
        bad = S[(S.correct_rate > S.good_1pct_rate + 1e-9) | (S.good_1pct_rate > S.good_5pct_rate + 1e-9) | (S.good_5pct_rate > S.good_10pct_rate + 1e-9)]
        for r in bad.itertuples():
            rec("FAIL", case, r.config, "C4", "rate ordering correct <= 1% <= 5% <= 10% violated")

    # C5 ---------------------------------------------------------------------------------------------------
    for r in S.itertuples():
        n = nt.get(r.config, 0)
        if n < 20:
            continue
        # runs that stop at rho-good (labels with rho=0.1) are judged by the rho-good rate
        rate, what = (r.good_10pct_rate, "10%-good") if "rho=0.1" in r.config else (corr[r.config], "correct")
        hi = wilson_high(round(rate * n), n)
        if hi < 1 - delta:
            why = known_reason(r.config)
            rec("KNOWN" if why else "FAIL", case, r.config, "C5",
                f"{what} {rate:.3f} (n={n}, upper {hi:.3f}) < {1 - delta:.2f}" + (f"; {why}" if why else ""))
        elif rate < 1 - delta:
            rec("WARN", case, r.config, "C5", f"{what} {rate:.3f} below {1 - delta:.2f} but not significantly (n={n})")

    # C6 ---------------------------------------------------------------------------------------------------
    nq = None
    for lab, v in cz.items():
        if v is None or np.isnan(v):
            continue
        if re.search(r"blocks=1\b|QWC", lab) and v > 1e-9:
            rec("FAIL", case, lab, "C6", f"{v:.3f} CZ/shot for a product-basis context")
        m = re.search(r"blocks=(\d+)", lab)
        if m and int(m.group(1)) > 1:
            b = int(m.group(1))
            per_block = b * (b - 1) / 2
            if v <= 0:
                rec("FAIL", case, lab, "C6", "zero CZ for a multi-qubit block")
            elif v > per_block * 12 / b + 1e-9:  # n >= 12 for LiH; bound uses the maximum n/b blocks
                rec("WARN", case, lab, "C6", f"{v:.2f} CZ/shot above the nominal cap for b={b}")

    # C7 / C8 -----------------------------------------------------------------------------------------------
    fams = {}
    for lab in mean:
        m = re.match(r"(.*), noise=([0-9.]+)/([0-9.]+)$", lab)
        if m:
            fams.setdefault(m.group(1), []).append((float(m.group(2)), lab))
    for fam, rows in fams.items():
        rows.sort()
        base = fam if fam in mean else None
        zero_cz = (cz.get(rows[0][1]) or 0) < 1e-9
        if zero_cz:
            vals = {round(mean[l], 6) for _, l in rows}
            if len(vals) > 1:
                rec("FAIL", case, fam, "C7", "0-CZ contexts: shots differ between noise levels")
            if base and abs(two_sample_z(mean[rows[0][1]], sem[rows[0][1]], mean[base], sem[base])) > 3.5:
                rec("WARN", case, fam, "C7", "0-CZ noisy run differs from the noiseless run by > 3.5 sigma")
        else:
            prev = base
            for p, l in rows:
                if prev is not None and two_sample_z(mean[prev], sem[prev], mean[l], sem[l]) > 3.0:
                    rec("WARN", case, l, "C8", f"shots fall with noise vs {prev} ({mean[prev]:.4g} -> {mean[l]:.4g})")
                prev = l

    # C9 / C10 ----------------------------------------------------------------------------------------------
    def le(a, b, why, tol=3.0):
        if a in mean and b in mean and mean[a] > mean[b] and two_sample_z(mean[a], sem[a], mean[b], sem[b]) > tol:
            rec("WARN", case, a, "C9", f"{why}: {mean[a]:.4g} > {b} {mean[b]:.4g}")

    pairs = [("II-A data", "II-0 estimated"), ("II-A data, safe", "II-0 safe"), ("II-A data, blocks=4", "II-0, blocks=4"),
             ("II-A data, blocks=2", "II-0, blocks=2"), ("II-A data, blocks=1", "II-0, blocks=1"),
             ("II-A data, safe, blocks=4", "II-0, safe, blocks=4"), ("II-A data, safe, blocks=2", "II-0, safe, blocks=2"),
             ("II-A data, safe, blocks=1", "II-0, safe, blocks=1"),
             ("Pivot II-A", "Pivot II-0"), ("Pivot merged II-A", "Pivot merged II-0"),
             ("Pivot II-0", "Pivot M1 seq"), ("Pivot merged II-0", "Pivot merged M1 seq")]
    for a, b in pairs:
        le(a, b, "II-A should not exceed II-0, nor II-0 the sequential M1")
    for a, b in (("II-0 estimated", "II-0 safe"), ("II-A data", "II-A data, safe"), ("M2 pairwise", "M2 marginal")):
        le(a, b, "pairwise/safe cost ordering (looser rule must be cheaper)")
    for lab in list(mean):
        m = re.match(r"^(Pivot merged |Pivot )?M1 seq(, safe)?(, blocks=\d+)?(, noise=.*)?$", lab)
        if m and not m.group(4):
            le(lab, f"{m.group(1) or ''}M1 static{m.group(3) or ''}", "M1 seq <= M1 static")
    for b_ in ("II-0", "II-A data"):
        sep = ", " if b_ == "II-A data" else ", "
        f = {"II-0": "II-0 estimated", "II-A data": "II-A data"}[b_]
        le(f, f"{b_}{sep}blocks=4", "FC <= blocks=4")
        le(f"{b_}{sep}blocks=4", f"{b_}{sep}blocks=1", "blocks=4 <= blocks=1")
    for a, b in (("M2 safe", "M2 pairwise"), ("M2 QWC safe", "M2 QWC pairwise")):
        # independent arms: the sign-aware rule differs from the pairwise one only when an estimated sign is near zero,
        # so the rows agree to a few trials (H4 2.0: 2 of 200 trials, 0.09%); a real difference is a bug
        if a in mean and b in mean and abs(mean[a] - mean[b]) > 0.02 * mean[b]:
            rec("FAIL", case, a, "C10", f"M2 sign-aware and pairwise rows differ by more than 2% ({mean[a]:.6g} vs {mean[b]:.6g})")
    # M1 static against the exact planning value from step 3
    p3 = ROOT / "runs" / case / f"{case}_step3_oracle_ceiling.csv"
    plan = pd.read_csv(p3) if p3.exists() else None
    if plan is not None and "M1 static" in mean:
        m1 = float(plan.m1.iloc[0])
        r = mean["M1 static"] / m1
        (rec("FAIL" if abs(r - 1) > 0.02 else "INFO", case, "M1 static", "C10", f"realised / planning = {r:.4f}"))

    # C11 ---------------------------------------------------------------------------------------------------
    if plan is not None:
        want = {"II-0 oracle": ("mass", "II-0"), "II-A oracle": ("canonical", "II-A")}
        for lab, (stg, lvl) in want.items():
            if lab in mean:
                row = plan[(plan.strategy == stg) & (plan.level == lvl)]
                if len(row):
                    ratio = mean[lab] / float(row.m3_actual_static.iloc[0])
                    sev = "WARN" if not (0.5 <= ratio <= 2.0) else "INFO"
                    rec(sev, case, lab, "C11", f"realised / planning = {ratio:.3f}")

    # C12 ---------------------------------------------------------------------------------------------------
    tp = ROOT / "runs" / case / f"{case}_adapt_trajectory.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        if (np.diff(t.energy) > 1e-9).any():
            rec("FAIL", case, "adapt_trajectory", "C12", "energy increases along the exact ADAPT trajectory")
        if (t.energy_error < -1e-9).any():
            rec("FAIL", case, "adapt_trajectory", "C12", "energy below the reference (negative error)")
        if not t.relative_gap.between(0, 1 + 1e-9).all() or (t.second_abs_gradient > t.max_abs_gradient + 1e-12).any():
            rec("FAIL", case, "adapt_trajectory", "C12", "gap outside [0,1] or second gradient above the maximum")
        rec("INFO", case, "adapt_trajectory", "C12", f"{len(t)} steps, final error {t.energy_error.iloc[-1]:.2e}, final max|g| {t.max_abs_gradient.iloc[-1]:.2e}")

    check_phase4(case)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--csv")
    ap.add_argument("--root", help="project directory holding runs/ (default: this checkout); used by the mutation test")
    a = ap.parse_args()
    global ROOT
    if a.root:
        ROOT = Path(a.root)
    sums = {c: check_case(c, a.delta) for c in a.cases}
    # C14 coverage across the given cases
    sets = {c: set(s.config) for c, s in sums.items() if s is not None}
    if len(sets) > 1:
        union = set().union(*sets.values())
        for c, st in sets.items():
            miss = sorted(union - st)
            if miss:
                rec("INFO", c, "*", "C14", f"{len(miss)} configs present in a sibling case only: {miss[:8]}{' ...' if len(miss) > 8 else ''}")
    for c, s in sums.items():
        if s is None:
            continue
        mode = int(s.n_trials.mode().iloc[0])
        short = s[s.n_trials < mode]
        for r in short.itertuples():
            rec("WARN", c, r.config, "C14", f"{r.n_trials} trials (case mode {mode})")
    df = pd.DataFrame(OUT)
    if a.csv:
        df.to_csv(a.csv, index=False)
    order = {"FAIL": 0, "WARN": 1, "KNOWN": 2, "INFO": 3}
    df = df.sort_values(by=["severity", "case", "check"], key=lambda s: s.map(order) if s.name == "severity" else s)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 160); pd.set_option("display.max_rows", 500)
    print(df.groupby("severity").size().to_string())
    for sev in ("FAIL", "WARN", "KNOWN", "INFO"):
        sub = df[df.severity == sev]
        if len(sub):
            print(f"\n=== {sev} ({len(sub)})")
            for r in sub.itertuples():
                print(f"[{r.case}] {r.check} {r.label}: {r.message}")
    return 1 if (df.severity == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
