#!/usr/bin/env python3
"""Paper A: bootstrap intervals for every headline ratio, from the stored per-trial files.

A ratio of two mean costs (or two medians) is a ratio of two random numbers; the paper quoted
it without an interval, and the costs have heavy tails (mean 7.9e5 against median 2.7e4 on one
pairwise row).  Here every ratio comes with a percentile interval from resampling the *trials*
(fixed states) or the *trajectories* (ADAPT runs) of numerator and denominator independently
(10,000 draws, fixed seed).  Trials of different methods use independent random streams, so an
unpaired bootstrap is the right resampling.

Ratios
    fixed states, per setting (``oracle start`` pairwise, ``sign-aware`` oracle start,
    ``a priori`` sign-aware with the bound start):
      * ours (II-A) against II-0                       -- the saving of the learned designs
      * every baseline against II-A                    -- the gain over the baselines
      * the strongest baseline (smallest mean) against II-A
    trajectories: every method against II-A safe, total selection shots to chemical accuracy.

Statistics: the ratio of means (the cost of a campaign adds up trial costs) and the ratio of
medians; a row whose relative half-width of the mean ratio exceeds ``--wide`` is flagged, so
that trials can be added where the interval is wide.

Outputs: ``runs/paper_a/bootstrap_fixed_state.csv``, ``runs/paper_a/bootstrap_trajectories.csv``
and (with ``--latex``) the headline table as a LaTeX fragment.

    python scripts/paper_a_bootstrap.py [--latex ../reports/generated/tab_headline_bootstrap.tex]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_trials import read_trials  # noqa: E402
from outputs import write_csv  # noqa: E402

DRAWS = 10_000
SEED = 20261009
CASES = ["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD", "LiH_R3p0_HF", "LiH_R3p0_ADAPT3",
         "LiH_R3p0_ADAPT5", "H2O_eq_CISD", "H2O_stretch_CISD", "H2O_eq_ADAPT11", "H2O_stretch_ADAPT8"]

# setting -> [(display name, step, stored label, role)]; role "ours" or "baseline" ("learned-II-0" is the reference)
ROWS = {
    "oracle start, pairwise rule": [
        ("II-0", "step4", "II-0 estimated", "reference"),
        ("II-A", "step4", "II-A data", "ours"),
        ("M1 seq", "step5", "M1 seq", "baseline"),
        ("M2 pairwise", "step5", "M2 pairwise", "baseline"),
        ("M2 marginal", "step5", "M2 marginal", "baseline"),
        ("Ikh reuse, FC", "step5", "Ikh reuse, FC", "baseline"),
        ("Ikh reuse, QWC", "step5", "Ikh reuse, QWC", "baseline"),
    ],
    "oracle start, sign-aware rule": [
        ("II-0", "step4", "II-0 safe", "reference"),
        ("II-A", "step5", "II-A data, safe", "ours"),
        ("M1 seq", "step5", "M1 seq, safe", "baseline"),
        ("M2", "step5", "M2 safe", "baseline"),
        ("Ikh reuse, FC", "step5", "Ikh reuse, FC, safe", "baseline"),
    ],
    "a priori start, sign-aware rule": [
        ("II-0", "step4", "II-0 safe, bound start", "reference"),
        ("II-A", "step5", "II-A data, safe, bound start", "ours"),
        ("M1 seq", "step5", "M1 seq, safe, bound start", "baseline"),
        ("M2", "step5", "M2 safe, bound start", "baseline"),
        ("M2 marginal", "step5", "M2 marginal, bound start", "baseline"),
        ("Ikh reuse, FC", "step5", "Ikh reuse, FC, safe, bound start", "baseline"),
    ],
}
TRAJECTORY_REFERENCE = "II-A data, safe"
TRAJECTORY_ORDER = ["II-0 safe", "M1 seq, safe", "M2 safe", "M2 marginal", "M1 static", "Ikh reuse, FC",
                    "II-0 safe + reuse, FC", "II-A data, safe + reuse, FC"]


def ratio_interval(num: np.ndarray, den: np.ndarray, stat: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval of ``stat(num) / stat(den)`` (independent resampling)."""
    f = np.mean if stat == "mean" else np.median
    point = float(f(num) / f(den))
    ia = rng.integers(0, num.size, size=(DRAWS, num.size))
    ib = rng.integers(0, den.size, size=(DRAWS, den.size))
    a = f(num[ia], axis=1)
    b = f(den[ib], axis=1)
    ratios = a / b
    return point, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def load_trials(runs: Path, case: str) -> dict[tuple[str, str], tuple[np.ndarray, float]]:
    """``(step, label) -> (shots of every trial, share correct)`` of one case (complete label sets only)."""
    out = {}
    for step in ("step4", "step5"):
        try:
            labels = read_trials(runs, case, step)
        except (SystemExit, FileNotFoundError):
            continue
        for label, rows in labels.items():
            n = int(rows[0]["n_trials"])
            if len(rows) < n:
                continue
            out[(step, label)] = (np.array([float(r["shots"]) for r in rows]),
                                  float(np.mean([int(r["correct"]) for r in rows])))
    return out


def fixed_state(runs: Path, wide: float) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for case in CASES:
        if not (runs / case).exists():
            continue
        trials = load_trials(runs, case)
        for setting, rows in ROWS.items():
            have = [(name, trials[(step, label)], role) for name, step, label, role in rows if (step, label) in trials]
            roles = {role: [(n, t) for n, t, r in have if r == role] for role in ("reference", "ours", "baseline")}
            if not roles["ours"]:
                continue
            ours_name, (ours, ours_correct) = roles["ours"][0]
            pairs = []
            if roles["reference"]:
                ref_name, (ref, _) = roles["reference"][0]
                pairs.append((f"{ref_name} / {ours_name}", ref, ours, "reference over ours"))
                pairs.append((f"{ours_name} / {ref_name}", ours, ref, "saving of ours"))
            baselines = [(n, t) for n, t in roles["baseline"]]
            for name, (shots, _) in baselines:
                pairs.append((f"{name} / {ours_name}", shots, ours, "baseline over ours"))
            if baselines:
                best_name, (best, _) = min(baselines, key=lambda item: item[1][0].mean())
                pairs.append((f"best baseline ({best_name}) / {ours_name}", best, ours, "best baseline over ours"))
            for label, num, den, kind in pairs:
                for stat in ("mean", "median"):
                    point, lo, hi = ratio_interval(num, den, stat, rng)
                    out.append({"case": case, "setting": setting, "ratio": label, "kind": kind, "stat": stat,
                                "value": point, "lo": lo, "hi": hi, "rel_halfwidth": (hi - lo) / (2 * point),
                                "n_num": num.size, "n_den": den.size,
                                "wide": bool(stat == "mean" and (hi - lo) / (2 * point) > wide)})
            # share of correct selections of the ours row: needed to read the rows honestly
            out.append({"case": case, "setting": setting, "ratio": f"correct share of {ours_name}", "kind": "correct",
                        "stat": "share", "value": ours_correct, "lo": float("nan"), "hi": float("nan"),
                        "rel_halfwidth": float("nan"), "n_num": ours.size, "n_den": 0, "wide": False})
    return out


def trajectories(runs: Path, wide: float) -> list[dict]:
    rng = np.random.default_rng(SEED + 1)
    out = []
    for case_dir in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "phase4").exists()):
        case = case_dir.name
        totals: dict[str, np.ndarray] = {}
        paths = sorted((case_dir / "phase4").glob(f"{case}_phase4*_trajectories.csv"))
        # main runs first (a method found in several files keeps the last), then the extra trajectories (other seed) pooled with them
        for path in [p for p in paths if "_phase4_extra" not in p.name] + [p for p in paths if "_phase4_extra" in p.name]:
            if f"{case}_phase4_sens_" in path.name or f"{case}_phase4_fixed_" in path.name:
                continue
            rows = list(csv.DictReader(path.open()))
            by = defaultdict(list)
            for r in rows:
                by[r["method"]].append(float(r["total_shots"]))
            for method, values in by.items():
                if method.startswith("Fixed "):
                    continue
                if "_phase4_extra" in path.name and method in totals:
                    totals[method] = np.concatenate([totals[method], np.array(values)])
                else:
                    totals[method] = np.array(values)
        if TRAJECTORY_REFERENCE not in totals:
            continue
        ours = totals[TRAJECTORY_REFERENCE]
        for method in [m for m in TRAJECTORY_ORDER if m in totals]:
            for stat in ("mean", "median"):
                point, lo, hi = ratio_interval(totals[method], ours, stat, rng)
                out.append({"case": case, "ratio": f"{method} / {TRAJECTORY_REFERENCE}", "stat": stat, "value": point,
                            "lo": lo, "hi": hi, "rel_halfwidth": (hi - lo) / (2 * point),
                            "n_num": totals[method].size, "n_den": ours.size,
                            "wide": bool(stat == "mean" and (hi - lo) / (2 * point) > wide)})
    return out


def _fmt(value: float) -> str:
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def headline_latex(rows: list[dict], setting: str) -> str:
    """One table: per state, the saving of II-A over II-0 and the best baseline over II-A, with intervals."""
    short = {"H4_square_eq_side1p0_CISD": "H$_4$ 1.0 \\AA, CISD", "H4_square_stretch_side2p0_CISD": "H$_4$ 2.0 \\AA, CISD",
             "LiH_R3p0_HF": "LiH, HF", "LiH_R3p0_ADAPT3": "LiH, ADAPT 3", "LiH_R3p0_ADAPT5": "LiH, ADAPT 5",
             "H2O_eq_CISD": "H$_2$O eq, CISD", "H2O_stretch_CISD": "H$_2$O str, CISD",
             "H2O_eq_ADAPT11": "H$_2$O eq, ADAPT 11", "H2O_stretch_ADAPT8": "H$_2$O str, ADAPT 8"}
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"State & II-0 / II-A & best baseline / II-A & correct (II-A) \\", r"\midrule"]
    for case in CASES:
        mine = [r for r in rows if r["case"] == case and r["setting"] == setting]
        saving = next((r for r in mine if r["kind"] == "reference over ours" and r["stat"] == "mean"), None)
        best = next((r for r in mine if r["kind"] == "best baseline over ours" and r["stat"] == "mean"), None)
        correct = next((r for r in mine if r["kind"] == "correct"), None)
        if saving is None:
            continue
        cell = lambda r: "--" if r is None else f"{_fmt(r['value'])} [{_fmt(r['lo'])}, {_fmt(r['hi'])}]"  # noqa: E731
        share = "--" if correct is None else f"{100 * correct['value']:.0f}\\%"
        lines.append(f"{short[case]} & {cell(saving)} & {cell(best)} & {share} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--wide", type=float, default=0.15, help="flag mean ratios whose relative half-width exceeds this")
    parser.add_argument("--latex", type=Path, default=None, help="write the headline table (a priori, sign-aware) here")
    args = parser.parse_args()
    fixed = fixed_state(args.runs, args.wide)
    traj = trajectories(args.runs, args.wide)
    out = args.runs / "paper_a"
    if fixed:
        write_csv(out / "bootstrap_fixed_state.csv", list(fixed[0]), fixed)
    if traj:
        write_csv(out / "bootstrap_trajectories.csv", list(traj[0]), traj)
    flagged = [r for r in fixed + traj if r["wide"]]
    print(f"{len(fixed)} fixed-state and {len(traj)} trajectory intervals; {len(flagged)} with relative half-width > {args.wide:.0%}")
    for r in flagged:
        print(f"  wide: {r['case']:32s} {r.get('setting', 'trajectories'):34s} {r['ratio']:50s} {r['value']:8.2f} "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]  n = {r['n_num']}/{r['n_den']}")
    for r in fixed:
        if r["kind"] in ("saving of ours",) and r["stat"] == "mean":
            print(f"{r['case']:32s} {r['setting']:32s} {r['ratio']:14s} {r['value']:6.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(headline_latex(fixed, "a priori start, sign-aware rule"), newline="\n")
        print(f"wrote {args.latex}")


if __name__ == "__main__":
    main()
