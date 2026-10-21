#!/usr/bin/env python3
"""Paper A: the headline comparison in one safe setting (sign-aware rule, a priori first radius, a raised minimum of shots).

Q9 of ``reports/draft.tex`` shows that the default minimum of 50 shots per context and fold (``n_min``, below which a held-out
covariance is replaced by its bound) lets the learned designs (II-A) underestimate their own variance on late ADAPT states, so
that II-A becomes dearer than II-0 there.  The cluster round ``ref3_minshots_*`` (2026-10-13) ran II-A with ``n_min = 400``
(a priori start, sign-aware rule, per-round intervals) on the 17 states of Table ``tab:apriori`` and on four ADAPT snapshots of H4.
This script compares those runs with the stored runs of the same states:

* II-A at ``n_min`` against II-A at the default of 50 (the price of the larger minimum);
* II-0 against II-A at ``n_min``; the II-0 reference is II-0 at the same ``n_min`` where such a run exists (``--ref matched``,
  the default) and II-0 at 50 otherwise; ``--ref default`` always uses II-0 at 50;
* the best published baseline against II-A at ``n_min``: the baseline of lowest mean cost among M1 seq, M2, M2 marginal and the
  reuse baseline of Ikhtiarudin et al. whose selections are at least 95% correct (a priori start, sign-aware rule where it
  applies); with ``--ref matched`` each baseline runs at the same ``n_min`` where such a run exists (round ``ref4_b400``,
  2026-10-19), else at its default of 50.

Every ratio is a ratio of mean context-shots with a 95% percentile interval from resampling the trials of numerator and
denominator independently (10,000 draws, fixed seed; ``paper_a_bootstrap.ratio_interval``).

Outputs: ``runs/paper_a/minshots_headline_n<n_min>[_default].csv`` (one row per state), ``runs/paper_a/minshots_facts_n<n_min>[_default].json`` (every number the draft
quotes from it), and the tabular-only body ``reports/generated/tab_minshots.tex`` (caption in the draft; paste with
``scripts/paper_a_inline.py``).

    python scripts/paper_a_minshots_headline.py [--nmin 400] [--ref matched|default]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_a_bootstrap import ratio_interval  # noqa: E402
from parallel import read_trials  # noqa: E402

SEED = 20261019
REPORTS = ROOT.parent / "reports"
# the 17 states of Table tab:apriori, in its order, then the four H4 snapshots of the 13-state comparison of Q9
STATES = [
    ("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0~\AA, CISD"),
    ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0~\AA, CISD"),
    ("LiH_R3p0_HF", r"LiH 3.0~\AA, HF"),
    ("LiH_R3p0_ADAPT3", r"LiH 3.0~\AA, ADAPT 3"),
    ("LiH_R3p0_ADAPT5", r"LiH 3.0~\AA, ADAPT 5"),
    ("H2O_eq_CISD", r"H$_2$O eq, CISD"),
    ("H2O_stretch_CISD", r"H$_2$O str, CISD"),
    ("H2O_eq_ADAPT11", r"H$_2$O eq, ADAPT 11"),
    ("H2O_stretch_ADAPT8", r"H$_2$O str, ADAPT 8"),
    ("LiH_R1p6_HF", r"LiH 1.6~\AA, HF"),
    ("LiH_R1p6_ADAPT2", r"LiH 1.6~\AA, ADAPT 2"),
    ("LiH_R1p6_ADAPT3", r"LiH 1.6~\AA, ADAPT 3"),
    ("LiH_R1p6_ADAPT4", r"LiH 1.6~\AA, ADAPT 4"),
    ("BeH2_ADAPT2", r"BeH$_2$, ADAPT 2"),
    ("BeH2_ADAPT4", r"BeH$_2$, ADAPT 4"),
    ("BeH2_ADAPT7", r"BeH$_2$, ADAPT 7"),
    ("BeH2_ADAPT9", r"BeH$_2$, ADAPT 9"),
    ("H4_square_eq_side1p0_ADAPT3", r"H$_4$ 1.0~\AA, ADAPT 3"),
    ("H4_square_eq_side1p0_ADAPT5", r"H$_4$ 1.0~\AA, ADAPT 5"),
    ("H4_square_stretch_side2p0_ADAPT2", r"H$_4$ 2.0~\AA, ADAPT 2"),
    ("H4_square_stretch_side2p0_ADAPT5", r"H$_4$ 2.0~\AA, ADAPT 5"),
]
SNAPSHOTS = {s for s, _ in STATES[17:]}
II_A_DEFAULT = ("step5", "II-A data, safe, bound start")
II_0_DEFAULT = ("step4", "II-0 safe, bound start")
BASELINES = [("M1 seq", "step5", "M1 seq, safe, bound start"), ("M2", "step5", "M2 safe, bound start"),
             ("M2 marg.", "step5", "M2 marginal, bound start"), ("Ikh", "step5", "Ikh reuse, FC, safe, bound start")]
VALID = 0.95  # a baseline counts as "best" only if at least this share of its selections is correct


def ii_a_labels(n: int) -> list[str]:
    return [f"II-A data, safe, bound start, min shots={n}, diagnose", f"II-A data, safe, bound start, min shots={n}"]


def ii_0_labels(n: int) -> list[str]:
    return [f"II-0, safe, bound start, min shots={n}, diagnose", f"II-0, safe, bound start, min shots={n}"]


def load(case: str) -> dict[tuple[str, str], tuple[np.ndarray, float]]:
    """``(step, label) -> (shots of every trial, share correct)``; only complete label sets."""
    out = {}
    for step in ("step4", "step5"):
        for label, rows in read_trials(ROOT / "runs", case, step).items():
            if len(rows) < int(rows[0]["n_trials"]):
                continue
            out[(step, label)] = (np.array([float(r["shots"]) for r in rows]), float(np.mean([int(r["correct"]) for r in rows])))
    return out


def first(trials: dict, labels: list[str]):
    for label in labels:
        if ("step5", label) in trials:
            return label, trials[("step5", label)]
    return None, None


def sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 2) >= 10:
        m, e = m / 10, e + 1
    return r"$" + f"{m:.2f}" + r"\times10^{" + str(e) + r"}$"


def cell(r: float, lo: float, hi: float) -> str:
    f = (lambda v: f"{v:.1f}") if hi >= 10 else (lambda v: f"{v:.2f}")
    return f"{f(r)} [{f(lo)}, {f(hi)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmin", type=int, default=400)
    parser.add_argument("--ref", choices=("matched", "default"), default="matched")
    parser.add_argument("--latex", type=Path, default=REPORTS / "generated" / "tab_minshots.tex")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows, lines = [], []
    for case, name in STATES:
        t = load(case)
        label_a, ours = first(t, ii_a_labels(args.nmin))
        if ours is None:
            print(f"{case}: no II-A run at min shots {args.nmin}")
            continue
        a, a_correct = ours
        a50, _ = t[II_A_DEFAULT]
        label_0, matched = first(t, ii_0_labels(args.nmin)) if args.ref == "matched" else (None, None)
        ref_label, (r0, r0_correct) = (label_0, matched) if matched is not None else (II_0_DEFAULT[1], t[II_0_DEFAULT])
        base, base_matched = [], []
        for bname, step, lab in BASELINES:
            raised = (step, f"{lab}, min shots={args.nmin}")
            if args.ref == "matched" and raised in t:
                base.append((bname, t[raised]))
                base_matched.append(bname)
            elif (step, lab) in t:
                base.append((bname, t[(step, lab)]))
        valid = [(bname, s) for bname, (s, c) in base if c >= VALID]
        best_name, best = min(valid, key=lambda item: item[1].mean())
        price = ratio_interval(a, a50, "mean", rng)
        vs0 = ratio_interval(r0, a, "mean", rng)
        vsb = ratio_interval(best, a, "mean", rng)
        row = {"case": case, "snapshot": case in SNAPSHOTS, "n_min": args.nmin, "trials": a.size, "ii_a_mean": a.mean(),
               "ii_a_correct": a_correct, "ii_a50_mean": a50.mean(), "price": price[0], "price_lo": price[1], "price_hi": price[2],
               "ii0_label": ref_label, "ii0_matched": matched is not None, "ii0_trials": r0.size, "ii0_mean": r0.mean(),
               "ii0_correct": r0_correct, "ii0_over_iia": vs0[0], "ii0_lo": vs0[1], "ii0_hi": vs0[2],
               "baselines_matched": ";".join(base_matched), "best_matched": best_name in base_matched,
               "best_baseline": best_name, "best_trials": best.size, "best_mean": best.mean(),
               "best_over_iia": vsb[0], "best_lo": vsb[1], "best_hi": vsb[2],
               "invalid_baselines": ";".join(f"{b}:{c:.2f}" for b, (s, c) in base if c < VALID),
               "min_correct_baselines": min(c for b, (s, c) in base)}
        # what the larger minimum costs the comparators: II-0 and the best valid baseline, each against its own default of 50
        default_valid = [t[(step, lab)][0] for _, step, lab in BASELINES if (step, lab) in t and t[(step, lab)][1] >= VALID]
        row["ii0_price"] = r0.mean() / t[II_0_DEFAULT][0].mean()
        row["best_price"] = best.mean() / min(s.mean() for s in default_valid)
        rows.append(row)
        lines.append(f"{name} & {sci(a.mean())} & {price[0]:.2f} & {cell(*vs0)} & {cell(*vsb)} & {best_name} \\\\")
        print(f"{case:34s} II-A {a.mean():.3e} ({a.size}, {a_correct:.2f})  x{price[0]:.2f} of n=50  "
              f"II-0[{'n' if matched is not None else '50'}]/II-A {vs0[0]:.2f} [{vs0[1]:.2f},{vs0[2]:.2f}]  "
              f"{best_name}/II-A {vsb[0]:.2f} [{vsb[1]:.2f},{vsb[2]:.2f}]  invalid: {row['invalid_baselines']}")
        if case == STATES[16][0]:
            lines.append(r"\midrule")
    # tabular body
    body = [r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
            rf"State & II-A & II-A$_{{{args.nmin}}}$ / II-A$_{{50}}$ & II-0 / II-A & best baseline / II-A & baseline \\", r"\midrule", *lines,
            r"\bottomrule", r"\end{tabular}", ""]
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text("\n".join(body), encoding="utf-8", newline="\n")
    out = ROOT / "runs" / "paper_a"
    tag = "" if args.ref == "matched" else "_default"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / f"minshots_headline_n{args.nmin}{tag}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # quoted numbers
    main17 = [r for r in rows if not r["snapshot"]]
    facts = {"n_min": args.nmin, "ref": args.ref, "states": len(rows), "states_main": len(main17),
             "all_correct_iia": all(r["ii_a_correct"] == 1.0 for r in rows),
             "min_correct_iia": min(r["ii_a_correct"] for r in rows)}
    for key, sub in (("all", rows), ("main17", main17)):
        win0 = [r for r in sub if r["ii0_lo"] > 1]
        lose0 = [r for r in sub if r["ii0_hi"] < 1]
        winb = [r for r in sub if r["best_lo"] > 1]
        loseb = [r for r in sub if r["best_hi"] < 1]
        facts[key] = {
            "ii0_over_iia_range": [min(r["ii0_over_iia"] for r in sub), max(r["ii0_over_iia"] for r in sub)],
            "ii0_matched": [r["case"] for r in sub if r["ii0_matched"]],
            "baselines_all_matched": all(r["best_matched"] for r in sub),
            "ii0_over_iia_where_above_one": sorted(r["ii0_over_iia"] for r in sub if r["ii0_over_iia"] > 1),
            "ii0_over_iia_where_below_one": sorted(r["ii0_over_iia"] for r in sub if r["ii0_over_iia"] < 1),
            "best_over_iia_where_below_one": sorted(r["best_over_iia"] for r in sub if r["best_over_iia"] < 1),
            "best_over_iia_where_above_one": sorted(r["best_over_iia"] for r in sub if r["best_over_iia"] > 1),
            "ii0_significantly_dearer": [r["case"] for r in win0], "ii0_significantly_cheaper": [r["case"] for r in lose0],
            "best_over_iia_range": [min(r["best_over_iia"] for r in sub), max(r["best_over_iia"] for r in sub)],
            "best_significantly_dearer": [r["case"] for r in winb], "best_significantly_cheaper": [r["case"] for r in loseb],
            "price_range": [min(r["price"] for r in sub), max(r["price"] for r in sub)],
            "price_where_above_one": sorted(r["price"] for r in sub if r["price"] > 1),
            "price_where_below_one": sorted(r["price"] for r in sub if r["price"] < 1),
            "ii0_over_iia_quartiles": [float(q) for q in np.percentile([r["ii0_over_iia"] for r in sub], [25, 50, 75])],
            "best_over_iia_quartiles": [float(q) for q in np.percentile([r["best_over_iia"] for r in sub], [25, 50, 75])],
            "trials_iia": int(sum(r["trials"] for r in sub)),
            "min_correct_ii0": min(r["ii0_correct"] for r in sub),
            "ii0_price_range": [min(r["ii0_price"] for r in sub), max(r["ii0_price"] for r in sub)],
            "best_price_range": [min(r["best_price"] for r in sub), max(r["best_price"] for r in sub)],
            "min_correct_baselines": min(r["min_correct_baselines"] for r in sub),
        }
    facts["rows"] = {r["case"]: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "case"} for r in rows}
    (out / f"minshots_facts_n{args.nmin}{tag}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in facts.items() if k != "rows"}, indent=1))
    print(f"wrote {args.latex}, {out}/minshots_headline_n{args.nmin}{tag}.csv, minshots_facts_n{args.nmin}{tag}.json")


if __name__ == "__main__":
    main()
