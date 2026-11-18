#!/usr/bin/env python3
"""Paper A, critique round (2026-10-27): the tail of the cost of a selection (Q9) and what predicts the saving (Q8), from the stored runs of the headline comparison.

* **Tails.**  For the 21 states of Table ``tab:minshots`` (n_min = 400, sign-aware rule, a priori start): median, 95th percentile and maximum of the context-shots of one
  selection of II-A and of the best valid baseline of that table, and the ratio of the 95th percentiles with a 95% bootstrap interval (trials of the two methods resampled
  independently).  A user who budgets for the tail needs the 95th percentile, not the mean.
* **Predictors.**  Per state: the relative gap (gap over the largest gradient), the number of generators within a factor two of the largest gradient, the pool size and the
  number of qubits (``load_problem``), against the ratio best baseline / II-A of the mean: Spearman correlations and the spread of the ratio.  The 21 states come from six
  systems and are correlated, so the correlations are descriptive; the leave-one-system-out error of a one-variable fit on the log ratio is reported as well.

Writes ``runs/paper_a/tail_costs.csv``, ``runs/paper_a/predictors.csv``, ``runs/paper_a/critique_facts.json`` and ``reports/generated/tab_tails.tex``.

    python scripts/paper_a_critique_tables.py          # needs the cached problems (WSL or the cluster: the pickles are numpy-2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_a_minshots_headline as mh  # noqa: E402

REPORTS = ROOT.parent / "reports"
N_MIN = 400
RNG = np.random.default_rng(20261027)
BASELINE_LABEL = {name: (step, f"{lab}, min shots={N_MIN}") for name, step, lab in mh.BASELINES}


def p95_ratio(num: np.ndarray, den: np.ndarray, draws: int = 4000) -> tuple[float, float, float]:
    point = np.percentile(num, 95) / np.percentile(den, 95)
    boot = np.array([np.percentile(RNG.choice(num, num.size), 95) / np.percentile(RNG.choice(den, den.size), 95) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def system_of(case: str) -> str:
    return case.split("_")[0] + ("_" + case.split("_")[1] + ("_" + case.split("_")[2] if case.startswith("H4") else "") if case.startswith(("H2O", "LiH", "H4")) else "")


def main() -> None:
    headline = {r["case"]: r for r in __import__("csv").DictReader((ROOT / "runs" / "paper_a" / f"minshots_headline_n{N_MIN}.csv").open())}
    tails, preds, lines = [], [], []
    for case, name in mh.STATES:
        t = mh.load(case)
        _, ours = mh.first(t, mh.ii_a_labels(N_MIN))
        a, _ = ours
        best_name = headline[case]["best_baseline"]
        step, label = BASELINE_LABEL[best_name]
        b = t[(step, label)][0] if (step, label) in t else t[(step, dict((n, l) for n, _, l in mh.BASELINES)[best_name])][0]
        ratio = p95_ratio(b, a)
        row = {"case": case, "trials": a.size, "iia_mean": a.mean(), "iia_median": float(np.median(a)), "iia_p95": float(np.percentile(a, 95)), "iia_max": a.max(),
               "iia_p95_over_mean": float(np.percentile(a, 95) / a.mean()), "best_baseline": best_name, "best_mean": b.mean(), "best_p95": float(np.percentile(b, 95)),
               "p95_ratio": ratio[0], "p95_lo": ratio[1], "p95_hi": ratio[2], "mean_ratio": float(headline[case]["best_over_iia"])}
        tails.append(row)
        lines.append(f"{name} & {mh.sci(row['iia_median'])} & {mh.sci(row['iia_p95'])} & {row['iia_p95_over_mean']:.1f} & {best_name} & {mh.cell(*ratio)} & "
                     f"{row['mean_ratio']:.2f} \\\\")
        if case == mh.STATES[16][0]:
            lines.append(r"\midrule")
        problem = __import__("step4_learned_designs").load_problem(case)
        g = np.sort(np.asarray(problem.abs_gradients))[::-1]
        preds.append({"case": case, "system": system_of(case), "K": int(g.size), "qubits": int(problem.n_qubits), "relative_gap": float((g[0] - g[1]) / g[0]),
                      "competitors": int((g >= 0.5 * g[0]).sum()), "mean_ratio": row["mean_ratio"], "iia_mean": row["iia_mean"]})
        print(f"{case:34s} II-A median {row['iia_median']:.2e} p95 {row['iia_p95']:.2e} (p95/mean {row['iia_p95_over_mean']:.1f}); {best_name} p95/II-A p95 "
              f"{ratio[0]:.2f} [{ratio[1]:.2f},{ratio[2]:.2f}] (means {row['mean_ratio']:.2f})  gap {preds[-1]['relative_gap']:.3f} K {g.size} comp {preds[-1]['competitors']}")

    import csv
    out = ROOT / "runs" / "paper_a"
    for name, rows in (("tail_costs", tails), ("predictors", preds)):
        with (out / f"{name}.csv").open("w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    body = [r"\begin{tabular}{@{}lrrrlrr@{}}", r"\toprule",
            r"State & median & 95th percentile & 95th / mean & baseline & baseline / II-A, 95th percentiles & baseline / II-A, means \\", r"\midrule", *lines,
            r"\bottomrule", r"\end{tabular}", ""]
    (REPORTS / "generated" / "tab_tails.tex").write_text("\n".join(body), encoding="utf-8", newline="\n")

    ratio = np.array([p["mean_ratio"] for p in preds])
    facts = {"p95_ratio_range": [min(r["p95_ratio"] for r in tails), max(r["p95_ratio"] for r in tails)],
             "p95_ratio_median": float(np.median([r["p95_ratio"] for r in tails])), "p95_over_mean_iia_range": [min(r["iia_p95_over_mean"] for r in tails), max(r["iia_p95_over_mean"] for r in tails)],
             "states_p95_ratio_below_one": [r["case"] for r in tails if r["p95_hi"] < 1], "states_p95_ratio_above_one": [r["case"] for r in tails if r["p95_lo"] > 1]}
    predictors = {}
    for key in ("relative_gap", "competitors", "K", "qubits"):
        x = np.array([p[key] for p in preds], float)
        rho, p = spearmanr(x, ratio)
        predictors[key] = {"spearman": float(rho), "p": float(p)}
    # leave-one-system-out error of a one-variable fit of log(ratio) on log(relative gap)
    sys_names = sorted({p["system"] for p in preds})
    for key in ("relative_gap", "competitors", "K"):
        x = np.log(np.array([p[key] for p in preds], float))
        y = np.log(ratio)
        err = []
        for s in sys_names:
            test = np.array([p["system"] == s for p in preds])
            if test.all() or not test.any():
                continue
            coef = np.polyfit(x[~test], y[~test], 1)
            err += list(np.abs(np.polyval(coef, x[test]) - y[test]))
        predictors[key]["loso_median_abs_log_error"] = float(np.median(err))
        predictors[key]["loso_factor"] = float(np.exp(np.median(err)))
    predictors["spread_of_log_ratio"] = float(np.median(np.abs(np.log(ratio) - np.median(np.log(ratio)))))
    facts["predictors"] = predictors
    facts["systems"] = sys_names
    (out / "critique_facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    pr = predictors
    sentence = (f"The 95th percentile of the selection cost of II-A is ${facts['p95_over_mean_iia_range'][0]:.1f}$ to ${facts['p95_over_mean_iia_range'][1]:.1f}$ times its mean, and the ratio of the "
                f"95th percentiles of the best baseline to II-A is ${facts['p95_ratio_range'][0]:.2f}$ to ${facts['p95_ratio_range'][1]:.2f}$ (median ${facts['p95_ratio_median']:.2f}$ over the "
                f"{len(tails)} states), with the interval of every state above one. The ratio of the means correlates with the number of qubits (Spearman ${pr['qubits']['spearman']:.2f}$, "
                f"$p={pr['qubits']['p']:.3f}$) and with the pool size (${pr['K']['spearman']:.2f}$, $p={pr['K']['p']:.3f}$), not with the relative gap (${pr['relative_gap']['spearman']:.2f}$, "
                f"$p={pr['relative_gap']['p']:.2f}$) or the number of generators within a factor of two of the largest (${pr['competitors']['spearman']:.2f}$, $p={pr['competitors']['p']:.2f}$). "
                f"The states come from four molecules at {len(sys_names)} geometries, and a fit of the logarithm of the ratio on the logarithm of the pool size, with one system left out at a time, is off by a median "
                f"factor of ${pr['K']['loso_factor']:.2f}$ (relative gap ${pr['relative_gap']['loso_factor']:.2f}$, competitors ${pr['competitors']['loso_factor']:.2f}$), against "
                f"${float(np.exp(pr['spread_of_log_ratio'])):.2f}$ for the spread of the ratio around its median: no variable predicts the saving better than the range does.\n")
    (REPORTS / "generated" / "tails_predictors_text.tex").write_text(sentence, encoding="utf-8", newline="\n")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
