#!/usr/bin/env python3
"""Tables and figure of the policy experiments (``energy_trajectory_policies.py``): what does a worse choice do to the trajectory?

For every case and policy (selection with injected errors, exact gradients):

* trajectories, share that reach the target (chemical accuracy, or what exact ADAPT reaches when it stalls above it),
  steps (mean, standard error) and the share of trajectories that need more steps than exact ADAPT;
* optimiser evaluations (mean, standard error) as a ratio to *tie-averaged* exact ADAPT (``rho_random:0``: random choice among
  the generators with the same |g|), because the evaluation count of the single exact path depends on how ties are broken
  (an arbitrary reference that moves the ratio by tens of percent); the ratio to the single exact path is kept as well;
* the energy error after each step against exact ADAPT's (mean over trajectories of the area under the curve over the first
  ``exact steps`` steps, as a ratio);
* the mean gradient shortfall of the choices and the mean ratio of their one-parameter lowering to the best one.

    python scripts/energy_policies_report.py [--runs runs/energy_policies] [--cases ...] [--evals-runs runs/energy_policies_gtol1e-5_cl]

Writes ``<runs>/report/policies_<case>.csv``, ``policies_all.csv`` and ``dose_response.pdf/png``.  With ``--evals-runs`` the bottom row of the figure (evaluations)
is taken from the report of another run set (the same policies at a looser optimiser tolerance: at the tight one the count carries the path noise of the
BFGS polishing) and the tight-tolerance values of the random draw are kept as a dotted grey line.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#666666"
SLACK = 1e-4  # Ha: a stalled trajectory has reached what exact ADAPT reaches when it is within this of it
LABEL = {"H4_square_eq_side1p0_HF": "H4 1.0", "H4_square_stretch_side2p0_HF": "H4 2.0", "LiH_R3p0_HF": "LiH 3.0",
         "BeH2_HF": "BeH2", "H2O_eq_HF": "H2O eq", "H2O_stretch_HF": "H2O str",
         "H2O_dz8o_eq_HF": "H2O dz8o eq", "H2O_dz8o_str_HF": "H2O dz8o str"}


def load(case_dir: Path):
    ref = json.loads((case_dir / "exact_reference.json").read_text())
    out = {}
    for path in sorted(case_dir.glob("policy_*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if rows:
            out[rows[0]["policy"]] = rows
    return ref, out


def se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarise(case: str, ref: dict, policies: dict) -> pd.DataFrame:
    exact_steps = ref["iterations"]
    target = max(1.6e-3, ref["final_error"] + SLACK) if ref["final_error"] >= 1.6e-3 else 1.6e-3
    exact_curve = np.array(ref["energy_errors"])
    base_rows = policies.get("rho_random:0")
    base_evals = np.mean([r["evals_total"] for r in base_rows]) if base_rows else float("nan")
    base_weighted = np.mean([r["evals_weighted"] for r in base_rows]) if base_rows else float("nan")
    exact_evals = sum(ref["evaluations"])
    exact_weighted = float("nan")
    if "exact" in policies:
        exact_weighted = policies["exact"][0]["evals_weighted"]
    out = []
    for spec, rows in sorted(policies.items(), key=lambda kv: order_key(kv[0])):
        steps = np.array([r["iterations"] for r in rows], float)
        evals = np.array([r["evals_total"] for r in rows], float)
        weighted = np.array([r["evals_weighted"] for r in rows], float)
        final = np.array([r["final_error"] for r in rows], float)
        n_ref = min(exact_steps, 10 ** 6)
        auc = []
        for r in rows:
            e = np.array(r["energy_errors"], float)
            e = np.concatenate([e, np.full(max(0, n_ref + 1 - len(e)), e[-1])])[: n_ref + 1]
            auc.append(e.sum())
        ref_auc = exact_curve[: n_ref + 1].sum()
        shortfall = np.concatenate([r["g_shortfall"] for r in rows]) if rows else np.zeros(1)
        ratios = np.concatenate([r["lowering_ratio"] for r in rows]) if rows else np.ones(1)
        out.append({
            "case": case, "system": LABEL.get(case, case), "policy": spec, "n": len(rows),
            "reached_pct": 100 * float(np.mean(final <= target)),
            "steps_mean": steps.mean(), "steps_se": se(steps), "exact_steps": exact_steps,
            "steps_ratio": steps.mean() / exact_steps,
            "more_steps_pct": 100 * float(np.mean(steps > exact_steps)), "fewer_steps_pct": 100 * float(np.mean(steps < exact_steps)),
            "evals_mean": evals.mean(), "evals_se": se(evals),
            "evals_ratio_tie_avg": evals.mean() / base_evals if base_evals == base_evals else float("nan"),
            "evals_ratio_tie_avg_se": se(evals) / base_evals if base_evals == base_evals else float("nan"),
            "evals_ratio_exact_path": evals.mean() / exact_evals,
            "weighted_ratio_tie_avg": weighted.mean() / base_weighted if base_weighted == base_weighted else float("nan"),
            "final_error_median_mHa": 1e3 * float(np.median(final)),
            "auc_ratio": float(np.mean(auc)) / ref_auc,
            "err20_mHa": 1e3 * float(np.mean([r["energy_errors"][min(20, len(r["energy_errors"]) - 1)] for r in rows])),
            "err_exact_steps_mHa": 1e3 * float(np.mean([r["energy_errors"][min(exact_steps, len(r["energy_errors"]) - 1)] for r in rows])),
            "exact_err20_mHa": 1e3 * float(exact_curve[min(20, len(exact_curve) - 1)]),
            "exact_err_exact_steps_mHa": 1e3 * float(exact_curve[-1]),
            "err_ratio_exact_steps": float(np.mean([r["energy_errors"][min(exact_steps, len(r["energy_errors"]) - 1)] for r in rows]) / exact_curve[-1]),
            "g_shortfall_mean": float(shortfall.mean()), "lowering_ratio_mean": float(ratios.mean()),
            "lowering_ratio_min": float(ratios.min()),
        })
    return pd.DataFrame(out)


def order_key(spec: str):
    name, _, param = spec.partition(":")
    families = ["exact", "rho_random", "rho_worst_g", "rho_worst_e", "tau_random", "nrho_random", "nrho_worst_e", "onep_greedy", "newton_exact", "newton_hf",
                "reopt_top", "reopt_all"]
    return families.index(name), float(param) if param else 0.0


def figure(table: pd.DataFrame, path: Path, evals_table: pd.DataFrame | None = None, evals_label: str = "") -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    systems = [s for s in LABEL.values() if s in set(table.system)]
    ncols = 4
    blocks = math.ceil(len(systems) / ncols)
    fig = plt.figure(figsize=(7.2, 3.9 * blocks))
    grid = fig.add_gridspec(2 * blocks, ncols, hspace=0.62, wspace=0.42)
    fams = [("rho_random", "random within rho of max |g|", BLUE, "o"), ("rho_worst_e", "worst energy within rho", RED, "s"),
            ("tau_random", "random within tau of best lowering", GREEN, "^"),
            ("nrho_random", "random within rho of |g|/sqrt(h_HF) (classical curvature)", PURPLE, "D")]
    axes = []
    for idx, system in enumerate(systems):
        block, col = divmod(idx, ncols)
        d = table[table.system == system]
        window = "dz8o" in system
        top = ("err_ratio_exact_steps", None, "energy error at step 40 / exact") if window else ("steps_ratio", None, "steps / exact")
        for row, (ycol, secol, ylabel) in enumerate([top, ("evals_ratio_tie_avg", "evals_ratio_tie_avg_se", "evaluations / tie-avg. exact")]):
            ax = fig.add_subplot(grid[2 * block + row, col])
            axes.append(ax)
            src = d
            if row == 1 and evals_table is not None and not evals_table[evals_table.system == system].empty:
                src = evals_table[evals_table.system == system]
                s9 = d[d.policy.str.startswith("rho_random:")].copy()
                if not s9.empty:
                    s9["x"] = s9.policy.str.split(":").str[1].astype(float)
                    s9 = s9.sort_values("x")
                    ax.plot(np.where(s9.x <= 0, 0.01, s9.x), s9[ycol], color=GREY, lw=0.8, ls=":", label="random within rho, tight tolerance 1e-9 (path noise)")
            for fam, text, colour, marker in fams:
                s = src[src.policy.str.startswith(fam + ":")].copy()
                if s.empty:
                    continue
                s["x"] = s.policy.str.split(":").str[1].astype(float)
                s = s.sort_values("x")
                x = np.where(s.x <= 0, 0.01, s.x)
                err = None
                if secol:
                    err = 1.96 * s[secol]
                elif "steps_se" in s:
                    err = 1.96 * s.steps_se / s.exact_steps
                ax.errorbar(x, s[ycol], yerr=err, color=colour, marker=marker, ms=2.5, lw=0.9, capsize=1.2, label=text)
            for spec, colour, text in (("onep_greedy", ORANGE, "energy-greedy (1 parameter)"), ("reopt_top:3", PURPLE, "energy-greedy (re-optimized, top 3)")):
                s = src[src.policy == spec]
                if not s.empty:
                    ax.axhline(float(s[ycol].iloc[0]), color=colour, lw=0.8, ls="--", label=text)
            ax.axhline(1.0, color=GREY, lw=0.6, ls=":")
            if row == 1 and evals_table is not None and src is d:
                ax.text(0.97, 0.96, "tight tolerance 1e-9", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=GREY)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(system)
            else:
                ax.set_xlabel("rho or tau (0 at 0.01)")
            if col == 0:
                ax.set_ylabel(ylabel + (f"\n{evals_label}" if row == 1 and evals_label else ""))
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh), labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/energy_policies"))
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--evals-runs", type=Path, default=None, help="run set whose report supplies the evaluation row of the figure (looser tolerance)")
    args = parser.parse_args()
    cases = args.cases or [p.name for p in sorted(args.runs.iterdir()) if (p / "exact_reference.json").exists()]
    out = args.runs / "report"
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for case in cases:
        ref, policies = load(args.runs / case)
        t = summarise(case, ref, policies)
        t.to_csv(out / f"policies_{case}.csv", index=False)
        tables.append(t)
        cols = ["policy", "n", "reached_pct", "steps_mean", "steps_se", "exact_steps", "more_steps_pct", "fewer_steps_pct",
                "evals_mean", "evals_ratio_tie_avg", "evals_ratio_exact_path", "final_error_median_mHa", "auc_ratio",
                "err20_mHa", "err_exact_steps_mHa", "g_shortfall_mean", "lowering_ratio_mean"]
        print(f"\n{LABEL.get(case, case)} (exact ADAPT {ref['iterations']} steps, {sum(ref['evaluations'])} evaluations)")
        print(t[cols].round(3).to_string(index=False))
    if tables:
        allt = pd.concat(tables, ignore_index=True)
        allt.to_csv(out / "policies_all.csv", index=False)
        evals_table, evals_label = None, ""
        if args.evals_runs is not None and (args.evals_runs / "report" / "policies_all.csv").exists():
            evals_table = pd.read_csv(args.evals_runs / "report" / "policies_all.csv")
            evals_label = f"({args.evals_runs.name.split('gtol')[-1].replace('_cl', '')} tolerance)"
        figure(allt, out / "dose_response", evals_table, evals_label)


if __name__ == "__main__":
    main()
