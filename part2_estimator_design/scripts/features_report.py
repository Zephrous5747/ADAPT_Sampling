#!/usr/bin/env python3
"""Tables and figures of the three feature studies (reads runs/features, writes runs/features/report).

Table 1  gradient |g_i| against the energy lowering of each generator (exact).
Table 2  confidence level 1 - delta against ADAPT iterations and optimisation steps.
Table 3  selection errors (non-best choices) against the optimisation cost.

    python scripts/features_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "features"
OUT = RUNS / "report"

# reference palette (dataviz skill, light mode): categorical slots in fixed order, blue ramp for ordered levels
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SYSTEM_ORDER = ["H4 1.0", "H4 2.0", "LiH 3.0", "H2O eq", "H2O str"]
CASE_LABEL = {"H4_square_eq_side1p0_HF": "H4 1.0", "H4_square_stretch_side2p0_HF": "H4 2.0",
              "LiH_R3p0_HF": "LiH 3.0", "H2O_eq_HF": "H2O eq", "H2O_stretch_HF": "H2O str"}
TEX = {"H4 1.0": r"H$_4$ 1.0~\AA", "H4 2.0": r"H$_4$ 2.0~\AA", "LiH 3.0": r"LiH 3.0~\AA",
       "H2O eq": r"H$_2$O eq", "H2O str": r"H$_2$O str"}
PRETTY = {"H4 1.0": "H$_4$ 1.0 Å", "H4 2.0": "H$_4$ 2.0 Å", "LiH 3.0": "LiH 3.0 Å",
          "H2O eq": "H$_2$O eq", "H2O str": "H$_2$O str"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
        "lines.linewidth": 2.0, "font.family": "DejaVu Sans"})


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- Table 1

def table1() -> pd.DataFrame:
    states = pd.read_csv(RUNS / "energy_lowering_states.csv")
    rows = []
    for system in SYSTEM_ORDER:
        d = states[states.system == system]
        if d.empty:
            continue
        sp = pd.to_numeric(d.spearman_1p, errors="coerce").dropna()
        top = d.best_g_is_best_dE_1p.astype(str).str.lower() == "true"
        share = pd.to_numeric(d.share_best_g_1p, errors="coerce")
        worst = pd.to_numeric(d.share_worst_rho_good_1p, errors="coerce")
        full_sp = pd.to_numeric(d.spearman_full, errors="coerce").dropna()
        full_share = pd.to_numeric(d.share_best_g_full, errors="coerce").dropna()
        rows.append({
            "system": system, "states": len(d),
            "spearman_median": sp.median(), "spearman_min": sp.min(),
            "top_g_is_best_dE_pct": 100 * top.mean(),
            "share_top_g_mean": share.mean(), "share_top_g_min": share.min(),
            "share_worst_rho_good_mean": worst.mean(), "share_worst_rho_good_min": worst.min(),
            "spearman_full_median": full_sp.median() if len(full_sp) else np.nan,
            "share_top_g_full_mean": full_share.mean() if len(full_share) else np.nan,
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tableS1_gradient_vs_energy.csv", index=False)
    lines = [r"\begin{table}[!htb]", r"\centering\small",
             r"\caption{Gradient against energy lowering (exact, STO-3G, UCCSD pool). For every state (HF, CISD and every "
             r"state of the exact ADAPT trajectory) and every generator, the energy lowering of a one-parameter move. "
             r"$\rho_s$: Spearman rank correlation of $|g_i|$ with the lowering (median and smallest over the states). "
             r"Top $|g|$ is best: share of states where the generator of largest $|g_i|$ also lowers the energy most. "
             r"Share of best lowering: lowering of the largest-$|g|$ choice, and of the worst choice within $\rho=0.1$ "
             r"of the largest $|g|$, divided by the best possible lowering (mean, smallest in brackets). Full re-opt.: "
             r"the same with all parameters re-optimized after the generator is appended (H$_4$, LiH).}",
             r"\label{tab:S_grad_energy}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{one-parameter move} & \multicolumn{2}{c}{full re-opt.} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"System & states & $\rho_s$ & top $|g|$ is best & share, top $|g|$ & share, worst $\rho$-good "
             r"& $\rho_s$ & share, top $|g|$ \\", r"\midrule"]
    for r in t.itertuples():
        full = (f"{r.spearman_full_median:.2f} & {r.share_top_g_full_mean:.2f}"
                if not np.isnan(r.spearman_full_median) else "-- & --")
        lines.append(f"{TEX[r.system]} & {r.states} & {r.spearman_median:.2f} ({r.spearman_min:.2f}) & "
                     f"{r.top_g_is_best_dE_pct:.0f}\\% & {r.share_top_g_mean:.2f} ({r.share_top_g_min:.2f}) & "
                     f"{r.share_worst_rho_good_mean:.2f} ({r.share_worst_rho_good_min:.2f}) & {full} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tableS1_gradient_vs_energy.tex").write_text("\n".join(lines) + "\n")
    return t


def figure1() -> None:
    gens = pd.read_csv(RUNS / "energy_lowering_generators.csv")
    gens["gain"] = -gens.dE1_mHa
    systems = [s for s in SYSTEM_ORDER if s in set(gens.system)]
    fig, axes = plt.subplots(1, len(systems), figsize=(2.6 * len(systems), 2.7), sharey=True)
    for ax, system, color in zip(np.atleast_1d(axes), systems, SERIES):
        d = gens[(gens.system == system) & (gens.abs_g > 1e-6) & (gens.gain > 1e-6)]
        ax.scatter(d.abs_g, d.gain, s=9, color=color, alpha=0.35, linewidths=0)
        # mark, per state, the largest-|g| generator (ring) and the best-lowering generator (cross)
        for _, s in gens[gens.system == system].groupby(["state", "k"], dropna=False):
            top = s.loc[s.abs_g.idxmax()]
            best = s.loc[s.gain.idxmax()]
            if top.gain > 1e-6:
                ax.scatter(top.abs_g, top.gain, s=36, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=3)
            if best.gain > 1e-6 and best.abs_g > 1e-6:
                ax.scatter(best.abs_g, best.gain, s=30, marker="x", color=INK, linewidths=1.0, zorder=3)
        x = np.logspace(-4, 0, 50)
        ax.plot(x, 1e3 * x ** 2 / 2, color=INK2, lw=1, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(1e-4, 1), ax.set_ylim(1e-4, 3e2)
        ax.set_title(PRETTY[system], fontsize=9, color=INK)
        ax.set_xlabel("|g$_i$| (Ha)")
    np.atleast_1d(axes)[0].set_ylabel("energy lowering, one-parameter move (mHa)")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="largest |g| at a state"),
               plt.Line2D([], [], marker="x", ls="", color=INK, label="largest lowering at a state"),
               plt.Line2D([], [], ls="--", color=INK2, lw=1, label="g$^2$/2 (curvature 1 Ha)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    save(fig, "figureS1_gradient_vs_energy")


# --------------------------------------------------------------------------- Table 1 (gradient-estimate correlation)

def load_pairs() -> pd.DataFrame:
    main_path = RUNS / "gradient_correlation_pairs.csv"
    d = pd.read_csv(main_path) if main_path.exists() else pd.DataFrame()
    hf_path = RUNS / "gradient_correlation_pairs_hf.csv"
    if hf_path.exists():  # HF states recomputed with the zero-variance guard
        d = pd.concat([d[d.state != "HF"] if len(d) else d, pd.read_csv(hf_path)], ignore_index=True)
    # further systems written to their own files (H2O: gradient_correlation_pairs_h2o*.csv)
    for extra in sorted(RUNS.glob("gradient_correlation_pairs_h2o*.csv")):
        d = pd.concat([d, pd.read_csv(extra)], ignore_index=True)
    d["lowering_ratio"] = d.gain_challenger_mHa / d.gain_leader_mHa
    return d


def table1_correlation() -> pd.DataFrame:
    from scipy.stats import spearmanr

    d = load_pairs()
    rows = []
    for system in SYSTEM_ORDER:
        for level in ("II-0", "II-A"):
            s = d[(d.system == system) & (d.design == level)]
            if s.empty:
                continue
            ok = s[np.isfinite(s.corr_abs)]
            top = ok[ok.challenger_rank == 2]
            fin = ok[np.isfinite(ok.lowering_ratio)]
            rows.append({
                "system": system, "design": level, "states": s.groupby(["state", "k"], dropna=False).ngroups,
                "pairs": len(ok), "undefined": len(s) - len(ok),
                "r_median": ok.corr_abs.median(), "abs_r_p90": ok.corr_abs.abs().quantile(0.9),
                "abs_r_max": ok.corr_abs.abs().max(),
                "chi_median": ok.chi.median(), "chi_min": ok.chi.min(), "chi_max": ok.chi.max(),
                "top_pair_r_median": top.corr_abs.median(), "top_pair_chi_median": top.chi.median(),
                "spearman_r_vs_lowering_ratio": spearmanr(fin.corr_abs, fin.lowering_ratio).statistic,
            })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table1_gradient_correlation_vs_energy.csv", index=False)

    # pooled bins of the correlation (both designs)
    ok = d[np.isfinite(d.corr_abs) & np.isfinite(d.lowering_ratio)]
    edges = [-1.0, -0.2, -0.05, 0.05, 0.2, 1.0]
    names = [r"$r<-0.2$", r"$-0.2\le r<-0.05$", r"$|r|<0.05$", r"$0.05\le r<0.2$", r"$r\ge0.2$"]
    bins = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        b = ok[(ok.corr_abs >= lo) & (ok.corr_abs < hi if hi < 1 else ok.corr_abs <= hi)]
        if b.empty:
            continue
        bins.append({"bin": name, "pairs": len(b), "chi_median": b.chi.median(),
                     "gap_rel_median": b.gap_rel.median(),
                     "lowering_ratio_median": b.lowering_ratio.median(),
                     "share_ratio_ge_0p9": 100 * (b.lowering_ratio >= 0.9).mean()})
    tb = pd.DataFrame(bins)
    tb.to_csv(OUT / "table1b_correlation_bins.csv", index=False)

    lines = [r"\begin{table}[!htb]", r"\centering\small",
             r"\caption{Correlation of the gradient estimates against energy lowering (exact covariances of each "
             r"state, STO-3G, UCCSD pool; HF, CISD and every exact ADAPT state). All gradients are estimated from "
             r"the same shared fully commuting contexts (II-0: fixed fragments; II-A: oracle coefficient splitting), "
             r"with one allocation over all non-zero gradients. For the leader $\ell$ (largest $|g|$) and every "
             r"challenger $j$: $r$ is the correlation of the estimates of $|g_\ell|$ and $|g_j|$, and "
             r"$\chi=\mathrm{Var}(|\hat g_\ell|-|\hat g_j|)/(\mathrm{Var}\,\hat g_\ell+\mathrm{Var}\,\hat g_j)$, the "
             r"factor by which the shared data change the shots of their comparison ($\chi<1$: cheaper). Top pair: "
             r"leader against the runner-up. $\rho_s$: Spearman correlation, over the pairs, of $r$ with the energy "
             r"lowering of the challenger divided by that of the leader (one-parameter move).}",
             r"\label{tab:corr_energy}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{all pairs} & \multicolumn{2}{c}{top pair} & \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
             r"System & design & pairs & $r$ & $|r|_{90\%}$ & $\chi$ (range) & $r$ & $\chi$ & $\rho_s(r,\,\Delta E)$ \\",
             r"\midrule"]
    for system in SYSTEM_ORDER:
        s = t[t.system == system]
        for i, r in enumerate(s.itertuples()):
            lines.append(f"{TEX[system] if i == 0 else ''} & {r.design} & {r.pairs} & {r.r_median:+.2f} & "
                         f"{r.abs_r_p90:.2f} & {r.chi_median:.2f} ({r.chi_min:.2f}--{r.chi_max:.2f}) & "
                         f"{r.top_pair_r_median:+.2f} & {r.top_pair_chi_median:.2f} & "
                         f"{r.spearman_r_vs_lowering_ratio:+.2f} \\\\")
    lines += [r"\midrule", r"\multicolumn{9}{l}{\emph{pooled over systems and designs, by correlation}} \\",
              r"bin & & pairs & & & $\chi$ & \multicolumn{2}{r}{rel.\ gap $|g|$} & $\Delta E_j/\Delta E_\ell$ \\"]
    for r in tb.itertuples():
        lines.append(f"{r.bin} & & {r.pairs} & & & {r.chi_median:.2f} & \\multicolumn{{2}}{{r}}{{{r.gap_rel_median:.2f}}} & "
                     f"{r.lowering_ratio_median:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table1_gradient_correlation_vs_energy.tex").write_text("\n".join(lines) + "\n")
    return t, tb


def figure1_correlation() -> None:
    d = load_pairs()
    d = d[np.isfinite(d.corr_abs) & np.isfinite(d.lowering_ratio)]
    systems = [s for s in SYSTEM_ORDER if s in set(d.system)]
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 2.9), sharey=True, squeeze=False)
    for ax, system in zip(axes[0], systems):
        for level, color in (("II-0", SERIES[0]), ("II-A", SERIES[1])):
            s = d[(d.system == system) & (d.design == level)]
            ax.scatter(s.corr_abs, s.lowering_ratio.clip(-0.05, 3), s=12, color=color, alpha=0.55, linewidths=0,
                       label=f"{level} design")
            top = s[s.challenger_rank == 2]
            ax.scatter(top.corr_abs, top.lowering_ratio.clip(-0.05, 3), s=34, facecolors="none", edgecolors=color,
                       linewidths=1.1)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set_xlim(-0.65, 0.65)
        ax.set_title(PRETTY[system], fontsize=9, color=INK)
        ax.set_xlabel("correlation r of the estimates\n(leader, challenger)")
    axes[0][0].set_ylabel("energy lowering, challenger / leader")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], label="II-0 (fixed fragments)"),
               plt.Line2D([], [], marker="o", ls="", color=SERIES[1], label="II-A (coefficient splitting)"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="leader vs runner-up")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09), fontsize=8)
    save(fig, "figure1_gradient_correlation_vs_energy")


# --------------------------------------------------------------------------- sweep data

def load_sweeps() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    traj, steps, refs = [], [], {}
    for case_dir in sorted(RUNS.glob("*_HF")):
        system = CASE_LABEL.get(case_dir.name)
        if system is None:
            continue
        ref_path = case_dir / "exact_reference.json"
        if ref_path.exists():
            refs[system] = json.loads(ref_path.read_text())
        for path in sorted(case_dir.glob("ci_*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                sel = r["selections"]
                traj.append({"system": system, "method": r["method"], "delta": r["delta"],
                             "confidence": 100 * (1 - r["delta"]), "index": r["index"],
                             "iterations": r["iterations"], "final_error_mHa": 1e3 * r["final_error"],
                             "reached": r["reached"],
                             "same_energy": abs(r["final_error"] - refs[system]["final_error"]) < 1e-4
                             if system in refs else r["reached"],
                             "selection_shots": r["selection_shots"],
                             "optimizer_evaluations": r["optimizer_evaluations"], "c_opt": r["c_opt_eta2"],
                             "selections": len(sel), "non_best": sum(not s["exact_best"] for s in sel),
                             "not_rho_good": len(sel) - r["rho_good"],
                             "mean_shortfall": float(np.mean([s["shortfall"] for s in sel])) if sel else 0.0})
                for s, ev, n in zip(sel, r["evaluations"], r["n_params"]):
                    steps.append({"system": system, "method": r["method"], "delta": r["delta"], "index": r["index"],
                                  "k": s["k"], "exact_best": s["exact_best"], "rank": s["rank"],
                                  "shortfall": s["shortfall"], "shots": s["shots"], "evaluations": ev, "n_params": n})
    return pd.DataFrame(traj), pd.DataFrame(steps), refs


# --------------------------------------------------------------------------- Table 2

def table2(traj: pd.DataFrame, refs: dict, method: str = "II-A data, safe") -> pd.DataFrame:
    d = traj[traj.method == method]
    rows = []
    for system in SYSTEM_ORDER:
        s = d[d.system == system]
        if s.empty:
            continue
        ref = refs[system]
        for delta, g in sorted(s.groupby("delta"), key=lambda kv: kv[0]):
            rows.append({
                "system": system, "confidence_pct": 100 * (1 - delta), "trajectories": len(g),
                "selection_shots_median": g.selection_shots.median(),
                "iterations_mean": g.iterations.mean(), "iterations_max": g.iterations.max(),
                "exact_iterations": ref["iterations"],
                "optimizer_evaluations_mean": g.optimizer_evaluations.mean(),
                "exact_optimizer_evaluations": ref["optimizer_evaluations"],
                "optimizer_evaluations_ratio": g.optimizer_evaluations.mean() / ref["optimizer_evaluations"],
                "optimizer_evaluations_sd_ratio": g.optimizer_evaluations.std(ddof=1) / ref["optimizer_evaluations"]
                if len(g) > 1 else np.nan,
                "c_opt_ratio": g.c_opt.mean() / ref["c_opt_eta2"],
                "c_opt_sd_ratio": g.c_opt.std(ddof=1) / ref["c_opt_eta2"] if len(g) > 1 else np.nan,
                "final_error_median_mHa": g.final_error_mHa.median(),
                "exact_final_error_mHa": 1e3 * ref["final_error"],
                "reached_pct": 100 * g.same_energy.mean(),
                "exact_best_pct": 100 * (1 - g.non_best.sum() / g.selections.sum()),
                "rho_good_pct": 100 * (1 - g.not_rho_good.sum() / g.selections.sum()),
            })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table2_confidence_vs_optimization.csv", index=False)
    lines = [r"\begin{table*}[!htb]", r"\centering\small",
             r"\caption{Confidence level of the selection against the cost of the optimization, along complete "
             r"measured ADAPT runs (II-A: shared contexts, elimination, learned splitting with cross-fitting; "
             r"sign-aware rule, a priori first radius, $\rho=0.1$; STO-3G, UCCSD pool; exact BFGS re-optimization). "
             r"Shots: median selection context-shots of a run. Steps: mean ADAPT iterations (largest in brackets). "
             r"Opt.\ evals: mean BFGS energy-and-gradient evaluations of a run, and its ratio to exact ADAPT "
             r"($\pm$ standard deviation over the runs; the count varies with the order in which tied generators "
             r"are added, even when every choice is correct); $C_\mathrm{opt}$: ratio of the modeled optimization measurement cost ($\eta=2$, energies to 1~mHa) to "
             r"exact ADAPT. $\Delta E$: median final energy error. Same $E$: share of runs whose final energy equals that of exact ADAPT (within 0.1~mHa; stretched H$_4$ stalls at 3.2~mHa for every method, a limit of the pool). Exact best / $\rho$-good: share of selections. Exact ADAPT row: the reference. H$_2$O: the recorded measured runs at 95\% (36 per geometry) with their optimization recomputed exactly; the other levels need the cluster.}",
             r"\label{tab:ci_opt}", r"\footnotesize\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
             r"System & conf. & runs & shots & steps & opt.\ evals & ratio & $C_\mathrm{opt}$ ratio & "
             r"$\Delta E$ (mHa) & same $E$ & exact best & $\rho$-good \\", r"\midrule"]
    for system in SYSTEM_ORDER:
        s = t[t.system == system]
        if s.empty:
            continue
        ref = refs[system]
        lines.append(f"{TEX[system]} & exact & -- & -- & {ref['iterations']} & {ref['optimizer_evaluations']} & "
                     f"1.00 & 1.00 & {1e3 * ref['final_error']:.2f} & -- & -- & -- \\\\")
        for r in s.sort_values("confidence_pct", ascending=False).itertuples():
            lines.append(f" & {r.confidence_pct:.0f}\\% & {r.trajectories} & {_sci(r.selection_shots_median)} & "
                         f"{r.iterations_mean:.1f} ({r.iterations_max}) & {r.optimizer_evaluations_mean:.0f} & "
                         f"{r.optimizer_evaluations_ratio:.2f}{_pm(r.optimizer_evaluations_sd_ratio)} & "
                         f"{r.c_opt_ratio:.2f}{_pm(r.c_opt_sd_ratio)} & {r.final_error_median_mHa:.2f} & "
                         f"{r.reached_pct:.0f}\\% & {r.exact_best_pct:.0f}\\% & {r.rho_good_pct:.0f}\\% \\\\")
        lines.append(r"\midrule" if system != SYSTEM_ORDER[-1] else "")
    if lines[-1] in (r"\midrule", ""):
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table2_confidence_vs_optimization.tex").write_text("\n".join(lines) + "\n")
    return t


def figure2(t: pd.DataFrame) -> None:
    systems = [s for s in SYSTEM_ORDER if s in set(t.system)]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    panels = [("selection_shots_median", None, "(a) selection shots per run (median)"),
              ("iterations_mean", None, "(b) ADAPT steps (parameters)"),
              ("optimizer_evaluations_ratio", "optimizer_evaluations_sd_ratio", "(c) optimizer evaluations / exact"),
              ("c_opt_ratio", "c_opt_sd_ratio", "(d) optimization cost C$_{opt}$ / exact")]
    for ax, (col, sd, title) in zip(axes, panels):
        for system, color in zip(SYSTEM_ORDER, SERIES):
            if system not in systems:
                continue
            s = t[t.system == system].sort_values("confidence_pct")
            if col == "iterations_mean":
                y = s[col] / s.exact_iterations
            else:
                y = s[col]
            err = s[sd] / np.sqrt(s.trajectories) if sd else None
            ax.errorbar(s.confidence_pct, y, yerr=err, marker="o", ms=5, color=color, capsize=2, elinewidth=1,
                        label=PRETTY[system])
        if col == "selection_shots_median":
            ax.set_yscale("log")
        else:
            ax.axhline(1.0, color=INK2, lw=1, ls="--")
            ax.set_ylim(0.5, 1.5)
        if col == "iterations_mean":
            ax.set_title("(b) ADAPT steps / exact ADAPT", fontsize=9, color=INK)
        else:
            ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("confidence level 1 − δ (%)")
        ax.invert_xaxis()
    axes[0].legend(fontsize=8, loc="best")
    save(fig, "figure2_confidence_vs_optimization")


# --------------------------------------------------------------------------- Table 3

def table3(traj: pd.DataFrame, steps: pd.DataFrame, refs: dict, method: str = "II-A data, safe") -> pd.DataFrame:
    rows = []

    def summary(system, group, label, g, ref):
        return {"system": system, "group": group, "label": label, "trajectories": len(g),
                "non_best_per_run": g.non_best.mean(), "not_rho_good_per_run": g.not_rho_good.mean(),
                "iterations_mean": g.iterations.mean(), "iterations_max": int(g.iterations.max()),
                "exact_iterations": ref["iterations"],
                "optimizer_evaluations_ratio": g.optimizer_evaluations.mean() / ref["optimizer_evaluations"],
                "optimizer_evaluations_sd_ratio": g.optimizer_evaluations.std(ddof=1) / ref["optimizer_evaluations"]
                if len(g) > 1 else np.nan,
                "c_opt_ratio": g.c_opt.mean() / ref["c_opt_eta2"],
                "final_error_median_mHa": g.final_error_mHa.median(),
                "same_energy_pct": 100 * g.same_energy.mean(),
                "reached_ca_pct": 100 * g.reached.mean(),
                "selection_shots_median": g.selection_shots.median()}

    for system in SYSTEM_ORDER:
        if system not in refs:
            continue
        ref = refs[system]
        s = traj[(traj.system == system) & (traj.method == method)]
        if not s.empty:
            s = s.assign(bucket=np.minimum(s.non_best, 3))
            for bucket, g in s.groupby("bucket"):
                rows.append(summary(system, "certified II-A (all levels)",
                                    "3 or more" if bucket == 3 else str(int(bucket)), g, ref))
        fixed = traj[(traj.system == system) & traj.method.str.startswith("Fixed")
                     & ~traj.method.str.endswith("uniform")]
        if not fixed.empty:
            # one allocation per system (pilot: no exact input, when it was run), and the budgets around the
            # smallest one at which at least 95% of the runs reach chemical accuracy
            alloc = "pilot" if fixed.method.str.endswith("pilot").any() else "designed"
            fixed = fixed[fixed.method.str.endswith(alloc)]
            budget = lambda name: int(name.split()[1].rstrip(","))  # noqa: E731
            stats = sorted(((budget(n), n, g.reached.mean()) for n, g in fixed.groupby("method")))
            ok = [b for b, _, e in stats if e >= 0.95]
            adequate = ok[0] if ok else stats[-1][0]
            below = [b for b, _, e in stats if b < adequate]
            keep = set(below[-3:] + [adequate])
            if len(below) > 3:
                keep.add(below[0])
            for b, name, _ in stats:
                if b in keep:
                    g = fixed[fixed.method == name]
                    rows.append(summary(system, "uncertified fixed budget",
                                        f"{_sci_plain(b)} shots/sel. ({alloc})", g, ref))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table3_errors_vs_optimization_cost.csv", index=False)
    lines = [r"\begin{table*}[!htb]", r"\centering\small",
             r"\caption{Selection errors against the cost of the optimization (measured ADAPT runs, STO-3G, UCCSD pool, "
             r"exact BFGS re-optimization). Top of each block: the certified II-A runs of Table~\ref{tab:ci_opt}, every "
             r"confidence level pooled, grouped by the number of selections in a run that were not the exact best "
             r"generator (exact ties count as best). Bottom: an uncertified selection with a fixed number of shots per "
             r"selection (allocation from variances estimated on a pilot of a fifth of the budget), which makes larger "
             r"errors. Non-best, not $\rho$-good: mean per run. Steps: mean ADAPT iterations (largest; exact ADAPT in "
             r"the system row). Opt.\ evals: mean BFGS evaluations of a run divided by exact ADAPT ($\pm$ standard "
             r"deviation over runs). $C_\mathrm{opt}$: modeled optimization measurement cost divided by exact ADAPT. "
             r"Same $E$: share of runs whose final energy equals exact ADAPT within 0.1~mHa; CA: share of runs that "
             r"reach chemical accuracy within ten steps more than exact ADAPT (stretched H$_4$: none can, the pool "
             r"stalls at 3.2~mHa). H$_2$O rows: the recorded measured runs at 95\% confidence, with their optimization recomputed exactly.}",
             r"\label{tab:err_opt}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"System & selection & runs & non-best & not $\rho$-good & steps & opt.\ evals & $C_\mathrm{opt}$ & "
             r"$\Delta E$ (mHa) & same $E$ & CA \\", r"\midrule"]
    for system in SYSTEM_ORDER:
        s = t[t.system == system]
        if s.empty:
            continue
        ref = refs[system]
        ca = "yes" if ref["final_error"] < 1.6e-3 else "no"
        lines.append(f"{TEX[system]} & exact ADAPT & -- & 0 & 0 & {ref['iterations']} & 1.00 & 1.00 & "
                     f"{1e3 * ref['final_error']:.2f} & -- & {ca} \\\\")
        for r in s.itertuples():
            label = (f"II-A, {r.label} non-best" if r.group.startswith("certified") else f"fixed {r.label}")
            sd = f" $\\pm${r.optimizer_evaluations_sd_ratio:.2f}" if np.isfinite(r.optimizer_evaluations_sd_ratio) else ""
            lines.append(f" & {label} & {r.trajectories} & {r.non_best_per_run:.1f} & {r.not_rho_good_per_run:.1f} & "
                         f"{r.iterations_mean:.1f} ({r.iterations_max}) & {r.optimizer_evaluations_ratio:.2f}{sd} & "
                         f"{r.c_opt_ratio:.2f} & {r.final_error_median_mHa:.2f} & {r.same_energy_pct:.0f}\\% & "
                         f"{r.reached_ca_pct:.0f}\\% \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table3_errors_vs_optimization_cost.tex").write_text("\n".join(lines) + "\n")
    return t


def figure3(traj: pd.DataFrame, refs: dict, method: str = "II-A data, safe") -> None:
    systems = [s for s in SYSTEM_ORDER if s in set(traj.system)]
    fig, axes = plt.subplots(2, len(systems), figsize=(3.4 * len(systems), 5.4), squeeze=False, sharex="col")
    jitter = np.random.default_rng(0)
    for col, system in enumerate(systems):
        ref = refs[system]
        groups = [(traj[(traj.system == system) & (traj.method == method)], SERIES[0], "certified II-A, all levels"),
                  (traj[(traj.system == system) & traj.method.str.startswith("Fixed")], SERIES[1],
                   "uncertified fixed budget")]
        for row, (ycol, yref, ylabel) in enumerate([("iterations", ref["iterations"], "ADAPT steps / exact"),
                                                    ("c_opt", ref["c_opt_eta2"], "C$_{opt}$ / exact")]):
            ax = axes[row][col]
            for g, color, label in groups:
                if g.empty:
                    continue
                x = g.non_best + jitter.uniform(-0.2, 0.2, len(g))
                ok = g.same_energy.to_numpy(bool)
                ax.scatter(x[ok], (g[ycol] / yref)[ok], s=16, color=color, alpha=0.6, linewidths=0, label=label)
                ax.scatter(x[~ok], (g[ycol] / yref)[~ok], s=18, facecolors="none", edgecolors=color, linewidths=1.0)
            top = traj[traj.system == system].non_best.max()
            ax.set_xlim(-0.6, max(top, 2) + 0.6)
            ax.axhline(1.0, color=INK2, lw=1, ls="--")
            if row == 0:
                ax.set_title(PRETTY[system], fontsize=9, color=INK)
            else:
                ax.set_xlabel("non-best selections in the run")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], label="certified II-A (all confidence levels)"),
               plt.Line2D([], [], marker="o", ls="", color=SERIES[1], label="uncertified fixed budget"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, label="final energy worse than exact ADAPT")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8)
    save(fig, "figure3_errors_vs_optimization_cost")


def _sci_plain(x: float) -> str:
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"{m:.1f}e{e}" if abs(m - round(m)) > 0.05 else f"{int(round(m))}e{e}"


def _pm(x: float) -> str:
    return f" $\\pm${x:.2f}" if np.isfinite(x) else ""


def _sci(x: float) -> str:
    e = int(np.floor(np.log10(x)))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import features_figures as ff
    ff.setup()
    pd.set_option("display.width", 250)
    if (RUNS / "energy_lowering_states.csv").exists():
        table1()  # supplementary: |g| against energy lowering
        ff.figure_gradient_energy(pd.read_csv(RUNS / "energy_lowering_generators.csv"), OUT)
    if list(RUNS.glob("gradient_correlation_pairs*.csv")):
        t1c, t1b = table1_correlation()
        ff.figure_correlation(load_pairs(), OUT)
        print(t1c.round(3).to_string())
        print(t1b.round(3).to_string())
    else:
        print("no gradient-correlation data yet (Table 1)")
    traj, steps, refs = load_sweeps()
    if traj.empty:
        print("no sweep data yet")
        return
    traj.to_csv(OUT / "all_trajectories.csv", index=False)
    t2 = table2(traj, refs)
    ff.figure_confidence(t2, OUT)
    t3 = table3(traj, steps, refs)
    ff.figure_errors(traj, refs, OUT)
    pd.set_option("display.width", 250)
    print(t2.round(3).to_string())
    print(t3.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
