"""Figures of the feature study, in the style of the paper's figures (scripts/paper_a_figures.py).

Okabe-Ito colours, one marker per series, 7.2 in wide (two-column width), 8 pt text, no grid,
open top and right spines, legend under the panels.  Called by ``features_report.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                       "#56B4E9", "#F0E442", "#666666")
# one colour and marker per molecule
SYSTEMS = {  # label -> (colour, marker, title)
    "H4 1.0": (BLUE, "o", "H$_4$ 1.0 Å"),
    "H4 2.0": (ORANGE, "^", "H$_4$ 2.0 Å"),
    "LiH 3.0": (GREEN, "v", "LiH 3.0 Å"),
    "H2O eq": (PURPLE, "D", "H$_2$O eq"),
    "H2O str": (RED, "s", "H$_2$O str"),
}
ORDER = list(SYSTEMS)


def setup() -> None:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --- Figure 1: correlation of the gradient estimates against the energy lowering ------------------------------------

def figure_correlation(pairs: pd.DataFrame, out: Path) -> None:
    d = pairs[np.isfinite(pairs.corr_abs) & np.isfinite(pairs.lowering_ratio)]
    systems = [s for s in ORDER if s in set(d.system)]
    if not systems:
        return
    fig, axes = plt.subplots(1, len(systems), figsize=(7.2, 2.3), sharey=True, squeeze=False)
    for ax, system in zip(axes[0], systems):
        for design, colour, marker in (("II-0", GREY, "s"), ("II-A", BLUE, "o")):
            s = d[(d.system == system) & (d.design == design)]
            ax.plot(s.corr_abs, s.lowering_ratio.clip(upper=3.0), ls="", marker=marker, ms=3, color=colour, alpha=0.75)
            top = s[s.challenger_rank == 2]
            ax.plot(top.corr_abs, top.lowering_ratio.clip(upper=3.0), ls="", marker=marker, ms=6, mfc="none",
                    mec=colour, mew=1.0)
        ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
        ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
        ax.set_xlim(-0.65, 0.65)
        ax.set_title(SYSTEMS[system][2])
        ax.set_xlabel("correlation $r$ (leader, challenger)")
    axes[0, 0].set_ylabel("energy lowering,\nchallenger / leader")
    handles = [plt.Line2D([], [], ls="", marker="s", ms=4, color=GREY, label="II-0 (fixed coefficients)"),
               plt.Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="II-A (learned coefficients)"),
               plt.Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="#222222", label="leader vs runner-up")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "figure1_gradient_correlation_vs_energy")


# --- Figure 2: confidence level against the optimisation ----------------------------------------------------------------

def figure_confidence(t2: pd.DataFrame, out: Path) -> None:
    systems = [s for s in ORDER if s in set(t2.system)]
    if not systems:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
    panels = [(axes[0, 0], "selection_shots_median", None, "selection shots per run\n(median)"),
              (axes[0, 1], "steps_ratio", None, "ADAPT steps / exact ADAPT"),
              (axes[1, 0], "optimizer_evaluations_ratio", "optimizer_evaluations_sd_ratio",
               "optimizer evaluations\n/ exact ADAPT"),
              (axes[1, 1], "c_opt_ratio", "c_opt_sd_ratio", "$C_{\\mathrm{opt}}$ / exact ADAPT")]
    t2 = t2.assign(steps_ratio=t2.iterations_mean / t2.exact_iterations)
    for ax, col, sd, label in panels:
        for system in systems:
            colour, marker, title = SYSTEMS[system]
            s = t2[t2.system == system].sort_values("confidence_pct")
            err = (s[sd] / np.sqrt(s.trajectories)).fillna(0.0) if sd else None
            ax.errorbar(s.confidence_pct, s[col], yerr=err, color=colour, marker=marker, ms=3.5, lw=1.1,
                        elinewidth=0.8, capsize=2, label=title)
        if col == "selection_shots_median":
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()  # at least one decade, ticks at powers of ten only
            mid = np.sqrt(lo * hi)
            ax.set_ylim(min(lo, mid / 10 ** 0.5), max(hi, mid * 10 ** 0.5))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
            ax.set_ylim(0.5, 1.5)
        ax.set_ylabel(label)
    for ax in axes[1]:
        ax.set_xlabel("confidence level $1-\\delta$ (%)")
    axes[0, 0].set_xlim(100.5, 66.5)  # shared x, fixed: from 99 % on the left to 68 % on the right
    axes[0, 0].set_xticks([99, 95, 90, 80, 68])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(systems), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "figure2_confidence_vs_optimization")


# --- Figure 3: selection errors against the cost of the optimisation ----------------------------------------------------

def figure_errors(traj: pd.DataFrame, refs: dict, out: Path, method: str = "II-A data, safe") -> None:
    systems = [s for s in ORDER if s in set(traj.system)]
    if not systems:
        return
    fig, axes = plt.subplots(2, len(systems), figsize=(7.2, 3.6), squeeze=False, sharey="row")
    jitter = np.random.default_rng(0)
    for col, system in enumerate(systems):
        ref = refs[system]
        certified = traj[(traj.system == system) & (traj.method == method)]
        fixed = traj[(traj.system == system) & traj.method.str.startswith("Fixed") & ~traj.method.str.endswith("uniform")]
        for row, (ycol, yref) in enumerate([("iterations", ref["iterations"]), ("c_opt", ref["c_opt_eta2"])]):
            ax = axes[row, col]
            for g, colour, marker in ((certified, BLUE, "o"), (fixed, ORANGE, "^")):
                if g.empty:
                    continue
                x = g.non_best.to_numpy() + jitter.uniform(-0.15, 0.15, len(g))
                y = (g[ycol] / yref).to_numpy()
                ok = g.reached.to_numpy(bool) | (ref["final_error"] >= 1.6e-3)  # stretched H4: none can reach
                ax.plot(x[ok], y[ok], ls="", marker=marker, ms=3, color=colour, alpha=0.7)
                ax.plot(x[~ok], y[~ok], ls="", marker=marker, ms=3.5, mfc="none", mec=colour, mew=0.9)
            ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
            top = traj[traj.system == system].non_best.max()
            ax.set_xlim(-0.6, max(top, 3) + 0.6)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
            if row == 0:
                ax.set_title(SYSTEMS[system][2])
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("non-best choices")
    axes[0, 0].set_ylabel("ADAPT steps\n/ exact ADAPT")
    axes[1, 0].set_ylabel("$C_{\\mathrm{opt}}$ / exact ADAPT")
    axes[0, 0].set_ylim(0.9, 2.1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_yticks([0.2, 0.5, 1, 2, 5])
    axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="certified II-A (all confidence levels)"),
               plt.Line2D([], [], ls="", marker="^", ms=4, color=ORANGE, label="uncertified fixed budget"),
               plt.Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec="#222222",
                          label="chemical accuracy not reached")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "figure3_errors_vs_optimization_cost")


# --- Figure S1: |g| against the energy lowering of a one-parameter move -------------------------------------------------

def figure_gradient_energy(gens: pd.DataFrame, out: Path) -> None:
    gens = gens.assign(gain=-gens.dE1_mHa)
    systems = [s for s in ORDER if s in set(gens.system)]
    if not systems:
        return
    fig, axes = plt.subplots(1, len(systems), figsize=(7.2, 2.0), sharey=True, squeeze=False)
    for ax, system in zip(axes[0], systems):
        colour = SYSTEMS[system][0]
        d = gens[(gens.system == system) & (gens.abs_g > 1e-6) & (gens.gain > 1e-6)]
        ax.plot(d.abs_g, d.gain, ls="", marker=".", ms=2.5, color=GREY, alpha=0.6)
        for _, s in gens[gens.system == system].groupby(["state", "k"], dropna=False):
            top, best = s.loc[s.abs_g.idxmax()], s.loc[s.gain.idxmax()]
            if top.gain > 1e-6:
                ax.plot(top.abs_g, top.gain, ls="", marker="o", ms=4, mfc="none", mec=colour, mew=0.9)
            if best.gain > 1e-6 and best.abs_g > 1e-6:
                ax.plot(best.abs_g, best.gain, ls="", marker="x", ms=3.5, color="#222222", mew=0.8)
        x = np.logspace(-4, 0, 50)
        ax.plot(x, 1e3 * x ** 2 / 2, color="#999999", lw=0.8, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(1e-4, 1), ax.set_ylim(1e-4, 3e2)
        ax.set_title(SYSTEMS[system][2])
        ax.set_xlabel("$|g_i|$ (Ha)")
    axes[0, 0].set_ylabel("energy lowering (mHa)")
    handles = [plt.Line2D([], [], ls="", marker=".", ms=5, color=GREY, label="generator at a state"),
               plt.Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec="#222222", label="largest $|g|$ at a state"),
               plt.Line2D([], [], ls="", marker="x", ms=4, color="#222222", label="largest lowering at a state"),
               plt.Line2D([], [], ls="--", lw=0.8, color="#999999", label="$g^2/2$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, "figureS1_gradient_vs_energy")
