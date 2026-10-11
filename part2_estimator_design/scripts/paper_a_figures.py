#!/usr/bin/env python3
"""Paper A: the figures (referee point on the absence of any figure), drawn from the stored runs.

``fig_pipeline``     the procedure of one selection, one box per step and the loop of rounds;
``fig_rounds``       cumulative shots and active candidates against the elimination round (II-0, II-A);
``fig_trajectory``   shots per selection and cumulative shots along an ADAPT trajectory, per method;
``fig_depth``        shots against the two-qubit gates per measured shot, one point per family of contexts;
``fig_fixed_budget`` the uncertified fixed-budget selection: success and steps against the budget, and the
                     total cost against the certified methods (needs ``runs/paper_a/fixed_budget.csv``).

PDF and PNG are written to ``--out`` (default ``../reports/figures``); a figure whose data are missing is skipped.
Colours are the Okabe-Ito palette (colour-blind safe) and every method has its own marker.

    python scripts/paper_a_figures.py [--only fig_depth fig_rounds]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                       "#56B4E9", "#F0E442", "#666666")
STYLE = {  # method -> (colour, marker, label)
    "II-0 safe": (GREY, "s", "II-0 (fixed coefficients)"),
    "II-A data, safe": (BLUE, "o", "II-A (learned coefficients)"),
    "M2 safe": (ORANGE, "^", "independent arms (M2)"),
    "M1 seq, safe": (GREEN, "v", "shared groups, no elimination (M1 seq)"),
    "Ikh reuse, FC": (PURPLE, "D", "reuse of energy data"),
}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


# --- the pipeline --------------------------------------------------------------------------------------------------------


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = {
        "pool": (0.2, 3.3, 2.6, 1.3, "Pool and Hamiltonian:\n$g_i=\\langle[H,G_i]\\rangle$,\nPauli products\nof $[H,G_i]$", SKY),
        "ctx": (3.2, 3.3, 2.6, 1.3, "Contexts:\ncommuting groups of\nthe products, one\ncircuit each", SKY),
        "des": (6.2, 3.3, 2.6, 1.3, "Designs $x_i$:\nweights of each product\nover the contexts that\nmeasure it", ORANGE),
        "shots": (9.2, 3.3, 2.6, 1.3, "Round $r$: allocate\nshots for radius $R_r$,\nmeasure, update\nestimates", GREEN),
        "stat": (9.2, 0.9, 2.6, 1.3, "Radii from the data\n(cross-fitted\ncovariances), $z$ for\nlevel $\\delta$", GREEN),
        "elim": (6.2, 0.9, 2.6, 1.3, "Eliminate candidates\nwhose upper bound\nis below the leader's\nlower bound", RED),
        "stop": (3.2, 0.9, 2.6, 1.3, "Stop when one candidate\nis left, or one is within\n$\\rho$ of the best", PURPLE),
        "out": (0.2, 0.9, 2.6, 1.3, "Add it to the ansatz,\nre-optimise the\nparameters (exact here)", PURPLE),
    }
    for key, (x, y, w, h, text, colour) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=colour, ec="#222222", lw=0.8, alpha=0.30))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4)

    def arrow(a, b, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, lw=0.9, color="#222222", connectionstyle=f"arc3,rad={rad}"))

    arrow((2.8, 3.95), (3.2, 3.95))
    arrow((5.8, 3.95), (6.2, 3.95))
    arrow((8.8, 3.95), (9.2, 3.95))
    arrow((10.5, 3.3), (10.5, 2.2))
    arrow((9.2, 1.55), (8.8, 1.55))
    arrow((6.2, 1.55), (5.8, 1.55))
    arrow((3.2, 1.55), (2.8, 1.55))
    arrow((7.5, 2.2), (7.5, 3.3))
    ax.text(7.7, 2.8, "not finished: refit designs\n(II-A, shots grew 1.5x),\n$R_{r+1}=0.9\\,R_r$", fontsize=6.1, ha="left", va="center")
    ax.text(0.2, 0.35, "Fixed once per state: contexts and circuits.  Repeated: designs (when refitted), shots, radii, elimination.  "
            "Nothing but the sampled outcomes enters a round.", fontsize=6.5, color="#333333")
    save(fig, out, "fig_pipeline")


# --- shots against the elimination round -------------------------------------------------------------------------------------


def fig_rounds(out: Path) -> None:
    cases = [("H4_square_eq_side1p0_CISD", "H$_4$ 1.0 Å, CISD"), ("H4_square_stretch_side2p0_CISD", "H$_4$ 2.0 Å, CISD"),
             ("LiH_R3p0_HF", "LiH, HF")]
    fig, axes = plt.subplots(2, len(cases), figsize=(7.2, 3.6), sharex="col")
    drawn = False
    for col, (case, title) in enumerate(cases):
        for config, label, colour, ls in (("II-0_estimated", "II-0", GREY, "-"), ("II-A_data", "II-A", BLUE, "-")):
            data = rows(RUNS / "paper_a" / f"{case}_{config}_evolution.csv")
            if not data:
                continue
            drawn = True
            r = [int(d["round"]) for d in data]
            axes[0, col].plot(r, [float(d["cumulative_shots"]) for d in data], color=colour, ls=ls, lw=1.4, label=label)
            axes[1, col].plot(r, [int(d["active"]) for d in data], color=colour, ls=ls, lw=1.4)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("elimination round")
        axes[1, col].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if not drawn:
        return
    axes[0, 0].set_ylabel("cumulative context-shots")
    axes[1, 0].set_ylabel("active candidates")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_rounds")


# --- cost along a trajectory --------------------------------------------------------------------------------------------------


def selections_by_method(case: str) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    phase4 = RUNS / case / "phase4"
    for path in sorted(phase4.glob(f"{case}_phase4*_selections.csv")):
        if "_phase4_fixed_" in path.name or "_phase4_sens_" in path.name:
            continue
        for r in rows(path):
            out[r["method"]][int(r["iteration"])].append(float(r["shots"]))
    return out


def fig_trajectory(out: Path) -> None:
    cases = [("H4_square_eq_side1p0_HF", "H$_4$ 1.0 Å"), ("LiH_R3p0_HF", "LiH 3.0 Å"), ("H2O_eq_HF", "H$_2$O eq")]
    fig, axes = plt.subplots(2, len(cases), figsize=(7.2, 3.8), sharex="col")
    drawn = False
    for col, (case, title) in enumerate(cases):
        data = selections_by_method(case)
        for method, (colour, marker, label) in STYLE.items():
            if method not in data:
                continue
            steps = sorted(data[method])
            median = [np.median(data[method][k]) for k in steps]
            axes[0, col].plot(steps, median, color=colour, marker=marker, ms=3, lw=1.1, label=label)
            axes[1, col].plot(steps, np.cumsum(median), color=colour, marker=marker, ms=3, lw=1.1)
            drawn = True
        for row in (0, 1):
            axes[row, col].set_yscale("log")
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("ADAPT step")
    if not drawn:
        return
    axes[0, 0].set_ylabel("per selection\n(median shots)")
    axes[1, 0].set_ylabel("cumulative\n(median shots)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    save(fig, out, "fig_trajectory")


# --- shots against circuit depth ---------------------------------------------------------------------------------------------


def fig_depth(out: Path) -> None:
    data = rows(RUNS / "paper_a" / "depth_frontier.csv")
    if not data:
        return
    cases = [("H4_square_eq_side1p0_CISD", "H$_4$ 1.0 Å, CISD"), ("LiH_R3p0_HF", "LiH, HF"), ("H2O_eq_CISD", "H$_2$O eq, CISD")]
    # (sign-aware row, pairwise row used where the sign-aware one was not run, colour, marker, label)
    methods = [("II-A, safe", "II-A", BLUE, "o", "II-A"), ("II-0, safe", "II-0", GREY, "s", "II-0"),
               ("M1 seq, safe", "M1 seq", GREEN, "v", "M1 seq"),
               ("M2 FC, safe", "M2 FC", ORANGE, "^", "M2 (independent arms)"), ("M2 QWC, safe", "M2 QWC", ORANGE, "^", None)]
    family_name = {"FC": "fully commuting", "blocks=4": "4-qubit blocks", "blocks=2": "2-qubit blocks", "QWC": "qubit-wise",
                   "pivot (published)": "pivot (published)", "pivot (merged)": "pivot (merged)", "independent arms": "independent"}
    fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 2.6), sharey=False)
    for ax, (case, title) in zip(axes, cases):
        mine = [r for r in data if r["case"] == case]
        for method, fallback, colour, marker, label in methods:
            pts = []
            for family in dict.fromkeys(r["family"] for r in mine):
                for name in (method, fallback):  # the sign-aware row, else the pairwise row if it was right in 95% of the trials
                    row = next((r for r in mine if r["family"] == family and r["method"] == name), None)
                    if row and row["cz_per_shot"] not in ("", "nan") and float(row["correct_rate"]) >= 0.95:
                        pts.append((float(row["cz_per_shot"]), float(row["shots_mean"]), family))
                        break
            pts = [p for p in pts if np.isfinite(p[0])]
            if not pts:
                continue
            pts.sort()
            if method.startswith("M2"):
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=colour, marker=marker, s=18, label=label, zorder=3)
            else:
                fcs = [p for p in pts if p[2] in ("FC", "blocks=4", "blocks=2", "QWC")]
                ax.plot([p[0] for p in fcs], [p[1] for p in fcs], color=colour, marker=marker, ms=4, lw=1.0, label=label)
                for p in pts:
                    if p[2].startswith("pivot"):
                        ax.scatter([p[0]], [p[1]], color=colour, marker=marker, s=22, facecolors="none", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("two-qubit gates per shot")
        ax.set_title(title)
    axes[0].set_ylabel("context-shots (mean)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.text(0.5, -0.04, "Lines: fully commuting, 4-qubit blocks, 2-qubit blocks, qubit-wise (right to left).  Open markers: pivot contexts "
             "(published and merged).  Sign-aware rule (pairwise where it was not run); rows right in at least 95% of the trials.",
             ha="center", fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig_depth")


# --- the fixed-budget baseline ------------------------------------------------------------------------------------------------


def fig_fixed_budget(out: Path) -> None:
    data = rows(RUNS / "paper_a" / "fixed_budget.csv")
    certified = {r["case"]: r for r in rows(RUNS / "paper_a" / "fixed_budget_summary.csv") if r["allocation"] == "designed"}
    if not data:
        return
    cases = list(dict.fromkeys(r["case"] for r in data))
    title = {"H4_square_eq_side1p0_HF": "H$_4$ 1.0 Å", "H4_square_stretch_side2p0_HF": "H$_4$ 2.0 Å",
             "LiH_R3p0_HF": "LiH 3.0 Å", "LiH_R1p6_HF": "LiH 1.6 Å", "H2O_eq_HF": "H$_2$O eq", "H2O_stretch_HF": "H$_2$O str"}
    fig, axes = plt.subplots(2, len(cases), figsize=(1.9 * len(cases) + 0.6, 3.8), sharey="row", squeeze=False)
    for col, case in enumerate(cases):
        for allocation, colour, marker in (("designed", BLUE, "o"), ("pilot", GREEN, "^"), ("uniform", ORANGE, "s")):
            mine = sorted((r for r in data if r["case"] == case and r["allocation"] == allocation), key=lambda r: int(r["budget"]))
            if not mine:
                continue
            b = [int(r["budget"]) for r in mine]
            axes[0, col].plot(b, [float(r["reached_rate"]) for r in mine], color=colour, marker=marker, ms=3, lw=1.0, label=f"{allocation} allocation")
            axes[1, col].plot(b, [float(r["total_shots_median"]) for r in mine], color=colour, marker=marker, ms=3, lw=1.0)
        c = certified.get(case)
        if c:
            for method, colour, ls, label in (("II-A data, safe", BLUE, "--", "II-A, certified"), ("II-0 safe", GREY, "--", "II-0, certified"),
                                              ("M2 safe", ORANGE, "--", "M2, certified")):
                value = float(c[f"certified {method}"])
                if np.isfinite(value):
                    axes[1, col].axhline(value, color=colour, ls=ls, lw=0.9, label=label)
        for row in (0, 1):
            axes[row, col].set_xscale("log")
        axes[1, col].set_yscale("log")
        axes[0, col].set_title(title.get(case, case))
        axes[1, col].set_xlabel("shots per selection")
    axes[0, 0].set_ylabel("trajectories reaching\nchemical accuracy")
    axes[1, 0].set_ylabel("total shots (median)")
    axes[0, 0].legend(frameon=False, fontsize=6)
    axes[1, -1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_fixed_budget")


FIGURES = {"fig_pipeline": fig_pipeline, "fig_rounds": fig_rounds, "fig_trajectory": fig_trajectory, "fig_depth": fig_depth,
           "fig_fixed_budget": fig_fixed_budget}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "reports" / "figures")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = parser.parse_args()
    for name in args.only:
        FIGURES[name](args.out)


if __name__ == "__main__":
    main()
