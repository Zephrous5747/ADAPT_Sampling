#!/usr/bin/env python3
"""Paper A, Results (RQ5): total cost of complete runs on H2O, split into selection shots and modeled optimization.

Reads ``cost_totals.csv`` of the three cluster rounds (optimizer threshold 1e-9, 1e-5, 1e-3), takes the certified methods at
1-delta = 95 % and plots the median total relative to II-A at the same threshold.  Writes ``reports/figures/fig_totalcost.pdf``.

    python scripts/paper_a_totalcost_figure.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "reports" / "figures"
THRESHOLDS = [("features", r"$10^{-9}$"), ("features_gtol1e-5_cl", r"$10^{-5}$"), ("features_gtol1e-3_cl", r"$10^{-3}$")]
CASES = [("H2O_eq_HF", r"H$_2$O at equilibrium"), ("H2O_stretch_HF", r"H$_2$O stretched")]
METHODS = [("II-A data, safe", "II-A (learned)"), ("II-0 safe", "II-0 (fixed)"), ("M2 safe", "M2 (independent elimination)"),
           ("M1 seq, safe", "M1 seq (sequential)")]
SEL, OPT = "#D55E00", "#BBBBBB"

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def load(folder: str, case: str) -> dict[str, dict]:
    rows = list(csv.DictReader((ROOT / "runs" / folder / "report" / "cost_totals.csv").open()))
    return {r["method"]: r for r in rows if r["case"] == case and abs(float(r["confidence_pct"]) - 95) < 0.5}


def main() -> None:
    fig, axes = plt.subplots(1, len(CASES), figsize=(7.2, 3.3), sharey=True)
    width = 0.2
    hatches = ["", "////", "xxxx", ".."]
    for ax, (case, title) in zip(axes, CASES):
        for g, (folder, _) in enumerate(THRESHOLDS):
            data = load(folder, case)
            ref = float(data["II-A data, safe"]["total_median"])
            for i, (key, _) in enumerate(METHODS):
                r = data[key]
                total = float(r["total_median"]) / ref
                sel = total * float(r["selection_share_pct"]) / 100
                x = g + (i - 1.5) * (width + 0.02)
                ax.bar(x, total - sel, width, color=OPT, edgecolor="#444444", lw=0.5)
                ax.bar(x, sel, width, bottom=total - sel, color=SEL, edgecolor="#444444", lw=0.5)
                ax.text(x, 0.06, str(i + 1), ha="center", va="bottom", fontsize=6.5)
                ax.text(x, total + 0.05, f"{float(r['selection_share_pct']):.2g}%", ha="center", va="bottom", fontsize=6.2, rotation=90)
        ax.axhline(1.0, color="#666666", lw=0.7, ls=":")
        ax.set_xticks(range(len(THRESHOLDS)))
        ax.set_xticklabels([t for _, t in THRESHOLDS])
        ax.set_xlabel(r"optimizer threshold $\epsilon_{\mathrm{opt}}$")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 5.8)
    axes[0].set_ylabel("total cost relative to II-A")
    from matplotlib.patches import Patch

    handles = [Patch(fc=SEL, ec="#444444", lw=0.5, label="selection shots"), Patch(fc=OPT, ec="#444444", lw=0.5, label=r"modeled optimization $C_{\mathrm{opt}}$")]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.5, 0.0, "Bars in each group, left to right: " + ";  ".join(f"{i + 1} {lab}" for i, (_, lab) in enumerate(METHODS)) + ".\n"
             "Number above a bar: share of the selection in the total.", ha="center", fontsize=6.8, linespacing=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_totalcost.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_totalcost.png", bbox_inches="tight", dpi=200)
    print("wrote", OUT / "fig_totalcost.pdf")


if __name__ == "__main__":
    main()
