#!/usr/bin/env python3
"""Energy against cumulative measurement cost along measured ADAPT runs: selection and optimisation together.

The feature study reports the selection shots (``selection_shots``) and the modelled optimisation cost (``C_opt``) of a run
separately.  What the protocol is judged by is the total, and by the energy reached for it.  For every run of
``runs/features/<case>/ci_*.jsonl`` the cost after step ``k`` is

    cum_k = sum_{j<k} (selection shots of step j  +  C_opt of the re-optimisation after step j)

and the energy error after step ``k`` is ``energy_errors[k]``.  Per case and method (confidence level, fixed budget) the table gives

* trajectories, steps, share that reach the target (chemical accuracy, or what exact ADAPT reaches when it stalls above it);
* the median total cost to the end of the run, split into selection and optimisation, and the selection share;
* the total cost to *first* reaching chemical accuracy (the run ends there, so it is the total),
* the same relative to the cheapest certified level, so that "shots saved by a lower confidence level" is read off the total.

    python scripts/energy_cost_curves.py [--runs runs/features] [--cases H4_square_eq_side1p0_HF ...]

Writes ``<runs>/report/cost_totals.csv`` and ``cost_curves.pdf/png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {"H4_square_eq_side1p0_HF": "H4 1.0", "H4_square_stretch_side2p0_HF": "H4 2.0", "LiH_R3p0_HF": "LiH 3.0",
         "H2O_eq_HF": "H2O eq", "H2O_stretch_HF": "H2O str"}
SLACK = 1e-4


def load_runs(case_dir: Path) -> pd.DataFrame:
    rows = []
    ref_path = case_dir / "exact_reference.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    for path in sorted(case_dir.glob("ci_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sel = np.array([s["shots"] for s in r["selections"]], float)
            copt = np.array(r["c_opt_steps"], float)
            n = min(len(sel), len(copt))
            cum = np.concatenate([[0.0], np.cumsum(sel[:n] + copt[:n])])
            rows.append({"method": r["method"], "delta": r["delta"], "index": r["index"], "source": r.get("source", "measured"),
                         "iterations": r["iterations"], "final_error": r["final_error"],
                         "selection": float(sel.sum()), "optimisation": float(copt.sum()),
                         "total": float(sel.sum() + copt.sum()), "cum": cum, "errors": np.array(r["energy_errors"], float),
                         "evals": r["optimizer_evaluations"]})
    df = pd.DataFrame(rows)
    df.attrs["ref"] = ref
    return df


def summarise(case: str, df: pd.DataFrame) -> pd.DataFrame:
    ref = df.attrs["ref"]
    target = 1.6e-3 if ref is None or ref["final_error"] < 1.6e-3 else ref["final_error"] + SLACK
    out = []
    for (method, delta), g in df.groupby(["method", "delta"]):
        reached = (g.final_error <= target).to_numpy()
        out.append({"case": case, "system": LABEL.get(case, case), "method": method, "confidence_pct": 100 * (1 - delta),
                    "n": len(g), "reached_pct": 100 * reached.mean(), "steps_mean": g.iterations.mean(),
                    "selection_median": g.selection.median(), "optimisation_median": g.optimisation.median(),
                    "total_median": g.total.median(), "total_mean": g.total.mean(),
                    "total_se": g.total.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                    "selection_share_pct": 100 * float(g.selection.median() / g.total.median()),
                    "evals_mean": g.evals.mean()})
    t = pd.DataFrame(out)
    certified = t[t.method.str.startswith("II-A")]
    if len(certified):
        base = certified.sort_values("total_mean").iloc[0]
        t["total_vs_cheapest_certified"] = t.total_mean / base.total_mean
        hi = certified.sort_values("confidence_pct").iloc[-1]
        t["total_vs_highest_confidence"] = t.total_mean / hi.total_mean
    return t


def curves(case: str, df: pd.DataFrame, ax) -> None:
    palette = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#666666"]
    groups = sorted(df.groupby(["method", "delta"]), key=lambda kv: (kv[0][0].startswith("Fixed"), kv[0][1]))
    for i, ((method, delta), g) in enumerate(groups):
        k = min(len(c) for c in g.cum)
        x = np.median(np.stack([c[:k] for c in g.cum]), axis=0)
        y = np.median(np.stack([e[:k] for e in g.errors]), axis=0)
        label = f"{method} ({100 * (1 - delta):.0f}%)" if method.startswith("II-A") else method
        ax.plot(x[1:], y[1:] * 1e3, marker="o", ms=2, lw=1, color=palette[i % len(palette)], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.6, color="#999999", lw=0.7, ls=":")
    ax.set_title(LABEL.get(case, case))
    ax.set_xlabel("cumulative shots (selection + optimisation)")


def totals_figure(frames: dict, out: Path) -> None:
    """Per system: total cost of every method (selection + optimisation) as a multiple of the II-A total at 95 % (or the highest level present)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(frames), figsize=(2.9 * len(frames), 3.2), squeeze=False)
    for ax, (case, df) in zip(axes[0], frames.items()):
        t = summarise(case, df)
        ref = t[(t.method.str.startswith("II-A")) & ((t.confidence_pct - 95).abs() < 0.5)]
        ref = (ref if len(ref) else t[t.method.str.startswith("II-A")].sort_values("confidence_pct").iloc[[-1]]).iloc[0]
        t = t.assign(label=[f"{m.replace(', safe', '').replace(' data', '')} {c:.0f}%" if m.startswith("II-A") else
                            (m.replace(", pilot", "") if m.startswith("Fixed") else m)
                            for m, c in zip(t.method, t.confidence_pct)])
        t = t.sort_values(["method", "confidence_pct"], ascending=[True, False]).reset_index(drop=True)
        y = np.arange(len(t))
        sel = t.selection_median / ref.total_median
        opt = t.optimisation_median / ref.total_median
        ax.barh(y, opt, color="#BBBBBB", label="optimisation (C_opt model)")
        ax.barh(y, sel, left=opt, color="#D55E00", label="selection shots")
        for yi, (s, o, r) in enumerate(zip(t.selection_share_pct, t.total_median / ref.total_median, t.reached_pct)):
            ax.text(o + 0.02, yi, f"{s:.2g}%" + ("" if r >= 95 else f" (reach {r:.0f}%)"), va="center", fontsize=5.5)
        ax.axvline(1.0, color="#666666", lw=0.7, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(t.label, fontsize=5.5)
        ax.set_xlim(0, max(1.6, float((t.total_median / ref.total_median).max()) * 1.35))
        ax.set_title(LABEL.get(case, case))
        ax.set_xlabel("total shots / II-A (95%)")
    axes[0][-1].legend(fontsize=5.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.35))
    fig.tight_layout()
    fig.savefig(out / "cost_totals.pdf", bbox_inches="tight")
    fig.savefig(out / "cost_totals.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/features"))
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    cases = args.cases or [p.name for p in sorted(args.runs.iterdir()) if p.is_dir() and p.name.endswith("_HF")]
    out = args.runs / "report"
    out.mkdir(parents=True, exist_ok=True)
    tables, frames = [], {}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for case in cases:
        df = load_runs(args.runs / case)
        if df.empty:
            continue
        frames[case] = df
        t = summarise(case, df)
        tables.append(t)
        cols = ["method", "confidence_pct", "n", "reached_pct", "steps_mean", "selection_median", "optimisation_median", "total_median",
                "selection_share_pct", "total_vs_highest_confidence"]
        print(f"\n{LABEL.get(case, case)}")
        print(t[[c for c in cols if c in t]].round(4).to_string(index=False))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "cost_totals.csv", index=False)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
        fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 2.8), squeeze=False)
        for ax, (case, df) in zip(axes[0], frames.items()):
            curves(case, df, ax)
        axes[0][0].set_ylabel("energy error (mHa), median")
        axes[0][0].legend(fontsize=5.5, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "cost_curves.pdf", bbox_inches="tight")
        fig.savefig(out / "cost_curves.png", bbox_inches="tight", dpi=200)
        totals_figure(frames, out)


if __name__ == "__main__":
    main()
