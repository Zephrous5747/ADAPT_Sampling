#!/usr/bin/env python3
"""Paper A, critique round (Q10): the device shot rate below which the saving in context-shots outweighs the classical time of the design.

For a state with a saving of ``dN`` context-shots per selection (mean shots of the best valid baseline minus mean shots of II-A, both at n_min = 400) and a classical time
``T`` of one II-A selection on one core, the learned selection is faster in wall-clock time than the baseline when the device runs at fewer than ``r* = dN / T`` shots per
second (the baselines are charged no classical time, which favors them).  The classical time is taken from the last column of Table ``tab:systems`` (``tab_systems.tex``; one
state per molecule) for the STO-3G systems and from the stored trials of the active-space states (``cas400_summary.csv``, hours per trial).  The design can be run on more than one
core and overlapped with the measurement of the next round, so ``r*`` is a conservative threshold.

Writes ``reports/generated/tab_breakeven.tex`` and ``runs/paper_a/breakeven.csv``.

    python scripts/paper_a_breakeven.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent / "reports"
SYSTEMS = [("H4", r"H$_4$", r"H\$_4\$"), ("LiH", "LiH", "LiH"), ("BeH2", r"BeH$_2$", r"BeH\$_2\$"), ("H2O", r"H$_2$O", r"H\$_2\$O")]


def classical_seconds() -> dict[str, float]:
    out = {}
    for line in (REPORTS / "generated" / "tab_systems.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        for key, label, _ in SYSTEMS:
            if cells[0].startswith(label) and len(cells) >= 8 and "CAS" not in cells[0]:
                out[key] = float(re.sub(r"[^0-9.]", "", cells[7].replace("{,}", "")))
    return out


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return rf"${float(m):.1f}\times10^{{{int(e)}}}$"


def main() -> None:
    seconds = classical_seconds()
    head = pd.read_csv(ROOT / "runs" / "paper_a" / "minshots_headline_n400.csv")
    head["system"] = head.case.map(lambda c: "H4" if c.startswith("H4") else "LiH" if c.startswith("LiH") else "BeH2" if c.startswith("BeH2") else "H2O")
    rows = []
    for key, label, _ in SYSTEMS:
        sub = head[head.system == key]
        saving = sub.best_mean - sub.ii_a_mean
        rate = saving / seconds[key]
        rows.append({"system": label, "states": len(sub), "classical_s": seconds[key], "saving_min": saving.min(), "saving_max": saving.max(), "rate_min": rate.min(), "rate_max": rate.max(),
                     "rate_median": float(rate.median())})
    cas = pd.read_csv(ROOT / "runs" / "paper_a" / "cas400_summary.csv")
    cas = cas[cas.n_min == 400]
    saving, hours, rate = [], [], []
    for state, g in cas.groupby("state"):
        a = g[g.method == "II-A"].iloc[0]
        others = g[g.method != "II-A"]
        others = others[others.within_rho >= 0.95]
        dn = others.shots_mean.min() - a.shots_mean
        saving.append(dn)
        hours.append(a.classical_hours_per_trial)
        rate.append(dn / (a.classical_hours_per_trial * 3600))
    rows.append({"system": r"H$_2$O CAS(8e,8o)", "states": len(saving), "classical_s": float(np.mean(hours)) * 3600, "saving_min": min(saving), "saving_max": max(saving),
                 "rate_min": min(rate), "rate_max": max(rate), "rate_median": float(np.median(rate)), "hours_min": min(hours), "hours_max": max(hours)})
    with (ROOT / "runs" / "paper_a" / "breakeven.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[-1]), restval="")
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"System & states & classical time per selection (s) & saving in shots per selection & break-even rate $r^\ast$ (shots/s) \\", r"\midrule"]
    for r in rows:
        t = f"{r['hours_min']:.1f} to {r['hours_max']:.1f}~h" if "CAS" in r["system"] else f"{r['classical_s']:,.0f}".replace(",", "{,}")
        lines.append(f"{r['system']} & {r['states']} & {t} & {sci(r['saving_min'])} to {sci(r['saving_max'])} & {sci(r['rate_min'])} to {sci(r['rate_max'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (REPORTS / "generated" / "tab_breakeven.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
