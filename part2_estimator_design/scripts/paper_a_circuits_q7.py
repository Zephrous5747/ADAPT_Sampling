#!/usr/bin/env python3
"""Paper A, second critique list (Q7): circuits per selection, from the runs under the labels ', circuits' (``circuit_batches``, ``contexts_max_round``).

For the six states and the four methods of ``cluster/jobs/q7_circuits.txt`` (n_min = 400, rho = 0): median rounds, median circuit configurations loaded per selection (``circuit_batches``:
contexts that receive shots, summed over the rounds), median largest number of contexts in one round, median shots, shots per circuit loaded, and whether the selections and shots
reproduce the runs under the plain labels (same trial numbers and seed).

Writes ``runs/paper_a/circuits_q7.csv``; ``reports/generated/tab_circuits.tex`` (tabular body).

    python scripts/paper_a_circuits_q7.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parent / "reports" / "generated"
STEMS = {"II-A": "II-A_data_safe_bound_start_min_shots=400", "II-0": "II-0_safe_bound_start_min_shots=400", "M1 seq": "M1_seq_safe_bound_start_min_shots=400",
         "M2": "M2_safe_bound_start_min_shots=400"}
NAMES = {"H4_square_eq_side1p0_CISD": r"H$_4$ 1.0~\AA, CISD", "H4_square_stretch_side2p0_ADAPT5": r"H$_4$ 2.0~\AA, ADAPT 5", "LiH_R3p0_HF": r"LiH 3.0~\AA, HF",
         "LiH_R3p0_ADAPT5": r"LiH 3.0~\AA, ADAPT 5", "LiH_R1p6_ADAPT3": r"LiH 1.6~\AA, ADAPT 3", "BeH2_ADAPT4": r"BeH$_2$, ADAPT 4"}


def main() -> None:
    rows = []
    for case in NAMES:
        for method, stem in STEMS.items():
            circ = glob.glob(str(ROOT / "runs" / case / "trials" / f"step5__{stem}*_circuits__shard*.csv"))
            if not circ:
                continue
            c = pd.read_csv(circ[0])
            if len(c) != int(c.n_trials.iloc[0]):
                continue
            ref = glob.glob(str(ROOT / "runs" / case / "trials" / f"step5__{stem}*__shard*.csv"))
            ref = [r for r in ref if "_circuits" not in r and "rho=" not in r and "anytime" not in r and "default" not in r]
            same = ""
            if ref:
                r = pd.read_csv(ref[0])
                if len(r) == len(c):
                    same = f"{int((r.selected.to_numpy() == c.selected.to_numpy()).sum())}/{len(c)}"
            rows.append({"case": case, "method": method, "trials": len(c), "rounds": c.rounds.median(), "circuit_batches": c.circuit_batches.median(),
                         "contexts_max_round": c.contexts_max_round.median(), "contexts_library": c.contexts_used.median(), "shots": c.shots.median(),
                         "shots_per_circuit": (c.shots / c.circuit_batches).median(), "correct": c.correct.mean(), "same_selection_as_plain": same})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "runs" / "paper_a" / "circuits_q7.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    body = []
    for case, name in NAMES.items():
        s = t[t.case == case].set_index("method")
        if "II-A" not in s.index:
            continue
        a = s.loc["II-A"]
        cb = [f"{s.loc[m, 'circuit_batches'] / a.circuit_batches:.2f}" for m in ("II-0", "M1 seq", "M2")]
        sh = [f"{s.loc[m, 'shots'] / a.shots:.2f}" for m in ("II-0", "M1 seq", "M2")]
        body.append(f"{name} & {a.rounds:.0f} & {a.contexts_library:.0f} & {a.circuit_batches:.0f} & " + " & ".join(cb) + " & " + " & ".join(sh) + r" \\")
    text = "\n".join([r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
                      r" & & & II-A & \multicolumn{3}{c}{circuits over II-A} & \multicolumn{3}{c}{shots over II-A} \\", r"\cmidrule(lr){5-7}\cmidrule(l){8-10}",
                      r"State & rounds & library & circuits & II-0 & M1 seq & M2 & II-0 & M1 seq & M2 \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])
    (GEN / "tab_circuits.tex").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
