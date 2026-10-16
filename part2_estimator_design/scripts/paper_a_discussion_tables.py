#!/usr/bin/env python3
"""LaTeX tables of the Discussion of Paper A, generated from the stored runs.

Run ``paper_a_depth_tables.py`` first (it collects the depth-frontier rows into ``runs/paper_a/depth_frontier.csv``);
the termination and score-axis tables come from ``paper_a_termination_cost.py`` and ``paper_a_score_axis_cost.py``;
the noise and radius-rule tables read the merged per-state tables ``runs/<case>/<case>_step5_external_baselines.csv``.
Written to ``reports/generated/discussion_tables.tex``; a cell is a mean cost in context-shots unless stated, a dash
means the configuration was not run.

    python scripts/paper_a_discussion_tables.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

STATES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0 CISD"), ("LiH_R3p0_HF", "LiH HF"),
          ("H2O_eq_CISD", r"H$_2$O eq CISD"), ("H2O_stretch_CISD", r"H$_2$O str CISD")]
FAMILIES = [("QWC", "qubit-wise (blocks of 1)"), ("blocks=2", "blocks of 2 qubits"), ("blocks=4", "blocks of 4 qubits"),
            ("FC", "fully commuting"), ("pivot (published)", "pivot, as published"), ("pivot (merged)", "pivot, merged")]
NOISE_LEVELS = (0.003, 0.01, 0.03, 0.06)


PRETTY = dict(STATES) | {
    "LiH_R3p0_ADAPT3": r"LiH ADAPT$_3$", "LiH_R3p0_ADAPT5": r"LiH ADAPT$_5$", "LiH_R3p0_ADAPT6": r"LiH ADAPT$_6$",
    "H2O_eq_ADAPT11": r"H$_2$O eq ADAPT$_{11}$", "H2O_eq_ADAPT17": r"H$_2$O eq ADAPT$_{17}$",
    "H2O_stretch_ADAPT8": r"H$_2$O str ADAPT$_8$", "H2O_stretch_ADAPT18": r"H$_2$O str ADAPT$_{18}$",
    "H4_square_eq_side1p0_ADAPT3": r"H$_4$ 1.0 ADAPT$_3$", "H4_square_eq_side1p0_ADAPT10": r"H$_4$ 1.0 ADAPT$_{10}$",
    "H4_square_stretch_side2p0_ADAPT3": r"H$_4$ 2.0 ADAPT$_3$", "H4_square_stretch_side2p0_ADAPT10": r"H$_4$ 2.0 ADAPT$_{10}$"}


def pretty(case: str) -> str:
    return PRETTY.get(case, case.replace("_", r"\_"))


def sci(x: float) -> str:
    if x != x:
        return "--"
    if x >= 1e5:
        m, e = f"{x:.2e}".split("e")
        return rf"${float(m):.2f}\times10^{{{int(e)}}}$"
    return f"{x:,.0f}".replace(",", r"\,")


def ratio_text(x: float) -> str:
    """A ratio to the cost of II-A: two decimals below 10 (0.58 must not print as 1), thin-spaced integers above."""
    return f"{x:.2f}" if x < 10 else f"{x:,.0f}".replace(",", r"\,")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def tabular(header: list[str], rows: list[list[str]], spec: str, caption: str, label: str, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!htb]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" if r[0] != r"\midrule" else r"\midrule" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines)


def depth_table(runs: Path) -> str | None:
    data = read(runs / "paper_a" / "depth_frontier.csv")
    if not data:
        return None
    index = {(r["case"], r["family"], r["method"]): r for r in data}
    rows: list[list[str]] = []
    swapped: dict[str, float] = {}  # state -> share of wrong selections of the pairwise II-A that its sign-aware row replaces
    for case, name in STATES:
        found = [f for f, _ in FAMILIES if any((case, f, m) in index for m in ("M1 seq", "II-A", "II-0"))]
        if not found:
            continue
        rows.append([rf"\multicolumn{{7}}{{l}}{{\emph{{{name}}}}}"] + [""] * 0)
        for family, text in FAMILIES:
            cells = [index.get((case, family, m)) for m in ("M1 static", "M1 seq", "II-0", "II-A")]
            if not any(cells):
                continue
            cz = next((float(c["cz_per_shot"]) for c in reversed(cells) if c and c["cz_per_shot"] not in ("", "nan")), float("nan"))
            seq, a = cells[1], cells[3]
            dagger = [""] * 4
            mark = ""
            # Where the pairwise II-A is not valid (the published pivot contexts: every product is measured in several contexts with few
            # shots each) its cell is replaced by the sign-aware row, with the sign-aware M1 seq in the ratio, and marked.
            safe_a, safe_seq = index.get((case, family, "II-A, safe")), index.get((case, family, "M1 seq, safe"))
            if a and float(a["correct_rate"]) < 0.95 and safe_a and float(safe_a["correct_rate"]) >= 0.95:
                swapped[case] = 1.0 - float(a["correct_rate"])
                cells[3], a = safe_a, safe_a
                seq = safe_seq or seq
                dagger[3], mark = r"$^\dagger$", r"$^\dagger$"
            ratio = f"{float(seq['shots_mean']) / float(a['shots_mean']):.2f}{mark}" if seq and a else "--"
            rows.append([text, "--" if cz != cz else f"{cz:.1f}"]
                        + [(sci(float(c["shots_mean"])) + d) if c else "--" for c, d in zip(cells, dagger)] + [ratio])
        for method, family in (("M2 FC", "FC"), ("M2 QWC", "QWC")):
            r = index.get((case, "independent arms", method))
            if r:
                # independent BAI over II-A on the same family of contexts (both with the pairwise rule)
                a = index.get((case, family, "II-A"))
                ratio = rf"{float(r['shots_mean']) / float(a['shots_mean']):.2f}$^\ddagger$" if a else "--"
                rows.append([f"{method} (independent arms)", f"{float(r['cz_per_shot']):.1f}", "--", sci(float(r["shots_mean"])), "--", "--", ratio])
        rows.append([r"\midrule"])
    if rows and rows[-1] == [r"\midrule"]:
        rows.pop()
    caption = ("Shots against measurement-circuit depth. Every method on every family of measurement contexts: "
               "mean context-shots over trials, mean two-qubit gates per shot, and the gain of II-A over the sequential M1 "
               "on the same contexts. Blocks of $b$ qubits commute block by block, so each circuit is a product of Cliffords "
               "on $b$ qubits (at most $b(b-1)/2$ CZ per block); blocks of one qubit are qubit-wise commuting, and the pivot "
               "contexts are those of Anastasiou \\textit{et al.} The rows of independent arms give the cost of independent BAI (M2) "
               "in the column of the sequential M1, and a dash marks the columns that do not apply to it; $^\\ddagger$its ratio is to II-A on "
               "the same family of contexts (fully commuting or qubit-wise commuting).")
    if swapped:
        wrong = " and ".join(f"{100 * w:.0f}\\% of the trials on {dict(STATES)[c]}" for c, w in swapped.items())
        caption += (r" $^\dagger$Sign-aware rule, and the sign-aware M1 seq in the ratio: on the published pivot contexts every product is "
                    r"measured in several contexts with few shots each, the plug-in variances of such contexts are unreliable, and the "
                    r"pairwise II-A selected the wrong generator in " + wrong + ".")
    return tabular(["contexts", "CZ / shot", "M1 static", "M1 seq", "II-0", "II-A", "M1 seq / II-A"], rows, "lrrrrrr",
                   caption, "tab:depth", wide=True)


def merged_rows(runs: Path, case: str) -> dict[str, dict]:
    out = {}
    for stem in ("step4_learned_designs", "step5_external_baselines"):
        for row in read(runs / case / f"{case}_{stem}.csv"):
            out[row["config"]] = row
    return out


def noise_table(runs: Path, cases=("H4_square_eq_side1p0_CISD", "LiH_R3p0_HF")) -> str | None:
    families = [("M1 seq", "M1 seq, FC"), ("II-A data", "II-A, FC"), ("II-A data, blocks=4", "II-A, blocks of 4"),
                ("II-A data, blocks=2", "II-A, blocks of 2"), ("II-A data, blocks=1", "II-A, QWC"), ("M1 seq, blocks=1", "M1 seq, QWC"),
                ("Pivot merged II-A", "II-A, pivot contexts"), ("M2 marginal", "M2, FC"), ("M2 QWC marginal", "M2, QWC")]
    rows, any_found = [], False
    for case, name in STATES:
        if case not in cases:
            continue
        table = merged_rows(runs, case)
        block = []
        for config, text in families:
            cells, cz = [], float("nan")
            base = table.get(config)
            if not (base and base.get("cz_per_shot_mean_mean") not in (None, "")) and config == "II-A data":
                base = table.get("M1 seq")  # the step-4 table does not record the gates per shot; the same contexts as M1 seq
            if base and base.get("cz_per_shot_mean_mean") not in (None, ""):
                cz = float(base["cz_per_shot_mean_mean"])
            for p in NOISE_LEVELS:
                row = table.get(f"{config}, noise={p:g}/0")
                if row:
                    any_found = True
                    cells.append(f"{100 * float(row['correct_rate']):.0f}\\% / {sci(float(row['shots_mean']))}")
                else:
                    cells.append("--")
            if any(c != "--" for c in cells):
                block.append([text, "--" if cz != cz else f"{cz:.1f}"] + cells)
        if block:
            rows.append([rf"\multicolumn{{{2 + len(NOISE_LEVELS)}}}{{l}}{{\emph{{{name}}}}}"])
            rows += block
    if not any_found:
        return None
    return tabular(["contexts and method", "CZ"] + [f"$p_2={p:g}$" for p in NOISE_LEVELS], rows, "lr" + "r" * len(NOISE_LEVELS),
                   "Noisy measurement circuits. A depolarizing error of probability $p_2$ after every CZ gate; each cell is the share "
                   "of selections that picked the exact best generator (of the noiseless gradients) and the mean cost. "
                   "The intervals know shot noise only.", "tab:noise", wide=True)


def radius_table(runs: Path, cases=("H4_square_eq_side1p0_CISD", "LiH_R3p0_HF", "LiH_R3p0_ADAPT3", "LiH_R3p0_ADAPT5")) -> str | None:
    methods = [("M1 static", "M1 static, selection z", "M1 static"), ("M1 seq", "M1 seq, selection z", "M1 seq"),
               ("M2 pairwise", "M2 pairwise, selection z", "M2, pairwise"), ("II-0 estimated", "II-0, selection z", "II-0"),
               ("II-A data", "II-A data, selection z", "II-A"),
               ("M1 seq, safe", "M1 seq, safe, selection z", "M1 seq, sign-aware"),
               ("M2 safe", "M2 safe, selection z", "M2, sign-aware"),
               ("II-A data, safe", "II-A data, safe, selection z", "II-A, sign-aware")]
    names = dict(STATES) | {"LiH_R3p0_ADAPT3": r"LiH ADAPT$_3$", "LiH_R3p0_ADAPT5": r"LiH ADAPT$_5$"}
    rows, found = [], False
    for case in cases:
        table = merged_rows(runs, case)
        block = []
        for strict, loose, text in methods:
            a, b = table.get(strict), table.get(loose)
            if not (a and b):
                continue
            found = True
            block.append([text, sci(float(a["shots_median"])), f"{100 * float(a['correct_rate']):.0f}\\%",
                          sci(float(b["shots_median"])), f"{100 * float(b['correct_rate']):.0f}\\%"])
        if block:
            rows.append([rf"\multicolumn{{5}}{{l}}{{\emph{{{names.get(case, case)}}}}}"])
            rows += block
    if not found:
        return None
    return tabular(["method", "median (Bonferroni)", "correct", "median (selection $z$)", "correct"], rows, "lrrrr",
                   "The radius rule. Median cost and share of exact selections with the Bonferroni radius of the paper and with the "
                   "less conservative $z=z_{\\delta/2}/\\sqrt2$ calibrated to the selection error (about $2.2$ times smaller at $K=26$).",
                   "tab:radius", wide=True)


def termination_table(runs: Path) -> str | None:
    rows_in = read(runs / "paper_a" / "termination_cost.csv") + read(runs / "paper_a" / "termination_cost_h2o.csv")
    final = {"H4_square_eq_side1p0_ADAPT10", "H4_square_stretch_side2p0_ADAPT10", "LiH_R3p0_ADAPT6", "H2O_eq_ADAPT17",
             "H2O_stretch_ADAPT18"}  # the last state of every exact trajectory: where a real run would have to stop
    keep = [r for r in rows_in if r["case"] in final and r["certify_shots"] not in ("", "nan") and float(r["certify_shots"]) == float(r["certify_shots"])]
    if not keep:
        return None
    rows = []
    # measured II-A trajectory totals (Q8); the H2O certification rows were computed before those runs existed
    totals = {r["case"]: float(r["total_shots_median"]) for r in read(runs / "paper_a" / "sota_trajectories.csv")
              if r["method"] == "II-A data, safe"}
    for r in keep:
        traj = r.get("trajectory_II-A", "")
        if traj in ("", "nan"):
            traj = totals.get(r["case"].split("_ADAPT")[0] + "_HF", "")
        ratio = r.get("over_trajectory_II-A")
        if ratio in ("", "nan", None) and traj != "":
            ratio = float(r["certify_shots"]) / float(traj)
        rows.append([pretty(r["case"]), f"{float(r['max_abs_gradient']):.1e}", f"{float(r['tau']):.1e}",
                     sci(float(r["certify_shots"])), sci(float(traj)) if traj not in ("", "nan") else "--",
                     f"{float(ratio):.2f}" if ratio not in ("", "nan", None) else "--"])
    return tabular(["state", r"$\max\lvert g\rvert$", r"$\tau$", "certify $\\max\\lvert g_i\\rvert<\\tau$", "trajectory, II-A", "ratio"], rows,
                   "lrrrrr", "Certifying the stop. Planning bound (exact variances, fully commuting contexts) for the shots that give every gradient "
                   "the radius $\\tau-\\lvert g_i\\rvert$, and the measured selection cost of the whole trajectory with II-A.", "tab:termination", wide=True)


def score_table(runs: Path) -> str | None:
    rows_in = [r for r in read(runs / "paper_a" / "score_axis_cost.csv") if int(r["landscape_points"]) == 5]
    if not rows_in:
        return None
    by_case: dict[str, dict[str, dict]] = {}
    for r in rows_in:
        by_case.setdefault(r["case"], {})["matched" if r["gap_matched"] == "True" else f"{float(r['epsilon']):g}"] = r
    rows = []
    for case, cells in by_case.items():
        matched, fixed = cells.get("matched"), cells.get("0.001")
        if not (matched and fixed):
            continue
        rows.append([pretty(case), fixed["n_generators"], sci(float(fixed["II-A"])),
                     sci(float(fixed["energy_score_cost"])), ratio_text(float(fixed["over_II-A"])),
                     f"{1e3 * float(matched['epsilon']):.2g}", sci(float(matched["energy_score_cost"])),
                     ratio_text(float(matched["over_II-A"]))])
    header = ["state", "$K$", "II-A", "energy scores, 1~mHa", "ratio", r"$\varepsilon_E$ matched (mHa)", "energy scores, matched", "ratio"]
    caption = (r"The cost side of an energy-based score. Each candidate needs its own energy landscape (five energies) on its own state, so "
               r"nothing is shared across candidates; energies are measured with the fully commuting groups of $\hat H$. Columns 4--5: every energy to 1~mHa. "
               r"Columns 6--8: every energy to the precision that resolves the energy lowerings $g^2/2h$ of the two best candidates ($h=1$~Ha assumed). "
               r"Ratios are to the mean cost of II-A.")
    return tabular(header, rows, "lrrrrrrr", caption, "tab:score", wide=True)


def splice_named(paper: Path, parts: dict[str, str | None]) -> list[str]:
    """Put every table between its ``% BEGIN GENERATED disc_<name>`` and ``% END GENERATED disc_<name>`` lines of the paper."""
    text = paper.read_text(encoding="utf-8")
    done = []
    for name, block in parts.items():
        begin, end = f"% BEGIN GENERATED disc_{name}", f"% END GENERATED disc_{name}"
        if block is None or begin not in text or end not in text:
            continue
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + begin + "\n" + block + "\n" + end + tail
        done.append(name)
    with paper.open("w", encoding="utf-8", newline="\n") as handle:  # the paper is LF-only; write_text would give CRLF on Windows
        handle.write(text)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("../reports/generated/discussion_tables.tex"))
    parser.add_argument("--splice", type=Path, default=None, help="paper whose % BEGIN/END GENERATED disc_<name> blocks are filled")
    args = parser.parse_args()
    parts = {"depth": depth_table(args.runs), "noise": noise_table(args.runs), "radius": radius_table(args.runs),
             "termination": termination_table(args.runs), "score": score_table(args.runs)}
    text = "% Generated by part2_estimator_design/scripts/paper_a_discussion_tables.py from the stored runs; do not edit by hand.\n"
    for name, block in parts.items():
        text += f"\n% --- {name} ---\n" + (block if block else f"% (no data for the {name} table yet)") + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    if args.splice is not None:
        print("spliced:", ", ".join(splice_named(args.splice, parts)) or "nothing (no markers or no data)")


if __name__ == "__main__":
    main()
