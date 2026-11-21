"""Hierarchy of the methods of the paper (Q1): the gain of every step, the total gain and the exact-covariance ceiling.

Arms (paper names): IG-SE (successive elimination, individual grouping: SE only), SG-FULL (shared contexts, no elimination: FC only), SG-SE
(both: the first combined baseline), SG-SE-CS (plus coefficient splitting and an allocation designed from covariances, the covariance taken from
measured outcomes), SG-SE-CS+prior (the covariance taken from a classical state), and the exact-covariance arm of runs/reframe_v10_oracle
(the covariance of the true state: a reference, not a bound on the shots of any procedure).

A gain is the mean selection shots of the earlier method over those of the later one on the same state; the table and the figure give its median
over the states with the minimum and the maximum.  Settings: rho = 0 ("strict", all states) and rho = 0.1.

    python scripts/reframe_hierarchy.py    ->  generated/gs_hierarchy_body.tex, figures/fig_gs_hierarchy.pdf, facts in generated/gs_facts.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reframe_summary as rs  # noqa: E402  (helpers: matrix_cells, load, shots, certified, state labels)

PAPER, GEN = rs.PAPER, rs.GEN
FIG = PAPER / "figures"
FIG.mkdir(exist_ok=True)

NAME = {"IG": "IG-SE", "FULL": "SG-FULL", "SE": "SG-SE", "CS": "SG-SE-CS", "PRIOR": "SG-SE-CS+prior", "CS_O": "exact covariance", "SE_O": "SG-SE, exact covariance"}
# (key, numerator, denominator, label)
STEPS = [
    ("ig_se", "IG", "SE", "SE only $\\to$ SE + FC (IG-SE $\\to$ SG-SE)"),
    ("full_se", "FULL", "SE", "FC only $\\to$ SE + FC (SG-FULL $\\to$ SG-SE)"),
    ("se_cs", "SE", "CS", "SE + FC $\\to$ SG-SE-CS (measured covariance)"),
    ("se_prior", "SE", "PRIOR", "SE + FC $\\to$ SG-SE-CS+prior (classical state)"),
    ("se_oracle", "SE", "CS_O", "SE + FC $\\to$ exact covariance (reference)"),
    ("se_o_cs_o", "SE_O", "CS_O", "SG-SE $\\to$ SG-SE-CS, both exact (CS alone)"),
    ("ig_cs", "IG", "CS", "IG-SE $\\to$ SG-SE-CS (total)"),
    ("ig_prior", "IG", "PRIOR", "IG-SE $\\to$ SG-SE-CS+prior (total)"),
    ("ig_oracle", "IG", "CS_O", "IG-SE $\\to$ exact covariance (total)"),
    ("cs_learn", "CS", "CS_O", "SG-SE-CS / exact covariance"),
    ("prior_learn", "PRIOR", "CS_O", "SG-SE-CS+prior / exact covariance"),
]


def per_state(setting: str) -> dict:
    cells, data = rs.matrix_cells(), rs.load("runs/reframe_v10_oracle")
    orc = defaultdict(dict)
    for key, rows in data.items():
        state, method, st, _ = key.split("__")
        orc[(state, st)][{"FC_SE_ORACLE": "SE_O", "FC_SE_CS_ORACLE": "CS_O"}[method]] = rows
    out: dict = defaultdict(dict)
    for (state, st), d in cells.items():
        if st != setting or state.endswith("CISD"):
            continue
        for arm, name in (("IG_SE", "IG"), ("SG_FULL", "FULL"), ("SG_SE", "SE"), ("SG_SE_CS", "CS"), ("SG_SE_CS+P", "PRIOR")):
            if arm in d and d[arm]:
                out[state][name] = (float(np.mean(rs.shots(d[arm]))), rs.certified(d[arm]) >= 0.95 * len(d[arm]))
        for name, rows in orc.get((state, setting), {}).items():
            out[state][name] = (float(np.mean(rs.shots(rows))), rs.certified(rows) >= 0.95 * len(rows))
    return out


def gains(per: dict, num: str, den: str) -> dict:
    return {s: d[num][0] / d[den][0] for s, d in per.items() if num in d and den in d and d[num][1] and d[den][1]}


def cell(v) -> str:
    v = np.array(list(v), float)
    return f"{np.median(v):.2f} ({v.min():.2f}--{v.max():.2f})"


def main() -> None:
    result: dict = {}
    for setting in ("strict", "rho010"):
        per = per_state(setting)
        result[setting] = {key: gains(per, num, den) for key, num, den, _ in STEPS}
    # at rho = 0.1 the reference was run on the 17 STO-3G states only: the rows that involve it, and the rows compared with them, use those states
    matched = set(result["rho010"]["se_oracle"])
    for key in ("se_cs", "se_prior", "se_oracle", "se_o_cs_o", "cs_learn", "prior_learn"):
        result["rho010"][key] = {s: v for s, v in result["rho010"][key].items() if s in matched}
    # table body
    lines = []
    for key, _, _, label in STEPS:
        g0, g1 = result["strict"][key], result["rho010"][key]
        lines.append(f"{label} & {len(g0)} / {len(g1)} & {cell(g0.values())} & {cell(g1.values())} \\\\")
        if key in ("se_o_cs_o", "ig_oracle"):
            lines.append(r"\midrule")
    (GEN / "gs_hierarchy_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", newline="\n")
    # facts for the text
    facts = {}
    for setting in ("strict", "rho010"):
        f = {}
        for key, g in result[setting].items():
            v = np.array(list(g.values()), float)
            f[key] = {"n": int(len(v)), "median": round(float(np.median(v)), 2), "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)}
        # share of the exact-covariance gain over SG-SE regained, log scale, states where that gain exceeds 1.15
        for arm, key in (("measured", "se_cs"), ("prior", "se_prior")):
            states = [s for s in result[setting][key] if s in result[setting]["se_oracle"] and result[setting]["se_oracle"][s] > 1.15]
            frac = [np.log(result[setting][key][s]) / np.log(result[setting]["se_oracle"][s]) for s in states]
            f[f"regained_{arm}"] = {"n": len(frac), "median": round(float(np.median(frac)), 2), "min": round(float(min(frac)), 2), "max": round(float(max(frac)), 2)}
        for arm, key in (("measured", "cs_learn"), ("prior", "prior_learn")):
            f[f"within_1p25_{arm}"] = [int(sum(v <= 1.25 for v in result[setting][key].values())), len(result[setting][key])]
        facts[setting] = f
    path = GEN / "gs_facts.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["hierarchy"] = facts
    path.write_text(json.dumps(old, indent=1, sort_keys=True), newline="\n")
    for key, _, _, label in STEPS:
        print(f"[hierarchy] {key:12s} rho0 {cell(result['strict'][key].values())} n={len(result['strict'][key])} | rho0.1 {cell(result['rho010'][key].values())} n={len(result['rho010'][key])}")
    print("[hierarchy]", json.dumps({k: facts['strict'][k] for k in facts['strict'] if k.startswith(('regained', 'within'))}))
    draw(result["strict"])


def draw(g: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GREEN, PINK, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#555555"
    RED = "#b3261e"
    # layout in inches from the top of the figure
    W, H = 7.2, 7.4
    T_A, SUB_A, AX_A_TOP, AX_A_H = 0.07, 0.29, 0.50, 2.20
    T_B, AX_B_TOP, AX_B_H = 2.86, 3.22, 2.16
    fig = plt.figure(figsize=(W, H))

    def fy(inches_from_top):
        return 1.0 - inches_from_top / H

    def med(key):
        v = np.array(list(g[key].values()), float)
        return np.median(v), v.min(), v.max()

    # ---------------------------------------------------------------- (a) the methods and the median gain of every step
    ax = fig.add_axes([0.01, fy(AX_A_TOP + AX_A_H), 0.98, AX_A_H / H])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")
    BH, BW = 13.0, 23.0
    X1, X2, X3 = 0.5, 38.5, 76.5
    YT, YM, YB = 38.0, 22.5, 7.0  # box centres
    DARK = "#33414f"
    # key: title, description, (FC, SE, CS), face colour, edge style, x, centre y
    boxes = {
        "IG": ("IG-SE", "elimination only: one circuit\nset for every candidate", (0, 1, 0), "#ececec", "-", X1, YT),
        "FULL": ("SG-FULL", "shared contexts only:\nno elimination (full pool)", (1, 0, 0), "#ececec", "-", X1, YB),
        "SE": ("SG-SE", "shared contexts + elimination:\nthe combined baseline", (1, 1, 0), "#d6e7f4", "-", X2, YM),
        "CS": ("SG-SE-CS", "proposed: + coefficient splitting;\ncovariance learned from the shots", (1, 1, 1), "#cdebdc", "-", X3, YT),
        "PRIOR": ("SG-SE-CS+prior", "proposed: CS with the covariance\nseeded by a classical state", (1, 1, 1), "#cdebdc", "-", X3, YM),
        "CS_O": ("Exact covariance", "CS with the covariance of the\ntrue state: a reference", (1, 1, 1), "white", "--", X3, YB),
    }
    for k, (title, desc, chips, fc, ls, x0, yc) in boxes.items():
        y0 = yc - BH / 2
        ax.add_patch(FancyBboxPatch((x0, y0), BW, BH, boxstyle="round,pad=0.2,rounding_size=1.1", fc=fc, ec="#333333", lw=0.9, ls=ls))
        cx = x0 + BW / 2
        ax.text(cx, y0 + BH - 0.8, title, ha="center", va="top", fontsize=7.8, fontweight="bold")
        ax.text(cx, y0 + BH - 3.7, desc, ha="center", va="top", fontsize=6.5, linespacing=1.15, color="#222222")
        for j, (on, tag) in enumerate(zip(chips, ("FC", "SE", "CS"))):
            xc = cx + (j - 1) * 6.4
            ax.add_patch(FancyBboxPatch((xc - 2.6, y0 + 0.9), 5.2, 3.4, boxstyle="round,pad=0.05,rounding_size=0.8", fc=DARK if on else "white", ec=DARK if on else "#b5b5b5", lw=0.6, ls="-" if on else (0, (2, 1.5))))
            ax.text(xc, y0 + 2.6, tag, ha="center", va="center", fontsize=6.2, fontweight="bold", color="white" if on else "#b5b5b5")

    def arrow(a, b, key, ls="-", t=0.5):
        xa, ya = boxes[a][5] + BW + 0.9, boxes[a][6]
        xb, yb = boxes[b][5] - 0.9, boxes[b][6]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=10, lw=1.2, color="#222222", ls=ls, shrinkA=0, shrinkB=0, zorder=2))
        m = med(key)[0]
        ax.text(xa + t * (xb - xa), ya + t * (yb - ya), f"$\\times${m:.2f}", ha="center", va="center", fontsize=8.4, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc="white", ec="#777777", lw=0.7))

    arrow("IG", "SE", "ig_se")
    arrow("FULL", "SE", "full_se")
    arrow("SE", "CS", "se_cs")
    arrow("SE", "PRIOR", "se_prior")
    arrow("SE", "CS_O", "se_oracle", ls=(0, (3, 2)))
    ax.text(X2 + 0.2, 12.6, "Ingredients", fontsize=6.6, fontweight="bold", va="top", ha="left")
    ax.text(X2 + 0.2, 10.2, "FC  shared measurement contexts\nSE  successive elimination\nCS  coefficient splitting with a\n      designed shot allocation", fontsize=6.4, va="top", ha="left", linespacing=1.3)
    fig.text(0.01, fy(T_A), "(a) The methods compared, and the median gain of each step ($\\rho=0$)", fontsize=8, fontweight="bold", va="top")
    fig.text(0.01, fy(SUB_A), "Arrow label: median over the states of (shots of the earlier method) / (shots of the later method). The range is in (b).", fontsize=7, va="top", color="#333333")

    # ---------------------------------------------------------------- (b) the range of every step
    ax = fig.add_axes([0.405, fy(AX_B_TOP + AX_B_H), 0.375, AX_B_H / H])
    rows = [
        ("ig_se", "SE only $\\to$ SE + FC", 0),
        ("full_se", "FC only $\\to$ SE + FC", 0),
        ("se_cs", "SE + FC $\\to$ SG-SE-CS", 1),
        ("se_prior", "SE + FC $\\to$ SG-SE-CS+prior", 1),
        ("se_oracle", "SE + FC $\\to$ exact covariance", 1),
        ("se_o_cs_o", "CS alone: exact cov. in both", 1),
        ("ig_cs", "SE only $\\to$ SG-SE-CS", 2),
        ("ig_prior", "SE only $\\to$ SG-SE-CS+prior", 2),
        ("ig_oracle", "SE only $\\to$ exact covariance", 2),
    ]
    groups = ["Combine\nthe baselines", "Add coefficient\nsplitting (CS) to\nSE + FC, with a\ncovariance source", "Total gain\nover SE only"]
    colors = {"H4": BLUE, "LiH": ORANGE, "BeH2": GREEN, "H2O": PINK, "CAS": GREY}
    LETTER = {"H4_a2_ADAPT2": "a", "H4_a1_ADAPT5": "b", "LiH_r16_ADAPT4": "c"}

    def fam(state):
        s = rs.split_state(state)[0]
        return "CAS" if "cas" in s else "H4" if s.startswith("H4") else "LiH" if s.startswith("LiH") else "BeH2" if s.startswith("BeH2") else "H2O"

    group_y = []
    ypos, y, last = [], 0.0, None
    for key, label, grp in rows:
        if last is not None and grp != last:
            y += 0.7
        ypos.append(y)
        y += 1.0
        last = grp
    rng = np.random.default_rng(7)
    for gi in range(3):
        ys = [yy for (kk, ll, gg), yy in zip(rows, ypos) if gg == gi]
        ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color="#f1f1f1" if gi % 2 == 0 else "white", zorder=0)
        group_y.append(np.mean(ys))
    ax.axvspan(0.55, 1.0, color="#d32f2f", alpha=0.07, lw=0, zorder=1)
    ax.axvline(1.0, color="black", lw=0.9, zorder=2)
    for (key, label, grp), y in zip(rows, ypos):
        for s, v in g[key].items():
            yy = y + rng.uniform(-0.24, 0.24)
            letter = "d" if (key == "full_se" and s == "H4_a2_ADAPT2") else LETTER.get(s)  # the same state loses for a different reason in the elimination step
            ringed = v < 1 and letter is not None
            ax.plot(v, yy, "o", ms=4.2 if ringed else 3.8, mfc=colors[fam(s)], mec=RED if ringed else "white", mew=1.1 if ringed else 0.3, zorder=5 if ringed else 3)
            if ringed:
                ax.annotate(letter, (v, yy), xytext=(0, 3.6), textcoords="offset points", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=RED, zorder=7)
        m, lo, hi = med(key)
        ax.plot([lo, hi], [y, y], color="#999999", lw=0.8, zorder=1.5)
        ax.plot([m, m], [y - 0.38, y + 0.38], color="black", lw=2.0, zorder=4)
        ax.text(1.03, y, f"{m:.2f}  ({lo:.2f}--{hi:.2f})", transform=ax.get_yaxis_transform(), va="center", fontsize=7.4)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 25)
    ax.set_xticks([0.6, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.6", "1", "2", "5", "10", "20"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.4)
    ax.set_ylim(max(ypos) + 0.6, -0.6)
    fig.canvas.draw()
    for gi, yy in enumerate(group_y):
        yfig = ax.transData.transform((1, yy))[1] / fig.bbox.height
        fig.text(0.012, yfig, groups[gi], fontsize=7.4, fontweight="bold", va="center", ha="left", color="#222222", linespacing=1.1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    trans = ax.get_xaxis_transform()
    drop = -0.125 # below the tick labels, in axes units
    ax.text(0.9, drop, "← later method needs more", transform=trans, ha="right", va="top", fontsize=6.9, color=RED, fontweight="bold")
    ax.text(1.12, drop, "needs fewer →", transform=trans, ha="left", va="top", fontsize=6.9, color="#2e7d32", fontweight="bold")
    axc = 0.405 + 0.375 / 2
    fig.text(axc, fy(AX_B_TOP + AX_B_H + 0.42), "shots of earlier method / shots of later method", ha="center", va="top", fontsize=7.5)
    ax.text(1.03, -0.62, "median  (min--max)", transform=ax.get_yaxis_transform(), fontsize=7.4, va="bottom", fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="none", color=c, ms=4.5, label=n) for n, c in (("H$_4$", BLUE), ("LiH", ORANGE), ("BeH$_2$", GREEN), ("H$_2$O", PINK), ("CAS(8e,8o)", GREY))]
    handles.append(Line2D([], [], color="black", lw=2.0, label="median"))
    handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=RED, mew=1.1, ms=4.5, label="ringed: below one, see key"))
    fig.legend(handles=handles, ncol=7, frameon=False, loc="center", bbox_to_anchor=(0.5, fy(AX_B_TOP + AX_B_H + 0.74)), fontsize=7, handletextpad=0.3, columnspacing=0.9)
    fig.text(0.01, fy(T_B), "(b) The same steps state by state ($\\rho=0$): each dot is one state, the bar the median, the line the range", fontsize=8, fontweight="bold", va="top")

    # ---------------------------------------------------------------- key: why a step can lose
    key_top = AX_B_TOP + AX_B_H + 0.93
    fig.text(0.01, fy(key_top), "Why a step can need more shots (ringed dots)", fontsize=7.8, fontweight="bold", va="top")
    key_lines = [
        ("a", "H₄ at 2.0 Å, ADAPT 2 (SE + FC → SG-SE-CS, 0.80): learning cost; with the exact covariance in both arms the step is 1.22, with the classical state 1.14."),
        ("b", "H₄ at 1.0 Å, ADAPT 5 (0.91): not the learning. The exact covariance in both arms gives 0.94, the classical state 0.84; cause not identified."),
        ("c", "LiH at 1.6 Å, ADAPT 4 (SE only → SE + FC, 0.94): a near tie (leading gap 2.6%, 3×10⁹ shots); sharing contexts saves nothing there."),
        ("d", "H₄ at 2.0 Å, ADAPT 2 (FC only → SE + FC, 0.89): SG-SE used a median of 26 checkpoints in 4 blocks, SG-FULL 9 in one; cause not isolated."),
        ("", "Other dots below one: FC only → SE + FC (0.92 to 0.99), the reference on H₄ at 2.0 Å, ADAPT 5 (0.92) and CS alone on LiH at 3.0 Å, ADAPT 3 (0.97)."),
    ]
    for i, (letter, text) in enumerate(key_lines):
        yk = key_top + 0.20 + 0.15 * i
        if letter:
            fig.text(0.012, fy(yk), letter, fontsize=7.2, fontweight="bold", color=RED, va="top")
        fig.text(0.028, fy(yk), text, fontsize=6.6, va="top")
    fig.savefig(FIG / "fig_gs_hierarchy.pdf")
    fig.savefig(FIG / "fig_gs_hierarchy.png", dpi=170)
    plt.close(fig)
    print("wrote", FIG / "fig_gs_hierarchy.pdf")


if __name__ == "__main__":
    main()
