#!/usr/bin/env python3
"""Tables and figures of the GS-1.0 results (the protocol of the paper), written next to the earlier-implementation ones.

Run scripts/reframe_summary.py first (it writes reports/prior/t24_matrix.csv and the generated/gs_*_body.tex files).

    python scripts/reframe_paper_results.py        (PAPER_DIR selects the folder that receives generated/ and figures/)

Writes
    generated/lowering_s1_table.tex   R4 with the optimizer of SI S1 (BFGS, gradient tolerance 1e-5), runs/features_s1
    figures/fig_lowering_s1.pdf       R4 scatter plots, same runs
    generated/gs_dose_body.tex        injected selection errors in exact-gradient ADAPT (runs/energy_policies*/report/policies_all.csv)
    figures/fig_dose.pdf              copy of the figure of those runs (drawn by part2_estimator_design/scripts/paper_a_trajectory_results.py)
    figures/fig_gs_fixed.pdf          R1: shots of each comparator over the proposed method, per state, strict
    figures/fig_gs_traj.pdf           R3: selection shots per step of the measured BeH2 trajectories
    generated/gs_results_facts.json   the numbers quoted in the text
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PART2 = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("PAPER_DIR", PART2.parent / "reports" / "reframed" / "generator_selection_paper_filled"))  # receives generated/ and figures/
RUNS = PART2 / "runs"
GEN, FIG = HERE / "generated", HERE / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
facts: dict = {}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"
SYSTEMS = ["H4 1.0", "H4 2.0", "LiH 3.0", "H2O eq", "H2O str"]
TEXNAME = {"H4 1.0": r"H$_4$ 1.0~\AA", "H4 2.0": r"H$_4$ 2.0~\AA", "LiH 3.0": r"LiH 3.0~\AA", "BeH2": r"BeH$_2$", "H2O eq": r"H$_2$O eq.", "H2O str": r"H$_2$O str.",
           "H2O dz8o eq": "CAS eq.", "H2O dz8o str": "CAS str."}

# ------------------------------------------------------------------ R4: gradient against full relaxation, optimizer of SI S1
states = pd.read_csv(RUNS / "features_s1" / "energy_lowering_states.csv")
gens = pd.read_csv(RUNS / "features_s1" / "energy_lowering_generators.csv")
states = states[states.spearman_full.notna() & states.state.isin(["HF", "ADAPT"])].copy()
states["k0"] = states.k.fillna(0)
SPREAD_MIN = 1e-6  # mHa: below this every reoptimized candidate reaches the same energy and the ranking is numerical noise


def candidates(row) -> pd.DataFrame:
    q = gens[(gens.system == row.system) & (gens.state == row.state)]
    if row.state == "ADAPT":
        q = q[q.k == row.k]
    return q[q.dEfull_mHa.notna()]


def spread(row) -> float:
    v = -candidates(row).dEfull_mHa.to_numpy()
    return float(v.max() - v.min())


states["spread"] = [spread(r) for r in states.itertuples()]
# A state is excluded when, in the tightly converged runs (gradient tolerance 1e-9, runs/features), every reoptimized candidate reaches the
# same energy: its ranking is then set by where the looser optimizer stops, not by the candidates (last ADAPT state of each H4 path).
_tight = pd.read_csv(RUNS / "features" / "energy_lowering_generators.csv")
_tight = _tight[_tight.dEfull_mHa.notna()].groupby(["system", "state", _tight.k.fillna(-1)]).dEfull_mHa.agg(lambda v: v.max() - v.min())
states["spread"] = [min(r.spread, _tight.get((r.system, r.state, -1 if pd.isna(r.k) else r.k), r.spread)) for r in states.itertuples()]
lines, low = [], []
for system in SYSTEMS:
    s = states[states.system == system].sort_values("k0")
    n_const = int((s.spread < SPREAD_MIN).sum())
    s = s[s.spread >= SPREAD_MIN]
    thirds = np.array_split(np.arange(len(s)), 3)
    med = [float(s.spearman_full.iloc[ix].median()) for ix in thirds]
    agree = s.best_g_is_best_dE_full.astype(str).eq("True").mean()
    inside = s.best_dE_in_rho_good_full.astype(str).eq("True").mean()
    n_cand = [len(candidates(r)) for r in s.itertuples()]
    sampled = s.spearman_sampled.dropna() if "spearman_sampled" in s else pd.Series(dtype=float)
    diff = (s.spearman_sampled - s.spearman_full).dropna() if "spearman_sampled" in s else pd.Series(dtype=float)
    lines.append(f"{TEXNAME[system]} & {len(s)}{f' (+{n_const})' if n_const else ''} & {min(n_cand)} to {max(n_cand)} & {s.spearman_full.median():.2f} & {s.spearman_full.min():.2f} & "
                 f"{s.spearman_full.max():.2f} & {med[0]:.2f} / {med[1]:.2f} / {med[2]:.2f} & {100 * agree:.0f}\\% & {100 * inside:.0f}\\% & "
                 f"{diff.median():+.2f} [{diff.min():+.2f}, {diff.max():+.2f}] \\\\")
    low.append({"system": system, "states": int(len(s)), "states_constant": n_const, "candidates": [int(min(n_cand)), int(max(n_cand))],
                "median": round(float(s.spearman_full.median()), 2), "min": round(float(s.spearman_full.min()), 2), "max": round(float(s.spearman_full.max()), 2),
                "thirds": [round(m, 2) for m in med], "top_pct": round(100 * float(agree)), "rho_set_pct": round(100 * float(inside)),
                "sampled_minus_full": [round(float(diff.median()), 2), round(float(diff.min()), 2), round(float(diff.max()), 2)], "last_k": int(s.k0.max())})
(GEN / "lowering_s1_table.tex").write_text(
    "\\begin{tabular}{@{}lrrrrrlrrl@{}}\\toprule\n"
    "System & States & Candidates & Median & Min & Max & Early / middle / late & Top & $\\rho$-set & Sampled $-$ full\\\\\\midrule\n" + "\n".join(lines)
    + "\n\\bottomrule\\end{tabular}\n", newline="\n")
facts["lowering_s1"] = low
for row in low:
    print("[R4]", row)

kept = states[states.spread >= SPREAD_MIN]
fig, axes = plt.subplots(len(SYSTEMS), 3, figsize=(6.6, 8.4))
for i, system in enumerate(SYSTEMS):
    s = kept[kept.system == system].sort_values("k0")
    for j, (tag, row) in enumerate(zip(("early", "middle", "late"), [s.iloc[0], s.iloc[len(s) // 2], s.iloc[-1]])):
        ax = axes[i, j]
        g = candidates(row)
        ax.scatter(g.abs_g, -g.dEfull_mHa, s=7, color=GREY, alpha=0.7, lw=0)
        top_g = g.loc[g.abs_g.idxmax()]
        ax.scatter([top_g.abs_g], [-top_g.dEfull_mHa], s=40, facecolor="none", edgecolor=RED, lw=1.2, label="largest |g|")
        top_e = g.loc[g.dEfull_mHa.idxmin()]
        ax.scatter([top_e.abs_g], [-top_e.dEfull_mHa], s=45, marker="*", color=BLUE, label="largest lowering")
        step = "HF" if row.state == "HF" else f"step {int(row.k)}"
        ax.ticklabel_format(useOffset=False)
        ax.set_title(f"{system}, {tag} ({step}), $\\rho_S$={row.spearman_full:.2f}", fontsize=7)
        if j == 0:
            ax.set_ylabel("lowering after full\nreoptimization (mHa)")
        if i == len(SYSTEMS) - 1:
            ax.set_xlabel("|g|")
        if i == 0 and j == 0:
            ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "fig_lowering_s1.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ R3: injected selection errors (exact gradients, exact optimization)
GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
pol = {tol: pd.read_csv(RUNS / folder / "report" / "policies_all.csv") for tol, folder in (("1e-9", "energy_policies"), ("1e-5", "energy_policies_gtol1e-5_cl"))}
lines, dose = [], {}
for system in ["H4 1.0", "H4 2.0", "LiH 3.0", "BeH2", "H2O eq", "H2O str"]:
    entry = {}
    for tol, t in pol.items():
        s = t[t.system == system]
        ratios = [float(s[s.policy == f"rho_random:{x:g}"].steps_ratio.iloc[0]) for x in GRID]
        reached = [float(s[s.policy == f"rho_random:{x:g}"].reached_pct.iloc[0]) for x in GRID]
        star, good = None, True
        for x, v in zip(GRID[:-1], ratios[:-1]):
            if v <= 1.03 and good:
                star = x
            else:
                good = False
        entry[tol] = {"exact_iterations": int(s[s.policy == "exact"].exact_steps.iloc[0]), "trajectories": int(s[s.policy == "rho_random:0.1"].n.iloc[0]),
                      "ratio": [round(v, 2) for v in ratios], "reached_pct": [round(v) for v in reached], "rho_star": star}
    dose[system] = entry
    a, b = entry["1e-5"], entry["1e-9"]
    lines.append(f"{TEXNAME[system]} & {a['exact_iterations']} & " + " & ".join(f"{v:.2f}" for v in a["ratio"]) + f" & {a['rho_star']:g} & {b['rho_star']:g} \\\\")
    print("[dose]", system, entry)
(GEN / "gs_dose_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", newline="\n")
t9 = pol["1e-9"]
for system in ("H2O dz8o eq", "H2O dz8o str"):
    s = t9[t9.system == system]
    dose[system] = {f"rho_{x:g}": {"error_ratio_after_40": round(float(s[s.policy == f"rho_random:{x:g}"].err_ratio_exact_steps.iloc[0]), 3),
                                   "draws": int(s[s.policy == f"rho_random:{x:g}"].n.iloc[0])} for x in (0.1, 0.3, 0.5)}
    dose[system]["exact_error_after_40_mHa"] = round(float(s[s.policy == "exact"].err_exact_steps_mHa.iloc[0]), 2)
    print("[dose]", system, dose[system])
facts["injected_errors"] = dose
shutil.copyfile(PART2.parent / "reports" / "figures" / "fig_dose.pdf", FIG / "fig_dose.pdf")

# ------------------------------------------------------------------ R1: fixed states, GS-1.0
m = pd.read_csv(PART2 / "reports" / "prior" / "t24_matrix.csv")
m = m[m.setting == "strict"]
order = list(dict.fromkeys(m.state))
SHORT = {"H4_a1": "H4 1.0", "H4_a2": "H4 2.0", "LiH_r16": "LiH 1.6", "LiH_r30": "LiH 3.0", "BeH2_r133": "BeH2", "H2O_eq": "H2O eq", "H2O_str": "H2O str",
         "H2O_cas_eq": "CAS eq", "H2O_cas_str": "CAS str"}


def short(state: str) -> str:
    system, tag = state.rsplit("_", 1)
    return SHORT[system] + " " + (tag if tag in ("HF", "CISD") else "A" + tag[5:])


core = [s for s in order if not s.endswith("CISD") and "_cas_" not in s]
cas = [s for s in order if "_cas_" in s and not s.endswith("CISD")]  # the CISD states are in their own SI table, not in the figure
order = core + cas
x = np.arange(len(order), dtype=float)
x[len(core):] += 1.0
fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.axhline(1.0, color="black", lw=0.8)
for arm, color, marker, off, name in (("SG_SE", BLUE, "o", -0.3, "SG-SE"), ("SG_FULL", GREEN, "^", -0.1, "SG-FULL"), ("IG_SE", RED, "D", 0.1, "IG-SE"),
                                      ("SG_SE_CS+P", ORANGE, "s", 0.3, "SG-SE-CS+prior")):
    d = m[(m.arm == arm) & m.ratio_to_SG_SE_CS.notna()].set_index("state")
    for filled in (True, False):
        pts = [(xi, d.loc[s]) for xi, s in zip(x, order) if s in d.index and (pd.isna(d.loc[s].flag) == filled)]
        if not pts:
            continue
        xs = np.array([p[0] for p in pts]) + off
        r = np.array([p[1].ratio_to_SG_SE_CS for p in pts])
        lo = np.array([p[1].lo for p in pts])
        hi = np.array([p[1].hi for p in pts])
        ax.errorbar(xs, r, yerr=[r - lo, hi - r], fmt=marker, ms=4.5, color=color, ecolor=color, elinewidth=0.8, capsize=1.5,
                    mfc=color if filled else "white", label=name if filled else None)
ax.set_yscale("log")
ax.set_ylabel("shots of method / shots of SG-SE-CS")
for a, b, text in ((len(core) + 0.4, len(order) + 0.6, "cc-pVDZ CAS"),):
    ax.axvspan(a, b, color="#eeeeee", zorder=0)
    ax.text(0.5 * (a + b), 0.97, text, ha="center", va="top", color=GREY, fontsize=7, transform=ax.get_xaxis_transform())
ax.legend(ncol=4, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))
ax.set_xticks(x)
ax.set_xticklabels([short(s) for s in order], rotation=55, ha="right", fontsize=7)
fig.savefig(FIG / "fig_gs_fixed.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ R3: measured trajectories, shots per step and totals
traj = {}
for f in sorted((RUNS / "reframe_traj").glob("BeH2_r133__*.jsonl")):
    _, method, setting = f.stem.split("__")
    traj[(method, setting)] = [json.loads(line) for line in open(f) if line.strip()]
panels = [("FC_SE", "SG-SE"), ("FC_SE_CS_DATA", "SG-SE-CS")]
assert all(("FC_SE_CS_DATA", st) in traj for st in ("strict", "rho005", "rho010", "rho020")), "data-only BeH2 trajectories are missing"
VIRIDIS = plt.get_cmap("viridis")
RHO = (("strict", VIRIDIS(0.0), r"$\rho=0$"), ("rho005", VIRIDIS(0.35), r"$\rho=0.05$"), ("rho010", VIRIDIS(0.62), r"$\rho=0.1$"), ("rho020", VIRIDIS(0.88), r"$\rho=0.2$"))
fig = plt.figure(figsize=(7.1, 2.7))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.4)
axes = [fig.add_subplot(grid[0, 0])]
axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
for ax, (method, name) in zip(axes, panels):
    for it in (2, 6, 9):  # iterations in which two symmetry-equivalent generators have equal gradients
        ax.axvspan(it - 0.35, it + 0.35, color="#eeeeee", zorder=0)
    for setting, color, label in RHO:
        per = np.array([[sel["shots"] for sel in t["selections"]] for t in traj[(method, setting)]], float).mean(axis=0)
        ax.plot(np.arange(1, len(per) + 1), per, marker="o", ms=3.2, lw=1.0, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration")
    ax.set_title(name, loc="left")
    ax.set_xticks(range(1, 11))
axes[0].set_ylim(3e7, 6e11)
axes[0].set_ylabel("mean selection shots")
axes[1].tick_params(labelleft=False)
axes[0].legend(frameon=False, ncol=2, loc="upper center", fontsize=6.2, columnspacing=0.8, handlelength=1.2)
axes[0].text(2, 0.03, "tied", ha="center", va="bottom", color=GREY, fontsize=6, transform=axes[0].get_xaxis_transform())
ax = fig.add_subplot(grid[0, 2])
tj = pd.read_csv(PART2 / "reports" / "prior" / "t24_traj.csv")
slots = [("BeH2_r133", st) for st in ("strict", "rho005", "rho010", "rho020")] + [("LiH_r16", st) for st in ("strict", "rho010")] + [("LiH_r30", st) for st in ("strict", "rho010")]
xs = np.array([0, 1.4, 2.8, 4.2, 6.0, 7.4, 9.2, 10.6])
RHOTXT = {"strict": "0", "rho005": "0.05", "rho010": "0.1", "rho020": "0.2"}
for arm, color, marker, off, name in (("SG_SE_CS", "black", "o", -0.2, "SG-SE-CS"), ("SG_SE", BLUE, "o", -0.07, "SG-SE"), ("SG_FULL", GREEN, "^", 0.07, "SG-FULL"),
                                      ("SG_SE_CS+P", ORANGE, "s", 0.2, "SG-SE-CS+prior")):
    first = True
    for system in ("BeH2_r133", "LiH_r16", "LiH_r30"):  # one thin line per arm and molecule through the tolerances
        pts = []
        for xi, (sys_, setting) in zip(xs, slots):
            row = tj[(tj.system == system) & (tj.setting == setting) & (tj.arm == arm)] if sys_ == system else tj.iloc[0:0]
            if len(row):
                pts.append((xi + off, float(row.mean_total_shots.iloc[0])))
        if pts:
            ax.plot(*zip(*pts), linestyle="-", lw=0.5, alpha=0.6, marker=marker, ms=3.0, color=color, label=name if first else None)
            first = False
ax.set_yscale("log")
ax.set_ylim(3e8, 6e11)
ax.set_xticks(xs)
ax.set_xticklabels([RHOTXT[st] for _, st in slots], fontsize=6.5)
for xc, text in ((2.1, r"BeH$_2$"), (6.7, "LiH 1.6"), (9.9, "LiH 3.0")):
    ax.text(xc, -0.2, text, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
ax.set_xlabel(r"tolerance $\rho$", labelpad=25)  # below the group labels at -0.2 (axes fraction)
ax.set_ylabel("total selection shots")
ax.set_title("trajectory totals", loc="left")
ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.0)
fig.savefig(FIG / "fig_gs_traj.pdf", bbox_inches="tight")
plt.close(fig)
facts["traj_figure_arms"] = [p[1] for p in panels] + ["totals: SG-SE-CS, SG-SE, SG-FULL, SG-SE-CS+prior"]

# ------------------------------------------------------------------ reproduction of the published qubit-wise elimination (separate implementation)
hi = {}
for variant in ("noreuse", "reuse"):
    for row in pd.read_csv(RUNS / "paper_a" / f"huang_izmaylov_reproduction_{variant}.csv").itertuples():
        hi.setdefault(row.case, {"published": row.published_reduction_pct})[variant] = (row.reduction_pct_mean, row.reduction_pct_sd, row.seeds,
                                                                                         row.naive_iterations, row.se_iterations, row.exact_adapt_iterations)
NAMES = {"H4_chain1p0_HF": r"H$_4$ chain", "LiH_R1p0_HF": "LiH", "BeH2_R1p0_HF": r"BeH$_2$"}
rows = []
for case, d in hi.items():
    rows.append(f"{NAMES[case]} & {d['published']:.1f} & {d['reuse'][0]:.1f} $\\pm$ {d['reuse'][1]:.1f} & {d['noreuse'][0]:.1f} $\\pm$ {d['noreuse'][1]:.1f} & "
                f"{d['reuse'][3]:.1f} / {d['reuse'][4]:.1f} \\\\")
(GEN / "gs_hi_body.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", newline="\n")
facts["hi_reproduction"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in hi.items()}

# ------------------------------------------------------------------ 40-generator horizon on cc-pVDZ CAS(8e,8o), one trajectory per method (draws what has finished)
horizon = {}
for folder in ("reframe_traj_cas", "reframe_traj_cas_fix"):
    for f in sorted((RUNS / folder).glob("H2O_cas_str__*__rho010.jsonl")):
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if rows:
            horizon[f.stem.split("__")[1]] = rows[0]
if horizon:
    NAMEMAP = {"FC_SE_CS_DATA": ("SG-SE-CS", "black", "-"), "FC_SE": ("SG-SE", BLUE, "-"), "FC_FULL": ("SG-FULL", GREEN, "-"), "FC_SE_CS": ("SG-SE-CS+prior", ORANGE, "-")}
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5))
    exact = next(iter(horizon.values()))["exact_energy_errors"]
    axes[0].plot(range(len(exact)), np.array(exact) * 1e3, color=GREY, lw=1.0, ls="--", label="exact-gradient ADAPT")
    for method, (name, color, ls) in NAMEMAP.items():
        if method not in horizon:
            continue
        t = horizon[method]
        err = np.array(t["energy_errors"]) * 1e3
        axes[0].plot(range(len(err)), err, color=color, lw=1.0, ls=ls, label=name)
        cum = np.concatenate([[np.nan], [s["cumulative_shots"] for s in t["selections"]]])
        axes[1].plot(range(len(cum)), cum, color=color, lw=1.0, ls=ls, label=name)
        facts.setdefault("horizon", {})[name] = {"iterations": t["iterations"], "total_shots": t["total_shots"], "final_error_mHa": round(float(err[-1]), 2),
                                                 "not_exact_best": int(sum(not s["exact_best"] for s in t["selections"])),
                                                 "max_shortfall": round(max(s["shortfall"] for s in t["selections"]), 4), "capped": int(sum(s["cap_hit"] for s in t["selections"]))}
    axes[0].set_yscale("log")
    axes[0].set_xlabel("ADAPT iteration")
    axes[0].set_ylabel("energy error (mHa)")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("ADAPT iteration")
    axes[1].set_ylabel("cumulative selection shots")
    fig.tight_layout()
    fig.savefig(FIG / "fig_gs_horizon.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[horizon] drawn for", sorted(horizon))

(GEN / "gs_results_facts.json").write_text(json.dumps(facts, indent=1), newline="\n")
print("wrote tables, figures and", GEN / "gs_results_facts.json")
