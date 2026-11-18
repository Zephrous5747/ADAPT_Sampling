"""Task T24 summary: every GS-1.0 result of the paper, with the data-only arm as the proposed method (decision D1 of 2026-11-16).

Record names keep the code names; the output uses the paper names:
    FC_FULL -> SG_FULL, FC_SE -> SG_SE, FC_SE_CS_DATA -> SG_SE_CS (proposed), FC_SE_CS -> SG_SE_CS+P (classical prior), FC_SE on the
    individual library -> IG_SE.
Sources (missing folders are skipped, their cells print "--"):
    runs/reframe_v2          strict and rho 0.1 matrix (T23).  FC_FULL at rho 0.1 there used the old certificate and is NOT used.
    runs/reframe_v4          FC_FULL at rho 0.1 with the dominance rule, data-only arm on the cc-pVDZ states (T24).
    runs/reframe_v2_cas_a, _b, _c   trials 5-29 of H2O_cas_eq_ADAPT10 SG_SE_CS+P at rho 0.1, merged into the cell of runs/reframe_v2.
    runs/reframe_v4_ig       IG_SE.
    runs/reframe_v4_growth   checkpoint factor 1.25.
    runs/reframe_v2_grp      qubit-wise and pivot context families (rho 0.1).
    runs/reframe_traj, reframe_traj_fix, reframe_traj_lih   measured trajectories.

   python scripts/reframe_summary.py            (all parts; writes reports/prior/t24_*.csv and the generated/ files of the paper)
   python scripts/reframe_summary.py matrix | traj | grouping | cost | growth | leaderout | leaderout_traj | rho_delta

A ratio is mean shots of the comparator over mean shots of the reference; its 95% interval resamples the trials of numerator and
denominator independently (10,000 resamples).  A ratio is flagged when either arm certifies fewer than 95% of its trials: it then compares
consumed budgets.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "reports" / "reframed" / "generator_selection_paper_filled"
GEN = PAPER / "generated"
OUT = ROOT / "reports" / "prior"
GEN.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
RESAMPLES = 10000
rng = np.random.default_rng(20261116)

SYSTEMS = {"H4_a1": r"H$_4$ 1.0~\AA", "H4_a2": r"H$_4$ 2.0~\AA", "LiH_r16": r"LiH 1.6~\AA", "LiH_r30": r"LiH 3.0~\AA", "BeH2_r133": r"BeH$_2$",
           "H2O_eq": r"H$_2$O eq.", "H2O_str": r"H$_2$O str.", "H2O_cas_eq": r"CAS eq.", "H2O_cas_str": r"CAS str."}
ARMS = ("SG_SE_CS", "SG_SE", "SG_FULL", "SG_SE_CS+P", "IG_SE")
TEX = {"SG_SE_CS": "SG-SE-CS", "SG_SE": "SG-SE", "SG_FULL": "SG-FULL", "SG_SE_CS+P": "SG-SE-CS+prior", "IG_SE": "IG-SE"}
FACTS: dict = {}


def load(folder: str) -> dict:
    out = {}
    for f in glob.glob(str(ROOT / folder / "*.jsonl")):
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if rows:
            out[os.path.basename(f)[:-6]] = rows
    return out


def split_state(state: str) -> tuple[str, str]:
    system, tag = state.rsplit("_", 1)
    return system, tag


def state_key(state: str):
    system, tag = split_state(state)
    rank = -1 if tag == "HF" else 999 if tag == "CISD" else int(tag[5:])
    return list(SYSTEMS).index(system), rank


def state_label(state: str) -> str:
    system, tag = split_state(state)
    return SYSTEMS[system] + ", " + (tag if tag in ("HF", "CISD") else "ADAPT " + tag[5:])


def group_of(state: str) -> str:
    return "cas" if "_cas_" in state else "cisd" if state.endswith("CISD") else "core"


def shots(rows) -> np.ndarray:
    return np.array([r["selection_shots"] for r in rows], float)


def certified(rows) -> int:
    return sum(r["status"] == "certified" for r in rows)


def ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    a = num[rng.integers(0, len(num), (RESAMPLES, len(num)))].mean(axis=1)
    b = den[rng.integers(0, len(den), (RESAMPLES, len(den)))].mean(axis=1)
    lo, hi = np.percentile(a / b, [2.5, 97.5])
    return float(num.mean() / den.mean()), float(lo), float(hi)


def matrix_cells() -> dict:
    """(state, setting) -> arm -> records."""
    cells: dict = defaultdict(dict)
    for key, rows in load("runs/reframe_v2").items():
        state, method, setting, _ = key.split("__")
        arm = {"FC_SE": "SG_SE", "FC_SE_CS": "SG_SE_CS+P", "FC_SE_CS_DATA": "SG_SE_CS"}.get(method)
        if method == "FC_FULL":
            arm = "SG_FULL" if setting == "strict" else "SG_FULL_oldrule"
        cells[(state, setting)][arm] = rows
    for folder in ("runs/reframe_v2_cas_a", "runs/reframe_v2_cas_b", "runs/reframe_v2_cas_c"):  # the 25 trials of H2O_cas_eq_ADAPT10 FC_SE_CS rho 0.1 that rf2_m4_cas did not finish (jobs rf2_m4_cas_res_a/b/c)
        for key, rows in load(folder).items():
            state, method, setting, _ = key.split("__")
            arm = {"FC_SE": "SG_SE", "FC_SE_CS": "SG_SE_CS+P", "FC_SE_CS_DATA": "SG_SE_CS"}[method]
            have = cells[(state, setting)].get(arm, [])
            seen = {r["trial"] for r in have}
            cells[(state, setting)][arm] = have + [r for r in rows if r["trial"] not in seen]
    for key, rows in load("runs/reframe_v4").items():
        state, method, setting, _ = key.split("__")
        if method == "FC_FULL" and all(r.get("full_dominance") for r in rows):
            cells[(state, setting)]["SG_FULL"] = rows
        elif method == "FC_SE_CS_DATA":
            cells[(state, setting)]["SG_SE_CS"] = rows
    for key, rows in load("runs/reframe_v4_ig").items():
        state, method, setting, _ = key.split("__")
        if method == "FC_SE":
            cells[(state, setting)]["IG_SE"] = rows
    return cells


def sci(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def summarize(values: list[float]) -> dict:
    v = np.array(values, float)
    return {"n": int(len(v)), "min": round(float(v.min()), 2), "median": round(float(np.median(v)), 2), "max": round(float(v.max()), 2),
            "geomean": round(float(np.exp(np.log(v).mean())), 2)} if len(v) else {"n": 0}


def load_gaps() -> dict:
    """Relative gap between the two largest gradient magnitudes per state (scripts/reframe_state_gaps.py); active-space states are not computed."""
    out = {}
    for path in [OUT / "state_gaps.csv", ROOT / "runs" / "state_gaps_cas" / "state_gaps_cas.csv"]:
        if path.exists():
            with open(path) as handle:
                out.update({r["state"]: float(r["rel_gap"]) for r in csv.DictReader(handle)})
    return out


def matrix() -> None:
    cells = matrix_cells()
    gaps = load_gaps()
    rows_csv, tex, tex_cisd = [], [], []
    stats: dict = defaultdict(lambda: defaultdict(list))  # (setting, arm) -> group -> [(state, r, lo, hi)]
    totals, totals_cisd = defaultdict(int), defaultdict(int)
    for setting in ("strict", "rho010"):
        states = sorted({s for s, t in cells if t == setting}, key=state_key)
        for state in states:
            cisd = state.endswith("CISD")  # the CISD states are reported in their own SI table and are not part of the headline statistics
            d = cells[(state, setting)]
            ref = d.get("SG_SE_CS")
            line = [state_label(state), f"{100 * gaps[state]:.1f}" if state in gaps else "--", "0" if setting == "strict" else "0.1"]
            if ref:
                line.append(f"{sci(shots(ref).mean())} ({certified(ref)}/{len(ref)})")
            else:
                line.append("--")
            for arm in ARMS:
                r = d.get(arm)
                if not r:
                    if arm != "SG_SE_CS":
                        line.append("--")
                    continue
                n, c = len(r), certified(r)
                wrong = sum(x["status"] == "certified" and not x["rho_good"] for x in r)
                miss = sum(x["block_failures"] + x["history_failures"] for x in r)
                tot = totals_cisd if cisd else totals
                tot["trials"] += n
                tot["certified"] += c
                tot["wrong"] += wrong
                tot["interval_misses"] += miss
                tot["maximizer_eliminated"] += sum(bool(x.get("maximizer_eliminated")) for x in r)
                rr = ("", "", "")
                flag = ""
                if ref and arm != "SG_SE_CS":
                    rr = ratio(shots(r), shots(ref))
                    ok = c >= 0.95 * n and certified(ref) >= 0.95 * len(ref)
                    flag = "" if ok else r"$^\dagger$"
                    line.append(f"{rr[0]:.2f}{flag} [{rr[1]:.2f}, {rr[2]:.2f}]")
                    if ok and not cisd:
                        stats[(setting, arm)][group_of(state)].append((state, rr[0], rr[1], rr[2]))
                elif arm != "SG_SE_CS":
                    line.append("--")
                rows_csv.append([state, setting, arm, n, c, wrong, miss, f"{shots(r).mean():.4g}", *(f"{v:.3f}" if v != "" else "" for v in rr),
                                 "" if flag == "" else "budget", f"{np.mean([x['wall_seconds'] for x in r]):.0f}",
                                 f"{max(x['peak_memory_bytes'] for x in r) / 1e9:.1f}"])
            (tex_cisd if cisd else tex).append(" & ".join(line) + r" \\")
        tex.append(r"\midrule")
        tex_cisd.append(r"\midrule")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "t24_matrix.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["state", "setting", "arm", "n", "certified", "wrong", "interval_misses", "mean_shots", "ratio_to_SG_SE_CS", "lo", "hi", "flag",
                    "mean_wall_s", "peak_GB"])
        w.writerows(rows_csv)
    (GEN / "gs_matrix_body.tex").write_text("\n".join(tex[:-1]) + "\n", newline="\n")
    (GEN / "gs_matrix_cisd_body.tex").write_text("\n".join(tex_cisd[:-1]) + "\n", newline="\n")
    print(f"[matrix] {dict(totals)}  (CISD states, separate table: {dict(totals_cisd)})")
    FACTS["matrix_totals"] = dict(totals)
    FACTS["matrix_totals_cisd"] = dict(totals_cisd)
    for (setting, arm), groups in sorted(stats.items()):
        everything = [x for g in groups.values() for x in g]
        entry = {"all": summarize([x[1] for x in everything])}
        for g, items in groups.items():
            entry[g] = summarize([x[1] for x in items])
        entry["below_one"] = [(s, round(r, 2)) for s, r, lo, hi in everything if hi < 1]
        entry["includes_one"] = [(s, round(r, 2)) for s, r, lo, hi in everything if lo <= 1 <= hi]
        FACTS[f"matrix.{setting}.{arm}"] = entry
        print(f"[matrix] {setting:7s} {arm:11s} " + "  ".join(f"{g}: {v}" for g, v in entry.items() if g not in ("below_one", "includes_one")))
        print(f"         below one (interval excludes 1): {entry['below_one']}   includes one: {entry['includes_one']}")
    # tolerance against strict for the same arm
    for arm in ("SG_SE_CS", "SG_SE", "SG_FULL", "SG_SE_CS+P"):
        vals = []
        for state in sorted({s for s, _ in cells if not s.endswith("CISD")}, key=state_key):
            a, b = cells.get((state, "rho010"), {}).get(arm), cells.get((state, "strict"), {}).get(arm)
            if a and b:
                vals.append((state, shots(a).mean() / shots(b).mean()))
        if vals:
            v = [x[1] for x in vals]
            FACTS[f"matrix.rho010_over_strict.{arm}"] = {"n": len(v), "median": round(float(np.median(v)), 2), "min": round(min(v), 2), "max": round(max(v), 2),
                                                          "below_0.9": [(s, round(x, 2)) for s, x in vals if x < 0.9], "above_1.1": [(s, round(x, 2)) for s, x in vals if x > 1.1]}
            print(f"[matrix] rho 0.1 / strict shots, {arm}: {FACTS[f'matrix.rho010_over_strict.{arm}']}")
    uncert = defaultdict(list)
    for (state, setting), d in sorted(cells.items(), key=lambda kv: (kv[0][1], state_key(kv[0][0]))):
        for arm, r in d.items():
            if arm in ARMS and certified(r) < 0.95 * len(r):
                uncert[f"{setting}.{arm}"].append(f"{state} {certified(r)}/{len(r)}")
    FACTS["matrix_uncertified"] = dict(uncert)
    print(f"[matrix] cells certified in fewer than 95% of trials: {dict(uncert)}")


def cost() -> None:
    cells = matrix_cells()
    tex = []
    for system in SYSTEMS:
        line = [SYSTEMS[system]]
        for arm in ("SG_SE", "SG_SE_CS", "SG_SE_CS+P"):
            wall, mem = [], []
            for (state, setting), d in cells.items():
                if split_state(state)[0] == system and setting == "strict" and arm in d and not state.endswith("CISD"):
                    wall.append(np.mean([x["wall_seconds"] for x in d[arm]]))
                    mem.append(max(x["peak_memory_bytes"] for x in d[arm]) / 1e9)
            if wall:
                span = f"{min(wall):,.0f}" if round(min(wall)) == round(max(wall)) else f"{min(wall):,.0f} to {max(wall):,.0f}"
                line += [span.replace(",", "{,}"), f"{max(mem):.1f}"]
                FACTS[f"cost.{system}.{arm}"] = {"wall_s_min": round(min(wall)), "wall_s_max": round(max(wall)), "peak_GB": round(max(mem), 1)}
            else:
                line += ["--", "--"]
        tex.append(" & ".join(line) + r" \\")
        print("[cost] " + " | ".join(line))
    (GEN / "gs_cost_body.tex").write_text("\n".join(tex) + "\n\\bottomrule\n", newline="\n")  # the rule is in the file: a rule after an input breaks booktabs


def growth() -> None:
    base, fine = matrix_cells(), defaultdict(dict)
    for key, rows in load("runs/reframe_v4_growth").items():
        state, method, setting, _ = key.split("__")
        arm = {"FC_FULL": "SG_FULL", "FC_SE": "SG_SE", "FC_SE_CS_DATA": "SG_SE_CS"}.get(method)
        if arm and not state.endswith("CISD"):  # CISD states are not part of the main study
            fine[(state, setting)][arm] = rows
    if not fine:
        print("[growth] runs/reframe_v4_growth not synced")
        return
    tex = []
    for (state, setting), d in sorted(fine.items(), key=lambda kv: (kv[0][1], state_key(kv[0][0]))):
        ref_f, ref_b = d.get("SG_SE_CS"), base.get((state, setting), {}).get("SG_SE_CS")
        parts = []
        cells = []
        for arm in ("SG_SE", "SG_FULL"):
            if arm in d and ref_f and arm in base.get((state, setting), {}) and ref_b:
                fine_ratio, coarse_ratio = shots(d[arm]).mean() / shots(ref_f).mean(), shots(base[(state, setting)][arm]).mean() / shots(ref_b).mean()
                parts.append(f"{arm} ratio factor 1.25: {fine_ratio:.2f}  factor 2: {coarse_ratio:.2f}")
                cells += [f"{coarse_ratio:.2f}", f"{fine_ratio:.2f}"]
        if ref_f and ref_b:
            parts.append(f"SG_SE_CS shots 1.25 / 2: {shots(ref_f).mean() / shots(ref_b).mean():.2f}")
            cells.append(f"{shots(ref_f).mean() / shots(ref_b).mean():.2f}")
            tex.append(" & ".join([state_label(state), "0" if setting == "strict" else "0.1", str(len(ref_f))] + cells) + r" \\")
        print(f"[growth] {state:18s} {setting:7s} " + "; ".join(parts))
    (GEN / "gs_growth_body.tex").write_text("\n".join(tex) + "\n\\bottomrule\n", newline="\n")  # the rule is in the file (booktabs)


def traj() -> None:
    recs: dict = defaultdict(dict)  # (system, setting) -> arm -> trajectories
    for folder in ("runs/reframe_traj", "runs/reframe_traj_lih", "runs/reframe_traj_fix"):
        for key, rows in load(folder).items():
            system, method, setting = key.split("__")
            arm = {"FC_SE": "SG_SE", "FC_SE_CS": "SG_SE_CS+P", "FC_SE_CS_DATA": "SG_SE_CS"}.get(method)
            if method == "FC_FULL":
                fixed = folder != "runs/reframe_traj" or setting in ("strict", "risk020") or all(r.get("full_dominance") for r in rows)
                arm = "SG_FULL" if fixed else "SG_FULL_oldrule"
            recs[(system, setting)][arm] = rows
    tex, rows_csv = [], []
    names = {"strict": "(0, 0.05)", "risk020": "(0, 0.20)", "rho005": "(0.05, 0.05)", "rho010": "(0.10, 0.05)", "rho020": "(0.20, 0.05)"}
    for system in sorted({s for s, _ in recs}):
        strict = recs.get((system, "strict"), {}).get("SG_SE")
        tie_steps = []
        if strict:
            n_steps = min(len(r["selections"]) for r in strict)
            tie_steps = [k for k in range(n_steps) if np.mean([r["selections"][k]["cap_hit"] for r in strict]) > 0.5]
            FACTS[f"traj.{system}.steps_at_cap_strict_SG_SE"] = [k + 1 for k in tie_steps]
        for setting in names:
            d = recs.get((system, setting), {})
            ref = d.get("SG_SE_CS")
            for arm in ARMS:
                r = d.get(arm)
                if not r:
                    continue
                n = len(r)
                total = np.array([x["total_shots"] for x in r], float)
                its = np.array([x["iterations"] for x in r], float)
                sel = [s for x in r for s in x["selections"]]
                cap = np.mean([s["cap_hit"] for s in sel])
                best = np.mean([s["exact_best"] for s in sel])
                good = np.mean([s["rho_good"] for s in sel])
                tie_share = np.mean([sum(x["selections"][k]["shots"] for k in tie_steps if k < len(x["selections"])) / x["total_shots"] for x in r]) if tie_steps else float("nan")
                rr = ("", "", "")
                if ref and arm != "SG_SE_CS":
                    rr = ratio(total, np.array([x["total_shots"] for x in ref], float))
                cell = f"{rr[0]:.2f} [{rr[1]:.2f}, {rr[2]:.2f}]" if rr[0] != "" else "--" if arm != "SG_SE_CS" else "1"
                tex.append(" & ".join([SYSTEMS[system], names[setting], TEX[arm], str(n), str(sum(bool(x["reached"]) for x in r)), f"{its.mean():.2f}",
                                       sci(total.mean()), cell, f"{100 * cap:.0f}", f"{100 * best:.0f}", f"{100 * tie_share:.0f}" if tie_steps else "--"]) + r" \\")
                rows_csv.append([system, setting, arm, n, sum(bool(x["reached"]) for x in r), f"{its.mean():.2f}", r[0]["exact_adapt_iterations"], f"{total.mean():.4g}",
                                 *(f"{v:.3f}" if v != "" else "" for v in rr), f"{cap:.3f}", f"{best:.3f}", f"{good:.3f}", f"{tie_share:.3f}",
                                 f"{1e3 * min(x['final_error'] for x in r):.4f}", f"{1e3 * max(x['final_error'] for x in r):.4f}"])
                FACTS[f"traj.{system}.{setting}.{arm}"] = {"n": n, "reached": sum(bool(x["reached"]) for x in r), "iterations": round(float(its.mean()), 2),
                                                            "mean_total_shots": float(f"{total.mean():.4g}"), "ratio_to_SG_SE_CS": [round(v, 2) for v in rr] if rr[0] != "" else None,
                                                            "capped_selections_pct": round(100 * cap, 1), "exact_best_pct": round(100 * best, 1), "rho_good_pct": round(100 * good, 1),
                                                            "share_of_shots_at_cap_steps_pct": None if not tie_steps else round(100 * tie_share, 1)}
                print(f"[traj] {system:10s} {setting:7s} {arm:11s} n={n} reached {sum(bool(x['reached']) for x in r)} iterations {its.mean():.2f} shots {total.mean():.3g} "
                      f"ratio {cell}  capped {100 * cap:.0f}%  exact-best {100 * best:.0f}%  rho-good {100 * good:.0f}%  shots at cap steps {100 * tie_share:.0f}%")
        tex.append(r"\midrule")
    with open(OUT / "t24_traj.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["system", "setting", "arm", "n", "reached", "mean_iterations", "exact_adapt_iterations", "mean_total_shots", "ratio_to_SG_SE_CS", "lo", "hi",
                    "capped_selection_share", "exact_best_share", "rho_good_share", "share_of_shots_at_cap_steps", "final_error_min_mHa", "final_error_max_mHa"])
        w.writerows(rows_csv)
    (GEN / "gs_traj_body.tex").write_text("\n".join(tex[:-1]) + "\n", newline="\n")


def leaderout() -> None:
    """T25: leader-out certificate (L_k >= (1 - rho) max over the other survivors of U_i) against the default Eq. (11), rho 0.1, same seeds
    (runs/reframe_v5_lo against runs/reframe_v2 `__rho010__nr`); ratios are over the same trials, with a 95% interval from resampling trials."""
    gaps = load_gaps()
    base, new_runs = load("runs/reframe_v2"), load("runs/reframe_v5_lo")
    label = {"FC_SE": "SG_SE", "FC_SE_CS_DATA": "SG_SE_CS"}
    rows, tex = [], []
    for key in sorted(new_runs, key=lambda k: (k.split("__")[1] != "FC_SE_CS_DATA", gaps.get(k.split("__")[0], 9))):
        state, method = key.split("__")[0], key.split("__")[1]
        new = {r["trial"]: r for r in new_runs[key]}
        old = {r["trial"]: r for r in base[f"{state}__{method}__rho010__nr"]}
        strict = {r["trial"]: r for r in base[f"{state}__{method}__strict__nr"]}
        trials = sorted(set(new) & set(old))
        a = np.array([new[i]["selection_shots"] for i in trials], float)
        o = np.array([old[i]["selection_shots"] for i in trials], float)
        s0 = np.array([strict[i]["selection_shots"] for i in trials if i in strict], float)
        idx = rng.integers(0, len(a), (RESAMPLES, len(a)))
        lo, hi = np.percentile(a[idx].mean(axis=1) / o[idx].mean(axis=1), [2.5, 97.5])
        certified_n = sum(new[i]["status"] == "certified" for i in trials)
        wrong = sum(new[i]["status"] == "certified" and not new[i]["rho_good"] for i in trials)
        never_more = float(np.mean(a <= o))
        rows.append((method, state, gaps.get(state, float("nan")), len(trials), a.mean() / o.mean(), lo, hi, a.mean() / s0.mean(), o.mean() / s0.mean(), certified_n, wrong, never_more))
        tex.append(" & ".join([state_label(state), f"{100 * gaps[state]:.1f}" if state in gaps else "--", TEX[label[method]], str(len(trials)), sci(o.mean()), sci(a.mean()),
                               f"{a.mean() / o.mean():.2f} [{lo:.2f}, {hi:.2f}]", f"{a.mean() / s0.mean():.2f}", f"{o.mean() / s0.mean():.2f}"]) + r" \\")
    (GEN / "gs_leaderout_body.tex").write_text(chr(10).join(tex) + chr(10) + chr(92) + "bottomrule" + chr(10), newline=chr(10))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "t25_leaderout.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["method", "state", "rel_gap", "n", "ratio_leader_out_over_default", "lo", "hi", "leader_out_over_strict", "default_over_strict", "certified", "wrong", "share_trials_never_more"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3], f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}", f"{r[8]:.3f}", r[9], r[10], f"{r[11]:.2f}"])
    for m in ("FC_SE_CS_DATA", "FC_SE"):
        v = np.array([r[4] for r in rows if r[0] == m]); vs = np.array([r[7] for r in rows if r[0] == m]); vd = np.array([r[8] for r in rows if r[0] == m])
        FACTS[f"leaderout.{label[m]}"] = {"n_states": int(len(v)), "ratio_min": round(float(v.min()), 2), "ratio_median": round(float(np.median(v)), 2), "ratio_max": round(float(v.max()), 2),
                                           "over_strict_median": round(float(np.median(vs)), 2), "default_over_strict_median": round(float(np.median(vd)), 2)}
        print(f"[leaderout] {label[m]}: {FACTS[f'leaderout.{label[m]}']}")
    FACTS["leaderout.totals"] = {"trials": int(sum(r[3] for r in rows)), "certified": int(sum(r[9] for r in rows)), "wrong": int(sum(r[10] for r in rows)),
                                  "never_more_min": round(min(r[11] for r in rows), 2)}
    print(f"[leaderout] {FACTS['leaderout.totals']}")


def leaderout_traj() -> None:
    """T25b: measured trajectories under the leader-out certificate (runs/reframe_traj_lo) against the default rule at the same setting
    (runs/reframe_traj, reframe_traj_lih, reframe_traj_fix), same seeds, paired by trial."""
    lo = load("runs/reframe_traj_lo")
    base: dict = {}
    for folder in ("runs/reframe_traj", "runs/reframe_traj_lih", "runs/reframe_traj_fix"):
        base.update(load(folder))
    label = {"FC_SE": "SG_SE", "FC_SE_CS_DATA": "SG_SE_CS"}
    names = {"rho005": "0.05", "rho010": "0.10", "rho020": "0.20"}
    tex, rows = [], []
    order = sorted(lo, key=lambda k: (k.split("__")[0], k.split("__")[1] != "FC_SE_CS_DATA", k.split("__")[2]))
    for key in order:
        system, method, setting = key.split("__")
        new = {r["trial"]: r for r in lo[key]}
        old = {r["trial"]: r for r in base[key]}
        trials = sorted(set(new) & set(old))
        a = np.array([new[i]["total_shots"] for i in trials], float)
        o = np.array([old[i]["total_shots"] for i in trials], float)
        idx = rng.integers(0, len(a), (RESAMPLES, len(a)))
        lo_ci, hi_ci = np.percentile(a[idx].mean(axis=1) / o[idx].mean(axis=1), [2.5, 97.5])
        reached = sum(bool(new[i]["reached"]) for i in trials)
        sel = [s for i in trials for s in new[i]["selections"]]
        sel_old = [s for i in trials for s in old[i]["selections"]]
        same = sum([s["selected"] for s in new[i]["selections"]] == [s["selected"] for s in old[i]["selections"]] for i in trials)
        dfinal = max(abs(new[i]["final_error"] - old[i]["final_error"]) for i in trials) * 1e3
        dtraj = [1e3 * float(np.max(np.abs(np.array(new[i]["energy_errors"]) - np.array(old[i]["energy_errors"])))) for i in trials]  # largest gap along each trajectory
        n_diff = sum(d > 1e-3 for d in dtraj)  # trials whose energy trajectory differs by more than 1 microHa somewhere
        never_more = float(np.mean(a <= o))
        rows.append((system, setting, label[method], len(trials), reached, a.mean(), o.mean(), a.mean() / o.mean(), lo_ci, hi_ci, never_more,
                     100 * np.mean([s["rho_good"] for s in sel]), 100 * np.mean([s["exact_best"] for s in sel]), 100 * np.mean([s["exact_best"] for s in sel_old]), same, dfinal, n_diff, max(dtraj)))
        tex.append(" & ".join([SYSTEMS[system], names[setting], TEX[label[method]], str(len(trials)), str(reached), sci(o.mean()), sci(a.mean()), f"{a.mean() / o.mean():.2f} [{lo_ci:.2f}, {hi_ci:.2f}]",
                               f"{100 * never_more:.0f}", f"{100 * np.mean([s['exact_best'] for s in sel_old]):.0f}", f"{100 * np.mean([s['exact_best'] for s in sel]):.0f}", str(n_diff), f"{max(dtraj):.2f}"]) + r" \\")
    (GEN / "gs_leaderout_traj_body.tex").write_text(chr(10).join(tex) + chr(10) + chr(92) + "bottomrule" + chr(10), newline=chr(10))  # the rule is in the file: a rule after an input breaks booktabs
    with open(OUT / "t25_leaderout_traj.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["system", "setting", "arm", "n", "reached", "mean_shots_leader_out", "mean_shots_default", "ratio", "lo", "hi", "share_trials_never_more",
                    "rho_good_pct", "exact_best_pct_leader_out", "exact_best_pct_default", "trials_same_sequence", "max_final_error_difference_mHa", "trials_energy_path_differs", "max_energy_path_difference_mHa"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4g}", f"{r[6]:.4g}", f"{r[7]:.3f}", f"{r[8]:.3f}", f"{r[9]:.3f}", f"{r[10]:.2f}", f"{r[11]:.1f}", f"{r[12]:.1f}", f"{r[13]:.1f}", r[14], f"{r[15]:.4f}", r[16], f"{r[17]:.3f}"])
    v = np.array([r[7] for r in rows])
    FACTS["leaderout_traj"] = {"n_series": len(rows), "trials": int(sum(r[3] for r in rows)), "reached": int(sum(r[4] for r in rows)), "ratio_min": round(float(v.min()), 2), "ratio_median": round(float(np.median(v)), 2),
                               "ratio_max": round(float(v.max()), 2), "max_final_error_difference_mHa": round(max(r[15] for r in rows), 4), "trials_energy_path_differs": int(sum(r[16] for r in rows)), "max_energy_path_difference_mHa": round(max(r[17] for r in rows), 2),
                               "by_arm": {m: [round(float(np.median([r[7] for r in rows if r[2] == m])), 2), round(float(min(r[7] for r in rows if r[2] == m)), 2), round(float(max(r[7] for r in rows if r[2] == m)), 2)] for m in ("SG_SE_CS", "SG_SE")}}
    for r in rows:
        print(f"[leaderout_traj] {r[0]:10s} {r[1]:7s} {r[2]:9s} n={r[3]} reached {r[4]} ratio {r[7]:.2f} [{r[8]:.2f}, {r[9]:.2f}] never_more {100 * r[10]:.0f}% exact_best {r[13]:.0f}->{r[12]:.0f}% same_seq {r[14]} dfinal {r[15]:.4f} path-differs {r[16]} max {r[17]:.2f}")
    print(f"[leaderout_traj] {FACTS['leaderout_traj']}")


def rho_delta() -> None:
    """T26: rho 0.1 together with delta 0.2 (runs/reframe_v6_delta) against delta 0.05 (runs/reframe_v2 `rho010__nr`, runs/reframe_v5_lo), for the default
    certificate and the leader-out variant, on the seven states of T25.  Streams differ (the setting id is in the seed), so ratios are of means with
    a 95% interval from resampling the trials of numerator and denominator independently."""
    gaps = load_gaps()
    base, lo05, new = load("runs/reframe_v2"), load("runs/reframe_v5_lo"), load("runs/reframe_v6_delta")
    label = {"FC_SE": "SG_SE", "FC_SE_CS_DATA": "SG_SE_CS"}
    rows, tex = [], []
    keys = sorted({tuple(k.split("__")[:2]) for k in new}, key=lambda k: (k[1] != "FC_SE_CS_DATA", gaps.get(k[0], 9)))
    def ratio_ci(a, b):
        ia = rng.integers(0, len(a), (RESAMPLES, len(a))); ib = rng.integers(0, len(b), (RESAMPLES, len(b)))
        lo_, hi_ = np.percentile(a[ia].mean(axis=1) / b[ib].mean(axis=1), [2.5, 97.5])
        return a.mean() / b.mean(), lo_, hi_
    for state, method in keys:
        sh = lambda rows_: np.array([r["selection_shots"] for r in rows_], float)
        d05, d02 = sh(base[f"{state}__{method}__rho010__nr"]), sh(new[f"{state}__{method}__rho010risk020__nr"])
        l05, l02 = sh(lo05[f"{state}__{method}__rho010__nr_lo"]), sh(new[f"{state}__{method}__rho010risk020__nr_lo"])
        s05 = sh(base[f"{state}__{method}__strict__nr"])
        rd, rl = ratio_ci(d02, d05), ratio_ci(l02, l05)
        cert = sum(r["status"] == "certified" for key in (f"{state}__{method}__rho010risk020__nr", f"{state}__{method}__rho010risk020__nr_lo") for r in new[key])
        wrong = sum(r["status"] == "certified" and not r["rho_good"] for key in (f"{state}__{method}__rho010risk020__nr", f"{state}__{method}__rho010risk020__nr_lo") for r in new[key])
        n = len(d02) + len(l02)
        rows.append((method, state, gaps.get(state, float("nan")), n, cert, wrong, d05.mean(), d02.mean(), *rd, l05.mean(), l02.mean(), *rl, d02.mean() / s05.mean(), l02.mean() / s05.mean()))
        tex.append(" & ".join([state_label(state), f"{100 * gaps[state]:.1f}" if state in gaps else "--", TEX[label[method]], sci(d05.mean()), sci(d02.mean()), f"{rd[0]:.2f} [{rd[1]:.2f}, {rd[2]:.2f}]",
                               sci(l05.mean()), sci(l02.mean()), f"{rl[0]:.2f} [{rl[1]:.2f}, {rl[2]:.2f}]", f"{l02.mean() / s05.mean():.2f}"]) + r" \\")
    (GEN / "gs_rhodelta_body.tex").write_text(chr(10).join(tex) + chr(10) + chr(92) + "bottomrule" + chr(10), newline=chr(10))
    with open(OUT / "t26_rho_delta.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["method", "state", "rel_gap", "trials", "certified", "wrong", "default_d005", "default_d02", "default_ratio", "default_lo", "default_hi",
                    "leaderout_d005", "leaderout_d02", "leaderout_ratio", "leaderout_lo", "leaderout_hi", "default_d02_over_strict", "leaderout_d02_over_strict"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3], r[4], r[5], *(f"{v:.4g}" for v in r[6:])])
    for m in ("FC_SE_CS_DATA", "FC_SE"):
        sel = [r for r in rows if r[0] == m]
        d = np.array([r[10 - 1] if False else r[8] for r in sel]); l = np.array([r[13] for r in sel])
        over_strict = np.array([r[16] for r in sel]); over_strict_d = np.array([r[17] for r in sel])
        FACTS[f"rho_delta.{label[m]}"] = {"n_states": len(sel), "default_delta_ratio_median": round(float(np.median(d)), 2), "default_delta_ratio_range": [round(float(d.min()), 2), round(float(d.max()), 2)],
                                          "leaderout_delta_ratio_median": round(float(np.median(l)), 2), "leaderout_delta_ratio_range": [round(float(l.min()), 2), round(float(l.max()), 2)],
                                          "default_d02_over_strict_median": round(float(np.median(over_strict)), 2), "leaderout_d02_over_strict_median": round(float(np.median(over_strict_d)), 2)}
        print(f"[rho_delta] {label[m]}: {FACTS[f'rho_delta.{label[m]}']}")
    FACTS["rho_delta.totals"] = {"trials": int(sum(r[3] for r in rows)), "certified": int(sum(r[4] for r in rows)), "wrong": int(sum(r[5] for r in rows))}
    print(f"[rho_delta] {FACTS['rho_delta.totals']}")
    for r in rows:
        print(f"[rho_delta] {r[1]:18s} {r[0]:14s} gap {100 * r[2]:5.1f}%  default d0.2/d0.05 {r[8]:.2f} [{r[9]:.2f}, {r[10]:.2f}]  leader-out {r[13]:.2f} [{r[14]:.2f}, {r[15]:.2f}]  lo(d0.2)/strict {r[17]:.2f}")


def grouping() -> None:
    base = load("runs/reframe_v2")
    fam: dict = defaultdict(dict)
    for key, rows in load("runs/reframe_v2_grp").items():
        state, method, setting, family = key.split("__")
        fam[(state, family)][method] = rows
    label = {"qwc": "qubit-wise", "pivot": "pivot, as published", "pivot_merged": "pivot, merged"}
    tex, out = [], []
    for (state, family) in sorted(fam, key=lambda k: (state_key(k[0]), list(label).index(k[1]))):
        d = fam[(state, family)]
        line = [state_label(state), label[family], f"{d[next(iter(d))][0]['n_contexts']:,}".replace(",", "{,}")]
        for method in ("FC_SE", "FC_SE_CS_DATA"):  # SG_SE and the proposed SG_SE_CS (data-only); the prior arm is not part of this table
            shared = base.get(f"{state}__{method}__rho010__nr")
            if method in d and shared:
                r = ratio(shots(d[method]), shots(shared))
                flag = "" if certified(d[method]) >= 0.95 * len(d[method]) else r"$^\dagger$"
                line.append(f"{r[0]:.2f}{flag} [{r[1]:.2f}, {r[2]:.2f}] ({len(d[method])})")
                out.append((state, family, method, r[0]))
            else:
                line.append("--")
        if "FC_SE" in d and "FC_SE_CS_DATA" in d:
            r = ratio(shots(d["FC_SE"]), shots(d["FC_SE_CS_DATA"]))
            line.append(f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]")
            out.append((state, family, "SE_over_CS", r[0]))
        else:
            line.append("--")
        tex.append(" & ".join(line) + r" \\")
        print("[grouping] " + " | ".join(line))
    (GEN / "gs_grouping_body.tex").write_text("\n".join(tex) + "\n\\bottomrule\n", newline="\n")  # the rule is in the file: a rule after an input breaks booktabs
    for family in label:
        for method in ("FC_SE", "FC_SE_CS_DATA", "SE_over_CS"):
            v = [x[3] for x in out if x[1] == family and x[2] == method]
            if v:
                FACTS[f"grouping.{family}.{method}"] = {"n": len(v), "min": round(min(v), 2), "max": round(max(v), 2)}
                print(f"[grouping] {family:13s} {method:12s} n={len(v)} {min(v):.2f} to {max(v):.2f}")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["matrix", "cost", "growth", "traj", "grouping", "leaderout", "leaderout_traj", "rho_delta"]
    for p in parts:
        {"matrix": matrix, "cost": cost, "growth": growth, "traj": traj, "grouping": grouping, "leaderout": leaderout, "leaderout_traj": leaderout_traj, "rho_delta": rho_delta}[p]()
    facts_file = GEN / "gs_facts.json"
    old = json.loads(facts_file.read_text()) if facts_file.exists() else {}
    old.update(FACTS)
    facts_file.write_text(json.dumps(old, indent=1, sort_keys=True), newline="\n")
    print("wrote", facts_file)
