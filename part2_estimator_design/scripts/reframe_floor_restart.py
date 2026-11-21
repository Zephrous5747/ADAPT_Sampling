"""Two sensitivity checks of the classical-prior arm of the paper (JCTC round 5), same states and trial ids as the matrix.

Floor (jobs rt10a-c, runs/reframe_v11_floor): a plan is rounded up to at least FLOOR shots per active context; the protocol uses 32.  Nine fixed
states, rho 0, 30 trials per cell, floors 8, 128 and 400 from the new runs and 32 from trials 0-29 of the matrix (runs/reframe_v2, same seeds).

Restart (jobs rt11a-d, runs/reframe_v12_restart): the matrix opens no new block after an elimination (--restart-ratio 0); the earlier harness opened
one after every elimination (--restart-ratio 1).  Eight cells, 30 trials, the same seeds as trials 0-29 of the matrix cell (paired).

Arms (code names): FC_SE_CS = SG-SE-CS+prior, FC_SE_CS_DATA = SG-SE-CS (measured covariance), FC_SE = SG-SE.  A ratio is the mean shots of the numerator
arm over those of the denominator arm, with the 95% interval of reframe_summary.ratio (independent resampling of trials).

    python scripts/reframe_floor_restart.py   ->  reports/prior/t29_floor.csv, t30_restart.csv, generated/gs_floor_body.tex, gs_restart_body.tex, facts
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reframe_summary as rs  # noqa: E402

GEN, OUT = rs.GEN, rs.OUT
ARM = {"FC_SE_CS": "SG_SE_CS+P", "FC_SE_CS_DATA": "SG_SE_CS", "FC_SE": "SG_SE"}
FLOORS = (8, 32, 128, 400)
FLOOR_STATES = ["H4_a1_ADAPT3", "H4_a1_ADAPT5", "H4_a2_ADAPT2", "LiH_r16_HF", "LiH_r16_ADAPT4", "LiH_r30_HF", "LiH_r30_ADAPT3", "BeH2_r133_ADAPT9", "H2O_eq_ADAPT11"]


def pick(rows, n=30):
    return [r for r in rows if r["trial"] < n]


def floor_cells() -> dict:
    cells = rs.matrix_cells()
    out: dict = {}
    for state in FLOOR_STATES:
        for arm_code, arm in ARM.items():
            out[(state, 32, arm_code)] = pick(cells[(state, "strict")][arm])
    for key, rows in rs.load("runs/reframe_v11_floor").items():
        state, method, _, tag = key.split("__")
        out[(state, int(tag[1:]), method)] = pick(rows)
    return out


def cell(rows):
    s = rs.shots(rows)
    return float(s.mean()), rs.certified(rows), len(rows)


def floor_part() -> None:
    data = floor_cells()
    facts: dict = {}
    table = []
    for state in FLOOR_STATES:
        row = {"state": state}
        for fl in FLOORS:
            d = {m: data[(state, fl, m)] for m in ARM}
            p, c, se = (rs.shots(d["FC_SE_CS"]), rs.shots(d["FC_SE_CS_DATA"]), rs.shots(d["FC_SE"]))
            r_pc = rs.ratio(p, c)
            r_sc = rs.ratio(se, c)
            cert = {m: rs.certified(d[m]) for m in ARM}
            row[fl] = {"prior_over_data": r_pc, "se_over_data": r_sc, "mean_prior": p.mean(), "mean_data": c.mean(), "mean_se": se.mean(),
                       "cert": cert, "n": len(c), "ok": all(v >= 0.95 * len(c) for v in cert.values())}  # a ratio of cells that hit the cap compares consumed budgets
        table.append(row)
    print("== floor: prior / data-only (95% interval), SG-SE / data-only")
    for row in table:
        print(f"{row['state']:20s} " + "  ".join(f"f{fl}: {row[fl]['prior_over_data'][0]:.2f} [{row[fl]['prior_over_data'][1]:.2f},{row[fl]['prior_over_data'][2]:.2f}] / {row[fl]['se_over_data'][0]:.2f}{'' if row[fl]['ok'] else ' (cap)'}" for fl in FLOORS))
    for fl in FLOORS:
        good = [row for row in table if row[fl]["ok"]]
        pc = np.array([row[fl]["prior_over_data"][0] for row in good])
        sc = np.array([row[fl]["se_over_data"][0] for row in good])
        cert = {m: sum(row[fl]["cert"][m] for row in table) for m in ARM}
        n = sum(row[fl]["n"] for row in table)
        facts[str(fl)] = {"cells_used": len(good), "cells_excluded_at_cap": [row["state"] for row in table if not row[fl]["ok"]], "prior_over_data_median": round(float(np.median(pc)), 2), "prior_over_data_min": round(float(pc.min()), 2), "prior_over_data_max": round(float(pc.max()), 2),
                          "prior_better_5pct": int((pc < 0.95).sum()), "prior_worse_5pct": int((pc > 1.05).sum()),
                          "se_over_data_median": round(float(np.median(sc)), 2), "se_over_data_min": round(float(sc.min()), 2), "se_over_data_max": round(float(sc.max()), 2),
                          "se_over_data_below1": int((sc < 1).sum()), "certified": cert, "trials_per_arm": n,
                          "prior_over_se_median": round(float(np.median([row[fl]["mean_prior"] / row[fl]["mean_se"] for row in good])), 2)}
        print(f"[floor {fl:3d}] prior/data median {np.median(pc):.2f} ({pc.min():.2f}-{pc.max():.2f}), better>5% {int((pc < .95).sum())}, worse>5% {int((pc > 1.05).sum())}; SG-SE/data median {np.median(sc):.2f} ({sc.min():.2f}-{sc.max():.2f}); certified {cert} of {n}")
    # the same states at every floor (cells that reached the cap at 128 or 400 are left out at all four floors)
    common = [row for row in table if all(row[fl]["ok"] for fl in FLOORS)]
    for fl in FLOORS:
        pc = np.array([row[fl]["prior_over_data"][0] for row in common])
        sc = np.array([row[fl]["se_over_data"][0] for row in common])
        facts[str(fl)].update({"common_n": len(common), "common_prior_over_data_median": round(float(np.median(pc)), 2), "common_prior_over_data_min": round(float(pc.min()), 2),
                               "common_prior_over_data_max": round(float(pc.max()), 2), "common_prior_better_5pct": int((pc < 0.95).sum()), "common_prior_worse_5pct": int((pc > 1.05).sum()),
                               "common_se_over_data_median": round(float(np.median(sc)), 2), "common_se_over_data_min": round(float(sc.min()), 2), "common_se_over_data_max": round(float(sc.max()), 2)})
        print(f"[floor {fl:3d}] same {len(common)} states: prior/data median {np.median(pc):.2f} ({pc.min():.2f}-{pc.max():.2f}) better {int((pc < .95).sum())} worse {int((pc > 1.05).sum())}; SG-SE/data median {np.median(sc):.2f} ({sc.min():.2f}-{sc.max():.2f})")
    # exact-style check of the U shape: geometric mean of prior/data over states
    for fl in FLOORS:
        pc = np.array([row[fl]["prior_over_data"][0] for row in table if row[fl]["ok"]])
        facts[str(fl)]["prior_over_data_geomean"] = round(float(np.exp(np.log(pc).mean())), 2)
    with open(OUT / "t29_floor.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["state", "floor", "n", "mean_prior", "mean_data", "mean_se", "prior_over_data", "lo", "hi", "se_over_data", "lo", "hi", "cert_prior", "cert_data", "cert_se"])
        for row in table:
            for fl in FLOORS:
                c = row[fl]
                w.writerow([row["state"], fl, c["n"], f"{c['mean_prior']:.5g}", f"{c['mean_data']:.5g}", f"{c['mean_se']:.5g}", *(f"{v:.3f}" for v in c["prior_over_data"]), *(f"{v:.3f}" for v in c["se_over_data"]),
                            c["cert"]["FC_SE_CS"], c["cert"]["FC_SE_CS_DATA"], c["cert"]["FC_SE"]])
    lines = []
    for row in table:
        cells = [f"{row[fl]['prior_over_data'][0]:.2f}" if row[fl]["ok"] else "--" for fl in FLOORS] + [f"{row[fl]['se_over_data'][0]:.2f}" if row[fl]["ok"] else "--" for fl in FLOORS]
        lines.append(rs.state_label(row["state"]) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    med = [f"{facts[str(fl)]['common_prior_over_data_median']:.2f}" for fl in FLOORS] + [f"{facts[str(fl)]['common_se_over_data_median']:.2f}" for fl in FLOORS]
    lines.append(f"Median over the {len(common)} states certified at every floor & " + " & ".join(med) + r" \\")
    (GEN / "gs_floor_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", newline="\n")
    rs.FACTS["floor"] = facts


RESTART_CELLS = [("H4_a2_ADAPT2", "rho010"), ("LiH_r16_ADAPT2", "rho010"), ("LiH_r30_HF", "rho010"), ("LiH_r30_ADAPT3", "rho010"), ("LiH_r30_ADAPT3", "strict"),
                 ("BeH2_r133_ADAPT9", "rho010"), ("H2O_eq_ADAPT11", "rho010"), ("H2O_str_ADAPT8", "rho010")]


def restart_part() -> None:
    cells = rs.matrix_cells()
    new = rs.load("runs/reframe_v12_restart")
    facts, table = {}, []
    print("== restart: ratios of mean shots, 30 paired trials")
    for state, setting in RESTART_CELLS:
        row = {"state": state, "setting": setting}
        for arm_code, arm in ARM.items():
            old_rows = pick(cells[(state, setting)][arm])
            new_rows = new[f"{state}__{arm_code}__{setting}__rs"]
            row[arm_code] = {"old": rs.shots(old_rows), "new": rs.shots(new_rows), "cert_old": rs.certified(old_rows), "cert_new": rs.certified(new_rows), "n": len(new_rows),
                             "blocks_new": float(np.mean([len(r.get("blocks", [])) for r in new_rows])), "blocks_old": float(np.mean([len(r.get("blocks", [])) for r in old_rows]))}
        row["prior_over_data_old"] = rs.ratio(row["FC_SE_CS"]["old"], row["FC_SE_CS_DATA"]["old"])
        row["prior_over_data_new"] = rs.ratio(row["FC_SE_CS"]["new"], row["FC_SE_CS_DATA"]["new"])
        row["se_over_data_old"] = rs.ratio(row["FC_SE"]["old"], row["FC_SE_CS_DATA"]["old"])
        row["se_over_data_new"] = rs.ratio(row["FC_SE"]["new"], row["FC_SE_CS_DATA"]["new"])
        row["data_restart_cost"] = rs.ratio(row["FC_SE_CS_DATA"]["new"], row["FC_SE_CS_DATA"]["old"])
        row["prior_restart_cost"] = rs.ratio(row["FC_SE_CS"]["new"], row["FC_SE_CS"]["old"])
        table.append(row)
        f = lambda t: f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"  # noqa: E731
        print(f"{state:18s} {setting:6s} prior/data: no restart {f(row['prior_over_data_old'])} -> restart {f(row['prior_over_data_new'])}; SG-SE/data {row['se_over_data_old'][0]:.2f} -> {row['se_over_data_new'][0]:.2f}; "
              f"restart cost data {row['data_restart_cost'][0]:.2f}x prior {row['prior_restart_cost'][0]:.2f}x; blocks data {row['FC_SE_CS_DATA']['blocks_old']:.1f}->{row['FC_SE_CS_DATA']['blocks_new']:.1f}; "
              f"cert new {row['FC_SE_CS']['cert_new']}/{row['FC_SE_CS_DATA']['cert_new']}/{row['FC_SE']['cert_new']} of {row['FC_SE']['n']}")
        facts[f"{state}.{setting}"] = {"prior_over_data_no_restart": [round(v, 2) for v in row["prior_over_data_old"]], "prior_over_data_restart": [round(v, 2) for v in row["prior_over_data_new"]],
                                       "se_over_data_no_restart": round(row["se_over_data_old"][0], 2), "se_over_data_restart": round(row["se_over_data_new"][0], 2),
                                       "data_only_restart_cost": round(row["data_restart_cost"][0], 2), "prior_restart_cost": round(row["prior_restart_cost"][0], 2)}
    po = np.array([r["prior_over_data_old"][0] for r in table])
    pn = np.array([r["prior_over_data_new"][0] for r in table])
    print(f"[restart] prior/data median {np.median(po):.2f} -> {np.median(pn):.2f}; better>5% {(po < .95).sum()} -> {(pn < .95).sum()}; worse>5% {(po > 1.05).sum()} -> {(pn > 1.05).sum()}")
    facts["summary"] = {"prior_over_data_median_no_restart": round(float(np.median(po)), 2), "prior_over_data_median_restart": round(float(np.median(pn)), 2),
                        "cells": len(table)}
    with open(OUT / "t30_restart.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["state", "setting", "n", "prior_over_data_no_restart", "lo", "hi", "prior_over_data_restart", "lo", "hi", "se_over_data_no_restart", "se_over_data_restart",
                    "data_restart_cost", "prior_restart_cost", "blocks_data_no_restart", "blocks_data_restart", "cert_prior", "cert_data", "cert_se"])
        for r in table:
            w.writerow([r["state"], r["setting"], r["FC_SE"]["n"], *(f"{v:.3f}" for v in r["prior_over_data_old"]), *(f"{v:.3f}" for v in r["prior_over_data_new"]),
                        f"{r['se_over_data_old'][0]:.3f}", f"{r['se_over_data_new'][0]:.3f}", f"{r['data_restart_cost'][0]:.3f}", f"{r['prior_restart_cost'][0]:.3f}",
                        f"{r['FC_SE_CS_DATA']['blocks_old']:.2f}", f"{r['FC_SE_CS_DATA']['blocks_new']:.2f}", r["FC_SE_CS"]["cert_new"], r["FC_SE_CS_DATA"]["cert_new"], r["FC_SE"]["cert_new"]])
    lines = []
    for r in table:
        po_, pn_ = r["prior_over_data_old"], r["prior_over_data_new"]
        lines.append(" & ".join([rs.state_label(r["state"]), r"$\rho=0$" if r["setting"] == "strict" else r"$\rho=0.1$", f"{po_[0]:.2f}", f"{pn_[0]:.2f} [{pn_[1]:.2f}, {pn_[2]:.2f}]",
                                 f"{r['data_restart_cost'][0]:.1f}", f"{r['prior_restart_cost'][0]:.1f}", f"{r['FC_SE_CS_DATA']['blocks_new']:.1f}",
                                 f"{r['se_over_data_old'][0]:.2f}", f"{r['se_over_data_new'][0]:.2f}" + (r"$^\dagger$" if r["FC_SE"]["cert_new"] < 0.95 * r["FC_SE"]["n"] else "")]) + r" \\")
    (GEN / "gs_restart_body.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", newline="\n")
    rs.FACTS["restart"] = facts


def main() -> None:
    floor_part()
    restart_part()
    path = GEN / "gs_facts.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["floor"], old["restart"] = rs.FACTS["floor"], rs.FACTS["restart"]
    path.write_text(json.dumps(old, indent=1, sort_keys=True), newline="\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
