#!/usr/bin/env python3
"""Paper A: the SOTA-baseline comparison tables, rebuilt from the stored trial summaries.

Fixed states
    ``runs/<case>/<case>_step4_learned_designs.csv`` (Part II's methods) and
    ``<case>_step5_external_baselines.csv`` (M1, M2, shot reuse), ``<case>_step7_ic_baseline.csv``
    (IC).  Every row is tagged with its *setting* -- ``oracle start`` (starting radius
    ``max|g|``, pairwise rule) or ``bound start`` (a-priori radius, sign-aware rule, no exact
    quantity) -- and with the ratio of its mean cost to the same setting's II-0 and II-A rows.

Trajectories
    ``runs/<case>/phase4/*_trajectories.csv`` (the original and every tagged run): median
    cumulative selection shots to chemical accuracy, relative to ``II-0 safe``.

Outputs: ``runs/paper_a/sota_fixed_state.csv`` and ``runs/paper_a/sota_trajectories.csv``; a
compact text version is printed.

    python scripts/paper_a_sota_tables.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outputs import write_csv  # noqa: E402

# (setting, method name shown, step, config label as stored)
FIXED = [
    ("oracle start", "II-0", "step4", "II-0 estimated"),
    ("oracle start", "II-A data", "step4", "II-A data"),
    ("oracle start", "M1 static (exact gap, exact variances)", "step5", "M1 static"),
    ("oracle start", "M1 seq (no elimination)", "step5", "M1 seq"),
    ("oracle start", "M2 independent, marginal rule", "step5", "M2 marginal"),
    ("oracle start", "M2 independent, pairwise rule", "step5", "M2 pairwise"),
    ("oracle start", "Ikh reuse, FC", "step5", "Ikh reuse, FC"),
    ("oracle start", "Ikh reuse, QWC", "step5", "Ikh reuse, QWC"),
    ("oracle start", "Ikh QWC, no reuse", "step5", "Ikh no reuse, QWC"),
    ("oracle start", "II-0 + reuse", "step5", "II-0 + reuse, FC"),
    ("oracle start", "II-A data + reuse", "step5", "II-A data + reuse, FC"),
    ("oracle start", "II-A data + reuse (home)", "step5", "II-A data + reuse (home), FC"),
    ("sign-aware", "II-0 safe", "step4", "II-0 safe"),
    ("sign-aware", "II-A safe", "step5", "II-A data, safe"),
    ("sign-aware", "M1 seq safe (oracle start)", "step5", "M1 seq, safe"),
    ("sign-aware", "M2 safe (oracle start)", "step5", "M2 safe"),
    ("sign-aware", "Ikh reuse safe", "step5", "Ikh reuse, FC, safe"),
    ("sign-aware", "II-0 safe + reuse", "step5", "II-0 safe + reuse, FC"),
    ("sign-aware", "II-A safe + reuse", "step5", "II-A data, safe + reuse, FC"),
    ("sign-aware", "II-A safe + reuse (home)", "step5", "II-A data, safe + reuse (home), FC"),
    ("sign-aware", "II-A safe, library, no data", "step5", "II-A data, safe, FC library, no data"),
    ("bound start", "II-0 safe", "step4", "II-0 safe, bound start"),
    ("bound start", "M1 seq safe", "step5", "M1 seq, safe, bound start"),
    ("bound start", "M2 safe", "step5", "M2 safe, bound start"),
    ("bound start", "M2 marginal", "step5", "M2 marginal, bound start"),
]
IC = [("IC oracle, anytime", "IC oracle, rule=safe, anytime"), ("IC estimated, anytime", "IC estimated, rule=safe, anytime"),
      ("IC oracle", "IC oracle, rule=safe"), ("IC estimated", "IC estimated, rule=safe")]


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["config"]: row for row in csv.DictReader(handle)}


def fixed_state(runs: Path) -> list[dict]:
    out = []
    for case_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        case = case_dir.name
        tables = {"step4": read_rows(case_dir / f"{case}_step4_learned_designs.csv"),
                  "step5": read_rows(case_dir / f"{case}_step5_external_baselines.csv")}
        if not tables["step5"]:
            continue
        # rho-tagged step-5 rows (pools) are stored under "<config>, rho=<rho>"
        for rho_suffix in sorted({k.split(", rho=")[1] for k in tables["step5"] if ", rho=" in k} | {""}):
            def find(step, label):
                rows = tables[step]
                return rows.get(label + (f", rho={rho_suffix}" if rho_suffix and step == "step5" else ""))
            reference = {"oracle start": find("step4", "II-0 estimated") or find("step5", "II-0 estimated"),
                         "bound start": find("step4", "II-0 safe, bound start"),
                         "sign-aware": find("step4", "II-0 safe")}
            ref_a = find("step4", "II-A data") or find("step5", "II-A data")
            for setting, name, step, label in FIXED:
                row = find(step, label) or (find("step5", label) if step == "step4" else None)
                if row is None:
                    continue
                mean = float(row["shots_mean"])
                ref = reference[setting]
                out.append({
                    "case": case, "rho": rho_suffix or "0", "setting": setting, "method": name,
                    "trials": int(row["n_trials"]), "shots_mean": mean, "shots_sem": float(row["shots_sem"]),
                    "shots_median": float(row["shots_median"]), "correct_rate": float(row["correct_rate"]),
                    "rho_good_rate": float(row.get("good_10pct_rate") or "nan"),
                    "vs_II0": mean / float(ref["shots_mean"]) if ref else float("nan"),
                    "vs_IIA": mean / float(ref_a["shots_mean"]) if ref_a and setting == "oracle start" else float("nan"),
                })
        ic = read_rows(case_dir / f"{case}_step7_ic_baseline.csv")
        for name, label in IC:
            for key, row in ic.items():
                if key.startswith(label):
                    out.append({"case": case, "rho": "0" if "rho=" not in key else key.split("rho=")[1],
                                "setting": "IC shots (unit differs)", "method": name + ("" if "free" not in key else " + free energy data"),
                                "trials": int(row["n_trials"]), "shots_mean": float(row["shots_mean"]),
                                "shots_sem": float(row["shots_sem"]), "shots_median": float(row["shots_median"]),
                                "correct_rate": float(row["correct_rate"]), "rho_good_rate": float(row.get("good_10pct_rate") or "nan"),
                                "vs_II0": float("nan"), "vs_IIA": float("nan")})
    return out


def trajectories(runs: Path) -> list[dict]:
    out = []
    for case_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        case = case_dir.name
        phase4 = case_dir / "phase4"
        if not phase4.exists():
            continue
        medians = {}
        for path in sorted(phase4.glob(f"{case}_phase4*_trajectories.csv")):
            rows = list(csv.DictReader(path.open()))
            for method in dict.fromkeys(r["method"] for r in rows):
                mine = [r for r in rows if r["method"] == method]
                totals = np.array([float(r["total_shots"]) for r in mine])
                medians[method] = {
                    "case": case, "method": method, "trajectories": len(mine),
                    "total_shots_median": float(np.median(totals)), "total_shots_mean": float(totals.mean()),
                    "reached_rate": float(np.mean([r["reached"] == "True" for r in mine])),
                    "iterations_mean": float(np.mean([float(r["iterations"]) for r in mine])),
                }
        reference = medians.get("II-0 safe")
        for method, row in medians.items():
            row["vs_II0_safe"] = (row["total_shots_median"] / reference["total_shots_median"]) if reference else float("nan")
            out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    fixed = fixed_state(args.runs)
    traj = trajectories(args.runs)
    if fixed:
        write_csv(args.runs / "paper_a" / "sota_fixed_state.csv", list(fixed[0]), fixed)
    if traj:
        write_csv(args.runs / "paper_a" / "sota_trajectories.csv", list(traj[0]), traj)
    case = None
    for r in fixed:
        if r["rho"] != "0" or r["case"] != case:
            case = r["case"]
            print(f"\n{r['case']}  (rho {r['rho']})")
        print(f"  {r['setting']:14s} {r['method']:40s} {r['shots_mean']:14,.0f} +- {r['shots_sem']:10,.0f}  "
              f"x{r['vs_II0']:6.2f} of II-0   correct {r['correct_rate']:.2f}")
    for r in traj:
        print(f"{r['case']:30s} {r['method']:32s} median {r['total_shots_median']:14,.0f}  x{r['vs_II0_safe']:5.2f} of II-0 safe "
              f"({r['trajectories']} trajectories, reached {r['reached_rate']:.2f})")


if __name__ == "__main__":
    main()
