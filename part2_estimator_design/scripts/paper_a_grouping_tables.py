#!/usr/bin/env python3
"""Grouping follow-up and radius-scale results of the cluster rounds of 2026-10-29/30: stored results to csv and LaTeX bodies.

* ``vaware``: II-A and II-0 at 400 shots, rho = 0, 100 trials, with the contexts of the covariance-aware insertion (Hartree-Fock covariances, contrast rows from the
  exact gradients; ``runs_group_vaware_hf``) over the same method with first fit by weight (``runs``), next to the sorted insertion by mass (``runs_group_mass``).
  Ratios of mean shots with 95% percentile intervals (``paper_a_bootstrap.ratio_interval``).
* ``oracle``: the oracle optimum of the shots (exact covariances, exact gradients, joint shots and splitting) for the groupings of ``paper_a_oracle_ceiling.py``,
  at rho = 0, over the optimum for the default grouping (``runs/paper_a/oracle_ceiling``).
* ``trajmass``: median selection shots along the HF trajectories at 400 shots with the mass grouping (one-line summaries of the line logs ``cluster/logs_fir/grp_traj_mass_*``,
  the phase-4 files keep only the method that finished last) against the first-fit runs of ``q2_methods400.csv``.
* ``r0``: II-A, II-0 and M2 at 400 shots with the a priori first radius multiplied by 2, 5 and 20, over the same method at the benchmark's scale 1.

Writes ``runs/paper_a/{vaware_hf,oracle_grouping,traj_mass,r0_scale}.csv`` and ``reports/generated/tab_{vaware,oracle,trajmass,r0scale}.tex``.

    python scripts/paper_a_grouping_tables.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_a_bootstrap import ratio_interval  # noqa: E402
from paper_a_round4_tables import iv, sci, state, write  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, MASS, VAWARE = ROOT / "runs", ROOT / "runs_group_mass", ROOT / "runs_group_vaware_hf"
OUT = RUNS / "paper_a"
STEMS = {"II-A": "II-A_data_safe_bound_start_min_shots=400_diagnose", "II-0": "II-0_safe_bound_start_min_shots=400_diagnose", "M2": "M2_safe_bound_start_min_shots=400"}


def trials(root: Path, case: str, stem: str) -> pd.DataFrame | None:
    f = glob.glob(str(root / case / "trials" / f"step5__{stem}__shard*.csv"))
    if not f:
        return None
    d = pd.read_csv(f[0])
    return d if len(d) == int(d.n_trials.iloc[0]) else None


def shots(d: pd.DataFrame) -> np.ndarray:
    return d.shots.to_numpy(float)


def vaware(rng) -> pd.DataFrame:
    rows = []
    for case in sorted(os.listdir(VAWARE)):
        row = {"case": case}
        for m in ("II-A", "II-0"):
            w, v, k = trials(RUNS, case, STEMS[m]), trials(VAWARE, case, STEMS[m]), trials(MASS, case, STEMS[m])
            if w is None or v is None:
                continue
            r = ratio_interval(shots(v), shots(w), "mean", rng)
            row.update({f"{m}_weight": w.shots.mean(), f"{m}_vaware": v.shots.mean(), f"{m}_ratio": r[0], f"{m}_lo": r[1], f"{m}_hi": r[2],
                        f"{m}_median_ratio": float(np.median(shots(v)) / np.median(shots(w))), f"{m}_correct": float(v.correct.mean()), f"{m}_max_over_median": float(shots(v).max() / np.median(shots(v))),
                        f"{m}_contexts_weight": float(w.contexts_used.mean()) if "contexts_used" in w else np.nan, f"{m}_contexts_vaware": float(v.contexts_used.mean()) if "contexts_used" in v else np.nan})
            if k is not None:
                row[f"{m}_mass_ratio"] = float(k.shots.mean() / w.shots.mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "vaware_hf.csv", index=False)
    rows_tex = []
    for _, r in t.iterrows():
        rows_tex.append(f"{state(r['case'])} & {iv(r['II-A_ratio'], r['II-A_lo'], r['II-A_hi'])} & {r['II-A_mass_ratio']:.2f} & {iv(r['II-0_ratio'], r['II-0_lo'], r['II-0_hi'])} & {r['II-0_mass_ratio']:.2f} & "
                        f"{r['II-A_contexts_vaware'] / r['II-A_contexts_weight']:.2f} & {100 * r['II-A_correct']:.0f}\\% \\\\")
    write("tab_vaware", ["State", "II-A cov.-aware", "II-A mass", "II-0 cov.-aware", "II-0 mass", "contexts", "II-A exact best"],
          rows_tex, "@{}lllrlrr@{}")
    return t


def oracle() -> pd.DataFrame:
    rows = []
    cases = sorted({Path(f).name.split("_rho0")[0] for f in glob.glob(str(OUT / "oracle_ceiling" / "*_rho0*.json")) if re.search(r"_rho0(_a|_b|_c|_vaware)?\.json$", f)})
    for case in cases:
        designs = {}
        for f in glob.glob(str(OUT / "oracle_ceiling" / f"{case}_rho0*.json")):
            if not re.search(r"_rho0(_a|_b|_c|_vaware)?\.json$", f):
                continue
            for rec in json.load(open(f)):
                designs.update(rec["designs"])
        if "weight" not in designs:
            continue
        w = designs["weight"]["shots"]
        row = {"case": case, "weight_shots": w, "weight_contexts": designs["weight"]["contexts"]}
        for k in ("II-0", "mass", "contrast", "family", "vaware", "vaware_hf", "weight+vaware"):
            if k in designs:
                row[f"{k}_over_weight"] = designs[k]["shots"] / w
                row[f"{k}_contexts"] = designs[k]["contexts"]
        cand = [designs[k]["shots"] / w for k in ("mass", "contrast", "family", "vaware", "vaware_hf", "weight+vaware") if k in designs]
        row["best_over_weight"] = min(cand + [1.0])
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "oracle_grouping.csv", index=False)
    f = lambda r, k: f"{r[k]:.2f}" if k in r and r[k] == r[k] else "--"  # noqa: E731
    body = [f"{state(r['case'])} & {sci(r['weight_shots'])} & {int(r['weight_contexts'])} & {f(r, 'mass_over_weight')} & {f(r, 'vaware_hf_over_weight')} & "
            f"{int(r['vaware_hf_contexts']) if r.get('vaware_hf_contexts') == r.get('vaware_hf_contexts') else '--'} & {r['best_over_weight']:.2f} \\\\" for _, r in t.iterrows()]
    write("tab_oracle", ["State", "first fit", "contexts", "mass", "cov.-aware", "contexts", "best of all"], body, "@{}lrrrrrr@{}")
    return t


LINE = re.compile(r"^(\S+)\s+(.+?) reached ([\d.]+)\s+iterations ([\d.]+) \(exact (\d+)\)\s+shots median\s+([\d,]+)\s+P90\s+([\d,]+)\s+exact-best ([\d.]+)\s+rho-good ([\d.]+)")


def trajmass() -> pd.DataFrame:
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "cluster" / "logs_fir" / "grp_traj_mass_*_line*.log"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = LINE.match(line.strip())
            if m:
                rows[(m.group(1), m.group(2).strip())] = {"case": m.group(1), "method": m.group(2).strip(), "reached": float(m.group(3)), "iterations": float(m.group(4)), "exact_iterations": int(m.group(5)),
                                                          "shots_median": float(m.group(6).replace(",", "")), "rho_good": float(m.group(9))}
    t = pd.DataFrame(list(rows.values()))
    ref = pd.read_csv(OUT / "q2_methods400.csv")
    ref["key"] = ref.method.str.split(",").str[0].str.split(" ").str[0]
    t["key"] = t.method.str.split(",").str[0].str.split(" ").str[0]
    a = t[t.key == "II-A"].set_index("case").shots_median
    t["median_over_IIA"] = [r.shots_median / a.get(r.case, np.nan) for r in t.itertuples()]
    t = t.merge(ref[["case", "key", "shots_median", "median_over_IIA"]].rename(columns={"shots_median": "shots_median_firstfit", "median_over_IIA": "over_IIA_firstfit"}), on=["case", "key"], how="left")
    t["mass_over_firstfit"] = t.shots_median / t.shots_median_firstfit
    t.to_csv(OUT / "traj_mass.csv", index=False)
    body = []
    for case in dict.fromkeys(t.case):
        s = t[t.case == case].set_index("key")
        body.append(f"{state(case)} & {sci(s.loc['II-A', 'shots_median'])} & {s.loc['II-A', 'mass_over_firstfit']:.2f} & {s.loc['II-0', 'median_over_IIA']:.2f} ({s.loc['II-0', 'over_IIA_firstfit']:.2f}) & "
                    f"{s.loc['M2', 'median_over_IIA']:.2f} ({s.loc['M2', 'over_IIA_firstfit']:.2f}) & {100 * s.rho_good.min():.1f}\\% \\\\")
    write("tab_trajmass", ["State", "II-A shots, mass", "II-A mass / first fit", "II-0 / II-A", "M2 / II-A", r"inside $\rho$-set"], body, "@{}lrrrrr@{}")
    return t


def r0(rng) -> pd.DataFrame:
    rows = []
    cases = sorted({Path(p).parent.parent.name for p in glob.glob(str(RUNS / "*" / "trials" / "step5__*_R0_x2__shard*.csv"))})
    base = {"II-A": "II-A_data_safe_bound_start_min_shots=400", "II-0": "II-0_safe_bound_start_min_shots=400", "M2": "M2_safe_bound_start_min_shots=400"}
    for case in cases:
        for m, stem in base.items():
            ref = trials(RUNS, case, STEMS[m])
            for k in (2, 5, 20):
                t = trials(RUNS, case, f"{stem}_R0_x{k}")
                if t is None or ref is None:
                    continue
                r = ratio_interval(shots(t), shots(ref), "mean", rng)
                rows.append({"case": case, "method": m, "scale": k, "ratio": r[0], "lo": r[1], "hi": r[2], "n": len(t), "exact_best": float(t.correct.mean()), "rounds_ratio": float(t.rounds.mean() / ref.rounds.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "r0_scale.csv", index=False)
    body = []
    for case in dict.fromkeys(t.case):
        for m in base:
            s = t[(t.case == case) & (t.method == m)].set_index("scale")
            if len(s) == 3:
                body.append(f"{state(case)} & {m} & " + " & ".join(iv(s.loc[k, "ratio"], s.loc[k, "lo"], s.loc[k, "hi"]) for k in (2, 5, 20)) + f" & {s.loc[20, 'rounds_ratio']:.2f} \\\\")
    write("tab_r0scale", ["State", "method", r"$R_0\times2$", r"$R_0\times5$", r"$R_0\times20$", r"rounds, $\times20$"], body, "@{}llllll@{}")
    return t


def main() -> None:
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for name, fn in (("vaware", lambda: vaware(rng)), ("oracle", oracle), ("trajmass", trajmass), ("r0", lambda: r0(rng))):
        print(f"\n=== {name}")
        print(fn().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
