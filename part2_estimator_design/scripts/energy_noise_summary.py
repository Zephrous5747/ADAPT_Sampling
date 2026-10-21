#!/usr/bin/env python3
"""Summary of the shot-noise optimiser runs against the noise-free runs at the same BFGS tolerance.

Reads ``runs/energy_policies_noisy_eps<e>_gtol<g>_cl/<case>/policy_{exact,rho_random_0.1}.jsonl`` (100 noise streams per policy, see
``energy_trajectory_policies.py --noise-eps``) and, as the noise-free reference, the tie-averaged exact ADAPT of ``runs/energy_policies_gtol1e-5_cl``
(``rho_random:0``).  Writes ``runs/energy_policies_noisy_summary.csv``: per system, noise level eps, tolerance gtol and policy the share that reaches the
target, the mean steps (and the noise-free steps), the mean evaluations and their ratio to the noise-free ones, the ratio of the weighted evaluations
sum_k ev_k (1 + 2 n_k) and the same ratio times (1 mHa / eps)^2 (the shots of an evaluation grow with the inverse square of the energy error).

    python scripts/energy_noise_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs"
SYS = {"H4_square_eq_side1p0_HF": "H4 1.0", "H4_square_stretch_side2p0_HF": "H4 2.0", "LiH_R3p0_HF": "LiH 3.0", "BeH2_HF": "BeH2",
       "H2O_eq_HF": "H2O eq", "H2O_stretch_HF": "H2O str"}
CONFIGS = ("eps1e-3_gtol1e-5", "eps1e-3_gtol3e-3", "eps1e-4_gtol1e-5", "eps1e-4_gtol3e-4")


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    out = []
    for case, name in SYS.items():
        free = rows(RUNS / "energy_policies_gtol1e-5_cl" / case / "policy_rho_random_0.jsonl")
        f_ev, f_w, f_steps = np.mean([r["evals_total"] for r in free]), np.mean([r["evals_weighted"] for r in free]), np.mean([r["iterations"] for r in free])
        ref = json.loads((RUNS / "energy_policies_gtol1e-5_cl" / case / "exact_reference.json").read_text())
        for tag in CONFIGS:
            eps = float(tag.split("_")[0][3:])
            for pol in ("exact", "rho_random_0.1"):
                rs = rows(RUNS / f"energy_policies_noisy_{tag}_cl" / case / f"policy_{pol}.jsonl")
                ev, w = np.array([r["evals_total"] for r in rs], float), np.array([r["evals_weighted"] for r in rs], float)
                out.append({"system": name, "eps": eps, "gtol": tag.split("_")[1][4:], "policy": pol, "n": len(rs),
                            "reached_pct": 100 * np.mean([r["reached"] for r in rs]), "steps": np.mean([r["iterations"] for r in rs]), "steps_free": f_steps,
                            "evals": ev.mean(), "evals_ratio": ev.mean() / f_ev, "weighted_ratio": w.mean() / f_w,
                            "shots_factor": (w.mean() / f_w) * (1e-3 / eps) ** 2,
                            "final_err_med_mHa": 1e3 * np.median([r["final_error"] for r in rs]), "ref_final_mHa": 1e3 * ref["final_error"],
                            "stalled_pct": 100 * np.mean([r["stalled"] for r in rs])})
    t = pd.DataFrame(out)
    t.to_csv(RUNS / "energy_policies_noisy_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
