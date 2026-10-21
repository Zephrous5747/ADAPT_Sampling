#!/usr/bin/env python3
"""Selection share on the 16-qubit cc-pVDZ window: certified II-A selection against the modelled optimisation of the step that follows.

For the state after ``k`` steps of the exact ADAPT path (k = 5, 10, 20, 30, 40) the certified selection (``step5_external_baselines`` II-A, a priori start,
sign-aware, rho 0.1, 16 trials: ``runs/<geometry>_ADAPT<k>/<geometry>_ADAPT<k>_step5_external_baselines.csv``, median and mean selection shots)
is followed by the re-optimisation after adding the (k+1)-th generator.  Its modelled cost is  C_opt = n_ev (1 + 2 n) m_E  with n = k + 1
parameters, n_ev the BFGS evaluations of that step on the exact path at the given gradient tolerance (``policy_exact.jsonl`` of a policy run
with ``--gtol``) and m_E the shots of one energy to 1 mHa at the state after the step (sorted-insertion groups of H, optimal allocation; as
``features_ci_sweep.Study.energy_shots``).  The state after the step is taken from the stored exact path (``.cache/trajectory/<case>_adapt.npz``, 41 states); for k = 40 the 40th step stands in for the 41st.

    python scripts/window_selection_share.py --policies runs/energy_policies_nrho_gtol1e-5 --out runs/window_share_gtol1e-5.csv

Writes one row per geometry and k: selection shots (median, mean), n_ev, n, m_E, C_opt, share = selection / (selection + C_opt).
Needs the window problem cache (a cluster job: 16 qubits, a few GB).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from pauli_ops import PauliEvaluator  # noqa: E402  (Part I)
from reuse import hamiltonian_terms  # noqa: E402
from trajectory import AdaptSystem  # noqa: E402

from paper_a_cost_decomposition import sorted_insertion_groups  # noqa: E402

ENERGY_ERROR = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True, help="policy-run directory with <case>/policy_exact.jsonl (the gradient tolerance of the model)")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30, 40])
    parser.add_argument("--geometries", nargs="+", default=["eq", "str"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for g in args.geometries:
        case = f"H2O_dz8o_{g}_HF"
        system = AdaptSystem.build(case)
        terms = hamiltonian_terms(system.problem, case)
        groups = sorted_insertion_groups(terms)
        traj = np.load(f".cache/trajectory/{case}_adapt.npz", allow_pickle=True)
        exact = json.loads((args.policies / case / "policy_exact.jsonl").read_text().splitlines()[0])
        for k in args.ks:
            csv_path = args.runs / f"H2O_dz8o_{g}_ADAPT{k}" / f"H2O_dz8o_{g}_ADAPT{k}_step5_external_baselines.csv"
            with csv_path.open() as handle:
                sel = next(r for r in csv.DictReader(handle) if r["config"].startswith("II-A"))
            # the exact path stores 40 steps: for k = 40 the cost of the 40th step stands in for the 41st
            idx = min(k, len(exact["evaluations"]) - 1)
            psi = np.asarray(traj["states"][min(k + 1, traj["states"].shape[0] - 1)], dtype=complex)
            evaluator = PauliEvaluator(system.full_state(psi))
            m_e = sum(evaluator.fragment_std({p: terms[p] for p in grp}) for grp in groups) ** 2 / ENERGY_ERROR ** 2
            n_ev, n = int(exact["evaluations"][idx]), int(exact["n_params"][idx])
            c_opt = n_ev * (1 + 2 * n) * m_e
            med, mean = float(sel["shots_median"]), float(sel["shots_mean"])
            rows.append({"geometry": g, "k": k, "trials": int(sel["n_trials"]), "selection_median": med, "selection_mean": mean,
                         "correct_rate": float(sel["correct_rate"]), "good_10pct_rate": float(sel["good_10pct_rate"]),
                         "n_ev": n_ev, "n_params": n, "m_E": m_e, "c_opt": c_opt, "share_median": med / (med + c_opt),
                         "share_mean": mean / (mean + c_opt), "max_gradient": float(np.abs(traj["gradients"][k]).max())})
            print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
