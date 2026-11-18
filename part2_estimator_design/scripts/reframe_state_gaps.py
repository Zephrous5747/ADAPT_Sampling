"""Stage-1 review items M6, M7 (2026-11-17): the relative gap between the two largest gradient magnitudes of every fixed state, the share of
Pauli products that are read from several contexts, and the diagonalizing-circuit size, from the same builds as the matrix runs.

    wsl ... python scripts/reframe_state_gaps.py [--workers 4] [states ...]        (writes reports/prior/state_gaps.csv)

gap = (|g_1| - |g_2|) / |g_1| over the candidates with a non-vanishing gradient (the arms of the selection); ties give 0.
The active-space states are built only when named on the command line (30-48 GB each; run them on a cluster node with --out).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import part1_bridge  # noqa: E402,F401
import reframe_run  # noqa: E402


def one(state_id: str) -> dict:
    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    case = reframe_run.case_of(state_id, protocol)
    selector, _, _ = reframe_run.build(case, 1e10, need_prior=False)
    cand = list(selector.candidates)
    g = np.sort(np.abs(selector.truth[cand]))[::-1]
    lib = selector.library
    mult = lib.multiplicity()
    needed = mult[: lib.n_required]
    two = lib.two_qubit_counts()
    return {"state": state_id, "case": case, "candidates": len(cand), "g1": float(g[0]), "g2": float(g[1]) if len(g) > 1 else 0.0,
            "rel_gap": float(1.0 - g[1] / g[0]) if len(g) > 1 else 1.0, "contexts": lib.n_contexts, "products": int(lib.n_required),
            "mean_contexts_per_product": float(needed.mean()), "share_in_several": float((needed > 1).mean()), "mean_two_qubit": float(two.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("states", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="default: reports/prior/state_gaps.csv")
    args = parser.parse_args()
    protocol = json.loads(reframe_run.PROTOCOL.read_text())
    states = args.states or [s["id"] for s in protocol["fixed_states"] if "_cas_" not in s["id"] and "dz8o" not in s["id"] and "cas" not in s["id"].lower()]
    print(len(states), "states", flush=True)
    out = args.out or ROOT / "reports" / "prior" / "state_gaps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(one, states):
            rows.append(row)
            print(row["state"], "gap %.3f" % row["rel_gap"], "contexts/product %.2f" % row["mean_contexts_per_product"], "CZ %.1f" % row["mean_two_qubit"], flush=True)
    rows.sort(key=lambda r: r["state"])
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
