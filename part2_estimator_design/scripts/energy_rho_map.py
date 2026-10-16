#!/usr/bin/env python3
"""What does a tolerance on the gradient (rho-good) guarantee for the energy lowering?  (reads ``energy_lowering_generators.csv``)

For every state of the benchmark (HF, CISD, every state of the exact ADAPT trajectory) and every generator the study has the
exact lowering of a one-parameter move (``dE1_mHa``) and, on the trajectory states of H4, LiH (and H2O when run with
``--full-reopt``), the lowering after re-optimising every parameter (``dEfull_mHa``).  Per state and tolerance rho the set of rho-good
generators is {i : |g_i| >= (1 - rho) max|g|}; the table gives, over the states of a system,

* ``worst``: the lowering of the worst rho-good generator divided by the best lowering of the state (mean, smallest);
* ``random``: the same for a uniformly drawn rho-good generator (mean over the set, then over the states);
* ``covered``: share of states whose energy-best generator is rho-good;
* ``rho_cover``: the smallest rho whose set contains the energy-best generator, 1 - |g_best-energy| / max|g| (median, largest) - how
  far down the gradient ranking the energy-best generator sits.

The rows of a system with several tied energy-best generators use the best-ranked one.  With ``--full`` the full re-optimisation
lowering replaces the one-parameter one (states that have it).

    python scripts/energy_rho_map.py [--runs runs/features] [--full]

Writes ``<runs>/report/rho_map.csv`` and ``rho_map_states.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RHOS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
NONZERO = 1e-8


def state_rows(system: str, key, g: pd.DataFrame, column: str) -> dict | None:
    d = g[g.abs_g > NONZERO]
    gain = -pd.to_numeric(d[column], errors="coerce")
    ok = gain.notna()
    d, gain = d[ok], gain[ok]
    if d.empty or gain.max() <= 0:
        return None
    absg = d.abs_g.to_numpy()
    gain = gain.to_numpy()
    gmax, best = absg.max(), gain.max()
    out = {"system": system, "state": key[0], "k": key[1], "gain_best_mHa": best, "n_live": len(d)}
    best_idx = np.flatnonzero(gain >= best * (1 - 1e-9))
    out["rho_cover"] = float(1.0 - absg[best_idx].max() / gmax)
    out["top_g_share"] = float(gain[absg >= gmax * (1 - 1e-9)].max() / best)
    for rho in RHOS:
        sel = absg >= (1 - rho) * gmax - 1e-12
        out[f"worst_{rho:g}"] = float(gain[sel].min() / best)
        out[f"random_{rho:g}"] = float(gain[sel].mean() / best)
        out[f"covered_{rho:g}"] = bool(sel[best_idx].any())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/features"))
    parser.add_argument("--full", action="store_true", help="use the full re-optimisation lowering where available")
    args = parser.parse_args()
    gens = pd.read_csv(args.runs / "energy_lowering_generators.csv")
    column = "dEfull_mHa" if args.full else "dE1_mHa"
    rows = []
    for (system, state, k), g in gens.groupby(["system", "state", gens.k.fillna(-1)], dropna=False, sort=False):
        r = state_rows(system, (state, k), g, column)
        if r:
            rows.append(r)
    states = pd.DataFrame(rows)
    out = args.runs / "report"
    out.mkdir(parents=True, exist_ok=True)
    states.to_csv(out / ("rho_map_states_full.csv" if args.full else "rho_map_states.csv"), index=False)
    table = []
    for system, s in states.groupby("system", sort=False):
        row = {"system": system, "states": len(s), "rho_cover_median": s.rho_cover.median(), "rho_cover_max": s.rho_cover.max(),
               "top_g_share_mean": s.top_g_share.mean(), "top_g_share_min": s.top_g_share.min()}
        for rho in RHOS:
            row[f"worst_{rho:g}_mean"] = s[f"worst_{rho:g}"].mean()
            row[f"worst_{rho:g}_min"] = s[f"worst_{rho:g}"].min()
            row[f"random_{rho:g}_mean"] = s[f"random_{rho:g}"].mean()
            row[f"covered_{rho:g}_pct"] = 100 * s[f"covered_{rho:g}"].mean()
        table.append(row)
    t = pd.DataFrame(table)
    t.to_csv(out / ("rho_map_full.csv" if args.full else "rho_map.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(f"lowering = {'full re-optimisation' if args.full else 'one-parameter move'}")
    cols = ["system", "states", "rho_cover_median", "rho_cover_max", "top_g_share_mean", "top_g_share_min"]
    print(t[cols].round(3).to_string(index=False))
    for rho in (0.05, 0.1, 0.2, 0.3):
        print(f"\nrho = {rho:g}")
        print(t[["system", f"worst_{rho:g}_mean", f"worst_{rho:g}_min", f"random_{rho:g}_mean", f"covered_{rho:g}_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
