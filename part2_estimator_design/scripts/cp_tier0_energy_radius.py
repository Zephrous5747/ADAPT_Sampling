"""Tier 0 check of the energy trust rule with a small energy radius r_E, without new sampling.

For each LiH state the rule (d0, s of the leave-one-system-out fit of the exact-energy calibration, runs/cp_v3x/calibration.json) is applied to
x_plus = (|E_c - E_q| + r_E) / (|E_q| - r_E) with the exact E_q (estimate = truth, noise ignored) for several radii; the trust it assigns is
turned into an expected Test A (A1) shot ratio R by log-linear interpolation of the constant-trust sweep (runs/cp_v2: trust 1 (data-only), 8, 32,
128, 512, static).  r_E = 0 reproduces the exact-energy variant, which is compared with its measured result (runs/cp_v3x) as a check of the lookup.
    python scripts/cp_tier0_energy_radius.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cp_report_tables as c  # noqa: E402  (testa_rows, testa_pct, TESTA)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAL = json.load(open(os.path.join(ROOT, "runs", "cp_v3x", "calibration.json")))
STATES = [("LiH_r16_m2", "LiH_R1p6_ADAPT2"), ("LiH_r30_m3", "LiH_R3p0_ADAPT3"), ("LiH_r30_m5", "LiH_R3p0_ADAPT5"), ("LiH_r30_m6", "LiH_R3p0_ADAPT6"),
          ("LiH_r30_m3_K2", "LiH_R3p0_ADAPT3"), ("LiH_r30_m5_K3", "LiH_R3p0_ADAPT5"), ("LiH_r16_m2_K1", "LiH_R1p6_ADAPT2")]
RADII = [("exact", 0.0), ("1.6 mHa", 1.6e-3), ("5 mHa", 5e-3), ("30 mHa", 3e-2), ("3.07 Ha (10^4 shots)", 3.07)]
SWEEP = [("NEUTRAL", 1.0), ("CLASSICAL_T8", 8.0), ("CLASSICAL_T32", 32.0), ("CLASSICAL_T128", 128.0), ("CLASSICAL_T512", 512.0)]


def tau_of(x, d0, s):
    if not math.isfinite(x):
        return 1.0 / (d0 + 1.0)
    return 1.0 / (d0 - math.expm1(-((x / s) ** 2)))


def sweep_ratio(R, N, tau):
    """expected A1 ratio at trust tau (log-linear interpolation of the constant-trust sweep; flat above 512)."""
    pts = []
    for arm, t in SWEEP:
        if arm == "NEUTRAL":
            pts.append((0.0, 1.0))
            continue
        p = c.testa_pct(R, N, "A1", arm)
        if p:
            pts.append((math.log(t), p[0]))
    xs, ys = zip(*pts)
    return float(np.interp(math.log(max(tau, 1.0)), xs, ys))


def main():
    print("state | E_c-E_q (mHa) | rule tau for r_E = " + ", ".join(r[0] for r in RADII))
    rows = []
    for state, case in STATES:
        R, meta = c.testa_rows("cp_v2", state)
        Rx, _ = c.testa_rows("cp_v3x", state)
        d = CAL["folds"]["loso_" + case]["fitted"]["CLASSICAL"]["energy"]
        e_c, e_q = meta["E_c"], meta["E_q"]
        gap = abs(e_c - e_q)
        taus = []
        for name, r in RADII:
            xp = (gap + r) / (abs(e_q) - r) if abs(e_q) > r else math.inf
            taus.append(tau_of(xp, d["d0"], d["s"]))
        rows.append((state, gap, d, taus, R, Rx))
        print(f"{state:15s} {1000*gap:8.3f}  d0={d['d0']:.5f} s={d['s']:.4f}  tau: " + "  ".join(f"{t:9.2f}" for t in taus))
    for N in (32, 128):
        print(f"\nA1, N={N}: expected R from the constant-trust sweep at the trust the rule assigns "
              "(columns: radii above; then constant 128, measured exact-energy rule from cp_v3x)")
        for state, gap, d, taus, R, Rx in rows:
            vals = [sweep_ratio(R, N, t) for t in taus]
            c128 = c.testa_pct(R, N, "A1", "CLASSICAL_T128")[0]
            xe = c.testa_pct(Rx, N, "A1", "CLASSICAL_XE")
            print(f"{state:15s} " + "  ".join(f"{v:6.3f}" for v in vals) + f"   | const128 {c128:6.3f}   measured rule (exact) {xe[0]:6.3f}")


if __name__ == "__main__":
    main()
