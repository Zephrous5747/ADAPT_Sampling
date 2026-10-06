#!/usr/bin/env python3
"""Cost of the confidence radii of Paper A against the information-theoretic minimum.

Discussion item: how much of a selection's cost is the price of a *valid* decision, as
opposed to the price of the information the decision needs?  The idealisation is two
Gaussian arms with equal variance ``s^2`` and gap ``D``, confidence ``1 - delta``:

* lower bound (Garivier and Kaufmann, COLT 2016): the total number of samples is at
  least ``T* kl(delta, 1 - delta)`` with ``T* = 8 s^2 / D^2`` for two arms;
* our rule: both arms are sampled until their radii satisfy ``2 R < D`` with
  ``R = z s / sqrt(n)``, so the total is ``8 z^2 s^2 / D^2``, where ``z`` is the
  per-round Bonferroni value ``Phi^-1(1 - delta / 2K)`` over the ``K`` gradients of the
  pool, or, for the anytime variant, the same with ``delta_r = 6 delta / (pi^2 r^2)``.

This is a back-of-the-envelope comparison: it ignores the elimination of the other
``K - 2`` arms, shared contexts and unequal variances.  The anytime-to-fixed ratio it
gives ((z_r / z)^2, 1.8 to 2.6 for r = 10 to 50) can be checked against the measured
1.8 to 2.6 of Q3 (anytime variant against per-round).  No state is needed.

    python scripts/price_of_validity.py [--delta 0.05]
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

POOLS = {"H4": 26, "LiH": 92, "H2O": 140}
ROUNDS = (10, 25, 50)


def kl(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()
    d = args.delta
    norm = NormalDist()
    bound = 8 * kl(d, 1 - d)
    print(f"delta = {d}: lower bound = {bound:.1f} s^2/D^2 total samples (two Gaussian arms)")
    print(f"{'pool':>5} {'K':>4} {'rule':>12} {'z':>6} {'cost':>7} {'vs bound':>9} {'vs per-round':>13}")
    for name, k in POOLS.items():
        z = norm.inv_cdf(1 - d / (2 * k))
        print(f"{name:>5} {k:>4} {'per-round':>12} {z:6.2f} {8 * z * z:7.0f} {8 * z * z / bound:9.1f} {1.0:13.2f}")
        for r in ROUNDS:
            dr = 6 * d / (math.pi ** 2 * r * r)
            za = norm.inv_cdf(1 - dr / (2 * k))
            print(f"{name:>5} {k:>4} {'anytime r=' + str(r):>12} {za:6.2f} {8 * za * za:7.0f} "
                  f"{8 * za * za / bound:9.1f} {(za / z) ** 2:13.2f}")


if __name__ == "__main__":
    main()
