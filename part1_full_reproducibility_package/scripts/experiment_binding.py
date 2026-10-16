#!/usr/bin/env python3
"""How over-resolved are the arms at M3's breadth breakpoint?

At the first breakpoint every arm is active at radius r0.  The allocation is sized
by its binding arms; every other arm reaches a smaller radius kappa_i * r0.  This
reports the fraction of arms that bind (kappa > 0.99), the median kappa, and the
leader's kappa.  A losing arm i is removed once the leader leads it by
(kappa_i + kappa_lead) r, so a leader resolved far below the target lets every
binding loser go at up to twice the radius the planning bound assumes; the last
column is that factor, 2 / (1 + kappa_lead), for a binding loser.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bai import elimination_thresholds
from cases import get_case
from problem_cache import load_or_build
from shot_models import allocate_context_shots

print(f"{'case':32s}{'arms':>6}{'contexts':>10}{'binding':>9}{'median kappa':>14}"
      f"{'kappa zero-g':>14}{'kappa leader':>14}{'1/(1+k_lead)':>14}")
for case in sys.argv[1:]:
    p = load_or_build(get_case(case), Path(".cache"))
    sig = p.parent_fragment_sigmas()
    t0 = elimination_thresholds(p.abs_gradients)[0]
    A = list(t0.active)
    shape = allocate_context_shots(sig[A], 1.0)
    live = shape > 0
    kappa = np.sqrt((sig[np.ix_(A, np.flatnonzero(live))] ** 2 / shape[live]).sum(axis=1))
    zero = p.abs_gradients[A] < 1e-6
    lead = A.index(int(np.argmax(p.abs_gradients)))
    kl = kappa[lead]
    print(f"{case:32s}{len(A):>6}{sig.shape[1]:>10}{np.mean(kappa > 0.99):>9.1%}"
          f"{np.median(kappa):>14.3f}"
          f"{np.median(kappa[zero]) if zero.any() else float('nan'):>14.3f}"
          f"{kl:>14.3f}{2.0 / (1.0 + kl):>14.3f}", flush=True)
