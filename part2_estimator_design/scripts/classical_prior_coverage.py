import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
from classical_prior_tables import STATES
from parallel import read_trials

runs = Path("runs")
arms = [("II-A data, safe, bound start, min shots=400, diagnose", "II-A n400"),
        ("II-A data, safe, bound start, min shots=50, diagnose", "II-A n50"),
        ("CPD split, sparse tau=100", "CPD sparse t100"),
        ("CPB split, sparse tau=100", "CPB sparse t100"),
        ("CPB split, HF tau=100", "CPB HF t100"),
        ("CPB split, true state tau=100", "CPB true t100"),
        ("CPB alloc, sparse tau=100", "CPB alloc sparse")]
print("share of trials with at least one interval that missed the exact value (miscovered_rounds>0) / best arm eliminated / correct selection")
print("state".ljust(20), *[a[1].ljust(17) for a in arms])
tot = {a[1]: [0, 0, 0, 0] for a in arms}
for case, name in STATES:
    labels = read_trials(runs, case, "step5")
    cells = []
    for label, short in arms:
        rows = labels.get(label)
        if not rows or len(rows) < int(rows[0]["n_trials"]):
            cells.append("-".ljust(17)); continue
        miss = np.mean([float(r.get("miscovered_rounds", 0) or 0) > 0 for r in rows])
        be = np.mean([str(r.get("best_eliminated", "")).lower() in ("1", "true") for r in rows])
        ok = np.mean([int(r["correct"]) for r in rows])
        cells.append(f"{miss:4.2f}/{be:4.2f}/{ok:4.2f}".ljust(17))
        t = tot[short]; t[0] += miss * len(rows); t[1] += be * len(rows); t[2] += ok * len(rows); t[3] += len(rows)
    print(name.ljust(20), *cells)
print("pooled".ljust(20), *[(f"{t[0]/t[3]:4.2f}/{t[1]/t[3]:4.2f}/{t[2]/t[3]:4.2f}" if t[3] else "-").ljust(17) for t in tot.values()])
