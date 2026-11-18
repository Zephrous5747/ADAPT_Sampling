import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
from classical_prior_tables import STATES, FAMILIES, load
from paper_a_bootstrap import ratio_interval

runs = Path("runs")
rng = np.random.default_rng(1)
R400 = "II-A data, safe, bound start, min shots=400, diagnose"
R50 = "II-A data, safe, bound start, min shots=50, diagnose"
print("== A: CPB arms against the n=400 data-only II-A (the draft headline), ratio of means [95% CI]; median ratio")
arms = [("CPB split, sparse tau=100", "sparse t100"), ("CPB split, HF tau=100", "HF t100"), ("CPB split, true state tau=100", "true t100"), (R50, "II-A n50")]
print("state".ljust(20), *[a[1].ljust(26) for a in arms])
geo = {a[1]: [] for a in arms}
for case, name in STATES:
    t = load(runs, case)
    if R400 not in t:
        continue
    ref = t[R400][0]
    cells = []
    for label, short in arms:
        if label not in t:
            cells.append("-".ljust(26)); continue
        s, c = t[label]
        r, lo, hi = ratio_interval(s, ref, "mean", rng)
        m = float(np.median(s) / np.median(ref))
        geo[short].append(r)
        cells.append(f"{r:5.2f} [{lo:4.2f},{hi:5.2f}] med {m:4.2f}".ljust(26))
    print(name.ljust(20), *cells)
print("geomean".ljust(20), *[f"{np.exp(np.mean(np.log(geo[a[1]]))):.2f} (n={len(geo[a[1]])})".ljust(26) for a in arms])

print()
print("== B: robust ratios (median of shots), CPD at n400 and CPB at n50, per arm: geometric mean over states of median ratio, #states with median ratio <0.95 / >1.05")
for fam in ("CPD", "CPB"):
    spec = FAMILIES[fam]
    ref_label = spec["reference"][0]
    for label, nm in spec["arms"]:
        v = []
        for case, name in STATES:
            t = load(runs, case)
            if ref_label in t and label in t:
                v.append(np.median(t[label][0]) / np.median(t[ref_label][0]))
        v = np.array(v)
        if v.size:
            print(fam, nm.ljust(44), f"geomean {np.exp(np.mean(np.log(v))):.3f} n={v.size}  <0.95: {(v < 0.95).sum()}  >1.05: {(v > 1.05).sum()}")

print()
print("== C: the tails: trials above 10x the arm median (count, max/median) for arms with a large mean/median ratio")
for case, name in STATES:
    try:
        from parallel import read_trials
        labels = read_trials(runs, case, "step5")
    except Exception:
        continue
    for label, rows in labels.items():
        if not label.startswith(("CP", "II-")):
            continue
        s = np.array([float(r["shots"]) for r in rows])
        if len(rows) >= int(rows[0]["n_trials"]) and s.mean() > 1.5 * np.median(s):
            print(name.ljust(20), label[:48].ljust(48), f"mean/median {s.mean() / np.median(s):6.1f}  max/median {s.max() / np.median(s):8.1f}  n>10x {(s > 10 * np.median(s)).sum()}  correct {np.mean([int(r['correct']) for r in rows]):.2f}")

print()
print("== D: share of the (data-only -> true-state) gap closed by the sparse prior, CPD static arms, states where the true-state design helps (ratio <0.97)")
for case, name in STATES:
    t = load(runs, case)
    a, b, c = R400, "CPD split, true state static", "CPD split, sparse static"
    if all(x in t for x in (a, b, c)):
        ref = t[a][0].mean()
        ro, rs = t[b][0].mean() / ref, t[c][0].mean() / ref
        flag = "" if ro < 0.97 else "  (no headroom)"
        print(name.ljust(20), f"oracle {ro:.2f}  sparse {rs:.2f}" + (f"  closed {(1 - rs) / (1 - ro):.0%}" if ro < 0.97 else flag))
