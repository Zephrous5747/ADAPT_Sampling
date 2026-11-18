"""BeH2 table of the classical-prior study: every arm on the four saved ADAPT states (steps 2, 4, 7, 9), mean shots and the sum over the four."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classical_prior_tables import load  # noqa: E402
from paper_a_bootstrap import ratio_interval  # noqa: E402

CASES = ["BeH2_ADAPT2", "BeH2_ADAPT4", "BeH2_ADAPT7", "BeH2_ADAPT9"]
GROUPS = [
    ("n400 (design only; radii from data)", "II-A data, safe, bound start, min shots=400, diagnose", [
        ("baseline II-0 (no split)", "II-0, safe, bound start, min shots=400, diagnose"),
        ("baseline II-A (sampled covariance)", "II-A data, safe, bound start, min shots=400, diagnose"),
        ("prior: sparse, static", "CPD split, sparse static"),
        ("prior: sparse, tau=100", "CPD split, sparse tau=100"),
        ("prior: sparse, tau=1000", "CPD split, sparse tau=1000"),
        ("prior: sparse, tau=100, eps=0.01", "CPD split, sparse tau=100, eps=0.01"),
        ("prior: HF, tau=100", "CPD split, HF tau=100"),
        ("ORACLE: true state, static", "CPD split, true state static"),
        ("ORACLE: true state, tau=100", "CPD split, true state tau=100"),
    ]),
    ("n50 (prior also in radii and planning)", "II-A data, safe, bound start, min shots=50, diagnose", [
        ("baseline II-0 (no split)", "II-0, safe, bound start, min shots=50, diagnose"),
        ("baseline II-A (sampled covariance)", "II-A data, safe, bound start, min shots=50, diagnose"),
        ("prior: sparse, allocation only, tau=100", "CPB alloc, sparse tau=100"),
        ("prior: sparse, split, tau=100", "CPB split, sparse tau=100"),
        ("prior: sparse, split, tau=1000", "CPB split, sparse tau=1000"),
        ("prior: HF, split, draft shrinkage", "CPB split, HF nu=100 shrink"),
        ("prior: HF, split, tau=100", "CPB split, HF tau=100"),
        ("ORACLE: true state, allocation only", "CPB alloc, true state tau=100"),
        ("ORACLE: true state, split, tau=100", "CPB split, true state tau=100"),
    ]),
]
rng = np.random.default_rng(3)
data = {c: load(ROOT / "runs", c) for c in CASES}
for title, ref_label, arms in GROUPS:
    print(f"\n### {title}; ratio = mean shots / II-A of this family (95% CI); last columns: sum of the four states' mean shots and its ratio\n")
    print("| arm | ADAPT 2 | ADAPT 4 | ADAPT 7 | ADAPT 9 | sum (shots) | sum / II-A | correct (min) |")
    print("|---|---|---|---|---|---|---|---|")
    ref_sum = sum(data[c][ref_label][0].mean() for c in CASES)
    for name, label in arms:
        cells, total, ok = [], 0.0, []
        for c in CASES:
            if label not in data[c]:
                cells.append("-"); total = np.nan; continue
            s, correct = data[c][label]
            r, lo, hi = ratio_interval(s, data[c][ref_label][0], "mean", rng)
            cells.append(f"{r:.2f} [{lo:.2f}, {hi:.2f}]" if label != ref_label else f"{s.mean():.3g}")
            total += s.mean(); ok.append(correct)
        print(f"| {name} | " + " | ".join(cells) + f" | {total:.3g} | {total / ref_sum:.2f} | {100 * min(ok):.0f}% |")
