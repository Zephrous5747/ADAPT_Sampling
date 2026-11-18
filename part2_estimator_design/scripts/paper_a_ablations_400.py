#!/usr/bin/env python3
"""Ablations at n_min = 400 (critique Q14, Q15 and round-3 Q1 edit 4): cross-fitting, the radius shrink factor gamma, II-A with the free energy data.

Reads the stored step-5 trials (100 per cell, seed 5, the same streams as the reference runs) and writes ``runs/paper_a/ablations_400.csv`` and
``ablations_400_facts.json``:

* ``crossfit``: II-A at 400 without cross-fitting (design fitted and applied on the same pooled shots) over II-A at 400 with it;
  also the shares of exact-best selections and of trials with a missed interval (the variant is biased and its radii are in-sample);
* ``gamma``: II-A, II-0 and M2 at 400 with gamma = 0.8 and 0.95 over the same method with the benchmark's 0.9;
* ``reuse``: II-A and II-0 with the free data of the last energy evaluation (shot reuse, FC contexts) over the same method without it, and over the
  reuse baseline of Ikhtiarudin et al. (``Ikh reuse, FC``) at 400.

Every ratio is a ratio of mean context-shots with a 95% percentile interval (``paper_a_bootstrap.ratio_interval``, 10,000 draws).

    python scripts/paper_a_ablations_400.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_a_bootstrap import ratio_interval  # noqa: E402
from parallel import slug as _slug  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
A400, I400, M2_400 = "II-A data, safe, bound start, min shots=400, diagnose", "II-0, safe, bound start, min shots=400, diagnose", "M2 safe, bound start, min shots=400"
CROSSFIT_STATES = ["H4_square_eq_side1p0_ADAPT5", "H4_square_stretch_side2p0_ADAPT5"]  # the job file lists the states; all cases with a no-crossfit file are used


def slug(label: str) -> str:
    return _slug(label)


def trials(case: str, label: str) -> pd.DataFrame | None:
    files = glob.glob(str(RUNS / case / "trials" / f"step5__{slug(label)}__shard*.csv"))
    return pd.read_csv(files[0]) if files else None


def ratio(num: pd.DataFrame, den: pd.DataFrame, rng) -> list[float]:
    return [round(x, 3) for x in ratio_interval(num.shots.to_numpy(float), den.shots.to_numpy(float), "mean", rng)]


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    cases = sorted({p.parent.parent.name for p in map(Path, glob.glob(str(RUNS / "*" / "trials" / "step5__*no_crossfit*.csv")))})
    for case in cases:
        a, b = trials(case, A400 + ", no crossfit"), trials(case, A400)
        if a is None or b is None:
            continue
        r = ratio(a, b, rng)
        rows.append({"study": "crossfit", "case": case, "variant": "II-A no cross-fit / II-A", "ratio": r[0], "lo": r[1], "hi": r[2], "n": len(a), "exact_best_variant": a.correct.mean(),
                     "exact_best_reference": b.correct.mean(), "missed_variant": float((a.miscovered_rounds > 0).mean()), "missed_reference": float((b.miscovered_rounds > 0).mean()),
                     "wrong_variant": int((~a.correct.astype(bool)).sum()), "wrong_reference": int((~b.correct.astype(bool)).sum())})
    for case in sorted({p.parent.parent.name for p in map(Path, glob.glob(str(RUNS / "*" / "trials" / "step5__*gamma=0.8*.csv")))}):
        for method, ref_label, label in (("II-A", A400, "II-A data, safe, bound start, min shots=400"), ("II-0", I400, "II-0, safe, bound start, min shots=400"), ("M2", M2_400, "M2 safe, bound start, min shots=400")):
            ref = trials(case, ref_label)
            for g in (0.8, 0.95):
                t = trials(case, f"{label}, gamma={g:g}")
                if t is None or ref is None:
                    continue
                r = ratio(t, ref, rng)
                rows.append({"study": "gamma", "case": case, "variant": f"{method} gamma={g:g} / gamma=0.9", "ratio": r[0], "lo": r[1], "hi": r[2], "n": len(t), "exact_best_variant": t.correct.mean(),
                             "exact_best_reference": ref.correct.mean(), "wrong_variant": int((~t.correct.astype(bool)).sum()), "wrong_reference": int((~ref.correct.astype(bool)).sum())})
    for case in sorted({p.parent.parent.name for p in map(Path, glob.glob(str(RUNS / "*" / "trials" / "step5__II-A_data_safe_reuse_FC_bound_start_min_shots=400__*.csv")))}):
        for method, rl, ref_label in (("II-A", "II-A data, safe + reuse, FC, bound start, min shots=400", A400), ("II-0", "II-0 safe + reuse, FC, bound start, min shots=400", I400)):
            t, ref = trials(case, rl), trials(case, ref_label)
            ikh = trials(case, "Ikh reuse, FC, safe, bound start, min shots=400")
            if t is None:
                continue
            r = ratio(t, ref, rng) if ref is not None else [np.nan] * 3
            row = {"study": "reuse", "case": case, "variant": f"{method} + reuse / {method}", "ratio": r[0], "lo": r[1], "hi": r[2], "n": len(t), "exact_best_variant": t.correct.mean(),
                   "mean_variant": float(t.shots.mean()), "mean_reference": float(ref.shots.mean()) if ref is not None else np.nan}
            if ikh is not None:
                rr = ratio(ikh, t, rng)
                row.update(ikh_over_variant=rr[0], ikh_lo=rr[1], ikh_hi=rr[2], mean_ikh=float(ikh.shots.mean()))
            rows.append(row)
    table = pd.DataFrame(rows)
    out = RUNS / "paper_a"
    table.to_csv(out / "ablations_400.csv", index=False)
    facts = {}
    for study, sub in table.groupby("study"):
        facts[study] = {"cells": int(len(sub)), "ratio_min": float(sub.ratio.min()), "ratio_max": float(sub.ratio.max()), "ratio_median": float(sub.ratio.median())}
    (out / "ablations_400_facts.json").write_text(json.dumps(facts, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(table.drop(columns=[c for c in table.columns if c.startswith("wrong")]).round(3).to_string(index=False))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
