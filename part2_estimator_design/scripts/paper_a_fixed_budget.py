#!/usr/bin/env python3
"""Paper A: the uncertified fixed-budget selection against the certified selections (referee point A1).

Every cost in the paper is the price of *certifying* the arg-max at confidence ``delta`` (or of a
``rho``-good choice).  The practical default of ADAPT-VQE has no certificate: it spends a fixed
number of shots on the gradients of each step, takes the largest estimate and moves on.  The runs
``phase4_adapt_selection.py --methods "Fixed <shots>, <designed|uniform>"`` follow whole ADAPT
trajectories under that rule, with the exact VQE re-optimisation after every step (as for the
certified methods), and record the steps to chemical accuracy, the final energy error and the
total shots.

This script collects them (``runs/<case>/phase4/*_phase4_fixed_*``) into

* ``runs/paper_a/fixed_budget.csv``: one row per case, allocation and budget: trajectories, the share
  that reached chemical accuracy, the steps (mean, against the exact ADAPT's), the final energy
  error (median, 90th percentile), the share of selections that were exactly right / within ``rho``
  of the best, and the total shots to the end of the trajectory;
* ``runs/paper_a/fixed_budget_summary.csv``: per case and allocation the cheapest budget at which at
  least ``--reach`` of the trajectories reach chemical accuracy in at most ``--extra-steps`` more
  steps than exact ADAPT, with its total cost next to the certified methods' (II-A, II-0, M2).

The decisive number is the *premium of certification*: certified II-A total / cheapest adequate
fixed-budget total.  Whether a fixed budget is adequate is unknown to the run that uses it; the
budget here is the one a user who knew the answer would pick (an oracle choice), so the premium
is an upper bound on what certification costs a user who has to guess.

    python scripts/paper_a_fixed_budget.py [--latex ../reports/generated/tab_fixed_budget.tex]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outputs import write_csv  # noqa: E402

NAME = re.compile(r"^Fixed (?P<shots>\d+), (?P<allocation>designed|uniform|pilot)$")
ALLOCATIONS = ("designed", "pilot", "uniform")
SLACK = 1e-4  # Ha: a trajectory that stalls reaches what exact ADAPT reaches when it is within this of its final error
CHEMICAL_ACCURACY = 1.6e-3
CERTIFIED = ("II-A data, safe", "II-0 safe", "M2 safe", "M1 seq, safe")
LABEL = {"H4_square_eq_side1p0_HF": "H$_4$ 1.0 \\AA", "H4_square_stretch_side2p0_HF": "H$_4$ 2.0 \\AA",
         "LiH_R3p0_HF": "LiH 3.0 \\AA", "LiH_R1p6_HF": "LiH 1.6 \\AA", "H2O_eq_HF": "H$_2$O eq",
         "H2O_stretch_HF": "H$_2$O str", "BeH2_HF": "BeH$_2$"}


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def collect(runs: Path) -> tuple[list[dict], dict]:
    """One row per (case, allocation, budget); several runs may hold the same budget (the half-decade grid and the finer
    one): the row with the most trajectories is kept.  A trajectory has *reached* the target when its final error is
    below chemical accuracy or, when exact ADAPT itself stalls above it (stretched H4), within ``SLACK`` of exact ADAPT's."""
    rows, certified = {}, {}
    for case_dir in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "phase4").exists()):
        case = case_dir.name
        phase4 = case_dir / "phase4"
        for path in sorted(phase4.glob(f"{case}_phase4_fixed_*_trajectories.csv")):
            stem = path.name[: -len("_trajectories.csv")]
            summary_rows = read(phase4 / f"{stem}_summary.csv") if (phase4 / f"{stem}_summary.csv").exists() else []
            if not summary_rows:
                continue
            exact_iterations = int(summary_rows[0]["exact_adapt_iterations"])
            exact_final = float(summary_rows[0].get("exact_adapt_final_error", 0.0) or 0.0)
            target = max(CHEMICAL_ACCURACY, exact_final + SLACK)
            selections = defaultdict_list()
            sel_path = phase4 / f"{stem}_selections.csv"
            if sel_path.exists():
                for row in read(sel_path):
                    selections[row["method"]].append(row)
            by = defaultdict_list()
            for row in read(path):
                by[row["method"]].append(row)
            for method, trajectories in by.items():
                m = NAME.match(method)
                if m is None:
                    continue
                shots = np.array([float(r["total_shots"]) for r in trajectories])
                steps = np.array([float(r["iterations"]) for r in trajectories])
                errors = np.array([float(r["final_error"]) for r in trajectories])
                mine = selections[method]
                key = (case, m["allocation"], int(m["shots"]))
                row = {
                    "case": case, "allocation": m["allocation"], "budget": int(m["shots"]),
                    "trajectories": len(trajectories), "exact_iterations": exact_iterations, "exact_final_error": exact_final,
                    "reached_rate": float(np.mean(errors <= target)), "steps_mean": float(steps.mean()),
                    "steps_over_exact": float(steps.mean() / exact_iterations) if exact_iterations else float("nan"),
                    "final_error_median": float(np.median(errors)), "final_error_p90": float(np.percentile(errors, 90)),
                    "exact_best_rate": float(np.mean([r["exact_best"] == "True" for r in mine])) if mine else float("nan"),
                    "rho_good_rate": float(np.mean([float(r["shortfall"]) <= 0.1 + 1e-12 for r in mine])) if mine else float("nan"),
                    "total_shots_median": float(np.median(shots)), "total_shots_mean": float(shots.mean()),
                }
                if key not in rows or row["trajectories"] > rows[key]["trajectories"]:
                    rows[key] = row
        # the certified methods on the same trajectories (their ordinary runs; rho = 0.1)
        for method, (totals, _, _) in certified_runs(phase4, case).items():
            if method in CERTIFIED:
                certified.setdefault(case, {})[method] = (float(np.median(totals)), len(totals))
    return sorted(rows.values(), key=lambda r: (r["case"], r["allocation"], r["budget"])), certified


def certified_runs(phase4: Path, case: str, rho: float | None = None) -> dict:
    """Certified trajectories of a case: ``{method: (total shots, steps, reached)}`` as arrays.  ``rho=None``: the main runs
    (rho = 0.1) with the extra trajectories (other seed, tag ``extra``) pooled; otherwise the sensitivity runs of that rho."""
    if rho is None:
        paths = [p for p in sorted(phase4.glob(f"{case}_phase4*_trajectories.csv"))
                 if "_phase4_fixed_" not in p.name and "_phase4_sens_" not in p.name]
    else:
        paths = sorted(phase4.glob(f"{case}_phase4_sens_rho{rho:g}*_trajectories.csv"))
    paths = [p for p in paths if "_phase4_extra" not in p.name] + [p for p in paths if "_phase4_extra" in p.name]
    pooled: dict[str, list] = {}
    for path in paths:
        by = defaultdict_list()
        for row in read(path):
            if not row["method"].startswith("Fixed "):
                by[row["method"]].append(row)
        for method, rows in by.items():
            pooled[method] = pooled[method] + rows if ("_phase4_extra" in path.name and method in pooled) else rows
    return {m: (np.array([float(r["total_shots"]) for r in rs]), np.array([float(r["iterations"]) for r in rs]),
                np.array([r["reached"] == "True" for r in rs])) for m, rs in pooled.items()}


def rho_rows(runs: Path, rows: list[dict], reach: float, extra_steps: float) -> list[dict]:
    """The certified II-A and II-0 at each tolerance rho against the cheapest adequate fixed budget (designed allocation)."""
    out = []
    for case in dict.fromkeys(r["case"] for r in rows):
        best = cheapest_adequate(rows, case, "designed", reach, extra_steps)
        phase4 = runs / case / "phase4"
        for rho in (0.05, 0.1, 0.2, 0.3):
            found = certified_runs(phase4, case, None if rho == 0.1 else rho)
            record = {"case": case, "rho": rho, "fixed_budget": best["budget"] if best else float("nan"),
                      "fixed_total": best["total_shots_median"] if best else float("nan"),
                      "exact_iterations": next(r["exact_iterations"] for r in rows if r["case"] == case)}
            for method, key in (("II-A data, safe", "iia"), ("II-0 safe", "ii0")):
                if method in found:
                    totals, steps, reached = found[method]
                    record.update({f"{key}_total": float(np.median(totals)), f"{key}_steps": float(steps.mean()),
                                   f"{key}_reached": float(reached.mean()), f"{key}_n": len(totals)})
                    record[f"{key}_over_fixed"] = float(np.median(totals)) / best["total_shots_median"] if best else float("nan")
            if "iia_total" in record or "ii0_total" in record:
                out.append(record)
    return out


def latex_rho(rho_table: list[dict]) -> str:
    def sci(x):
        if x is None or not np.isfinite(x):
            return "--"
        exp = int(np.floor(np.log10(x)))
        return f"${x / 10 ** exp:.1f}\\times10^{{{exp}}}$"

    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"System & $\rho$ & II-A total (steps) & II-0 total (steps) & fixed total & II-A / fixed \\", r"\midrule"]
    previous = None
    for r in rho_table:
        name = LABEL.get(r["case"], r["case"]) if r["case"] != previous else ""
        if previous is not None and r["case"] != previous:
            lines.append(r"\midrule")
        previous = r["case"]

        def cell(key):
            if f"{key}_total" not in r:
                return "--"
            return f"{sci(r[key + '_total'])} ({r[key + '_steps']:.1f})"
        ratio = r.get("iia_over_fixed", float("nan"))
        lines.append(f"{name} & {r['rho']:g} & {cell('iia')} & {cell('ii0')} & {sci(r['fixed_total'])} & "
                     f"{'--' if not np.isfinite(ratio) else f'{ratio:.2f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


def cheapest_adequate(rows: list[dict], case: str, allocation: str, reach: float, extra_steps: float):
    pool = [r for r in rows if r["case"] == case and r["allocation"] == allocation
            and r["reached_rate"] >= reach and r["steps_mean"] <= r["exact_iterations"] + extra_steps]
    return min(pool, key=lambda r: r["budget"]) if pool else None


def summary(rows: list[dict], certified: dict, reach: float, extra_steps: float) -> list[dict]:
    out = []
    for case in dict.fromkeys(r["case"] for r in rows):
        for allocation in ALLOCATIONS:
            best = cheapest_adequate(rows, case, allocation, reach, extra_steps)
            record = {"case": case, "allocation": allocation, "reach": reach, "extra_steps": extra_steps,
                      "budget": best["budget"] if best else float("nan"),
                      "total_shots_median": best["total_shots_median"] if best else float("nan"),
                      "steps_mean": best["steps_mean"] if best else float("nan"),
                      "exact_iterations": next(r["exact_iterations"] for r in rows if r["case"] == case)}
            for method in CERTIFIED:
                value = certified.get(case, {}).get(method)
                record[f"certified {method}"] = value[0] if value else float("nan")
                record[f"premium {method}"] = (value[0] / best["total_shots_median"]) if (value and best) else float("nan")
            out.append(record)
    return out


def latex(summary_rows: list[dict]) -> str:
    def sci(x):
        if not np.isfinite(x):
            return "--"
        exp = int(np.floor(np.log10(x)))
        return f"${x / 10 ** exp:.1f}\\times10^{{{exp}}}$"

    by_case = {}
    for r in summary_rows:
        by_case.setdefault(r["case"], {})[r["allocation"]] = r
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"System & exact steps & designed: total (steps) & pilot: total & II-A (certified) & II-A / designed & II-A / pilot \\", r"\midrule"]
    for case, d in by_case.items():
        a, b = d.get("designed"), d.get("pilot")
        steps = f" ({a['steps_mean']:.1f})" if a and np.isfinite(a["steps_mean"]) else ""
        prem_a = f"{a['premium II-A data, safe']:.1f}" if a and np.isfinite(a["premium II-A data, safe"]) else "--"
        prem_b = f"{b['premium II-A data, safe']:.1f}" if b and np.isfinite(b["premium II-A data, safe"]) else "--"
        lines.append(f"{LABEL.get(case, case)} & {a['exact_iterations']} & {sci(a['total_shots_median'])}{steps} & "
                     f"{sci(b['total_shots_median']) if b else '--'} & {sci(a['certified II-A data, safe'])} & {prem_a} & {prem_b} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--reach", type=float, default=0.95, help="share of trajectories that must reach chemical accuracy")
    parser.add_argument("--extra-steps", type=float, default=1.0, help="mean steps allowed beyond exact ADAPT's")
    parser.add_argument("--latex", type=Path, default=None)
    parser.add_argument("--latex-rho", type=Path, default=None, help="premium of certification at other tolerances rho")
    args = parser.parse_args()
    rows, certified = collect(args.runs)
    if not rows:
        print("no fixed-budget runs found")
        return
    out = args.runs / "paper_a"
    write_csv(out / "fixed_budget.csv", list(rows[0]), rows)
    summ = summary(rows, certified, args.reach, args.extra_steps)
    write_csv(out / "fixed_budget_summary.csv", list(summ[0]), summ)
    for case in dict.fromkeys(r["case"] for r in rows):
        print(f"\n{case}  (exact ADAPT {next(r['exact_iterations'] for r in rows if r['case'] == case)} steps)")
        for allocation in ALLOCATIONS:
            for r in sorted((r for r in rows if r["case"] == case and r["allocation"] == allocation), key=lambda r: r["budget"]):
                print(f"  {allocation:8s} {r['budget']:>12,d}/step  reached {r['reached_rate']:.2f}  steps {r['steps_mean']:5.1f}  "
                      f"final dE median {r['final_error_median']:.2e}  exact-best {r['exact_best_rate']:.2f}  "
                      f"rho-good {r['rho_good_rate']:.2f}  total {r['total_shots_median']:12,.0f}")
    print("\ncheapest adequate budgets (reach >= {:.0%}, <= exact + {:g} steps):".format(args.reach, args.extra_steps))
    for r in summ:
        c = {m: r[f"certified {m}"] for m in CERTIFIED}
        print(f"  {r['case']:28s} {r['allocation']:8s} budget {r['budget']:>12,.0f}  total {r['total_shots_median']:13,.0f}  "
              f"II-A {c['II-A data, safe']:13,.0f} (x{r['premium II-A data, safe']:5.1f})  II-0 x{r['premium II-0 safe']:5.1f}  "
              f"M2 x{r['premium M2 safe']:5.1f}")
    table = rho_rows(args.runs, rows, args.reach, args.extra_steps)
    if table:
        write_csv(out / "fixed_rho.csv", list(table[0]), table)
        print("\ncertified II-A / II-0 at other tolerances against the cheapest adequate fixed budget (designed allocation):")
        for r in table:
            print(f"  {r['case']:28s} rho {r['rho']:4.2f}  II-A {r.get('iia_total', float('nan')):13,.0f} ({r.get('iia_steps', float('nan')):4.1f} steps, "
                  f"n={r.get('iia_n', 0)})  II-0 {r.get('ii0_total', float('nan')):13,.0f}  fixed {r['fixed_total']:13,.0f}  II-A/fixed {r.get('iia_over_fixed', float('nan')):5.2f}")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex(summ), newline="\n")
        print(f"wrote {args.latex}")
    if args.latex_rho and table:
        args.latex_rho.write_text(latex_rho(table), newline="\n")
        print(f"wrote {args.latex_rho}")


if __name__ == "__main__":
    main()
