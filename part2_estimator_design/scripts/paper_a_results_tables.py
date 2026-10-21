#!/usr/bin/env python3
"""Paper A, step 3 of the compression plan: the three tables that the Results are built on, from the stored runs.

``tab_systems``   T1: the benchmark systems and the structure of the shared measurements (replaces the old Tables I and XXVII)
``tab_planning``  T2: planning bounds, what sharing, elimination and both save against the no-sharing baseline (replaces Table II)
``tab_learning``  T3: what learned estimators add, finite shots, fixed and trajectory states (replaces Tables V, VI, VII)

Tabular bodies are written to ``reports/generated/<name>.tex`` (caption and table note stand in the draft; paste with ``paper_a_inline.py``).

    python scripts/paper_a_results_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = ROOT / "runs"
PART1 = ROOT.parent / "part1_full_reproducibility_package" / "runs"
OUT = ROOT.parent / "reports" / "generated"


def rows_of(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ok(x: float) -> bool:
    return x == x


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def sci(x: float, d: int = 2) -> str:
    if not ok(x):
        return "--"
    if x < 1e4:
        return thousands(x)
    e = len(str(int(x))) - 1
    return f"${x / 10 ** e:.{d}f}\\times10^{{{e}}}$"


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------- T1

SCALING = ["paper_a", "paper_a_n2", "paper_a_dz", "paper_a_dz11o", "paper_a_631g"]
T1_ROWS = [("H4_square_eq_side1p0_CISD", r"H$_4$", "H4_square_eq_side1p0_HF"), ("LiH_R3p0_HF", "LiH", "LiH_R3p0_HF"),
           ("BeH2_HF", r"BeH$_2$", None), ("H2O_eq_HF", r"H$_2$O", "H2O_eq_HF"),
           ("H2O_dz8o_eq_HF", r"H$_2$O, cc-pVDZ window", None), ("H6_chain_HF", r"H$_6$ chain\textsuperscript{a}", None),
           ("N2_HF", r"N$_2$\textsuperscript{a}", None)]


def separate_contexts(case: str | None) -> float:
    """Contexts when every gradient is grouped on its own (the groups of independent elimination)."""
    if case is None:
        return float("nan")
    p = PART1 / case / f"{case}_M2_BAIFCIG_groups.csv"
    return sum(1 for _ in p.open()) - 1 if p.exists() else float("nan")


def classical_seconds(case: str) -> float:
    """Mean classical seconds of one complete II-A selection on a fixed state of the molecule (as in the old scaling table)."""
    from paper_a_referee_tables import classical_seconds as old

    return old(case)


def tab_systems() -> None:
    scal = {}
    for folder in SCALING:
        for r in rows_of(RUNS / folder / "scaling.csv"):
            scal[r["case"]] = r
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"System & Qubits & $K$ & Pauli products & Contexts $N$ & \shortstack{Gradients\\per product} & \shortstack{CZ gates\\per circuit} & \shortstack{II-A selection\\(s)}\\",
             r"\midrule"]
    for case, name, part1 in T1_ROWS:
        r = scal.get(case)
        if r is None:
            continue
        k, b0, n = fnum(r["generators"]), fnum(r["universal_support"]), fnum(r["library_contexts"])
        uses = k * fnum(r["commutator_terms_mean"]) / b0
        sep = separate_contexts(part1)
        ctx = thousands(n) + (f" ({thousands(sep)})" if ok(sep) else "")
        sec = classical_seconds(case)
        lines.append(f"{name} & {int(fnum(r['qubits']))} & {int(k)} & {thousands(b0)} & {ctx} & {uses:.1f} & {fnum(r['two_qubit_mean']):.1f} & "
                     f"{thousands(sec) if ok(sec) else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_systems", lines)


# ---------------------------------------------------------------------------------------------------------- T2

T2_STATES = [("H4_square_eq_side1p0_HF", r"H$_4$ 1.0~\AA, HF"), ("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0~\AA, CISD"),
             ("H4_square_stretch_side2p0_HF", r"H$_4$ 2.0~\AA, HF"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0~\AA, CISD"),
             ("LiH_R3p0_HF", r"LiH 3.0~\AA, HF"), ("H2O_eq_HF", r"H$_2$O eq, HF"), ("H2O_eq_CISD", r"H$_2$O eq, CISD"),
             ("H2O_stretch_HF", r"H$_2$O str, HF"), ("H2O_stretch_CISD", r"H$_2$O str, CISD")]


def planning(case: str) -> dict | None:
    """M1, M2 (part 1), M3 on actual precision (II-0 level of the oracle-ceiling run) and the no-sharing baseline."""
    ceil = [r for r in rows_of(RUNS / case / f"{case}_step3_oracle_ceiling.csv") if r["strategy"] == "mass" and r["level"] == "II-0"]
    p1 = PART1 / case
    ns = p1 / f"{case}_NOSHARE_summary.json"
    if not ceil or not ns.exists():
        return None
    c = ceil[0]
    summ = json.loads((p1 / f"{case}_summary.json").read_text()) if (p1 / f"{case}_summary.json").exists() else {}
    gap = fnum(summ.get("top_abs_gradient")) - fnum(summ.get("second_abs_gradient"))
    return {"m1": fnum(c["part1_m1"]), "m2": fnum(c["part1_m2"]), "m3": fnum(c["m3_actual_static"]),
            "base": float(json.loads(ns.read_text())["total_context_shots"]), "gap": gap}


def ratio(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def tab_planning() -> None:
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Saving against no sharing} & \multicolumn{2}{c}{M3 against} & \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"State & Gap & Sharing (M1) & Elimination (M2) & Both (M3) & M1 & M2 & Interaction\\", r"\midrule"]
    for case, name in T2_STATES:
        d = planning(case)
        if d is None:
            continue
        s, e, b = d["base"] / d["m1"], d["base"] / d["m2"], d["base"] / d["m3"]
        lines.append(f"{name} & {d['gap']:.4f} & {ratio(s)} & {ratio(e)} & {ratio(b)} & {ratio(d['m1'] / d['m3'])} & {ratio(d['m2'] / d['m3'])} & {b / (s * e):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_planning", lines)


# ---------------------------------------------------------------------------------------------------------- T3

T3_STATES = [("H4_square_eq_side1p0_CISD", r"H$_4$ 1.0~\AA, CISD"), ("H4_square_stretch_side2p0_CISD", r"H$_4$ 2.0~\AA, CISD"), ("LiH_R3p0_HF", r"LiH 3.0~\AA, HF"),
             ("H2O_eq_CISD", r"H$_2$O eq, CISD"), ("H2O_stretch_CISD", r"H$_2$O str, CISD"),
             ("H4_square_eq_side1p0_ADAPT5", r"H$_4$ 1.0~\AA, iteration 5"), ("LiH_R3p0_ADAPT3", r"LiH 3.0~\AA, iteration 3"),
             ("LiH_R3p0_ADAPT5", r"LiH 3.0~\AA, iteration 5"), ("H2O_eq_ADAPT11", r"H$_2$O eq, iteration 11"),
             ("H2O_stretch_ADAPT8", r"H$_2$O str, iteration 8")]


def tab_learning() -> None:
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Change against II-0} & \\", r"\cmidrule(lr){3-4}",
             r"State & II-0 (shots) & II-A, learned & \shortstack{II-A, exact\\covariances} & \shortstack{Correct\\selections}\\", r"\midrule"]
    for i, (case, name) in enumerate(T3_STATES):
        if i == 5:
            lines.append(r"\midrule")
        cfg = {r["config"]: r for r in rows_of(RUNS / case / f"{case}_step4_learned_designs.csv")}
        a0, a, ao, o0 = cfg.get("II-0 estimated"), cfg.get("II-A data"), cfg.get("II-A oracle"), cfg.get("II-0 oracle")
        if not (a0 and a):
            continue
        base = fnum(a0["shots_mean"])
        learned = f"${100 * (fnum(a['shots_mean']) / base - 1):+.0f}\\%$"
        exact = f"${100 * (fnum(ao['shots_mean']) / fnum(o0['shots_mean']) - 1):+.0f}\\%$" if (ao and o0 and i < 5) else "--"
        corr = 100 * fnum(a["correct_rate"])
        lines.append(f"{name} & {sci(base)} & {learned} & {exact} & {corr:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_learning", lines)


if __name__ == "__main__":
    tab_systems()
    tab_planning()
    tab_learning()
