#!/usr/bin/env python3
"""Work order CP-1.2: calibration (leave-one-family-out), Test A (A1/A2 design quality) and Test B (selection shots).

    python scripts/cp_order_run.py calibrate --workers 24
    python scripts/cp_order_run.py testa --states H4_a2_m3 --reps 30 --workers 30
    python scripts/cp_order_run.py testb --states H4_a2_m3 --reps 30 --workers 30

Outputs go to ``runs/cp_order``: ``calibration.json`` (all grid scores, the fitted parameters per held-out family and source), ``testa/*.jsonl``,
``testb/*.jsonl``.  Test A and B read the frozen calibration of the state's family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")  # Clarabel's Rust pool ignores OMP_NUM_THREADS (see src/cp_order.py)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import part1_bridge  # noqa: E402,F401
from cp_order import HISTORY_SHOTS, TrueCovariance, history_strength  # noqa: E402
from cp_states import ALL_STATES, FAMILIES, STATES, arm_inputs, build_state, design_covariance, make_record, nested_counts  # noqa: E402
from cp_order import NeutralMoments  # noqa: E402
from cp_testb import run_cp  # noqa: E402
from note_protocol import integer_counts  # noqa: E402

FIXED_SOURCES = ("FC_SE", "FC_FULL")  # gate G2 arms of Test B: frozen reconstruction, with / without elimination
RESTART_SUFFIX = {"_NR": 0.0, "_H50": 0.5, "_H25": 0.25}  # restart policy of an arm (task T22); no suffix: a new block after every elimination

D0_GRID = (1 / 8, 1 / 32, 1 / 128)
S_MULT = (0.25, 1.0, 4.0)
CONSTANTS = (8.0, 32.0, 128.0)
# Wider grids of the in-family calibration (``--wide``): trust 1/d0 at zero discrepancy down to 1/2048, decay scale up to 16 x the median x_+,
# constants from 1 up to the static prior.  The first calibration put every best setting on the edge of the work order's grid.
D0_WIDE = (1 / 8, 1 / 32, 1 / 128, 1 / 512, 1 / 2048)
S_WIDE = (0.25, 1.0, 4.0, 16.0)
CONSTANTS_WIDE = (1.0, 2.0, 4.0, 8.0, 32.0, 128.0, 512.0, 2048.0, math.inf)
PREFIXES = (0, 8, 32, 128)
NS = "CP-1.2"
_SEL: dict = {}


def rng_of(*key) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, (NS,) + key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def load_selectors(states, solve_seconds: float) -> None:
    for state_id in states:
        started = time.perf_counter()
        _SEL[state_id] = build_state(state_id, solve_seconds=solve_seconds)
        s = _SEL[state_id]
        print(f"{state_id}: {len(s.candidates)} candidates, {s.n_contexts} contexts, L_E {s.energy.L}, r_E {s.energy.radius:.3f} Ha, "
              f"E_c - E_q = {1e3 * (s.energy_c - s.energy_q):+.3f} mHa ({time.perf_counter() - started:.0f}s)", flush=True)


def true_vmax(sel, designs: dict, counts: np.ndarray) -> tuple[float, dict]:
    v = sel.true_variances(designs, counts)
    return (max(v.values()) if v else 0.0), v


# --- calibration -----------------------------------------------------------------------------------------------------------------------


_PRE: dict = {}  # (state, replicate) -> (pre-measurement record, counts at N = 32); built in the parent, inherited by the forked workers


def _calibration_unit(args) -> dict:
    """One design solve over all arms for one (state, replicate, source, trust value); source None is the NEUTRAL reference.

    One unit takes about 100 s on LiH (92 arms, 57-65 of them through CVXPY), so the 192 cores are filled with units rather than with
    one serial task per state and replicate (the first version ran 6 LiH tasks of about 43 units each on 6 cores).
    """
    state_id, rep, source, value = args
    sel = _SEL[state_id]
    record, counts = _PRE[(state_id, rep)]
    arms = list(sel.candidates)
    uniform = np.full(sel.n_contexts, float(HISTORY_SHOTS))
    started = time.perf_counter()
    if source is None:
        prior, tau, _ = arm_inputs(sel, "NEUTRAL", record)
    elif source == "CLASSICAL":
        prior, tau, _ = arm_inputs(sel, "CLASSICAL", record, {"const": value})
    else:
        prior, _, _ = arm_inputs(sel, "PREVIOUS", record, {"const": value})
        tau = history_strength(value, HISTORY_SHOTS)
    designs, routes = sel.a1(arms, design_covariance(sel, counts, prior, tau))
    return {"state": state_id, "rep": rep, "source": source, "value": value, "v": true_vmax(sel, designs, uniform)[0],
            "routes": {r: list(routes.values()).count(r) for r in set(routes.values())}, "seconds": round(time.perf_counter() - started, 1)}


def calibrate(args) -> None:
    from cp_order import tau_of, x_upper, y_upper

    d0_grid, s_mult, constants = (D0_WIDE, S_WIDE, CONSTANTS_WIDE) if args.wide else (D0_GRID, S_MULT, CONSTANTS)
    groups = None
    if args.groups:  # in-family calibration: each group is fitted on its own states (in-sample), no held-out family
        groups = {spec.split("=")[0]: spec.split("=")[1].split(",") for spec in args.groups}
        states = list(dict.fromkeys(s for members in groups.values() for s in members))
    else:
        states = list(args.states)
    load_selectors(states, args.solve_seconds)
    records = {}
    for state_id in states:
        sel = _SEL[state_id]
        for rep in range(args.cal_reps):
            rec = make_record(sel, rng_of("cal-pre", state_id, rep))
            _PRE[(state_id, rep)] = (rec, nested_counts(sel, rng_of("cal-cur", state_id, rep), prefixes=(32,))[32])
            x = x_upper(sel.energy_c, rec["E_q"], rec["r_q"])
            y = y_upper(rec["E_q"], rec["r_q"], rec["E_prev"], rec["r_prev"]) if sel.has_previous else None
            if args.exact_energy:  # exact relative discrepancies: |E_c - E_q| / |E_q| and |E_q - E_prev| / |E_q|
                x = abs(sel.energy_c - sel.energy_q) / abs(sel.energy_q)
                y = abs(sel.energy_q - sel.energy_prev) / abs(sel.energy_q) if sel.has_previous else None
            records[(state_id, rep)] = {"x_plus": x, "y_plus": y}
    folds = {}
    needed: dict = {}  # (state, rep) -> {source: set of tau_E values}
    fold_defs = (list(groups.items()) if groups else
                 [(f, [s for s in states if ALL_STATES[s][2] != f]) for f in FAMILIES if any(ALL_STATES[s][2] == f for s in states)])
    for family, train in fold_defs:
        xs = [records[(s, r)]["x_plus"] for s in train for r in range(args.cal_reps) if math.isfinite(records[(s, r)]["x_plus"]) and records[(s, r)]["x_plus"] > 0]
        ys = [records[(s, r)]["y_plus"] for s in train for r in range(args.cal_reps)
              if records[(s, r)]["y_plus"] is not None and math.isfinite(records[(s, r)]["y_plus"]) and records[(s, r)]["y_plus"] > 0]
        x50 = float(np.median(xs)) if xs else None
        y50 = float(np.median(ys)) if ys else None
        folds[family] = {"train": train, "x50": x50, "y50": y50}
        for s in train:
            for r in range(args.cal_reps):
                slot = needed.setdefault((s, r), {"CLASSICAL": set(), "PREVIOUS": set()})
                for source, ref, key in (("CLASSICAL", x50, "x_plus"), ("PREVIOUS", y50, "y_plus")):
                    if source == "PREVIOUS" and not _SEL[s].has_previous:
                        continue
                    for c in constants:
                        slot[source].add(c)
                    value = records[(s, r)][key]
                    if ref is None or value is None:
                        continue
                    for d0 in d0_grid:
                        for mult in s_mult:
                            slot[source].add(tau_of(value, d0, ref * mult))
    units = []
    for (s, r), slots in needed.items():
        units.append((s, r, None, None))
        units += [(s, r, source, float(value)) for source, values in slots.items() for value in sorted(values)]
    units.sort(key=lambda u: -len(_SEL[u[0]].candidates) * _SEL[u[0]].n_contexts)  # the large states first
    # finished units are kept in a checkpoint file: a requeued or restarted job continues instead of starting over
    checkpoint = args.out / "calibration_units.jsonl"
    results = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                results[(row["state"], row["rep"], row["source"], repr(row["value"]))] = row
    todo = [u for u in units if (u[0], u[1], u[2], repr(u[3])) not in results]
    print(f"{len(units)} calibration units ({len(units) - len(todo)} already in the checkpoint), {args.workers} workers", flush=True)
    started = time.perf_counter()
    with open(checkpoint, "a") as handle, ProcessPoolExecutor(max(1, args.workers), mp_context=mp.get_context("fork")) as pool:
        for k, future in enumerate(as_completed([pool.submit(_calibration_unit, u) for u in todo]), start=1):
            row = future.result()
            results[(row["state"], row["rep"], row["source"], repr(row["value"]))] = row
            handle.write(json.dumps(row, default=float) + "\n")
            handle.flush()
            print(f"  [{k}/{len(todo)}] {row['state']} rep {row['rep']} {row['source']} {row['value']} v={row['v']:.4g} routes {row['routes']} "
                  f"({row['seconds']}s, elapsed {time.perf_counter() - started:.0f}s)", flush=True)
    scores = {}
    for (s, r, source, value), row in results.items():
        slot = scores.setdefault((s, r), {"state": s, "rep": r, "sources": {}})
        if source is None:
            slot["neutral"] = row["v"]
        else:
            slot["sources"].setdefault(source, {})[value] = row["v"]
    route_totals: dict = {}
    for row in results.values():
        for name, count in row["routes"].items():
            route_totals[name] = route_totals.get(name, 0) + count
    print("design routes over all units:", route_totals, flush=True)
    out = {"protocol": NS, "grid": {"d0": d0_grid, "s_mult": s_mult, "constants": constants}, "groups": groups, "routes": route_totals,
           "fallback_units": sorted({f"{s}|{r}|{src}|{val}" for (s, r, src, val), row in results.items() if row["routes"].get("fallback")}),
           "folds": {}, "scores": []}
    for family, info in folds.items():
        fitted = {}
        for source, ref, key in (("CLASSICAL", info["x50"], "x_plus"), ("PREVIOUS", info["y50"], "y_plus")):
            settings = []
            if ref is not None:
                settings += [{"kind": "energy", "d0": d0, "s": ref * mult, "mult": mult} for d0 in d0_grid for mult in s_mult]
            settings += [{"kind": "const", "value": c} for c in constants]
            rows = []
            for setting in settings:
                per_family: dict = {}
                for s in info["train"]:
                    if source == "PREVIOUS" and not _SEL[s].has_previous:
                        continue
                    losses = []
                    for r in range(args.cal_reps):
                        row = scores[(s, r)]
                        neutral = row["neutral"]
                        if neutral <= 1e-14:
                            continue  # analytically zero-variance state: a control, excluded from the loss
                        if setting["kind"] == "energy":
                            value = records[(s, r)][key]
                            tau = tau_of(value, setting["d0"], setting["s"])
                        else:
                            tau = setting["value"]
                        v = row["sources"][source][repr(float(tau))]
                        losses.append(math.log(max(v, 1e-300) / neutral))
                    if losses:
                        per_family.setdefault(ALL_STATES[s][2], []).append(float(np.mean(losses)))
                if not per_family:
                    continue
                loss = float(np.mean([np.mean(v) for v in per_family.values()]))
                rows.append({**setting, "loss": loss, "n_states": sum(len(v) for v in per_family.values())})
            out["scores"].append({"held_out": family, "source": source, "rows": rows})
            for kind in ("energy", "const"):
                candidates = [r_ for r_ in rows if r_["kind"] == kind]
                if not candidates:
                    continue
                best = min(candidates, key=lambda r_: (round(r_["loss"], 10), -r_.get("d0", 0.0), r_.get("s", 0.0)))
                best = dict(best)
                if kind == "energy":
                    best["range_limited"] = best["d0"] in (d0_grid[0], d0_grid[-1]) or best["mult"] in (s_mult[0], s_mult[-1])
                else:
                    best["range_limited"] = best["value"] in (constants[0], constants[-1])
                fitted.setdefault(source, {})[kind] = best
        out["folds"][family] = {"train": info["train"], "x50": info["x50"], "y50": info["y50"], "fitted": fitted}
    out["records"] = {f"{s}|{r}": v for (s, r), v in records.items()}
    path = args.out / "calibration.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print("calibration written", path, flush=True)
    for family, info in out["folds"].items():
        print(family, json.dumps(info["fitted"], default=float), flush=True)


# --- Test A ----------------------------------------------------------------------------------------------------------------------------


def frozen(calibration: dict, family: str, source: str, kind: str) -> dict | None:
    fit = calibration["folds"][family]["fitted"].get(source, {}).get(kind)
    if fit is None:
        return None
    return {"d0": fit["d0"], "s": fit["s"]} if kind == "energy" else {"const": fit["value"]}


def _a2(sel, cov, arms, budget: float):
    designs, plan, info = sel.joint(arms, cov, "A2", budget=budget)
    frozen_reconstruction = plan is None
    if plan is None:
        designs = {j: sel.x0[j] for j in arms}
        plan = sel.fixed_plan(arms, cov, "A2", budget=budget)
    counts = integer_counts(plan, int(round(budget)), 32).astype(float)
    return designs, counts, info, frozen_reconstruction


def _testa_task(args) -> list[dict]:
    state_id, rep, calibration = args
    sel = _SEL[state_id]
    family = sel.family
    record = make_record(sel, rng_of("A-pre", state_id, rep))
    streams = nested_counts(sel, rng_of("A-cur", state_id, rep), PREFIXES)
    arms = list(sel.candidates)
    L = sel.n_contexts
    budget = float(HISTORY_SHOTS * L)
    order = sorted(arms, key=lambda j: -abs(sel.truth[j]))
    top = order[:1]
    if calibration.get("xe"):
        # exact-energy variant: the energy rule of the work order with the exact relative discrepancy x = |E_c - E_q| / |E_q| (evaluator-side
        # information, never a selector input), fitted leaving the system out (group ``loso_<case>`` of the calibration), against the constant 128
        fit = frozen(calibration, "loso_" + ALL_STATES[state_id][0], "CLASSICAL", "energy")
        defs = [("NEUTRAL", "NEUTRAL", None, True), ("CLASSICAL_T128", "CLASSICAL", {"const": 128.0}, True),
                ("CLASSICAL_XE", "CLASSICAL", {**fit, "exact": True}, True)]
        fit_p = frozen(calibration, "loso_" + ALL_STATES[state_id][0], "PREVIOUS", "energy")
        if sel.has_previous and fit_p is not None:  # previous-state source: exact y = |E_q - E_prev| / |E_q|, against the full 128-outcome history
            defs += [("PREVIOUS_Tinf", "PREVIOUS", {"const": math.inf}, True), ("PREVIOUS_XE", "PREVIOUS", {**fit_p, "exact": True}, True)]
    elif calibration.get("v2"):
        # second round: a sweep of constant trust (no energy rule: fitted in-family it is a constant of about 230), A2 for four arms only
        defs = [("NEUTRAL", "NEUTRAL", None, True)]
        defs += [(f"CLASSICAL_T{'inf' if c == math.inf else int(c)}", "CLASSICAL", {"const": c}, c == 128.0) for c in (8.0, 32.0, 128.0, 512.0, math.inf)]
        defs += [("PREVIOUS_T128", "PREVIOUS", {"const": 128.0}, False), ("PREVIOUS_Tinf", "PREVIOUS", {"const": math.inf}, True)]
    else:
        defs = [("NEUTRAL", "NEUTRAL", None, True),
                ("CLASSICAL", "CLASSICAL", frozen(calibration, family, "CLASSICAL", "energy"), True),
                ("CLASSICAL_CONST", "CLASSICAL", frozen(calibration, family, "CLASSICAL", "const"), True),
                ("PREVIOUS", "PREVIOUS", frozen(calibration, family, "PREVIOUS", "energy"), True),
                ("PREVIOUS_CONST", "PREVIOUS", frozen(calibration, family, "PREVIOUS", "const"), True)]
    rows = []
    for name, source, params, with_a2 in defs:
        if source == "PREVIOUS" and not sel.has_previous:
            continue
        if source in ("CLASSICAL", "PREVIOUS") and name != "NEUTRAL" and params is None:
            continue
        for N in PREFIXES:
            prior, tau, info = arm_inputs(sel, source, record, params)
            cov = design_covariance(sel, streams[N], prior, tau)
            started = time.perf_counter()
            d1, routes = sel.a1(arms, cov)
            v1, per1 = true_vmax(sel, d1, np.full(L, float(HISTORY_SHOTS)))
            modes = [("A1", v1, per1)]
            solver = fixed = None
            if with_a2:
                d2, n2, solver, fixed = _a2(sel, cov, arms, budget)
                v2, per2 = true_vmax(sel, d2, n2)
                modes.append(("A2", v2, per2))
            for mode, v, per in modes:
                rows.append({"state": state_id, "rep": rep, "N": N, "arm": name, "mode": mode, "vmax": v,
                             "v_top": max(per[j] for j in top) if per else 0.0, "info": info,
                             "routes": {r: list(routes.values()).count(r) for r in set(routes.values())} if mode == "A1" else None,
                             "solver": solver if mode == "A2" else None, "frozen_reconstruction": fixed if mode == "A2" else None,
                             "seconds": round(time.perf_counter() - started, 2)})
    # evaluator-only attainable reference with the true covariance (once per replicate)
    cov = TrueCovariance(sel.oracle)
    d1, _ = sel.a1(arms, cov)
    v1, per1 = true_vmax(sel, d1, np.full(L, float(HISTORY_SHOTS)))
    d2, n2, solver, fixed = _a2(sel, cov, arms, budget)
    v2, per2 = true_vmax(sel, d2, n2)
    for mode, v, per in (("A1", v1, per1), ("A2", v2, per2)):
        rows.append({"state": state_id, "rep": rep, "N": None, "arm": "TRUE", "mode": mode, "vmax": v, "v_top": max(per[j] for j in top) if per else 0.0,
                     "solver": solver if mode == "A2" else None})
    # the fixed reference reconstruction at the same budgets, as a floor of what splitting can do
    fixed_designs = {j: sel.x0[j] for j in arms}
    rows.append({"state": state_id, "rep": rep, "N": None, "arm": "FIXED_B0", "mode": "A1",
                 "vmax": max(sel.true_variances(fixed_designs, np.full(L, float(HISTORY_SHOTS))).values(), default=0.0)})
    rows.append({"state": state_id, "rep": rep, "N": None, "arm": "meta", "mode": "meta", "tau_neutral": 1.0,
                 "x_plus": arm_inputs(sel, "CLASSICAL", record, {"const": 1.0})[2].get("x_plus"), "x_exact": abs(sel.energy_c - sel.energy_q) / abs(sel.energy_q),
                 "E_hat": record["E_q"], "r_E": record["r_q"], "E_c": sel.energy_c, "E_q": sel.energy_q,
                 "top_gradients": [float(abs(sel.truth[j])) for j in order[:3]]})
    return rows


# --- Test B ----------------------------------------------------------------------------------------------------------------------------


def split_source(source: str) -> tuple[str, float]:
    """``(base source, restart ratio)``: a suffix ``_NR`` (never open a new block after an elimination), ``_H50`` / ``_H25`` (new block only when the
    requested set of the block has shrunk to 50% / 25%) selects the restart policy of the arm; no suffix is the protocol as specified (restart after
    every elimination)."""
    for suffix, ratio in RESTART_SUFFIX.items():
        if source.endswith(suffix):
            return source[:-len(suffix)], ratio
    return source, 1.0


def _testb_task(args) -> dict:
    state_id, rep, full_source, calibration = args
    source, restart_ratio = split_source(full_source)
    sel = _SEL[state_id]
    record = make_record(sel, rng_of("B-pre", state_id, rep))
    kind = "CLASSICAL" if source.startswith("CLASSICAL") else source  # CLASSICAL_INF: the static classical prior (trust infinity, never updated)
    if source == "NEUTRAL":
        params = None
    elif source == "CLASSICAL_INF":
        params = {"const": math.inf}
    elif source.startswith("CLASSICAL_T"):  # constant trust given in the name: CLASSICAL_T1000 is trust 1000 (equivalent outcomes per context)
        params = {"const": float(source[len("CLASSICAL_T"):])}
    elif calibration.get("v2"):  # second round: constant trust (classical 128; previous-state: the full 128-outcome history)
        params = {"const": 128.0 if source == "CLASSICAL" else math.inf}
    else:
        params = frozen(calibration, sel.family, source, "energy")
    design, eliminate = "program", True
    if source in FIXED_SOURCES:  # gate G2: FC_SE and FC_FULL, frozen reconstruction with the measured-covariance allocation (neutral start)
        prior, tau, info = NeutralMoments(sel.dim), 1.0, {"tau": 1.0, "arm": source}
        design, eliminate = "fixed", source != "FC_FULL"
    else:
        prior, tau, info = arm_inputs(sel, kind, record, params)
    started = time.perf_counter()
    result = run_cp(sel, rng_of("B-cur", state_id, full_source, rep), prior, tau, cap=calibration.get("cap", 1e7), batch_fix=calibration.get("batch_fix", False),
                     rho=float(calibration.get("rho", 0.0)), design=design, eliminate=eliminate, restart_ratio=restart_ratio)
    result["batch_fix"] = bool(calibration.get("batch_fix", False))
    result.update(state=state_id, rep=rep, source=full_source, info=info, total_seconds=round(time.perf_counter() - started, 2))
    return result


def run_jobs(function, tasks, workers: int, path: Path, key) -> int:
    done = set()
    if path.exists():
        done = {key(json.loads(line)) for line in path.read_text().splitlines() if line.strip()}
    todo = [t for t in tasks if key_of_task(t) not in done]
    if not todo:
        return 0
    with open(path, "a") as handle, ProcessPoolExecutor(max(1, workers), mp_context=mp.get_context("fork")) as pool:
        for future in as_completed([pool.submit(function, t) for t in todo]):
            result = future.result()
            for row in (result if isinstance(result, list) else [result]):
                handle.write(json.dumps(row, default=float) + "\n")
            handle.flush()
    return len(todo)


def key_of_task(task):
    return tuple(t for t in task[:3] if not isinstance(t, dict))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "testa", "testb"))
    parser.add_argument("--states", nargs="+", default=list(STATES))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--cal-reps", type=int, default=3)
    parser.add_argument("--v2", action="store_true", help="second round: constant-trust arms (no calibration file), trust sweep in Test A")
    parser.add_argument("--exact-energy", action="store_true", help="calibrate / test the energy rule with the exact relative energy discrepancy (evaluator variant)")
    parser.add_argument("--alt-fallback", action="store_true", help="replace a failed conic solve by the alternating solver instead of the frozen reconstruction")
    parser.add_argument("--sweeps", type=int, default=2, help="sweeps of the alternating design solver (programs too large for the conic solver)")
    parser.add_argument("--batch-fix", action="store_true", help="Test B first-checkpoint cap max(4096, 64 L_active) instead of 4096 (keeps the allocation when L > 128)")
    parser.add_argument("--cap", type=float, default=1e7, help="Test B shot cap per trial")
    parser.add_argument("--rho", type=float, default=0.0, help="Test B: stop when the leader is certified rho-good (0 = strict identification of the largest gradient)")
    parser.add_argument("--wide", action="store_true", help="wider grids and constants up to the static prior (in-family calibration)")
    parser.add_argument("--groups", nargs="+", default=None, metavar="NAME=STATE,STATE",
                        help="in-family calibration: fit each named group on its own states (in-sample) instead of leaving a family out")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solve-seconds", type=float, default=60.0)
    parser.add_argument("--calibration", type=Path, default=Path("runs/cp_order/calibration.json"))
    parser.add_argument("--sources", nargs="+", default=["NEUTRAL", "CLASSICAL", "PREVIOUS"])
    parser.add_argument("--out", type=Path, default=Path("runs/cp_order"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate":
        calibrate(args)
        return
    calibration = ({"v2": True, "cap": args.cap} if args.v2 else
                   {**json.loads(args.calibration.read_text()), "cap": args.cap, "xe": args.exact_energy})
    calibration["batch_fix"] = args.batch_fix
    calibration["rho"] = args.rho
    load_selectors(args.states, args.solve_seconds)
    for selector in _SEL.values():
        selector.sweeps = args.sweeps
        selector.alternating_fallback = args.alt_fallback
    reps = range(args.rep_start, args.rep_start + args.reps)
    for state_id in args.states:
        if args.command == "testa":
            path = args.out / "testa" / f"{state_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            finished = set()
            if path.exists():
                finished = {json.loads(line)["rep"] for line in path.read_text().splitlines() if line.strip()}
            tasks = [(state_id, r, calibration) for r in reps if r not in finished]
            began = time.perf_counter()
            with open(path, "a") as handle, ProcessPoolExecutor(max(1, args.workers), mp_context=mp.get_context("fork")) as pool:
                for future in as_completed([pool.submit(_testa_task, t) for t in tasks]):
                    for row in future.result():
                        handle.write(json.dumps(row, default=float) + "\n")
                    handle.flush()
            print(f"{state_id}: test A, {len(tasks)} replicates ({time.perf_counter() - began:.0f}s)", flush=True)
        else:
            sources = [s for s in args.sources if not (s == "PREVIOUS" and not _SEL[state_id].has_previous)]
            for source in sources:
                path = args.out / "testb" / f"{state_id}__{source}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                finished = set()
                if path.exists():
                    finished = {json.loads(line)["rep"] for line in path.read_text().splitlines() if line.strip()}
                tasks = [(state_id, r, source, calibration) for r in reps if r not in finished]
                began = time.perf_counter()
                with open(path, "a") as handle, ProcessPoolExecutor(max(1, args.workers), mp_context=mp.get_context("fork")) as pool:
                    for future in as_completed([pool.submit(_testb_task, t) for t in tasks]):
                        handle.write(json.dumps(future.result(), default=float) + "\n")
                        handle.flush()
                print(f"{state_id} {source}: test B, {len(tasks)} replicates ({time.perf_counter() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
