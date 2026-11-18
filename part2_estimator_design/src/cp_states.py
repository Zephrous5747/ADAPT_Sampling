"""The six states of work order CP-1.2, their pre-selection records and the arms (sources of initial moments)."""
from __future__ import annotations

import numpy as np

from classical_prior import MixedMoments
from cp_order import (EPS_C, HISTORY_SHOTS, CPSelector, EnergyRecord, NeutralMoments, PreviousMoments, history_strength, tau_of,
                      x_upper, y_upper)
from note_protocol import NoteCovariance
from reframe import Counts

STATES = {  # id -> (target case, preceding case, family, pilot state)
    "H2_eq_m0": ("H2_R0p7414_ADAPT0", None, "H2", True),
    "H2_str_m1": ("H2_R1p5_ADAPT1", "H2_R1p5_ADAPT0", "H2", False),
    "H4_a1_m1": ("H4_square_eq_side1p0_ADAPT1", "H4_square_eq_side1p0_ADAPT0", "H4", False),
    "H4_a2_m3": ("H4_square_stretch_side2p0_ADAPT3", "H4_square_stretch_side2p0_ADAPT2", "H4", True),
    "LiH_r16_m1": ("LiH_R1p6_ADAPT1", "LiH_R1p6_ADAPT0", "LiH", False),
    "LiH_r30_m3": ("LiH_R3p0_ADAPT3", "LiH_R3p0_ADAPT2", "LiH", True),
}
FAMILIES = ("H2", "H4", "LiH")
# Siblings of a state with a degraded classical state (half of the determinants, as the ``half`` arms of note_run.py): the same quantum state and
# records, a prior that is wrong by some mHa.  Used by the in-family calibration only; they have no preceding-state source and are not in STATES,
# so the defaults of cp_order_run.py (all of STATES) are unchanged.
# Second round (2026-11-11): states with a unique largest gradient (LiH 1.6 m=1 has an exact tie and can never be certified), states whose
# sparse classical state is not exact (LiH 3.0 m=5: overlap 0.998, m=6: 0.988; water 0.9986-0.9999), and STO-3G water.
EXTRA = {
    "LiH_r16_m2": ("LiH_R1p6_ADAPT2", "LiH_R1p6_ADAPT1", "LiH", False),
    "LiH_r30_m5": ("LiH_R3p0_ADAPT5", "LiH_R3p0_ADAPT4", "LiH", False),
    "LiH_r30_m6": ("LiH_R3p0_ADAPT6", "LiH_R3p0_ADAPT5", "LiH", False),
    "H2O_eq_m11": ("H2O_eq_ADAPT11", "H2O_eq_ADAPT10", "H2O", False),
    "H2O_str_m8": ("H2O_stretch_ADAPT8", "H2O_stretch_ADAPT7", "H2O", False),
    # water trajectory stages (2026-11-12; Test A only): early / mid / late along the STO-3G trajectories (equilibrium 0..17, stretched 0..18)
    "H2O_eq_m3": ("H2O_eq_ADAPT3", "H2O_eq_ADAPT2", "H2O", False),
    "H2O_eq_m7": ("H2O_eq_ADAPT7", "H2O_eq_ADAPT6", "H2O", False),
    "H2O_eq_m16": ("H2O_eq_ADAPT16", "H2O_eq_ADAPT15", "H2O", False),
    "H2O_str_m3": ("H2O_stretch_ADAPT3", "H2O_stretch_ADAPT2", "H2O", False),
    "H2O_str_m13": ("H2O_stretch_ADAPT13", "H2O_stretch_ADAPT12", "H2O", False),
    "H2O_str_m17": ("H2O_stretch_ADAPT17", "H2O_stretch_ADAPT16", "H2O", False),
}
VARIANTS = {"LiH_r16_m1_K1": "LiH_r16_m1", "LiH_r30_m3_K2": "LiH_r30_m3", "LiH_r16_m2_K1": "LiH_r16_m2", "LiH_r30_m5_K3": "LiH_r30_m5"}
ALL_STATES = {**STATES, **EXTRA}
ALL_STATES.update({variant: ALL_STATES[base] for variant, base in VARIANTS.items()})


def build_state(state_id: str, solve_seconds: float = 60.0, cap: float = 1e7) -> CPSelector:
    from classical_prior import sparse_state_for_case
    from part1_bridge import load_problem
    from pauli_ops import PauliEvaluator
    from reuse import hamiltonian_terms
    from sampler import OracleMoments
    from step4_learned_designs import setup

    case, prev_case, family, _ = ALL_STATES[state_id]
    if state_id in VARIANTS:
        prev_case = None
    problem, library, oracle, _hf, base, coords = setup(case, "mass", "mass")

    def normalized(moments) -> np.ndarray:
        d = np.clip(moments.distributions_matrix(), 0.0, None)
        return d / d.sum(axis=1, keepdims=True)

    state, info = sparse_state_for_case(case)
    if state_id in VARIANTS:
        half = max(1, int(info["K"]) // 2)
        state, info = sparse_state_for_case(case, keep=half)
    classical = OracleMoments(library, state)
    for alpha in range(library.n_contexts):
        classical.group_means(alpha)
    selector = CPSelector(problem, library, normalized(oracle), base, coords, classical, cap=cap, solve_seconds=solve_seconds)
    selector.state_id, selector.case, selector.family, selector.info = state_id, case, family, info
    selector.oracle = oracle
    terms = hamiltonian_terms(problem, case)

    def evaluate(s) -> float:
        return float(PauliEvaluator(s).fragment_mean_std(terms)[0])

    reference = evaluate(problem.evaluator.state)
    selector.energy_q = float(problem.state_energy)
    selector.energy_c = selector.energy_q + evaluate(state) - reference
    selector.energy = EnergyRecord(problem, case, terms, selector.energy_q)
    selector.energy_dist_q = selector.energy.distributions(problem.evaluator.state)
    selector.has_previous = prev_case is not None
    if prev_case is not None:
        previous = load_problem(prev_case)
        selector.energy_prev = float(previous.state_energy)
        selector.prev_dist = normalized(OracleMoments(library, previous.evaluator.state))
        selector.energy_dist_prev = selector.energy.distributions(previous.evaluator.state)
    return selector


def make_record(selector: CPSelector, rng: np.random.Generator) -> dict:
    """The common pre-selection data of one replicate: energy records and the synthetic history of the preceding state."""
    record = {"E_q": selector.energy.sample(rng, selector.energy_dist_q), "r_q": selector.energy.radius}
    if selector.has_previous:
        record["E_prev"] = selector.energy.sample(rng, selector.energy_dist_prev)
        record["r_prev"] = selector.energy.radius
        counts = rng.multinomial(HISTORY_SHOTS, selector.prev_dist)
        record["history"] = {a: counts[a].astype(np.float64) for a in range(selector.n_contexts)}
    return record


def arm_inputs(selector: CPSelector, source: str, record: dict, params: dict | None = None):
    """``(prior, tau, info)`` of one arm; ``params``: ``{"d0", "s"}`` for the energy rule or ``{"const": value}`` for the control."""
    params = params or {}
    if source == "NEUTRAL":
        return NeutralMoments(selector.dim), 1.0, {"tau": 1.0}
    if source == "CLASSICAL":
        x_plus = x_upper(selector.energy_c, record["E_q"], record["r_q"])
        if params.get("exact"):  # evaluator-side variant of the work order: the exact relative discrepancy instead of the uncertainty-aware x_+
            x_plus = abs(selector.energy_c - selector.energy_q) / abs(selector.energy_q)
        tau = float(params["const"]) if "const" in params else tau_of(x_plus, params["d0"], params["s"])
        return MixedMoments(selector.prior, EPS_C), tau, {"tau": tau, "x_plus": x_plus, "E_c": selector.energy_c,
                                                          "E_hat": record["E_q"], "r_E": record["r_q"]}
    if source == "PREVIOUS":
        if not selector.has_previous:
            return NeutralMoments(selector.dim), 1.0, {"tau": 1.0, "fallback": "no preceding state"}
        y_plus = y_upper(record["E_q"], record["r_q"], record["E_prev"], record["r_prev"])
        if params.get("exact"):  # evaluator-side variant: the exact relative energy change between the two states
            y_plus = abs(selector.energy_q - selector.energy_prev) / abs(selector.energy_q)
        tau_e = float(params["const"]) if "const" in params else tau_of(y_plus, params["d0"], params["s"])
        tau = history_strength(tau_e, HISTORY_SHOTS)
        return PreviousMoments(record["history"], HISTORY_SHOTS, selector.dim), tau, {
            "tau": tau, "tau_E": tau_e, "y_plus": y_plus, "E_hat": record["E_q"], "E_prev_hat": record["E_prev"]}
    raise ValueError(source)


def design_covariance(selector: CPSelector, counts: Counts, prior, tau) -> NoteCovariance:
    return NoteCovariance(selector.library, counts, prior, tau)


def nested_counts(selector: CPSelector, rng: np.random.Generator, prefixes=(0, 8, 32, 128)) -> dict:
    """``{N: Counts}`` of nested prefixes of one 128-outcome stream per group (cumulative sums of independent multinomials)."""
    out = {}
    cumulative = np.zeros((selector.n_contexts, selector.dim))
    previous = 0
    everything = np.arange(selector.n_contexts)
    for N in sorted(prefixes):
        if N > previous:
            cumulative += rng.multinomial(N - previous, selector.dist)
            previous = N
        counts = Counts(selector.n_contexts)
        if N > 0:
            counts.add(everything, cumulative.copy())
        out[N] = counts
    return out
