#!/usr/bin/env python3
"""ADAPT trajectories under selection policies with *injected* selection errors (exact gradients, no shots).

Why.  The question of Paper A is not whether the selection finds the largest gradient, but what a worse choice does to the
energy trajectory (steps to chemical accuracy, optimiser evaluations, final energy).  A measured selection makes few errors
and most of them are ties or rho-good, so the measured runs cannot show where the trajectory starts to suffer.  Here the
error is put in on purpose and its size is the dose:

* ``exact``            argmax |g| (ties to the lowest index): exact ADAPT, the reference of all costs
* ``rho_random:R``     uniformly among the generators with |g| >= (1 - R) max|g|  (R = 0: only tied generators)
* ``rho_worst_g:R``    the smallest |g| inside that set (adversarial in the gradient)
* ``rho_worst_e:R``    the smallest one-parameter energy lowering inside that set (adversarial in the energy)
* ``tau_random:T``     uniformly among the generators whose one-parameter lowering is >= (1 - T) of the best
                       (a tolerance on the energy instead of on the gradient)
* ``onep_greedy``      the largest one-parameter lowering (energy-greedy, no re-optimisation)
* ``reopt_top:M``      among the M largest |g|, the generator with the lowest energy after re-optimising every parameter
                       (energy-greedy reference; ``reopt_all``: every generator with a gradient)
* ``newton_exact``     the largest second-order estimate g_i^2 / (2 h_i) of the lowering, h_i = d2E/dtheta2 at the current state
                       (curvature measured exactly: an oracle)
* ``newton_hf``        the same with the curvature h_i taken once at the Hartree-Fock reference (a basis state: every Pauli
                       expectation value is known classically, so the measured gradients are the only shots)
* ``nrho_random:R``    uniformly among the generators whose curvature-scaled gradient |g_i| / sqrt(h_i) (h_i at the HF reference)
                       is within R of the largest: the rho-good set of the selection run on classically rescaled arms
* ``nrho_worst_e:R``   the smallest one-parameter lowering inside that set (adversarial in the energy)

Every trajectory records, per step, the optimiser evaluations, the parameters, the energy error, the gradient shortfall
1 - |g_sel|/max|g| of the choice and the ratio of its one-parameter lowering to the best one, so that the harm of a
choice can be read against its error.  Trajectory ``i`` uses the random stream ``SeedSequence(seed).spawn(n)[i]`` for every
policy, so policies are compared on common random numbers.

    python scripts/energy_trajectory_policies.py --case H4_square_eq_side1p0_HF --trajectories 200 --workers 8 \\
        --policies exact rho_random:0 rho_random:0.1 rho_random:0.3 rho_worst_e:0.1 onep_greedy reopt_top:5

Output: ``runs/energy_policies/<case>/policy_<spec>.jsonl`` (one line per trajectory; the run resumes from these files).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from parallel import run_trials, slug  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem  # noqa: E402

from features_energy_lowering import light_system, one_parameter_lowering  # noqa: E402

NONZERO = 1e-8       # a generator "has a gradient" above this (as in the feature study)
TIE = 1e-9           # relative tolerance for tied gradients
GRADIENT_STOP = 1e-6  # ADAPT stalls below this (as trajectory.run_adapt)
ENERGY_STALL = False  # with a loose optimiser tolerance: also stall when three steps in a row gain < 1e-6 Ha in total (see set_optimizer_tolerance)
DETERMINISTIC = {"exact", "onep_greedy", "reopt_top", "reopt_all", "newton_exact", "newton_hf"}


@dataclass
class Policy:
    spec: str
    name: str
    param: float | None

    @classmethod
    def parse(cls, spec: str) -> "Policy":
        name, _, param = spec.partition(":")
        if name not in {"exact", "rho_random", "rho_worst_g", "rho_worst_e", "tau_random", "onep_greedy",
                        "reopt_top", "reopt_all", "newton_exact", "newton_hf", "nrho_random", "nrho_worst_e"}:
            raise ValueError(f"unknown policy {spec!r}")
        return cls(spec, name, float(param) if param else None)

    @property
    def deterministic(self) -> bool:
        return self.name in DETERMINISTIC

    @property
    def needs_lowering(self) -> bool:
        return self.name in {"rho_worst_e", "tau_random", "onep_greedy", "nrho_worst_e"}


def set_optimizer_tolerance(gtol: float) -> None:
    """Use BFGS with this gradient tolerance (max-norm) for every re-optimisation, the exact reference included.  The default of
    ``AdaptSystem.optimise`` is 1e-9, far below what a sampled gradient can resolve (a parameter-shift gradient from shots has
    a noise floor of the order of 1e-3 Ha/rad); the evaluation counts depend strongly on it."""
    from scipy.optimize import minimize

    import trajectory

    def optimise(self, ops, start):
        result = minimize(self._energy_and_gradient, start, args=(ops,), jac=True, method="BFGS",
                          options={"gtol": gtol, "maxiter": 2000})
        self.last_evaluations = int(result.nfev)
        return np.asarray(result.x), float(result.fun)

    AdaptSystem.optimise = optimise
    # ADAPT stalls when every |g_i| is below a threshold; the residual gradients of generators that are zero by symmetry are of the order of the
    # optimiser's tolerance, so the threshold has to move with it (otherwise stretched H4 keeps adding generators for 30 more steps)
    global GRADIENT_STOP, ENERGY_STALL
    GRADIENT_STOP = 1e-6
    ENERGY_STALL = gtol > 1e-7
    trajectory.run_adapt.__kwdefaults__["gradient_tolerance"] = GRADIENT_STOP


def curvature(system: AdaptSystem, psi: np.ndarray) -> np.ndarray:
    """h_i = d2E/dtheta2 at theta = 0 of exp(theta G_i) psi for every generator: 2 <psi|H G^2|psi> + 2 <G psi|H|G psi>."""
    h_psi = system.hamiltonian @ psi
    out = np.zeros(len(system.generators))
    for i, g in enumerate(system.generators):
        v1 = g @ psi
        v2 = g @ v1
        out[i] = 2.0 * float(h_psi @ v2) + 2.0 * float(v1 @ (system.hamiltonian @ v1))
    return out


_HF_CURVATURE: dict = {}


def newton_gain(g: np.ndarray, h: np.ndarray) -> np.ndarray:
    """g^2 / (2 h) where the curvature is positive; |g| scaled by the median positive curvature elsewhere (no estimate)."""
    pos = h > 1e-12
    scale = np.median(h[pos]) if pos.any() else 1.0
    return np.where(pos, g * g / (2.0 * np.where(pos, h, 1.0)), g * g / (2.0 * scale))


def choose(policy: Policy, system: AdaptSystem, ops: list[int], params: np.ndarray, g: np.ndarray, gain: np.ndarray,
           rng: np.random.Generator, psi: np.ndarray | None = None) -> int:
    absg = np.abs(g)
    live = absg > NONZERO
    gmax = absg.max()
    name = policy.name
    if name == "exact":
        return int(np.argmax(absg))
    if name in {"rho_random", "rho_worst_g", "rho_worst_e"}:
        rho = policy.param or 0.0
        cand = np.flatnonzero(live & (absg >= (1.0 - max(rho, TIE)) * gmax))
        if name == "rho_random":
            return int(rng.choice(cand))
        key = absg[cand] if name == "rho_worst_g" else gain[cand]
        worst = cand[key <= key.min() + TIE * max(abs(key.min()), 1e-30)]
        return int(rng.choice(worst))
    if name == "tau_random":
        tau = policy.param or 0.0
        best = gain[live].max()
        cand = np.flatnonzero(live & (gain >= (1.0 - max(tau, TIE)) * best))
        return int(rng.choice(cand))
    if name == "onep_greedy":
        masked = np.where(live, gain, -np.inf)
        return int(np.argmax(masked))
    if name in {"nrho_random", "nrho_worst_e"}:
        key = id(system)
        if key not in _HF_CURVATURE:
            _HF_CURVATURE[key] = curvature(system, system.hf)
        h = _HF_CURVATURE[key]
        pos = h > 1e-12
        scale = np.median(h[pos]) if pos.any() else 1.0
        scaled = absg / np.sqrt(np.where(pos, h, scale))
        smax = scaled[live].max()
        rho = policy.param or 0.0
        cand = np.flatnonzero(live & (scaled >= (1.0 - max(rho, TIE)) * smax))
        if name == "nrho_random":
            return int(rng.choice(cand))
        worst = cand[gain[cand] <= gain[cand].min() + TIE * max(abs(gain[cand].min()), 1e-30)]
        return int(rng.choice(worst))
    if name in {"newton_exact", "newton_hf"}:
        if name == "newton_hf":
            key = id(system)
            if key not in _HF_CURVATURE:
                _HF_CURVATURE[key] = curvature(system, system.hf)
            h = _HF_CURVATURE[key]
        else:
            h = curvature(system, psi)
        est = np.where(live, newton_gain(g, h), -np.inf)
        return int(np.argmax(est))
    if name in {"reopt_top", "reopt_all"}:
        order = [int(i) for i in np.argsort(-absg, kind="stable") if live[i]]
        if name == "reopt_top":
            order = order[: int(policy.param or 1)]
        best, best_e = order[0], np.inf
        for i in order:
            _, energy = system.optimise(list(ops) + [i], np.append(params, 0.0))
            if energy < best_e - 1e-14:
                best, best_e = i, energy
        return best
    raise AssertionError(name)


def adapt_with_policy(system: AdaptSystem, policy: Policy, rng: np.random.Generator, *, tolerance: float,
                      max_iterations: int, diagnose: bool = True) -> dict:
    """Same loop as ``trajectory.run_adapt`` with the choice made by ``policy``."""
    ops: list[int] = []
    params = np.zeros(0)
    errors, evaluations, n_params, labels, shortfall, lowering_ratio = [], [], [], [], [], []
    started = time.perf_counter()
    stalled = False
    for k in range(max_iterations + 1):
        psi = system.state(ops, params)
        energy = system.energy(psi)
        g = system.gradients(psi)
        errors.append(energy - system.fci_energy)
        if errors[-1] < tolerance or k == max_iterations:
            break
        if np.abs(g).max() < GRADIENT_STOP:
            stalled = True
            break
        if ENERGY_STALL and len(errors) >= 4 and errors[-4] - errors[-1] < 3e-6:
            stalled = True  # the symmetry-forbidden generators of a loosely optimised state keep a residual gradient: stop on the energy instead
            break
        gain = None
        if diagnose or policy.needs_lowering:
            lowering, _ = one_parameter_lowering(system, psi)
            gain = -lowering
        arm = choose(policy, system, ops, params, g, gain, rng, psi)
        absg = np.abs(g)
        shortfall.append(float(1.0 - absg[arm] / absg.max()))
        if gain is not None:
            live = absg > NONZERO
            best = float(gain[live].max())
            lowering_ratio.append(float(gain[arm] / best) if best > 0 else 1.0)
        ops.append(arm)
        labels.append(system.labels[arm])
        params, _ = system.optimise(ops, np.append(params, 0.0))
        evaluations.append(int(system.last_evaluations))
        n_params.append(len(ops))
    return {"policy": policy.spec, "iterations": len(ops), "final_error": errors[-1],
            "reached": bool(errors[-1] < tolerance), "stalled": stalled,
            "arms": [int(a) for a in ops], "evaluations": evaluations, "n_params": n_params,
            "evals_total": int(sum(evaluations)),
            "evals_weighted": float(sum(e * (1 + 2 * n) for e, n in zip(evaluations, n_params))),
            "energy_errors": errors, "selected": labels, "g_shortfall": shortfall,
            "lowering_ratio": lowering_ratio, "seconds": round(time.perf_counter() - started, 2)}


def exact_reference(system: AdaptSystem, tolerance: float, max_iterations: int = 40) -> dict:
    """Exact ADAPT (largest |g|, ties to the lowest index) through the loop of the policies, with the stall rules in force."""
    r = adapt_with_policy(system, Policy.parse("exact"), np.random.default_rng(0), tolerance=tolerance, max_iterations=max_iterations,
                          diagnose=False)
    return {"iterations": r["iterations"], "final_error": r["final_error"], "energy_errors": r["energy_errors"],
            "evaluations": r["evaluations"], "n_params": r["n_params"], "added": r["selected"], "arms": r["arms"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--policies", nargs="+", default=["exact", "rho_random:0", "rho_random:0.1"])
    parser.add_argument("--trajectories", type=int, default=100, help="for the random policies")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--extra-steps", type=int, default=15, help="steps allowed beyond exact ADAPT's")
    parser.add_argument("--tolerance", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--gtol", type=float, default=None,
                        help="BFGS gradient tolerance of every re-optimisation (default: that of AdaptSystem.optimise, 1e-9)")
    parser.add_argument("--no-diagnose", action="store_true", help="skip the one-parameter lowering of every step "
                        "(only for policies that do not need it)")
    parser.add_argument("--out", type=Path, default=Path("runs/energy_policies"))
    args = parser.parse_args()

    policies = [Policy.parse(s) for s in args.policies]
    if args.gtol is not None:
        set_optimizer_tolerance(args.gtol)
    system = light_system(args.case)
    out = args.out / args.case
    out.mkdir(parents=True, exist_ok=True)
    ref = exact_reference(system, args.tolerance)
    max_iterations = ref["iterations"] + args.extra_steps
    ref["max_iterations"] = max_iterations
    (out / "exact_reference.json").write_text(json.dumps(ref, indent=1))
    print(f"{args.case}: exact ADAPT {ref['iterations']} steps to {ref['final_error'] * 1e3:.3f} mHa; policies "
          f"{', '.join(p.spec for p in policies)}", flush=True)

    for policy in policies:
        path = out / f"policy_{slug(policy.spec)}.jsonl"
        n = 1 if policy.deterministic else args.trajectories
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    done.add(int(json.loads(line)["index"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        clock = time.perf_counter()

        def one(rng, p=policy):
            return adapt_with_policy(system, p, rng, tolerance=args.tolerance, max_iterations=max_iterations,
                                     diagnose=not args.no_diagnose)

        def save(index, result, p=path):
            with p.open("a") as handle:
                handle.write(json.dumps({"index": index, **result}) + "\n")

        run_trials(one, n, args.seed, workers=args.workers, skip=done, on_result=save)
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        steps = np.array([r["iterations"] for r in rows])
        evs = np.array([r["evals_total"] for r in rows])
        print(f"{args.case} {policy.spec:18s} n={len(rows):4d} reached {np.mean([r['reached'] for r in rows]):.2f} "
              f"steps {steps.mean():6.2f} (exact {ref['iterations']}) evals {evs.mean():8.1f} "
              f"({time.perf_counter() - clock:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
