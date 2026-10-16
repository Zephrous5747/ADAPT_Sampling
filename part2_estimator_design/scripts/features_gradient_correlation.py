#!/usr/bin/env python3
"""Feature study: correlation of the gradient estimates (shared measurements) against energy lowering.

All gradients are estimated from the same shared fully commuting contexts, so their
estimates are correlated.  For every state (HF, CISD and every state of the exact ADAPT
trajectory) we compute, from the exact one-shot covariances of the state,

* the covariance matrix of the gradient estimates for the shared design
  (II-0: fixed fragments; II-A: oracle coefficient splitting), with the shots of one
  allocation over all non-zero arms (the correlation does not depend on the total);
* for the leader (largest |g|) and every challenger j: the correlation of the two
  *absolute* gradient estimates, r = s_l s_j Cov(g_l, g_j) / (sd_l sd_j), and
  chi = Var(s_l g_l - s_j g_j) / (Var g_l + Var g_j) = 1 - 2 r sd_l sd_j / (Var_l + Var_j):
  chi < 1 means the shared data make the leader-challenger comparison cheaper
  (shots for a fixed confidence scale with chi);
* the exact energy lowering of a one-parameter move with the leader and with j, and the
  share of the leader's lowering that is lost if j is chosen instead.

Output: ``runs/features/gradient_correlation_pairs.csv`` (one row per state and challenger).

    OMP_NUM_THREADS=1 python scripts/features_gradient_correlation.py --cases H4_square_eq_side1p0_HF
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import part1_bridge  # noqa: E402,F401
from allocation import allocate  # noqa: E402
from contexts import build_context_library  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from trajectory import CHEMICAL_ACCURACY, AdaptSystem, problem_at, run_adapt  # noqa: E402

from features_energy_lowering import GEOMETRIES, cisd_vector, one_parameter_lowering  # noqa: E402

NONZERO = 1e-8


def arm_blocks(p) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Context -> (Pauli ids, coefficients) of one arm's design."""
    out = {}
    for k, alpha in enumerate(p.ctx_ids):
        lo, hi = p.ctx_ptr[k], p.ctx_ptr[k + 1]
        keep = np.abs(p.x[lo:hi]) > 0
        if keep.any():
            out[int(alpha)] = (p.coord_pauli[lo:hi][keep], p.x[lo:hi][keep])
    return out


def covariance_matrix(design: DesignSet, arms: list[int], moments, shots: np.ndarray, lead: int) -> np.ndarray:
    """Variances of every arm and covariances of the leader with every arm (other entries left zero)."""
    blocks = [arm_blocks(design.problems[i]) for i in arms]
    n = len(arms)
    cov = np.zeros((n, n))
    pairs = [(a, a) for a in range(n)] + [(lead, b) for b in range(n) if b != lead]
    for a, b in pairs:
        if True:
            total = 0.0
            for alpha in set(blocks[a]) & set(blocks[b]):
                if shots[alpha] <= 0:
                    continue
                pa, xa = blocks[a][alpha]
                pb, xb = blocks[b][alpha]
                ids = np.concatenate([pa, pb])
                C = moments.covariance(alpha, ids)
                total += float(xa @ C[: pa.size, pa.size:] @ xb) / shots[alpha]
            cov[a, b] = cov[b, a] = total
    return cov


def pair_rows(label, kind, k, problem, design, arms, moments, gain, level):
    sig = design.sigmas(arms)
    shots = allocate(sig, 1.0)
    # contexts whose fragments have (numerically) zero variance get ~0 shots; floor them so that
    # 0/0 round-off cannot dominate (their contribution is then ~1e-17 / 1e-9 of the largest)
    shots = np.maximum(shots, 1e-9 * shots.max())
    g = problem.gradients[arms]
    absg = np.abs(g)
    lead = int(np.argmax(absg))
    cov = covariance_matrix(design, arms, moments, shots, lead)
    s = np.sign(g)
    rows = []
    for j in range(len(arms)):
        if j == lead:
            continue
        vl, vj = cov[lead, lead], cov[j, j]
        scale = max(np.max(np.diag(cov)), 1e-300)
        if min(vl, vj) <= 1e-10 * scale:  # an estimate with (numerically) no variance: r is undefined
            r = chi = np.nan
        else:
            r = s[lead] * s[j] * cov[lead, j] / np.sqrt(vl * vj)
            chi = (vl + vj - 2 * s[lead] * s[j] * cov[lead, j]) / (vl + vj)
        gl, gj = gain[arms[lead]], gain[arms[j]]
        rows.append({"system": label, "state": kind, "k": "" if k is None else k, "design": level,
                     "leader": arms[lead], "challenger": arms[j],
                     "challenger_rank": int(np.sum(absg > absg[j] + 1e-12)) + 1,
                     "abs_g_leader": absg[lead], "abs_g_challenger": absg[j],
                     "gap_rel": (absg[lead] - absg[j]) / absg[lead],
                     "corr_abs": r, "chi": chi, "var_leader": vl, "var_challenger": vj,
                     "gain_leader_mHa": 1e3 * gl, "gain_challenger_mHa": 1e3 * gj,
                     "energy_loss_rel": (gl - gj) / gl if gl > 0 else np.nan,
                     "total_shots_unit_eps": float(shots.sum())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        default=["H4_square_eq_side1p0_HF", "H4_square_stretch_side2p0_HF", "LiH_R3p0_HF"])
    parser.add_argument("--levels", nargs="+", default=["II-0", "II-A"])
    parser.add_argument("--states", nargs="+", default=["HF", "CISD", "ADAPT"])
    parser.add_argument("--name", default="gradient_correlation_pairs.csv")
    parser.add_argument("--ks", nargs="*", type=int, default=None,
                        help="only these ADAPT steps (with --states ADAPT); lets one geometry run as several jobs")
    parser.add_argument("--out", type=Path, default=Path("runs/features"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / args.name
    rows = []
    for case in args.cases:
        label, cisd_case = GEOMETRIES[case]
        clock = time.perf_counter()
        system = AdaptSystem.build(case)
        library = build_context_library(system.problem, "mass")
        states = [("HF", None, system.hf)]
        if cisd_case is not None:
            from features_energy_lowering import light_system

            states.append(("CISD", None, cisd_vector(light_system(case), cisd_case)))
        exact = run_adapt(system, tolerance=CHEMICAL_ACCURACY)
        states += [("ADAPT", s.iteration, system.state(s.ops, s.params)) for s in exact[1:-1]]
        for kind, k, psi in states:
            if kind not in args.states:
                continue
            if kind == "ADAPT" and args.ks is not None and k not in args.ks:
                continue
            full = system.full_state(psi)
            problem = problem_at(system.problem, full, f"{case}_{kind}{k or ''}", energy=system.energy(psi),
                                 sector_gradients=system.gradients(psi))
            arms = [int(i) for i in np.flatnonzero(np.abs(problem.gradients) > NONZERO)]
            if len(arms) < 2:
                continue
            lowering, _ = one_parameter_lowering(system, psi)
            gain = -lowering
            moments = OracleMoments(library, full)
            for level in args.levels:
                design = DesignSet(build_fragment_problems(problem, library, moments, level), library.n_contexts)
                if level != "II-0":
                    design.solve(arms, 1.0)
                rows += pair_rows(label, kind, k, problem, design, arms, moments, gain, level)
        print(f"{label}: {len(states)} states ({time.perf_counter() - clock:.0f}s)", flush=True)
        _write(path, rows)


def _write(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
