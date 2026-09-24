#!/usr/bin/env python3
"""Step 0 gate: the shot-level sampler against the analytic covariance and Part I.

At a fixed allocation (Part I's M1 allocation, scaled), the gradient estimates are
drawn many times in two independent ways:

* **shot level** -- one multinomial draw of outcomes per context after its
  measurement circuit, estimates from the sample means of the measured group
  (:mod:`sampler`, Part II);
* **Gaussian surrogate** -- Part I's correlated estimator noise, exact mean and
  covariance ``sum_alpha K_alpha / n_alpha``.

The gate passes when the shot-level estimates are unbiased, their per-arm
variances and cross-arm correlations match the analytic covariance within Monte
Carlo error, and their standardised distribution is indistinguishable from the
surrogate's (two-sample Kolmogorov-Smirnov) at the scale Part I runs at.  A
low-shot scale (``--low-shot``, a few shots in the lightest context) is reported
as well, to show where the Gaussian limit starts to fail.

Example::

    python scripts/step0_validate.py --cases H4_square_eq_side1p0_HF LiH_R3p0_HF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from contexts import build_context_library  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from online import OnlineConfig, OnlineM3  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from part1_bridge import (  # noqa: E402
    DEFAULT_DELTA,
    allocate_context_shots,
    epsilon_from_radius,
    load_problem,
    z_from_delta,
)
from sampler import OracleMoments, walsh_hadamard  # noqa: E402


def draw_shot_level(model: OnlineM3, shots: np.ndarray, rng, replicates: int) -> np.ndarray:
    carrying = np.unique(model._ctx)
    out = np.empty((replicates, model.n_arms))
    P = model.distributions[carrying]
    n = shots[carrying]
    for r in range(replicates):
        counts = rng.multinomial(n, P)
        means = np.zeros(model.distributions.shape)
        means[carrying] = walsh_hadamard(counts) / n[:, None]
        contributions = model._weight * means[model._ctx, model._zmask]
        out[r] = np.bincount(model._gen, weights=contributions, minlength=model.n_arms)
    return out


def draw_gaussian(model: OnlineM3, shots: np.ndarray, rng, replicates: int) -> np.ndarray:
    covariance = np.einsum("a,aij->ij", np.where(shots > 0, 1.0 / np.maximum(shots, 1), 0.0), model.covariances)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    factor = vectors * np.sqrt(np.clip(values, 0.0, None))
    return model.problem.gradients + rng.standard_normal((replicates, model.n_arms)) @ factor.T


def check(case: str, replicates: int, scale: float, seed: int) -> dict:
    problem = load_problem(case)
    library = build_context_library(problem, "canonical")
    moments = OracleMoments(library, problem.evaluator.state)
    design = DesignSet(build_fragment_problems(problem, library, moments, "II-0"), library.n_contexts)
    model = OnlineM3(problem, library, moments, design, OnlineConfig(rule="pairwise"))

    z = z_from_delta(DEFAULT_DELTA, problem.n_generators)
    allocation = allocate_context_shots(model.sigmas, epsilon_from_radius(problem.top_gap() / 2, z))
    shots = np.maximum(np.ceil(allocation * scale).astype(np.int64), model._minimum)
    analytic = np.einsum("a,aij->ij", 1.0 / np.maximum(shots, 1) * (shots > 0), model.covariances)
    sd = np.sqrt(np.diag(analytic))

    rng = np.random.default_rng(seed)
    shot_level = draw_shot_level(model, shots, rng, replicates)
    gaussian = draw_gaussian(model, shots, rng, replicates)

    watched = [i for i in range(problem.n_generators) if sd[i] > 0]
    error = shot_level[:, watched] - problem.gradients[watched]
    bias_z = np.abs(error.mean(axis=0)) / (sd[watched] / np.sqrt(replicates))
    variance_ratio = error.var(axis=0, ddof=1) / sd[watched] ** 2
    empirical_corr = np.corrcoef(shot_level[:, watched], rowvar=False)
    analytic_corr = analytic[np.ix_(watched, watched)] / np.outer(sd[watched], sd[watched])
    corr_error = float(np.abs(empirical_corr - analytic_corr).max())

    order = problem.ranking()
    binding = int(np.argmax(model.squared.sum(axis=1)))
    ks = {}
    for name, i in (("winner", order[0]), ("runner_up", order[1]), ("binding", binding)):
        a = (shot_level[:, i] - problem.gradients[i]) / sd[i]
        b = (gaussian[:, i] - problem.gradients[i]) / sd[i]
        ks[name] = float(stats.ks_2samp(a, b).pvalue)
        ks[f"{name}_skew"] = float(stats.skew(a))
        ks[f"{name}_excess_kurtosis"] = float(stats.kurtosis(a))

    tolerance_var = 5.0 * np.sqrt(2.0 / replicates)
    row = {
        "case_id": case,
        "scale": scale,
        "replicates": replicates,
        "total_shots": int(shots.sum()),
        "min_shots_in_a_context": int(shots[shots > 0].min()),
        "max_bias_z": float(bias_z.max()),
        "max_variance_ratio_error": float(np.abs(variance_ratio - 1).max()),
        "variance_tolerance": tolerance_var,
        "max_correlation_error": corr_error,
        "correlation_tolerance": 5.0 / np.sqrt(replicates),
        **{f"ks_p_{k}" if not k.endswith(("skew", "kurtosis")) else k: v for k, v in ks.items()},
    }
    row["passed"] = bool(
        row["max_bias_z"] < 5.0
        and row["max_variance_ratio_error"] < tolerance_var
        and corr_error < row["correlation_tolerance"]
        and min(ks["winner"], ks["runner_up"], ks["binding"]) > 1e-3
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=[
        "H4_square_eq_side1p0_HF", "H4_square_eq_side1p0_CISD",
        "H4_square_stretch_side2p0_HF", "H4_square_stretch_side2p0_CISD", "LiH_R3p0_HF"])
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low-shot", type=float, default=None,
                        help="also run at this fraction of the M1 allocation")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        scales = [1.0] + ([args.low_shot] if args.low_shot else [])
        for scale in scales:
            row = check(case, args.replicates, scale, args.seed)
            rows.append(row)
            print(
                f"{case:32s} scale {scale:<8g} shots {row['total_shots']:>12,}  "
                f"bias_z {row['max_bias_z']:5.2f}  var err {row['max_variance_ratio_error']:.3f} "
                f"(tol {row['variance_tolerance']:.3f})  corr err {row['max_correlation_error']:.3f} "
                f"(tol {row['correlation_tolerance']:.3f})  KS p winner {row['ks_p_winner']:.2f} "
                f"binding {row['ks_p_binding']:.2f}  {'PASS' if row['passed'] else 'FAIL'}",
                flush=True,
            )
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out / "step0_sampler_check.csv", columns, rows)
    write_json(args.out / "step0_sampler_check_meta.json", {"run": run_record()})
    if not all(r["passed"] for r in rows if r["scale"] == 1.0):
        raise SystemExit("Step 0 sampler gate failed")


if __name__ == "__main__":
    main()
