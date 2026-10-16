#!/usr/bin/env python3
"""Shot reuse and the evolution of a selection (Paper A, questions Q3 and Q5).

For each case and configuration a few trials are run with a run log and with the
design refit instrumented.  Two outputs:

Reuse (``runs/paper_a/reuse.csv``)
    the average number of *active* candidates whose current estimator uses the
    data of one context-shot, accumulated over the whole selection,
    ``sum_r sum_alpha n_alpha^(r) |{i in A_r : alpha in supp x_i}| / sum n``,
    with ``x_i`` the design in force when round ``r``'s shots were planned (II-0:
    the parent context of every Pauli; learned levels: the last fold designs).
    Per-gradient measurement (M2) has reuse one by construction.
Evolution (``runs/paper_a/<case>_<config>_evolution.csv``, first trial)
    per round: radius, |A_r|, |B_r| (union support of the active gradients), the
    parent contexts carrying it, contexts measured, cumulative shots; per refit:
    the median relative error of the learned covariance of the true leader's
    coordinates against the exact one (validation only), the share of the
    leader's coefficient mass moved off its parent contexts, and the leader's
    variance under the learned design relative to its II-0 variance, both at the
    shots held and with exact covariances.

    python scripts/paper_a_run_diagnostics.py --cases LiH_R3p0_HF --configs "II-0 estimated" "II-A data"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from design import FragmentProblem  # noqa: E402
from learning import LearnedM3  # noqa: E402
from outputs import run_record, write_csv, write_json  # noqa: E402
from parallel import slug  # noqa: E402
from runlog import RunLog  # noqa: E402
from step4_learned_designs import CONFIGS, setup  # noqa: E402


class Instrumented:
    """Wraps ``LearnedM3._refit`` to keep each refit's designs and leader diagnostics."""

    def __init__(self, learner: LearnedM3, oracle, leader: int) -> None:
        self.learner = learner
        self.oracle = oracle
        self.leader = leader
        ctx, pauli, targets = learner.split_coords[leader]
        self.exact = FragmentProblem(leader, ctx, pauli, targets, learner.library.home, oracle.covariance)
        self.home = learner._home_x(self.exact)
        self.refits: list[dict] = []
        self.designs = None
        self._original = learner._refit
        learner._refit = self._refit

    def _refit(self, model, active):
        designs = self._original(model, active)
        self.designs = designs
        shots = [model.shots(0), model.shots(1)]
        held = (shots[0] + shots[1]).astype(float)
        learnable = np.minimum(shots[0], shots[1]) >= self.learner.config.min_fold_shots
        p = self.exact
        errors = []
        for k, alpha in enumerate(p.ctx_ids):
            if not learnable[alpha]:
                continue
            paulis = p.coord_pauli[p.ctx_ptr[k]:p.ctx_ptr[k + 1]]
            exact = self.oracle.covariance(int(alpha), paulis)
            norm = np.linalg.norm(exact)
            if norm > 1e-12:
                errors.append(np.linalg.norm(model.covariance(int(alpha), paulis, 0) - exact) / norm)
        x = np.zeros(p.n_coordinates)
        self.learner._embed(p, designs[0][self.leader], 1.0, x)
        moved = float(np.abs(x - self.home).sum() / (2 * np.abs(self.home).sum())) if self.home.any() else 0.0
        ratio = float(p.variance(held, x) / p.variance(held, self.home))
        self.refits.append({"shots": int(held.sum()), "learnable_contexts": int(learnable.sum()),
                            "leader_cov_error_median": float(np.median(errors)) if errors else np.nan,
                            "leader_mass_moved": moved, "leader_variance_ratio": ratio})
        return designs


def supports(learner) -> list[np.ndarray]:
    return [np.asarray(p.pauli_ids) for p in learner.base]


def informed_contexts(learner, designs) -> list[set[int]]:
    if designs is None:
        return [set(np.unique(p.coord_ctx).tolist()) for p in learner.base]
    return [set(np.unique(np.concatenate([designs[0][i][0], designs[1][i][0]])).tolist())
            for i in range(learner.n_arms)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "LiH_R3p0_HF"])
    parser.add_argument("--configs", nargs="+", default=["II-0 estimated", "II-A data"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    reuse_rows = []
    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, "mass")
        leader = int(np.argmax(problem.abs_gradients))
        sup = None
        for name in args.configs:
            seeds = np.random.SeedSequence(args.seed).spawn(args.trials)
            values = []
            for trial in range(args.trials):
                learner = LearnedM3(problem, library, oracle, prior, base, coords, CONFIGS[name])
                sup = sup or supports(learner)
                probe = Instrumented(learner, oracle, leader)
                log = RunLog()
                plan_designs = []  # designs in force when each round's shots were planned

                original_round = log.round

                def round_hook(lr, _probe=probe, _orig=original_round, _plans=plan_designs, **kw):
                    _plans.append(informed_contexts(lr, _probe.designs_before))
                    _probe.designs_before = _probe.designs
                    return _orig(lr, **kw)

                probe.designs_before = None
                log.round = round_hook
                outcome = learner.run(np.random.default_rng(seeds[trial]), log=log)
                informed_weight, shots = 0.0, 0.0
                evolution = []
                cumulative = 0
                for r, record in enumerate(log.rounds):
                    active = record["active"]
                    informed = plan_designs[r]
                    for alpha_text, (f0, f1) in record["shots_added"].items():
                        alpha = int(alpha_text)
                        added = f0 + f1
                        informed_weight += added * sum(alpha in informed[i] for i in active)
                        shots += added
                    cumulative = record["cumulative_shots"]
                    if trial == 0:
                        union = np.unique(np.concatenate([sup[i] for i in active]))
                        carrying = np.unique(library.home[union])
                        evolution.append({"round": record["round"], "radius": record["radius"],
                                          "active": len(active), "support": int(union.size),
                                          "parent_contexts_of_support": int(carrying.size),
                                          "contexts_receiving_shots": len(record["shots_added"]),
                                          "cumulative_shots": cumulative, "refit": record["refit"],
                                          "eliminated": len(record["eliminated"])})
                values.append(informed_weight / shots)
                if trial == 0:
                    for row, refit in zip([e for e in evolution if e["refit"]], probe.refits):
                        row.update(refit)
                    columns = list(dict.fromkeys(k for e in evolution for k in e))
                    write_csv(args.out / "paper_a" / f"{case}_{slug(name)}_evolution.csv", columns, evolution)
                print(f"{case:30s} {name:20s} trial {trial}: reuse {values[-1]:.2f} shots {outcome.shots:,.0f} "
                      f"rounds {outcome.rounds} correct {outcome.correct}", flush=True)
            reuse_rows.append({"case_id": case, "config": name, "trials": args.trials,
                               "reuse_mean": float(np.mean(values)), "reuse_min": float(np.min(values)),
                               "reuse_max": float(np.max(values)), "pool": problem.n_generators})
    write_csv(args.out / "paper_a" / "reuse.csv", list(reuse_rows[0]), reuse_rows)
    write_json(args.out / "paper_a" / "diagnostics_meta.json",
               {"trials": args.trials, "seed": args.seed, "configs": {k: CONFIGS[k].label for k in args.configs},
                "run": run_record()})


if __name__ == "__main__":
    main()
