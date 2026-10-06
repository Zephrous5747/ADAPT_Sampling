"""The Phase 3 record of one complete generator-selection decision.

The work order (Required outputs) asks for every benchmark and method:

(i)    Hamiltonian, generator labels and the coefficient matrix ``A``;
(ii)   required and auxiliary Pauli libraries with their generation rule;
(iii)  the context library with each context's Clifford diagonalisation;
(iv)   the designs ``C`` / ``B`` of every redesign round;
(v)    ``||A - BC||`` with a hard failure beyond tolerance;
(vi)   raw bitstrings or sufficient statistics of every context;
(vii)  empirical / prior / shrinkage covariance matrices;
(viii) shot allocations and all batchwise redesign decisions;
(ix)   estimates, intervals, active sets, eliminations and the selection;
(x)    context-shots, circuit cost and classical time;
(xi)   oracle references in separate validation files.

:func:`write_static` writes (i)-(iii) once per case.  A :class:`RunLog` collects
(iv)-(x) from :meth:`learning.LearnedM3.run` and writes them with
:meth:`RunLog.save`; the oracle part goes to its own ``*_validation.json``.  The
sufficient statistic of a context is its outcome histogram, which is saved per
fold instead of the bitstrings (the order of shots carries no information here).
(v) is enforced inside the run itself, for every designed row at every refit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_static(out: Path, problem, library, *, hamiltonian_terms: dict | None = None) -> dict:
    """(i)-(iii): the problem, the library and the contexts of one case."""
    from part1_bridge import gradient_matrix

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    A = gradient_matrix(problem, library.n_library)
    sp.save_npz(out / "A.npz", A.tocsr())
    (out / "generators.json").write_text(json.dumps(
        {"labels": problem.labels, "kinds": problem.kinds}, indent=1))
    if hamiltonian_terms is not None:
        (out / "hamiltonian.json").write_text(json.dumps(
            {k: [float(np.real(v)), float(np.imag(v))] for k, v in hamiltonian_terms.items()}))
    multiplicity = library.multiplicity()
    (out / "library.json").write_text(json.dumps({
        "labels": library.labels,
        "n_required": library.n_required,
        "home_context": library.home.tolist(),
        "multiplicity": multiplicity.tolist(),
        "auxiliary_rule": "none: the learned designs use II-0 / II-A coordinates only; II-B "
                          "auxiliaries (Step 3) are library Paulis of other gradients measured "
                          "by >= 2 contexts carrying the generator",
        "completion_strategy": library.strategy,
    }))
    contexts = []
    for c in library.contexts:
        contexts.append({
            "index": c.index,
            "home_members": c.home_members.tolist(),
            "generators": c.generators.tolist(),
            "circuit": [list(map(_jsonable, gate)) for gate in c.circuit.gates],
            "two_qubit_gates": int(c.circuit.two_qubit_count),
            "members": c.members.tolist(),
            "member_zmask": c.member_zmask.tolist(),
            "member_sign": c.member_sign.tolist(),
        })
    (out / "contexts.json").write_text(json.dumps({
        "n_qubits": library.n_qubits,
        "encoding": "generators are packed symplectic vectors (symplectic.pack); the circuit "
                    "maps member P to sign * Z**zmask (bit q of zmask = qubit n-1-q)",
        "contexts": contexts,
    }))
    return {"A_shape": list(A.shape), "n_contexts": library.n_contexts, "n_library": library.n_library}


class RunLog:
    """Per-round and per-refit record of one run of :class:`learning.LearnedM3`."""

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self.refits: list[dict] = []
        self._designs: dict[str, np.ndarray] = {}
        self._final: dict[str, np.ndarray] = {}
        self.summary: dict = {}
        self.validation: dict = {}

    # --- hooks called by LearnedM3.run --------------------------------------------

    def round(self, learner, *, rounds, radius, epsilon, active, added, spent, estimates,
              covariance, decision, contrasts, refitted, seconds) -> None:
        grew, fold0, fold1 = added
        sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        measured = spent > 0
        self.rounds.append(_jsonable({
            "round": rounds,
            "radius": radius,
            "epsilon": epsilon,
            "active": list(active),
            "shots_added": {int(a): [int(x), int(y)] for a, x, y in zip(grew, fold0, fold1)},
            "cumulative_shots": int(spent.sum()),
            "refit": refitted,
            "estimates": {int(a): float(estimates[a]) for a in active},
            "sd": {int(a): float(v) for a, v in zip(active, sd)},
            "lower": {int(a): float(v) for a, v in zip(active, decision.lower)},
            "upper": {int(a): float(v) for a, v in zip(active, decision.upper)},
            "sign_resolved": {int(a): bool(v) for a, v in zip(active, decision.resolved)},
            "leader": decision.leader,
            "eliminated": [{"arm": a, "by": b, "how": h} for a, b, h in decision.eliminated],
            "contrasts": [{"lead": l, "other": i, "t": t, "estimate": v, "sd": float(np.sqrt(max(var, 0.0)))}
                          for (l, i, t), (v, var) in (contrasts or {}).items()],
            "contexts_measured": int(measured.sum()),
            "two_qubit_gates_of_measured_contexts": int(learner.cz[measured].sum()),
            "two_qubit_gates_times_shots": int((spent * learner.cz).sum()),
            "classical_seconds": seconds,
        }))

    def refit(self, learner, epoch: int, rounds: int, model, designs) -> None:
        """(iv) the fold designs as sparse ``C`` rows, (vii) covariance diagnostics, (viii)."""
        shots = [model.shots(0), model.shots(1)]
        for f in (0, 1):
            rows, ctx, pauli, x = [], [], [], []
            for arm, (c, p, v) in enumerate(designs[f]):
                rows.append(np.full(c.size, arm))
                ctx.append(c)
                pauli.append(p)
                x.append(v)
            prefix = f"refit{epoch:02d}_fold{f}"
            self._designs[f"{prefix}_row"] = np.concatenate(rows).astype(np.int32)
            self._designs[f"{prefix}_ctx"] = np.concatenate(ctx).astype(np.int32)
            self._designs[f"{prefix}_pauli"] = np.concatenate(pauli).astype(np.int32)
            self._designs[f"{prefix}_x"] = np.concatenate(x)
        learnable = np.minimum(shots[0], shots[1]) >= learner.config.min_fold_shots
        weights = None
        if model.nu is not None and not np.isinf(model.nu):
            total = shots[0] + shots[1]
            weights = model.nu / (model.nu + total)
        split = [sum(np.unique(d[1]).size < d[1].size for d in designs[f]) for f in (0, 1)]
        self.refits.append(_jsonable({
            "epoch": epoch,
            "round": rounds,
            "cumulative_shots": int(shots[0].sum() + shots[1].sum()),
            "learnable_contexts": int(learnable.sum()),
            "designs_kept_by_guard": learner.guard_kept,
            "arms_with_split_coefficients": split,
            "mean_prior_weight": None if weights is None else float(weights[learnable].mean()) if learnable.any() else None,
        }))

    def final(self, learner, model, spent, selected, extra) -> None:
        """(vi) outcome histograms per fold; (vii) covariance matrices of the most used contexts."""
        used = np.flatnonzero(spent > 0)
        self._final["counts_contexts"] = used.astype(np.int32)
        for f in (0, 1):
            self._final[f"counts_fold{f}"] = sp.csr_matrix(model.counts[f][used])
        # Covariances over the library Paulis of the heaviest contexts (all of them
        # would be large): empirical per fold, prior, and the shrinkage actually used.
        heavy = used[np.argsort(-spent[used])][:10]
        for alpha in heavy:
            paulis = np.intersect1d(learner.library.contexts[int(alpha)].members,
                                    np.arange(learner.library.n_required))
            key = f"cov_ctx{int(alpha)}"
            self._final[f"{key}_paulis"] = paulis.astype(np.int32)
            for f in (0, 1):
                self._final[f"{key}_model_fold{f}"] = model.covariance(int(alpha), paulis, f)
            if model.prior is not None and model.prior != "flat":
                self._final[f"{key}_prior"] = model.prior.covariance(int(alpha), paulis)
        self.summary = _jsonable({
            "selected": selected,
            "selected_label": learner.problem.labels[selected],
            "context_shots": int(spent.sum()),
            "rounds": len(self.rounds),
            "refits": len(self.refits),
            "config": learner.config.label,
            **{k: v for k, v in extra.items() if k not in ("shortfall", "miscovered_rounds", "best_eliminated")},
        })
        truth = learner.problem.gradients
        best = int(np.argmax(np.abs(truth)))
        self.validation = _jsonable({
            "note": "oracle reference values; never used by the run",
            "exact_gradients": truth,
            "exact_best": best,
            "exact_best_label": learner.problem.labels[best],
            "selected_is_exact_best": selected == best,
            "relative_shortfall": extra["shortfall"],
            "rounds_with_an_interval_missing_the_truth": extra["miscovered_rounds"],
            "best_arm_eliminated": extra["best_eliminated"],
        })

    # --- output ------------------------------------------------------------------------

    def save(self, out: Path, stem: str) -> list[Path]:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        paths = [out / f"{stem}_rounds.jsonl", out / f"{stem}_refits.json", out / f"{stem}_designs.npz",
                 out / f"{stem}_data.npz", out / f"{stem}_summary.json", out / f"{stem}_validation.json"]
        paths[0].write_text("".join(json.dumps(r) + "\n" for r in self.rounds))
        paths[1].write_text(json.dumps(self.refits, indent=1))
        np.savez_compressed(paths[2], **self._designs)
        sparse = {k: v for k, v in self._final.items() if sp.issparse(v)}
        dense = {k: v for k, v in self._final.items() if not sp.issparse(v)}
        for k, v in sparse.items():
            dense[f"{k}_data"], dense[f"{k}_indices"], dense[f"{k}_indptr"] = v.data, v.indices, v.indptr
            dense[f"{k}_shape"] = np.array(v.shape)
        np.savez_compressed(paths[3], **dense)
        paths[4].write_text(json.dumps(self.summary, indent=1))
        paths[5].write_text(json.dumps(self.validation, indent=1))
        return paths
