#!/usr/bin/env python3
"""Step 5: the external baselines run the way Part II's own methods run.

Fixed state, sampled measurement outcomes, variances estimated from the shots (no
exact variance in any row except ``M1 static``, which is handed the exact gap and the
exact variances as in Part I).  Configurations:

* ``M1 static``: one allocation for the whole pool at radius ``gap / 2``, one draw;
* ``M1 seq``: shared groups, uniform precision for every gradient, shrinking radius,
  elimination switched off (``LearnedM3(elimination="off")``): the sequential,
  non-oracle form of all-gradient estimation, with II-0's rule and estimated radii;
* ``M2 ...``: independent best-arm identification (:class:`baselines.IndependentBAI`):
  ``marginal`` is Successive Elimination on ``|g|`` as in Huang and Izmaylov's
  formulation, ``pairwise``/``safe`` are the rules of II-0 on independent estimates;
* ``Ikh ...``: the shot-reuse baseline of Ikhtiarudin et al. (:mod:`reuse`): all-gradient
  estimation (no elimination) in which every Pauli measured by the energy contexts of the
  last VQE evaluation is read from them, those shots being free; ``FC`` or ``QWC`` groups.
  ``... no reuse`` rows are the same grouping without the free data;
* ``II-0 + reuse`` and ``II-A data + reuse``: Part II's methods given the same free data
  (on the same library, so II-A may split coefficients between energy and parent contexts).

The rows named ``... bound start`` use the a-priori starting radius, i.e. no exact
quantity enters the run.  II-0 and II-A rows to compare with are written by
``step4_learned_designs.py`` (``II-0 estimated``, ``II-A data``, ``II-0 safe, bound
start``, ``II-A data, safe``) into the same trial directories.

Example::

    OMP_NUM_THREADS=1 python scripts/step5_external_baselines.py --cases H4_square_eq_side1p0_CISD \\
        --trials 200 --workers 12
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import part1_bridge  # noqa: E402,F401
from baselines import IndependentBAI, IndependentConfig, IndependentContexts, StaticM1  # noqa: E402
from design import DesignSet, build_fragment_problems  # noqa: E402
from learning import LearnedM3, LearningConfig  # noqa: E402
from merge_trials import merge_case  # noqa: E402
from online import summarise  # noqa: E402
from outputs import run_record, write_json  # noqa: E402
from parallel import (  # noqa: E402
    Checkpoint,
    completed_trials,
    parse_shard,
    run_trials,
    trial_path,
    write_trials,
)
from contexts import block_commuting_groups, build_context_library, contiguous_blocks  # noqa: E402
from noise import NoisyMoments  # noqa: E402
from pivot import describe as describe_pivot  # noqa: E402
from pivot import apply_merged_home, merged_fragment_problems, pivot_fragment_problems, pivot_library  # noqa: E402
from reuse import ReuseLibrary, ReuseSpec, hamiltonian_terms, reuse_fragment_problems  # noqa: E402
from sampler import OracleMoments  # noqa: E402
from states import hartree_fock_state  # noqa: E402  (Part I)
from step4_learned_designs import setup  # noqa: E402


Reuse = ReuseSpec

# Noisy measurement circuits (``--gate-error``, ``--readout-error``): a depolarising error after every CZ and readout flips.
NOISE = {"two_qubit": 0.0, "readout": 0.0}


def moments_for(library, state):
    """Oracle moments of the (possibly noisy) measurement circuits of a library."""
    if NOISE["two_qubit"] or NOISE["readout"]:
        return NoisyMoments(library, state, NOISE["two_qubit"], NOISE["readout"])
    return OracleMoments(library, state)


@dataclasses.dataclass(frozen=True)
class PivotSpec:
    """A method on the pivot contexts of Anastasiou et al. (:mod:`pivot`): ``(Hamiltonian term, class)``.

    ``learning`` is ``"static"`` (M1 static: one allocation, one draw) or a :class:`LearningConfig`
    (sequential M1 with elimination off, or shared elimination), run with every gradient read from
    the pivot contexts its Paulis came from.  ``classes``: ``"auto"`` (the ``2N`` anchored classes
    of the qubit-type pools, first-fit for UCCSD) or ``"firstfit"`` (commuting classes of the pool
    strings by first-fit insertion, which needs fewer classes than the anchored ones).
    """

    learning: object
    classes: str = "auto"
    assign: str = "split"  # "split": the published scheme; "merged": one pivot context per product (stronger)


@dataclasses.dataclass(frozen=True)
class BlockSpec:
    """A method on block-wise commuting contexts (:func:`contexts.block_commuting_groups`).

    ``size`` qubits per block: 1 is qubit-wise commutation (product measurements, no entangling
    gates), 2 and 4 cap the CZ gates of every circuit at ``n/2`` and ``3n/2`` by construction, and
    ``n`` is Part I's fully commuting grouping.  ``learning`` as in :class:`PivotSpec`.
    """

    learning: object
    size: int


@dataclasses.dataclass(frozen=True)
class StaticSpec:
    """M1 static with a chosen confidence factor (``"selection"``: the less conservative z of :func:`part1_bridge.confidence_z`)."""

    confidence: str = "bonferroni"


M1_SEQ = LearningConfig(level="II-0", prior="none", nu=0.0, elimination="off")
II_0 = LearningConfig(level="II-0", prior="none", nu=0.0)
II_A = LearningConfig(prior="none", nu=0.0)
# The sign-aware family: the pairwise rule's plug-in signs can eliminate the best arm while a gradient is near zero (Sec. Q3), and the
# free data of the reuse rows made that happen in 7.5% of LiH trials; these rows use the rule that is safe on the good event.
II_0_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", radius_min_shots=50)
II_A_SAFE = LearningConfig(prior="none", nu=0.0, rule="safe", radius_min_shots=50)
M1_SEQ_SAFE = LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", radius_min_shots=50, elimination="off")

CONFIGS = {
    # Part II's own methods, so that a run on another pool (``<case>@<pool>``) is self-contained.
    "II-0 estimated": II_0,
    "II-A data": II_A,
    "M1 static": "static",
    "M1 seq": M1_SEQ,
    "M1 seq, safe, bound start": LearningConfig(level="II-0", prior="none", nu=0.0, rule="safe", start="bound",
                                                radius_min_shots=50, elimination="off"),
    "M2 marginal": IndependentConfig(rule="marginal", start="oracle"),
    "M2 pairwise": IndependentConfig(rule="pairwise", start="oracle"),
    "M2 safe": IndependentConfig(rule="safe", start="oracle"),
    "M2 safe, bound start": IndependentConfig(rule="safe", start="bound"),
    "M2 marginal, bound start": IndependentConfig(rule="marginal", start="bound"),
    # Huang and Izmaylov's fragmentation: qubit-wise commuting groups of each commutator (sorted insertion)
    "M2 QWC marginal": IndependentConfig(rule="marginal", start="oracle", grouping="qwc"),
    "M2 QWC pairwise": IndependentConfig(rule="pairwise", start="oracle", grouping="qwc"),
    "M2 QWC safe": IndependentConfig(rule="safe", start="oracle", grouping="qwc"),
    "Ikh reuse, FC": Reuse(M1_SEQ, "fc"),
    "Ikh reuse, QWC": Reuse(M1_SEQ, "qwc"),
    "Ikh no reuse, QWC": Reuse(M1_SEQ, "qwc", free_data=False),
    "II-0 + reuse, FC": Reuse(II_0, "fc"),
    "II-A data + reuse, FC": Reuse(II_A, "fc"),
    # sign-aware family (oracle start)
    "II-A data, safe": II_A_SAFE,
    "M1 seq, safe": M1_SEQ_SAFE,
    "Ikh reuse, FC, safe": Reuse(M1_SEQ_SAFE, "fc"),
    "II-0 safe + reuse, FC": Reuse(II_0_SAFE, "fc"),
    "II-A data, safe + reuse, FC": Reuse(II_A_SAFE, "fc"),
    # II-A starting from the home design, free data available as extra coordinates (the splitting decides)
    "II-A data + reuse (home), FC": Reuse(II_A, "fc", True, "home"),
    "II-A data, safe + reuse (home), FC": Reuse(II_A_SAFE, "fc", True, "home"),
    "II-A data, safe, FC library, no data": Reuse(II_A_SAFE, "fc", False, "home"),
    # pivot-based grouping (Anastasiou et al.), every gradient read from the contexts of its own pivots
    "Pivot M1 static": PivotSpec("static"),
    "Pivot M1 seq": PivotSpec(M1_SEQ),
    "Pivot II-0": PivotSpec(II_0),
    "Pivot M1 seq, safe": PivotSpec(M1_SEQ_SAFE),
    "Pivot II-0, safe": PivotSpec(II_0_SAFE),
    "Pivot M1 static, first-fit classes": PivotSpec("static", "firstfit"),
    "Pivot M1 seq, first-fit classes": PivotSpec(M1_SEQ, "firstfit"),
    # each product read from one pivot context (greedy set cover): stronger than the published scheme
    "Pivot merged M1 static": PivotSpec("static", "auto", "merged"),
    "Pivot merged M1 seq": PivotSpec(M1_SEQ, "auto", "merged"),
    "Pivot merged II-0": PivotSpec(II_0, "auto", "merged"),
    # the learned splitting restricted to the pivot contexts: our estimators on the baseline's own circuits
    "Pivot merged II-A": PivotSpec(II_A, "auto", "merged"),
    "Pivot merged II-A, safe": PivotSpec(II_A_SAFE, "auto", "merged"),
    "Pivot merged II-0, safe": PivotSpec(II_0_SAFE, "auto", "merged"),
    "Pivot merged M1 static, first-fit classes": PivotSpec("static", "firstfit", "merged"),
    "Pivot merged M1 seq, first-fit classes": PivotSpec(M1_SEQ, "firstfit", "merged"),
}
# block-wise commuting contexts: the shots-against-circuit-depth frontier (size 1: QWC, no entangling gates)
for _size in (1, 2, 4):
    CONFIGS.update({
        f"M1 static, blocks={_size}": BlockSpec("static", _size),
        f"M1 seq, blocks={_size}": BlockSpec(M1_SEQ, _size),
        f"II-0, blocks={_size}": BlockSpec(II_0, _size),
        f"II-A data, blocks={_size}": BlockSpec(II_A, _size),
        f"M1 seq, safe, blocks={_size}": BlockSpec(M1_SEQ_SAFE, _size),
        f"II-0, safe, blocks={_size}": BlockSpec(II_0_SAFE, _size),
        f"II-A data, safe, blocks={_size}": BlockSpec(II_A_SAFE, _size),
    })
# the radius rule: every method with the less conservative confidence factor z = z_{delta/2} / sqrt(2) of Part I, to see whether
# any ranking depends on the Bonferroni convention (the costs fall by about (z_B / z_sel)^2, 3-5x, and so does the share of correct selections)
SEL = "selection"
CONFIGS.update({
    "M1 static, selection z": StaticSpec(SEL),
    "M1 seq, selection z": dataclasses.replace(M1_SEQ, confidence=SEL),
    "II-0, selection z": dataclasses.replace(II_0, confidence=SEL),
    "II-A data, selection z": dataclasses.replace(II_A, confidence=SEL),
    "M1 seq, safe, selection z": dataclasses.replace(M1_SEQ_SAFE, confidence=SEL),
    "II-0, safe, selection z": dataclasses.replace(II_0_SAFE, confidence=SEL),
    "II-A data, safe, selection z": dataclasses.replace(II_A_SAFE, confidence=SEL),
    "M2 pairwise, selection z": IndependentConfig(rule="pairwise", start="oracle", confidence=SEL),
    "M2 marginal, selection z": IndependentConfig(rule="marginal", start="oracle", confidence=SEL),
    "M2 safe, selection z": IndependentConfig(rule="safe", start="oracle", confidence=SEL),
})
EXTENSIONS = (Reuse, PivotSpec, BlockSpec, StaticSpec)
DEFAULT = [n for n, c in CONFIGS.items() if not isinstance(c, EXTENSIONS) and not n.startswith("M2 QWC")
           and not n.endswith("selection z") and n not in ("II-0 estimated", "II-A data", "II-A data, safe", "M1 seq, safe")]
POOL_CONFIGS = ["II-0 estimated", "II-A data", "M1 static", "M1 seq", "M2 pairwise", "M2 marginal"]
REUSE_NAMES = [n for n, c in CONFIGS.items() if isinstance(c, Reuse)]
PIVOT_NAMES = [n for n, c in CONFIGS.items() if isinstance(c, PivotSpec)]
BLOCK_NAMES = [n for n, c in CONFIGS.items() if isinstance(c, BlockSpec)]


def reuse_setup(problem, case: str, grouping: str, strategy: str, energy_error: float):
    """Library with the energy contexts, the free data, and the designs on it."""
    terms = hamiltonian_terms(problem, case)
    rl = ReuseLibrary(problem, terms, strategy, grouping)
    library = rl.library
    oracle = moments_for(library, problem.evaluator.state)
    credit = rl.credit(oracle, energy_error)
    base = reuse_fragment_problems(problem, rl, oracle, credit)
    base_home = build_fragment_problems(problem, library, oracle, "II-0")  # plain home assignment
    split = build_fragment_problems(problem, library, oracle, "II-A")
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in split]
    prior = OracleMoments(library, hartree_fock_state(problem.n_qubits, problem.metadata["n_electrons"]))
    return {"library": library, "oracle": oracle, "prior": prior, "base": base, "base_home": base_home, "coords": coords,
            "credit": credit, "overlap": rl.overlap(problem), "energy_groups": len(rl.energy)}


def pivot_setup(problem, case: str, classes: str, assign: str = "split", splitting: bool = False):
    """Library of the pivot contexts, the oracle moments and every gradient's reading of them.

    ``assign="split"`` is the published scheme (every pivot that produces a product measures it);
    ``"merged"`` reads each product from one pivot context chosen by greedy set cover.  With
    ``splitting`` the coordinates of II-A (a product may be read from any context that measures it)
    are built as well, for the learned designs restricted to the pivot contexts.
    """
    terms = hamiltonian_terms(problem, case)
    library, structure = pivot_library(problem, terms, classes)
    if assign == "merged":
        apply_merged_home(library, structure)
    oracle = moments_for(library, problem.evaluator.state)
    make = pivot_fragment_problems if assign == "split" else merged_fragment_problems
    base = make(problem, structure, library, oracle)
    reference = build_fragment_problems(problem, library, oracle, "II-A") if splitting else base
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in reference]
    return {"library": library, "oracle": oracle, "base": base, "coords": coords,
            "structure": describe_pivot(structure, library)}


def block_setup(problem, size: int, splitting: bool = False):
    """Block-wise commuting contexts of ``size`` qubits, the oracle moments and the designs on them."""
    blocks = contiguous_blocks(problem.n_qubits, size)
    groups = block_commuting_groups(problem.universal_support, blocks)
    library = build_context_library(problem, "canonical", groups=groups, blocks=blocks)
    oracle = moments_for(library, problem.evaluator.state)
    base = build_fragment_problems(problem, library, oracle, "II-0")
    reference = build_fragment_problems(problem, library, oracle, "II-A") if splitting else base
    coords = [(p.coord_ctx, p.coord_pauli, dict(zip(p.pauli_ids.tolist(), p.pauli_target))) for p in reference]
    cz = library.two_qubit_counts()
    return {"library": library, "oracle": oracle, "base": base, "coords": coords,
            "structure": {"classes": f"blocks={size}", "n_classes": len(blocks), "contexts": library.n_contexts,
                          "members_mean": float(np.mean([len(c.members) for c in library.contexts])),
                          "two_qubit_mean": float(cz.mean()), "two_qubit_max": int(cz.max()),
                          "bound_two_qubit": len(blocks) * size * (size - 1) // 2}}


def is_static(config) -> bool:
    return (config == "static" or isinstance(config, StaticSpec)
            or (isinstance(config, (PivotSpec, BlockSpec)) and config.learning == "static"))


def with_rho(config, rho: float):
    """The configuration stopping at a ``rho``-good selection (``rho = 0``: exact identification)."""
    if rho <= 0 or is_static(config):
        return config
    if isinstance(config, (Reuse, PivotSpec, BlockSpec)):
        return dataclasses.replace(config, learning=dataclasses.replace(config.learning, rho=rho))
    return dataclasses.replace(config, rho=rho)


def static_radius(problem, rho: float) -> float:
    return rho * float(problem.abs_gradients.max()) / 2.0 if rho > 0 else problem.top_gap() / 2.0


def build_model(name, problem, case, strategy, shared, cache, energy_error, rho=0.0):
    config = with_rho(CONFIGS[name], rho)
    library, oracle, prior, base, coords = shared
    if config == "static" or isinstance(config, StaticSpec):
        design = DesignSet(build_fragment_problems(problem, library, oracle, "II-0"), library.n_contexts)
        confidence = config.confidence if isinstance(config, StaticSpec) else "bonferroni"
        return StaticM1(problem, library, oracle, design, radius=static_radius(problem, rho), confidence=confidence)
    if isinstance(config, PivotSpec):
        splitting = isinstance(config.learning, LearningConfig) and config.learning.level != "II-0"
        key = ("pivot", config.classes, config.assign, splitting)
        if key not in cache:
            started = time.perf_counter()
            cache[key] = pivot_setup(problem, case, config.classes, config.assign, splitting)
            s = cache[key]["structure"]
            print(f"{case}: pivot library ({s['classes']} classes, {s['n_classes']} of them): {s['contexts']} contexts, "
                  f"{s['members_mean']:.1f} products each, {s['two_qubit_mean']:.1f} two-qubit gates on average "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
        info = cache[key]
        if config.learning == "static":
            design = DesignSet(info["base"], info["library"].n_contexts)
            return StaticM1(problem, info["library"], info["oracle"], design, radius=static_radius(problem, rho))
        return LearnedM3(problem, info["library"], info["oracle"], None, info["base"], info["coords"], config.learning)
    if isinstance(config, BlockSpec):
        splitting = isinstance(config.learning, LearningConfig) and config.learning.level != "II-0"
        key = ("blocks", config.size, splitting)
        if key not in cache:
            started = time.perf_counter()
            cache[key] = block_setup(problem, config.size, splitting)
            s = cache[key]["structure"]
            print(f"{case}: contexts of {config.size}-qubit blocks: {s['contexts']} contexts, {s['members_mean']:.1f} "
                  f"products each, {s['two_qubit_mean']:.1f} two-qubit gates on average (at most {s['two_qubit_max']}; "
                  f"bound {s['bound_two_qubit']}) ({time.perf_counter() - started:.0f}s)", flush=True)
        info = cache[key]
        if config.learning == "static":
            design = DesignSet(info["base"], info["library"].n_contexts)
            return StaticM1(problem, info["library"], info["oracle"], design, radius=static_radius(problem, rho))
        return LearnedM3(problem, info["library"], info["oracle"], None, info["base"], info["coords"], config.learning)
    if isinstance(config, LearningConfig):
        return LearnedM3(problem, library, oracle, prior, base, coords, config)
    if isinstance(config, IndependentConfig):
        key = ("independent", config.grouping)
        if key not in cache:
            started = time.perf_counter()
            cache[key] = IndependentContexts(problem, progress=True, grouping=config.grouping)
            cache[key].set_state(problem.evaluator.state, NOISE["two_qubit"], NOISE["readout"])
            print(f"{case}: {cache[key].n_contexts} independent {config.grouping.upper()} groups built in "
                  f"{time.perf_counter() - started:.0f}s ({cache[key].cz.mean():.1f} two-qubit gates on average)", flush=True)
        return IndependentBAI(problem, cache[key], config)
    key = ("reuse", config.grouping)
    if key not in cache:
        started = time.perf_counter()
        cache[key] = reuse_setup(problem, case, config.grouping, strategy, energy_error)
        info = cache[key]
        print(f"{case}: {config.grouping} reuse library: {info['library'].n_contexts} contexts, "
              f"{info['energy_groups']} energy groups, credit {info['credit'].sum():,} shots, overlap "
              f"{info['overlap']['covered_fraction']:.2f} of the support / "
              f"{info['overlap']['covered_mass_fraction']:.2f} of the coefficient mass "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    info = cache[key]
    credit = info["credit"] if config.free_data else None
    # "no reuse" is plain grouping (home assignment); "home" lets the learned splitting use the free data on its own
    base = info["base"] if (config.free_data and config.assign == "fixed") else info["base_home"]
    return LearnedM3(problem, info["library"], info["oracle"], info["prior"], base, info["coords"],
                     config.learning, credit=credit)


def label_of(name: str) -> str:
    config = CONFIGS[name]
    if config == "static":
        return "M1 static (gap/2, exact variances)"
    if isinstance(config, StaticSpec):
        return f"M1 static (gap/2, exact variances)/confidence={config.confidence}"
    if isinstance(config, PivotSpec):
        learning = "M1 static (gap/2, exact variances)" if config.learning == "static" else config.learning.label
        return f"{learning}/pivot contexts/classes={config.classes}/assign={config.assign}"
    if isinstance(config, BlockSpec):
        learning = "M1 static (gap/2, exact variances)" if config.learning == "static" else config.learning.label
        return f"{learning}/{config.size}-qubit block contexts"
    if isinstance(config, Reuse):
        return (f"{config.learning.label}/{config.grouping}/{'reuse' if config.free_data else 'no reuse'}"
                + (f"/assign={config.assign}" if config.assign != "fixed" else ""))
    return config.label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["H4_square_eq_side1p0_CISD", "H4_square_stretch_side2p0_CISD"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT, choices=list(CONFIGS))
    parser.add_argument("--strategy", default="mass")
    parser.add_argument("--rho", type=float, default=0.0,
                        help="stop at a rho-good selection (|g| >= (1 - rho) max|g|); 0: exact identification. "
                             "Needed on pools with exactly tied leaders; M1 static then uses radius rho max|g| / 2")
    parser.add_argument("--energy-error", type=float, default=1e-3,
                        help="standard error (Ha) of the last energy evaluation whose data the reuse rows hold for free")
    parser.add_argument("--gate-error", type=float, default=0.0,
                        help="two-qubit depolarising error probability after every CZ of the measurement circuits (src/noise.py)")
    parser.add_argument("--readout-error", type=float, default=0.0, help="bit-flip probability of every readout")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="worker processes (use OMP_NUM_THREADS=1)")
    parser.add_argument("--shard", default="1/1", help="K/N: run every N-th trial starting at K")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    shard = parse_shard(args.shard)

    NOISE.update(two_qubit=args.gate_error, readout=args.readout_error)
    noise_tag = f", noise={args.gate_error:g}/{args.readout_error:g}" if (args.gate_error or args.readout_error) else ""
    for case in args.cases:
        problem, library, oracle, prior, base, coords = setup(case, args.strategy)
        if noise_tag:  # the fully commuting library of the main study, sampled through noisy circuits
            oracle = moments_for(library, problem.evaluator.state)
            base = build_fragment_problems(problem, library, oracle, "II-0")
        shared = (library, oracle, prior, base, coords)
        cache: dict = {}
        done_names = []
        for name in args.configs:
            if is_static(CONFIGS[name]) and args.rho <= 0 and problem.top_gap() < 1e-9:
                print(f"{case}: {name}: tied leaders, exact identification has no finite cost; skipped", flush=True)
                continue
            label = (name if args.rho <= 0 else f"{name}, rho={args.rho:g}") + noise_tag
            path = trial_path(args.out, case, "step5", label, shard)
            finished = completed_trials(path, args.trials, shard) if args.resume else None
            if finished is not None:
                print(f"{case:30s} {name:28s} already complete in {path.name}; skipped", flush=True)
                continue
            try:
                model = build_model(name, problem, case, args.strategy, shared, cache, args.energy_error, args.rho)
            except ValueError as error:  # e.g. a cached problem whose orbitals do not match a rebuilt H
                print(f"{case}: {name}: SKIPPED: {error}", flush=True)
                continue
            checkpoint = Checkpoint(path)
            if not args.resume:
                checkpoint.clear()
            done = checkpoint.load()
            if done:
                print(f"{case:30s} {name:28s} resuming: {len(done)} trials", flush=True)

            def runner(rng, model=model):
                outcome = model.run(rng)
                return outcome, dict(outcome.extra)

            started = time.perf_counter()
            results = run_trials(runner, args.trials, args.seed, workers=args.workers, shard=shard,
                                 skip=done, on_result=checkpoint.add)
            seconds = time.perf_counter() - started
            results = sorted(list(done.items()) + results, key=lambda item: item[0])
            write_trials(path, label, args.trials, args.seed, [(i, o, e) for i, (o, e) in results])
            checkpoint.clear()
            summary = summarise([o for _, (o, _) in results])
            print(f"{case:30s} {name:28s} shots {summary['shots_mean']:14,.0f} +- {summary['shots_sem']:11,.0f}  "
                  f"median {summary['shots_median']:13,.0f}  correct {summary['correct_rate']:.3f}  "
                  f"({seconds:.0f}s)", flush=True)
            done_names.append(label)
        if shard == (1, 1):
            merge_case(args.out, case, "step5")
        else:
            print(f"{case}: shard {args.shard} written; run scripts/merge_trials.py --step step5 when all shards are in")
        meta_path = args.out / case / f"{case}_step5_meta.json"
        try:  # several processes of one case write this file; a half-written one is rebuilt rather than fatal
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except json.JSONDecodeError:
            meta = {}
        meta.setdefault("configs", {}).update({k: label_of(k.split(", rho=")[0].split(", noise=")[0]) for k in done_names})
        for key, info in cache.items():
            if isinstance(key, tuple) and key[0] == "reuse":
                meta.setdefault("reuse_libraries", {})[key[1]] = {
                    "contexts": info["library"].n_contexts, "energy_groups": info["energy_groups"],
                    "credit_shots": int(info["credit"].sum()), "overlap": info["overlap"]}
            elif isinstance(key, tuple) and key[0] == "pivot":
                meta.setdefault("pivot_libraries", {})[f"{key[1]}/{key[2]}"] = info["structure"]
            elif isinstance(key, tuple) and key[0] == "blocks":
                meta.setdefault("block_libraries", {})[str(key[1])] = info["structure"]
        meta.update({"trials": args.trials, "seed": args.seed, "strategy": args.strategy, "energy_error": args.energy_error, "rho": args.rho,
                     "seeding": "trial i uses SeedSequence(seed).spawn(trials)[i]", "run": run_record()})
        scratch = meta_path.with_name(f"{meta_path.name}.{os.getpid()}.tmp")  # atomic: concurrent writers never interleave
        write_json(scratch, meta)
        os.replace(scratch, meta_path)


if __name__ == "__main__":
    main()
