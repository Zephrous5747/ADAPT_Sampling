"""Independent trials, run in parallel and split into shards.

Every trial gets its own random stream, ``SeedSequence(seed).spawn(n_trials)[i]``,
so a trial's result depends only on ``(seed, i)``: it is the same whether the
trials run in one process, in many worker processes, or in shards on different
machines.  ``shard = (k, n)`` runs trials ``k-1, k-1+n, k-1+2n, ...``; the union of
all ``n`` shards is exactly the full set.

Workers are forked after the model is built, so the (large, read-only) model is
shared copy-on-write rather than pickled, and only the per-trial results travel
back.  Set ``OMP_NUM_THREADS=1`` when running many workers.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

_RUNNER: Callable | None = None


def parse_shard(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if not match or not 1 <= int(match.group(1)) <= int(match.group(2)):
        raise ValueError(f"shard must look like K/N with 1 <= K <= N, got {text!r}")
    return int(match.group(1)), int(match.group(2))


def shard_indices(n_trials: int, shard: tuple[int, int]) -> list[int]:
    k, n = shard
    return list(range(k - 1, n_trials, n))


def _call(job):
    index, seed = job
    return index, _RUNNER(np.random.default_rng(seed))


def run_trials(runner: Callable, n_trials: int, seed: int, *, workers: int = 1,
               shard: tuple[int, int] = (1, 1)) -> list[tuple[int, object]]:
    """``[(trial index, runner(rng))]`` for the trials of this shard, in order."""
    global _RUNNER
    seeds = np.random.SeedSequence(seed).spawn(n_trials)
    jobs = [(i, seeds[i]) for i in shard_indices(n_trials, shard)]
    _RUNNER = runner
    try:
        if workers <= 1:
            return [_call(job) for job in jobs]
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            return sorted(pool.map(_call, jobs, chunksize=1), key=lambda item: item[0])
    finally:
        _RUNNER = None


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", label).strip("_")


def trial_path(out: Path, case: str, step: str, label: str, shard: tuple[int, int]) -> Path:
    return Path(out) / case / "trials" / f"{step}__{slug(label)}__shard{shard[0]}of{shard[1]}.csv"


def write_trials(path: Path, label: str, n_trials: int, seed: int, results) -> Path:
    """One row per trial; ``results`` holds ``(index, OnlineOutcome, extra dict)``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, outcome, extra in results:
        rows.append({"label": label, "trial": index, "n_trials": n_trials, "seed": seed,
                     "selected": outcome.selected, "correct": int(outcome.correct),
                     "shots": outcome.shots, "rounds": outcome.rounds, **extra})
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["label"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
