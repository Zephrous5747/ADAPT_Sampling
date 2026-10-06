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
import json
import multiprocessing as mp
import os
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
               shard: tuple[int, int] = (1, 1), skip=(), on_result: Callable | None = None
               ) -> list[tuple[int, object]]:
    """``[(trial index, runner(rng))]`` for the trials of this shard, in order.

    Trials in ``skip`` are not run (they finished in an earlier, interrupted job).
    ``on_result(index, result)`` is called in this process as each trial finishes,
    which is where checkpoints are written.
    """
    global _RUNNER
    seeds = np.random.SeedSequence(seed).spawn(n_trials)
    skip = set(skip)
    jobs = [(i, seeds[i]) for i in shard_indices(n_trials, shard) if i not in skip]
    _RUNNER = runner
    results = []
    try:
        if workers <= 1:
            finished = map(_call, jobs)
            for item in finished:
                results.append(item)
                if on_result is not None:
                    on_result(*item)
        else:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
                for item in _unordered(pool, jobs):
                    results.append(item)
                    if on_result is not None:
                        on_result(*item)
    finally:
        _RUNNER = None
    return sorted(results, key=lambda item: item[0])


def _unordered(pool, jobs):
    """Results as they finish (``pool.map`` would hold them back until all are done)."""
    from concurrent.futures import as_completed

    futures = [pool.submit(_call, job) for job in jobs]
    for future in as_completed(futures):
        yield future.result()


class Checkpoint:
    """Finished trials of one run, appended as JSON lines so an interrupted job can resume.

    Each line holds the trial index, the :class:`online.OnlineOutcome` fields and
    the extra columns of its trial row.  The file sits next to the trial file it
    will become (``<trial file>.partial.jsonl``) and is removed once that is written.
    """

    def __init__(self, trial_file: Path) -> None:
        self.path = Path(str(trial_file) + ".partial.jsonl")

    def load(self) -> dict[int, tuple[object, dict]]:
        from online import OnlineOutcome

        done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line cut off by the time limit
                    continue
                done[int(r["trial"])] = (OnlineOutcome(int(r["selected"]), bool(r["correct"]), float(r["shots"]),
                                                       int(r["rounds"]), extra=r["outcome_extra"]), r["extra"])
        return done

    def add(self, index: int, result) -> None:
        outcome, extra = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"trial": index, "selected": int(outcome.selected), "correct": bool(outcome.correct),
                           "shots": float(outcome.shots), "rounds": int(outcome.rounds),
                           "outcome_extra": outcome.extra, "extra": extra}, default=_plain)
        with self.path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def _plain(value):
    """JSON fallback for NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def completed_trials(path: Path, n_trials: int, shard: tuple[int, int]) -> list[dict] | None:
    """The rows of a finished trial file, or ``None`` unless it holds exactly this shard's trials."""
    if not Path(path).exists():
        return None
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[0]["n_trials"]) != n_trials:
        return None
    if sorted(int(r["trial"]) for r in rows) != shard_indices(n_trials, shard):
        return None
    return rows


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


EXTRA_COLUMNS = {"shortfall": float, "miscovered_rounds": int, "best_eliminated": lambda v: v == "True",
                 "stopped_rho": lambda v: v == "True", "design_seconds": float,
                 "statistics_seconds": float, "contexts_used": int, "cz_per_shot_mean": float}


def read_trials(out: Path, case: str, step: str) -> dict[str, list[dict]]:
    """Every trial row of one case and step, grouped by label (all shards)."""
    by_label: dict[str, list[dict]] = {}
    for path in sorted((Path(out) / case / "trials").glob(f"{step}__*.csv")):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                by_label.setdefault(row["label"], []).append(row)
    return by_label


def outcomes_from_rows(rows: list[dict]):
    """Rebuild :class:`online.OnlineOutcome` objects, with diagnostics when recorded."""
    from online import OnlineOutcome

    result = []
    for r in rows:
        extra = {k: cast(r[k]) for k, cast in EXTRA_COLUMNS.items() if r.get(k) not in (None, "")}
        result.append(OnlineOutcome(int(r["selected"]), bool(int(r["correct"])), float(r["shots"]),
                                    int(r["rounds"]), extra=extra))
    return result


def check_complete(case: str, label: str, rows: list[dict]) -> None:
    n_trials = int(rows[0]["n_trials"])
    indices = sorted(int(r["trial"]) for r in rows)
    if indices != list(range(n_trials)):
        missing = sorted(set(range(n_trials)) - set(indices))
        duplicated = len(indices) - len(set(indices))
        raise SystemExit(f"{case} / {label}: {len(missing)} trials missing, {duplicated} duplicated")
