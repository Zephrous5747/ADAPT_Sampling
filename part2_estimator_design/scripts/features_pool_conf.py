#!/usr/bin/env python3
"""Copy the rows of a confidence sweep into another directory with a ``source`` tag, so that two sweeps of the same (method, level, index) pool.

``energy_conf_noninferiority.py`` identifies a run by (method, delta, index, source) and lets later directories win; two sweeps with
different seeds both number their trajectories from 0, so the second one is tagged (``--tag``) before it is passed after the first.

    python scripts/features_pool_conf.py --src runs/features_conf_gtol1e-5_cl --dst runs/features_conf_gtol1e-5_cl_tagged --tag direct11
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    n = 0
    for path in sorted(args.src.glob("*/ci_*.jsonl")):
        out = args.dst / path.parent.name / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["source"] = args.tag
                rows.append(json.dumps(r))
        out.write_text("\n".join(rows) + "\n")
        n += len(rows)
    for ref in args.src.glob("*/exact_reference.json"):
        (args.dst / ref.parent.name).mkdir(parents=True, exist_ok=True)
        (args.dst / ref.parent.name / ref.name).write_text(ref.read_text())
    print(f"{n} rows tagged '{args.tag}' -> {args.dst}")


if __name__ == "__main__":
    main()
