#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/rebuild_part1_tables_from_cached.py --cached-root data/cached --out reproduced_tables
python scripts/validate_cached_outputs.py --tables-dir reproduced_tables
python scripts/make_part1_report_from_tables.py --tables-dir reproduced_tables --out reproduced_tables/part1_tables_report.md
