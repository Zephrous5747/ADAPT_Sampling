#!/usr/bin/env python3
"""Labels of ``draft.tex`` for ``supplement.tex``, as a static file (``reports/draft_labels.tex``).

``xr-hyper`` reads ``draft.aux``, which exists only where ``draft.tex`` was compiled in the same folder; on Overleaf, where one main file is
compiled, the references of the SI to the main text then come out as ``??``.  This script copies the ``\\newlabel`` lines of ``draft.aux`` as
``r@M-<label>`` definitions, which the SI reads with ``\\input{draft_labels}`` and which work anywhere (no hyperlinks into the draft).

    python scripts/paper_a_draft_labels.py --aux <build folder>/draft.aux      # then rebuild supplement.tex
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "reports" / "draft_labels.tex"
LINE = re.compile(r"^\\newlabel\{([^}]+)\}\{(.*)\}\s*$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aux", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.aux.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            rows.append(f"\\expandafter\\gdef\\csname r@M-{m.group(1)}\\endcsname{{{m.group(2)}}}")
    text = ("% Labels of draft.tex for supplement.tex (M- prefix); written by scripts/paper_a_draft_labels.py from draft.aux. Do not edit.\n"
            "\\makeatletter\n" + "\n".join(rows) + "\n\\makeatother\n")
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(rows)} labels)")


if __name__ == "__main__":
    main()
