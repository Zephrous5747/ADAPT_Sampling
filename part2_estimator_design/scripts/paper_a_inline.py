#!/usr/bin/env python3
"""Keep ``reports/draft.tex`` self-contained: the only files it needs are ``figures/*.pdf``.

The tables written by the generators (``reports/generated/tab_*.tex``) and the bibliography (the ``.bbl`` that BibTeX
makes from ``references.bib``) are pasted into the draft between marker comments

    % BEGIN INLINE <name>   ...   % END INLINE <name>

so the draft compiles on its own (Overleaf: upload ``draft.tex`` and ``figures/``).  Run again after a generator has
rewritten a table: the text between the markers is replaced.

    python scripts/paper_a_inline.py                     # first time: turns \\input{generated/<name>} into marked blocks
    python scripts/paper_a_inline.py --bbl draft.bbl     # (re)paste the bibliography built by BibTeX
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[2] / "reports"
INPUT = re.compile(r"\\input\{generated/(?P<name>[A-Za-z0-9_]+)\}")
BLOCK = re.compile(r"% BEGIN INLINE (?P<name>[A-Za-z0-9_]+)\n(?P<body>.*?)% END INLINE (?P=name)\n", re.S)


def block(name: str, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    return f"% BEGIN INLINE {name}\n{text}% END INLINE {name}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=REPORTS / "draft.tex")
    parser.add_argument("--generated", type=Path, default=REPORTS / "generated")
    parser.add_argument("--bbl", type=Path, default=None, help="BibTeX output to paste in place of \\bibliography")
    args = parser.parse_args()
    text = args.draft.read_text(encoding="utf-8").replace("\r\n", "\n")

    def first_time(match: re.Match) -> str:
        name = match["name"]
        return block(name, (args.generated / f"{name}.tex").read_text(encoding="utf-8").replace("\r\n", "\n"))

    text, n_new = INPUT.subn(first_time, text)

    def refresh(match: re.Match) -> str:
        name = match["name"]
        if name == "BIBLIOGRAPHY":
            return match[0]
        path = args.generated / f"{name}.tex"
        return block(name, path.read_text(encoding="utf-8").replace("\r\n", "\n")) if path.exists() else match[0]

    text, n_all = BLOCK.subn(refresh, text)
    if args.bbl is not None:
        bbl = args.bbl.read_text(encoding="utf-8").replace("\r\n", "\n")
        pasted = block("BIBLIOGRAPHY", bbl)
        if "% BEGIN INLINE BIBLIOGRAPHY" in text:
            text = re.sub(r"% BEGIN INLINE BIBLIOGRAPHY\n.*?% END INLINE BIBLIOGRAPHY\n", lambda _: pasted, text, flags=re.S)
        else:
            old = "\\bibliographystyle{apsrev4-2}\n\\bibliography{references}"
            assert text.count(old) == 1, "bibliography commands not found"
            text = text.replace(old, pasted)
    args.draft.write_text(text, encoding="utf-8", newline="\n")
    print(f"inlined {n_new} new tables, refreshed {n_all} blocks; bibliography {'pasted' if args.bbl else 'unchanged'}")


if __name__ == "__main__":
    main()
