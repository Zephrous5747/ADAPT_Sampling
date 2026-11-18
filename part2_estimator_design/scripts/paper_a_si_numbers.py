#!/usr/bin/env python3
"""Paper A (SI rebuild): provenance check of the numbers in the hand-written parts of the Supporting Information.

Every number in the prose of ``reports/si_parts/*.tex`` that was written by hand (the evidence blocks ``b*.tex`` by default) must occur in one of the sources
that the numbers come from: the SI before the rebuild (``reports/archive/supplement_v48_before_rebuild.tex``, whose numbers were checked against the stored
runs), the main text (``reports/draft.tex``), the generated tables (``reports/generated/*.tex``) and the archived draft before the compression.  A number
that occurs in none of them is a typing error or a new result without a table; both need a decision.  The check is syntactic (a number may occur in the
sources for another reason), so it finds typos, not wrong readings.

Numbers are compared as written without the sign, the thousands separator and the mantissa/exponent form: ``$4.6\\times10^{10}$`` gives ``4.6`` and ``10``
and ``10`` (the exponent), ``$2{,}244{,}862$`` gives ``2244862``.  Numbers inside ``\\ref``, ``\\label``, ``\\path`` and ``\\cite`` and the labels of tables are
ignored, as are comments and the bodies of ``% BEGIN INLINE`` blocks (generated tables).

    python scripts/paper_a_si_numbers.py                 # the evidence blocks b*.tex
    python scripts/paper_a_si_numbers.py --parts 's1_*'  # other hand-written parts
Exit status 1 if a number is not found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[2] / "reports"
SOURCES = [REPORTS / "archive" / "supplement_v48_before_rebuild.tex", REPORTS / "draft.tex", REPORTS / "archive" / "draft_full_pre_compression.tex"]

NUMBER = re.compile(r"(?<![\w.])\d+(?:\{,\}\d{3})*(?:\.\d+)?")


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def strip_markup(text: str) -> str:
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"% BEGIN INLINE (\w+)\n.*?% END INLINE \1\n", "", text, flags=re.S)
    text = re.sub(r"\\(?:ref|eqref|label|cite|path|pageref|siref)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:pc)\{[0-9.]+cm\}", " ", text)  # widths of the ledger cells
    text = re.sub(r"\\(?:setlength|tabcolsep|footnotesize)\S*", " ", text)
    text = re.sub(r"\\begin\{(?:tabular|longtable)\}\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:section|subsection)\{[^}]*\}", " ", text)
    text = re.sub(r"\{\\,\\scriptsize\[", " [", text)
    return text


def numbers(text: str) -> list[str]:
    out = []
    for m in NUMBER.finditer(text):
        tok = m.group(0).replace("{,}", "")
        out.append(tok)
    return out


def corpus() -> set[str]:
    tokens: set[str] = set()
    paths = [p for p in SOURCES if p.exists()] + sorted((REPORTS / "generated").glob("*.tex"))
    for p in paths:
        t = read(p).replace("{,}", "")
        tokens.update(NUMBER.findall(t))
        tokens.update(re.findall(r"\d+(?:\.\d+)?", t))  # also numbers glued to letters or signs (10^{-3}, $-21\%$)
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", default="[bc][0-9]*,s1_baselines", help="comma-separated globs of the parts in reports/si_parts (without .tex): the evidence blocks b*, "
                        "the robustness parts c* and the table of the baselines")
    args = parser.parse_args()
    known = corpus()
    bad = 0
    paths = sorted({p for pattern in args.parts.split(",") for p in (REPORTS / "si_parts").glob(pattern.strip() + ".tex")})
    for path in paths:
        text = strip_markup(read(path))
        missing = []
        for k, line in enumerate(text.split("\n"), 1):
            for tok in numbers(line):
                if tok not in known and tok.lstrip("0") not in known:
                    missing.append((k, tok, line.strip()[:90]))
        print(f"{path.name}: {len(numbers(text))} numbers, {len(missing)} not found in the sources")
        for k, tok, ctx in missing:
            print(f"   {tok:>12s}   {ctx}")
        bad += len(missing)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
