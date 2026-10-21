#!/usr/bin/env python3
"""Keep ``reports/draft.tex`` self-contained although it cites the Supporting Information.

The main text cites a section or table of the SI as ``SI Sec.~\\siref{sec:si_contexts}`` or ``SI Table~\\siref{tab:settings}``.  The
numbers ("S1.2", "S1") are those of ``reports/supplement.tex``; instead of ``xr-hyper`` (which needs ``supplement.aux`` next to the
draft, so the draft would not compile alone on Overleaf) this script pastes them between

    % BEGIN INLINE SIREFS   ...   % END INLINE SIREFS

in the preamble of the draft, one ``\\def`` per cited label.  Run it after the SI was built (``supplement.aux`` exists) and whenever
the SI changes its structure or the draft cites another SI label:

    python scripts/paper_a_si_refs.py --aux <path to supplement.aux>

``static_numbers`` derives the numbers of the SI sections and subsections from ``supplement.tex`` alone (the test of
``tests/test_make_si.py`` compares them with the pasted block).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[2] / "reports"
BLOCK = re.compile(r"% BEGIN INLINE SIREFS\n.*?% END INLINE SIREFS\n", re.S)
CITE = re.compile(r"\\siref\{([^}]*)\}")
DEF = re.compile(r"\\expandafter\\def\\csname SIref:([^}\\]*)\\endcsname\{([^}]*)\}")


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def cited_labels(draft: str) -> list[str]:
    """Labels of the SI that the draft cites (the macro definition itself and the pasted block are not citations)."""
    body = BLOCK.sub("", draft)
    body = re.sub(r"\\newcommand\{\\siref\}.*", "", body)
    body = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("%"))  # comments mention the macro
    return sorted(set(CITE.findall(body)))


def aux_numbers(aux: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)}


def static_numbers(supplement: str) -> dict[str, str]:
    """Numbers of the labels that follow a ``\\section`` or ``\\subsection`` of the SI (S1, S1.2): counted from the source."""
    out: dict[str, str] = {}
    sec = sub = 0
    current = None
    for line in supplement.splitlines():
        if re.match(r"\\section\{", line):
            sec, sub = sec + 1, 0
            current = f"S{sec}"
        elif re.match(r"\\subsection\{", line):
            sub += 1
            current = f"S{sec}.{sub}"
        elif re.match(r"\\subsubsection\{|\\paragraph\{|\\begin\{", line):
            current = None
        for lab in re.findall(r"^\\label\{([^}]*)\}", line):
            if current and lab.startswith("sec:"):
                out[lab] = current
    return out


def block(numbers: dict[str, str]) -> str:
    lines = [f"\\expandafter\\def\\csname SIref:{lab}\\endcsname{{{numbers[lab]}}}\n" for lab in sorted(numbers)]
    return "% BEGIN INLINE SIREFS\n" + "".join(lines) + "% END INLINE SIREFS\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, default=REPORTS / "draft.tex")
    parser.add_argument("--aux", type=Path, required=True, help="supplement.aux of a build of reports/supplement.tex")
    args = parser.parse_args()
    draft = read(args.draft)
    assert len(BLOCK.findall(draft)) == 1, "the draft needs exactly one % BEGIN/END INLINE SIREFS block"
    labels = cited_labels(draft)
    aux = aux_numbers(read(args.aux))
    missing = [lab for lab in labels if lab not in aux]
    assert not missing, f"cited labels not in {args.aux.name}: {missing}"
    draft = BLOCK.sub(lambda _: block({lab: aux[lab] for lab in labels}), draft)
    args.draft.write_bytes(draft.encode("utf-8"))
    print(f"{len(labels)} SI labels pasted into {args.draft.name}: " + ", ".join(f"{lab}={aux[lab]}" for lab in labels))


if __name__ == "__main__":
    main()
