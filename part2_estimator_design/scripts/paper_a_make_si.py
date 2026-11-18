#!/usr/bin/env python3
"""Build ``reports/supplement.tex``, the Supporting Information of Paper A, from the full draft and the findings report.

Steps 1 and 2 of the compression plan (``reports/compression_plan_jctc.tex``): the SI is cut out of the draft *before* anything is
removed from it, so that nothing is lost.  Its prose and floats come from ``reports/archive/draft_full_pre_compression.tex`` (the
draft of 19 October, before the Methods were compressed; ``FULL``), not from the live ``reports/draft.tex``, which the compression
steps shorten.  The live draft is read only for its labels (cross references from the SI into the main text) and to decide which
floats stay in the main text.  The script copies, per SI section (S1-S10, see ``SECTIONS``), whole subsections or named paragraphs
(floats taken out), the floats named by label, and sections or table blocks of ``reports/energy_trajectory_findings.tex``.
Text is copied verbatim; trimming and rewriting of the SI prose is a later step.

* The SI is a separate document with the same preamble as the draft; tables and figures are numbered S1, S2, ... and sections S1, ...
* The main text and the SI refer to each other with ``the static label file draft_labels.tex (scripts/paper_a_draft_labels.py; xr-hyper fails on Overleaf)``: prefix ``M-`` for labels of ``draft.tex`` used inside the SI,
  prefix ``S-`` for labels of the SI used inside the draft.  A label that is defined in the SI keeps its name there, so a label may
  exist in both documents (the SI keeps the long version of what the main text summarizes).
* A hand-written part (``file`` item) may place a float of the full draft with a line ``%%FLOAT <label>``; the float is restyled like the other
  copied floats (``paper_a_si_clean.CAPTIONS``).  A line ``%%TABULAR <name>`` places the ``tabular`` of the generated table
  ``reports/generated/<name>.tex`` (inside ``\fitcol``), so that several generated tables can be panels of one float.  A section may carry its own label as a third element of its ``SECTIONS`` entry (default ``sec:si<i>``).
* ``% BEGIN INLINE <name>`` blocks of the findings report are kept, so ``scripts/paper_a_inline.py --draft ../reports/supplement.tex``
  refreshes the generated tables in the SI as well.
* Every float of the full draft must be either in the SI or in ``MAIN_ONLY`` (the floats that stay in the main text; they must still
  exist in the live draft); the script stops otherwise.
* The bibliography is inline, as in the draft: build once with ``\\bibliography{references}``, then ``--paste-bbl supplement.bbl``.

    python scripts/paper_a_make_si.py                      # writes reports/supplement.tex
    python scripts/paper_a_make_si.py --paste-bbl <bbl>    # pastes the BibTeX output between the markers (in place)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import paper_a_si_clean as clean

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent / "reports"
DRAFT = REPORTS / "draft.tex"  # the live draft: labels and the floats that stay in the main text
FULL = REPORTS / "archive" / "draft_full_pre_compression.tex"  # the draft before the compression: source of the SI prose and floats
FINDINGS = REPORTS / "energy_trajectory_findings.tex"
OUT = REPORTS / "supplement.tex"

# floats of the full draft that stay in the main text of the compressed paper (everything else goes to the SI); the Results of the live
# draft still hold all floats until their rewrite (step 3 of the plan), so this set is not computed from the live draft
MAIN_ONLY = {"fig:pipeline", "fig:rounds", "fig:depth", "fig:dose", "tab:minshots"}

# labels of the findings report that collide with labels of the draft: renamed inside the SI
RENAME_FINDINGS: dict = {}  # the findings figure fig:dose is no longer in the SI (the main text has the figure)

FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", re.S)
STOP = re.compile(r"\\paragraph\{|\\subsection\{|\\section\{|\\begin\{(?:table|figure)")

# SI sections: (title, [items]).  Items: ("text", latex) | ("sub", subsection-prefix[, label]) | ("subsec", title, label)
#   | ("para", region, prefix) | ("float", label) | ("file", path relative to reports/)
#   | ("fsec", findings-section-title-prefix) | ("fsub", findings-subsection-prefix) | ("fblock", findings-inline-name)
# "sub" copies a whole subsection of the full draft; "subsec" opens a new subsection that the following "para" items (paragraphs of the
# full draft) fill; "file" copies a hand-written part (S1, the extended methods).  A label is added to the subsection so that the main
# text can cite it as SI Sec.~\siref{<label>}.
# No duplicates: an item of the full draft (a float, a subsection or a paragraph) that the live draft still states verbatim is skipped,
# and a section whose items are all skipped is left out.  The SI therefore holds what the main text does not: the Results text and
# floats appear here once step 3 of the plan has cut them from the main text.
SECTIONS = [
    ("Notation, settings and extended methods", [
        ("file", "si_parts/s1_notation.tex"),
        ("file", "si_parts/s1_methods_a.tex"),
        ("file", "si_parts/s1_baselines.tex"),
        ("file", "si_parts/s1_methods_b.tex"),
    ], "sec:si1"),
    # Evidence for the claims of the Results (rebuild of 2026-10-26): hand-written blocks A to C, one claim ledger; block E is still the
    # section below (Total cost...) until it is rebuilt
    ("Evidence for the claims of the Results", [
        ("file", "si_parts/b0_ledger.tex"),
        ("file", "si_parts/b1_planning.tex"),
        ("file", "si_parts/b2_learned.tex"),
        ("file", "si_parts/b3_baselines.tex"),
        ("file", "si_parts/b3b_round4.tex"),
        ("file", "si_parts/b4_trajectory.tex"),
        ("file", "si_parts/b5_total.tex"),
    ], "sec:evidence"),
    # robustness and diagnostics of the comparison (to be rewritten: the copied paragraphs of the old Results are kept for now)
    # robustness and diagnostics of the comparison and of the optimizer model (hand-written parts c*; the fidelity paragraph and the Discussion
    # paragraphs are still the copies of the full draft)
    ("Robustness and diagnostics", [
        ("file", "si_parts/c1_shots.tex"),
        ("subsec", "Fidelity of the baselines", "sec:si_fidelity"),
        ("para", "Discussion", "What was compared, and how faithfully."),
        ("file", "si_parts/c2_depth.tex"),
        ("file", "si_parts/c3_optimizer.tex"),
        ("file", "si_parts/c4_stop.tex"),
    ], "sec:si4"),
    ("Implementation checks and reproduction", [
        ("file", "si_parts/s_checks.tex"),
        ("file", "si_parts/s_repro.tex"),
    ], "sec:si7"),
]

INLINE = re.compile(r"% BEGIN INLINE (?P<name>[A-Za-z0-9_]+)\n(?P<body>.*?)% END INLINE (?P=name)\n", re.S)


def expand_inline(text: str) -> str:
    """Fill the % BEGIN/END INLINE blocks of a hand-written part with the generated tables (reports/generated/<name>.tex)."""

    def fill(m: re.Match) -> str:
        path = REPORTS / "generated" / f"{m['name']}.tex"
        body = read(path) if path.exists() else m["body"]
        return f"% BEGIN INLINE {m['name']}\n{body}" + ("" if body.endswith("\n") else "\n") + f"% END INLINE {m['name']}\n"

    return INLINE.sub(fill, text)


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def strip_floats(text: str) -> str:
    return FLOAT.sub("", text)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collapse(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"


class Source:
    def __init__(self, draft: str, findings: str) -> None:
        self.draft = draft
        self.findings = findings
        self.floats = {}
        for m in FLOAT.finditer(draft):
            lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
            if lab:
                assert lab.group(1) not in self.floats, f"duplicate label {lab.group(1)}"
                self.floats[lab.group(1)] = m.group(0)

    # ---- draft
    def region(self, name: str) -> str:
        """Text of a part of the draft: a section name ('Methods', 'Discussion') or a Q subsection ('Q11:')."""
        if re.fullmatch(r"Q\d+:", name):
            return self.subsection(name)
        m = re.search(r"\\section\{" + re.escape(name), self.draft)
        assert m, f"section {name!r} not found"
        n = re.search(r"\\section\{", self.draft[m.end():])
        return self.draft[m.start(): m.end() + (n.start() if n else len(self.draft))]

    def subsection(self, prefix: str) -> str:
        ms = [m for m in re.finditer(r"\\subsection\{" + re.escape(prefix), self.draft)]
        assert len(ms) == 1, f"subsection {prefix!r}: {len(ms)} matches"
        m = ms[0]
        n = re.search(r"\\subsection\{|\\section\{|% BEGIN INLINE BIBLIOGRAPHY", self.draft[m.end():])
        return self.draft[m.start(): m.end() + (n.start() if n else len(self.draft))]

    def paragraph(self, region: str, prefix: str) -> str:
        text = self.region(region)
        ms = [m for m in re.finditer(r"\\paragraph\{" + re.escape(prefix), text)]
        assert len(ms) == 1, f"paragraph {prefix!r} in {region!r}: {len(ms)} matches"
        m = ms[0]
        n = STOP.search(text[m.end():])
        return text[m.start(): m.end() + (n.start() if n else len(text))]

    # ---- findings report
    def fsection(self, prefix: str) -> str:
        ms = [m for m in re.finditer(r"\\section\{" + re.escape(prefix), self.findings)]
        assert len(ms) == 1, f"findings section {prefix!r}: {len(ms)} matches"
        m = ms[0]
        n = re.search(r"\\section\{|\\appendix", self.findings[m.end():])
        body = self.findings[m.start(): m.end() + (n.start() if n else len(self.findings))]
        return body.replace("\\section{", "\\subsection{").replace("\\subsubsection{", "\\subsubsection{")

    def fsubsection(self, prefix: str) -> str:
        ms = [m for m in re.finditer(r"\\subsection\{" + re.escape(prefix), self.findings)]
        assert len(ms) == 1, f"findings subsection {prefix!r}: {len(ms)} matches"
        m = ms[0]
        n = re.search(r"\\subsection\{|\\section\{|\\appendix", self.findings[m.end():])
        return self.findings[m.start(): m.end() + (n.start() if n else len(self.findings))].replace("\\subsection{", "\\subsubsection{", 1)

    def fblock(self, name: str) -> str:
        m = re.search(r"% BEGIN INLINE " + re.escape(name) + r"\n.*?% END INLINE " + re.escape(name) + r"\n", self.findings, re.S)
        assert m, f"findings block {name!r} not found"
        return m.group(0)


def preamble(draft: str) -> str:
    head = draft[: draft.index("\\begin{document}")]
    # the draft cites the SI with \siref (numbers pasted by paper_a_si_refs.py); the SI cites the draft (xr-hyper, prefix M-) below
    head, n = re.subn(
        r"% numbers of the Supporting Information sections[^\n]*\n% from supplement\.aux[^\n]*\n\\newcommand\{\\siref\}[^\n]*\n"
        r"% BEGIN INLINE SIREFS\n.*?% END INLINE SIREFS\n",
        "",
        head,
        flags=re.S,
    )
    assert n == 1, "the draft preamble has no \\siref block"
    # the SI is a one-column document: its tables are wide and must stay legible (the draft uses the two-column reprint layout)
    head, n = re.subn(r"\\documentclass\[aps,pra,reprint,", r"\\documentclass[aps,pra,onecolumn,", head)
    assert n == 1, "documentclass line of the draft changed"
    extra = (
        "\\newcommand{\\Copt}{C_{\\mathrm{opt}}}\n"
        "\\newcommand{\\pending}{\\textcolor{red}{(pending)}}\n"
        "% table note below the tabular: details belong here, the caption is a title\n"
        "\\providecommand{\\tabnote}[1]{\\par\\smallskip{\\footnotesize\\raggedright #1\\par}}\n"
        "\\newcommand{\\notationref}{Table~\\ref{tab:notation}}  % the generated table notes cite the table of notation with this macro\n"
        "% a wrapping table cell: p-columns do not compile under revtex4-2 with this preamble\n"
        "\\newcommand{\\pc}[2]{\\parbox[t]{#1}{\\raggedright #2}}\n"
        "\\usepackage{placeins}  % \\FloatBarrier at the end of an evidence block keeps its tables in it\n"
        "\\input{draft_labels}  % labels of draft.tex with the prefix M- (static file of scripts/paper_a_draft_labels.py; works without draft.aux, e.g. on Overleaf)\n"
        "% table of contents: wide number boxes (S1.13)\n"
        "\\makeatletter\n"
        "\\renewcommand*\\l@section{\\@dottedtocline{1}{0em}{2.6em}}\n"
        "\\renewcommand*\\l@subsection{\\@dottedtocline{2}{1.6em}{3.4em}}\n"
        "\\renewcommand*\\l@subsubsection{\\@dottedtocline{3}{5em}{4.2em}}\n"
        "\\makeatother\n"
        "\\renewcommand{\\thesection}{S\\arabic{section}}\n"
        "\\renewcommand{\\thesubsection}{\\thesection.\\arabic{subsection}}\n"
        "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
        "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"
        "\\renewcommand{\\theequation}{S\\arabic{equation}}\n"
        "\\renewcommand{\\thesubsubsection}{\\thesubsection.\\arabic{subsubsection}}\n"
        "\\makeatletter\\renewcommand\\p@subsection{}\\renewcommand\\p@subsubsection{}\\makeatother\n"
        "% like \\fitcol, for tables that span the full text width\n"
        "\\newcommand{\\fitwide}[1]{\\sbox{\\fitbox}{#1}\\ifdim\\wd\\fitbox>\\textwidth\\resizebox{\\textwidth}{!}{\\usebox{\\fitbox}}\\else\\usebox{\\fitbox}\\fi}\n"
    )
    return head + extra


def build():
    draft, full, findings = read(DRAFT), read(FULL), read(FINDINGS)
    src = Source(full, findings)
    draft_labels = set(re.findall(r"\\label\{([^}]*)\}", draft))
    main_floats = set(MAIN_ONLY)
    assert main_floats <= set(src.floats), f"MAIN_ONLY names unknown floats: {sorted(main_floats - set(src.floats))}"
    assert main_floats <= draft_labels, f"MAIN_ONLY floats missing from the live draft: {sorted(main_floats - draft_labels)}"
    # the main text as it stands, without floats: what the SI must not repeat
    main_text = norm(strip_floats(draft[draft.index("\\begin{document}"): draft.index("% BEGIN INLINE BIBLIOGRAPHY")]))
    body: list[str] = []
    used: set[str] = set()
    src.skipped = []  # items of the full draft that the live draft still holds (they reach the SI when the main text drops them)
    for i, (title, items, *opt) in enumerate(SECTIONS, start=1):
        parts: list[str] = []
        n_content = 0
        for item in items:
            kind = item[0]
            if kind == "text":
                parts.append(item[1] + "\n")
            elif kind == "raw":
                parts.append(item[1])
                n_content += 1
            elif kind == "file":
                text = expand_inline(read(REPORTS / item[1]))

                def place(mm: re.Match) -> str:  # a float of the full draft inside a hand-written part
                    lab = mm.group(1)
                    assert lab in src.floats, f"%%FLOAT {lab!r}: not a float of the full draft"
                    assert lab not in main_floats and lab not in draft_labels, f"%%FLOAT {lab}: the float stays in the main text"
                    assert lab not in used, f"%%FLOAT {lab}: placed twice"
                    used.add(lab)
                    return src.floats[lab].rstrip("\n") + "\n"

                text = re.sub(r"(?m)^%%FLOAT (\S+)[ \t]*$", place, text)

                def tabular(mm: re.Match) -> str:  # the tabular of a generated table, as one panel of a float
                    path = REPORTS / "generated" / f"{mm.group(1)}.tex"
                    assert path.exists(), f"%%TABULAR {mm.group(1)}: reports/generated/{mm.group(1)}.tex does not exist"
                    found = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", read(path), re.S)
                    assert found, f"%%TABULAR {mm.group(1)}: no tabular in the file"
                    return "\\fitcol{%\n" + found.group(0) + "}"

                text = re.sub(r"(?m)^%%TABULAR (\S+)[ \t]*$", tabular, text)
                parts.append(text.rstrip("\n") + "\n")
                n_content += 1
            elif kind == "subhead":
                parts.append(f"\\subsection{{{item[1]}}}\n")
            elif kind == "subsec":
                parts.append(f"\\subsection{{{item[1]}}}\n\\label{{{item[2]}}}\n")
            elif kind in ("sub", "para"):
                if kind == "sub":
                    text = strip_floats(src.subsection(item[1]))
                    what = item[1]
                else:
                    text = strip_floats(src.paragraph(item[1], item[2])) + "\n"
                    what = item[1] + " / " + item[2]
                if norm(text) in main_text:
                    src.skipped.append(what)
                    continue
                if kind == "sub" and len(item) > 2:  # a label for SI Sec.~\siref{<label>} in the main text, after the heading line
                    text = re.sub(r"(\\subsection\{[^\n]*\}\n)", lambda m: m.group(1) + "\\label{%s}\n" % item[2], text, count=1)
                    assert "\\label{%s}" % item[2] in text
                parts.append("% source: full draft, " + what + "\n" + text)
                n_content += 1
            elif kind == "float":
                lab = item[1]
                assert lab in src.floats, f"float {lab!r} not in the full draft"
                assert lab not in main_floats, f"{lab} is a main-text float"
                if lab in draft_labels:  # the live draft still has it
                    src.skipped.append(lab)
                    continue
                used.add(lab)
                parts.append(src.floats[lab] + "\n")
                n_content += 1
            elif kind == "fsec":
                parts.append("% source: findings report, " + item[1] + "\n" + src.fsection(item[1]) + "\n")
                n_content += 1
            elif kind == "fsub":
                text = src.fsubsection(item[1])
                if not any("\\subsection{" in p for p in parts):  # no subsection above it (its Q-subsection is still in the main text)
                    text = text.replace("\\subsubsection{", "\\subsection{", 1)
                parts.append("% source: findings report, " + item[1] + "\n" + text + "\n")
                n_content += 1
            elif kind == "fblock":
                parts.append(src.fblock(item[1]) + "\n")
                n_content += 1
            else:
                raise AssertionError(kind)
        if n_content:  # a section whose items are all still in the main text is left out
            body.append(f"\n% {'=' * 100}\n% S{i}\n% {'=' * 100}\n\\section{{{title}}}\n\\label{{{opt[0] if opt else f'sec:si{i}'}}}\n")
            body.extend(parts)
    in_main = {lab for lab in src.floats if lab in draft_labels}
    missing = set(src.floats) - used - in_main - set(clean.DROP_FLOATS)
    assert not missing, f"floats of the full draft neither in the SI nor in the live draft: {sorted(missing)}"
    assert not (used & in_main), f"floats in both the SI and the live draft: {sorted(used & in_main)}"
    src.main_floats = main_floats
    text = clean.apply("".join(body))  # SI wording for the Results-derived parts (headings, captions, Q labels)

    # labels of the findings report that collide with the draft
    for old, new in RENAME_FINDINGS.items():
        text = text.replace("\\label{%s}" % old, "\\label{%s}" % new)
        text = re.sub(r"\\ref\{%s\}" % re.escape(old), "\\\\ref{%s}" % new, text)
    # findings figures were made for a full-width page: put them in the column
    text = text.replace("width=0.92\\textwidth", "width=\\columnwidth")
    # a p-column inside \fitcol does not compile under revtex4-2 ("Extra \or"); the generators use plain columns
    assert "{p{" not in re.sub(r"\\begin\{tabular\}\{@\{\}p\{", "", text), "a p-column table is in the SI"

    si_labels = re.findall(r"\\label\{([^}]*)\}", text)
    dup = {x for x in si_labels if si_labels.count(x) > 1}
    assert not dup, f"duplicate labels in the SI: {sorted(dup)}"
    si_set = set(si_labels)

    def fix(m: re.Match) -> str:
        cmd, lab = m.group(1), m.group(2)
        if lab in si_set or lab.startswith("M-"):
            return m.group(0)
        if lab in draft_labels:
            return "\\%s{M-%s}" % (cmd, lab)
        return m.group(0)

    text = re.sub(r"\\(ref|eqref|pageref)\{([^}]*)\}", fix, text)
    text = collapse(text)

    head = preamble(draft)
    front = (
        "\\begin{document}\n"
        "% ============================================================================================\n"
        "% TO BE COMPLETED BY THE AUTHORS: title, author list and affiliations as in draft.tex.\n"
        "% This file is generated by scripts/paper_a_make_si.py from archive/draft_full_pre_compression.tex and the findings report;\n"
        "% The sections are hand-written (reports/si_parts/*.tex: setup s1_*, evidence b*, robustness c*, checks s_*); the generated tables are pasted from reports/generated/;\n"
        "% only the paragraph on the fidelity of the baselines and the Discussion paragraphs of the last robustness subsection are still copied from the full draft.\n"
        "% ============================================================================================\n"
        # the title and the authors are those of the main text, so the two documents cannot disagree
        "\\title{Supporting Information for: " + re.search(r"\\title\{(.*)\}\n", draft).group(1) + "}\n"
        + re.search(r"\\author\{.*\}\n", draft).group(0)
        + "\\maketitle\n"  # no \date: revtex then prints no "Dated:" line
        "\\tableofcontents\n"
    )
    tail = "\n\\bibliographystyle{apsrev4-2}\n\\bibliography{references}\n\n\\end{document}\n"
    out = head + front + text + tail
    return out, src, si_set, draft_labels


def report(out: str, draft_labels: set[str], si_set: set[str]) -> None:
    refs = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]*)\}", out))
    undefined = sorted(r for r in refs if r not in si_set and not (r.startswith("M-") and r[2:] in draft_labels))
    words = len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", re.sub(r"% BEGIN INLINE (\w+)\n.*?% END INLINE \1\n", "", re.sub(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", "", out, flags=re.S), flags=re.S))))
    n_float_labels = len(re.findall(r"\\label\{(?:tab|fig):", out))
    print(f"SI: {len(out.splitlines())} lines, about {words} words of prose, {n_float_labels} table/figure labels")
    print("refs to M- labels:", sum(1 for r in refs if r.startswith("M-")))
    print("references to labels defined nowhere:", undefined if undefined else "none")
    mentions = [i + 1 for i, line in enumerate(out.splitlines()) if re.search(r"\bher\b|colleague|\bTZS\b", line)]
    print("lines that mention the colleague's study (decision D5):", mentions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paste-bbl", type=Path, default=None, help="paste this BibTeX output between the bibliography markers (in place)")
    args = parser.parse_args()
    if args.paste_bbl:
        text = read(OUT)
        bbl = read(args.paste_bbl)
        if not bbl.endswith("\n"):
            bbl += "\n"
        block = "% BEGIN INLINE BIBLIOGRAPHY\n" + bbl + "% END INLINE BIBLIOGRAPHY\n"
        if "% BEGIN INLINE BIBLIOGRAPHY" in text:
            text = re.sub(r"% BEGIN INLINE BIBLIOGRAPHY\n.*?% END INLINE BIBLIOGRAPHY\n", lambda _: block, text, flags=re.S)
        else:
            old = "\\bibliographystyle{apsrev4-2}\n\\bibliography{references}\n"
            assert text.count(old) == 1
            text = text.replace(old, block)
        OUT.write_bytes(text.encode("utf-8"))
        print("pasted", args.paste_bbl, "into", OUT)
        return
    out, src, si_set, draft_labels = build()
    if OUT.exists():  # keep the inline bibliography of the previous version (rebuild it with --paste-bbl after a change of citations)
        old = re.search(r"% BEGIN INLINE BIBLIOGRAPHY\n.*?% END INLINE BIBLIOGRAPHY\n", read(OUT), re.S)
        tail = "\\bibliographystyle{apsrev4-2}\n\\bibliography{references}\n"
        if old and out.count(tail) == 1:
            out = out.replace(tail, old.group(0))
            print("kept the inline bibliography of the existing file")
    OUT.write_bytes(out.encode("utf-8"))
    print("wrote", OUT)
    print(f"{len(src.skipped)} items of the full draft are still in the main text and not repeated here (they reach the SI when the main text drops them)")
    report(out, draft_labels, si_set)


if __name__ == "__main__":
    sys.exit(main())
