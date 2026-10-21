"""The Supporting Information generator (scripts/paper_a_make_si.py) places every float of the full draft and keeps its references."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_a_make_si as si  # noqa: E402
import paper_a_si_refs as refs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (si.DRAFT.exists() and si.FULL.exists() and si.FINDINGS.exists()), reason="reports/ is not part of this checkout"
)


def test_every_float_of_the_full_draft_is_in_the_si_or_in_the_main_text():
    out, src, si_set, draft_labels = si.build()  # build() stops with an AssertionError for a float that has no decision
    in_si = {lab for lab in si_set if lab.startswith(("tab:", "fig:"))}
    for lab in src.floats:  # in the SI or still in the live draft: never in neither (lost) and never in both (repeated)
        if lab in si.clean.DROP_FLOATS:  # replaced by a float of the main text
            assert lab not in in_si, lab
            continue
        assert (lab in in_si) != (lab in draft_labels), lab
    assert si.MAIN_ONLY <= draft_labels, "a float that stays in the main text is missing from the live draft"


def test_the_si_repeats_no_sentence_and_no_equation_of_the_main_text():
    out = si.build()[0]
    clean = lambda t: si.norm(re.sub(r"(?m)^%.*$", "", re.sub(r"% BEGIN INLINE (\w+)\n.*?% END INLINE \1\n", "", si.strip_floats(t), flags=re.S)))
    draft = si.read(si.DRAFT)
    main = clean(draft[draft.index("\\begin{document}"): draft.index("% BEGIN INLINE BIBLIOGRAPHY")])
    body = clean(out[out.index("\\begin{document}"): out.index("% BEGIN INLINE BIBLIOGRAPHY") if "% BEGIN INLINE BIBLIOGRAPHY" in out else len(out)]).replace("{M-", "{")
    sentences = [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z\\$(])", body) if len(re.findall(r"[A-Za-z]{2,}", s)) >= 8]
    repeated = [s[:100] for s in sentences if s in main]
    assert not repeated, repeated[:5]
    label = lambda e: re.sub(r"\\label\{[^}]*\}", "", e)
    equations = [m.group(0) for m in re.finditer(r"\\begin\{(equation|gather|align)\}.*?\\end\{\1\}", out, flags=re.S)]
    in_main = [e[:80] for e in equations if si.norm(label(e)) in si.norm(label(draft))]
    assert not in_main, in_main[:5]


def test_references_from_the_draft_into_the_si_exist():
    """\\siref{x} in draft.tex must name a label of the generated SI."""
    out, src, si_set, draft_labels = si.build()
    cited = set(refs.cited_labels(si.read(si.DRAFT)))
    assert cited, "the draft cites no SI section: the compressed Methods should"
    assert not cited - si_set, sorted(cited - si_set)


def test_si_numbers_pasted_into_the_draft_match_the_si_source():
    """The block between % BEGIN/END INLINE SIREFS has one definition per cited label, and the numbers of the sections are the
    ones counted in supplement.tex (a stale block means: rerun paper_a_si_refs.py with the aux of a new SI build)."""
    draft = si.read(si.DRAFT)
    block = refs.BLOCK.search(draft)
    assert block, "no SIREFS block in the draft"
    pasted = dict(refs.DEF.findall(block.group(0)))
    assert set(pasted) == set(refs.cited_labels(draft)), "the pasted labels differ from the cited ones"
    static = refs.static_numbers(si.read(si.OUT))
    for lab, number in pasted.items():
        if lab.startswith("sec:"):
            assert static[lab] == number, (lab, static[lab], number)
    assert "xr-hyper" not in draft.split("\\begin{document}")[0], "the draft must not need supplement.aux"


def test_references_to_the_main_text_exist_in_the_draft():
    out, src, si_set, draft_labels = si.build()
    refs = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]*)\}", out))
    undefined = [r for r in refs if r not in si_set and not (r.startswith("M-") and r[2:] in draft_labels)]
    assert not undefined, undefined


def test_no_paragraph_column_in_a_float():
    out = si.build()[0]
    assert not re.search(r"\\begin\{tabular\}\{[^}]*p\{", out)


def test_generation_is_deterministic():
    assert si.build()[0] == si.build()[0]
