#!/usr/bin/env python3
"""Re-splice the generated Q8 and Discussion blocks of ``reports/draft.tex`` without touching the Q9-Q12 block.

``paper_a_q8_text.py --paper`` rewrites everything from Q8 to the Discussion heading, and the Q9-Q12 block (between
``% BEGIN REFEREE RESULTS`` and ``% END REFEREE RESULTS``) sits in that range.  This script cuts the block out, runs the three
splicers (``paper_a_sota_latex.py``, ``paper_a_q8_text.py``, ``paper_a_discussion_tables.py``) on the rest, and puts the block back in
front of the Discussion.  Run ``paper_a_sota_tables.py`` and ``paper_a_depth_tables.py`` first, and ``paper_a_inline.py`` after.

    python scripts/paper_a_splice_all.py [--draft ../reports/draft.tex] [--check]     # --check: write to a scratch copy only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEGIN, END = "% BEGIN REFEREE RESULTS\n", "% END REFEREE RESULTS\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=HERE.parents[1] / "reports" / "draft.tex")
    parser.add_argument("--check", action="store_true", help="leave the draft alone; print the path of the spliced copy")
    args = parser.parse_args()
    text = args.draft.read_text(encoding="utf-8").replace("\r\n", "\n")
    a, b = text.index(BEGIN), text.index(END) + len(END)
    block = text[a:b]
    work = Path(tempfile.mkdtemp()) / "draft_spliced.tex"
    work.write_text(text[:a] + text[b:].lstrip("\n"), encoding="utf-8", newline="\n")
    for script, flags in (("paper_a_sota_latex.py", ["--splice"]), ("paper_a_q8_text.py", ["--paper"]), ("paper_a_discussion_tables.py", ["--splice"])):
        done = subprocess.run([sys.executable, str(HERE / script), *flags, str(work)], capture_output=True, text=True, cwd=HERE.parent)
        print(f"{script}: {'ok' if done.returncode == 0 else 'FAILED'}  {(done.stdout.strip().splitlines() or [''])[-1][:160]}")
        if done.returncode:
            print(done.stderr[-800:])
            sys.exit(1)
    out = work.read_text(encoding="utf-8").replace("\r\n", "\n")
    marker = "\\section{Discussion}"
    assert out.count(marker) == 1
    out = out.replace(marker, block + "\n" + marker)
    if args.check:
        work.write_text(out, encoding="utf-8", newline="\n")
        print(work)
    else:
        args.draft.write_text(out, encoding="utf-8", newline="\n")
        print(f"spliced {args.draft}")


if __name__ == "__main__":
    main()
