#!/usr/bin/env python3
"""Copy built displays to a chosen asset directory and validate manuscript inputs.

Run after build_all.py. Table bodies are copied verbatim; the only derived file is the
row block of Table 1(b), which main.tex places between booktabs rules of a combined table.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "artifacts"
PAPER = HERE.parent / "build/paper-artifacts"
FIGURES = ["ab_fig2", "ab_rater_noise", "ab_skewness", "bon_prompt_conditional_scope",
           "rm_fig_worked_example", "fig1_workflow"]


def table_rows(source: Path) -> str:
    """Rows between the header's \\midrule and \\bottomrule of a generated tabular."""
    lines = source.read_text().splitlines()
    start, end = lines.index(r"\midrule"), lines.index(r"\bottomrule")
    return "\n".join(lines[start + 1:end]) + "\n"


def main() -> None:
    global OUT, PAPER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUT,
                        help="Output directory produced by build_all.py")
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    args = parser.parse_args()
    OUT, PAPER = args.outdir.resolve(), args.paper_dir.resolve()
    for kind in ("tables", "figures"):
        (PAPER / kind).mkdir(parents=True, exist_ok=True)
    for table in sorted((OUT / "tables").glob("*.tex")):
        shutil.copyfile(table, PAPER / "tables" / table.name)
    for name in FIGURES:
        shutil.copyfile(OUT / "figures" / f"{name}.pdf", PAPER / "figures" / f"{name}.pdf")
    (PAPER / "tables" / "flex_table1b_prepool_rows.tex").write_text(
        table_rows(OUT / "tables" / "flex_table1b_prepool.tex"))

    loaded = set()
    for tex in ("main.tex", "appendix.tex", "checklist.tex"):
        source = PAPER / tex
        if not source.exists():
            continue
        text = source.read_text()
        loaded |= set(re.findall(r"\\(?:input|tablerows)\{(tables/[^}]+)\}", text))
        loaded |= {f"{p}" for p in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)}
    missing = sorted(p for p in loaded if not (PAPER / p).exists())
    if missing:
        sys.exit("missing manuscript inputs:\n  " + "\n  ".join(missing))
    print(f"synced {len(list((OUT / 'tables').glob('*.tex')))} tables and {len(FIGURES)} figures; "
          f"{len(loaded)} manuscript inputs present")


if __name__ == "__main__":
    main()
