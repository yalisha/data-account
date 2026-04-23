#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MANUSCRIPT_DIR="$ROOT_DIR/manuscript"
SECTIONS_DIR="$SCRIPT_DIR/sections"

mkdir -p "$SECTIONS_DIR"

convert() {
  local src="$1"
  local dest="$2"
  pandoc \
    -f gfm+tex_math_dollars \
    -t latex \
    --syntax-highlighting=none \
    --wrap=none \
    --shift-heading-level-by=-1 \
    "$src" \
    -o "$dest"
}

convert "$MANUSCRIPT_DIR/01_引言.md" "$SECTIONS_DIR/01_introduction.tex"
convert "$MANUSCRIPT_DIR/02_理论与假说.md" "$SECTIONS_DIR/02_theory.tex"
convert "$MANUSCRIPT_DIR/03_研究设计.md" "$SECTIONS_DIR/03_design.tex"
convert "$MANUSCRIPT_DIR/04_实证检验.md" "$SECTIONS_DIR/04_results.tex"
convert "$MANUSCRIPT_DIR/05_进一步分析.md" "$SECTIONS_DIR/05_further_analysis.tex"
convert "$MANUSCRIPT_DIR/06_异质性分析.md" "$SECTIONS_DIR/06_heterogeneity.tex"
convert "$MANUSCRIPT_DIR/07_结论.md" "$SECTIONS_DIR/07_conclusion.tex"
convert "$MANUSCRIPT_DIR/09_参考文献.md" "$SECTIONS_DIR/09_references.tex"
convert "$MANUSCRIPT_DIR/附录.md" "$SECTIONS_DIR/appendix.tex"

python3 - <<'PY'
from pathlib import Path


def wrap_landscape_table(path_str: str, title: str, begin_line: str, tabcolsep: str = "3pt") -> None:
    path = Path(path_str)
    lines = path.read_text().splitlines()

    title_idx = next(i for i, line in enumerate(lines) if title in line)
    def_idx = next(i for i in range(title_idx + 1, len(lines)) if lines[i].startswith(r"{\def\LTcaptype{}"))
    begin_idx = next(i for i in range(def_idx + 1, len(lines)) if lines[i].startswith(r"\begin{longtable}"))
    end_idx = next(i for i in range(begin_idx + 1, len(lines)) if lines[i].strip() == r"\end{longtable}")
    close_idx = next(i for i in range(end_idx + 1, len(lines)) if lines[i].strip() == "}")

    lines[begin_idx] = begin_line
    prefix = [
        r"\begin{landscape}",
        r"\begingroup",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\scriptsize",
    ]
    suffix = [
        r"\endgroup",
        r"\end{landscape}",
    ]

    lines[def_idx:def_idx] = prefix
    close_idx += len(prefix)
    lines[close_idx + 1:close_idx + 1] = suffix
    path.write_text("\n".join(lines) + "\n")


wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/04_results.tex",
    "表6 构念边界与测度补强",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{3.0cm}*{7}{>{\centering\arraybackslash}p{2.0cm}}@{}}",
)
wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/04_results.tex",
    "表6A 增强测度与章节限定口径检验",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{3.2cm}*{6}{>{\centering\arraybackslash}p{2.3cm}}@{}}",
)
wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/04_results.tex",
    "表7 时间顺序与内生性识别",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{3.0cm}*{5}{>{\centering\arraybackslash}p{2.55cm}}@{}}",
)
wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/04_results.tex",
    "表9 稳健性检验",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{3.0cm}*{7}{>{\centering\arraybackslash}p{2.05cm}}@{}}",
)
wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/appendix.tex",
    "表A1 替代切点与补充异质性检验",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{4.2cm}*{6}{>{\centering\arraybackslash}p{2.0cm}}@{}}",
)
wrap_landscape_table(
    "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/appendix.tex",
    "表A2 增强测度的补充结果",
    r"\begin{longtable}[]{@{}>{\raggedright\arraybackslash}p{2.6cm}>{\centering\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{2.4cm}>{\centering\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{3.5cm}@{}}",
)
PY

python3 - <<'PY'
from pathlib import Path

path = Path("/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex/sections/04_results.tex")
text = path.read_text()
text = text.replace(
    r"\pandocbounded{\includegraphics[keepaspectratio]{results/v16_tables/psm_density.png}}",
    "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.78\\linewidth]{results/v16_tables/psm_density.png}\n\\end{figure}",
)
path.write_text(text)
PY
