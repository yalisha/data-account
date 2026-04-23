#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

"$SCRIPT_DIR/regen_sections.sh"

cd "$SCRIPT_DIR"
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
cp main.pdf manuscript_elsevier_v15.pdf
