#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/docs/report"
REPORT_TEX="$REPORT_DIR/analysis_report.tex"

if [[ ! -f "$REPORT_TEX" ]]; then
  echo "[error] missing report source: $REPORT_TEX" >&2
  exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
  cd "$REPORT_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error analysis_report.tex
elif command -v pdflatex >/dev/null 2>&1; then
  cd "$REPORT_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error analysis_report.tex
  pdflatex -interaction=nonstopmode -halt-on-error analysis_report.tex
elif command -v tectonic >/dev/null 2>&1; then
  cd "$REPORT_DIR"
  tectonic analysis_report.tex
else
  cat >&2 <<'EOF'
[error] No LaTeX builder found.

Install one of these first:
  sudo apt-get update
  sudo apt-get install -y latexmk texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-pictures lmodern

Then rerun:
  bash scripts/build_report.sh
EOF
  exit 1
fi

echo "[ok] built $REPORT_DIR/analysis_report.pdf"
