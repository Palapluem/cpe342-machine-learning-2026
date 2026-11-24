# CPE 342 Term Project Report — Karena Player Intelligence System

Technical report source and print-ready PDF for **Team ML004** (*CPE 342 Machine Learning*, Semester 1/2026, KMUTT).

## Package Structure

| Path | Purpose |
|---|---|
| `report.tex` | Main XeLaTeX technical report source (English, Sarabun 12 pt, Consolas monospace) |
| `frontpage/frontpage.tex` | Official KMUTT cover page (`Team: ML004`, members from `AGENTS.md`, Dr. Boonyarit Changaival) |
| `frontpage/frontpage.example.tex` | Reusable cover-page template |
| `frontpage/assets/` | KMUTT logo (`KMUTT_Logo.png`, `KMUTT_Logo.svg`) |
| `assets/` | 16 copied 300-dpi diagnostic & alignment figures from `project/task1..5/assets/` |
| `evidence/README.md` | Artifact checksums, figure provenance, and documented source conflicts |
| `.latexmkrc` | Configures `latexmk` to compile with `xelatex` |
| `output/pdf/CPE342_TermProject_Report.pdf` | Final print-ready A4 PDF |

## Build Instructions

Requires `xelatex`, `latexmk`, and system fonts `Sarabun` and `Consolas`. Build into a temporary directory so intermediate `.aux`/`.log`/`.toc` files never pollute `project/report/`:

```sh
cd project/report
build_dir="$(mktemp -d -t cpe342-report)"
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" -jobname=CPE342_TermProject_Report report.tex
cp "$build_dir/CPE342_TermProject_Report.pdf" output/pdf/
rm -rf "$build_dir"
```
