@echo off
REM Patent Application LaTeX Build Script
REM Requires: MiKTeX or TeX Live installed and in PATH
REM Install MiKTeX: https://miktex.org/download

echo Building patent_application.pdf ...

REM First pass (resolves cross-references)
pdflatex -interaction=nonstopmode patent_application.tex

REM Bibliography pass
bibtex patent_application

REM Second and third pass (resolves all references)
pdflatex -interaction=nonstopmode patent_application.tex
pdflatex -interaction=nonstopmode patent_application.tex

echo.
echo Done. Output: patent_application.pdf
echo.

REM Open PDF if possible
if exist patent_application.pdf (
    start patent_application.pdf
)
