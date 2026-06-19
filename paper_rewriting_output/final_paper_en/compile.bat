@echo off
chcp 65001 >nul
set PATH=D:\latex\miktex\miktex\bin\x64;%PATH%

echo ========== Step 1: xelatex compile ==========
xelatex -interaction=nonstopmode main.tex

echo ========== Step 2: bibtex references ==========
bibtex main

echo ========== Step 3: xelatex (resolve citations) ==========
xelatex -interaction=nonstopmode main.tex

echo ========== Step 4: xelatex final (cross-refs) ==========
xelatex -interaction=nonstopmode main.tex

echo ========== Step 5: Clean temp files ==========
del /f /q main.aux main.bbl main.blg main.log main.out main.synctex.gz 2>nul
echo ========== Compilation Complete! ==========
pause
