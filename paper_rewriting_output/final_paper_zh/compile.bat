@echo off
chcp 65001 >nul
set PATH=D:\latex\miktex\miktex\bin\x64;%PATH%

echo ========== 第1步：xelatex 编译 ==========
xelatex -interaction=nonstopmode main.tex

echo ========== 第2步：bibtex 处理参考文献 ==========
bibtex main

echo ========== 第3步：xelatex 再次编译（解析引用） ==========
xelatex -interaction=nonstopmode main.tex

echo ========== 第4步：xelatex 最终编译（交叉引用到位） ==========
xelatex -interaction=nonstopmode main.tex

echo ========== 第5步：清理编译临时文件 ==========
del /f /q main.aux main.bbl main.blg main.log main.out main.synctex.gz 2>nul
echo ========== 编译完成！ ==========
pause
