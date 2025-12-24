@echo off
chcp 65001 >nul
echo ============================================
echo   AI Anime Video Creator
echo ============================================
echo.
echo Starting application...
echo Open your browser to: http://localhost:7860
echo.

set PYTHON=C:\Users\cchol\AppData\Local\Programs\Python\Python314\python.exe
cd /d "%~dp0"

"%PYTHON%" frontend\app.py

pause

