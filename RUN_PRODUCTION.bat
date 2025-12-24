@echo off
echo ============================================
echo   AI Anime Video Creator - PRODUCTION
echo ============================================
echo.

:: Python path
set PYTHON=C:\Users\cchol\AppData\Local\Programs\Python\Python314\python.exe
set PIP=C:\Users\cchol\AppData\Local\Programs\Python\Python314\Scripts\pip.exe

:: Check Python
if not exist "%PYTHON%" (
    echo ERROR: Python not found
    pause
    exit /b 1
)

echo Python: %PYTHON%
%PYTHON% --version
echo.

cd /d "%~dp0"

:: Create dirs
echo Creating directories...
if not exist "outputs" mkdir outputs
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs
echo Done.
echo.

:: Install dependencies
echo ============================================
echo   Installing Dependencies...
echo ============================================
echo.

echo Installing core packages...
%PIP% install --upgrade pip --quiet

echo Installing Gradio...
%PIP% install gradio --quiet

echo Installing TTS...
%PIP% install edge-tts --quiet

echo Installing video processing...
%PIP% install imageio imageio-ffmpeg pillow numpy --quiet

echo Installing moviepy (for audio merge)...
%PIP% install moviepy --quiet

echo Installing audio tools...
%PIP% install mutagen --quiet

echo Installing translation...
%PIP% install googletrans==4.0.0-rc1 --quiet

echo Installing utilities...
%PIP% install loguru --quiet

echo.
echo ============================================
echo   All Dependencies Installed!
echo ============================================
echo.

echo ============================================
echo   Starting Application...
echo ============================================
echo.
echo FEATURES:
echo   [+] Advanced Animated Video Generation
echo   [+] Multi-Language Translation (6+ languages)
echo   [+] Auto-translate script to voice language
echo   [+] Professional UI - All buttons working
echo   [+] Voice preview and templates
echo   [+] Share functionality enabled
echo.
echo LANGUAGES SUPPORTED:
echo   Japanese, Hindi, Kannada, English, Korean, Chinese
echo.
echo SERVER: http://localhost:7860
echo Press Ctrl+C to stop
echo.

:: Set encoding
chcp 65001 >nul

:: Run app
%PYTHON% app_final.py

pause

