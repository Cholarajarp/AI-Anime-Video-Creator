@echo off
echo ============================================
echo   AI Anime Video Creator - Enhanced Version
echo ============================================
echo.

:: Set Python path
set PYTHON_PATH=C:\Users\cchol\AppData\Local\Programs\Python\Python314
set PYTHON=%PYTHON_PATH%\python.exe
set PIP=%PYTHON_PATH%\Scripts\pip.exe

:: Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo Found Python at: %PYTHON%
%PYTHON% --version
echo.

:: Navigate to project directory
cd /d "%~dp0"

:: Create directories
echo Creating directories...
if not exist "outputs" mkdir outputs
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs
echo Done.
echo.

:: Install core dependencies
echo ============================================
echo   Installing Core Dependencies...
echo ============================================
echo.

echo Installing essential packages...
%PIP% install --upgrade pip
%PIP% install -r requirements_core.txt

echo.
echo ============================================
echo   Starting Enhanced Application...
echo ============================================
echo.
echo Features:
echo   - Multi-language TTS (Japanese, Hindi, Kannada, etc.)
echo   - Placeholder video generation (works without ComfyUI)
echo   - Enhanced modern UI
echo   - Generation history tracking
echo   - Voice preview
echo   - Share functionality
echo.
echo Open your browser to: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

:: Set encoding for Unicode support
chcp 65001 >nul

:: Run the enhanced application
%PYTHON% frontend\app_enhanced.py

pause

