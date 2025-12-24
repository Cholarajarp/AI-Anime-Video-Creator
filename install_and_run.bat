@echo off
echo ============================================
echo   AI Anime Video Creator - Setup and Run
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
echo Working directory: %CD%
echo.

:: Create directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "outputs" mkdir outputs
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\motion_modules" mkdir models\motion_modules
if not exist "models\loras" mkdir models\loras
if not exist "models\vae" mkdir models\vae
echo Done.
echo.

:: Install dependencies
echo ============================================
echo   Installing Dependencies...
echo ============================================
echo.

%PIP% install --upgrade pip

:: Install core packages one by one to avoid conflicts
echo Installing Gradio...
%PIP% install gradio>=4.44.0

echo Installing FastAPI...
%PIP% install fastapi uvicorn python-multipart

echo Installing Edge-TTS...
%PIP% install edge-tts

echo Installing FFmpeg-Python...
%PIP% install ffmpeg-python imageio imageio-ffmpeg

echo Installing other dependencies...
%PIP% install pydub aiofiles aiohttp httpx websockets websocket-client
%PIP% install pydantic pydantic-settings python-dotenv pyyaml
%PIP% install loguru rich pillow

echo.
echo ============================================
echo   Dependencies Installed!
echo ============================================
echo.

:: Check FFmpeg
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: FFmpeg not found in PATH
    echo Please install FFmpeg from https://ffmpeg.org/
    echo Or use: winget install ffmpeg
    echo.
)

echo.
echo ============================================
echo   Starting Application...
echo ============================================
echo.
echo Open your browser to: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

:: Set encoding for Unicode support
chcp 65001 >nul

:: Run the application directly
%PYTHON% frontend\app.py

pause

