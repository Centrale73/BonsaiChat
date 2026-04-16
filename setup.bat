@echo off
:: BonsaiChat — One-click setup (Windows)
:: agent-zero-features branch
:: Run this once before launching BonsaiChat for the first time.

setlocal enabledelayedexpansion
echo.
echo  ========================================
echo   BonsaiChat Setup
echo  ========================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)
echo [OK] Python found.

:: ── 2. Create venv ───────────────────────────────────────────────────────
if not exist ".venv" (
    echo [*] Creating virtual environment...
    python -m venv .venv
)
echo [OK] Virtual environment ready.

:: ── 3. Install dependencies ──────────────────────────────────────────────
echo [*] Installing Python dependencies (this may take a few minutes)...
call .venv\Scripts\pip install --upgrade pip --quiet
call .venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause & exit /b 1
)
echo [OK] Dependencies installed.

:: ── 4. Create required directories ───────────────────────────────────────
if not exist "bin"         mkdir bin
if not exist "models"      mkdir models
if not exist "memory_data" mkdir memory_data
echo [OK] Directories ready.

:: ── 5. llama-server check ────────────────────────────────────────────────
if not exist "bin\llama-server.exe" (
    echo.
    echo [NOTICE] bin\llama-server.exe not found.
    echo   Download it from: https://github.com/ggml-org/llama.cpp/releases
    echo   Place the llama-server.exe inside the bin\ folder.
    echo.
) else (
    echo [OK] llama-server.exe found.
)

:: ── 6. .env from template ────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [OK] .env created from template.
    )
)

echo.
echo  ========================================
echo   Setup complete!
echo.
echo   To run BonsaiChat:
echo     .venv\Scripts\python app.py
echo.
echo   To build the .exe:
echo     .venv\Scripts\pyinstaller BonsaiChat.spec
echo  ========================================
echo.
pause
