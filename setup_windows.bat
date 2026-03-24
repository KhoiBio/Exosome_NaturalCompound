@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: ExoRAG — Windows Auto Setup Script
:: Double-click this file to set up ExoRAG on Windows
:: ============================================================

title ExoRAG Setup
color 0A

echo.
echo  =====================================================
echo   ExoRAG — Exosome / Cosmetic / Natural Compound RAG
echo   Windows Setup Script
echo  =====================================================
echo.

:: ── Check Python ─────────────────────────────────────────────

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found.
    echo.
    echo  Please install Python 3.11 from:
    echo  https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: On the installer first screen,
    echo  check "Add Python to PATH" before clicking Install Now.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%

:: Check version is 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if %PYMAJ% LSS 3 (
    echo  ERROR: Python 3.10 or higher required. Found %PYVER%
    pause
    exit /b 1
)
if %PYMAJ% EQU 3 if %PYMIN% LSS 10 (
    echo  ERROR: Python 3.10 or higher required. Found %PYVER%
    pause
    exit /b 1
)
echo  OK

:: ── Create virtual environment ────────────────────────────────

echo.
echo [2/5] Creating Python virtual environment...
if exist venv (
    echo  venv\ already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Created venv\
)

:: ── Activate venv ─────────────────────────────────────────────

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment activated.

:: ── Upgrade pip ───────────────────────────────────────────────

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded.

:: ── Install dependencies ──────────────────────────────────────

echo.
echo [4/5] Installing dependencies (this may take 3-5 minutes)...
echo  Installing: biopython, chromadb, sentence-transformers,
echo              anthropic, streamlit, pandas, tqdm ...
echo.

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  WARNING: Some packages may have failed.
    echo  Trying chromadb with --no-binary flag...
    python -m pip install chromadb --no-binary chromadb
)

echo.
echo  Dependencies installed.

:: ── Create .env if not exists ─────────────────────────────────

echo.
echo [5/5] Setting up configuration file...
if exist .env (
    echo  .env already exists, skipping.
) else (
    copy .env.example .env >nul
    echo  Created .env from template.
)

:: Create data and outputs directories
if not exist data mkdir data
if not exist outputs mkdir outputs
if not exist chroma_db mkdir chroma_db

:: ── Done ──────────────────────────────────────────────────────

echo.
echo  =====================================================
echo   Setup Complete!
echo  =====================================================
echo.
echo  NEXT STEPS:
echo.
echo  1. Add your Claude API key to .env:
echo     - Open .env in Notepad
echo     - Replace sk-ant-your-key-here with your real key
echo     - Get a key at: https://console.anthropic.com/
echo     - Also set your ENTREZ_EMAIL to your real email
echo.
echo  2. Fetch PubMed literature (run once, ~5-10 min):
echo     python scripts\01_fetch_pubmed.py
echo.
echo  3. Build local database (run once, ~15-25 min):
echo     python scripts\02_build_index.py
echo.
echo  4. Launch the app:
echo     streamlit run scripts\04_app.py
echo     Then open: http://localhost:8501
echo.
echo  NOTE: Steps 2 and 3 only need to run ONCE.
echo  After that, just run step 4 to use the app.
echo.
echo  See WINDOWS_SETUP.md for full instructions and troubleshooting.
echo.
pause
