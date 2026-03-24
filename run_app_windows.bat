@echo off
:: ============================================================
:: ExoRAG — Launch App (Windows)
:: Double-click this after setup is complete
:: ============================================================

title ExoRAG
color 0B

echo.
echo  Starting ExoRAG...
echo  Opening at: http://localhost:8501
echo.
echo  Press Ctrl+C to stop the app.
echo.

:: Activate virtual environment
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo  ERROR: Virtual environment not found.
    echo  Please run setup_windows.bat first.
    pause
    exit /b 1
)

:: Check chroma_db exists
if not exist chroma_db\* (
    echo  WARNING: Database not found.
    echo  Run these first:
    echo    python scripts\01_fetch_pubmed.py
    echo    python scripts\02_build_index.py
    echo.
    pause
    exit /b 1
)

:: Launch app
streamlit run scripts\04_app.py
