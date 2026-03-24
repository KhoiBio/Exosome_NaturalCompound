#!/bin/bash
# ============================================================
# ExoRAG — Mac / Linux Setup Script
# Run: bash setup_mac_linux.sh
# ============================================================

set -e

echo ""
echo " ====================================================="
echo "  ExoRAG — Setup (Mac / Linux)"
echo " ====================================================="
echo ""

# ── Check Python ─────────────────────────────────────────────

echo "[1/5] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo " ERROR: python3 not found."
    echo ""
    echo " Mac:   brew install python@3.11"
    echo "        or download from https://python.org/downloads"
    echo " Linux: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

PYVER=$(python3 --version | awk '{print $2}')
echo " Found Python $PYVER"

# ── Create virtual environment ────────────────────────────────

echo ""
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo " venv/ already exists, skipping."
else
    python3 -m venv venv
    echo " Created venv/"
fi

source venv/bin/activate
echo " Virtual environment activated."

# ── Upgrade pip ───────────────────────────────────────────────

echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip --quiet
echo " Done."

# ── Install dependencies ──────────────────────────────────────

echo ""
echo "[4/5] Installing dependencies..."
pip install -r requirements.txt

# Mac Apple Silicon fallback for chromadb
if [[ "$(uname -m)" == "arm64" ]]; then
    echo " Detected Apple Silicon — checking chromadb..."
    pip install chromadb --no-binary chromadb 2>/dev/null || true
fi

echo " Done."

# ── Create .env ───────────────────────────────────────────────

echo ""
echo "[5/5] Setting up .env..."
mkdir -p data outputs chroma_db

if [ -f ".env" ]; then
    echo " .env already exists, skipping."
else
    cp .env.example .env
    echo " Created .env from template."
fi

# ── Done ──────────────────────────────────────────────────────

echo ""
echo " ====================================================="
echo "  Setup Complete!"
echo " ====================================================="
echo ""
echo " NEXT STEPS:"
echo ""
echo " 1. Edit .env — add your Claude API key:"
echo "    nano .env   (or open in any text editor)"
echo "    ANTHROPIC_API_KEY=sk-ant-..."
echo "    Get a key at: https://console.anthropic.com/"
echo ""
echo " 2. Fetch PubMed papers (run once, ~5-10 min):"
echo "    python scripts/01_fetch_pubmed.py"
echo ""
echo " 3. Build local database (run once, ~15-25 min):"
echo "    python scripts/02_build_index.py"
echo ""
echo " 4. Launch the app:"
echo "    streamlit run scripts/04_app.py"
echo "    Then open: http://localhost:8501"
echo ""
