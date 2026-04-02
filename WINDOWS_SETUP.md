# ExoRAG — Windows Setup Guide
### Complete instructions for running ExoRAG on Windows 10/11

---

## What You Need Before Starting

- [ ] Windows 10 or 11
- [ ] Internet connection
- [ ] An Anthropic Claude API key (get one free at https://console.anthropic.com)
- [ ] ~1GB free disk space
- [ ] 20–30 minutes for first-time setup
- [ ] Ollama
- [ ] Python 3.10 +

---
## Short list of what need to be done in order 
## if you want to save time

1. Start PowerShell
2. cd to the downloaded folder
3. cd Exosome_NaturalCompound-main
4. run setup for the window with .\setup_windows.bat
5. edit claude api and PubMed email in .env via Notepad
6. bypass window security: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
7. .\venv\Scripts\activate
8. install requirement: pip install biopython requests chromadb sentence-transformers anthropic streamlit pandas tqdm python-dotenv
9. fetch pubmed: python scripts\01_fetch_pubmed.py
10. build chromaDB local index: python .\scripts\02_build_index.py
11. Download Ollama for free local LLM:https://ollama.com/
12. prepare Ollama : ollama pull llama3.2:latest
13. initialize the app : streamlit run scripts/04_app.py





## Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (click the yellow "Download Python 3.11.x" button)
3. Run the installer
4. **CRITICAL:** On the first screen, check ✅ **"Add Python to PATH"** before clicking Install
5. Click **"Install Now"**
6. When done, click **"Disable path length limit"** if prompted (recommended)

**Verify Python installed correctly:**
Open Command Prompt (press `Win + R`, type `cmd`, press Enter) and run:
```
python --version
```
You should see: `Python 3.11.x`

If you see an error, Python is not in PATH — reinstall and make sure to check "Add Python to PATH".

---

## Step 2 — Download ExoRAG from GitHub

**Option A — With Git (recommended):**
```
git clone https://github.com/YOUR_USERNAME/exorag.git
cd exorag
```

**Option B — Without Git:**
1. Go to the GitHub page
2. Click green **"Code"** button → **"Download ZIP"**
3. Extract the ZIP to a folder (e.g., `C:\Users\YourName\exorag`)
4. Open Command Prompt and navigate there:
```
cd C:\Users\YourName\exorag
```

---

## Step 3 — Run the Automatic Setup Script

Double-click **`setup_windows.bat`** in the exorag folder.

Or run it from Command Prompt:
```
setup_windows.bat
```

This script will:
- Create a Python virtual environment
- Install all dependencies
- Copy `.env.example` to `.env`
- Tell you what to do next

**If you get a security warning:** Right-click the .bat file → Properties → check "Unblock" → OK, then run again.

---

## Step 4 — Add Your Claude API Key

1. Open the `exorag` folder in File Explorer
2. Find the file named `.env` (it may show as just `env` if file extensions are hidden)
3. Right-click → Open with → Notepad
4. Find the line: `ANTHROPIC_API_KEY=sk-ant-your-key-here`
5. Replace `sk-ant-your-key-here` with your actual API key
6. Change `ENTREZ_EMAIL=yourname@email.com` to your real email
7. Save and close Notepad

**Where to get your Claude API key:**
- Go to https://console.anthropic.com/
- Sign in or create a free account
- Click "API Keys" in the left sidebar
- Click "Create Key" → copy the key (starts with `sk-ant-`)

---

## Step 5 — Fetch PubMed Literature

In Command Prompt (make sure you're in the exorag folder):
```
venv\Scripts\activate
python scripts\01_fetch_pubmed.py
```

This takes **5–10 minutes**. You'll see progress for each search query.
Creates: `data\abstracts.json`

---

## Step 6 — Build the Local Database

```
python scripts\02_build_index.py
```

This takes **15–25 minutes** on first run (downloads the embedding model ~80MB, then processes all papers).
On all future runs: ~3–5 minutes.
Creates: `chroma_db\` folder

**You only need to run this once.** The database is saved to disk.

---

## Step 7 — Launch the App

```
streamlit run scripts\04_app.py
```

Your browser will open automatically at: **http://localhost:8501**

If the browser doesn't open, go to http://localhost:8501 manually.

**To stop the app:** Press `Ctrl + C` in the Command Prompt window.

---

## Running the App Again Later

Every time you want to use ExoRAG after the first setup:

```
cd C:\Users\YourName\exorag
venv\Scripts\activate
streamlit run scripts\04_app.py
```

That's it — Steps 5 and 6 only need to run once.

---

## Troubleshooting

### "python is not recognized as an internal or external command"
Python is not in PATH. Reinstall Python and check "Add Python to PATH" on the first screen.

### "pip is not recognized"
Try: `python -m pip install -r requirements.txt` instead of `pip install`

### ChromaDB install error
Run: `python -m pip install chromadb --no-binary chromadb`

### "Microsoft Visual C++ 14.0 or greater is required"
Some packages need C++ build tools. Install from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Click "Download Build Tools" → install with "C++ build tools" workload checked.

### Streamlit won't open in browser
Go to http://localhost:8501 manually in Chrome or Edge.

### "Port 8501 is already in use"
Run: `streamlit run scripts\04_app.py --server.port 8502`
Then go to http://localhost:8502

### App opens but says "Could not load knowledge base"
You need to run Steps 5 and 6 first (fetch + build index).

### Slow performance
Normal on 8GB RAM — the embedding model uses ~500MB. Close other applications while indexing.

---

## File Structure After Setup

```
exorag/
├── venv/              ← Python environment (created by setup)
├── data/
│   └── abstracts.json ← PubMed papers (created by Step 5)
├── chroma_db/         ← Local vector database (created by Step 6)
├── outputs/           ← Query logs and reports
├── scripts/           ← All Python scripts
├── .env               ← Your API key (never share this file)
└── ...
```

---

## Quick Reference — Commands

| Action | Command |
|--------|---------|
| Activate environment | `venv\Scripts\activate` |
| Fetch papers | `python scripts\01_fetch_pubmed.py` |
| Build database | `python scripts\02_build_index.py` |
| Launch app | `streamlit run scripts\04_app.py` |
| CLI query mode | `python scripts\03_query_cli.py` |
| Corpus analytics | `python scripts\05_explorer.py` |
| Stop any script | `Ctrl + C` |

---

## Cost

| Component | Cost |
|-----------|------|
| Python, ChromaDB, embeddings | **Free** |
| PubMed API | **Free** |
| Claude API (queries) | ~$0.001 per query (Haiku model) |
| Building the index | **Free** (local embeddings) |

A typical session of 20 queries costs about **$0.02 total**.
