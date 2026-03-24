# 🌿 ExoRAG — Exosome · Cosmetic · Natural Compound Intelligence

A fully local RAG (Retrieval-Augmented Generation) pipeline that mines PubMed literature 
on exosomes, cosmetic applications, and natural compounds — then answers product development 
questions using Claude AI.

**Every answer is grounded in real PubMed papers with clickable citations.**

---

## What It Does

- 🔍 **Fetches** 400–700 PubMed abstracts across 13 targeted queries (exosome + skin, hair, natural compounds, plant nanoparticles, cosmetic actives, etc.)
- 💾 **Indexes** them into a local ChromaDB vector database using free local embeddings
- 🤖 **Answers** product development questions via Claude AI with cited sources
- 📊 **Explores** compound frequency, publication trends, and research gaps

## Architecture

```
PubMed API (free) → abstracts.json → ChromaDB (local disk)
                                           ↑
                              sentence-transformers (local CPU, free)
                                           ↓
                    Query → embed → similarity search → top-K chunks
                                           ↓
                         Context + system prompt → Claude API → Answer
```

## Cost

| Component | Cost |
|-----------|------|
| PubMed fetch | Free |
| ChromaDB (local) | Free |
| Embeddings (local) | Free |
| Claude API (per query) | ~$0.001 (Haiku) |

---

## Quick Start

### Windows
```
1. Double-click setup_windows.bat
2. Edit .env — add your Claude API key
3. Double-click run_app_windows.bat
```
→ See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for full instructions

### Mac / Linux
```bash
bash setup_mac_linux.sh
# Edit .env — add your Claude API key
python scripts/01_fetch_pubmed.py
python scripts/02_build_index.py
streamlit run scripts/04_app.py
```

---

## Prerequisites

- Python 3.10+
- Anthropic Claude API key → [console.anthropic.com](https://console.anthropic.com)
- ~1GB disk space
- No GPU required — runs on CPU

---

## Scripts

| Script | What it does | When to run |
|--------|-------------|-------------|
| `01_fetch_pubmed.py` | Pull abstracts from PubMed | Once (or monthly to update) |
| `02_build_index.py` | Embed + store in ChromaDB | Once after fetching |
| `03_query_cli.py` | CLI query interface | Anytime (testing) |
| `04_app.py` | Streamlit demo app | Anytime |
| `05_explorer.py` | Corpus analytics | Anytime |

---

## Demo Queries

- *"Which natural compounds in exosomes show the strongest anti-aging skin evidence?"*
- *"What plant-derived exosome-like nanoparticles work best for cosmetic delivery?"*
- *"Where are the product gaps in natural compound exosome cosmetics?"*
- *"Compare curcumin vs quercetin vs resveratrol for exosome skin delivery."*
- *"What's the evidence for exosome + natural compound combinations in wound healing?"*

---

## Topics Covered

- Exosome + skin / anti-aging / wound healing
- Exosome + hair growth / alopecia
- Curcumin, quercetin, resveratrol, berberine, EGCG, ginger
- Plant-derived exosome-like nanoparticles (ginger, grapefruit, aloe vera, garlic)
- Exosome drug loading / formulation strategies
- Topical / transdermal exosome delivery
- Cosmetic actives (retinol, vitamin C, niacinamide, ceramide) + exosome
- Melanin / skin brightening / pigmentation

---

## Built With

- [ChromaDB](https://www.trychroma.com/) — local vector database
- [sentence-transformers](https://www.sbert.net/) — local embeddings (all-MiniLM-L6-v2)
- [Anthropic Claude](https://www.anthropic.com/) — LLM for answer generation
- [Biopython / NCBI Entrez](https://biopython.org/) — PubMed data ingestion
- [Streamlit](https://streamlit.io/) — web UI

---

## License

MIT — free to use, modify, and distribute.
