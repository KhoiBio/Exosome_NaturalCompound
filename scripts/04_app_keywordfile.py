#!/usr/bin/env python3
"""
04_app.py — ExoRAG Literature Engine
======================================
Tabs:
  1. Literature Q&A   — RAG query with cited answers + source snippets
  2. Paper Summaries  — Fetch free full-text PDFs and AI-summarize each paper
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION    = "exorag_cosmetic"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Prompts ──────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are a Senior Scientific Advisor specializing in exosome biology
and extracellular vesicle research. Answer with a scientific focus using retrieved
PubMed literature. Cite as "Author et al., YEAR (PMID: XXXXX)".
Clearly distinguish in vitro / in vivo / clinical findings.
Be thorough, structured, and highlight key mechanistic insights."""

SUMMARY_SYSTEM = """You are a scientific literature analyst. Summarize the provided
paper content into a concise, structured scientific summary.

Return ONLY a valid JSON object with exactly these keys:
{
  "background": "1-2 sentence context and research gap",
  "objective": "1 sentence stating the main aim",
  "methods": "2-3 sentences on key experimental approaches",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "conclusion": "1-2 sentences on significance and implications",
  "limitations": "1 sentence on study limitations or caveats",
  "relevance_to_exosomes": "1-2 sentences on how this relates to exosome/EV biology"
}

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

DEMO_QUERIES = [
    "What are the key mechanisms of exosome biogenesis and secretion?",
    "How are miRNAs selectively loaded into exosomes?",
    "What surface markers best identify exosomes vs other EVs?",
    "How do tumor-derived exosomes promote immune evasion?",
    "What methods are most effective for exosome isolation and characterization?",
    "How do mesenchymal stem cell exosomes promote tissue regeneration?",
    "What is the role of exosomes in cell-to-cell communication?",
    "How are plant-derived nanoparticles used as exosome mimetics?",
]

# ─── PDF / Full-text fetching ──────────────────────────────────────────────────

UNPAYWALL_EMAIL = os.getenv("ENTREZ_EMAIL", "researcher@email.com")

def get_doi_from_pmid(pmid: str) -> str | None:
    """Use NCBI eutils to get DOI from PMID."""
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "rettype": "xml", "retmode": "xml"}
        r = requests.get(url, params=params, timeout=10)
        # Parse DOI from XML
        match = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', r.text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def get_free_pdf_url(doi: str) -> str | None:
    """Query Unpaywall for open-access PDF URL."""
    if not doi:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Best OA location first
            best = data.get("best_oa_location")
            if best:
                return best.get("url_for_pdf") or best.get("url")
            # Fallback: scan all locations
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]
    except Exception:
        pass
    return None


def get_pmc_pdf_url(pmid: str) -> str | None:
    """Check PubMed Central for free full text."""
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            "dbfrom": "pubmed", "db": "pmc",
            "id": pmid, "retmode": "json"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        links = data.get("linksets", [{}])[0].get("linksetdbs", [])
        for link in links:
            if link.get("dbto") == "pmc":
                ids = link.get("links", [])
                if ids:
                    pmcid = ids[0]
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
    except Exception:
        pass
    return None


def fetch_pdf_text(pdf_url: str, max_chars: int = 8000) -> str | None:
    """Download PDF and extract text using pdfminer or fallback to raw."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        r = requests.get(pdf_url, headers=headers, timeout=20, stream=True)
        if r.status_code != 200:
            return None
        content_type = r.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not pdf_url.endswith(".pdf"):
            return None

        # Try pdfminer
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io
            pdf_bytes = r.content
            output = io.StringIO()
            extract_text_to_fp(
                io.BytesIO(pdf_bytes), output,
                laparams=LAParams(), output_type="text", codec="utf-8"
            )
            text = output.getvalue().strip()
            if text:
                return text[:max_chars]
        except ImportError:
            pass

        # Fallback: try pypdf
        try:
            import pypdf, io
            pdf_bytes = r.content
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages = [page.extract_text() or "" for page in reader.pages[:15]]
            text = "\n".join(pages).strip()
            if text:
                return text[:max_chars]
        except Exception:
            pass

    except Exception:
        pass
    return None


def resolve_full_text(pmid: str, title: str) -> dict:
    """
    Attempt to get free full text for a paper.
    Returns dict with keys: pmid, source, pdf_url, text, status
    """
    result = {"pmid": pmid, "title": title, "source": None,
               "pdf_url": None, "text": None, "status": "not_found"}

    # 1. Try PMC first (most reliable)
    pmc_url = get_pmc_pdf_url(pmid)
    if pmc_url:
        text = fetch_pdf_text(pmc_url)
        if text:
            result.update({"source": "PMC", "pdf_url": pmc_url,
                           "text": text, "status": "success"})
            return result
        result.update({"source": "PMC", "pdf_url": pmc_url, "status": "pdf_unreadable"})

    # 2. Try Unpaywall via DOI
    doi = get_doi_from_pmid(pmid)
    if doi:
        oa_url = get_free_pdf_url(doi)
        if oa_url:
            text = fetch_pdf_text(oa_url)
            if text:
                result.update({"source": "Unpaywall", "pdf_url": oa_url,
                               "text": text, "status": "success"})
                return result
            result.update({"source": "Unpaywall", "pdf_url": oa_url,
                           "status": "pdf_unreadable"})

    # 3. Try Europe PMC API
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"EXT_ID:{pmid} SRC:MED", "format": "json",
                    "resultType": "core"},
            timeout=10
        )
        if r.status_code == 200:
            hits = r.json().get("resultList", {}).get("result", [])
            for hit in hits:
                if hit.get("isOpenAccess") == "Y":
                    pmcid = hit.get("pmcid")
                    if pmcid:
                        pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                        text = fetch_pdf_text(pdf_url)
                        if text:
                            result.update({"source": "EuropePMC", "pdf_url": pdf_url,
                                           "text": text, "status": "success"})
                            return result
    except Exception:
        pass

    if result["pdf_url"]:
        result["status"] = "pdf_unreadable"
    return result


# ─── LLM Backends ─────────────────────────────────────────────────────────────

def check_claude() -> bool:
    return bool(ANTHROPIC_KEY and not ANTHROPIC_KEY.startswith("sk-ant-your"))

def check_ollama() -> tuple:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []

def stream_response(backend, ollama_model, system, user):
    if "Claude" in backend:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        with client.messages.stream(
            model=CLAUDE_MODEL, max_tokens=3000,
            system=system, messages=[{"role": "user", "content": user}]
        ) as s:
            for t in s.text_stream:
                yield t
    else:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": ollama_model,
                  "prompt": f"{system}\n\n{user}", "stream": True},
            stream=True, timeout=180
        )
        for line in r.iter_lines():
            if line:
                try:
                    d = json.loads(line)
                    t = d.get("response", "")
                    if t:
                        yield t
                    if d.get("done"):
                        break
                except Exception:
                    continue

def call_llm(system: str, user: str, backend: str, ollama_model: str) -> str:
    if "Claude" in backend:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        r = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2000,
            system=system, messages=[{"role": "user", "content": user}]
        )
        return r.content[0].text
    else:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": ollama_model,
                  "prompt": f"{system}\n\n{user}",
                  "stream": False, "format": "json"},
            timeout=180
        )
        return r.json().get("response", "")

def parse_json_safe(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r"(\{.*\})", r"(\[.*\])"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    raise ValueError(f"Could not parse JSON:\n{text[:300]}")


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION, embedding_function=ef)

def retrieve(collection, query: str, k: int) -> list:
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "metadata": meta, "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks):
        m = c["metadata"]
        parts.append(
            f"[Source {i+1}]\nTitle: {m['title']}\n"
            f"Authors: {m['authors']} | Year: {m['year']} | Journal: {m['journal']}\n"
            f"PMID: {m['pmid']}\nText: {c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: #f4f3ef; }

.hero {
    background: #0d1117;
    border-radius: 0px;
    padding: 2.2rem 2.8rem 1.8rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #30e3ca;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #30e3ca;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem;
    color: #f0ede6;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-title em {
    font-style: italic;
    color: #30e3ca;
}
.hero-sub {
    color: #6b7280;
    font-size: 0.85rem;
    margin: 0;
    font-weight: 300;
}
.hero-pills {
    margin-top: 1rem;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(48,227,202,0.08);
    border: 1px solid rgba(48,227,202,0.25);
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    color: #30e3ca;
    letter-spacing: 0.05em;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: white;
    border: 1px solid #e5e2d9;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    border-top: 3px solid #30e3ca;
}
.stat-num {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    color: #0d1117;
    line-height: 1;
}
.stat-lbl {
    color: #9ca3af;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

.demo-btn-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 1.2rem;
}

.answer-box {
    background: white;
    border: 1px solid #e5e2d9;
    border-left: 4px solid #30e3ca;
    border-radius: 6px;
    padding: 1.8rem;
    line-height: 1.8;
    font-size: 0.92rem;
    color: #1f2937;
}

.source-card {
    background: white;
    border: 1px solid #e5e2d9;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
}
.source-title {
    font-weight: 600;
    color: #0d1117;
    font-size: 0.88rem;
    line-height: 1.3;
    margin-bottom: 4px;
}
.source-meta {
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.03em;
}
.sim-bar-wrap {
    background: #f4f3ef;
    border-radius: 3px;
    height: 4px;
    margin-top: 6px;
}
.sim-bar {
    height: 4px;
    border-radius: 3px;
    background: #30e3ca;
}

.summary-card {
    background: white;
    border: 1px solid #e5e2d9;
    border-radius: 6px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.summary-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0ede6;
}
.summary-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.1rem;
    color: #0d1117;
    line-height: 1.3;
    flex: 1;
}
.oa-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 3px 8px;
    border-radius: 3px;
    white-space: nowrap;
    letter-spacing: 0.05em;
}
.oa-yes { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.oa-no  { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.oa-err { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

.finding-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 6px;
    font-size: 0.85rem;
}
.finding-bullet {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #30e3ca;
    flex-shrink: 0;
    margin-top: 6px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 4px;
}
.section-text {
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.6;
}

.badge-claude {
    background: #fff3e0; color: #c2410c;
    border: 1px solid #fed7aa; border-radius: 3px;
    padding: 2px 8px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
}
.badge-ollama {
    background: #d1fae5; color: #065f46;
    border: 1px solid #a7f3d0; border-radius: 3px;
    padding: 2px 8px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
}
.setup-box {
    background: #fef3c7; border: 1px solid #fde68a;
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.82rem;
}

/* Streamlit overrides */
.stButton > button {
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stExpander"] {
    border: 1px solid #e5e2d9 !important;
    border-radius: 6px !important;
}
</style>
"""


# ─── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "qa_query": "",
        "last_chunks": [],
        "summaries_cache": {},   # pmid -> summary dict
        "summaries_queue": [],   # list of metadata dicts to summarize
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ExoRAG — Literature Engine",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    # ── Backend detection ──
    claude_ok              = check_claude()
    ollama_ok, ollama_models = check_ollama()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            'letter-spacing:0.15em;color:#9ca3af;text-transform:uppercase">'
            '⬡ ExoRAG</p>', unsafe_allow_html=True
        )
        st.divider()
        st.markdown("**LLM Backend**")

        options = []
        if claude_ok:  options.append("🟠 Claude API  (Recommended)")
        if ollama_ok:  options.append("🟢 Ollama  (Free / Local)")
        if not options: options.append("⚠️  No LLM found — see setup")

        backend = st.radio("Choose LLM:", options)

        ollama_sel = OLLAMA_MODEL
        if "Ollama" in backend and ollama_models:
            ollama_sel = st.selectbox("Ollama model:", ollama_models)

        if "⚠️" in backend:
            st.markdown("""
            <div class="setup-box">
            <b>Option A — Claude API:</b><br>
            Get key at <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a><br>
            Add to <code>.env</code>: <code>ANTHROPIC_API_KEY=sk-ant-...</code><br><br>
            <b>Option B — Ollama (free):</b><br>
            Download <a href="https://ollama.com" target="_blank">ollama.com</a><br>
            Run: <code>ollama pull llama3.2</code><br>
            Run: <code>ollama serve</code>
            </div>""", unsafe_allow_html=True)

        st.divider()
        k_val = st.slider("Chunks retrieved (K)", 3, 20, 8)
        show_snippets = st.toggle("Show abstract snippets", value=True)
        st.divider()
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#9ca3af">'
            'PubMed · ChromaDB · MiniLM<br>Unpaywall · PMC · EuropePMC</p>',
            unsafe_allow_html=True
        )

    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Exosome Research Intelligence</div>
        <p class="hero-title">ExoRAG <em>Literature Engine</em></p>
        <p class="hero-sub">Ask questions across your PubMed corpus. Retrieve free full-text PDFs. AI-summarize every paper.</p>
        <div class="hero-pills">
            <span class="hero-pill">RAG Q&A</span>
            <span class="hero-pill">Full-text PDF retrieval</span>
            <span class="hero-pill">Structured summaries</span>
            <span class="hero-pill">PMC · Unpaywall · EuropePMC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load DB ──
    try:
        collection = load_collection()
        n_chunks   = collection.count()
    except Exception as e:
        st.error(f"Knowledge base not found: {e}")
        st.info("Run:\n```\npython scripts/01_fetch_pubmed.py\npython scripts/02_build_index.py\n```")
        return

    llm_label = "Claude" if "Claude" in backend else ollama_sel if "Ollama" in backend else "—"

    # Stats row
    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    for num, lbl in [
        (f"{n_chunks:,}", "indexed chunks"),
        (f"~{n_chunks//3:,}", "pubmed papers"),
        (str(k_val), "retrieved / query"),
        (llm_label, "llm engine"),
    ]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    no_llm = "⚠️" in backend

    # ── Tabs ──
    tab1, tab2 = st.tabs(["💬 Literature Q&A", "📄 Paper Summaries"])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — Literature Q&A
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'letter-spacing:0.1em;color:#9ca3af;text-transform:uppercase;margin-bottom:0.8rem">'
            'Quick start — example queries</p>',
            unsafe_allow_html=True
        )

        # Demo query buttons
        cols = st.columns(2)
        selected_demo = None
        for i, dq in enumerate(DEMO_QUERIES):
            if cols[i % 2].button(f"↳ {dq}", key=f"dq_{i}", use_container_width=True):
                selected_demo = dq

        if selected_demo:
            st.session_state["qa_query"] = selected_demo

        st.divider()

        query = st.text_area(
            "Scientific question:",
            value=st.session_state.get("qa_query", ""),
            height=90,
            placeholder="e.g. What are the mechanisms by which tumor-derived exosomes modulate T cell function?",
            key="qa_input",
        )

        col_run, col_send = st.columns([1, 5])
        run_qa = col_run.button("🔍 Analyze", type="primary", key="run_qa")

        if run_qa and query.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                with st.spinner("Retrieving relevant literature..."):
                    chunks  = retrieve(collection, query.strip(), k_val)
                    context = build_context(chunks)
                    # Store for Paper Summaries tab
                    st.session_state["last_chunks"] = chunks

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n"
                    f"---\n\nSCIENTIFIC QUESTION: {query.strip()}"
                )

                col_ans, col_src = st.columns([3, 2])

                with col_ans:
                    badge_cls = "badge-claude" if "Claude" in backend else "badge-ollama"
                    badge_txt = "Claude API" if "Claude" in backend else f"Ollama · {ollama_sel}"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:0.8rem">'
                        f'<span style="font-family:\'Instrument Serif\',serif;font-size:1.15rem">Analysis</span>'
                        f'<span class="{badge_cls}">{badge_txt}</span></div>',
                        unsafe_allow_html=True
                    )
                    placeholder = st.empty()
                    full_text = ""
                    try:
                        for token in stream_response(backend, ollama_sel, RAG_SYSTEM, user_msg):
                            full_text += token
                            placeholder.markdown(
                                f'<div class="answer-box">{full_text}▌</div>',
                                unsafe_allow_html=True
                            )
                        placeholder.markdown(
                            f'<div class="answer-box">{full_text}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"LLM error: {e}")

                with col_src:
                    st.markdown(
                        f'<span style="font-family:\'Instrument Serif\',serif;font-size:1.15rem">'
                        f'Sources</span> '
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#9ca3af">'
                        f'{len(chunks)} chunks retrieved</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    for i, c in enumerate(chunks):
                        m   = c["metadata"]
                        sim = c["similarity"]
                        sim_pct = int(sim * 100)
                        sim_color = "#30e3ca" if sim > 0.7 else "#fbbf24" if sim > 0.5 else "#f87171"

                        with st.expander(
                            f"[{i+1}] {m['title'][:55]}{'...' if len(m['title'])>55 else ''} ({m['year']})",
                            expanded=i == 0
                        ):
                            st.markdown(
                                f'<div class="source-title">{m["title"]}</div>'
                                f'<div class="source-meta">{m["authors"]}</div>'
                                f'<div class="source-meta">{m["journal"]} · {m["year"]} · '
                                f'<a href="{m["url"]}" target="_blank">PMID {m["pmid"]}</a></div>'
                                f'<div class="sim-bar-wrap"><div class="sim-bar" '
                                f'style="width:{sim_pct}%;background:{sim_color}"></div></div>'
                                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                                f'color:#9ca3af;margin-top:3px">relevance {sim:.3f}</div>',
                                unsafe_allow_html=True
                            )
                            if show_snippets:
                                st.markdown(
                                    f'<div style="margin-top:0.8rem;padding-top:0.8rem;'
                                    f'border-top:1px solid #f0ede6;font-size:0.8rem;'
                                    f'color:#6b7280;line-height:1.6">'
                                    f'{c["text"][:350]}...</div>',
                                    unsafe_allow_html=True
                                )

                # Queue these sources for Paper Summaries tab
                st.success(
                    f"✓ Analysis complete. Go to **📄 Paper Summaries** tab to fetch "
                    f"full-text PDFs and AI-summarize all {len(chunks)} source papers."
                )

        elif run_qa:
            st.warning("Please enter a question.")

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — Paper Summaries
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            '<p style="font-family:\'Instrument Serif\',serif;font-size:1.5rem;margin-bottom:0.2rem">'
            'Paper Summaries</p>'
            '<p style="font-size:0.85rem;color:#6b7280;margin-bottom:1.2rem">'
            'Fetches free full-text PDFs from PMC, Unpaywall, and EuropePMC. '
            'AI-generates a structured summary for each source paper from your last Q&A.</p>',
            unsafe_allow_html=True
        )

        chunks = st.session_state.get("last_chunks", [])

        if not chunks:
            st.info(
                "No sources loaded yet. Run a query in the **💬 Literature Q&A** tab first — "
                "the retrieved source papers will appear here for summarization."
            )
        else:
            # Deduplicate by PMID
            seen_pmids = set()
            unique_sources = []
            for c in chunks:
                pmid = c["metadata"]["pmid"]
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    unique_sources.append(c)

            st.markdown(
                f'<div style="background:white;border:1px solid #e5e2d9;border-radius:6px;'
                f'padding:1rem 1.4rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:12px">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#9ca3af">'
                f'{len(unique_sources)} UNIQUE PAPERS FROM LAST QUERY</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if no_llm:
                st.error("No LLM configured — see sidebar. LLM is needed to generate summaries.")
            else:
                col_btn, col_info = st.columns([1, 4])
                run_all = col_btn.button(
                    "⬇ Fetch PDFs & Summarize All",
                    type="primary", key="run_summaries"
                )
                col_info.markdown(
                    '<p style="font-size:0.8rem;color:#9ca3af;padding-top:0.5rem">'
                    'Tries PMC → Unpaywall → EuropePMC for each paper. '
                    'Falls back to abstract if no free PDF found.</p>',
                    unsafe_allow_html=True
                )

                if run_all:
                    progress_bar = st.progress(0, text="Starting...")
                    status_box   = st.empty()

                    for idx, c in enumerate(unique_sources):
                        m    = c["metadata"]
                        pmid = m["pmid"]

                        # Skip if already cached
                        if pmid in st.session_state["summaries_cache"]:
                            progress_bar.progress(
                                (idx + 1) / len(unique_sources),
                                text=f"[{idx+1}/{len(unique_sources)}] {m['title'][:50]}... (cached)"
                            )
                            continue

                        progress_bar.progress(
                            (idx + 1) / len(unique_sources),
                            text=f"[{idx+1}/{len(unique_sources)}] Fetching: {m['title'][:50]}..."
                        )
                        status_box.markdown(
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#6b7280">→ Resolving PMID {pmid}...</div>',
                            unsafe_allow_html=True
                        )

                        # Attempt full-text fetch
                        ft = resolve_full_text(pmid, m["title"])
                        time.sleep(0.5)  # polite delay

                        # Use full text if available, else abstract
                        if ft["status"] == "success" and ft["text"]:
                            content_for_llm = ft["text"]
                            text_source = f"Full text ({ft['source']})"
                        else:
                            content_for_llm = c["text"]  # abstract chunk
                            text_source = "Abstract only"

                        # Generate summary
                        status_box.markdown(
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#6b7280">→ Summarizing ({text_source})...</div>',
                            unsafe_allow_html=True
                        )
                        try:
                            user_msg = (
                                f"Paper title: {m['title']}\n"
                                f"Authors: {m['authors']}\n"
                                f"Journal: {m['journal']} ({m['year']})\n"
                                f"PMID: {pmid}\n\n"
                                f"Content:\n{content_for_llm[:6000]}"
                            )
                            raw     = call_llm(SUMMARY_SYSTEM, user_msg, backend, ollama_sel)
                            summary = parse_json_safe(raw)
                        except Exception as e:
                            summary = {
                                "background": "Could not parse LLM summary.",
                                "objective": str(e)[:100],
                                "methods": "",
                                "key_findings": [],
                                "conclusion": "",
                                "limitations": "",
                                "relevance_to_exosomes": "",
                            }

                        st.session_state["summaries_cache"][pmid] = {
                            "metadata":    m,
                            "summary":     summary,
                            "text_source": text_source,
                            "pdf_url":     ft.get("pdf_url"),
                            "oa_status":   ft["status"],
                            "oa_source":   ft.get("source"),
                        }

                    progress_bar.empty()
                    status_box.empty()
                    st.success(f"✓ Summarized {len(unique_sources)} papers.")

                # ── Render cached summaries ──
                cache = st.session_state.get("summaries_cache", {})
                rendered = [
                    cache[c["metadata"]["pmid"]]
                    for c in unique_sources
                    if c["metadata"]["pmid"] in cache
                ]

                if rendered:
                    st.divider()
                    # Export button
                    export_data = json.dumps(
                        [{
                            "pmid":    r["metadata"]["pmid"],
                            "title":   r["metadata"]["title"],
                            "authors": r["metadata"]["authors"],
                            "journal": r["metadata"]["journal"],
                            "year":    r["metadata"]["year"],
                            "pdf_url": r.get("pdf_url"),
                            "text_source": r["text_source"],
                            "summary": r["summary"],
                        } for r in rendered],
                        indent=2
                    )
                    st.download_button(
                        "⬇ Export summaries as JSON",
                        data=export_data,
                        file_name=f"exorag_summaries_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        key="export_summaries"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    for r in rendered:
                        m       = r["metadata"]
                        summary = r["summary"]
                        oa      = r["oa_status"]
                        src     = r.get("oa_source", "")
                        pdf_url = r.get("pdf_url")
                        tsrc    = r["text_source"]

                        # OA badge
                        if oa == "success":
                            badge = f'<span class="oa-badge oa-yes">✓ {src} · Full text</span>'
                        elif pdf_url:
                            badge = f'<span class="oa-badge oa-no">⚠ {src} · PDF unreadable</span>'
                        else:
                            badge = '<span class="oa-badge oa-err">Abstract only</span>'

                        st.markdown(
                            f'<div class="summary-card">'
                            f'<div class="summary-header">'
                            f'<div class="summary-title">{m["title"]}</div>'
                            f'{badge}'
                            f'</div>'

                            # Meta row
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                            f'color:#9ca3af;margin-bottom:1rem">'
                            f'{m["authors"]} &nbsp;·&nbsp; {m["journal"]} &nbsp;·&nbsp; {m["year"]} '
                            f'&nbsp;·&nbsp; <a href="{m["url"]}" target="_blank">PMID {m["pmid"]}</a>'
                            + (f' &nbsp;·&nbsp; <a href="{pdf_url}" target="_blank">PDF ↗</a>' if pdf_url else '')
                            + f'</div>',
                            unsafe_allow_html=True
                        )

                        # Summary sections
                        sections = [
                            ("Background", summary.get("background", "")),
                            ("Objective",  summary.get("objective", "")),
                            ("Methods",    summary.get("methods", "")),
                        ]
                        cols = st.columns(3)
                        for col, (label, text) in zip(cols, sections):
                            col.markdown(
                                f'<div class="section-label">{label}</div>'
                                f'<div class="section-text">{text}</div>',
                                unsafe_allow_html=True
                            )

                        # Key findings
                        findings = summary.get("key_findings", [])
                        if findings:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                '<div class="section-label">Key Findings</div>',
                                unsafe_allow_html=True
                            )
                            for f in findings:
                                st.markdown(
                                    f'<div class="finding-item">'
                                    f'<div class="finding-bullet"></div>'
                                    f'<div class="section-text">{f}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                        # Bottom row
                        col_c, col_l, col_r = st.columns(3)
                        col_c.markdown(
                            f'<div class="section-label">Conclusion</div>'
                            f'<div class="section-text">{summary.get("conclusion","")}</div>',
                            unsafe_allow_html=True
                        )
                        col_l.markdown(
                            f'<div class="section-label">Limitations</div>'
                            f'<div class="section-text">{summary.get("limitations","")}</div>',
                            unsafe_allow_html=True
                        )
                        col_r.markdown(
                            f'<div class="section-label">Exosome Relevance</div>'
                            f'<div class="section-text">{summary.get("relevance_to_exosomes","")}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                            f'color:#d1d5db;text-align:right;margin-top:-0.5rem;margin-bottom:1rem">'
                            f'summarized from: {tsrc}</div>',
                            unsafe_allow_html=True
                        )

                elif not run_all:
                    st.info("Click **Fetch PDFs & Summarize All** to generate summaries for the papers above.")


if __name__ == "__main__":
    main()
