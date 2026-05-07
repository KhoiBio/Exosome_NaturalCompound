#!/usr/bin/env python3
"""
04_app.py — ExoRAG Literature Engine
======================================
Tabs:
  1. Literature Q&A   — RAG query with cited answers + source snippets
  2. Paper Summaries  — AI-summarize the source papers used in the Q&A answer
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
COLLECTION    = os.getenv("CHROMA_COLLECTION", "exorag_cosmetic")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen3:4b")

UNPAYWALL_EMAIL = os.getenv("ENTREZ_EMAIL", "researcher@email.com")

# ─── Prompts ──────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are a Senior Scientific Advisor. Answer with a scientific focus
using the retrieved PubMed literature provided. Cite as "Author et al., YEAR (PMID: XXXXX)".
Clearly distinguish in vitro / in vivo / clinical findings.
Be thorough, structured, and highlight key mechanistic insights."""

# Generic — works for any biomedical topic (TPU, wound healing, exosomes, etc.)
SUMMARY_SYSTEM = """You are a scientific literature analyst. Summarize the provided
paper into a concise, structured summary.

Return ONLY a valid JSON object with exactly these keys:
{
  "background": "1-2 sentence context and research gap addressed",
  "objective": "1 sentence stating the main aim of the study",
  "methods": "2-3 sentences on key experimental approaches used",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "conclusion": "1-2 sentences on significance and implications",
  "limitations": "1 sentence on study limitations or caveats",
  "relevance": "1-2 sentences on why this paper matters for the research area"
}

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

DEMO_QUERIES = [
    "What are the key mechanisms of exosome biogenesis and secretion?",
    "How are miRNAs selectively loaded into exosomes?",
    "What surface markers best identify exosomes vs other EVs?",
    "How do tumor-derived exosomes promote immune evasion?",
    "What methods are most effective for exosome isolation and characterization?",
    "How do mesenchymal stem cell exosomes promote tissue regeneration?",
    "What is the role of TPU scaffolds in wound healing?",
    "How do fibroblasts interact with electrospun polymer scaffolds?",
]

# ─── PDF / Full-text fetching ─────────────────────────────────────────────────

def get_doi_from_pmid(pmid: str) -> str | None:
    try:
        params = {"db": "pubmed", "id": pmid, "rettype": "xml", "retmode": "xml"}
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params, timeout=10
        )
        match = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', r.text)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return None


def get_free_pdf_url(doi: str) -> str | None:
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}",
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            best = data.get("best_oa_location")
            if best:
                return best.get("url_for_pdf") or best.get("url")
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]
    except Exception:
        pass
    return None


def get_pmc_pdf_url(pmid: str) -> str | None:
    try:
        params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"}
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params=params, timeout=10
        )
        data = r.json()
        links = data.get("linksets", [{}])[0].get("linksetdbs", [])
        for link in links:
            if link.get("dbto") == "pmc":
                ids = link.get("links", [])
                if ids:
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{ids[0]}/pdf/"
    except Exception:
        pass
    return None


def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts before sending to LLM."""
    # Fix hyphenated line breaks (e.g. "fibro-\nblast" -> "fibroblast")
    text = re.sub(r'-\n(\w)', r'\1', text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse excessive whitespace
    text = re.sub(r' {2,}', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_pdf_text(pdf_url: str, max_chars: int = 40000) -> str | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        r = requests.get(pdf_url, headers=headers, timeout=20, stream=True)
        if r.status_code != 200:
            return None
        content_type = r.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not pdf_url.endswith(".pdf"):
            return None
        pdf_bytes = r.content
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io
            output = io.StringIO()
            extract_text_to_fp(io.BytesIO(pdf_bytes), output, laparams=LAParams(),
                                output_type="text", codec="utf-8")
            text = output.getvalue().strip()
            if text:
                return clean_pdf_text(text)[:max_chars]
        except ImportError:
            pass
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages = [page.extract_text() or "" for page in reader.pages[:15]]
            text = "\n".join(pages).strip()
            if text:
                return clean_pdf_text(text)[:max_chars]
        except Exception:
            pass
    except Exception:
        pass
    return None


def resolve_full_text(pmid: str, title: str) -> dict:
    result = {"pmid": pmid, "title": title, "source": None,
              "pdf_url": None, "text": None, "status": "not_found"}

    # 1. PMC
    pmc_url = get_pmc_pdf_url(pmid)
    if pmc_url:
        text = fetch_pdf_text(pmc_url)
        if text:
            result.update({"source": "PMC", "pdf_url": pmc_url,
                           "text": text, "status": "success"})
            return result
        result.update({"source": "PMC", "pdf_url": pmc_url, "status": "pdf_unreadable"})

    # 2. Unpaywall
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

    # 3. EuropePMC
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"EXT_ID:{pmid} SRC:MED", "format": "json",
                    "resultType": "core"},
            timeout=10
        )
        if r.status_code == 200:
            for hit in r.json().get("resultList", {}).get("result", []):
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
            return True, [m["name"] for m in r.json().get("models", [])]
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
            json={"model": ollama_model, "prompt": f"{system}\n\n{user}", "stream": True},
            stream=True, timeout=600
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
            model=CLAUDE_MODEL, max_tokens=4000,
            system=system, messages=[{"role": "user", "content": user}]
        )
        return r.content[0].text
    else:
        # Ollama: no "format":"json" — it fights with structured prompts on smaller models.
        # Use num_predict to prevent truncation (default is often only 128 tokens).
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":       ollama_model,
                "prompt":      f"{system}\n\n{user}",
                "stream":      False,
                "num_predict": 2000,   # ~1500 words output — enough for full summary JSON
                "temperature": 0.1,    # low temp = more deterministic JSON structure
            },
            timeout=300  # longer timeout for bigger models / longer papers
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
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f3ef; }

.hero {
    background: #0d1117;
    padding: 2.2rem 2.8rem 1.8rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #30e3ca;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.25em; color: #30e3ca;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Instrument Serif', serif; font-size: 2.4rem;
    color: #f0ede6; margin: 0 0 0.3rem 0; line-height: 1.1;
}
.hero-title em { font-style: italic; color: #30e3ca; }
.hero-sub { color: #6b7280; font-size: 0.85rem; margin: 0; font-weight: 300; }
.hero-pills { margin-top: 1rem; display: flex; gap: 8px; flex-wrap: wrap; }
.hero-pill {
    background: rgba(48,227,202,0.08); border: 1px solid rgba(48,227,202,0.25);
    border-radius: 3px; padding: 3px 10px; font-size: 0.68rem;
    font-family: 'DM Mono', monospace; color: #30e3ca; letter-spacing: 0.05em;
}

.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.5rem; }
.stat-card {
    background: white; border: 1px solid #e5e2d9; border-radius: 6px;
    padding: 1rem 1.2rem; border-top: 3px solid #30e3ca;
}
.stat-num { font-family: 'Instrument Serif', serif; font-size: 2rem; color: #0d1117; line-height: 1; }
.stat-lbl {
    color: #9ca3af; font-size: 0.68rem; font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px;
}

.answer-box {
    background: white; border: 1px solid #e5e2d9; border-left: 4px solid #30e3ca;
    border-radius: 6px; padding: 1.8rem; line-height: 1.8;
    font-size: 0.92rem; color: #1f2937;
}
.sim-bar-wrap { background: #f4f3ef; border-radius: 3px; height: 4px; margin-top: 6px; }
.sim-bar { height: 4px; border-radius: 3px; background: #30e3ca; }

.oa-badge {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    padding: 3px 8px; border-radius: 3px; white-space: nowrap; letter-spacing: 0.05em;
}
.oa-yes { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.oa-no  { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.oa-err { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

.badge-claude {
    background: #fff3e0; color: #c2410c; border: 1px solid #fed7aa;
    border-radius: 3px; padding: 2px 8px; font-size: 0.72rem; font-family: 'DM Mono', monospace;
}
.badge-ollama {
    background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;
    border-radius: 3px; padding: 2px 8px; font-size: 0.72rem; font-family: 'DM Mono', monospace;
}
.setup-box {
    background: #fef3c7; border: 1px solid #fde68a;
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.82rem;
}
.stButton > button { border-radius: 4px !important; }
div[data-testid="stExpander"] { border: 1px solid #e5e2d9 !important; border-radius: 6px !important; }
</style>
"""

# ─── Session state ─────────────────────────────────────────────────────────────

def init_state():
    for k, v in {
        "qa_query": "",
        "last_chunks": [],        # chunks from most recent Q&A
        "last_question": "",      # question that produced last_chunks
        "summaries_cache": {},    # pmid -> summary result dict
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ExoRAG — Literature Engine",
        page_icon="⬡", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    claude_ok              = check_claude()
    ollama_ok, ollama_models = check_ollama()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            'letter-spacing:0.15em;color:#9ca3af;text-transform:uppercase">⬡ ExoRAG</p>',
            unsafe_allow_html=True
        )
        st.divider()
        st.markdown("**LLM Backend**")

        options = []
        if claude_ok:   options.append("🟠 Claude API  (Recommended)")
        if ollama_ok:   options.append("🟢 Ollama  (Free / Local)")
        if not options: options.append("⚠️  No LLM found — see setup")

        backend    = st.radio("Choose LLM:", options)
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
            <a href="https://ollama.com" target="_blank">ollama.com</a> →
            <code>ollama pull llama3.2</code> → <code>ollama serve</code>
            </div>""", unsafe_allow_html=True)

        st.divider()
        k_val         = st.slider("Chunks retrieved (K)", 3, 20, 8)
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
        <div class="hero-eyebrow">Biomedical Research Intelligence</div>
        <p class="hero-title">ExoRAG <em>Literature Engine</em></p>
        <p class="hero-sub">Ask questions across your PubMed corpus · Auto-summarize source papers · Retrieve free full-text PDFs</p>
        <div class="hero-pills">
            <span class="hero-pill">RAG Q&A</span>
            <span class="hero-pill">Source paper summaries</span>
            <span class="hero-pill">PMC · Unpaywall · EuropePMC</span>
            <span class="hero-pill">Export JSON</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load DB ──
    try:
        collection = load_collection()
        n_chunks   = collection.count()
    except Exception as e:
        st.error(f"Knowledge base not found: {e}")
        st.info("Run:\n```\npython 01_fetch_pubmed.py conditions.txt --mode free --max 500\npython 02_build_index.py\n```")
        return

    no_llm    = "⚠️" in backend
    llm_label = "Claude" if "Claude" in backend else ollama_sel if "Ollama" in backend else "—"

    # Stats
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

    # ── Tabs ──
    tab1, tab2 = st.tabs(["💬 Literature Q&A", "📄 Source Paper Summaries"])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 — Literature Q&A
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'letter-spacing:0.1em;color:#9ca3af;text-transform:uppercase;margin-bottom:0.8rem">'
            'Example queries</p>', unsafe_allow_html=True
        )

        cols = st.columns(2)
        selected_demo = None
        for i, dq in enumerate(DEMO_QUERIES):
            if cols[i % 2].button(f"↳ {dq}", key=f"dq_{i}", use_container_width=True):
                selected_demo = dq
        if selected_demo:
            st.session_state["qa_query"] = selected_demo

        st.divider()

        query  = st.text_area(
            "Scientific question:",
            value=st.session_state.get("qa_query", ""),
            height=90,
            placeholder="e.g. How do fibroblasts interact with electrospun TPU scaffolds in wound healing?",
            key="qa_input",
        )
        run_qa = st.button("🔍 Analyze", type="primary", key="run_qa")

        if run_qa and query.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                with st.spinner("Retrieving relevant literature..."):
                    chunks  = retrieve(collection, query.strip(), k_val)
                    context = build_context(chunks)
                    # ── Store for Tab 2 ──────────────────────────────────────
                    st.session_state["last_chunks"]   = chunks
                    st.session_state["last_question"] = query.strip()
                    # Clear old summaries so Tab 2 reflects the new question
                    st.session_state["summaries_cache"] = {}

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
                    full_text   = ""
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
                        f'<span style="font-family:\'Instrument Serif\',serif;font-size:1.15rem">Sources</span> '
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#9ca3af">'
                        f'{len(chunks)} chunks</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    for i, c in enumerate(chunks):
                        m       = c["metadata"]
                        sim     = c["similarity"]
                        sim_pct = int(sim * 100)
                        sim_col = "#30e3ca" if sim > 0.7 else "#fbbf24" if sim > 0.5 else "#f87171"

                        with st.expander(
                            f"[{i+1}] {m['title'][:55]}{'...' if len(m['title'])>55 else ''} ({m['year']})",
                            expanded=i == 0
                        ):
                            st.markdown(
                                f'<div style="font-weight:600;color:#0d1117;font-size:0.88rem;'
                                f'line-height:1.3;margin-bottom:4px">{m["title"]}</div>'
                                f'<div style="color:#9ca3af;font-family:\'DM Mono\',monospace;'
                                f'font-size:0.68rem">{m["authors"]}</div>'
                                f'<div style="color:#9ca3af;font-family:\'DM Mono\',monospace;'
                                f'font-size:0.68rem">{m["journal"]} · {m["year"]} · '
                                f'<a href="{m["url"]}" target="_blank">PMID {m["pmid"]}</a></div>'
                                f'<div class="sim-bar-wrap"><div class="sim-bar" '
                                f'style="width:{sim_pct}%;background:{sim_col}"></div></div>'
                                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                                f'color:#9ca3af;margin-top:3px">relevance {sim:.3f}</div>',
                                unsafe_allow_html=True
                            )
                            if show_snippets:
                                st.markdown(
                                    f'<div style="margin-top:0.8rem;padding-top:0.8rem;'
                                    f'border-top:1px solid #f0ede6;font-size:0.8rem;'
                                    f'color:#6b7280;line-height:1.6">{c["text"][:350]}...</div>',
                                    unsafe_allow_html=True
                                )

                st.success(
                    f"✓ Done. Switch to **📄 Source Paper Summaries** to read full summaries "
                    f"of the {len(set(c['metadata']['pmid'] for c in chunks))} source papers."
                )

        elif run_qa:
            st.warning("Please enter a question.")

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 — Source Paper Summaries
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        chunks   = st.session_state.get("last_chunks", [])
        question = st.session_state.get("last_question", "")

        if not chunks:
            st.info(
                "No sources yet. Ask a question in **💬 Literature Q&A** first — "
                "the papers used to answer it will appear here for summarization."
            )
            return

        # Deduplicate by PMID, preserving order
        seen, unique_sources = set(), []
        for c in chunks:
            pmid = c["metadata"]["pmid"]
            if pmid not in seen:
                seen.add(pmid)
                unique_sources.append(c)

        # Header
        st.markdown(
            f'<p style="font-family:\'Instrument Serif\',serif;font-size:1.5rem;margin-bottom:0.2rem">'
            f'Source Paper Summaries</p>'
            f'<p style="font-size:0.85rem;color:#6b7280;margin-bottom:0.3rem">'
            f'Summarizing the <b>{len(unique_sources)} papers</b> used to answer:</p>'
            f'<p style="font-size:0.9rem;color:#0d1117;font-style:italic;'
            f'border-left:3px solid #30e3ca;padding-left:0.8rem;margin-bottom:1.2rem">'
            f'"{question}"</p>',
            unsafe_allow_html=True
        )

        if no_llm:
            st.error("No LLM configured — see sidebar.")
            return

        col_btn, col_info = st.columns([1, 4])
        run_all = col_btn.button("⬇ Fetch & Summarize All", type="primary", key="run_summaries")
        col_info.markdown(
            '<p style="font-size:0.8rem;color:#9ca3af;padding-top:0.5rem">'
            'Tries PMC → Unpaywall → EuropePMC for free full text. '
            'Falls back to abstract if unavailable. Results cached per session.</p>',
            unsafe_allow_html=True
        )

        if run_all:
            progress = st.progress(0, text="Starting...")
            status   = st.empty()
            cache    = st.session_state["summaries_cache"]

            for idx, c in enumerate(unique_sources):
                m    = c["metadata"]
                pmid = m["pmid"]
                pct  = (idx + 1) / len(unique_sources)

                if pmid in cache:
                    progress.progress(pct, text=f"[{idx+1}/{len(unique_sources)}] {m['title'][:45]}... (cached)")
                    continue

                progress.progress(pct, text=f"[{idx+1}/{len(unique_sources)}] Fetching PMID {pmid}...")
                status.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#6b7280">'
                    f'→ Resolving full text for: {m["title"][:70]}...</div>',
                    unsafe_allow_html=True
                )

                ft = resolve_full_text(pmid, m["title"])
                time.sleep(0.5)

                if ft["status"] == "success" and ft["text"]:
                    content_for_llm = ft["text"]
                    text_source     = f"Full text ({ft['source']})"
                else:
                    content_for_llm = c["text"]
                    text_source     = "Abstract only"

                status.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#6b7280">'
                    f'→ Summarizing ({text_source})...</div>',
                    unsafe_allow_html=True
                )

                try:
                    user_msg = (
                        f"Paper title: {m['title']}\n"
                        f"Authors: {m['authors']}\n"
                        f"Journal: {m['journal']} ({m['year']})\n"
                        f"PMID: {pmid}\n\n"
                        f"Content:\n{content_for_llm[:40000]}"
                    )
                    raw     = call_llm(SUMMARY_SYSTEM, user_msg, backend, ollama_sel)
                    summary = parse_json_safe(raw)
                except Exception as e:
                    summary = {
                        "background": "Summary generation failed.",
                        "objective":  str(e)[:120],
                        "methods":    "", "key_findings": [],
                        "conclusion": "", "limitations": "", "relevance": "",
                    }

                cache[pmid] = {
                    "metadata":    m,
                    "chunk_text":  c["text"],
                    "similarity":  c["similarity"],
                    "summary":     summary,
                    "text_source": text_source,
                    "pdf_url":     ft.get("pdf_url"),
                    "oa_status":   ft["status"],
                    "oa_source":   ft.get("source"),
                }

            progress.empty()
            status.empty()
            st.success(f"✓ Summarized {len(unique_sources)} papers.")

        # ── Render summaries ─────────────────────────────────────────────────
        cache    = st.session_state.get("summaries_cache", {})
        rendered = [cache[c["metadata"]["pmid"]] for c in unique_sources
                    if c["metadata"]["pmid"] in cache]

        if not rendered:
            if not run_all:
                st.info("Click **Fetch & Summarize All** above to generate summaries.")
            return

        # Export
        st.divider()
        export_data = json.dumps([{
            "pmid":        r["metadata"]["pmid"],
            "title":       r["metadata"]["title"],
            "authors":     r["metadata"]["authors"],
            "journal":     r["metadata"]["journal"],
            "year":        r["metadata"]["year"],
            "url":         r["metadata"]["url"],
            "pdf_url":     r.get("pdf_url"),
            "text_source": r["text_source"],
            "similarity":  r["similarity"],
            "summary":     r["summary"],
        } for r in rendered], indent=2)

        st.download_button(
            "⬇ Export all summaries as JSON",
            data=export_data,
            file_name=f"exorag_summaries_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key="export_summaries"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Render each paper as a contained st.container() block
        # (avoids the broken open-div-across-st.markdown bug)
        for rank, r in enumerate(rendered):
            m       = r["metadata"]
            summary = r["summary"]
            oa      = r["oa_status"]
            src     = r.get("oa_source", "")
            pdf_url = r.get("pdf_url")
            tsrc    = r["text_source"]
            sim     = r["similarity"]

            # OA badge html string
            if oa == "success":
                oa_badge = f'<span class="oa-badge oa-yes">✓ {src} · Full text</span>'
            elif pdf_url:
                oa_badge = f'<span class="oa-badge oa-no">⚠ {src} · PDF unreadable</span>'
            else:
                oa_badge = '<span class="oa-badge oa-err">Abstract only</span>'

            pdf_link = f' · <a href="{pdf_url}" target="_blank" style="color:#30e3ca">PDF ↗</a>' if pdf_url else ""

            with st.container(border=True):
                # ── Title row ────────────────────────────────────────────────
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:0.5rem">'
                    f'<div style="font-family:\'Instrument Serif\',serif;font-size:1.1rem;'
                    f'color:#0d1117;line-height:1.3;flex:1">'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                    f'color:#9ca3af;margin-right:6px">[{rank+1}]</span>{m["title"]}</div>'
                    f'{oa_badge}</div>'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                    f'color:#9ca3af;margin-bottom:0.8rem">'
                    f'{m["authors"]} · {m["journal"]} · {m["year"]} · '
                    f'<a href="{m["url"]}" target="_blank">PMID {m["pmid"]}</a>'
                    f'{pdf_link} · relevance {sim:.3f} · summarized from: {tsrc}</div>',
                    unsafe_allow_html=True
                )

                # ── Background / Objective / Methods ─────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Background</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("background","")}</div>',
                    unsafe_allow_html=True
                )
                c2.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Objective</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("objective","")}</div>',
                    unsafe_allow_html=True
                )
                c3.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Methods</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("methods","")}</div>',
                    unsafe_allow_html=True
                )

                # ── Key findings ──────────────────────────────────────────────
                findings = summary.get("key_findings", [])
                if findings:
                    st.markdown(
                        '<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                        'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;'
                        'margin:0.8rem 0 0.4rem 0">Key Findings</div>',
                        unsafe_allow_html=True
                    )
                    for f in findings:
                        st.markdown(
                            f'<div style="display:flex;gap:10px;align-items:flex-start;'
                            f'margin-bottom:4px;font-size:0.85rem">'
                            f'<div style="width:6px;height:6px;border-radius:50%;'
                            f'background:#30e3ca;flex-shrink:0;margin-top:6px"></div>'
                            f'<div style="color:#374151;line-height:1.6">{f}</div></div>',
                            unsafe_allow_html=True
                        )

                # ── Conclusion / Limitations / Relevance ──────────────────────
                c4, c5, c6 = st.columns(3)
                c4.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Conclusion</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("conclusion","")}</div>',
                    unsafe_allow_html=True
                )
                c5.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Limitations</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("limitations","")}</div>',
                    unsafe_allow_html=True
                )
                c6.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                    f'letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;margin-bottom:4px">'
                    f'Relevance</div>'
                    f'<div style="font-size:0.85rem;color:#374151;line-height:1.6">'
                    f'{summary.get("relevance","")}</div>',
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
