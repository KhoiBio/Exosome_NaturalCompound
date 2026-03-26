#!/usr/bin/env python3
"""
04_app.py
=========
Streamlit UI for ExoRAG.
Supports two LLM backends — user chooses at runtime in the sidebar:
  - Claude API  (best quality, requires Anthropic API key)
  - Ollama      (free, fully local, no API key needed)

Usage:
    streamlit run scripts/04_app.py
    Opens: http://localhost:8501
"""

import os
import json
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

SYSTEM_PROMPT = """You are a Senior Scientific Advisor specializing in exosome-based cosmetics
and natural compound therapeutics. Your expertise covers exosome biology, natural compound
pharmacology, skincare formulation, plant-derived nanoparticles, and translational research.

Answer questions with a PRODUCT DEVELOPMENT focus using the retrieved PubMed literature:
1. **Key findings** — what the evidence actually shows
2. **Ingredient/compound candidates** — specific actives with promise
3. **Formulation insights** — loading methods, delivery routes, stability considerations
4. **Evidence quality** — distinguish in vitro / in vivo / clinical clearly
5. **Product opportunities** — research gaps = innovation white space

Always cite papers as "AuthorLastName et al., YEAR". Be specific and actionable.
Format responses with clear headers for complex questions."""

DEMO_QUERIES = [
    "Which natural compounds in exosomes show the strongest anti-aging skin evidence?",
    "What plant-derived exosome-like nanoparticles work best for cosmetic delivery?",
    "How are lipophilic compounds like curcumin loaded into exosomes for skincare?",
    "What's the evidence for exosome + natural compound combinations in wound healing?",
    "Which ingredients show promise for exosome-based hair growth formulations?",
    "Where are the product gaps in natural compound exosome cosmetics?",
    "Compare curcumin vs quercetin vs resveratrol for exosome skin delivery.",
    "What natural compounds brighten skin when delivered via exosomes?",
]


# ─── LLM Backends ─────────────────────────────────────────────────────────────

def check_claude_available() -> bool:
    return bool(ANTHROPIC_KEY and not ANTHROPIC_KEY.startswith("sk-ant-your"))


def check_ollama_available() -> tuple:
    """Returns (is_running: bool, model_list: list)"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []


def stream_claude(context: str, query: str):
    """Stream tokens from Claude API."""
    import anthropic
    client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    user_msg = f"RETRIEVED LITERATURE CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {query}"
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_ollama(context: str, query: str, model: str):
    """Stream tokens from local Ollama."""
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"RETRIEVED LITERATURE CONTEXT:\n\n{context}\n\n"
        f"---\n\nQUESTION: {query}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    )
    for line in resp.iter_lines():
        if line:
            try:
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except Exception:
                continue


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
        {
            "text":       doc,
            "metadata":   meta,
            "similarity": round(1 - dist, 4),
        }
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
            f"[Source {i+1}]\n"
            f"Title: {m['title']}\n"
            f"Authors: {m['authors']} | Year: {m['year']} | Journal: {m['journal']}\n"
            f"PMID: {m['pmid']}\n"
            f"Text: {c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ─── UI ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ExoRAG — Cosmetic Intelligence",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #faf9f6; color: #1a1a2e; }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
    }
    .hero-title { font-family: 'DM Serif Display', serif; font-size: 2.2rem; margin: 0 0 0.3rem 0; color: #e8f4f8; }
    .hero-sub   { color: #a8d8ea; font-size: 0.95rem; font-weight: 300; margin: 0; }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25); border-radius: 20px;
        padding: 3px 12px; font-size: 0.75rem; color: #cce5ff;
        margin-right: 8px; margin-top: 12px; font-family: 'DM Mono', monospace;
    }
    .stat-card {
        background: white; border: 1px solid #e8e8e8; border-radius: 12px;
        padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-num { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #0f3460; line-height: 1; }
    .stat-lbl { color: #888; font-size: 0.75rem; margin-top: 4px; }
    .answer-box {
        background: white; border: 1px solid #e0e7f0; border-left: 4px solid #0f3460;
        border-radius: 12px; padding: 1.5rem; line-height: 1.7;
    }
    .badge-claude {
        display: inline-block; background: #fff3e0; color: #e65100;
        border: 1px solid #ffcc80; border-radius: 8px;
        padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
    }
    .badge-ollama {
        display: inline-block; background: #e8f5e9; color: #1b5e20;
        border: 1px solid #a5d6a7; border-radius: 8px;
        padding: 3px 10px; font-size: 0.78rem; font-weight: 600;
    }
    .setup-box {
        background: #fff8e1; border: 1px solid #ffe082;
        border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Detect available backends ──────────────────────────────────────────────
    claude_ok              = check_claude_available()
    ollama_ok, ollama_models = check_ollama_available()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌿 ExoRAG")
        st.divider()

        # ── LLM selector ──────────────────────────────────────────────────────
        st.markdown("**🤖 LLM Backend**")

        options = []
        if claude_ok:
            options.append("🟠 Claude API  (Recommended)")
        if ollama_ok:
            options.append("🟢 Ollama  (Free / Local)")
        if not options:
            options.append("⚠️  No LLM found — see setup below")

        backend = st.radio("Choose LLM:", options)

        # Ollama model picker
        ollama_model_selected = OLLAMA_MODEL
        if "Ollama" in backend and ollama_models:
            ollama_model_selected = st.selectbox(
                "Ollama model:",
                ollama_models,
                help="Recommended: llama3.2 or mistral"
            )

        # Setup help when nothing is configured
        if "⚠️" in backend:
            st.divider()
            st.markdown("""
            <div class="setup-box">
            <b>Set up an LLM to get started:</b><br><br>
            <b>Option A — Claude API:</b><br>
            1. Get a free key at <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a><br>
            2. Add to <code>.env</code>:<br>
            <code>ANTHROPIC_API_KEY=sk-ant-...</code><br>
            3. Restart the app<br><br>
            <b>Option B — Ollama (no account needed):</b><br>
            1. Download from <a href="https://ollama.com" target="_blank">ollama.com</a><br>
            2. In a terminal: <code>ollama pull llama3.2</code><br>
            3. Then: <code>ollama serve</code><br>
            4. Restart the app
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        k_val         = st.slider("Chunks retrieved (K)", 3, 15, 8)
        show_snippets = st.toggle("Show source snippets", value=True)
        st.divider()
        st.markdown("**Stack**")
        st.markdown("""
        - 🧬 **Data:** PubMed Entrez API
        - 💾 **Vector DB:** ChromaDB (local)
        - 🔢 **Embeddings:** MiniLM-L6 (local)
        - 🖥 **UI:** Streamlit
        """)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <p class="hero-title">🌿 ExoRAG</p>
        <p class="hero-sub">Exosome · Cosmetic · Natural Compound Literature Intelligence</p>
        <span class="hero-badge">PubMed</span>
        <span class="hero-badge">ChromaDB</span>
        <span class="hero-badge">Local Embeddings</span>
        <span class="hero-badge">Claude API / Ollama</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load ChromaDB ─────────────────────────────────────────────────────────
    try:
        collection = load_collection()
        n_chunks   = collection.count()
        est_papers = n_chunks // 3
    except Exception as e:
        st.error(f"Knowledge base not found: {e}")
        st.info("Run these first:\n```\npython scripts/01_fetch_pubmed.py\npython scripts/02_build_index.py\n```")
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    llm_label = (
        "Claude" if "Claude" in backend
        else ollama_model_selected if "Ollama" in backend
        else "—"
    )
    for col, num, lbl in zip(
        st.columns(4),
        [f"{n_chunks:,}", f"~{est_papers}", str(k_val), llm_label],
        ["Indexed Chunks",  "PubMed Papers",  "Retrieved / Query", "LLM Engine"],
    ):
        col.markdown(
            f'<div class="stat-card"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demo queries ──────────────────────────────────────────────────────────
    st.markdown("**Quick Demo Queries**")
    selected = None
    for i, dq in enumerate(DEMO_QUERIES):
        with st.columns(2)[i % 2]:
            if st.button(f"▶ {dq[:62]}...", key=f"dq_{i}", use_container_width=True):
                selected = dq

    st.divider()

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.text_area(
        "Ask a product development question:",
        value=selected or "",
        height=85,
        placeholder="e.g. Which plant-derived exosome nanoparticles show the best skin penetration?",
    )
    run = st.button("🔍 Search & Analyze", type="primary")

    # ── Results ───────────────────────────────────────────────────────────────
    if run and query.strip():
        if "⚠️" in backend:
            st.error("No LLM configured — see the sidebar for setup instructions.")
            return

        q       = query.strip()
        chunks  = retrieve(collection, q, k_val)
        context = build_context(chunks)

        col_ans, col_src = st.columns([3, 2])

        with col_ans:
            if "Claude" in backend:
                badge = '<span class="badge-claude">🟠 Claude API</span>'
            else:
                badge = f'<span class="badge-ollama">🟢 Ollama · {ollama_model_selected}</span>'

            st.markdown(f"#### 💡 Analysis &nbsp; {badge}", unsafe_allow_html=True)

            placeholder = st.empty()
            full_text   = ""

            try:
                generator = (
                    stream_claude(context, q)
                    if "Claude" in backend
                    else stream_ollama(context, q, ollama_model_selected)
                )
                for token in generator:
                    full_text += token
                    placeholder.markdown(
                        f'<div class="answer-box">{full_text}▌</div>',
                        unsafe_allow_html=True,
                    )
                placeholder.markdown(
                    f'<div class="answer-box">{full_text}</div>',
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"LLM error: {e}")
                if "Claude" in backend:
                    st.info("Check your ANTHROPIC_API_KEY in .env")
                else:
                    st.info("Make sure Ollama is running: open a terminal and run `ollama serve`")

        with col_src:
            st.markdown(f"#### 📄 Sources ({len(chunks)})")
            for i, chunk in enumerate(chunks):
                m   = chunk["metadata"]
                sim = chunk["similarity"]
                icon = "🟢" if sim > 0.7 else "🟡" if sim > 0.5 else "🔴"
                with st.expander(
                    f"{icon} [{i+1}] {m['title'][:50]}... ({m['year']})",
                    expanded=(i < 2),
                ):
                    st.markdown(f"**{m['title']}**")
                    st.markdown(f"*{m['authors']}*")
                    st.caption(f"`{m['journal']}` · {m['year']} · PMID {m['pmid']}")
                    st.caption(f"Relevance: **{sim:.3f}**")
                    st.markdown(f"[Open in PubMed ↗]({m['url']})")
                    if show_snippets:
                        st.divider()
                        st.caption(chunk["text"][:380] + "...")

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "q": q, "llm": llm_label,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

    elif run:
        st.warning("Please enter a question.")

    if st.session_state.get("history"):
        st.divider()
        with st.expander(f"📜 Session history ({len(st.session_state.history)} queries)"):
            for item in reversed(st.session_state.history):
                st.markdown(f"`{item['time']}` [{item['llm']}] — {item['q']}")


if __name__ == "__main__":
    main()
