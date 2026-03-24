#!/usr/bin/env python3
"""
04_app.py
=========
Streamlit UI for ExoRAG — your interview demo.
Uses Claude as the LLM, local ChromaDB + local embeddings.

Usage:
    streamlit run scripts/04_app.py
    Opens: http://localhost:8501
"""

import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import anthropic

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION    = "exorag_cosmetic"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")

SYSTEM_PROMPT = """You are a Senior Scientific Advisor specializing in exosome-based cosmetics 
and natural compound therapeutics. Your expertise covers exosome biology, natural compound 
pharmacology, skincare formulation, plant-derived nanoparticles, and translational research.

Answer questions with a PRODUCT DEVELOPMENT focus using the retrieved PubMed literature:
1. **Key findings** — what the evidence actually shows
2. **Ingredient/compound candidates** — specific actives with promise  
3. **Formulation insights** — loading methods, delivery routes, stability considerations
4. **Evidence quality** — distinguish in vitro / in vivo / clinical clearly
5. **Product opportunities** — research gaps = innovation white space

Cite papers as "AuthorLastName et al., YEAR". Be specific and actionable.
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


# ─── Cached Resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION, embedding_function=ef)


@st.cache_resource
def load_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# ─── RAG Functions ────────────────────────────────────────────────────────────

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

    # ── CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #faf9f6; color: #1a1a2e; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        margin: 0 0 0.3rem 0;
        color: #e8f4f8;
    }
    .hero-sub {
        color: #a8d8ea;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #cce5ff;
        margin-right: 8px;
        margin-top: 12px;
        font-family: 'DM Mono', monospace;
    }
    .stat-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-num {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #0f3460;
        line-height: 1;
    }
    .stat-lbl {
        color: #888;
        font-size: 0.75rem;
        margin-top: 4px;
    }
    .answer-box {
        background: white;
        border: 1px solid #e0e7f0;
        border-left: 4px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
    }
    .source-card {
        background: white;
        border: 1px solid #eaecef;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s;
    }
    .source-title {
        font-weight: 600;
        font-size: 0.83rem;
        color: #0f3460;
        margin-bottom: 3px;
        line-height: 1.3;
    }
    .source-meta {
        font-size: 0.72rem;
        color: #888;
        font-family: 'DM Mono', monospace;
    }
    .sim-pill {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 0.68rem;
        font-family: 'DM Mono', monospace;
        margin-left: 6px;
    }
    .topic-pill {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 0.68rem;
        margin-left: 4px;
    }
    .demo-section { margin-bottom: 1rem; }
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-size: 0.78rem;
        text-align: left;
        white-space: normal;
        height: auto;
        padding: 6px 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🌿 ExoRAG Settings")
        st.divider()
        k_val         = st.slider("Chunks retrieved (K)", 3, 15, 8)
        show_snippets = st.toggle("Show source text", value=True)
        st.divider()
        st.markdown("**Stack**")
        st.markdown("""
        - 🧬 **Data:** PubMed Entrez API
        - 💾 **Vector DB:** ChromaDB (local)
        - 🤖 **Embeddings:** MiniLM-L6-v2 (local)
        - 🟠 **LLM:** Claude 3.5 Haiku
        - 🖥 **UI:** Streamlit
        """)
        st.divider()
        st.markdown(f"**Model:** `{CLAUDE_MODEL}`")

    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <p class="hero-title">🌿 ExoRAG</p>
        <p class="hero-sub">Exosome · Cosmetic · Natural Compound Literature Intelligence</p>
        <span class="hero-badge">PubMed</span>
        <span class="hero-badge">ChromaDB</span>
        <span class="hero-badge">Claude AI</span>
        <span class="hero-badge">Local Embeddings</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Load resources ──
    try:
        collection = load_collection()
        client     = load_client()
        n_chunks   = collection.count()
        est_papers = n_chunks // 3
    except Exception as e:
        st.error(f"Could not load knowledge base: {e}")
        st.info("Run: `python scripts/02_build_index.py`")
        return

    # ── Stats ──
    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, f"{n_chunks:,}", "Indexed Chunks"),
        (c2, f"~{est_papers}", "PubMed Papers"),
        (c3, str(k_val), "Retrieved / Query"),
        (c4, "Claude", "LLM Engine"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demo queries ──
    st.markdown("**Quick Demo Queries**")
    cols = st.columns(2)
    selected = None
    for i, dq in enumerate(DEMO_QUERIES):
        with cols[i % 2]:
            label = f"▶ {dq[:62]}{'...' if len(dq) > 62 else ''}"
            if st.button(label, key=f"dq_{i}", use_container_width=True):
                selected = dq

    st.divider()

    # ── Query input ──
    query = st.text_area(
        "Ask a product development question:",
        value=selected or "",
        height=85,
        placeholder="e.g. Which plant-derived exosome nanoparticles show the best skin penetration for natural cosmetics?",
    )

    run = st.button("🔍 Search & Analyze", type="primary")

    # ── Results ──
    if run and query.strip():
        q = query.strip()

        # Retrieve
        with st.spinner("Retrieving literature..."):
            chunks = retrieve(collection, q, k_val)

        context = build_context(chunks)
        user_msg = (
            f"RETRIEVED LITERATURE CONTEXT:\n\n{context}\n\n"
            f"---\n\nQUESTION: {q}"
        )

        col_ans, col_src = st.columns([3, 2])

        with col_ans:
            st.markdown("#### 💡 Analysis")
            placeholder = st.empty()
            full_text   = ""

            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=1800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    placeholder.markdown(
                        f'<div class="answer-box">{full_text}▌</div>',
                        unsafe_allow_html=True,
                    )

            placeholder.markdown(
                f'<div class="answer-box">{full_text}</div>',
                unsafe_allow_html=True,
            )

        with col_src:
            st.markdown(f"#### 📄 Sources ({len(chunks)})")
            for i, chunk in enumerate(chunks):
                m   = chunk["metadata"]
                sim = chunk["similarity"]
                sim_color = "🟢" if sim > 0.7 else "🟡" if sim > 0.5 else "🔴"
                topic_label = m.get("topic", "").replace("_", " ")

                with st.expander(
                    f"{sim_color} [{i+1}] {m['title'][:50]}... ({m['year']})",
                    expanded=(i < 2)
                ):
                    st.markdown(f"**{m['title']}**")
                    st.markdown(f"*{m['authors']}*")
                    st.caption(f"`{m['journal']}` · {m['year']} · PMID {m['pmid']}")
                    st.caption(f"Relevance: **{sim:.3f}** · Topic: *{topic_label}*")
                    st.markdown(f"[Open in PubMed ↗]({m['url']})")
                    if show_snippets:
                        st.divider()
                        snippet = chunk["text"][:380]
                        st.caption(snippet + ("..." if len(chunk["text"]) > 380 else ""))

        # Session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "q":    q,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

    elif run:
        st.warning("Please enter a question.")

    # ── History ──
    if st.session_state.get("history"):
        st.divider()
        with st.expander(f"📜 Session history ({len(st.session_state.history)} queries)"):
            for item in reversed(st.session_state.history):
                st.markdown(f"`{item['time']}` — {item['q']}")


if __name__ == "__main__":
    main()
