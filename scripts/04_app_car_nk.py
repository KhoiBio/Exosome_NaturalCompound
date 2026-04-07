#!/usr/bin/env python3
"""
04_app.py — ExoRAG Decision Engine
====================================
Focus: CAR-NK-derived exosomes as cancer immunotherapy

Tabs:
  1. Literature Q&A         — RAG query with cited answers
  2. Target Ranker          — Score cancer targets by CAR-NK exosome evidence
  3. Mechanism Explorer     — Map NK kill mechanisms carried in exosomes
  4. Experimental Readiness — Can we run this assay tomorrow?
  5. Study Comparator       — In vitro vs in vivo vs clinical evidence gaps
  6. Feedback Loop          — Log internal results, build proprietary intelligence
"""

import os
import json
import re
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
COLLECTION    = "exorag_car_nk"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Prompts ──────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are a Senior Scientific Advisor specializing in CAR-NK cell therapy,
NK cell-derived exosomes, and cancer immunotherapy. Your focus is on the paradigm where
CAR-NK cells PRODUCE exosomes that are harvested and used as cell-free cancer therapeutics.

Answer with a TRANSLATIONAL R&D focus using retrieved PubMed literature.
Cite as "Author et al., YEAR (PMID: XXXXX)". Distinguish in vitro / in vivo / clinical evidence.
Highlight mechanistic insights about how NK-derived exosomes kill tumor cells
(perforin, granzyme, TRAIL, FasL, NKG2D ligands, CAR antigen recognition)."""

RANKER_SYSTEM = """You are a cancer immunotherapy R&D analyst. Rank cancer targets for
CAR-NK-derived exosome therapy based on the retrieved literature.

You MUST respond with ONLY a valid JSON array. No text before or after. No markdown. No explanation.

Each array item MUST have exactly these keys:
{"target": "name", "score": 7.5, "evidence_level": "Moderate", "cancer_type": "one phrase",
 "mechanism": "one phrase", "exosome_relevance": "one sentence", "novelty": "Medium", "summary": "one sentence"}

score: number 0-10
evidence_level: exactly one of: High, Moderate, Low, Preliminary
novelty: exactly one of: High, Medium, Low

OUTPUT ONLY THE JSON ARRAY. START WITH [ AND END WITH ]."""

MECHANISM_SYSTEM = """You are a CAR-NK exosome biologist. Map the cytotoxic mechanisms
carried by NK cell-derived exosomes based on the literature.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Each key is a mechanism category. Each value is an array of mechanism objects.

Required format:
{"Mechanism Category": [{"mechanism": "name", "evidence_strength": 3, "cargo": "protein/RNA", "notes": "one sentence"}]}

evidence_strength: integer 1-5 only (1=weak, 5=strong)

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

READINESS_SYSTEM = """You are a cancer immunology lab scientist. Assess experimental readiness
for testing CAR-NK-derived exosomes against a specific cancer target.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Required format:
{"target": "name", "overall_readiness": 7, "exosome_source": "NK-92 or primary NK",
 "isolation_method": "ultracentrifugation", "cancer_cell_lines": ["U87", "Raji"],
 "assays": ["cytotoxicity", "flow cytometry"], "timeline_days": 14,
 "blockers": ["CAR engineering complexity"], "next_step": "One concrete next action."}

overall_readiness: integer 0-10
timeline_days: integer

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

COMPARATOR_SYSTEM = """You are a translational cancer immunologist. Compare evidence across
study types for CAR-NK exosome therapy.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Required format:
{"in_vitro": {"strength": 7, "key_findings": ["finding 1", "finding 2"], "gaps": ["gap 1"]},
 "in_vivo": {"strength": 4, "key_findings": ["finding 1"], "gaps": ["gap 1", "gap 2"]},
 "clinical": {"strength": 1, "key_findings": [], "gaps": ["gap 1", "gap 2", "gap 3"]},
 "overall_verdict": "one paragraph summary",
 "biggest_translational_gap": "one sentence"}

strength: integer 0-10

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

# ─── Demo content ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What cytotoxic cargo do NK cell-derived exosomes carry to kill tumor cells?",
    "How do CAR-NK cells produce exosomes that inherit CAR targeting?",
    "What cancer types show the best response to NK-derived exosome therapy?",
    "Compare NK-derived exosomes vs CAR-T cells: safety and efficacy evidence.",
    "What are the best methods to isolate and purify CAR-NK-derived exosomes?",
    "How does perforin and granzyme packaging into NK exosomes work?",
    "What is the evidence for NK exosomes in glioblastoma treatment?",
    "What exosome engineering strategies improve CAR display on the surface?",
]

CANCER_TARGETS = [
    "CD19 — B-cell malignancies (ALL, CLL, lymphoma)",
    "CD20 — Non-Hodgkin lymphoma",
    "HER2 — Breast / gastric cancer",
    "EGFR — Glioblastoma / lung cancer",
    "GD2 — Neuroblastoma / glioma",
    "BCMA — Multiple myeloma",
    "CD33 — AML (acute myeloid leukemia)",
    "Mesothelin — Pancreatic / ovarian / mesothelioma",
    "PD-L1 — Solid tumors (immune checkpoint)",
    "EpCAM — Epithelial solid tumors",
    "CD123 — AML / BPDCN",
    "NKG2D ligands — Pan-tumor (stress ligands)",
]

MECHANISM_CATEGORIES = [
    "Perforin / Granzyme-mediated apoptosis",
    "TRAIL / FasL death receptor signaling",
    "CAR antigen-specific targeting",
    "NKG2D / DNAM-1 activating receptor cargo",
    "miRNA / siRNA gene silencing payload",
    "Immune checkpoint modulation (PD-1/PD-L1)",
    "Tumor microenvironment remodeling",
    "Cytokine delivery (IL-15, IFN-γ)",
]

CANCER_TYPES = [
    "Glioblastoma (GBM)",
    "Acute Myeloid Leukemia (AML)",
    "B-cell ALL / lymphoma",
    "Multiple Myeloma",
    "Breast Cancer",
    "Lung Cancer (NSCLC)",
    "Pancreatic Cancer",
    "Ovarian Cancer",
    "Neuroblastoma",
    "Colorectal Cancer",
]


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

def call_llm(system: str, user: str, backend: str, ollama_model: str, stream: bool = False):
    if "Claude" in backend:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        if stream:
            return client.messages.stream(
                model=CLAUDE_MODEL, max_tokens=2000,
                system=system, messages=[{"role": "user", "content": user}]
            )
        else:
            r = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=2000,
                system=system, messages=[{"role": "user", "content": user}]
            )
            return r.content[0].text
    else:
        prompt = f"{system}\n\n{user}"
        if stream:
            return requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": True},
                stream=True, timeout=180
            )
        else:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=180
            )
            return r.json().get("response", "")

def stream_response(backend, ollama_model, system, user):
    if "Claude" in backend:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        with client.messages.stream(
            model=CLAUDE_MODEL, max_tokens=2000,
            system=system, messages=[{"role": "user", "content": user}]
        ) as s:
            for t in s.text_stream:
                yield t
    else:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": ollama_model, "prompt": f"{system}\n\n{user}", "stream": True},
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

def parse_json_response(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    cleaned = text
    cleaned = re.sub(r",\s*([}\]])", r"", cleaned)
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r"//.*", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if objects:
        parsed = []
        for obj in objects:
            try:
                parsed.append(json.loads(obj))
            except Exception:
                obj_clean = re.sub(r",\s*}", "}", obj)
                try:
                    parsed.append(json.loads(obj_clean))
                except Exception:
                    continue
        if parsed:
            return parsed
    raise ValueError(f"Could not parse JSON from response. Raw text:\n{text[:300]}")


def normalize_ranker_data(data) -> list:
    if isinstance(data, list):
        normalized = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # handle "target" or "compound" or "name" key
                if "compound" in item and "target" not in item:
                    item["target"] = item.pop("compound")
                if "name" in item and "target" not in item:
                    item["target"] = item.pop("name")
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({
                    "target": item,
                    "score": max(8.0 - i * 0.5, 1.0),
                    "evidence_level": "Unknown",
                    "cancer_type": "",
                    "mechanism": "",
                    "exosome_relevance": "",
                    "novelty": "Unknown",
                    "summary": "",
                })
        return normalized
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict) and any(k in first for k in ["target", "compound", "name"]):
                    normalized = []
                    for item in val:
                        if isinstance(item, dict):
                            if "compound" in item and "target" not in item:
                                item["target"] = item.pop("compound")
                            if "name" in item and "target" not in item:
                                item["target"] = item.pop("name")
                            normalized.append(item)
                    return normalized
        values = list(data.values())
        if values and isinstance(values[0], dict) and "target" in values[0]:
            return list(values)
    return []


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

def retrieve_filtered(collection, query: str, k: int, study_type: str = None) -> list:
    """Retrieve with optional study_type filter (clinical / in_vivo / in_vitro)."""
    where = {"study_type": study_type} if study_type else None
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
    )
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
            f"Study type: {m.get('study_type','unknown')} | "
            f"Relevance score: {m.get('relevance_score','?')}\n"
            f"PMID: {m['pmid']}\nText: {c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f0f2f5; }
.hero {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1f3c 50%, #0a3d62 100%);
    border-radius: 16px; padding: 1.8rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0 0 0.2rem 0; color: #e8f4ff; }
.hero-sub   { color: #7ec8e3; font-size: 0.9rem; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; color: #b3e0ff;
    margin-right: 6px; margin-top: 10px; font-family: 'DM Mono', monospace;
}
.stat-card {
    background: white; border: 1px solid #e0e8f0; border-radius: 12px;
    padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #0a3d62; line-height: 1; }
.stat-lbl { color: #888; font-size: 0.72rem; margin-top: 4px; }
.answer-box {
    background: white; border: 1px solid #d0e4f0; border-left: 4px solid #0a3d62;
    border-radius: 12px; padding: 1.5rem; line-height: 1.7;
}
.rank-card {
    background: white; border: 1px solid #d8e8f4; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.rank-name  { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #0a3d62; }
.score-pill {
    display: inline-block; border-radius: 20px; padding: 3px 12px;
    font-family: 'DM Mono', monospace; font-size: 0.85rem; font-weight: 700;
}
.score-high   { background: #e3f2fd; color: #0d47a1; }
.score-mid    { background: #fff8e1; color: #e65100; }
.score-low    { background: #fce4ec; color: #880e4f; }
.mech-card {
    background: white; border: 1px solid #c8dff0; border-left: 4px solid #1565c0;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.mech-title { font-weight: 600; color: #1565c0; font-size: 0.95rem; margin-bottom: 0.4rem; }
.evidence-bar {
    background: #e3f2fd; border-radius: 6px; height: 8px; margin-top: 6px;
}
.badge-claude {
    display: inline-block; background: #fff3e0; color: #e65100;
    border: 1px solid #ffcc80; border-radius: 8px; padding: 3px 10px;
    font-size: 0.78rem; font-weight: 600;
}
.badge-ollama {
    display: inline-block; background: #e8f5e9; color: #1b5e20;
    border: 1px solid #a5d6a7; border-radius: 8px; padding: 3px 10px;
    font-size: 0.78rem; font-weight: 600;
}
.comparator-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #d0e4f0; margin-bottom: 0.8rem;
}
.setup-box {
    background: #e8f4fd; border: 1px solid #90caf9;
    border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.83rem;
}
.paradigm-box {
    background: linear-gradient(135deg, #0d1f3c, #0a3d62);
    border-radius: 10px; padding: 1rem 1.4rem; color: #b3e0ff;
    font-size: 0.85rem; margin-bottom: 1rem; font-family: 'DM Mono', monospace;
}
</style>
"""


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ExoRAG — CAR-NK Immunotherapy",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Backend detection ──
    claude_ok              = check_claude()
    ollama_ok, ollama_models = check_ollama()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🧬 ExoRAG")
        st.divider()

        # Paradigm reminder
        st.markdown("""
        <div class="paradigm-box">
        CAR-NK → secretes → Exosomes<br>
        ↓ harvest & purify<br>
        Exosome → kills → Tumor cell<br>
        <br>
        <span style="color:#7ec8e3">Cell-free · Off-the-shelf · Low CRS risk</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🤖 LLM Backend**")
        options = []
        if claude_ok:   options.append("🟠 Claude API  (Recommended)")
        if ollama_ok:   options.append("🟢 Ollama  (Free / Local)")
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
        k_val         = st.slider("Chunks retrieved (K)", 3, 15, 8)
        show_snippets = st.toggle("Show source snippets", value=True)
        st.divider()
        st.markdown("**Stack**")
        st.markdown("- 🧬 PubMed · ChromaDB · MiniLM · Streamlit")

    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <p class="hero-title">🧬 ExoRAG — CAR-NK Immunotherapy Engine</p>
        <p class="hero-sub">CAR-NK-Derived Exosomes · Cell-Free Cancer Therapy · Literature Intelligence</p>
        <span class="hero-badge">Literature Q&A</span>
        <span class="hero-badge">Target Ranker</span>
        <span class="hero-badge">Mechanism Explorer</span>
        <span class="hero-badge">Study Comparator</span>
    </div>""", unsafe_allow_html=True)

    # ── Load DB ──
    try:
        collection = load_collection()
        n_chunks   = collection.count()
    except Exception as e:
        st.error(f"Knowledge base not found: {e}")
        st.info("Run:\n```\npython scripts/01_fetch_pubmed.py\npython scripts/02_build_index.py\n```")
        return

    llm_label = "Claude" if "Claude" in backend else ollama_sel if "Ollama" in backend else "—"

    for col, num, lbl in zip(
        st.columns(4),
        [f"{n_chunks:,}", f"~{n_chunks//3}", str(k_val), llm_label],
        ["Indexed Chunks", "PubMed Papers", "Retrieved/Query", "LLM Engine"],
    ):
        col.markdown(
            f'<div class="stat-card"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Literature Q&A",
        "🎯 Target Ranker",
        "⚙️ Mechanism Explorer",
        "🔬 Experimental Readiness",
        "📊 Study Comparator",
        "🗂️ Feedback Loop",
    ])

    no_llm = "⚠️" in backend

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Literature Q&A
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### Quick Demo Queries")
        selected = None
        cols = st.columns(2)
        for i, dq in enumerate(DEMO_QUERIES):
            with cols[i % 2]:
                if st.button(f"▶ {dq[:65]}...", key=f"dq_{i}", use_container_width=True):
                    selected = dq
        st.divider()

        query = st.text_area(
            "Ask a research or translational question:",
            value=selected or "",
            height=80,
            placeholder="e.g. What cytotoxic mechanisms do NK-derived exosomes carry to kill GBM cells?",
            key="qa_query",
        )

        col_filter, _ = st.columns([2, 3])
        with col_filter:
            study_filter = st.selectbox(
                "Filter by study type:",
                ["All", "clinical", "in_vivo", "in_vitro", "review"],
                key="study_filter",
            )

        run_qa = st.button("🔍 Search & Analyze", type="primary", key="run_qa")

        if run_qa and query.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
                return

            sf = None if study_filter == "All" else study_filter
            try:
                chunks = retrieve_filtered(collection, query.strip(), k_val, sf)
            except Exception:
                chunks = retrieve(collection, query.strip(), k_val)

            context  = build_context(chunks)
            user_msg = f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\nQUESTION: {query.strip()}"

            col_ans, col_src = st.columns([3, 2])
            with col_ans:
                badge = f'<span class="{"badge-claude" if "Claude" in backend else "badge-ollama"}">{"🟠 Claude API" if "Claude" in backend else f"🟢 Ollama · {ollama_sel}"}</span>'
                st.markdown(f"#### 💡 Analysis &nbsp; {badge}", unsafe_allow_html=True)
                placeholder = st.empty()
                full_text   = ""
                try:
                    for token in stream_response(backend, ollama_sel, RAG_SYSTEM, user_msg):
                        full_text += token
                        placeholder.markdown(f'<div class="answer-box">{full_text}▌</div>', unsafe_allow_html=True)
                    placeholder.markdown(f'<div class="answer-box">{full_text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"LLM error: {e}")

            with col_src:
                st.markdown(f"#### 📄 Sources ({len(chunks)})")
                for i, c in enumerate(chunks):
                    m    = c["metadata"]
                    sim  = c["similarity"]
                    icon = "🟢" if sim > 0.7 else "🟡" if sim > 0.5 else "🔴"
                    study_icon = {"clinical": "🏥", "in_vivo": "🐭", "in_vitro": "🧫", "review": "📖"}.get(m.get("study_type",""), "📄")
                    with st.expander(f"{icon} [{i+1}] {m['title'][:48]}... ({m['year']})", expanded=i < 2):
                        st.markdown(f"**{m['title']}**")
                        st.markdown(f"*{m['authors']}*")
                        st.caption(f"`{m['journal']}` · {m['year']} · PMID {m['pmid']}")
                        st.caption(f"Relevance: **{sim:.3f}** | {study_icon} {m.get('study_type','?')}")
                        st.markdown(f"[PubMed ↗]({m['url']})")
                        if show_snippets:
                            st.divider()
                            st.caption(c["text"][:320] + "...")

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Target Ranker
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🎯 Cancer Target Ranker")
        st.markdown(
            "Select a cancer type and get antigen targets ranked by evidence for "
            "**CAR-NK-derived exosome** therapy."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            cancer_type = st.selectbox("Cancer type:", CANCER_TYPES, key="rank_cancer")
        with col2:
            top_n = st.slider("Top N targets", 3, 8, 5, key="rank_n")

        run_rank = st.button("🎯 Rank Targets", type="primary", key="run_rank")

        if run_rank:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"CAR-NK exosome {cancer_type} target antigen cytotoxicity"
                chunks   = retrieve(collection, search_q, min(k_val + 4, 15))
                context  = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Rank the top {top_n} cancer antigen targets for CAR-NK-derived exosome "
                    f"therapy in: {cancer_type}\n"
                    f"Focus on targets where NK exosomes carrying CAR + cytotoxic cargo "
                    f"(perforin/granzyme/TRAIL) show evidence.\n"
                    f"Return a JSON array with {top_n} targets ranked by overall score."
                )

                with st.spinner(f"Ranking targets for {cancer_type}..."):
                    try:
                        raw  = call_llm(RANKER_SYSTEM, user_msg, backend, ollama_sel)
                        data = normalize_ranker_data(parse_json_response(raw))

                        if not data:
                            st.warning("No targets extracted. Try reducing K or switching to Claude API.")
                            st.code(raw[:800])
                        else:
                            st.markdown(f"### Top {len(data)} Targets for **{cancer_type}**")
                            st.caption(f"Based on {len(chunks)} retrieved PubMed chunks · CAR-NK exosome focus")
                            st.divider()

                        for i, item in enumerate(data):
                            score = float(item.get("score", 0))
                            score_class = "score-high" if score >= 7 else "score-mid" if score >= 5 else "score-low"
                            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"

                            st.markdown(
                                f'<div class="rank-card">'
                                f'<span style="font-size:1.3rem">{medal}</span> '
                                f'<span class="rank-name"> {item.get("target","")}</span> '
                                f'<span class="score-pill {score_class}">{score:.1f}/10</span><br><br>'
                                f'<b>Cancer type:</b> {item.get("cancer_type","")}<br>'
                                f'<b>Mechanism:</b> {item.get("mechanism","")}<br>'
                                f'<b>Evidence level:</b> {item.get("evidence_level","")}&nbsp;&nbsp;'
                                f'<b>Novelty:</b> {item.get("novelty","")}<br>'
                                f'<b>Exosome relevance:</b> {item.get("exosome_relevance","")}<br><br>'
                                f'<i>{item.get("summary","")}</i>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.code(raw[:1500] if raw else "Empty response.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Mechanism Explorer
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### ⚙️ Mechanism Explorer")
        st.markdown(
            "Map the cytotoxic cargo and kill mechanisms carried by NK cell-derived exosomes. "
            "This is the core science behind using CAR-NK exosomes as therapeutic agents."
        )

        mech_target = st.selectbox("Focus on mechanism category:", MECHANISM_CATEGORIES, key="mech_target")
        run_mech = st.button("⚙️ Explore Mechanisms", type="primary", key="run_mech")

        if run_mech:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"NK exosome {mech_target} cancer cytotoxicity cargo"
                chunks   = retrieve(collection, search_q, min(k_val + 4, 15))
                context  = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Map the mechanisms for: {mech_target}\n"
                    f"Focus specifically on what molecular cargo is packaged into NK-derived exosomes "
                    f"and how it mediates tumor cell killing. Return JSON."
                )

                with st.spinner(f"Mapping mechanisms for {mech_target}..."):
                    try:
                        raw  = call_llm(MECHANISM_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        st.markdown(f"### Mechanism Map: **{mech_target}**")
                        st.caption(f"Based on {len(chunks)} PubMed sources")
                        st.divider()

                        for category, items in data.items():
                            if not items:
                                continue
                            with st.expander(f"**{category}** — {len(items)} mechanism(s)", expanded=True):
                                for item in items:
                                    if isinstance(item, str):
                                        item = {"mechanism": item, "evidence_strength": 0, "cargo": "", "notes": ""}
                                    elif not isinstance(item, dict):
                                        continue
                                    strength = min(max(int(item.get("evidence_strength", 0)), 0), 5)
                                    stars    = "⭐" * strength + "☆" * (5 - strength)
                                    bar_w    = strength * 20

                                    st.markdown(
                                        f'<div class="mech-card">'
                                        f'<span class="mech-title">{item.get("mechanism","")}</span> '
                                        f'<span style="font-size:0.8rem">{stars if strength > 0 else ""}</span><br>'
                                        f'<span style="font-size:0.82rem;color:#1565c0"><b>Cargo:</b> {item.get("cargo","")}</span><br>'
                                        f'<span style="font-size:0.82rem;color:#555">{item.get("notes","")}</span>'
                                        f'<div class="evidence-bar"><div style="background:#1565c0;width:{bar_w}%;height:8px;border-radius:6px"></div></div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.code(raw[:1500] if raw else "Empty response.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API.")

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Experimental Readiness
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 🔬 Experimental Readiness")
        st.markdown(
            "Can we test CAR-NK-derived exosomes against this target **this week**? "
            "Get a lab-ready protocol assessment from the literature."
        )

        col1, col2 = st.columns(2)
        with col1:
            ready_target = st.selectbox("Cancer target:", CANCER_TARGETS, key="ready_target")
        with col2:
            ready_cancer = st.selectbox("Cancer type:", CANCER_TYPES, key="ready_cancer")

        nk_source = st.radio(
            "NK cell source:",
            ["NK-92 cell line (easiest)", "Primary NK (PBMC-derived)", "iPSC-derived NK", "Cord blood NK"],
            horizontal=True,
            key="nk_source",
        )

        run_ready = st.button("🔬 Assess Readiness", type="primary", key="run_ready")

        if run_ready:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                target_name = ready_target.split("—")[0].strip()
                search_q = (
                    f"CAR-NK exosome {target_name} {ready_cancer} "
                    f"protocol isolation assay cytotoxicity"
                )
                chunks   = retrieve(collection, search_q, k_val)
                context  = build_context(chunks)
                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Target antigen: {target_name}\n"
                    f"Cancer type: {ready_cancer}\n"
                    f"NK cell source: {nk_source}\n"
                    f"Assess readiness to test CAR-NK-derived exosomes in the lab. Return only valid JSON."
                )

                with st.spinner(f"Assessing readiness for {target_name} / {ready_cancer}..."):
                    try:
                        raw  = call_llm(READINESS_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        score     = int(data.get("overall_readiness", 0))
                        score_col = "#0d47a1" if score >= 7 else "#e65100" if score >= 4 else "#b71c1c"
                        bar_pct   = score * 10

                        st.markdown(f"### {target_name} × {ready_cancer} — Readiness Assessment")
                        st.divider()

                        st.markdown(
                            f"""<div style="background:white;border:1px solid #d0e4f0;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">
                            <div style="font-size:0.85rem;color:#888;margin-bottom:6px">OVERALL READINESS</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:2.5rem;color:{score_col};line-height:1">{score}/10</div>
                            <div style="background:#f0f0f0;border-radius:8px;height:10px;margin-top:10px">
                              <div style="background:{score_col};width:{bar_pct}%;height:10px;border-radius:8px"></div>
                            </div>
                            <div style="margin-top:10px;font-size:0.85rem;color:#333"><b>Next step:</b> {data.get("next_step","")}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**🧫 Exosome Production**")
                            st.markdown(f"- **NK source:** {data.get('exosome_source','')}")
                            st.markdown(f"- **Isolation:** {data.get('isolation_method','')}")
                            st.markdown(f"- **Timeline:** ~{data.get('timeline_days','?')} days to first data")
                        with c2:
                            st.markdown("**🔬 Cell Lines & Assays**")
                            for cl in data.get("cancer_cell_lines", []):
                                st.markdown(f"- {cl}")
                            st.markdown("**Assays:**")
                            for assay in data.get("assays", []):
                                st.markdown(f"- {assay}")
                        with c3:
                            st.markdown("**⚠️ Potential Blockers**")
                            for b in data.get("blockers", []):
                                st.markdown(f"- {b}")

                        st.divider()
                        st.caption(f"Based on {len(chunks)} PubMed sources")

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.code(raw[:1500] if raw else "Empty response.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API.")

    # ════════════════════════════════════════════════════════════════
    # TAB 5 — Study Comparator
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### 📊 Study Comparator")
        st.markdown(
            "Compare the strength of evidence across **in vitro → in vivo → clinical** "
            "for a given CAR-NK exosome + cancer combination. Identify translational gaps."
        )

        col1, col2 = st.columns(2)
        with col1:
            comp_cancer = st.selectbox("Cancer type:", CANCER_TYPES, key="comp_cancer")
        with col2:
            comp_target = st.text_input(
                "Antigen target (optional):",
                placeholder="e.g. CD19, HER2, GD2",
                key="comp_target",
            )

        run_comp = st.button("📊 Compare Evidence", type="primary", key="run_comp")

        if run_comp:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                target_str = f"{comp_target} " if comp_target.strip() else ""
                search_q   = f"CAR-NK NK exosome {target_str}{comp_cancer} clinical vivo vitro"
                chunks     = retrieve(collection, search_q, min(k_val + 4, 15))
                context    = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Cancer: {comp_cancer}\n"
                    f"Target: {comp_target or 'any'}\n"
                    f"Compare the evidence across in vitro, in vivo, and clinical studies "
                    f"for CAR-NK-derived exosome therapy. Identify the biggest translational gap. "
                    f"Return only valid JSON."
                )

                with st.spinner(f"Comparing evidence for {comp_cancer}..."):
                    try:
                        raw  = call_llm(COMPARATOR_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        st.markdown(f"### Evidence Comparison: **{comp_cancer}** {('· ' + comp_target) if comp_target else ''}")
                        st.caption(f"Based on {len(chunks)} PubMed sources")
                        st.divider()

                        # Evidence strength bars
                        cols = st.columns(3)
                        for col, key, label, icon in [
                            (cols[0], "in_vitro",  "In Vitro",  "🧫"),
                            (cols[1], "in_vivo",   "In Vivo",   "🐭"),
                            (cols[2], "clinical",  "Clinical",  "🏥"),
                        ]:
                            section  = data.get(key, {})
                            strength = int(section.get("strength", 0))
                            bar_col  = "#0d47a1" if strength >= 7 else "#e65100" if strength >= 4 else "#b71c1c"
                            with col:
                                st.markdown(
                                    f'<div class="comparator-card">'
                                    f'<div style="font-size:1.2rem">{icon} <b>{label}</b></div>'
                                    f'<div style="font-family:DM Mono;font-size:1.8rem;color:{bar_col}">{strength}/10</div>'
                                    f'<div style="background:#f0f0f0;border-radius:6px;height:8px;margin:8px 0">'
                                    f'<div style="background:{bar_col};width:{strength*10}%;height:8px;border-radius:6px"></div></div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                if section.get("key_findings"):
                                    st.markdown("**Key findings:**")
                                    for f in section["key_findings"]:
                                        st.markdown(f"✅ {f}")
                                if section.get("gaps"):
                                    st.markdown("**Gaps:**")
                                    for g in section["gaps"]:
                                        st.markdown(f"⚠️ {g}")

                        st.divider()
                        st.markdown("### 🔍 Overall Verdict")
                        st.markdown(
                            f'<div class="answer-box">{data.get("overall_verdict","")}</div>',
                            unsafe_allow_html=True,
                        )
                        if data.get("biggest_translational_gap"):
                            st.error(f"**Biggest translational gap:** {data['biggest_translational_gap']}")

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.code(raw[:1500] if raw else "Empty response.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API.")

    # ════════════════════════════════════════════════════════════════
    # TAB 6 — Feedback Loop
    # ════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("#### 🗂️ Feedback Loop — Internal Experiment Log")
        st.markdown(
            "Log your CAR-NK exosome lab results. Over time this builds "
            "**proprietary knowledge** no published paper has."
        )

        FEEDBACK_FILE = Path(os.getenv("OUTPUT_DIR", "./outputs")) / "feedback_log.json"
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

        def load_feedback():
            if FEEDBACK_FILE.exists():
                try:
                    return json.loads(FEEDBACK_FILE.read_text())
                except Exception:
                    return []
            return []

        def save_feedback(records):
            FEEDBACK_FILE.write_text(json.dumps(records, indent=2))

        st.markdown("### ➕ Log a New Experiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            fb_target   = st.text_input("Cancer target / antigen:", key="fb_target",
                                         placeholder="e.g. CD19, HER2, GD2")
            fb_cancer   = st.selectbox("Cancer type:", CANCER_TYPES, key="fb_cancer")
        with col2:
            fb_nk_src   = st.text_input("NK cell source:", key="fb_nk_src",
                                         placeholder="e.g. NK-92, primary PBMC")
            fb_method   = st.text_input("Exosome isolation method:", key="fb_method",
                                         placeholder="e.g. ultracentrifugation, SEC")
        with col3:
            fb_result   = st.radio("Result:", ["✅ Cytotoxicity observed", "⚠️ Partial", "❌ No effect"],
                                    key="fb_result")
            fb_score    = st.slider("Efficacy score:", 0, 10, 5, key="fb_score")

        fb_assay = st.text_input("Assay used:", key="fb_assay",
                                  placeholder="e.g. LDH release, flow cytometry, CytoTox-Glo")
        fb_notes = st.text_area("Notes / observations:", height=80, key="fb_notes",
                                 placeholder="e.g. 40% cytotoxicity at 50µg/mL exosome dose, "
                                             "abolished by anti-TRAIL blocking antibody")
        fb_date  = st.date_input("Date:", key="fb_date")

        if st.button("💾 Save Experiment", type="primary", key="save_fb"):
            if fb_target.strip():
                records = load_feedback()
                records.append({
                    "id":               len(records) + 1,
                    "target":           fb_target.strip(),
                    "cancer_type":      fb_cancer,
                    "nk_source":        fb_nk_src,
                    "isolation_method": fb_method,
                    "assay":            fb_assay,
                    "result":           fb_result,
                    "score":            fb_score,
                    "notes":            fb_notes,
                    "date":             str(fb_date),
                    "logged_at":        datetime.now().isoformat(),
                })
                save_feedback(records)
                st.success(f"✅ Saved experiment: {fb_target} × {fb_cancer}")
                st.rerun()
            else:
                st.warning("Please enter a cancer target.")

        st.divider()

        records = load_feedback()

        if not records:
            st.info("No experiments logged yet. Start logging your lab results above.")
        else:
            st.markdown(f"### 📋 Experiment Log ({len(records)} entries)")

            worked   = sum(1 for r in records if "Cytotoxicity" in r.get("result",""))
            partial  = sum(1 for r in records if "Partial" in r.get("result",""))
            failed   = sum(1 for r in records if "No effect" in r.get("result",""))
            avg_score = sum(r.get("score",0) for r in records) / len(records) if records else 0

            s1, s2, s3, s4 = st.columns(4)
            for col, num, lbl, color in [
                (s1, worked,           "✅ Cytotoxic",  "#0d47a1"),
                (s2, partial,          "⚠️ Partial",    "#e65100"),
                (s3, failed,           "❌ No Effect",  "#b71c1c"),
                (s4, f"{avg_score:.1f}/10", "Avg Score", "#0a3d62"),
            ]:
                col.markdown(
                    f'<div class="stat-card"><div class="stat-num" style="color:{color}">{num}</div>'
                    f'<div class="stat-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            filter_result = st.multiselect(
                "Filter by result:",
                ["✅ Cytotoxicity observed", "⚠️ Partial", "❌ No effect"],
                default=["✅ Cytotoxicity observed", "⚠️ Partial", "❌ No effect"],
                key="fb_filter",
            )
            filtered = [r for r in records if r.get("result","") in filter_result]

            for r in reversed(filtered):
                result_color  = {"✅ Cytotoxicity observed": "#e3f2fd", "⚠️ Partial": "#fff8e1", "❌ No effect": "#fce4ec"}.get(r.get("result",""), "white")
                border_color  = {"✅ Cytotoxicity observed": "#0d47a1", "⚠️ Partial": "#e65100", "❌ No effect": "#c62828"}.get(r.get("result",""), "#ccc")

                with st.expander(
                    f"{r.get('result','')} **{r.get('target','')}** × {r.get('cancer_type','')} "
                    f"| Score: {r.get('score','')}/10 | {r.get('date','')}",
                    expanded=False,
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Target:** {r.get('target','')}")
                        st.markdown(f"**Cancer:** {r.get('cancer_type','')}")
                        st.markdown(f"**NK source:** {r.get('nk_source','')}")
                        st.markdown(f"**Isolation:** {r.get('isolation_method','')}")
                    with c2:
                        st.markdown(f"**Assay:** {r.get('assay','')}")
                        st.markdown(f"**Result:** {r.get('result','')}")
                        st.markdown(f"**Score:** {r.get('score','')}/10")
                        st.markdown(f"**Date:** {r.get('date','')}")
                    if r.get("notes"):
                        st.markdown(f"**Notes:** {r.get('notes','')}")

            st.divider()

            # AI synthesis
            if len(records) >= 3 and not no_llm:
                st.markdown("### 🤖 AI Synthesis of Internal Results")
                st.caption("Combines your experiment log with PubMed for proprietary insights")

                if st.button("🧠 Synthesize Intelligence", key="synthesize"):
                    summary_data = [
                        {k: r.get(k,"") for k in ["target","cancer_type","nk_source","result","score","assay","notes"]}
                        for r in records
                    ]
                    search_q = " ".join(set(
                        f"{r.get('target','')} {r.get('cancer_type','')}" for r in records[:5]
                    ))
                    chunks   = retrieve(collection, search_q + " CAR-NK exosome", min(k_val, 8))
                    context  = build_context(chunks)

                    synthesis_prompt = f"""You are analyzing PROPRIETARY internal CAR-NK exosome lab results
combined with published literature.

INTERNAL EXPERIMENT LOG:
{json.dumps(summary_data, indent=2)}

PUBLISHED LITERATURE CONTEXT:
{context}

Provide a strategic synthesis covering:
1. Which cancer targets / NK sources are working vs failing internally vs what literature predicts
2. Surprising discrepancies between your results and published data
3. Top 3 actionable recommendations based on BOTH internal results AND literature
4. Which target/cancer combinations to prioritize next
5. Any patterns in what drives cytotoxicity (NK source, isolation method, assay, dose)

Be specific. Use the actual targets, cancer types, and scores from the internal data."""

                    with st.spinner("Synthesizing your data with PubMed..."):
                        placeholder = st.empty()
                        full_text   = ""
                        for token in stream_response(backend, ollama_sel, RAG_SYSTEM, synthesis_prompt):
                            full_text += token
                            placeholder.markdown(
                                f'<div class="answer-box">{full_text}▌</div>',
                                unsafe_allow_html=True,
                            )
                        placeholder.markdown(
                            f'<div class="answer-box">{full_text}</div>',
                            unsafe_allow_html=True,
                        )
            elif len(records) < 3:
                st.info(f"Log at least 3 experiments to unlock AI synthesis. ({len(records)}/3 logged)")

            st.divider()
            if st.button("⬇️ Export to JSON", key="export_fb"):
                st.download_button(
                    label="Download feedback_log.json",
                    data=json.dumps(records, indent=2),
                    file_name="feedback_log.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
