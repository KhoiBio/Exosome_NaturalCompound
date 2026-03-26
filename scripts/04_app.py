#!/usr/bin/env python3
"""
04_app.py — ExoRAG Decision Engine
====================================
Tabs:
  1. Literature Q&A        — RAG query with cited answers
  2. Compound Ranker       — Score compounds by evidence, safety, novelty
  3. Claim Generator       — Auto-generate cosmetic claim language
  4. Mechanism Clusters    — Group compounds by biological target
  5. Experimental Readiness — Can we test this tomorrow?
  6. Feedback Loop         — Log results, build company intelligence
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
COLLECTION    = "exorag_cosmetic"
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Prompts ──────────────────────────────────────────────────────────────────

RAG_SYSTEM = """You are a Senior Scientific Advisor specializing in exosome-based cosmetics
and natural compound therapeutics. Answer with a PRODUCT DEVELOPMENT focus using retrieved
PubMed literature. Cite as "Author et al., YEAR". Distinguish in vitro/in vivo/clinical."""

RANKER_SYSTEM = """You are a cosmetic R&D decision analyst. Rank natural compounds from the literature.

You MUST respond with ONLY a valid JSON array. No text before or after. No markdown. No explanation.

Each array item MUST have exactly these keys:
{"compound": "name", "score": 7.5, "evidence_level": "Moderate", "mechanism": "one phrase", "safety_profile": "one sentence", "novelty": "Medium", "summary": "one sentence"}

score: number 0-10
evidence_level: exactly one of: High, Moderate, Low, Preliminary
novelty: exactly one of: High, Medium, Low

Example of valid output (DO NOT copy this, use real data):
[{"compound": "Curcumin", "score": 8.2, "evidence_level": "High", "mechanism": "NF-kB inhibition", "safety_profile": "Well-established safety profile in cosmetics.", "novelty": "Medium", "summary": "Strong anti-inflammatory evidence in multiple cell models."}]

OUTPUT ONLY THE JSON ARRAY. START WITH [ AND END WITH ]."""

CLAIM_SYSTEM = """You are a cosmetic claims specialist. Generate claim language from scientific literature.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Required format:
{"primary_claim": "short claim here", "supporting_claims": ["claim 1", "claim 2", "claim 3"], "evidence_basis": "in vitro", "caution": "one sentence note"}

evidence_basis must be exactly one of: in vitro, in vivo, clinical, mixed

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

READINESS_SYSTEM = """You are a cosmetic R&D lab scientist. Assess experimental readiness.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Required format:
{"compound": "name", "overall_readiness": 7, "solubility": "Lipid-soluble", "typical_concentration": "5-10 uM", "cell_models": ["HaCaT", "fibroblast"], "assays": ["MTT", "ELISA"], "suppliers": ["Sigma-Aldrich"], "timeline_days": 7, "blockers": ["poor solubility"], "next_step": "Order from Sigma and prepare DMSO stock."}

overall_readiness: integer 0-10
solubility: exactly one of: Water-soluble, Lipid-soluble, Both, Unknown
timeline_days: integer (days to first data point)

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

CLUSTER_SYSTEM = """You are a cosmetic R&D scientist. Cluster natural compounds by mechanism.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Each key is a mechanism name. Each value is an array of compound objects.

Required format:
{"Mechanism Name": [{"compound": "name", "evidence_strength": 3, "notes": "one sentence"}]}

evidence_strength: integer 1-5 only (1=weak, 5=strong)

Example (DO NOT copy, use real data from literature):
{"Collagen Stimulation": [{"compound": "Vitamin C", "evidence_strength": 4, "notes": "Directly stimulates collagen synthesis in fibroblasts."}]}

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

READINESS_SYSTEM = """You are a cosmetic R&D lab scientist. Assess experimental readiness.

You MUST respond with ONLY a valid JSON object. No text before or after. No markdown.

Required format:
{"compound": "name", "overall_readiness": 7, "solubility": "Lipid-soluble", "typical_concentration": "5-10 uM", "cell_models": ["HaCaT", "fibroblast"], "assays": ["MTT", "ELISA"], "suppliers": ["Sigma-Aldrich"], "timeline_days": 7, "blockers": ["poor solubility"], "next_step": "Order from Sigma and prepare DMSO stock."}

overall_readiness: integer 0-10
solubility: exactly one of: Water-soluble, Lipid-soluble, Both, Unknown
timeline_days: integer (days to first data point)

OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }."""

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

APPLICATIONS = [
    "Anti-aging / wrinkle reduction",
    "Skin brightening / hyperpigmentation",
    "Wound healing / skin repair",
    "Hair growth / alopecia",
    "Anti-inflammatory / acne",
    "Moisturizing / barrier function",
    "Collagen synthesis",
    "Antioxidant protection",
]

MECHANISMS = [
    "Melanin inhibition / skin brightening",
    "Collagen synthesis / anti-aging",
    "Antioxidant / ROS scavenging",
    "Anti-inflammatory",
    "Wnt/β-catenin (hair growth)",
    "Cell proliferation / wound healing",
    "Barrier function / ceramide",
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
    """Single entry point for both Claude and Ollama."""
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
                json={"model": ollama_model, "prompt": prompt, "stream": False,
                      "format": "json"},   # force JSON output from Ollama
                timeout=180
            )
            return r.json().get("response", "")

def stream_response(backend, ollama_model, system, user):
    """Yield tokens from either backend."""
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
    """
    Robust JSON parser — handles all the ways Ollama can mangle output:
    - Markdown fences (```json ... ```)
    - Extra text before/after the JSON
    - Single quotes instead of double quotes
    - Trailing commas
    - Truncated output
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON array [...] or object {...}
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Fix common Ollama issues: trailing commas, single quotes
    cleaned = text
    cleaned = re.sub(r",\s*([}\]])", r"", cleaned)   # trailing commas
    cleaned = cleaned.replace("'", '"')                  # single → double quotes
    cleaned = re.sub(r"//.*", "", cleaned)               # strip JS comments
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract individual JSON objects and build array
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
    """
    Normalize whatever Ollama returns into a list of compound dicts.
    Handles all the weird formats llama3.2 produces:
      - Correct: [{"compound": "X", "score": 8, ...}]
      - Flat obj: {"compound1": "Aloe Vera", "compound2": "Vit C", "rank": "[0,1,2]"}
      - Simple list: ["Aloe Vera", "Vitamin C"]
      - Nested: {"compounds": [...]}
      - Dict of dicts: {"1": {"compound": "X",...}, "2": {...}}
    """
    # Already a list of dicts — ideal case
    if isinstance(data, list):
        normalized = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                # Plain string — wrap it
                normalized.append({
                    "compound": item,
                    "score": max(8.0 - i * 0.5, 1.0),
                    "evidence_level": "Unknown",
                    "mechanism": "",
                    "safety_profile": "",
                    "novelty": "Unknown",
                    "summary": "",
                })
        return normalized

    # Dict with any list value that looks like compounds
    if isinstance(data, dict):
        # Check all values for a list of compound-like dicts
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict) and any(k in first for k in ["compound", "name", "ingredient"]):
                    # Normalize "name" or "ingredient" key to "compound"
                    normalized = []
                    for item in val:
                        if isinstance(item, dict):
                            if "name" in item and "compound" not in item:
                                item["compound"] = item.pop("name")
                            if "ingredient" in item and "compound" not in item:
                                item["compound"] = item.pop("ingredient")
                            normalized.append(item)
                    return normalized

        # Dict of dicts: {"1": {"compound":...}, "2": {...}}
        values = list(data.values())
        if values and isinstance(values[0], dict) and "compound" in values[0]:
            return list(values)

        # Flat object: {"compound1": "Aloe", "compound2": "VitC", ...}
        # Extract compound-like keys and build minimal dicts
        compound_names = []
        for k, v in data.items():
            if isinstance(v, str) and k.lower().startswith("compound"):
                compound_names.append(v)
        if compound_names:
            return [
                {
                    "compound": name,
                    "score": max(8.0 - i * 0.5, 1.0),
                    "evidence_level": "Unknown",
                    "mechanism": "",
                    "safety_profile": "",
                    "novelty": "Unknown",
                    "summary": f"Identified by LLM for this application.",
                }
                for i, name in enumerate(compound_names)
            ]

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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f8f9fb; }
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 1.8rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0 0 0.2rem 0; color: #e8f4f8; }
.hero-sub   { color: #a8d8ea; font-size: 0.9rem; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25); border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; color: #cce5ff;
    margin-right: 6px; margin-top: 10px; font-family: 'DM Mono', monospace;
}
.stat-card {
    background: white; border: 1px solid #e8e8e8; border-radius: 12px;
    padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #0f3460; line-height: 1; }
.stat-lbl { color: #888; font-size: 0.72rem; margin-top: 4px; }
.answer-box {
    background: white; border: 1px solid #e0e7f0; border-left: 4px solid #0f3460;
    border-radius: 12px; padding: 1.5rem; line-height: 1.7;
}
.rank-card {
    background: white; border: 1px solid #e0e7f0; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.rank-name  { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #0f3460; }
.score-pill {
    display: inline-block; border-radius: 20px; padding: 3px 12px;
    font-family: 'DM Mono', monospace; font-size: 0.85rem; font-weight: 700;
}
.score-high   { background: #e8f5e9; color: #1b5e20; }
.score-mid    { background: #fff8e1; color: #e65100; }
.score-low    { background: #fce4ec; color: #880e4f; }
.claim-card {
    background: white; border: 1px solid #c8e6c9; border-left: 4px solid #2e7d32;
    border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
}
.primary-claim {
    font-family: 'DM Serif Display', serif; font-size: 1.3rem;
    color: #1b5e20; margin-bottom: 0.8rem;
}
.cluster-card {
    background: white; border: 1px solid #e3f2fd; border-left: 4px solid #1565c0;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.mechanism-title { font-weight: 600; color: #1565c0; font-size: 0.95rem; margin-bottom: 0.4rem; }
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
.setup-box {
    background: #fff8e1; border: 1px solid #ffe082;
    border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.83rem;
}
</style>
"""


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="ExoRAG Decision Engine",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Backend detection ──
    claude_ok              = check_claude()
    ollama_ok, ollama_models = check_ollama()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🌿 ExoRAG")
        st.divider()
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
        <p class="hero-title">🌿 ExoRAG Decision Engine</p>
        <p class="hero-sub">Exosome · Cosmetic · Natural Compound R&D Intelligence</p>
        <span class="hero-badge">Literature Q&A</span>
        <span class="hero-badge">Compound Ranker</span>
        <span class="hero-badge">Claim Generator</span>
        <span class="hero-badge">Mechanism Clusters</span>
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

    # Stats
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
        "🏆 Compound Ranker",
        "📣 Claim Generator",
        "🧬 Mechanism Clusters",
        "🔬 Experimental Readiness",
        "📊 Feedback Loop",
    ])

    no_llm = "⚠️" in backend

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Literature Q&A
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### Quick Demo Queries")
        selected = None
        for i, dq in enumerate(DEMO_QUERIES):
            with st.columns(2)[i % 2]:
                if st.button(f"▶ {dq[:62]}...", key=f"dq_{i}", use_container_width=True):
                    selected = dq
        st.divider()

        query = st.text_area(
            "Ask a product development question:",
            value=selected or "",
            height=80,
            placeholder="e.g. Which plant-derived exosome nanoparticles show the best skin penetration?",
            key="qa_query",
        )
        run_qa = st.button("🔍 Search & Analyze", type="primary", key="run_qa")

        if run_qa and query.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
                return

            chunks  = retrieve(collection, query.strip(), k_val)
            context = build_context(chunks)
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
                    with st.expander(f"{icon} [{i+1}] {m['title'][:48]}... ({m['year']})", expanded=i < 2):
                        st.markdown(f"**{m['title']}**")
                        st.markdown(f"*{m['authors']}*")
                        st.caption(f"`{m['journal']}` · {m['year']} · PMID {m['pmid']}")
                        st.caption(f"Relevance: **{sim:.3f}**")
                        st.markdown(f"[PubMed ↗]({m['url']})")
                        if show_snippets:
                            st.divider()
                            st.caption(c["text"][:320] + "...")

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Compound Ranker
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🏆 Compound Ranker")
        st.markdown("Select an application and get compounds ranked by evidence, safety, and novelty.")

        col1, col2 = st.columns([2, 1])
        with col1:
            application = st.selectbox("Target application:", APPLICATIONS, key="rank_app")
        with col2:
            top_n = st.slider("Top N compounds", 3, 10, 5, key="rank_n")

        run_rank = st.button("🏆 Rank Compounds", type="primary", key="run_rank")

        if run_rank:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"natural compounds exosome {application} evidence efficacy"
                chunks   = retrieve(collection, search_q, min(k_val + 4, 15))
                context  = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Rank the top {top_n} natural compounds for: {application}\n"
                    f"Return a JSON array with {top_n} compounds ranked by overall score."
                )

                with st.spinner(f"Ranking compounds for {application}..."):
                    try:
                        raw  = call_llm(RANKER_SYSTEM, user_msg, backend, ollama_sel)
                        data = normalize_ranker_data(parse_json_response(raw))

                        if not data:
                            st.warning("No compounds extracted. Try reducing K to 3-4 or switching to Claude API.")
                            st.code(raw[:800])
                        else:
                            st.markdown(f"### Top {len(data)} Compounds for **{application}**")
                            st.caption(f"Based on {len(chunks)} retrieved PubMed chunks · Ranked by evidence + safety + novelty")
                            st.divider()

                        for i, item in enumerate(data):
                            score = float(item.get("score", 0))
                            score_class = "score-high" if score >= 7 else "score-mid" if score >= 5 else "score-low"
                            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"

                            with st.container():
                                c1, c2 = st.columns([4, 1])
                                with c1:
                                    st.markdown(
                                        f'<div class="rank-card">'
                                        f'<span style="font-size:1.3rem">{medal}</span> '
                                        f'<span class="rank-name"> {item.get("compound","")}</span> '
                                        f'<span class="score-pill {score_class}">{score:.1f}/10</span><br><br>'
                                        f'<b>Mechanism:</b> {item.get("mechanism","")}<br>'
                                        f'<b>Evidence:</b> {item.get("evidence_level","")}&nbsp;&nbsp;'
                                        f'<b>Novelty:</b> {item.get("novelty","")}<br>'
                                        f'<b>Safety:</b> {item.get("safety_profile","")}<br><br>'
                                        f'<i>{item.get("summary","")}</i>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.markdown("**Raw LLM response:**")
                        st.code(raw[:1500] if raw else "Empty response — try again or reduce K.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API for reliable JSON.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Claim Generator
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 📣 Cosmetic Claim Generator")
        st.markdown("Generate compliant marketing claim language grounded in PubMed evidence.")

        col1, col2 = st.columns(2)
        with col1:
            compound = st.text_input("Compound / ingredient:", placeholder="e.g. curcumin, ginger extract, quercetin", key="claim_compound")
        with col2:
            claim_app = st.selectbox("Application:", APPLICATIONS, key="claim_app")

        evidence_type = st.radio(
            "Evidence basis to emphasize:",
            ["Best available (mixed)", "In vitro only", "In vivo only", "Clinical only"],
            horizontal=True,
            key="claim_evidence",
        )

        run_claim = st.button("📣 Generate Claims", type="primary", key="run_claim")

        if run_claim and compound.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"{compound} exosome {claim_app} evidence mechanism"
                chunks   = retrieve(collection, search_q, k_val)
                context  = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Compound: {compound}\nApplication: {claim_app}\n"
                    f"Evidence emphasis: {evidence_type}\n\n"
                    f"Generate cosmetic claim language. Return only valid JSON."
                )

                with st.spinner(f"Generating claims for {compound}..."):
                    try:
                        raw  = call_llm(CLAIM_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        ev   = data.get("evidence_basis", "mixed")
                        ev_color = {"clinical": "🟢", "in vivo": "🟡", "in vitro": "🔵", "mixed": "🟣"}.get(ev, "⚪")

                        st.markdown(
                            f'<div class="claim-card">'
                            f'<div class="primary-claim">"{data.get("primary_claim","")}"</div>'
                            f'<b>Evidence basis:</b> {ev_color} {ev}<br><br>'
                            f'<b>Supporting claims:</b><br>',
                            unsafe_allow_html=True,
                        )

                        for sc in data.get("supporting_claims", []):
                            st.markdown(f"- {sc}")

                        st.markdown(
                            f'<br><b>⚠️ Regulatory note:</b> <i>{data.get("caution","")}</i>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        st.divider()
                        st.markdown(f"**📄 Based on {len(chunks)} PubMed sources:**")
                        for c in chunks[:4]:
                            m = c["metadata"]
                            st.caption(f"• {m['authors']} ({m['year']}) — {m['title'][:70]}... [PMID {m['pmid']}]({m['url']})")

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.markdown("**Raw LLM response:**")
                        st.code(raw[:1500] if raw else "Empty response — try again or reduce K.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API for reliable JSON.")

        elif run_claim:
            st.warning("Please enter a compound name.")

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Mechanism Clusters
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 🧬 Mechanism Clusters")
        st.markdown("Group natural compounds by biological mechanism for a target application.")
        st.markdown("Think: **target → mechanism → compound** instead of just listing papers.")

        target = st.selectbox("Select biological target:", MECHANISMS, key="cluster_target")
        run_cluster = st.button("🧬 Cluster Compounds", type="primary", key="run_cluster")

        if run_cluster:
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"natural compounds exosome {target} mechanism pathway"
                chunks   = retrieve(collection, search_q, min(k_val + 4, 15))
                context  = build_context(chunks)

                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Cluster natural compounds by mechanism for target: {target}\n"
                    f"Return JSON with mechanism names as keys."
                )

                with st.spinner(f"Clustering compounds for {target}..."):
                    try:
                        raw  = call_llm(CLUSTER_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        st.markdown(f"### Mechanism Map: **{target}**")
                        st.caption(f"Based on {len(chunks)} PubMed sources")
                        st.divider()

                        for mechanism, compounds in data.items():
                            if not compounds:
                                continue
                            with st.expander(f"**{mechanism}** — {len(compounds)} compound(s)", expanded=True):
                                for item in compounds:
                                    # Handle both dict format AND plain string format from Ollama
                                    if isinstance(item, str):
                                        item = {"compound": item, "evidence_strength": 0, "notes": ""}
                                    elif not isinstance(item, dict):
                                        continue
                                    strength = min(max(int(item.get("evidence_strength", 0)), 0), 5)
                                    stars    = "⭐" * strength + "☆" * (5 - strength)
                                    st.markdown(
                                        f'<div class="cluster-card">'
                                        f'<span class="mechanism-title">{item.get("compound","")}</span> '
                                        f'<span style="font-size:0.8rem">{stars if strength > 0 else ""}</span><br>'
                                        f'<span style="font-size:0.82rem; color:#555">{item.get("notes","")}</span>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.markdown("**Raw LLM response:**")
                        st.code(raw[:1500] if raw else "Empty response — try again or reduce K.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API for reliable JSON.")



    # ════════════════════════════════════════════════════════════════
    # TAB 5 — Experimental Readiness
    # ════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### 🔬 Experimental Readiness")
        st.markdown("Can we test this compound **tomorrow**? Get a lab-ready assessment from the literature.")

        col1, col2 = st.columns(2)
        with col1:
            ready_compound = st.text_input(
                "Compound:", placeholder="e.g. curcumin, quercetin, ginger extract",
                key="ready_compound"
            )
        with col2:
            ready_app = st.selectbox("Application:", APPLICATIONS, key="ready_app")

        run_ready = st.button("🔬 Assess Readiness", type="primary", key="run_ready")

        if run_ready and ready_compound.strip():
            if no_llm:
                st.error("No LLM configured — see sidebar.")
            else:
                search_q = f"{ready_compound} exosome concentration cell model assay protocol"
                chunks   = retrieve(collection, search_q, k_val)
                context  = build_context(chunks)
                user_msg = (
                    f"RETRIEVED LITERATURE:\n\n{context}\n\n---\n\n"
                    f"Compound: {ready_compound}\nApplication: {ready_app}\n"
                    f"Assess experimental readiness for lab testing. Return only valid JSON."
                )
                with st.spinner(f"Assessing readiness for {ready_compound}..."):
                    try:
                        raw  = call_llm(READINESS_SYSTEM, user_msg, backend, ollama_sel)
                        data = parse_json_response(raw)

                        score     = int(data.get("overall_readiness", 0))
                        score_col = "#1b5e20" if score >= 7 else "#e65100" if score >= 4 else "#b71c1c"
                        bar_pct   = score * 10

                        st.markdown(f"### {data.get('compound', ready_compound)} — Readiness Assessment")
                        st.divider()

                        # Readiness score bar
                        st.markdown(
                            f"""<div style="background:white;border:1px solid #e0e7f0;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">
                            <div style="font-size:0.85rem;color:#888;margin-bottom:6px">OVERALL READINESS</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:2.5rem;color:{score_col};line-height:1">{score}/10</div>
                            <div style="background:#f0f0f0;border-radius:8px;height:10px;margin-top:10px">
                              <div style="background:{score_col};width:{bar_pct}%;height:10px;border-radius:8px"></div>
                            </div>
                            <div style="margin-top:10px;font-size:0.85rem;color:#333"><b>Next step:</b> {data.get("next_step","")}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # Details grid
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**🧪 Lab Properties**")
                            st.markdown(f"- **Solubility:** {data.get('solubility','Unknown')}")
                            st.markdown(f"- **Concentration:** {data.get('typical_concentration','Unknown')}")
                            st.markdown(f"- **Timeline:** ~{data.get('timeline_days','?')} days to first data")

                        with c2:
                            st.markdown("**🔬 Cell Models & Assays**")
                            for cm in data.get("cell_models", []):
                                st.markdown(f"- {cm}")
                            st.markdown("**Assays:**")
                            for assay in data.get("assays", []):
                                st.markdown(f"- {assay}")

                        with c3:
                            st.markdown("**🛒 Suppliers**")
                            for s in data.get("suppliers", []):
                                st.markdown(f"- {s}")
                            if data.get("blockers"):
                                st.markdown("**⚠️ Potential blockers:**")
                                for b in data.get("blockers", []):
                                    st.markdown(f"- {b}")

                        st.divider()
                        st.caption(f"Based on {len(chunks)} PubMed sources")

                    except Exception as e:
                        st.error(f"JSON parse error: {e}")
                        st.markdown("**Raw LLM response:**")
                        st.code(raw[:1500] if raw else "Empty response — try again or reduce K.")
                        st.info("💡 Reduce K to 3-4, or switch to Claude API for reliable JSON.")

        elif run_ready:
            st.warning("Please enter a compound name.")

    # ════════════════════════════════════════════════════════════════
    # TAB 6 — Feedback Loop
    # ════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("#### 📊 Feedback Loop — Company R&D Intelligence")
        st.markdown("Log your lab results. Over time this builds **proprietary knowledge** no one else has.")

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

        # ── Log new result ──
        st.markdown("### ➕ Log a New Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            fb_compound  = st.text_input("Compound:", key="fb_compound",
                                          placeholder="e.g. curcumin")
            fb_app       = st.selectbox("Application:", APPLICATIONS, key="fb_app")
        with col2:
            fb_model     = st.text_input("Cell model / system:", key="fb_model",
                                          placeholder="e.g. HaCaT, fibroblast, in vivo")
            fb_conc      = st.text_input("Concentration used:", key="fb_conc",
                                          placeholder="e.g. 5µM, 10µg/mL")
        with col3:
            fb_result    = st.radio("Result:", ["✅ Worked", "⚠️ Partial", "❌ Failed"],
                                     key="fb_result")
            fb_score     = st.slider("Efficacy score (your judgment):", 0, 10, 5, key="fb_score")

        fb_notes = st.text_area("Notes / observations:", height=80, key="fb_notes",
                                 placeholder="e.g. Good viability at 5µM, collagen upregulation 40%, but cytotoxic at 20µM")
        fb_date  = st.date_input("Date:", key="fb_date")

        if st.button("💾 Save Result", type="primary", key="save_fb"):
            if fb_compound.strip():
                records = load_feedback()
                records.append({
                    "id":         len(records) + 1,
                    "compound":   fb_compound.strip(),
                    "application": fb_app,
                    "cell_model": fb_model,
                    "concentration": fb_conc,
                    "result":     fb_result,
                    "score":      fb_score,
                    "notes":      fb_notes,
                    "date":       str(fb_date),
                    "logged_at":  datetime.now().isoformat(),
                })
                save_feedback(records)
                st.success(f"✅ Saved result for **{fb_compound}**")
                st.rerun()
            else:
                st.warning("Please enter a compound name.")

        st.divider()

        # ── View logged results ──
        records = load_feedback()

        if not records:
            st.info("No results logged yet. Start logging your lab experiments above to build your internal knowledge base.")
        else:
            st.markdown(f"### 📋 Internal Results Database ({len(records)} entries)")

            # Summary stats
            worked   = sum(1 for r in records if "Worked" in r.get("result",""))
            partial  = sum(1 for r in records if "Partial" in r.get("result",""))
            failed   = sum(1 for r in records if "Failed" in r.get("result",""))
            avg_score = sum(r.get("score",0) for r in records) / len(records) if records else 0

            s1, s2, s3, s4 = st.columns(4)
            for col, num, lbl, color in [
                (s1, worked,          "✅ Worked",   "#1b5e20"),
                (s2, partial,         "⚠️ Partial",  "#e65100"),
                (s3, failed,          "❌ Failed",   "#b71c1c"),
                (s4, f"{avg_score:.1f}/10", "Avg Score", "#0f3460"),
            ]:
                col.markdown(
                    f'<div class="stat-card"><div class="stat-num" style="color:{color}">{num}</div>'
                    f'<div class="stat-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Filter
            filter_result = st.multiselect(
                "Filter by result:",
                ["✅ Worked", "⚠️ Partial", "❌ Failed"],
                default=["✅ Worked", "⚠️ Partial", "❌ Failed"],
                key="fb_filter",
            )

            filtered = [r for r in records if r.get("result","") in filter_result]

            # Display records newest first
            for r in reversed(filtered):
                result_color = {"✅ Worked": "#e8f5e9", "⚠️ Partial": "#fff8e1", "❌ Failed": "#fce4ec"}.get(r.get("result",""), "white")
                border_color = {"✅ Worked": "#2e7d32", "⚠️ Partial": "#e65100", "❌ Failed": "#c62828"}.get(r.get("result",""), "#ccc")

                with st.expander(
                    f"{r.get('result','')} **{r.get('compound','')}** — {r.get('application','')} "
                    f"| Score: {r.get('score','')}/10 | {r.get('date','')}",
                    expanded=False,
                ):
                    st.markdown(
                        f'<div style="background:{result_color};border-left:4px solid {border_color};'
                        f'border-radius:8px;padding:1rem">',
                        unsafe_allow_html=True,
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Compound:** {r.get('compound','')}")
                        st.markdown(f"**Application:** {r.get('application','')}")
                        st.markdown(f"**Cell model:** {r.get('cell_model','')}")
                        st.markdown(f"**Concentration:** {r.get('concentration','')}")
                    with c2:
                        st.markdown(f"**Result:** {r.get('result','')}")
                        st.markdown(f"**Efficacy score:** {r.get('score','')}/10")
                        st.markdown(f"**Date:** {r.get('date','')}")
                    if r.get("notes"):
                        st.markdown(f"**Notes:** {r.get('notes','')}")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            # ── AI synthesis of internal data ──
            if len(records) >= 3 and not no_llm:
                st.markdown("### 🤖 AI Synthesis of Your Internal Results")
                st.caption("Combines your lab data with PubMed literature for proprietary insights")

                if st.button("🧠 Synthesize Internal Intelligence", key="synthesize"):
                    summary_data = [
                        {k: r[k] for k in ["compound","application","result","score","notes","cell_model"]}
                        for r in records
                    ]
                    search_q  = " ".join(set(r["compound"] for r in records[:5]))
                    chunks    = retrieve(collection, search_q + " exosome cosmetic", min(k_val, 8))
                    context   = build_context(chunks)

                    synthesis_prompt = f"""You are analyzing PROPRIETARY internal lab results combined with published literature.

INTERNAL LAB RESULTS:
{json.dumps(summary_data, indent=2)}

PUBLISHED LITERATURE CONTEXT:
{context}

Provide a strategic synthesis covering:
1. Which compounds are working vs failing internally vs what literature predicts
2. Surprising discrepancies between your results and published data
3. Top 3 actionable recommendations based on BOTH internal results AND literature
4. Compounds worth prioritizing for next experiments
5. Any patterns in what works (cell model, concentration, application type)

Be specific and use the actual compound names and scores from the internal data."""

                    with st.spinner("Synthesizing your internal data with PubMed..."):
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
                st.info(f"Log at least 3 results to unlock AI synthesis. ({len(records)}/3 logged)")

            # Export
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
