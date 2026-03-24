#!/usr/bin/env python3
"""
03_query_cli.py
===============
Command-line interface for querying ExoRAG using Claude as the LLM.
Good for testing queries before the interview.

Usage:
    python scripts/03_query_cli.py
    python scripts/03_query_cli.py --query "exosome skin anti-aging"
"""

import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import anthropic

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR     = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
OUTPUT_DIR     = Path(os.getenv("OUTPUT_DIR", "./outputs"))
COLLECTION     = "exorag_cosmetic"
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
RETRIEVAL_K    = int(os.getenv("RETRIEVAL_K", "8"))

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Scientific Advisor specializing in exosome-based cosmetics 
and natural compound therapeutics. Your expertise covers:
- Exosome and extracellular vesicle biology
- Natural compound pharmacology and cosmetic applications
- Skincare formulation and active ingredient science
- Plant-derived nanoparticle delivery systems
- Anti-aging, skin brightening, and wound healing research

You are given retrieved PubMed literature. Answer questions with a PRODUCT DEVELOPMENT focus:

1. **Key findings** — what the evidence actually shows
2. **Compound/ingredient candidates** — specific actives with promise
3. **Formulation insights** — delivery methods, loading strategies, stability
4. **Evidence quality** — clearly distinguish in vitro / in vivo / clinical
5. **Product opportunities** — gaps in the literature = white space for innovation

Always cite papers as "AuthorLastName et al., YEAR". 
Be specific, actionable, and direct. Avoid vague summaries.
If the context lacks sufficient information, say so clearly."""

# ─── Demo Queries ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "Which natural compounds loaded into exosomes show the strongest evidence for anti-aging skin benefits?",
    "What plant-derived exosome-like nanoparticles have been studied for cosmetic applications?",
    "How are natural compounds loaded into exosomes and which method works best for skincare actives?",
    "What is the evidence for exosomes in wound healing and skin regeneration using natural ingredients?",
    "Which exosome-based formulations show promise for hair growth and scalp health?",
    "What are the research gaps in natural compound exosome cosmetics — where are the product opportunities?",
    "Compare the evidence for curcumin vs quercetin vs resveratrol in exosome-based skin delivery.",
    "What natural compounds affect melanin production when delivered via exosomes for skin brightening?",
]


# ─── Setup ────────────────────────────────────────────────────────────────────

def load_collection():
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_db/ not found. Run: python scripts/02_build_index.py")
        sys.exit(1)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION, embedding_function=embedding_fn)
    return collection


def load_claude():
    if not ANTHROPIC_KEY or "your-key" in ANTHROPIC_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

def retrieve(collection, query: str, k: int = RETRIEVAL_K) -> list:
    results = collection.query(query_texts=[query], n_results=k)
    chunks  = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":       doc,
            "metadata":   meta,
            "similarity": round(1 - dist, 4),
        })
    return chunks


def format_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks):
        m = c["metadata"]
        parts.append(
            f"[Source {i+1}]\n"
            f"Title: {m['title']}\n"
            f"Authors: {m['authors']} | Year: {m['year']} | Journal: {m['journal']}\n"
            f"PMID: {m['pmid']} | URL: {m['url']}\n"
            f"Relevance: {c['similarity']:.3f}\n"
            f"Text: {c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def query_rag(collection, client, query: str, k: int = RETRIEVAL_K) -> dict:
    chunks  = retrieve(collection, query, k)
    context = format_context(chunks)

    user_msg = (
        f"RETRIEVED LITERATURE CONTEXT:\n\n{context}\n\n"
        f"---\n\nQUESTION (answer with product development focus): {query}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )

    answer = response.content[0].text

    return {
        "query":     query,
        "answer":    answer,
        "model":     CLAUDE_MODEL,
        "sources":   [
            {
                "title":      c["metadata"]["title"],
                "authors":    c["metadata"]["authors"],
                "year":       c["metadata"]["year"],
                "journal":    c["metadata"]["journal"],
                "pmid":       c["metadata"]["pmid"],
                "url":        c["metadata"]["url"],
                "topic":      c["metadata"].get("topic", ""),
                "similarity": c["similarity"],
            }
            for c in chunks
        ],
        "timestamp": datetime.now().isoformat(),
    }


def log_query(result: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "query_log.json"
    existing = []
    if log_file.exists():
        try:
            existing = json.loads(log_file.read_text())
        except Exception:
            pass
    existing.append(result)
    log_file.write_text(json.dumps(existing, indent=2))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, help="Run a single query")
    parser.add_argument("--k",     type=int, default=RETRIEVAL_K)
    args = parser.parse_args()

    print("=" * 65)
    print("ExoRAG — Cosmetic + Natural Compound Intelligence")
    print(f"LLM: {CLAUDE_MODEL} | Embeddings: local MiniLM")
    print("=" * 65)

    print("\nLoading ChromaDB...", end=" ", flush=True)
    collection = load_collection()
    print(f"OK ({collection.count()} chunks)")

    client = load_claude()

    if args.query:
        result = query_rag(collection, client, args.query, args.k)
        print(f"\n{'='*65}\nANSWER:\n{'='*65}")
        print(result["answer"])
        print(f"\n{'='*65}\nSources:")
        for s in result["sources"]:
            print(f"  • {s['authors']} ({s['year']}) — {s['title'][:60]}...")
            print(f"    {s['url']}")
        log_query(result)
        return

    # Interactive mode
    print("\nDEMO QUERIES — type number or your own question:")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i}] {q}")
    print("  [q] Quit\n")

    while True:
        user_input = input("Query > ").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            break
        if user_input.isdigit() and 1 <= int(user_input) <= len(DEMO_QUERIES):
            query = DEMO_QUERIES[int(user_input) - 1]
            print(f"Running: {query}\n")
        elif user_input:
            query = user_input
        else:
            continue

        result = query_rag(collection, client, query, args.k)

        print(f"\n{'='*65}\nANSWER\n{'='*65}")
        print(result["answer"])
        print(f"\nSources ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  • {s['authors']} ({s['year']}) — {s['title'][:55]}...")
            print(f"    sim={s['similarity']:.3f} | {s['url']}")
        log_query(result)
        print()


if __name__ == "__main__":
    main()
