#!/usr/bin/env python3
"""
02_build_index.py
=================
Chunks abstracts and indexes them into a LOCAL ChromaDB vector database.
Uses sentence-transformers for FREE local embeddings (no API key needed).

Model: all-MiniLM-L6-v2
  - ~80MB download on first run
  - Fast, good quality for biomedical text
  - Runs entirely on CPU (no GPU needed)

ChromaDB stores data in: ./chroma_db/  (local folder, persists on disk)

Usage:
    python scripts/02_build_index.py
"""

import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_FILE    = Path(os.getenv("DATA_DIR", "./data")) / "abstracts.json"
CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION    = "exorag_car_nk"      # updated: CAR-NK exosome cancer focus
CHUNK_SIZE    = 380                  # words per chunk (~500 tokens)
CHUNK_OVERLAP = 50                   # word overlap between chunks
BATCH_SIZE    = 100                  # chunks per ChromaDB batch


# ─── Local Embedding Function ─────────────────────────────────────────────────

def get_embedding_fn():
    print("  Loading local embedding model: all-MiniLM-L6-v2")
    print("  (First run downloads ~80MB — subsequent runs are instant)")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_documents(records: list) -> tuple:
    """Convert abstract records into ChromaDB-ready (docs, metadatas, ids)."""
    documents, metadatas, ids = [], [], []

    for record in tqdm(records, desc="  Chunking"):
        full_text = f"TITLE: {record['title']}\n\nABSTRACT: {record['abstract']}"
        chunks    = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            doc_id = f"pmid_{record['pmid']}_c{i}"
            meta   = {
                "pmid":           record["pmid"],
                "title":          record["title"][:200],
                "authors":        ", ".join(record["authors"][:3]),
                "journal":        record["journal"][:100],
                "year":           record["year"],
                "url":            record["url"],
                "chunk_index":    i,
                "total_chunks":   len(chunks),
                "topic":          record.get("search_label", "general"),
                "keywords":       ", ".join(record.get("keywords", [])[:5]),
                "study_type":     record.get("study_type", "unknown"),
                "relevance_score": str(record.get("relevance_score", 0)),  # str for Chroma compat
            }
            documents.append(chunk)
            metadatas.append(meta)
            ids.append(doc_id)

    return documents, metadatas, ids


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ExoRAG — Build Local ChromaDB Index")
    print("Focus  : CAR-NK-Derived Exosomes as Cancer Immunotherapy")
    print("Embeddings: all-MiniLM-L6-v2 (local, free)")
    print("=" * 65)

    # Load abstracts
    if not INPUT_FILE.exists():
        print(f"\nERROR: {INPUT_FILE} not found.")
        print("Run first: python scripts/01_fetch_pubmed.py")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        records = json.load(f)
    print(f"\nLoaded {len(records)} abstracts from {INPUT_FILE}")

    # Print topic breakdown
    from collections import Counter
    topics = Counter(r.get("search_label", "unknown") for r in records)
    print("\nTopic breakdown:")
    for topic, count in topics.most_common():
        print(f"  {topic:<50} {count:>4} papers")

    # Print relevance score distribution
    high   = sum(1 for r in records if r.get("relevance_score", 0) >= 0.7)
    exact  = sum(1 for r in records if r.get("relevance_score", 0) >= 1.0)
    print(f"\nRelevance tiers:")
    print(f"  Exact paradigm (CAR-NK → exosome → cancer): {exact}")
    print(f"  High relevance (score ≥ 0.7)              : {high}")
    print(f"  Total                                      : {len(records)}")

    # Setup embeddings
    print("\nSetting up embeddings...")
    embedding_fn = get_embedding_fn()

    # Setup ChromaDB (local persistent)
    print(f"\nInitializing ChromaDB at: {CHROMA_DIR.resolve()}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate collection for clean rebuild
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        print(f"  Dropping existing collection '{COLLECTION}' (clean rebuild)...")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Created collection: '{COLLECTION}'")

    # Build chunks
    print(f"\nChunking {len(records)} abstracts...")
    documents, metadatas, ids = build_documents(records)
    print(f"  Total chunks: {len(documents)}")

    # Embed + index in batches
    print(f"\nEmbedding and indexing ({BATCH_SIZE} chunks/batch)...")
    print("  Estimated time: 5–15 min on CPU (runs once, then instant)")

    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="  Indexing"):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE],
        )

    # Verify
    count = collection.count()
    print(f"\n{'=' * 65}")
    print(f"✓ Indexed {count} chunks into ChromaDB")
    print(f"  Location : {CHROMA_DIR.resolve()}")
    print(f"  Size     : ~{sum(f.stat().st_size for f in CHROMA_DIR.rglob('*') if f.is_file()) // 1024 // 1024} MB")

    # Sanity check — updated to CAR-NK exosome cancer context
    print("\nSanity check — querying 'CAR-NK derived exosome cancer killing'...")
    results = collection.query(
        query_texts=["CAR-NK derived exosome cancer killing perforin granzyme"],
        n_results=3
    )
    for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  [{j+1}] {meta['title'][:65]}...")
        print(f"       {meta['year']} | {meta['journal'][:45]}")
        print(f"       study_type={meta['study_type']} | relevance={meta['relevance_score']}")

    print(f"\n✓ Done — next step: streamlit run scripts/04_app.py")


if __name__ == "__main__":
    main()
