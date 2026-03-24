#!/usr/bin/env python3
"""
05_explorer.py
==============
Corpus analytics — surfaces compound frequency, topic trends,
top journals, and year-over-year publication growth.

Usage:
    python scripts/05_explorer.py

Outputs:
    outputs/compound_stats.json
    outputs/report.txt
"""

import os
import json
from pathlib import Path
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR  = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR", "./outputs"))
COLLECTION  = "exorag_cosmetic"

# ── Compound keywords ──────────────────────────────────────────────────────────
COMPOUNDS = {
    "Curcumin":          ["curcumin", "curcuminoid"],
    "Quercetin":         ["quercetin"],
    "Resveratrol":       ["resveratrol"],
    "Berberine":         ["berberine"],
    "EGCG / Green Tea":  ["egcg", "epigallocatechin", "green tea"],
    "Ginger / Shogaol":  ["ginger", "shogaol", "gingerol", "zingiber"],
    "Ginsenoside":       ["ginsenoside", "panax"],
    "Luteolin":          ["luteolin"],
    "Retinol":           ["retinol", "retinoic acid", "vitamin a"],
    "Vitamin C":         ["vitamin c", "ascorbic acid", "ascorbate"],
    "Niacinamide":       ["niacinamide", "nicotinamide"],
    "Hyaluronic Acid":   ["hyaluronic acid", "hyaluronan"],
    "Collagen Peptide":  ["collagen peptide", "hydrolyzed collagen"],
    "Aloe Vera":         ["aloe vera", "aloe barbadensis"],
    "Grape / Resveratrol":["grape", "vitis vinifera"],
    "Garlic":            ["garlic", "allicin", "allium sativum"],
    "Grapefruit":        ["grapefruit", "citrus paradisi"],
    "Sulforaphane":      ["sulforaphane", "broccoli"],
    "Piperine":          ["piperine", "black pepper"],
    "Honokiol":          ["honokiol", "magnolia"],
    "Fisetin":           ["fisetin"],
    "Apigenin":          ["apigenin"],
    "Naringenin":        ["naringenin", "naringin"],
    "Andrographolide":   ["andrographolide"],
    "Ceramide":          ["ceramide"],
}

INDICATIONS = [
    "anti-aging", "wrinkle", "wound healing", "skin brightening", "whitening",
    "hair growth", "alopecia", "inflammation", "acne", "hyperpigmentation",
    "moisturizing", "collagen", "antioxidant", "UV", "photoaging",
    "cancer", "tumor", "diabetes", "cardiovascular",
]

DELIVERY = [
    "topical", "transdermal", "intradermal", "intravenous", "oral",
    "intranasal", "injection", "electroporation", "sonication",
    "extrusion", "incubation",
]


def get_all_docs():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION, embedding_function=ef)
    total      = collection.count()

    docs, metas = [], []
    batch = 500
    offset = 0
    while offset < total:
        result = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        docs.extend(result["documents"])
        metas.extend(result["metadatas"])
        offset += batch

    return docs, metas, total


def count_keyword(docs: list, keywords: list) -> int:
    count = 0
    for doc in docs:
        d = doc.lower()
        if any(kw in d for kw in keywords):
            count += 1
    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ExoRAG — Corpus Explorer")
    print("=" * 65)

    print("\nLoading ChromaDB...", end=" ", flush=True)
    docs, metas, total = get_all_docs()
    print(f"{total} chunks loaded")

    # Unique papers
    seen_pmids = set()
    for m in metas:
        seen_pmids.add(m.get("pmid", ""))
    n_papers = len(seen_pmids)

    # Compound counts
    print("\nCounting compound mentions...")
    compound_counts = {
        name: count_keyword(docs, kws)
        for name, kws in COMPOUNDS.items()
    }
    compound_counts = dict(sorted(compound_counts.items(), key=lambda x: x[1], reverse=True))

    # Indication counts
    indication_counts = {
        ind: count_keyword(docs, [ind])
        for ind in INDICATIONS
    }
    indication_counts = dict(sorted(indication_counts.items(), key=lambda x: x[1], reverse=True))

    # Delivery counts
    delivery_counts = {
        route: count_keyword(docs, [route])
        for route in DELIVERY
    }

    # Year distribution (unique papers)
    year_counter = Counter()
    seen_for_year = set()
    for m in metas:
        pmid = m.get("pmid", "")
        year = m.get("year", "")
        if pmid not in seen_for_year and year.isdigit() and 2000 <= int(year) <= 2026:
            year_counter[year] += 1
            seen_for_year.add(pmid)

    # Topic distribution
    topic_counter = Counter()
    seen_for_topic = set()
    for m in metas:
        pmid  = m.get("pmid", "")
        topic = m.get("topic", "unknown")
        if pmid not in seen_for_topic:
            topic_counter[topic] += 1
            seen_for_topic.add(pmid)

    # Journal distribution
    journal_counter = Counter()
    seen_for_journal = set()
    for m in metas:
        pmid    = m.get("pmid", "")
        journal = m.get("journal", "Unknown")
        if pmid not in seen_for_journal:
            journal_counter[journal] += 1
            seen_for_journal.add(pmid)

    # Save JSON stats
    stats = {
        "total_chunks":        total,
        "unique_papers":       n_papers,
        "compound_mentions":   compound_counts,
        "indication_mentions": indication_counts,
        "delivery_mentions":   delivery_counts,
        "year_distribution":   dict(sorted(year_counter.items())),
        "topic_distribution":  dict(topic_counter.most_common()),
        "top_journals":        dict(journal_counter.most_common(20)),
    }
    stats_file = OUTPUT_DIR / "compound_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2))

    # ── Print report ──────────────────────────────────────────────────────────
    lines = []
    lines += [
        "=" * 65,
        "ExoRAG — Corpus Analysis Report",
        "Focus: Exosome · Cosmetic · Natural Compounds",
        "=" * 65,
        f"Total indexed chunks : {total:,}",
        f"Unique papers        : {n_papers:,}",
        "",
    ]

    lines += ["TOP NATURAL COMPOUNDS / COSMETIC INGREDIENTS:"]
    for i, (name, count) in enumerate(list(compound_counts.items())[:18], 1):
        bar = "█" * min(count // 3, 35)
        lines.append(f"  {i:2}. {name:<22} {count:4d}  {bar}")
    lines.append("")

    lines += ["TOP THERAPEUTIC / COSMETIC INDICATIONS:"]
    for ind, count in list(indication_counts.items())[:12]:
        lines.append(f"  • {ind.capitalize():<28} {count:4d} mentions")
    lines.append("")

    lines += ["DELIVERY ROUTES:"]
    for route, count in sorted(delivery_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {route.capitalize():<20} {count:4d} mentions")
    lines.append("")

    lines += ["PUBLICATION TREND (2018–2026):"]
    for year in sorted(year_counter.keys()):
        if int(year) >= 2018:
            count = year_counter[year]
            bar   = "▓" * min(count // 2, 45)
            lines.append(f"  {year}: {bar} ({count})")
    lines.append("")

    lines += ["TOPIC COVERAGE:"]
    for topic, count in topic_counter.most_common():
        lines.append(f"  • {topic.replace('_', ' '):<42} {count:4d} papers")
    lines.append("")

    lines += ["TOP JOURNALS:"]
    for journal, count in journal_counter.most_common(12):
        lines.append(f"  • {journal[:55]:<55} {count:3d}")

    report = "\n".join(lines)
    print("\n" + report)

    report_file = OUTPUT_DIR / "report.txt"
    report_file.write_text(report)

    print(f"\n✓ Saved: {stats_file}")
    print(f"✓ Saved: {report_file}")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
