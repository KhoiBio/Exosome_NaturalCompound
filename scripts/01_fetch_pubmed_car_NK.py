#!/usr/bin/env python3
"""
01_fetch_pubmed.py
==================
Fetches PubMed abstracts focused on exosome / extracellular vesicle biology,
with condition terms loaded from a plain-text file (comma-separated).

Usage:
    python 01_fetch_pubmed.py conditions.txt

conditions.txt example:
    aging, senescence, anti-aging, rejuvenation, SASP, inflammaging,
    neurodegeneration, Alzheimer's, wound healing, cancer, fibrosis

How it works:
    - Each keyword in the text file becomes its own labeled search batch:
        (exosome OR "extracellular vesicle" OR sEV ...) AND ("<keyword>")
    - Results are deduplicated by PMID, scored, and saved to data/abstracts.json.

No API key required — uses NCBI Entrez (free, 3 req/sec max).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from Bio import Entrez
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

Entrez.email = os.getenv("ENTREZ_EMAIL", "researcher@email.com")
OUTPUT_FILE  = Path(os.getenv("DATA_DIR", "./data")) / "abstracts.json"
DELAY        = 0.4    # seconds between requests (NCBI allows 3/sec)
MAX_PER_TERM = 150    # default max results per condition keyword

# ─── Exosome/EV identity — always the left side of every AND query ────────────

EV_CORE = (
    '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] '
    'OR "small EV"[Title/Abstract] OR sEV[Title/Abstract] '
    'OR microvesicle[Title/Abstract] OR exomere[Title/Abstract] '
    'OR nanovesicle[Title/Abstract] OR "exosome-mimetic"[Title/Abstract] '
    'OR "biomimetic nanoparticle"[Title/Abstract] '
    'OR "plant-derived exosome"[Title/Abstract] '
    'OR "plant-derived nanoparticle"[Title/Abstract])'
)

# ─── Entity / scoring metadata ────────────────────────────────────────────────

COMPOUND_KEYWORDS = [
    "curcumin", "quercetin", "resveratrol", "egcg", "epigallocatechin",
    "vitamin c", "ascorbic acid", "niacinamide", "retinol", "kojic acid",
    "luteolin", "berberine", "glycyrrhizin", "lycopene", "ginger", "gingerol",
    "shogaol", "aloe vera", "ceramide", "hyaluronic acid", "collagen peptide",
    "naringenin", "apigenin", "fisetin", "sulforaphane", "piperine", "honokiol",
    "andrographolide", "ginsenoside", "green tea",
]

EFFECT_KEYWORDS = [
    "anti-inflammatory", "antioxidant", "collagen", "melanin", "brightening",
    "whitening", "anti-aging", "elasticity", "hair growth", "wound healing",
    "skin barrier", "moisturizing", "pigmentation", "fibroblast proliferation",
    "senescence", "rejuvenation", "longevity", "inflammaging", "sasp",
]

CELL_TYPES = [
    "keratinocyte", "fibroblast", "melanocyte", "stem cell",
    "hacat", "hdfn", "b16f10", "huvec", "mesenchymal",
]

NEGATIVE_TERMS = ["cancer", "tumor", "chemotherapy", "carcinoma", "leukemia"]


def extract_entities(text: str) -> dict:
    t = text.lower()
    return {
        "compounds":  list(set(c for c in COMPOUND_KEYWORDS if c in t)),
        "effects":    list(set(e for e in EFFECT_KEYWORDS   if e in t)),
        "cell_types": list(set(c for c in CELL_TYPES        if c in t)),
    }


def detect_study_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["clinical trial", "randomized", "rct", "human subjects"]):
        return "clinical"
    if any(k in t for k in ["in vivo", "mouse model", "rat model", "animal study"]):
        return "in_vivo"
    if any(k in t for k in ["in vitro", "cell line", "cell culture", "cell-based"]):
        return "in_vitro"
    if any(k in t for k in ["review", "meta-analysis", "systematic review"]):
        return "review"
    return "unknown"


def score_evidence(study_type: str) -> float:
    return {"clinical": 1.0, "in_vivo": 0.8, "in_vitro": 0.6,
            "review": 0.5, "unknown": 0.3}.get(study_type, 0.3)


def score_relevance(text: str, condition_terms: list[str]) -> float:
    """Score based on how many condition keywords from the file appear in the text."""
    t = text.lower()
    hits = sum(1 for term in condition_terms if term.lower() in t)
    neg  = sum(1 for term in NEGATIVE_TERMS if term in t)
    return round(max((hits - neg * 2) / max(len(condition_terms), 10), 0.0), 3)


def score_recency(year: str) -> float:
    try:
        return round(1 / (1 + (2026 - int(year))), 3)
    except Exception:
        return 0.1


def compute_final_score(evidence: float, relevance: float, recency: float) -> float:
    return round(0.40 * evidence + 0.40 * relevance + 0.20 * recency, 3)


# ─── PubMed fetch helpers ─────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int) -> list:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_batch_with_retry(pmids: list, retries: int = 3, wait: int = 5) -> list:
    for attempt in range(retries):
        try:
            handle  = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records["PubmedArticle"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  Network error (attempt {attempt+1}/{retries}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  Skipping batch after {retries} failed attempts: {e}")
                return []


def fetch_details(pmids: list) -> list:
    if not pmids:
        return []
    all_records = []
    for i in range(0, len(pmids), 25):
        batch = pmids[i:i+25]
        records = fetch_batch_with_retry(batch)
        all_records.extend(records)
        time.sleep(DELAY)
    return all_records


def parse_record(record: dict, label: str, condition_terms: list[str]) -> dict | None:
    try:
        article = record["MedlineCitation"]["Article"]
        pmid    = str(record["MedlineCitation"]["PMID"])
        title   = str(article.get("ArticleTitle", "")).strip()

        # Abstract (structured or plain)
        abstract_text = ""
        abstract_obj  = article.get("Abstract", {})
        if abstract_obj:
            raw = abstract_obj.get("AbstractText", "")
            if isinstance(raw, list):
                parts = []
                for part in raw:
                    label_tag = part.attributes.get("Label", "") if hasattr(part, "attributes") else ""
                    text = str(part).strip()
                    parts.append(f"{label_tag}: {text}" if label_tag else text)
                abstract_text = " ".join(parts)
            else:
                abstract_text = str(raw).strip()

        if not abstract_text or len(abstract_text) < 80:
            return None

        # Authors
        authors = []
        for author in article.get("AuthorList", [])[:5]:
            last     = str(author.get("LastName", ""))
            initials = str(author.get("Initials", ""))
            if last:
                authors.append(f"{last} {initials}".strip())

        # Journal + year
        journal_info = article.get("Journal", {})
        journal      = str(journal_info.get("Title", "Unknown Journal"))
        pub_date     = journal_info.get("JournalIssue", {}).get("PubDate", {})
        year_raw     = pub_date.get("Year", pub_date.get("MedlineDate", "2000"))
        year         = str(year_raw)[:4]

        # Keywords & MeSH
        keywords = []
        for kw_group in record["MedlineCitation"].get("KeywordList", []):
            for kw in kw_group:
                keywords.append(str(kw))

        mesh = []
        for m in record["MedlineCitation"].get("MeshHeadingList", []):
            mesh.append(str(m.get("DescriptorName", "")))

        # ── Scoring ──────────────────────────────────────────────────────────
        full_text   = f"{title} {abstract_text}"
        entities    = extract_entities(full_text)
        study_type  = detect_study_type(full_text)
        ev_score    = score_evidence(study_type)
        rel_score   = score_relevance(full_text, condition_terms)
        rec_score   = score_recency(year)
        final_score = compute_final_score(ev_score, rel_score, rec_score)

        return {
            "pmid":           pmid,
            "title":          title,
            "abstract":       abstract_text,
            "authors":        authors,
            "journal":        journal,
            "year":           year,
            "keywords":       keywords[:10],
            "mesh_terms":     mesh[:10],
            "search_label":   label,
            "url":            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "fetched_at":     datetime.now().isoformat(),
            # Intelligence fields
            "entities":       entities,
            "study_type":     study_type,
            "evidence_score": ev_score,
            "relevance_score": rel_score,
            "recency_score":  rec_score,
            "final_score":    final_score,
        }
    except Exception:
        return None


def deduplicate(records: list) -> list:
    seen, unique = set(), []
    for r in records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            unique.append(r)
    return unique


# ─── Keyword file parser ──────────────────────────────────────────────────────

def load_conditions(filepath: str) -> list[str]:
    """
    Reads a text file of comma-separated keywords.
    Handles multi-line files, strips whitespace, skips blanks.

    Example file contents:
        aging, senescence, anti-aging,
        rejuvenation, SASP, inflammaging,
        neurodegeneration, wound healing
    """
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    raw = path.read_text(encoding="utf-8")
    # Split on commas, collapse newlines, strip whitespace
    terms = [t.strip() for t in raw.replace("\n", ",").split(",")]
    terms = [t for t in terms if t]  # drop empty strings

    if not terms:
        print(f"ERROR: No keywords found in {filepath}")
        sys.exit(1)

    return terms


def build_queries(condition_terms: list[str], max_per_term: int) -> list[dict]:
    """
    Build one PubMed query per condition term:
        EV_CORE AND ("<term>"[Title/Abstract])
    """
    queries = []
    for term in condition_terms:
        label = term.lower().replace(" ", "_").replace("'", "").replace("/", "_")
        query = f'{EV_CORE} AND ("{term}"[Title/Abstract])'
        queries.append({
            "label": f"ev_{label}",
            "query": query,
            "max":   max_per_term,
            "term":  term,
        })
    return queries


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch PubMed exosome abstracts filtered by condition keywords.",
        epilog="Example: python 01_fetch_pubmed.py conditions.txt --max 200 --out data/abstracts.json"
    )
    parser.add_argument(
        "conditions_file",
        help="Path to a text file with condition keywords, comma-separated. "
             "Example: aging, senescence, anti-aging, wound healing"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_PER_TERM,
        help=f"Max PubMed results per condition keyword (default: {MAX_PER_TERM})"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output JSON file path (default: {OUTPUT_FILE})"
    )
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load keywords ────────────────────────────────────────────────────────
    condition_terms = load_conditions(args.conditions_file)
    queries         = build_queries(condition_terms, args.max)

    print("=" * 65)
    print("ExoRAG — PubMed Fetcher (keyword-file mode)")
    print("=" * 65)
    print(f"Entrez email     : {Entrez.email}")
    print(f"Conditions file  : {args.conditions_file}")
    print(f"Keywords loaded  : {len(condition_terms)}")
    print(f"Max per keyword  : {args.max}")
    print(f"Output           : {output_path}")
    print(f"\nKeywords: {', '.join(condition_terms)}")
    print()

    all_parsed = []

    for q in queries:
        print(f"\n[{q['label']}]  term: \"{q['term']}\"")
        print(f"  Searching (max {q['max']})...", end=" ", flush=True)
        pmids = search_pubmed(q["query"], q["max"])
        print(f"{len(pmids)} PMIDs found")
        time.sleep(DELAY)

        if not pmids:
            continue

        print(f"  Fetching abstracts...", end=" ", flush=True)
        records = fetch_details(pmids)
        print(f"{len(records)} records fetched")

        parsed = []
        for r in records:
            p = parse_record(r, q["label"], condition_terms)
            if p:
                parsed.append(p)

        print(f"  Parsed: {len(parsed)} usable abstracts")
        all_parsed.extend(parsed)

    unique = deduplicate(all_parsed)

    print(f"\n{'=' * 65}")
    print(f"Total fetched  : {len(all_parsed)}")
    print(f"After dedup    : {len(unique)}")

    unique_sorted = sorted(unique, key=lambda x: x.get("final_score", 0), reverse=True)

    with open(output_path, "w") as f:
        json.dump(unique_sorted, f, indent=2)

    print(f"Saved to       : {output_path}")

    if unique_sorted:
        s = unique_sorted[0]
        print(f"\nTop-ranked record:")
        print(f"  PMID         : {s['pmid']}")
        print(f"  Title        : {s['title'][:70]}...")
        print(f"  Year         : {s['year']} | Journal: {s['journal'][:45]}")
        print(f"  Study type   : {s.get('study_type', '?')}")
        print(f"  Final score  : {s.get('final_score', '?')}")
        print(f"  Compounds    : {', '.join(s.get('entities', {}).get('compounds', [])[:5]) or 'none detected'}")

    scores = [r.get("final_score", 0) for r in unique_sorted]
    if scores:
        print(f"\nScore distribution:")
        print(f"  Top paper    : {max(scores):.3f}")
        print(f"  Average      : {sum(scores)/len(scores):.3f}")
        print(f"  Bottom paper : {min(scores):.3f}")
        clinical = sum(1 for r in unique_sorted if r.get("study_type") == "clinical")
        in_vivo  = sum(1 for r in unique_sorted if r.get("study_type") == "in_vivo")
        in_vitro = sum(1 for r in unique_sorted if r.get("study_type") == "in_vitro")
        review   = sum(1 for r in unique_sorted if r.get("study_type") == "review")
        print(f"\nStudy types:")
        print(f"  Clinical : {clinical}")
        print(f"  In vivo  : {in_vivo}")
        print(f"  In vitro : {in_vitro}")
        print(f"  Review   : {review}")

    print(f"\n✓ Done — next step: python scripts/02_build_index.py")


if __name__ == "__main__":
    main()
