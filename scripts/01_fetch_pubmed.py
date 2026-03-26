#!/usr/bin/env python3
"""
01_fetch_pubmed.py
==================
Fetches PubMed abstracts focused on:
  - Exosome / extracellular vesicle biology
  - Cosmetic & skincare applications
  - Natural compound delivery & therapeutics
  - Plant-derived exosome-like nanoparticles

Saves results to: data/abstracts.json
No API key required — uses NCBI Entrez (free).

Usage:
    python scripts/01_fetch_pubmed.py
"""

import os
import json
import time
import sys
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

# ─── Search Queries ───────────────────────────────────────────────────────────
# Organized by topic: exosome biology, cosmetics, natural compounds, delivery

QUERIES = [

    # ── Exosome + Cosmetic / Skin ──────────────────────────────────────────
    {
        "label": "exosome_skin_cosmetic",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "cosmetic"[Title/Abstract] OR '
            '"anti-aging"[Title/Abstract] OR "skincare"[Title/Abstract] OR '
            '"dermatology"[Title/Abstract] OR "wound healing"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "exosome_collagen_elastin",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("collagen"[Title/Abstract] OR "elastin"[Title/Abstract] OR '
            '"hyaluronic acid"[Title/Abstract] OR "fibroblast"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "exosome_hair_scalp",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("hair"[Title/Abstract] OR "scalp"[Title/Abstract] OR '
            '"alopecia"[Title/Abstract] OR "hair growth"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Natural Compounds + Exosome ────────────────────────────────────────
    {
        "label": "natural_compound_exosome_general",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("natural compound"[Title/Abstract] OR "phytochemical"[Title/Abstract] '
            'OR "plant extract"[Title/Abstract] OR "herbal"[Title/Abstract] '
            'OR "bioactive"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "curcumin_exosome",
        "query": (
            'curcumin[Title/Abstract] AND '
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] '
            'OR "nanoparticle"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "quercetin_resveratrol_exosome",
        "query": (
            '(quercetin[Title/Abstract] OR resveratrol[Title/Abstract] OR '
            'berberine[Title/Abstract] OR "green tea"[Title/Abstract] OR '
            'EGCG[Title/Abstract] OR luteolin[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "ginger_turmeric_exosome",
        "query": (
            '(ginger[Title/Abstract] OR "zingiber"[Title/Abstract] OR '
            'turmeric[Title/Abstract] OR "curcuma"[Title/Abstract] OR '
            'gingerol[Title/Abstract] OR shogaol[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] '
            'OR "nanoparticle"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Plant-Derived Exosome-Like Nanoparticles ───────────────────────────
    {
        "label": "plant_exosome_nanoparticle",
        "query": (
            '"plant-derived exosome"[Title/Abstract] OR '
            '"plant exosome-like nanoparticle"[Title/Abstract] OR '
            '"plant-derived nanoparticle"[Title/Abstract] OR '
            '"plant-derived extracellular vesicle"[Title/Abstract]'
        ),
        "max": 100,
    },
    {
        "label": "fruit_vegetable_exosome",
        "query": (
            '("grape"[Title/Abstract] OR "grapefruit"[Title/Abstract] OR '
            '"aloe vera"[Title/Abstract] OR "garlic"[Title/Abstract] OR '
            '"lemon"[Title/Abstract] OR "carrot"[Title/Abstract] OR '
            '"blueberry"[Title/Abstract] OR "tomato"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] '
            'OR "nanoparticle"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Exosome Drug Delivery + Formulation ───────────────────────────────
    {
        "label": "exosome_drug_loading",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("drug loading"[Title/Abstract] OR "cargo loading"[Title/Abstract] '
            'OR "encapsulation"[Title/Abstract] OR "electroporation"[Title/Abstract] '
            'OR "sonication"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "exosome_topical_transdermal",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("topical"[Title/Abstract] OR "transdermal"[Title/Abstract] '
            'OR "skin penetration"[Title/Abstract] OR "percutaneous"[Title/Abstract])'
        ),
        "max": 100,
    },

    # ── Anti-inflammatory / Antioxidant ───────────────────────────────────
    {
        "label": "exosome_anti_inflammatory_natural",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("anti-inflammatory"[Title/Abstract] OR "antioxidant"[Title/Abstract]) '
            'AND ("natural"[Title/Abstract] OR "plant"[Title/Abstract] OR '
            '"compound"[Title/Abstract] OR "extract"[Title/Abstract])'
        ),
        "max": 100,
    },

    # ── Cosmetic Ingredients Specific ─────────────────────────────────────
    {
        "label": "cosmetic_active_ingredients_exosome",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("retinol"[Title/Abstract] OR "vitamin C"[Title/Abstract] OR '
            '"niacinamide"[Title/Abstract] OR "peptide"[Title/Abstract] OR '
            '"ceramide"[Title/Abstract] OR "hyaluronic acid"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_melanin_pigmentation",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("melanin"[Title/Abstract] OR "pigmentation"[Title/Abstract] OR '
            '"whitening"[Title/Abstract] OR "brightening"[Title/Abstract] OR '
            '"melanocyte"[Title/Abstract])'
        ),
        "max": 60,
    },
]


# ─── Functions ────────────────────────────────────────────────────────────────


# ─── Entity Extraction ────────────────────────────────────────────────────────

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
]

CELL_TYPES = [
    "keratinocyte", "fibroblast", "melanocyte", "stem cell",
    "hacat", "hdfn", "b16f10", "huvec",
]

COSMETIC_TERMS = [
    "skin", "cosmetic", "dermatology", "anti-aging", "hair", "scalp",
    "melanin", "fibroblast", "keratinocyte", "topical", "transdermal",
    "wound", "collagen", "brightening", "moisturizing",
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


def score_cosmetic_relevance(text: str) -> float:
    t = text.lower()
    pos = sum(1 for term in COSMETIC_TERMS  if term in t)
    neg = sum(1 for term in NEGATIVE_TERMS if term in t)
    return round(max((pos - neg * 2) / 10, 0.0), 3)


def score_recency(year: str) -> float:
    try:
        return round(1 / (1 + (2026 - int(year))), 3)
    except Exception:
        return 0.1


def compute_final_score(evidence: float, cosmetic: float, recency: float) -> float:
    return round(0.40 * evidence + 0.40 * cosmetic + 0.20 * recency, 3)

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


def parse_record(record: dict, label: str) -> dict | None:
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

        # ── Scoring layer ──────────────────────────────────────
        full_text      = f"{title} {abstract_text}"
        entities       = extract_entities(full_text)
        study_type     = detect_study_type(full_text)
        ev_score       = score_evidence(study_type)
        cos_score      = score_cosmetic_relevance(full_text)
        rec_score      = score_recency(year)
        final_score    = compute_final_score(ev_score, cos_score, rec_score)

        return {
            "pmid":              pmid,
            "title":             title,
            "abstract":          abstract_text,
            "authors":           authors,
            "journal":           journal,
            "year":              year,
            "keywords":          keywords[:10],
            "mesh_terms":        mesh[:10],
            "search_label":      label,
            "url":               f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "fetched_at":        datetime.now().isoformat(),
            # ── Intelligence fields ──
            "entities":          entities,
            "study_type":        study_type,
            "evidence_score":    ev_score,
            "cosmetic_score":    cos_score,
            "recency_score":     rec_score,
            "final_score":       final_score,
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ExoRAG — PubMed Fetcher")
    print("Focus: Exosome · Cosmetic · Natural Compounds")
    print("=" * 65)
    print(f"Entrez email : {Entrez.email}")
    print(f"Output       : {OUTPUT_FILE}")
    print(f"Queries      : {len(QUERIES)}")
    print()

    all_parsed = []

    for q in QUERIES:
        print(f"\n[{q['label']}]")
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
            p = parse_record(r, q["label"])
            if p:
                parsed.append(p)

        print(f"  Parsed: {len(parsed)} usable abstracts")
        all_parsed.extend(parsed)

    unique = deduplicate(all_parsed)

    print(f"\n{'=' * 65}")
    print(f"Total fetched  : {len(all_parsed)}")
    print(f"After dedup    : {len(unique)}")

    # Sort by final_score — best papers first
    unique_sorted = sorted(unique, key=lambda x: x.get("final_score", 0), reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique_sorted, f, indent=2)

    print(f"Saved to       : {OUTPUT_FILE}")

    if unique_sorted:
        s = unique_sorted[0]
        print(f"\nTop-ranked record:")
        print(f"  PMID        : {s['pmid']}")
        print(f"  Title       : {s['title'][:70]}...")
        print(f"  Year        : {s['year']} | Journal: {s['journal'][:45]}")
        print(f"  Study type  : {s.get('study_type','?')}")
        print(f"  Final score : {s.get('final_score','?')}")
        print(f"  Compounds   : {', '.join(s.get('entities',{}).get('compounds',[])[:5]) or 'none detected'}")

    # Print score distribution
    scores = [r.get("final_score", 0) for r in unique_sorted]
    if scores:
        print(f"\nScore distribution:")
        print(f"  Top paper   : {max(scores):.3f}")
        print(f"  Average     : {sum(scores)/len(scores):.3f}")
        print(f"  Bottom paper: {min(scores):.3f}")
        clinical = sum(1 for r in unique_sorted if r.get("study_type") == "clinical")
        in_vivo  = sum(1 for r in unique_sorted if r.get("study_type") == "in_vivo")
        in_vitro = sum(1 for r in unique_sorted if r.get("study_type") == "in_vitro")
        print(f"\nStudy types:")
        print(f"  Clinical : {clinical}")
        print(f"  In vivo  : {in_vivo}")
        print(f"  In vitro : {in_vitro}")

    print(f"\n✓ Done — next step: python scripts/02_build_index.py")


if __name__ == "__main__":
    main()
