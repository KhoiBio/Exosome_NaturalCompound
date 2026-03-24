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

def search_pubmed(query: str, max_results: int) -> list:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_details(pmids: list) -> list:
    if not pmids:
        return []
    all_records = []
    for i in range(0, len(pmids), 50):
        batch = pmids[i:i+50]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        all_records.extend(records["PubmedArticle"])
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

        return {
            "pmid":         pmid,
            "title":        title,
            "abstract":     abstract_text,
            "authors":      authors,
            "journal":      journal,
            "year":         year,
            "keywords":     keywords[:10],
            "mesh_terms":   mesh[:10],
            "search_label": label,
            "url":          f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "fetched_at":   datetime.now().isoformat(),
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

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"Saved to       : {OUTPUT_FILE}")

    if unique:
        s = unique[0]
        print(f"\nSample record:")
        print(f"  PMID    : {s['pmid']}")
        print(f"  Title   : {s['title'][:70]}...")
        print(f"  Year    : {s['year']} | Journal: {s['journal'][:45]}")
        print(f"  Abstract: {len(s['abstract'])} chars")

    print(f"\n✓ Done — next step: python scripts/02_build_index.py")


if __name__ == "__main__":
    main()
