import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import time
import re

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EMAIL = "your_email@example.com"
TOOL = "cosmetic_rag_engine"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

SEARCH_QUERIES = [
    ("natural compounds skin", "cosmetic_general"),
    ("plant extract fibroblast collagen", "anti_aging"),
    ("natural compounds melanin inhibition", "skin_brightening"),
    ("hair growth natural extract keratinocyte", "hair_growth"),
]

MAX_RESULTS = 50


# ─────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────

COMPOUND_KEYWORDS = [
    "curcumin", "quercetin", "resveratrol", "egcg",
    "epigallocatechin gallate", "vitamin c", "ascorbic acid",
    "niacinamide", "retinol", "kojic acid", "luteolin",
    "berberine", "glycyrrhizin", "lycopene"
]

EFFECT_KEYWORDS = [
    "anti-inflammatory", "antioxidant", "collagen",
    "melanin", "brightening", "whitening",
    "anti-aging", "elasticity", "hair growth",
    "wound healing"
]

CELL_TYPES = [
    "keratinocyte", "fibroblast", "melanocyte", "stem cell"
]


def extract_entities(text: str):
    text = text.lower()

    compounds = [c for c in COMPOUND_KEYWORDS if c in text]
    effects = [e for e in EFFECT_KEYWORDS if e in text]
    cells = [c for c in CELL_TYPES if c in text]

    return {
        "compounds": list(set(compounds)),
        "effects": list(set(effects)),
        "cell_types": list(set(cells)),
    }


# ─────────────────────────────────────────────────────────────
# EVIDENCE SCORING
# ─────────────────────────────────────────────────────────────

def detect_study_type(text: str):
    text = text.lower()

    if "clinical trial" in text or "randomized" in text:
        return "clinical"
    elif "in vivo" in text or "mouse" in text or "rat" in text:
        return "in_vivo"
    elif "in vitro" in text or "cell line" in text:
        return "in_vitro"
    elif "review" in text:
        return "review"
    return "unknown"


def score_evidence(study_type: str):
    return {
        "clinical": 1.0,
        "in_vivo": 0.8,
        "in_vitro": 0.6,
        "review": 0.5,
        "unknown": 0.3
    }.get(study_type, 0.3)


# ─────────────────────────────────────────────────────────────
# COSMETIC RELEVANCE
# ─────────────────────────────────────────────────────────────

COSMETIC_TERMS = [
    "skin", "cosmetic", "dermatology", "anti-aging",
    "hair", "scalp", "melanin", "fibroblast",
    "keratinocyte", "topical", "transdermal"
]

NEGATIVE_TERMS = [
    "cancer", "tumor", "chemotherapy"
]


def score_cosmetic_relevance(text: str):
    text = text.lower()

    pos = sum(1 for t in COSMETIC_TERMS if t in text)
    neg = sum(1 for t in NEGATIVE_TERMS if t in text)

    score = pos - (neg * 2)
    return max(score / 10, 0)


# ─────────────────────────────────────────────────────────────
# RECENCY
# ─────────────────────────────────────────────────────────────

def score_recency(year: str):
    try:
        y = int(year)
        return 1 / (1 + (2026 - y))
    except:
        return 0.1


# ─────────────────────────────────────────────────────────────
# PUBMED FETCH
# ─────────────────────────────────────────────────────────────

def search_pubmed(query, max_results=20):
    url = BASE_URL + "esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL
    }

    r = requests.get(url, params=params)
    return r.json()["esearchresult"]["idlist"]


def fetch_details(id_list):
    ids = ",".join(id_list)
    url = BASE_URL + "efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
        "tool": TOOL,
        "email": EMAIL
    }

    r = requests.get(url, params=params)
    return ET.fromstring(r.text)


# ─────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────

def parse_record(article, label):
    try:
        medline = article.find("MedlineCitation")

        pmid = medline.findtext("PMID", default="")

        article_data = medline.find("Article")
        title = article_data.findtext("ArticleTitle", default="")

        abstract = article_data.find("Abstract")
        abstract_text = ""
        if abstract is not None:
            abstract_text = " ".join(
                [t.text for t in abstract.findall("AbstractText") if t.text]
            )

        journal = article_data.findtext("Journal/Title", default="")

        year = article_data.findtext("Journal/JournalIssue/PubDate/Year", default="")

        authors = []
        for a in article_data.findall("AuthorList/Author"):
            last = a.findtext("LastName", "")
            if last:
                authors.append(last)

        keywords = [
            k.text for k in medline.findall("KeywordList/Keyword") if k.text
        ]

        mesh = [
            m.findtext("DescriptorName", "")
            for m in medline.findall("MeshHeadingList/MeshHeading")
        ]

        # ─── SCORING + ENTITIES ───
        full_text = f"{title} {abstract_text}".lower()

        entities = extract_entities(full_text)

        study_type = detect_study_type(full_text)
        evidence_score = score_evidence(study_type)
        cosmetic_score = score_cosmetic_relevance(full_text)
        recency_score = score_recency(year)

        final_score = (
            0.4 * evidence_score +
            0.4 * cosmetic_score +
            0.2 * recency_score
        )

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "authors": authors,
            "journal": journal,
            "year": year,
            "keywords": keywords[:10],
            "mesh_terms": mesh[:10],
            "search_label": label,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",

            "entities": entities,
            "study_type": study_type,
            "evidence_score": round(evidence_score, 3),
            "cosmetic_score": round(cosmetic_score, 3),
            "recency_score": round(recency_score, 3),
            "final_score": round(final_score, 3),

            "fetched_at": datetime.now().isoformat(),
        }

    except Exception as e:
        print("Parse error:", e)
        return None


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    all_results = []

    for query, label in SEARCH_QUERIES:
        print(f"🔎 Searching: {query}")

        ids = search_pubmed(query, MAX_RESULTS)
        time.sleep(0.5)

        xml_root = fetch_details(ids)

        for article in xml_root.findall(".//PubmedArticle"):
            parsed = parse_record(article, label)
            if parsed:
                all_results.append(parsed)

    # Deduplicate by PMID
    unique = {r["pmid"]: r for r in all_results}.values()

    # Sort by score
    ranked = sorted(unique, key=lambda x: x["final_score"], reverse=True)

    # Save
    with open("ranked_cosmetic_papers.json", "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"\n✅ Saved {len(ranked)} ranked papers")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
