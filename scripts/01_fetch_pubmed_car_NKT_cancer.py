#!/usr/bin/env python3
"""
01_fetch_pubmed.py
==================
Fetches PubMed abstracts focused on:
  - CAR-NK cell therapy & exosomes
  - CAR-T cell therapy & exosomes
  - Exosome-mediated cancer immunotherapy
  - NK cell-derived extracellular vesicles
  - Exosome engineering for cancer targeting

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
# Organized by topic: CAR-NK, CAR-T, exosome cancer immunotherapy

QUERIES = [

    # ── CAR-NK + Exosome ──────────────────────────────────────────────────
    {
        "label": "car_nk_exosome_core",
        "query": (
            '("CAR-NK"[Title/Abstract] OR "chimeric antigen receptor NK"[Title/Abstract] '
            'OR "CAR NK cell"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] '
            'OR "nanoparticle"[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "car_nk_cancer_therapy",
        "query": (
            '("CAR-NK"[Title/Abstract] OR "chimeric antigen receptor NK"[Title/Abstract] '
            'OR "CAR NK"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract] OR oncology[Title/Abstract] '
            'OR leukemia[Title/Abstract] OR lymphoma[Title/Abstract] OR glioma[Title/Abstract] '
            'OR "solid tumor"[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "nk_cell_derived_exosome",
        "query": (
            '("NK cell-derived exosome"[Title/Abstract] OR '
            '"NK-derived extracellular vesicle"[Title/Abstract] OR '
            '"natural killer cell exosome"[Title/Abstract] OR '
            '"NK cell exosome"[Title/Abstract] OR '
            '"NK-derived nanovesicle"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "nk_cell_exosome_cytotoxicity",
        "query": (
            '("natural killer"[Title/Abstract] OR "NK cell"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND (cytotoxic[Title/Abstract] OR "tumor killing"[Title/Abstract] OR '
            '"anti-tumor"[Title/Abstract] OR "antitumor"[Title/Abstract] OR '
            '"cancer"[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "car_nk_engineering",
        "query": (
            '("CAR-NK"[Title/Abstract] OR "CAR NK"[Title/Abstract]) '
            'AND ("engineering"[Title/Abstract] OR "design"[Title/Abstract] OR '
            '"manufacture"[Title/Abstract] OR "expansion"[Title/Abstract] OR '
            '"iPSC"[Title/Abstract] OR "cord blood"[Title/Abstract])'
        ),
        "max": 150,
    },

    # ── CAR-T + Exosome ───────────────────────────────────────────────────
    {
        "label": "car_t_exosome_core",
        "query": (
            '("CAR-T"[Title/Abstract] OR "CAR T cell"[Title/Abstract] OR '
            '"chimeric antigen receptor T"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "car_t_exosome_cancer",
        "query": (
            '("CAR-T"[Title/Abstract] OR "CAR T"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract] OR '
            'leukemia[Title/Abstract] OR lymphoma[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "car_t_exosome_delivery",
        "query": (
            '("CAR-T"[Title/Abstract] OR "CAR T cell"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("drug delivery"[Title/Abstract] OR "payload"[Title/Abstract] OR '
            '"cargo"[Title/Abstract] OR "loading"[Title/Abstract])'
        ),
        "max": 100,
    },

    # ── Exosome Cancer Immunotherapy (Broad) ──────────────────────────────
    {
        "label": "exosome_cancer_immunotherapy",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND (immunotherapy[Title/Abstract] OR "immune checkpoint"[Title/Abstract] OR '
            '"PD-1"[Title/Abstract] OR "PD-L1"[Title/Abstract] OR '
            '"checkpoint blockade"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "exosome_tumor_microenvironment",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("tumor microenvironment"[Title/Abstract] OR "TME"[Title/Abstract]) '
            'AND (immunotherapy[Title/Abstract] OR "immune evasion"[Title/Abstract] OR '
            '"immune suppression"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "exosome_cancer_drug_delivery",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("drug delivery"[Title/Abstract] OR "targeted delivery"[Title/Abstract] OR '
            '"nanocarrier"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract] OR oncology[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "engineered_exosome_cancer",
        "query": (
            '("engineered exosome"[Title/Abstract] OR "modified exosome"[Title/Abstract] OR '
            '"functionalized exosome"[Title/Abstract] OR '
            '"surface-engineered extracellular vesicle"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract])'
        ),
        "max": 150,
    },

    # ── NK Cell Biology + Cancer ───────────────────────────────────────────
    {
        "label": "nk_cell_cancer_therapy",
        "query": (
            '("natural killer cell"[Title/Abstract] OR "NK cell"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract]) '
            'AND ("adoptive therapy"[Title/Abstract] OR "immunotherapy"[Title/Abstract] OR '
            '"cell therapy"[Title/Abstract])'
        ),
        "max": 200,
    },
    {
        "label": "nk_cell_solid_tumor",
        "query": (
            '("natural killer cell"[Title/Abstract] OR "NK cell"[Title/Abstract] OR '
            '"CAR-NK"[Title/Abstract]) '
            'AND "solid tumor"[Title/Abstract]'
        ),
        "max": 150,
    },
    {
        "label": "nk_cell_glioblastoma",
        "query": (
            '("natural killer cell"[Title/Abstract] OR "NK cell"[Title/Abstract] OR '
            '"CAR-NK"[Title/Abstract]) '
            'AND (glioblastoma[Title/Abstract] OR "GBM"[Title/Abstract] OR '
            '"glioma"[Title/Abstract] OR "brain tumor"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "nk_cell_hematologic_malignancy",
        "query": (
            '("natural killer cell"[Title/Abstract] OR "NK cell"[Title/Abstract] OR '
            '"CAR-NK"[Title/Abstract]) '
            'AND (leukemia[Title/Abstract] OR lymphoma[Title/Abstract] OR '
            '"AML"[Title/Abstract] OR "ALL"[Title/Abstract] OR "CLL"[Title/Abstract] OR '
            '"multiple myeloma"[Title/Abstract])'
        ),
        "max": 150,
    },

    # ── Exosome + Specific Cancers ─────────────────────────────────────────
    {
        "label": "exosome_glioblastoma",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND (glioblastoma[Title/Abstract] OR "GBM"[Title/Abstract] OR '
            '"glioma"[Title/Abstract] OR "brain tumor"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "exosome_breast_lung_cancer",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("breast cancer"[Title/Abstract] OR "lung cancer"[Title/Abstract] OR '
            '"NSCLC"[Title/Abstract] OR "triple negative"[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "exosome_leukemia_lymphoma",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND (leukemia[Title/Abstract] OR lymphoma[Title/Abstract] OR '
            '"AML"[Title/Abstract] OR "multiple myeloma"[Title/Abstract])'
        ),
        "max": 150,
    },

    # ── Exosome Biomarkers + Cancer Diagnosis ─────────────────────────────
    {
        "label": "exosome_cancer_biomarker",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("biomarker"[Title/Abstract] OR "liquid biopsy"[Title/Abstract] OR '
            '"circulating"[Title/Abstract] OR "diagnosis"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract])'
        ),
        "max": 150,
    },

    # ── Exosome Cargo + Immune Modulation ─────────────────────────────────
    {
        "label": "exosome_mirna_cancer",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("miRNA"[Title/Abstract] OR "microRNA"[Title/Abstract] OR '
            '"siRNA"[Title/Abstract] OR "lncRNA"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract])'
        ),
        "max": 150,
    },
    {
        "label": "exosome_perforin_granzyme",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("perforin"[Title/Abstract] OR "granzyme"[Title/Abstract] OR '
            '"FasL"[Title/Abstract] OR "TRAIL"[Title/Abstract] OR '
            '"NKG2D"[Title/Abstract] OR "DNAM-1"[Title/Abstract])'
        ),
        "max": 100,
    },

    # ── Clinical Trials + Manufacturing ───────────────────────────────────
    {
        "label": "car_nk_clinical_trial",
        "query": (
            '("CAR-NK"[Title/Abstract] OR "CAR NK"[Title/Abstract]) '
            'AND ("clinical trial"[Title/Abstract] OR "phase I"[Title/Abstract] OR '
            '"phase II"[Title/Abstract] OR "GMP"[Title/Abstract] OR '
            '"clinical study"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "exosome_clinical_cancer",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("clinical trial"[Title/Abstract] OR "phase I"[Title/Abstract] OR '
            '"phase II"[Title/Abstract] OR "clinical study"[Title/Abstract]) '
            'AND (cancer[Title/Abstract] OR tumor[Title/Abstract])'
        ),
        "max": 100,
    },
]


# ─── Entity Extraction ────────────────────────────────────────────────────────

IMMUNOTHERAPY_KEYWORDS = [
    "CAR-NK", "CAR-T", "chimeric antigen receptor", "natural killer",
    "adoptive cell therapy", "checkpoint inhibitor", "PD-1", "PD-L1",
    "CTLA-4", "immune checkpoint", "bispecific", "NK cell",
    "T cell", "cytokine", "IL-15", "IL-2", "interferon",
]

CANCER_KEYWORDS = [
    "glioblastoma", "GBM", "glioma", "leukemia", "lymphoma", "AML", "ALL",
    "breast cancer", "lung cancer", "NSCLC", "colorectal", "pancreatic",
    "ovarian", "prostate", "melanoma", "solid tumor", "multiple myeloma",
    "hepatocellular", "bladder cancer", "renal cell",
]

EXOSOME_KEYWORDS = [
    "exosome", "extracellular vesicle", "nanoparticle", "microvesicle",
    "exosome-like", "nanovesicle", "membrane vesicle",
]

MECHANISM_KEYWORDS = [
    "cytotoxicity", "apoptosis", "perforin", "granzyme", "FasL", "TRAIL",
    "NKG2D", "DNAM-1", "ADCC", "tumor killing", "anti-tumor",
    "drug delivery", "cargo loading", "miRNA", "siRNA", "immunosuppression",
    "tumor microenvironment", "TME", "immune evasion",
]

CELL_TYPES = [
    "NK-92", "iPSC", "cord blood", "peripheral blood", "K562",
    "Jurkat", "HeLa", "U87", "PBMC", "T cell", "dendritic cell",
]

NEGATIVE_TERMS = []  # No exclusions — all cancer context is relevant


def extract_entities(text: str) -> dict:
    t = text.lower()
    return {
        "immunotherapy":  list(set(k for k in IMMUNOTHERAPY_KEYWORDS if k.lower() in t)),
        "cancers":        list(set(k for k in CANCER_KEYWORDS        if k.lower() in t)),
        "exosome_terms":  list(set(k for k in EXOSOME_KEYWORDS       if k.lower() in t)),
        "mechanisms":     list(set(k for k in MECHANISM_KEYWORDS      if k.lower() in t)),
        "cell_types":     list(set(k for k in CELL_TYPES             if k.lower() in t)),
    }


def detect_study_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["clinical trial", "randomized", "rct", "human subjects", "phase i", "phase ii"]):
        return "clinical"
    if any(k in t for k in ["in vivo", "mouse model", "rat model", "animal study", "xenograft"]):
        return "in_vivo"
    if any(k in t for k in ["in vitro", "cell line", "cell culture", "cell-based"]):
        return "in_vitro"
    if any(k in t for k in ["review", "meta-analysis", "systematic review"]):
        return "review"
    return "unknown"


def score_evidence(study_type: str) -> float:
    return {"clinical": 1.0, "in_vivo": 0.8, "in_vitro": 0.6,
            "review": 0.5, "unknown": 0.3}.get(study_type, 0.3)


def score_relevance(text: str) -> float:
    """Score relevance to CAR-NK / exosome / cancer topic."""
    t = text.lower()
    score = 0.0
    # High-value: CAR-NK + exosome combo
    car_nk = any(k in t for k in ["car-nk", "car nk", "chimeric antigen receptor nk"])
    exo    = any(k in t for k in ["exosome", "extracellular vesicle", "nanovesicle"])
    cancer = any(k in t for k in ["cancer", "tumor", "leukemia", "lymphoma", "glioma"])
    if car_nk and exo:  score += 0.5
    if car_nk:          score += 0.2
    if exo and cancer:  score += 0.2
    if cancer:          score += 0.1
    return round(min(score, 1.0), 3)


def score_recency(year: str) -> float:
    try:
        return round(1 / (1 + (2026 - int(year))), 3)
    except Exception:
        return 0.1


def compute_final_score(evidence: float, relevance: float, recency: float) -> float:
    return round(0.40 * evidence + 0.40 * relevance + 0.20 * recency, 3)


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
        full_text   = f"{title} {abstract_text}"
        entities    = extract_entities(full_text)
        study_type  = detect_study_type(full_text)
        ev_score    = score_evidence(study_type)
        rel_score   = score_relevance(full_text)
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
            # ── Intelligence fields ──
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ExoRAG — PubMed Fetcher")
    print("Focus: CAR-NK · CAR-T · Exosome · Cancer Immunotherapy")
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
        print(f"  PMID         : {s['pmid']}")
        print(f"  Title        : {s['title'][:70]}...")
        print(f"  Year         : {s['year']} | Journal: {s['journal'][:45]}")
        print(f"  Study type   : {s.get('study_type','?')}")
        print(f"  Final score  : {s.get('final_score','?')}")
        print(f"  Immunotherapy: {', '.join(s.get('entities',{}).get('immunotherapy',[])[:5]) or 'none detected'}")
        print(f"  Cancers      : {', '.join(s.get('entities',{}).get('cancers',[])[:5]) or 'none detected'}")

    # Print score distribution
    scores = [r.get("final_score", 0) for r in unique_sorted]
    if scores:
        print(f"\nScore distribution:")
        print(f"  Top paper    : {max(scores):.3f}")
        print(f"  Average      : {sum(scores)/len(scores):.3f}")
        print(f"  Bottom paper : {min(scores):.3f}")

        clinical = sum(1 for r in unique_sorted if r.get("study_type") == "clinical")
        in_vivo  = sum(1 for r in unique_sorted if r.get("study_type") == "in_vivo")
        in_vitro = sum(1 for r in unique_sorted if r.get("study_type") == "in_vitro")
        reviews  = sum(1 for r in unique_sorted if r.get("study_type") == "review")
        car_nk   = sum(1 for r in unique_sorted
                       if any("car-nk" in e.lower() or "car nk" in e.lower()
                              for e in r.get("entities", {}).get("immunotherapy", [])))

        print(f"\nStudy types:")
        print(f"  Clinical : {clinical}")
        print(f"  In vivo  : {in_vivo}")
        print(f"  In vitro : {in_vitro}")
        print(f"  Reviews  : {reviews}")
        print(f"\nCAR-NK papers: {car_nk}")

    print(f"\n✓ Done — next step: python scripts/02_build_index.py")


if __name__ == "__main__":
    main()
