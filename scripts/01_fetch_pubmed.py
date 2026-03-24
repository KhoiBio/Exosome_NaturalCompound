#!/usr/bin/env python3
"""
01_fetch_pubmed.py
==================
Fetches PubMed abstracts focused on:
  - Exosome / extracellular vesicle biology
  - Cosmetic & skincare applications
  - Natural compound delivery & therapeutics
  - Plant-derived exosome-like nanoparticles
  - Invitrx-specific: PEM, Wharton's jelly, cord blood, amniotic,
    Reluma actives, flow cytometry characterization, GMP/HCT/P

Saves results to: data/abstracts.json
No API key required — uses NCBI Entrez (free).

Usage:
    python scripts/01_fetch_pubmed.py
"""

import os
import json
import time
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

QUERIES = [

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — ORIGINAL QUERIES
    # ══════════════════════════════════════════════════════════════════════════

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


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — INVITRX-SPECIFIC QUERIES
    # Mapped to: Reluma PEM technology, Invitra HCT/P product lines (cord blood,
    # Wharton's jelly, amniotic fluid/membrane), exosome therapeutics summit
    # focus, flow cytometry + ELISA characterization, GMP clean room context
    # ══════════════════════════════════════════════════════════════════════════

    # ── Polypeptide Enriched Media (PEM) — Reluma's core technology ────────
    {
        "label": "pem_polypeptide_growth_factor_skin",
        "query": (
            '("polypeptide"[Title/Abstract] OR "growth factor"[Title/Abstract] OR '
            '"matrix protein"[Title/Abstract] OR "peptide complex"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "anti-aging"[Title/Abstract] OR '
            '"luminosity"[Title/Abstract] OR "fibroblast"[Title/Abstract] OR '
            '"keratinocyte"[Title/Abstract] OR "epidermis"[Title/Abstract])'
        ),
        "max": 100,
    },
    {
        "label": "conditioned_media_secretome_skin",
        "query": (
            '("conditioned medium"[Title/Abstract] OR "secretome"[Title/Abstract] OR '
            '"cell-conditioned media"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "cosmetic"[Title/Abstract] OR '
            '"anti-aging"[Title/Abstract] OR "wound"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "stem_cell_secretome_cosmetic_rejuvenation",
        "query": (
            '"stem cell"[Title/Abstract] AND '
            '("secretome"[Title/Abstract] OR "paracrine"[Title/Abstract] OR '
            '"conditioned medium"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "cosmetic"[Title/Abstract] OR '
            '"rejuvenation"[Title/Abstract] OR "anti-aging"[Title/Abstract])'
        ),
        "max": 100,
    },

    # ── Wharton's Jelly MSC ────────────────────────────────────────────────
    {
        "label": "whartons_jelly_exosome_regenerative",
        "query": (
            '("Wharton jelly"[Title/Abstract] OR "WJ-MSC"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] OR '
            '"secretome"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "wound"[Title/Abstract] OR '
            '"regenerative"[Title/Abstract] OR "anti-inflammatory"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "umbilical_cord_msc_tissue_repair",
        "query": (
            '("umbilical cord MSC"[Title/Abstract] OR "UC-MSC"[Title/Abstract] OR '
            '"Wharton jelly"[Title/Abstract]) '
            'AND ("tissue repair"[Title/Abstract] OR "wound healing"[Title/Abstract] OR '
            '"anti-inflammatory"[Title/Abstract] OR "regenerative"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Umbilical Cord Blood Plasma (hUCBP) ───────────────────────────────
    {
        "label": "cord_blood_plasma_skin_growth_factor",
        "query": (
            '("cord blood plasma"[Title/Abstract] OR '
            '"umbilical cord blood plasma"[Title/Abstract] OR '
            '"cord blood"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "wound"[Title/Abstract] OR '
            '"growth factor"[Title/Abstract] OR "cosmetic"[Title/Abstract] OR '
            '"regenerative"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "cord_blood_exosome_therapeutic",
        "query": (
            '("cord blood"[Title/Abstract] OR "umbilical cord"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("therapeutic"[Title/Abstract] OR "anti-inflammatory"[Title/Abstract] OR '
            '"regenerative"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Amniotic Fluid / Membrane ──────────────────────────────────────────
    {
        "label": "amniotic_fluid_exosome_cosmetic",
        "query": (
            '("amniotic fluid"[Title/Abstract] OR "amniotic membrane"[Title/Abstract] OR '
            '"amnion"[Title/Abstract]) '
            'AND (exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract] OR '
            '"growth factor"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "wound"[Title/Abstract] OR '
            '"cosmetic"[Title/Abstract] OR "regenerative"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "amniotic_membrane_wound_burn_healing",
        "query": (
            '("amniotic membrane"[Title/Abstract] OR "amniotic fluid"[Title/Abstract]) '
            'AND ("wound healing"[Title/Abstract] OR "tissue repair"[Title/Abstract] OR '
            '"burn"[Title/Abstract] OR "ulcer"[Title/Abstract] OR '
            '"anti-inflammatory"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Burn / Diabetic Ulcer / Ocular — Invitrx founding indications ─────
    {
        "label": "cell_therapy_burn_diabetic_ulcer",
        "query": (
            '("stem cell"[Title/Abstract] OR "cell therapy"[Title/Abstract]) '
            'AND ("burn"[Title/Abstract] OR "skin graft"[Title/Abstract] OR '
            '"diabetic ulcer"[Title/Abstract] OR "chronic wound"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_ocular_ophthalmic_surface",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("ocular"[Title/Abstract] OR "ophthalmic"[Title/Abstract] OR '
            '"cornea"[Title/Abstract] OR "dry eye"[Title/Abstract] OR '
            '"ocular surface"[Title/Abstract])'
        ),
        "max": 60,
    },

    # ── EV Characterization — directly ties to your flow cytometry + ELISA ─
    {
        "label": "exosome_flow_cytometry_surface_markers",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("flow cytometry"[Title/Abstract] OR '
            '"nanoparticle tracking analysis"[Title/Abstract] OR "NTA"[Title/Abstract]) '
            'AND ("CD9"[Title/Abstract] OR "CD63"[Title/Abstract] OR '
            '"CD81"[Title/Abstract] OR "tetraspanin"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_elisa_cytokine_quantification",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("ELISA"[Title/Abstract] OR "cytokine"[Title/Abstract] OR '
            '"TGF-beta"[Title/Abstract] OR "VEGF"[Title/Abstract] OR '
            '"EGF"[Title/Abstract] OR "IGF"[Title/Abstract] OR "FGF"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "wound"[Title/Abstract] OR '
            '"cosmetic"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_mirna_small_rna_cargo_skin",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("miRNA"[Title/Abstract] OR "small RNA"[Title/Abstract] OR '
            '"non-coding RNA"[Title/Abstract] OR "RNA cargo"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "fibroblast"[Title/Abstract] OR '
            '"keratinocyte"[Title/Abstract] OR "wound"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_proteomics_cargo_profiling",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("proteomics"[Title/Abstract] OR "protein cargo"[Title/Abstract] OR '
            '"mass spectrometry"[Title/Abstract]) '
            'AND ("skin"[Title/Abstract] OR "cosmetic"[Title/Abstract] OR '
            '"stem cell"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Exosome Isolation + GMP Manufacturing ─────────────────────────────
    {
        "label": "exosome_isolation_gmp_scale_up",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("isolation"[Title/Abstract] OR "purification"[Title/Abstract] OR '
            '"ultracentrifugation"[Title/Abstract] OR "size exclusion"[Title/Abstract]) '
            'AND ("GMP"[Title/Abstract] OR "manufacturing"[Title/Abstract] OR '
            '"scale-up"[Title/Abstract] OR "clinical grade"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "exosome_stability_storage_lyophilization",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("stability"[Title/Abstract] OR "storage"[Title/Abstract] OR '
            '"lyophilization"[Title/Abstract] OR "freeze-drying"[Title/Abstract] OR '
            '"shelf life"[Title/Abstract] OR "formulation"[Title/Abstract])'
        ),
        "max": 60,
    },

    # ── HCT/P Regulatory + FDA Safety ─────────────────────────────────────
    {
        "label": "hctp_fda_regulatory_quality_control",
        "query": (
            '("HCT/P"[Title/Abstract] OR "human cell tissue"[Title/Abstract] OR '
            '"tissue bank"[Title/Abstract]) '
            'AND ("safety"[Title/Abstract] OR "regulatory"[Title/Abstract] OR '
            '"FDA"[Title/Abstract] OR "quality control"[Title/Abstract] OR '
            '"lot release"[Title/Abstract])'
        ),
        "max": 60,
    },
    {
        "label": "exosome_cosmetic_clinical_safety",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("safety"[Title/Abstract] OR "clinical trial"[Title/Abstract] OR '
            '"toxicity"[Title/Abstract] OR "adverse effect"[Title/Abstract]) '
            'AND ("cosmetic"[Title/Abstract] OR "topical"[Title/Abstract] OR '
            '"skin"[Title/Abstract])'
        ),
        "max": 60,
    },

    # ── Plastic Surgery / Reconstructive ──────────────────────────────────
    {
        "label": "cell_therapy_plastic_surgery_scar",
        "query": (
            '("stem cell"[Title/Abstract] OR "cell therapy"[Title/Abstract] OR '
            'exosome[Title/Abstract]) '
            'AND ("plastic surgery"[Title/Abstract] OR "reconstructive"[Title/Abstract] OR '
            '"fat grafting"[Title/Abstract] OR "scar"[Title/Abstract] OR '
            '"keloid"[Title/Abstract])'
        ),
        "max": 60,
    },

    # ── Exosome Cancer + Gene Engineering — Asia summit 2025-2026 ─────────
    {
        "label": "exosome_cancer_immunotherapy_delivery",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("cancer"[Title/Abstract] OR "tumor"[Title/Abstract] OR '
            '"immunotherapy"[Title/Abstract]) '
            'AND ("drug delivery"[Title/Abstract] OR "therapeutic"[Title/Abstract] OR '
            '"engineered"[Title/Abstract])'
        ),
        "max": 80,
    },
    {
        "label": "engineered_exosome_surface_modification",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("gene engineering"[Title/Abstract] OR '
            '"surface modification"[Title/Abstract] OR "engineered"[Title/Abstract] OR '
            '"functionalized"[Title/Abstract]) '
            'AND ("targeted delivery"[Title/Abstract] OR "therapeutic"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Longevity / Anti-Aging Biology ────────────────────────────────────
    {
        "label": "exosome_longevity_senescence_aging",
        "query": (
            '(exosome[Title/Abstract] OR "extracellular vesicle"[Title/Abstract]) '
            'AND ("aging"[Title/Abstract] OR "longevity"[Title/Abstract] OR '
            '"senescence"[Title/Abstract] OR "rejuvenation"[Title/Abstract] OR '
            '"healthspan"[Title/Abstract])'
        ),
        "max": 80,
    },

    # ── Stem Cell QC / Cell Viability ─────────────────────────────────────
    {
        "label": "stem_cell_qc_viability_potency_gmp",
        "query": (
            '"stem cell"[Title/Abstract] '
            'AND ("quality control"[Title/Abstract] OR "cell viability"[Title/Abstract] OR '
            '"potency assay"[Title/Abstract] OR "lot release"[Title/Abstract] OR '
            '"GMP"[Title/Abstract] OR "sterility"[Title/Abstract])'
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

        authors = []
        for author in article.get("AuthorList", [])[:5]:
            last     = str(author.get("LastName", ""))
            initials = str(author.get("Initials", ""))
            if last:
                authors.append(f"{last} {initials}".strip())

        journal_info = article.get("Journal", {})
        journal      = str(journal_info.get("Title", "Unknown Journal"))
        pub_date     = journal_info.get("JournalIssue", {}).get("PubDate", {})
        year_raw     = pub_date.get("Year", pub_date.get("MedlineDate", "2000"))
        year         = str(year_raw)[:4]

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

    n_original = 14
    n_invitrx  = len(QUERIES) - n_original

    print("=" * 65)
    print("ExoRAG — PubMed Fetcher")
    print("Focus: Exosome · Cosmetic · Natural Compounds · Invitrx")
    print("=" * 65)
    print(f"Entrez email      : {Entrez.email}")
    print(f"Output            : {OUTPUT_FILE}")
    print(f"Total queries     : {len(QUERIES)}")
    print(f"  Original        : {n_original}")
    print(f"  Invitrx-specific: {n_invitrx}")
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

        parsed = [parse_record(r, q["label"]) for r in records]
        parsed = [p for p in parsed if p]
        print(f"  Parsed: {len(parsed)} usable abstracts")
        all_parsed.extend(parsed)

    unique = deduplicate(all_parsed)

    print(f"\n{'=' * 65}")
    print(f"Total fetched     : {len(all_parsed)}")
    print(f"After dedup       : {len(unique)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"Saved to          : {OUTPUT_FILE}")

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
