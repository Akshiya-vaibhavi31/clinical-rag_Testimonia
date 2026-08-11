"""
build_corpus.py

Phase 4: Build the Unified Corpus.

Reads the deduplicated, normalized data from Phase 2/3 and converts every
trial and every abstract into ONE consistent document shape:

    {
      "document_id": "...",
      "source_type": "clinical_trial" | "pubmed_abstract",
      "source_id": "NCT04..." | "38123456",
      "title": "...",
      "condition_query": "type 2 diabetes",
      "sections": [
          {"section_name": "...", "text": "..."},
          ...
      ],
      "metadata": {...}
    }

Why sections instead of one big text blob: a clinical trial has genuinely
different kinds of content (a summary, a list of who can join, a list of
outcomes) that we will want to chunk DIFFERENTLY later (Phase 5), based on
what Phase 3 taught us about their length characteristics. Splitting into
sections now means Phase 5 can just say "chunk each section according to
its own rules" instead of re-parsing raw trial/abstract data again.

Usage:
    python -m src.build_corpus
"""

import json
from pathlib import Path

from src.config import RAW_DATA_DIR, PROJECT_ROOT, TARGET_CONDITIONS

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def trial_to_document(trial: dict, condition_query: str) -> dict:
    """
    Convert one normalized trial record into the unified schema.
    Each meaningful text field becomes its own section, and empty/missing
    fields are simply skipped rather than included as empty sections.
    """
    sections = []

    if trial.get("brief_summary"):
        sections.append({"section_name": "brief_summary", "text": trial["brief_summary"]})

    if trial.get("detailed_description"):
        sections.append({"section_name": "detailed_description", "text": trial["detailed_description"]})

    if trial.get("eligibility_criteria_text"):
        sections.append({"section_name": "eligibility_criteria", "text": trial["eligibility_criteria_text"]})

    # Outcomes are lists of dicts (measure + time_frame) — flatten them into
    # one readable text block rather than leaving them as raw nested JSON.
    primary_outcomes = trial.get("primary_outcomes") or []
    if primary_outcomes:
        outcome_lines = [
            f"- {o.get('measure', '')} (time frame: {o.get('timeFrame', 'not specified')})" for o in primary_outcomes
        ]
        sections.append(
            {
                "section_name": "primary_outcomes",
                "text": "Primary outcomes:\n" + "\n".join(outcome_lines),
            }
        )

    return {
        "document_id": f"trial_{trial.get('nct_id')}",
        "source_type": "clinical_trial",
        "source_id": trial.get("nct_id"),
        "title": trial.get("brief_title"),
        "condition_query": condition_query,
        "sections": sections,
        "metadata": {
            "overall_status": trial.get("overall_status"),
            "study_type": trial.get("study_type"),
            "phases": trial.get("phases"),
            "lead_sponsor": trial.get("lead_sponsor"),
            "minimum_age": trial.get("minimum_age"),
            "sex": trial.get("sex"),
            "start_date": trial.get("start_date"),
            "completion_date": trial.get("completion_date"),
        },
    }


def abstract_to_document(article: dict, condition_query: str) -> dict:
    """Convert one PubMed abstract record into the unified schema."""
    sections = []

    if article.get("abstract"):
        sections.append({"section_name": "abstract", "text": article["abstract"]})

    return {
        "document_id": f"pubmed_{article.get('pmid')}",
        "source_type": "pubmed_abstract",
        "source_id": article.get("pmid"),
        "title": article.get("title"),
        "condition_query": condition_query,
        "sections": sections,
        "metadata": {
            "journal": article.get("journal"),
            "pub_year": article.get("pub_year"),
            "mesh_terms": article.get("mesh_terms"),
            "doi": article.get("doi"),
            "pmcid": article.get("pmcid"),
        },
    }


def run():
    all_documents = []
    seen_trial_ids = set()
    seen_pmids = set()

    # --- Trials ---
    for condition in TARGET_CONDITIONS:
        safe_name = condition.replace(" ", "_")
        path = RAW_DATA_DIR / f"trials_normalized_{safe_name}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue

        for trial in load_jsonl(path):
            nct_id = trial.get("nct_id")
            if not nct_id or nct_id in seen_trial_ids:
                continue  # skip duplicates, same logic as Phase 3
            seen_trial_ids.add(nct_id)

            doc = trial_to_document(trial, condition)
            if doc["sections"]:  # skip documents that ended up with no usable text
                all_documents.append(doc)

    trial_doc_count = len(all_documents)
    print(f"Built {trial_doc_count} trial documents")

    # --- Abstracts ---
    for condition in TARGET_CONDITIONS:
        safe_name = condition.replace(" ", "_")
        path = RAW_DATA_DIR / f"pubmed_{safe_name}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue

        for article in load_jsonl(path):
            pmid = article.get("pmid")
            if not pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            doc = abstract_to_document(article, condition)
            if doc["sections"]:
                all_documents.append(doc)

    abstract_doc_count = len(all_documents) - trial_doc_count
    print(f"Built {abstract_doc_count} abstract documents")

    # --- Save the unified corpus ---
    output_path = PROCESSED_DIR / "corpus.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + "\n")

    total_sections = sum(len(doc["sections"]) for doc in all_documents)
    print(f"\nTotal documents: {len(all_documents)}")
    print(f"Total sections across all documents: {total_sections}")
    print(f"Corpus saved to: {output_path}")

    # --- Show two example documents so we can visually sanity-check the shape ---
    print("\n--- Example trial document ---")
    trial_example = next((d for d in all_documents if d["source_type"] == "clinical_trial"), None)
    if trial_example:
        print(json.dumps(trial_example, indent=2)[:1200])

    print("\n--- Example abstract document ---")
    abstract_example = next((d for d in all_documents if d["source_type"] == "pubmed_abstract"), None)
    if abstract_example:
        print(json.dumps(abstract_example, indent=2)[:1200])


if __name__ == "__main__":
    run()
