"""
citations.py

Phase 12: Citation System.

Builds a structured citation record for every piece of evidence used in an
answer, connecting it to everything the project brief requires:
NCT ID / PMID, PMCID, DOI (when available), article/trial title, section,
and the actual evidence chunk text.

Also maintains a persistent citation log (data/citations/citation_log.jsonl)
— one entry per question asked, recording exactly which chunks were used to
answer it. This is what makes "click a citation to inspect the evidence"
possible even without a web frontend yet (Phase 19): inspect_citation.py
(the CLI companion to this module) can look up any citation by ID and show
its full record, and the log means we can always trace back "what evidence
was actually shown for this specific past answer."

HONEST LIMITATION: not every source has every field. DOI and PMCID only
exist for PubMed articles that have them (not all do — DOI is common but not
universal, and PMCID only exists if the paper is deposited in PubMed
Central). Clinical trials don't have a DOI/PMCID at all — their permanent
identifier is the NCT ID itself. We show "not available" rather than
omitting the field silently, so it's clear the absence was checked for.
"""

import json
from datetime import datetime, timezone

from src.config import PROJECT_ROOT

CITATIONS_DIR = PROJECT_ROOT / "data" / "citations"
CITATIONS_DIR.mkdir(parents=True, exist_ok=True)
CITATION_LOG_PATH = CITATIONS_DIR / "citation_log.jsonl"


def build_citation_record(chunk: dict) -> dict:
    """
    Build a complete citation record from an evidence chunk, connecting it
    to every field the project brief requires.
    """
    metadata = chunk.get("metadata", {})
    is_trial = chunk["source_type"] == "clinical_trial"

    return {
        "chunk_id": chunk["chunk_id"],
        "source_type": chunk["source_type"],
        "nct_id": chunk["source_id"] if is_trial else None,
        "pmid": chunk["source_id"] if not is_trial else None,
        "pmcid": metadata.get("pmcid") if not is_trial else None,
        "doi": metadata.get("doi") if not is_trial else None,
        "title": chunk["title"],
        "section": chunk["section_name"],
        "evidence_text": chunk["text"],
        "condition": chunk.get("condition_query"),
    }


def format_citation_display(record: dict) -> str:
    """Human-readable summary of a citation record, used in CLI output."""
    lines = [f"Title: {record['title']}"]

    if record["nct_id"]:
        lines.append(f"NCT ID: {record['nct_id']}")
        lines.append(f"Link: https://clinicaltrials.gov/study/{record['nct_id']}")
    if record["pmid"]:
        lines.append(f"PMID: {record['pmid']}")
        lines.append(f"Link: https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/")

    lines.append(f"PMCID: {record['pmcid'] or 'not available'}")
    lines.append(f"DOI: {record['doi'] or 'not available'}")
    lines.append(f"Section: {record['section']}")
    lines.append(f"Evidence text:\n{record['evidence_text']}")

    return "\n".join(lines)


def log_citations_for_answer(question: str, evidence_chunks: list[dict], verification_report: dict):
    """
    Append one entry to the persistent citation log, recording exactly what
    evidence was used to answer a given question. This creates an audit
    trail — you can always look back and see precisely which chunks
    (down to the exact text) supported any past answer.
    """
    citation_records = [build_citation_record(chunk) for chunk in evidence_chunks]

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "citations": citation_records,
        "verification_passed": verification_report.get("passed"),
        "fabricated_count": verification_report.get("fabricated_count"),
    }

    with open(CITATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
