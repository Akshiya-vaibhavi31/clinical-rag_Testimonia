"""
build_source_index.py

Phase 18: populates the `sources` table from the existing corpus.jsonl
(built in Phase 4). This is a one-time (or re-run-when-corpus-changes)
indexing step — analogous to build_vector_db.py, but for the lightweight
structured SQL index instead of the vector database.

Usage:
    python -m src.build_source_index
"""

import json

from src.config import PROJECT_ROOT
from src.database import init_db, upsert_source

CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"


def run():
    print("Initializing database schema...")
    init_db()

    print(f"Loading corpus from {CORPUS_PATH}...")
    count = 0
    skipped_missing_title = 0

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            metadata = doc.get("metadata", {})

            # BUG FOUND VIA REAL DATA: some corpus records have a missing or
            # empty title (likely a PubMed record whose <ArticleTitle> field
            # was malformed or absent in the original XML — the same kind of
            # messy real-world data we've hit several times before in this
            # project). Our schema correctly requires a non-null title, so
            # rather than crash the whole indexing run over one bad record,
            # we substitute a clear placeholder and report exactly which
            # source_id was affected — surfacing the data quality issue
            # instead of silently hiding it.
            title = doc.get("title")
            if not title or not title.strip():
                title = f"(untitled — {doc.get('source_type', 'unknown')} {doc.get('source_id', 'unknown')})"
                skipped_missing_title += 1
                print(f"  WARNING: missing title for source_id={doc.get('source_id')}, using placeholder")

            if doc["source_type"] == "clinical_trial":
                phases = metadata.get("phases")
                phase_str = ", ".join(phases) if phases else None
                upsert_source(
                    source_id=doc["source_id"],
                    source_type=doc["source_type"],
                    title=title,
                    condition_query=doc.get("condition_query"),
                    phase=phase_str,
                    overall_status=metadata.get("overall_status"),
                )
            else:
                upsert_source(
                    source_id=doc["source_id"],
                    source_type=doc["source_type"],
                    title=title,
                    condition_query=doc.get("condition_query"),
                    pub_year=metadata.get("pub_year"),
                    doi=metadata.get("doi"),
                    pmcid=metadata.get("pmcid"),
                )
            count += 1

    print(f"\nIndexed {count} sources into the database.")
    if skipped_missing_title:
        print(
            f"Note: {skipped_missing_title} source(s) had a missing title and used a placeholder — see warnings above."
        )


if __name__ == "__main__":
    run()
