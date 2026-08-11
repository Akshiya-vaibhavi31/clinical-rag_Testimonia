"""
inspect_citation.py

Companion to citations.py — this is our stand-in for "click a citation to
inspect the supporting evidence" (Phase 12's requirement), since we don't
have a web frontend yet (that's Phase 19). Instead, you can look up any
NCT ID or PMID from an answer's citations and see its full record here.

Usage:
    python -m src.inspect_citation --id NCT01057251
    python -m src.inspect_citation --id 42572726
"""

import argparse
import json

from src.config import PROJECT_ROOT
from src.citations import build_citation_record, format_citation_display

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def inspect(source_id: str):
    chunks = load_jsonl(CHUNKS_PATH)
    matches = [c for c in chunks if c["source_id"] == source_id]

    if not matches:
        print(f"No chunks found for source ID: {source_id}")
        print(
            "Check the ID is correct (e.g. 'NCT01057251' for trials, or the bare number for PubMed, e.g. '42572726')."
        )
        return

    print(f"Found {len(matches)} chunk(s) for source ID: {source_id}\n")
    for i, chunk in enumerate(matches, start=1):
        record = build_citation_record(chunk)
        print(f"--- Chunk {i}/{len(matches)} ({chunk['chunk_id']}) ---")
        print(format_citation_display(record))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="NCT ID (e.g. NCT01057251) or PMID (e.g. 42572726)")
    args = parser.parse_args()

    inspect(args.id)
