"""
ingest_pubmed.py

Run this script to download PubMed abstracts for all target conditions
and save them to disk. This is Phase 2, step 2.

Usage:
    python -m src.ingest_pubmed
"""

import csv
import json
from datetime import datetime, timezone

from src.clients.pubmed_client import fetch_pmids_across_years, fetch_abstracts, parse_pubmed_xml
from src.config import RAW_DATA_DIR, MANIFEST_DIR, TARGET_CONDITIONS, ABSTRACTS_PER_CONDITION

# efetch works better in smaller batches than one huge request —
# NCBI recommends keeping batches modest to avoid timeouts.
BATCH_SIZE = 100


def run():
    manifest_rows = []

    for condition in TARGET_CONDITIONS:
        print(f"Searching PubMed for: {condition}")

        pmids = fetch_pmids_across_years(condition, total_results=ABSTRACTS_PER_CONDITION, years_back=5)
        print(f"  Found {len(pmids)} matching PMIDs (spread across the last 5 years)")

        all_articles = []
        for i in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[i : i + BATCH_SIZE]
            xml_text = fetch_abstracts(batch)
            articles = parse_pubmed_xml(xml_text)
            all_articles.extend(articles)
            print(f"  Fetched batch {i // BATCH_SIZE + 1}: {len(articles)} articles")

        safe_name = condition.replace(" ", "_")
        output_path = RAW_DATA_DIR / f"pubmed_{safe_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for article in all_articles:
                f.write(json.dumps(article) + "\n")

        # Data quality check: how many articles actually have a usable abstract?
        # Many PubMed records are letters, editorials, or conference abstracts
        # with no structured abstract text — worth knowing this rate up front.
        missing_abstract = sum(1 for a in all_articles if not a.get("abstract"))
        print(f"  Missing abstract text: {missing_abstract}/{len(all_articles)}")

        manifest_rows.append(
            {
                "source": "pubmed",
                "query": condition,
                "record_count": len(all_articles),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_file": output_path.name,
            }
        )

    manifest_path = MANIFEST_DIR / "ingestion_log_pubmed.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = sum(row["record_count"] for row in manifest_rows)
    print(f"\nDone. Total abstracts collected: {total}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    run()
