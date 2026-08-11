"""
ingest_trials.py

Run this script to download clinical trial data for all target conditions
and save it to disk. This is Phase 2, step 1.

Usage:
    python -m src.ingest_trials
"""

import csv
import json
from datetime import datetime, timezone

from src.clients.clinicaltrials_client import fetch_trials_for_condition, normalize_trial
from src.config import RAW_DATA_DIR, MANIFEST_DIR, TARGET_CONDITIONS, TRIALS_PER_CONDITION


def run():
    manifest_rows = []

    for condition in TARGET_CONDITIONS:
        print(f"Fetching trials for: {condition}")

        raw_studies = fetch_trials_for_condition(condition, max_records=TRIALS_PER_CONDITION)
        print(f"  Retrieved {len(raw_studies)} raw records")

        # Save raw, untouched API responses — always keep the original data
        # so we can re-process it later without hitting the API again.
        safe_name = condition.replace(" ", "_")
        raw_path = RAW_DATA_DIR / f"trials_raw_{safe_name}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for study in raw_studies:
                f.write(json.dumps(study) + "\n")

        # Save normalized (flattened) version — this is what later phases will use.
        normalized = [normalize_trial(s) for s in raw_studies]
        normalized_path = RAW_DATA_DIR / f"trials_normalized_{safe_name}.jsonl"
        with open(normalized_path, "w", encoding="utf-8") as f:
            for trial in normalized:
                f.write(json.dumps(trial) + "\n")

        # Basic data quality check: how many records are missing key fields?
        missing_summary = sum(1 for t in normalized if not t.get("brief_summary"))
        missing_eligibility = sum(1 for t in normalized if not t.get("eligibility_criteria_text"))
        print(f"  Missing brief_summary: {missing_summary}/{len(normalized)}")
        print(f"  Missing eligibility criteria: {missing_eligibility}/{len(normalized)}")

        manifest_rows.append(
            {
                "source": "clinicaltrials.gov",
                "query": condition,
                "record_count": len(normalized),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_file": raw_path.name,
                "normalized_file": normalized_path.name,
            }
        )

    # Track what we did, when, and how much data we got — this is what makes
    # the pipeline reproducible and debuggable later.
    manifest_path = MANIFEST_DIR / "ingestion_log_trials.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = sum(row["record_count"] for row in manifest_rows)
    print(f"\nDone. Total trials collected: {total}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    run()
