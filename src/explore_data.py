"""
explore_data.py

Phase 3: Data Exploration.

Loads every .jsonl file we collected in Phase 2, and reports:
  - record counts per source/condition
  - missing-field rates
  - text length distributions (word counts)
  - duplicate ID checks
  - a few saved charts (PNG files) for visual inspection

Usage:
    python -m src.explore_data
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.config import RAW_DATA_DIR, PROJECT_ROOT, TARGET_CONDITIONS

FIGURES_DIR = PROJECT_ROOT / "data" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file (one JSON object per line) into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def word_count(text) -> int:
    """Count words in a text field, treating None/empty as 0 words."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def explore_trials():
    print("=" * 60)
    print("CLINICAL TRIALS")
    print("=" * 60)

    all_trials = []
    for condition in TARGET_CONDITIONS:
        safe_name = condition.replace(" ", "_")
        path = RAW_DATA_DIR / f"trials_normalized_{safe_name}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        records = load_jsonl(path)
        for r in records:
            r["_condition_query"] = condition
        all_trials.extend(records)

    df = pd.DataFrame(all_trials)
    print(f"\nTotal trials loaded (with duplicates): {len(df)}")

    # Duplicate check: same NCT ID appearing more than once. This happens
    # when a single trial matches more than one of our condition searches
    # (e.g. a trial studying "diabetes and hypertension" gets pulled by both
    # queries). We keep the first occurrence and drop the rest so each trial
    # is only represented once in our final corpus.
    dup_count = df["nct_id"].duplicated().sum()
    print(f"Duplicate NCT IDs found: {dup_count}")
    df = df.drop_duplicates(subset="nct_id", keep="first").reset_index(drop=True)
    print(f"Total trials after deduplication: {len(df)}")

    # Missing-field rates
    print("\nMissing field rates:")
    for col in ["brief_summary", "eligibility_criteria_text", "detailed_description"]:
        missing_pct = df[col].isna().sum() / len(df) * 100
        print(f"  {col}: {missing_pct:.1f}% missing")

    # Text length stats (word counts) — informs chunk sizing later
    df["eligibility_word_count"] = df["eligibility_criteria_text"].apply(word_count)
    df["summary_word_count"] = df["brief_summary"].apply(word_count)

    print("\nEligibility criteria length (words):")
    print(df["eligibility_word_count"].describe().round(1))

    print("\nBrief summary length (words):")
    print(df["summary_word_count"].describe().round(1))

    # Phase distribution
    print("\nStudy phase distribution:")
    phase_counts = df["phases"].apply(lambda p: p[0] if isinstance(p, list) and p else "UNKNOWN").value_counts()
    print(phase_counts)

    # --- Charts ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["eligibility_word_count"].hist(bins=30, ax=axes[0])
    axes[0].set_title("Eligibility criteria length (words)")
    axes[0].set_xlabel("Word count")
    axes[0].set_ylabel("Number of trials")

    df["_condition_query"].value_counts().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Trials collected per condition")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = FIGURES_DIR / "trials_overview.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nChart saved: {out_path}")

    return df


def explore_pubmed():
    print("\n" + "=" * 60)
    print("PUBMED ABSTRACTS")
    print("=" * 60)

    all_articles = []
    for condition in TARGET_CONDITIONS:
        safe_name = condition.replace(" ", "_")
        path = RAW_DATA_DIR / f"pubmed_{safe_name}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        records = load_jsonl(path)
        for r in records:
            r["_condition_query"] = condition
        all_articles.extend(records)

    df = pd.DataFrame(all_articles)
    print(f"\nTotal abstracts loaded (with duplicates): {len(df)}")

    # Same reasoning as trials: an abstract can match more than one
    # condition search, so we deduplicate by PMID (PubMed's unique article ID)
    # and keep only the first occurrence.
    dup_count = df["pmid"].duplicated().sum()
    print(f"Duplicate PMIDs found: {dup_count}")
    df = df.drop_duplicates(subset="pmid", keep="first").reset_index(drop=True)
    print(f"Total abstracts after deduplication: {len(df)}")

    missing_abstract_pct = df["abstract"].apply(lambda a: not a).sum() / len(df) * 100
    print(f"Missing abstract text: {missing_abstract_pct:.1f}%")

    df["abstract_word_count"] = df["abstract"].apply(word_count)
    print("\nAbstract length (words):")
    print(df["abstract_word_count"].describe().round(1))

    print("\nPublication year distribution:")
    print(df["pub_year"].value_counts().sort_index(ascending=False).head(10))

    # --- Charts ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["abstract_word_count"].hist(bins=30, ax=axes[0])
    axes[0].set_title("Abstract length (words)")
    axes[0].set_xlabel("Word count")
    axes[0].set_ylabel("Number of articles")

    df["_condition_query"].value_counts().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Abstracts collected per condition")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = FIGURES_DIR / "pubmed_overview.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nChart saved: {out_path}")

    return df


def run():
    trials_df = explore_trials()
    pubmed_df = explore_pubmed()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records across both sources: {len(trials_df) + len(pubmed_df)}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
